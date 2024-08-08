#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::list<std::string> inputArgs(cl::Positional, cl::desc("<input args>"),
                                       cl::ZeroOrMore, cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// States
//===----------------------------------------------------------------------===//
/// Classes that store states.
/// The simulator defines 2 maps - newValueStates and oldValueStates - to
/// represent the previous (used during the signals' collection to prevent "data
/// race conditions") and current (where we collect new values to). The map
/// updaters (inside the Simulator class as well), in fact, containing
/// references to corresponding pair of new and old valueStates, lets assign the
/// newly collected values from newValueStates to oldValueStates when the
/// collection is finished.

// The state in circuit's value states
class ValueState {
public:
  ValueState(Value val) : val(val) {}

  virtual ~ValueState() {};

protected:
  Value val;
};

/// The state, templated with concrete hasndshake value type
template <typename Ty>
class TypedValueState : public ValueState {
public:
  TypedValueState(TypedValue<Ty> val) : ValueState(val) {}
};

class ChannelState : public TypedValueState<handshake::ChannelType> {
  // Give the corresponding RW API an access to data members
  friend struct ChannelConsumerRW;
  friend struct ChannelProducerRW;
  // Give the corresponding Updater an access to data members
  friend class ChannelUpdater;
  // Temporary solution for the testing function printValueStates() function
  // (which one will be removed in the future)
  friend class Simulator;

public:
  using TypedValueState<handshake::ChannelType>::TypedValueState;

  ChannelState(TypedValue<handshake::ChannelType> channel)
      : TypedValueState<handshake::ChannelType>(channel) {

    llvm::TypeSwitch<Type, void>(channel.getType().getDataType())
        .Case<IntegerType>([&](IntegerType intType) {
          data = APInt(intType.getWidth(), 0, intType.isSigned());
        })
        .Case<FloatType>([&](FloatType floatType) {
          data = APFloat(floatType.getFloatSemantics());
        })
        .Default([&](auto) {
          emitError(channel.getLoc())
              << "Unsuported date type " << channel.getType()
              << ", we111 should probably report an error and stop";
        });
  }

protected:
  bool valid = false;
  bool ready = false;
  llvm::Any data = {};
  // Maybe some additional data signals here
};

class ControlState : public TypedValueState<handshake::ControlType> {
  // Give the corresponding RW API an access to data members
  friend struct ControlConsumerRW;
  friend struct ControlProducerRW;
  // Give the corresponding Updater an access to data members
  friend class ControlUpdater;
  // Temporary solution for the testing function printValueStates() function
  // (which one will be removed in the future)
  friend class Simulator;

public:
  using TypedValueState<handshake::ControlType>::TypedValueState;

  ControlState(TypedValue<handshake::ControlType> control)
      : TypedValueState<handshake::ControlType>(control) {}

protected:
  bool valid = false;
  bool ready = false;
};

//===----------------------------------------------------------------------===//
// Updater
//===----------------------------------------------------------------------===//
/// Classes to update old values states with new ones after finishing the data
/// collection.
/// The map updaters (defined inside the Simulator class) contains references to
/// the corresponding pair of new and old valueStates, assigns the newly
/// collected values from newValueStates to oldValueStates when the collection
/// is finished.

/// Base Updater class
class Updater {
public:
  virtual ~Updater() = default;
  // Check if, for some oldValueState, newValueState, oldValueState ==
  // newValueState
  virtual bool check() = 0;
  // Update oldValueState with the values of newValueState
  virtual void update() = 0;
  // Make valid signal of the state true (For example, when need manual
  // management for circuit's external inputs)
  virtual void setValid() = 0;
  // For the particular valueState, remove valid if ready is set
  virtual void resetValid() = 0;
};

/// Templated updater class that contains pair <oldValueState, newValueState>
template <typename State>
class DoubleUpdater : public Updater {
protected:
  State &oldState;
  State &newState;

public:
  DoubleUpdater(State &oldState, State &newState)
      : oldState(oldState), newState(newState) {}
};

/// Update valueStates with "Channel type": valid, ready, data
class ChannelUpdater : public DoubleUpdater<ChannelState> {
public:
  ChannelUpdater(ChannelState &oldState, ChannelState &newState)
      : DoubleUpdater<ChannelState>(oldState, newState) {}
  bool check() override {
    return newState.valid == oldState.valid && newState.ready == oldState.ready;
  }
  void setValid() override {
    newState.valid = true;
    update();
  }
  void resetValid() override {
    if (newState.ready)
      newState.valid = false;
    update();
  }
  void update() override {
    oldState.valid = newState.valid;
    oldState.ready = newState.ready;
    oldState.data = newState.data;
  }
};

/// Update valueStates with "Control type": valid, ready
class ControlUpdater : public DoubleUpdater<ControlState> {
public:
  ControlUpdater(ControlState &oldState, ControlState &newState)
      : DoubleUpdater<ControlState>(oldState, newState) {}
  bool check() override {
    return newState.valid == oldState.valid && newState.ready == oldState.ready;
  }
  void setValid() override {
    newState.valid = true;
    update();
  }
  void resetValid() override {
    if (newState.ready)
      newState.valid = false;
    update();
  }
  void update() override {
    oldState.valid = newState.valid;
    oldState.ready = newState.ready;
  }
};

//===----------------------------------------------------------------------===//
// Readers & writers
//===----------------------------------------------------------------------===//

/// Classes with references to oldValueStates and newValueStates to represent
/// readers/writers API (that is opened to access for the user). The map<
/// <value, Operation*>, RW*> rws (inside the simulator class) stores these
/// classes and is deleted by the simulator itself. In fact, RW is a sort of the
/// union of the StateReader and StateWriter classes from your code

/// Base RW
/// Virtual destructor is not necessary but maybe add some virtual func members
/// later
class RW {
public:
  RW(Value val) : val(val) {}
  virtual ~RW() {};

private:
  Value val;
};

/// Templated RW
template <typename Ty>
class TypedRW : public RW {
public:
  TypedRW(TypedValue<Ty> val) : RW(val) {}
};

////--- Channel (data)

/// In this case the user can change the valid signal, but has ReadOnly access
/// to the valid and data ones.
struct ChannelConsumerRW : public TypedRW<handshake::ChannelType> {
  const bool &valid;
  const Any &data;
  bool &ready;

  using TypedRW<handshake::ChannelType>::TypedRW;
  ChannelConsumerRW(TypedValue<handshake::ChannelType> channel,
                    ChannelState &reader, ChannelState &writer)
      : TypedRW<handshake::ChannelType>(channel), valid(reader.valid),
        data(reader.data), ready(writer.ready) {}
};

/// In this case the user can change the valid and data signals, but has
/// ReadOnly access to the ready one.
struct ChannelProducerRW : public TypedRW<handshake::ChannelType> {
  bool &valid;
  Any &data;
  const bool &ready;

  using TypedRW<handshake::ChannelType>::TypedRW;
  ChannelProducerRW(TypedValue<handshake::ChannelType> channel,
                    ChannelState &reader, ChannelState &writer)
      : TypedRW<handshake::ChannelType>(channel), valid(writer.valid),
        data(writer.data), ready(reader.ready) {}
};

////--- Control (no data)

/// In this case the user can change the ready signal, but has
/// ReadOnly access to the valid one.
struct ControlConsumerRW : public TypedRW<handshake::ControlType> {
  const bool &valid;
  bool &ready;

  using TypedRW<handshake::ControlType>::TypedRW;
  ControlConsumerRW(TypedValue<handshake::ControlType> control,
                    ControlState &reader, ControlState &writer)
      : TypedRW<handshake::ControlType>(control), valid(reader.valid),
        ready(writer.ready) {}
};

/// In this case the user can change the valid signal, but has
/// ReadOnly access to the ready one.
struct ControlProducerRW : public TypedRW<handshake::ControlType> {
  bool &valid;
  const bool &ready;

  using TypedRW<handshake::ControlType>::TypedRW;
  ControlProducerRW(TypedValue<handshake::ControlType> control,
                    ControlState &reader, ControlState &writer)
      : TypedRW<handshake::ControlType>(control), valid(writer.valid),
        ready(reader.ready) {}
};

//===----------------------------------------------------------------------===//
// Execution Model
//===----------------------------------------------------------------------===//
/// Classes to represent execution models
/// Base class
class ExecutionModel {

public:
  ExecutionModel(Operation *op) : op(op) {}
  // Execute the model: reset if isReset == true, clock iteration otherwise
  virtual void exec(bool isReset) = 0;

  virtual ~ExecutionModel() {}

protected:
  Operation *op;
};

/// Typed execution model
template <typename Op>
class OpExecutionModel : public ExecutionModel {
  // Give an access to the getState function for the simulator
  friend class Simulator;

public:
  OpExecutionModel(Op op) : ExecutionModel(op.getOperation()) {}

protected:
  Op getOperation() { return cast<Op>(op); }
  // Get exact RW state type (ChannelProducerRW / ChannelConsumerRW /
  // ControlProducerRW / ControlConsumerRW)
  template <typename State>
  static State *
  getState(Value val, Op op,
           mlir::DenseMap<std::pair<OpOperand *, Operation *>, RW *> &rws) {
    return static_cast<State *>(rws[{&*val.getUses().begin(), op}]);
  }
};

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//
/// Components required for the internal state. Some of them will be used in
/// future.

class Mul4Stage {
public:
  Mul4Stage(unsigned bitwidth) { this->bitwidth = bitwidth; }
  void exec(bool isReset, bool ce, const APInt &a, const APInt &b, APInt &p) {
    if (isReset) {
      // here for example we do nothing on the reset
    } else if (ce) {
      p = APInt(a * b);
    }
  }

private:
  // In fact, some components don't need the generics due to the fact of using
  // llvm::Any for their data fields.
  unsigned bitwidth;
};

class Join {
public:
  Join(unsigned size) : size(size) {}

  void exec(bool isRst, const std::vector<bool> &insValid, bool outsReady,
            std::vector<std::reference_wrapper<bool>> &insReady,
            bool &outsValid) {
    outsValid = true;
    for (unsigned i = 0; i < size; ++i) {
      outsValid = outsValid && insValid[i];
    }

    for (unsigned i = 0; i < size; ++i) {
      insReady[i].get() = outsReady;
      for (unsigned j = 0; j < size; ++j) {
        if (i != j) {
          insReady[i].get() = insReady[i].get() && insValid[j];
        }
      }
    }
  }

private:
  unsigned size;
};

class DelayBuffer {
public:
  DelayBuffer(unsigned size = 32) : size(size) {}
  bool exec(bool isReset, bool validIn, bool readyIn, bool &validOut) {
    if (isReset) {
      regs.resize(size, false);
      regs[0] = validIn;
      counter = size;
    } else {
      if (counter == 0)
        regs[counter] = validIn;
      else
        regs[counter] = regs[counter - 1];

      ++counter;
    }

    if (counter == size) {
      validOut = regs[size - 1];
      counter = 0;
      return true;
    }
    return false;
  }

private:
  unsigned size;
  unsigned counter = 0;
  std::vector<bool> regs;
};

class OEHB {
public:
  OEHB(unsigned dataWidth) { this->dataWidth = dataWidth; }

  void exec(bool isReset, const Any &ins, bool insValid, bool outsReady,
            bool &insReady, Any &outs, bool &outsValid) {
    if (isReset) {
      outsValid = false;
      insReady = true;
      outs = 0;
    } else {
      outsValid = insValid;
      insReady = !insValid || outsReady;
      llvm::outs() << "ins_ready = " << insReady << "!\n";
      if (insValid && outsReady) {
        outs = ins;
      }
    }
  }

private:
  unsigned dataWidth;
};

class TEHBDataless {
public:
  TEHBDataless() = default;
  void exec(bool isReset, bool insValid, bool outsReady, bool &insReady,
            bool &outsValid) {
    if (isReset) {
      insReady = true;
    } else {
      insReady = !insValid || outsReady;
    }
    outsValid = insValid;
  }
};

//===----------------------------------------------------------------------===//
// Cocrete Execution Models
//===----------------------------------------------------------------------===//
class SinkModel : public OpExecutionModel<handshake::SinkOp> {
public:
  using OpExecutionModel<handshake::SinkOp>::OpExecutionModel;
  SinkModel(handshake::SinkOp sinkOp,
            mlir::DenseMap<std::pair<OpOperand *, Operation *>, RW *> &rws)
      : OpExecutionModel<handshake::SinkOp>(sinkOp), op(sinkOp) {

    ins = getState<ControlConsumerRW>(op.getOperand(), op, rws);
  }

  void exec(bool isReset) override { ins->ready = true; }

private:
  handshake::SinkOp op;
  ControlConsumerRW *ins;
};

class ReturnModel : public OpExecutionModel<handshake::ReturnOp> {
public:
  using OpExecutionModel<handshake::ReturnOp>::OpExecutionModel;
  ReturnModel(handshake::ReturnOp returnOp,
              mlir::DenseMap<std::pair<OpOperand *, Operation *>, RW *> &rws)
      : OpExecutionModel<handshake::ReturnOp>(returnOp), op(returnOp),
        returnTEHB() {
    ins = getState<ChannelConsumerRW>(op.getInputs().front(), op, rws);
    outs = getState<ChannelProducerRW>(op.getOutputs().front(), op, rws);

    auto temp =
        cast<handshake::ChannelType>(returnOp.getInputs().front().getType());
    datawidth = temp.getDataBitWidth();
  }

  void exec(bool isReset) override {
    bool regNotFull = false, regEnable = false;
    Any dataReg;

    returnTEHB.exec(isReset, ins->valid, outs->ready, regNotFull, outs->valid);
    regEnable = regNotFull && ins->valid && !outs->ready;

    if (isReset) {
      dataReg = APInt(datawidth, 0);
    } else if (regEnable) {
      dataReg = ins->data;
    }
    if (regNotFull) {
      outs->data = ins->data;
    } else {
      outs->data = dataReg;
    }
    ins->ready = regNotFull;
  }

private:
  handshake::ReturnOp op;
  unsigned datawidth;

  ChannelConsumerRW *ins;
  ChannelProducerRW *outs;

  TEHBDataless returnTEHB;
  Any dataReg;
};

/// In fact, there's no need in a specific class for End - it just sets
/// the ready for the circuit on reset (e.g. once). To be refactored later
class EndMemlessModel : public OpExecutionModel<handshake::EndOp> {
public:
  using OpExecutionModel<handshake::EndOp>::OpExecutionModel;
  EndMemlessModel(
      handshake::EndOp endOp,
      mlir::DenseMap<std::pair<OpOperand *, Operation *>, RW *> &rws)
      : OpExecutionModel<handshake::EndOp>(endOp), op(endOp) {
    ins = getState<ChannelConsumerRW>(op.getInputs().front(), op, rws);
    auto temp =
        cast<handshake::ChannelType>(endOp.getInputs().front().getType());
    bitwidth = temp.getDataBitWidth();
  }

  void exec(bool isReset) override {
    if (isReset)
      ins->ready = true;
  }

private:
  handshake::EndOp op;
  unsigned bitwidth;

  ChannelConsumerRW *ins;
};

// Mutual component for binary operations: AddI, AndI, CmpI, MulI, OrI, ShlI,
// ShrsI, ShruI, SubI, XorI
using BinaryCompFunc = std::function<llvm::Any(llvm::Any, llvm::Any)>;
template <typename Op>
class GenericBinaryOpModel : public OpExecutionModel<Op> {
public:
  using OpExecutionModel<Op>::OpExecutionModel;
  GenericBinaryOpModel(
      Op op, mlir::DenseMap<std::pair<OpOperand *, Operation *>, RW *> &rws,
      const BinaryCompFunc &callback, unsigned latency)
      : OpExecutionModel<Op>(op), op(op), callback(callback), latency(latency),
        bitwidth(cast<handshake::ChannelType>(op.getResult().getType())
                     .getDataBitWidth()),
        lhs(OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
            op.getLhs(), op, rws)),
        rhs(OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
            op.getRhs(), op, rws)),
        result(OpExecutionModel<Op>::template getState<ChannelProducerRW>(
            op.getResult(), op, rws)),
        binJoin(2) {}

  void exec(bool isReset) override {
    std::vector<bool> insValid{lhs->valid, rhs->valid};
    std::vector<std::reference_wrapper<bool>> insReady{lhs->ready, rhs->ready};
    // Set hadClock if the new iteration of the outer loop has started
    if (!isReset) {
      hadClock = true;
    }

    binJoin.exec(isReset, insValid, result->ready, insReady, result->valid);
    // Store data internally if there were no data && lhs and rhs are valid
    // (that is may transfer their data)
    if (!hasData && result->valid && hadClock) {
      hasData = true;
      resData = any_cast<APInt>(callback(lhs->data, rhs->data));
      result->valid = false;
      counter = 0;
    }
    // Increase latency counter here
    if (hasData && hadClock) {
      ++counter;
      hadClock = false;
    }
    // Now we are ready to transfer data
    if (counter > latency) {
      // Transfer is the consumer is ready to accept
      if (result->ready)
        result->valid = true;

      hasData = false;
      result->data = resData;
    } else {
      // latency
      result->valid = false;
    }
  }

private:
  Op op;
  BinaryCompFunc callback;
  unsigned latency = 0;
  unsigned bitwidth;
  ChannelConsumerRW *lhs, *rhs;
  ChannelProducerRW *result;
  Join binJoin;

  APInt resData = APInt(bitwidth, 0);
  unsigned counter = 0;
  bool hasData = false;
  bool hadClock = true;
};

//===----------------------------------------------------------------------===//
// Simulator
//===----------------------------------------------------------------------===//

class Simulator {
public:
  Simulator(handshake::FuncOp funcOp) : funcOp(funcOp) {
    // Iterate through all the values of the circuit
    for (BlockArgument arg : funcOp.getArguments()) {
      associateState(arg, funcOp.getOperation(), funcOp->getLoc());
    }
    for (Operation &op : funcOp.getOps()) {
      for (OpResult res : op.getResults()) {
        associateState(res, &op, op.getLoc());
      }
    }

    // register models for all ops of the funcOp
    for (Operation &op : funcOp.getOps()) {
      associateModel(&op);
    }
  }

  void reset() {
    // Collect all the states
    for (auto &[op, model] : opModels) {
      model->exec(true);
    }
    // update old values with collected ones
    for (auto [val, state] : updaters) {
      state->update();
    }
  }

  void simulate(llvm::ArrayRef<std::string> inputArg) {
    // Counter of channels
    unsigned channelCount = 0;
    // All values that match channels
    llvm::SmallVector<Value> channelArgs;
    for (BlockArgument arg : funcOp.getArguments()) {
      if (isa<handshake::ChannelType>(arg.getType())) {
        ++channelCount;
        channelArgs.push_back(arg);
      }
    }
    // Check if the number of data values is divisible by the number of
    // ChannelStates
    assert(inputArg.size() % channelCount == 0 &&
           "The amount of input data values must be divisible by the number of "
           "the input channels!");

    // Iterate through all the collected channel values...
    size_t k = 0;
    for (auto val : channelArgs) {
      for (auto &opRes : val.getUses()) {
        auto channel = cast<TypedValue<handshake::ChannelType>>(val);
        auto *channelArg =
            OpExecutionModel<handshake::FuncOp>::getState<ChannelProducerRW>(
                val, funcOp, rws);
        // ...and update the corresponding data fields
        llvm::TypeSwitch<Type, void>(channel.getType().getDataType())
            .Case<IntegerType>([&](IntegerType intType) {
              auto argNumVal = APInt(intType.getWidth(), inputArgs[k], 10);
              channelArg->data = argNumVal;
            })
            .Case<FloatType>([&](FloatType floatType) {
              auto argNumVal =
                  APFloat(floatType.getFloatSemantics(), inputArgs[k]);
              channelArg->data = argNumVal;
            })
            .Default([&](auto) {
              emitError(channel.getLoc())
                  << "Unsuported date type " << channel.getType()
                  << ", we111 should probably report an error and stop";
            });
        ++k;
        updaters[&opRes]->update();
      }
    }
    // Set all inputs valid to true
    for (BlockArgument arg : funcOp.getArguments()) {
      for (auto &opRes : arg.getUses()) {
        updaters[&opRes]->setValid();
      }
    }

    auto returnOp = *funcOp.getOps<handshake::ReturnOp>().begin();
    auto *res =
        OpExecutionModel<handshake::ReturnOp>::getState<ChannelProducerRW>(
            returnOp.getOutputs().front(), returnOp, rws);
    unsigned eqStateCount = 0;
    // Outer loop: for clock cycles
    // Stops when return.input->valid becomes true or limit of cycles is reached
    while (true) {
      // True only once on each clkRisingEdge
      bool isClock = true;
      // Inner loop: for signal propagation within one clkRisingEdge
      // Stops when there's no change in valueStates
      while (true) {
        // Execute each model
        for (auto &[op, model] : opModels) {
          model->exec(!isClock);
        }
        bool isFin = true;
        // Check if states have changed
        for (auto [val, state] : updaters) {
          isFin = isFin && state->check();
        }
        if (isFin)
          break;
        // Update oldStates
        for (auto [val, state] : updaters) {
          state->update();
        }
        isClock = false;
      }

      if (res->valid) {
        for (auto [val, state] : updaters) {
          state->resetValid();
        }
        break;
      }

      bool isFin = true;
      // Check if states have changed
      for (auto [val, state] : updaters) {
        isFin = isFin && state->check();
      }
      if (isFin) {
        ++eqStateCount;
        if (eqStateCount >= cyclesLimit)
          break;
      }

      // At the end of each cycle reset each input's valid signal to false if
      // the corresponding ready was true
      for (auto [val, state] : updaters) {
        state->resetValid();
      }
    }
  }

  // Just a temporary function to print the results of the simulation to
  // standart output
  void printResults() {
    auto returnOp = *funcOp.getOps<handshake::ReturnOp>().begin();
    llvm::outs() << "Result:\n";
    for (auto res : returnOp.getOutputs()) {
      auto *state =
          OpExecutionModel<handshake::ReturnOp>::getState<ChannelProducerRW>(
              res, returnOp, rws);
      if (state->data.has_value())
        llvm::outs() << llvm::any_cast<APInt>(state->data);
      llvm::outs() << "\n";
    }
  }

  // Just a temporary function to print the current valueStates to
  // standart output
  void printValuesStates() {
    for (auto x : newValuesStates) {
      auto *val = x.first;
      auto *state = x.second;
      llvm::outs() << val->get() << ":\n";
      llvm::TypeSwitch<Type, void>(val->get().getType())
          .template Case<handshake::ChannelType>(
              [&](handshake::ChannelType channelType) {
                auto *cl = static_cast<ChannelState *>(state);
                llvm::outs() << cl->valid << " " << cl->ready << " ";
                if (cl->data.has_value())
                  llvm::outs() << llvm::any_cast<APInt>(cl->data);
                llvm::outs() << "\n";
              })
          .template Case<handshake::ControlType>(
              [&](handshake::ControlType controlType) {
                auto *cl = static_cast<ControlState *>(state);
                llvm::outs() << cl->valid << " " << cl->ready << "\n";
              })
          .template Default([&](auto) {
            llvm::outs() << "Value " << val
                         << " has11 unsupported type, we should probably "
                            "report an error and stop";
          });
    }
  }

  ~Simulator() {
    for (auto [_, model] : opModels)
      delete model;
    for (auto [_, state] : oldValuesStates)
      delete state;
    for (auto [_, state] : newValuesStates)
      delete state;
    for (auto [p, rw] : rws)
      delete rw;
    for (auto [_, state] : updaters)
      delete state;
  }

private:
  // Maybe need it some day, let it be the part of simulator class
  handshake::FuncOp funcOp;
  // Map for execution models
  mlir::DenseMap<Operation *, ExecutionModel *> opModels;
  // Map the stores RW API classes
  mlir::DenseMap<std::pair<OpOperand *, Operation *>, RW *> rws;
  // Map that stores the oldValuesStates we read on the current iteration (to
  // collect new outputs)
  mlir::DenseMap<OpOperand *, ValueState *> oldValuesStates;
  // And this map is for newly collected values (we can'not update
  // oldValuesStates during collection, because some component can read changed
  // values then)
  mlir::DenseMap<OpOperand *, ValueState *> newValuesStates;
  // Map to update oldValuesStates with the corresponding values of
  // newValuesStates at the end of the collection process
  mlir::DenseMap<OpOperand *, Updater *> updaters;
  // Set the number of the iterations for the simulator to execute before force
  // break
  unsigned cyclesLimit = 100;

  // Register the Model inside opNodels
  template <typename Model, typename Op, typename... Args>
  void registerModel(Op op, Args &&...modelArgs) {
    ExecutionModel *model =
        new Model(op, rws, std::forward<Args>(modelArgs)...);
    opModels.insert({op, model});
  }

  // Determine the concrete Model type
  void associateModel(Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<handshake::SinkOp>([&](handshake::SinkOp sinkOp) {
          registerModel<SinkModel, handshake::SinkOp>(sinkOp);
        })
        .Case<handshake::MulIOp>([&](handshake::MulIOp muliOp) {
          BinaryCompFunc callback = [](llvm::Any lhs, llvm::Any rhs) {
            APInt mul = any_cast<APInt>(lhs) * any_cast<APInt>(rhs);
            return mul;
          };
          registerModel<GenericBinaryOpModel<handshake::MulIOp>>(muliOp,
                                                                 callback, 4);
        })
        .Case<handshake::ReturnOp>([&](handshake::ReturnOp returnOp) {
          registerModel<ReturnModel, handshake::ReturnOp>(returnOp);
        })
        .Case<handshake::EndOp>([&](handshake::EndOp endOp) {
          registerModel<EndMemlessModel, handshake::EndOp>(endOp);
        })
        .Case<handshake::AddIOp>([&](handshake::AddIOp addiOp) {
          BinaryCompFunc callback = [](llvm::Any lhs, llvm::Any rhs) {
            APInt add = any_cast<APInt>(lhs) + any_cast<APInt>(rhs);
            return add;
          };
          registerModel<GenericBinaryOpModel<handshake::AddIOp>>(addiOp,
                                                                 callback, 0);
        })
        .Default([&](auto) {
          emitError(op->getLoc())
              << "Operation " << op
              << " has unsupported type, we should probably "
                 "report an error and stop";
        });
  }

  // Register the State state where needed (oldValueStates, newValueStates,
  // updaters, rws).
  template <typename State, typename Updater, typename Producer,
            typename Consumer, typename Ty>
  void registerState(Value val, Operation *producerOp, Ty type) {
    TypedValue<Ty> typedVal = cast<TypedValue<Ty>>(val);
    for (auto &opOp : val.getUses()) {
      State *oldState = new State(typedVal);
      State *newState = new State(typedVal);
      Updater *upd = new Updater(*oldState, *newState);
      Producer *producerRW = new Producer(typedVal, *oldState, *newState);
      Consumer *consumerRW = new Consumer(typedVal, *oldState, *newState);
      auto *consumerOp = opOp.getOwner();

      oldValuesStates.insert({&opOp, oldState});
      newValuesStates.insert({&opOp, newState});
      updaters.insert({&opOp, upd});
      rws.insert({{&opOp, producerOp}, producerRW});
      rws.insert({{&opOp, consumerOp}, consumerRW});
    }
  }

  // Determine if the state belongs to channel or control
  void associateState(Value val, Operation *producerOp, Location loc) {
    llvm::TypeSwitch<Type, void>(val.getType())
        .Case<handshake::ChannelType>([&](handshake::ChannelType channelType) {
          registerState<ChannelState, ChannelUpdater, ChannelProducerRW,
                        ChannelConsumerRW>(val, producerOp, channelType);
        })
        .Case<handshake::ControlType>([&](handshake::ControlType controlType) {
          registerState<ControlState, ControlUpdater, ControlProducerRW,
                        ControlConsumerRW>(val, producerOp, controlType);
        })
        .Default([&](auto) {
          emitError(loc) << "Value " << val
                         << " has unsupported type, we should probably "
                            "report an error and stop";
        });
  }
};

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "simulator");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFilename
                 << "': " << error.message() << "\n";
    return 1;
  }

  // We only need the Handshake and HW dialects
  MLIRContext context;
  context.loadDialect<handshake::HandshakeDialect, arith::ArithDialect>();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  // Assume a single Handshake function in the module
  handshake::FuncOp funcOp = *modOp->getOps<handshake::FuncOp>().begin();

  Simulator sim(funcOp);

  sim.reset();
  sim.simulate(inputArgs);
  sim.printResults();
}
