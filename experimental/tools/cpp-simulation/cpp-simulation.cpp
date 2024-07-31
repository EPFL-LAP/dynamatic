#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
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
  // give the corresponding RW API an access to data members
  friend struct ChannelConsumerRW;
  friend struct ChannelProducerRW;
  // give the corresponding Updater an access to data members
  friend class ChannelUpdater;

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
  // maybe some additional data signals here
};

class ControlState : public TypedValueState<handshake::ControlType> {
  // give the corresponding RW API an access to data members
  friend struct ControlConsumerRW;
  friend struct ControlProducerRW;
  // give the corresponding Updater an access to data members
  friend class ControlUpdater;

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
  virtual void update() = 0;
};

/// Templated updater class that contains pair <oldValueState, newValueState>
template <typename State>
class DoubleUpdater : public Updater {
protected:
  State &cons;
  State &prod;

public:
  DoubleUpdater(State &cons, State &prod) : cons(cons), prod(prod) {}
};

/// Update valueStates with "Channel type": valid, ready, data
class ChannelUpdater : public DoubleUpdater<ChannelState> {
public:
  ChannelUpdater(ChannelState &cons, ChannelState &prod)
      : DoubleUpdater<ChannelState>(cons, prod) {}
  void update() override {
    cons.valid = prod.valid;
    cons.ready = prod.ready;
    cons.data = prod.data;
  }
};

/// Update valueStates with "Control type": valid, ready
class ControlUpdater : public DoubleUpdater<ControlState> {
public:
  ControlUpdater(ControlState &cons, ControlState &prod)
      : DoubleUpdater<ControlState>(cons, prod) {}
  void update() override {
    cons.valid = prod.valid;
    cons.ready = prod.ready;
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

class ExecutionModel {
public:
  ExecutionModel(Operation *op) : op(op) {}
  virtual void
  exec(bool isReset,
       mlir::DenseMap<std::pair<Value, Operation *>, RW *> &rws) = 0;

  // Function to get the concrete RW API class (State = ChannelProducerRW /
  // ChannelConsumerRW / ControlProducerRW / ControlConsumerRW)
  template <typename State>
  State *getState(RW *rw) {
    return static_cast<State *>(rw);
  }

  virtual ~ExecutionModel() {}

protected:
  Operation *op;
};

template <typename Op>
class OpExecutionModel : public ExecutionModel {
public:
  OpExecutionModel(Op op) : ExecutionModel(op.getOperation()) {}

protected:
  Op getOperation() { return cast<Op>(op); }
};

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//
/// Mostly components required for the internal state

class Mul4Stage {
public:
  Mul4Stage(unsigned bitwidth) : bitwidth(bitwidth) {}
  void exec(bool isReset, bool ce, const APInt &a, const APInt &b, APInt &p) {
    if (isReset) {
      // here for example we do nothing on the reset
    } else if (ce) {
      p = a * b;
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
  void exec(bool isReset, bool validIn, bool readyIn, bool &validOut) {
    if (isReset) {
      if (size == 1) {
        validOut = validIn;
      } else {
        validOut = false;
      }
    } else if (readyIn) {
      validOut = validIn;
    }
  }

private:
  unsigned size;
};

class OEHB {
public:
  OEHB(unsigned dataWidth) : dataWidth(dataWidth) {}

  void exec(bool isReset, const Any &ins, bool insValid, bool outsReady,
            bool &insReady, Any &outs, bool &outsValid) {
    if (isReset) {
      outsValid = false;
      insReady = true;
      outs = 0;
    } else {
      outsValid = insValid;
      insReady = !insValid || outsReady;
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
      insReady = 1;
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
  SinkModel(handshake::SinkOp sinkOp)
      : OpExecutionModel<handshake::SinkOp>(sinkOp), op(sinkOp) {}

  void exec(bool isReset,
            mlir::DenseMap<std::pair<Value, Operation *>, RW *> &rws) override {
    auto *ins = getState<ControlConsumerRW>(rws[{op.getOperand(), op}]);
    ins->ready = true;
  }

private:
  handshake::SinkOp op;
};

class MulIModel : public OpExecutionModel<handshake::MulIOp> {
public:
  using OpExecutionModel<handshake::MulIOp>::OpExecutionModel;
  MulIModel(handshake::MulIOp muliOp)
      : OpExecutionModel<handshake::MulIOp>(muliOp), op(muliOp),
        bitwidth(cast<handshake::ChannelType>(muliOp.getLhs().getType())
                     .getDataBitWidth()),
        muliJoin(2), muliMul4Stage(bitwidth), muliDelayBuffer(3),
        muliOEHB(bitwidth) {}

  void exec(bool isReset,
            mlir::DenseMap<std::pair<Value, Operation *>, RW *> &rws) override {
    auto *lhs = getState<ChannelConsumerRW>(rws[{op.getLhs(), op}]);
    auto *rhs = getState<ChannelConsumerRW>(rws[{op.getRhs(), op}]);
    auto *result = getState<ChannelProducerRW>(rws[{op.getResult(), op}]);
    auto lhsData = llvm::any_cast<APInt>(lhs->data);
    auto rhsData = llvm::any_cast<APInt>(rhs->data);

    bool joinValid = false, buffValid = false, oehbReady = false;
    Any oehbDataIn, oehbDataOut;
    APInt resData;

    std::vector<bool> insValid{lhs->valid, rhs->valid};
    std::vector<std::reference_wrapper<bool>> insReady{lhs->ready, rhs->ready};

    for (int i = 0; i < 10; ++i) {
      muliJoin.exec(isReset, insValid, oehbReady, insReady, joinValid);
      muliMul4Stage.exec(isReset, oehbReady, lhsData, rhsData, resData);
      result->data = resData;
      muliDelayBuffer.exec(isReset, joinValid, oehbReady, buffValid);
      muliOEHB.exec(isReset, oehbDataIn, buffValid, result->ready, oehbReady,
                    oehbDataOut, result->valid);
    }
  }

private:
  handshake::MulIOp op;
  unsigned bitwidth;
  Join muliJoin;
  Mul4Stage muliMul4Stage;
  DelayBuffer muliDelayBuffer;
  OEHB muliOEHB;
};

class ReturnModel : public OpExecutionModel<handshake::ReturnOp> {
public:
  using OpExecutionModel<handshake::ReturnOp>::OpExecutionModel;
  ReturnModel(handshake::ReturnOp returnOp)
      : OpExecutionModel<handshake::ReturnOp>(returnOp), op(returnOp),
        returnTEHB() {

    auto temp =
        cast<handshake::ChannelType>(returnOp.getInputs().front().getType());
    datawidth = temp.getDataBitWidth();
  }

  void exec(bool isReset,
            mlir::DenseMap<std::pair<Value, Operation *>, RW *> &rws) override {
    auto *ins = getState<ChannelConsumerRW>(rws[{op.getInputs().front(), op}]);
    auto *outs =
        getState<ChannelProducerRW>(rws[{op.getOutputs().front(), op}]);
    bool regEnable = false, regNotFull = false;
    Any dataReg;

    for (int i = 0; i < 10; ++i) {
      returnTEHB.exec(isReset, ins->valid, outs->ready, regNotFull,
                      outs->valid);
      if (isReset) {
        dataReg = 0;
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
  }

private:
  handshake::ReturnOp op;
  unsigned datawidth;
  TEHBDataless returnTEHB;
};

class EndModel : public OpExecutionModel<handshake::EndOp> {
public:
  using OpExecutionModel<handshake::EndOp>::OpExecutionModel;
  EndModel(handshake::EndOp endOp)
      : OpExecutionModel<handshake::EndOp>(endOp), op(endOp) {
    auto temp =
        cast<handshake::ChannelType>(endOp.getInputs().front().getType());
    bitwidth = temp.getDataBitWidth();
  }

  void exec(bool isReset,
            mlir::DenseMap<std::pair<Value, Operation *>, RW *> &rws) override {

    auto *ins = getState<ChannelConsumerRW>(rws[{op.getInputs().front(), op}]);
    auto *outs =
        getState<ChannelProducerRW>(rws[{op.getReturnValues().front(), op}]);

    ins->ready = ins->valid && outs->ready;
    outs->valid = ins->valid;
    outs->data = ins->data;
  }

private:
  handshake::EndOp op;
  unsigned bitwidth;
};

//===----------------------------------------------------------------------===//
// Simulator
//===----------------------------------------------------------------------===//

class Simulator {
public:
  Simulator(handshake::FuncOp funcOp) : funcOp(funcOp) {
    // Iterate through all the values of the circuit
    for (BlockArgument arg : funcOp.getArguments()) {
      // get the consumer op of the value
      auto *user = *arg.getUsers().begin();
      associateState(arg, funcOp.getOperation(), user, funcOp->getLoc());
    }

    for (Operation &op : funcOp.getOps()) {
      for (OpResult res : op.getResults()) {
        // get the consumer op of the value
        auto *user = *res.getUsers().begin();
        associateState(res, &op, user, op.getLoc());
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
      model->exec(true, rws);
    }
    // update old values with colldected ones
    for (auto [val, state] : updaters) {
      state->update();
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
  mlir::DenseMap<std::pair<Value, Operation *>, RW *> rws;
  // Map that stores the oldValuesStates we read on the current iteration (to
  // collect new outputs)
  mlir::DenseMap<Value, ValueState *> oldValuesStates;
  // And this map is for newly collected values (we can'not update
  // oldValuesStates during collection, because some component can read changed
  // values then)
  mlir::DenseMap<Value, ValueState *> newValuesStates;
  // Map to update oldValuesStates with the corresponding values of
  // newValuesStates at the end of the collection process
  mlir::DenseMap<Value, Updater *> updaters;

  // Register the Model inside opNodels
  template <typename Model, typename Op, typename... Args>
  void registerModel(Op op, Args &&...modelArgs) {
    ExecutionModel *model = new Model(op, std::forward<Args>(modelArgs)...);
    opModels.insert({op, model});
  }

  // Determine the concrete Model type
  void associateModel(Operation *op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::SinkOp>([&](handshake::SinkOp sinkOp) {
          registerModel<SinkModel, handshake::SinkOp>(sinkOp);
        })
        .Case<handshake::MulIOp>([&](handshake::MulIOp muliOp) {
          registerModel<MulIModel, handshake::MulIOp>(muliOp);
        })
        .Case<handshake::ReturnOp>([&](handshake::ReturnOp returnOp) {
          registerModel<ReturnModel, handshake::ReturnOp>(returnOp);
        })
        .Case<handshake::EndOp>([&](handshake::EndOp endOp) {
          registerModel<EndModel, handshake::EndOp>(endOp);
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
  void registerState(Value val, Operation *producerOp, Operation *consumerOp,
                     Ty type) {
    TypedValue<Ty> typedVal = cast<TypedValue<Ty>>(val);
    State *oldState = new State(typedVal);
    State *newState = new State(typedVal);
    Updater *upd = new Updater(*oldState, *newState);
    Producer *producerRW = new Producer(typedVal, *oldState, *newState);
    Consumer *consumerRW = new Consumer(typedVal, *oldState, *newState);

    oldValuesStates.insert({val, oldState});
    newValuesStates.insert({val, newState});
    updaters.insert({val, upd});
    rws.insert({{val, producerOp}, producerRW});
    rws.insert({{val, consumerOp}, consumerRW});
  }

  // Determine if the state belongs to channel or control
  void associateState(Value val, Operation *producerOp, Operation *consumerOp,
                      Location loc) {
    llvm::TypeSwitch<Type, void>(val.getType())
        .Case<handshake::ChannelType>([&](handshake::ChannelType channelType) {
          registerState<ChannelState, ChannelUpdater, ChannelProducerRW,
                        ChannelConsumerRW>(val, producerOp, consumerOp,
                                           channelType);
        })
        .Case<handshake::ControlType>([&](handshake::ControlType controlType) {
          registerState<ControlState, ControlUpdater, ControlProducerRW,
                        ControlConsumerRW>(val, producerOp, consumerOp,
                                           controlType);
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
}