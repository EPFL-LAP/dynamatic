#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <string>
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

// Binding for llvm::Any that lets the user check if the value was changed
// It's required for updaters to stop the internal loop
struct Data {
  llvm::Any data;
  llvm::hash_code hash;
  unsigned bitwidth;

  Data() = default;

  Data(const APInt &other) {
    data = other;
    hash = llvm::hash_value(other);
    bitwidth = other.getBitWidth();
  }
  Data(const APFloat &other) {
    data = other;
    hash = llvm::hash_value(other);
    bitwidth = other.getSizeInBits(other.getSemantics());
  }

  Data &operator=(const APInt &other) {
    this->data = other;
    hash = llvm::hash_value(other);
    bitwidth = other.getBitWidth();
    return *this;
  }

  Data &operator=(const APFloat &other) {
    this->data = other;
    hash = llvm::hash_value(other);
    bitwidth = other.getSizeInBits(other.getSemantics());
    return *this;
  }

  bool hasValue() const { return data.has_value(); }

  template <class T>
  friend T dataCast(const Data &value);
  template <class T>
  friend T dataCast(Data &value);
  template <class T>
  friend T dataCast(Data &&value);
  template <class T>
  friend const T *dataCast(const Data *value);
  template <class T>
  friend T *dataCast(Data *value);
};

template <class T>
T dataCast(const Data &value) {
  return llvm::any_cast<T>(value.data);
}
template <class T>
T dataCast(Data &value) {
  return llvm::any_cast<T>(value.data);
}
template <class T>
T dataCast(Data &&value) {
  return llvm::any_cast<T>(value.data);
}
template <class T>
const T *dataCast(const Data *value) {
  return llvm::any_cast<T>(value->data);
}
template <class T>
T *dataCast(Data *value) {
  return llvm::any_cast<T>(value->data);
}

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

public:
  using TypedValueState<handshake::ChannelType>::TypedValueState;

  ChannelState(TypedValue<handshake::ChannelType> channel)
      : TypedValueState<handshake::ChannelType>(channel) {

    llvm::TypeSwitch<mlir::Type>(channel.getType().getDataType())
        .Case<IntegerType>([&](IntegerType intType) {
          data = APInt(std::max(1, (int)intType.getWidth()), 0,
                       intType.isSigned());
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
  Data data = {};
  // Maybe some additional data signals here
};

class ControlState : public TypedValueState<handshake::ControlType> {
  // Give the corresponding RW API an access to data members
  friend struct ControlConsumerRW;
  friend struct ControlProducerRW;
  // Give the corresponding Updater an access to data members
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
    return newState.valid == oldState.valid &&
           newState.ready == oldState.ready &&
           newState.data.hash == oldState.data.hash;
  }
  void setValid() override {
    newState.valid = true;
    update();
  }
  void resetValid() override {
    if (newState.ready) {
      newState.valid = false;
    }
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
  RW() = default;
  virtual ~RW() {};
};

class ProducerRW : public RW {
protected:
  enum ProducerDescendants { D_ChannelProducerRW, D_ControlProducerRW };

public:
  bool &valid;
  const bool &ready;

public:
  ProducerDescendants getType() const { return prod; }
  ProducerRW(bool &valid, const bool &ready,
             ProducerDescendants p = D_ControlProducerRW)
      : valid(valid), ready(ready), prod(p) {}
  ProducerRW(ProducerRW &p) : valid(p.valid), ready(p.ready), prod(p.prod) {}

  virtual ~ProducerRW() {}

private:
  const ProducerDescendants prod;
};

class ConsumerRW : public RW {
public:
  const bool &valid;
  bool &ready;

protected:
  enum ConsumerDescendants { D_ChannelConsumerRW, D_ControlConsumerRW };

public:
  ConsumerDescendants getType() const { return cons; }

  ConsumerRW(const bool &valid, bool &ready,
             ConsumerDescendants c = D_ControlConsumerRW)
      : valid(valid), ready(ready), cons(c) {}

  ConsumerRW(ConsumerRW &c) : valid(c.valid), ready(c.ready), cons(c.cons) {}

  virtual ~ConsumerRW() {};

private:
  const ConsumerDescendants cons;
};

////--- Control (no data)

/// In this case the user can change the ready signal, but has
/// ReadOnly access to the valid one.
struct ControlConsumerRW : public ConsumerRW {
  ControlConsumerRW(ControlState &reader, ControlState &writer)
      : ConsumerRW(reader.valid, writer.ready, D_ControlConsumerRW) {}

  static bool classof(const ConsumerRW *c) {
    return c->getType() == D_ControlConsumerRW;
  }
};

/// In this case the user can change the valid signal, but has
/// ReadOnly access to the ready one.
struct ControlProducerRW : public ProducerRW {
  ControlProducerRW(ControlState &reader, ControlState &writer)
      : ProducerRW(writer.valid, reader.ready, D_ControlProducerRW) {}

  static bool classof(const ProducerRW *c) {
    return c->getType() == D_ControlProducerRW;
  }
};

////--- Channel (data)

/// In this case the user can change the valid signal, but has ReadOnly access
/// to the valid and data ones.
struct ChannelConsumerRW : public ConsumerRW {
  const Data &data;

  ChannelConsumerRW(ChannelState &reader, ChannelState &writer)
      : ConsumerRW(reader.valid, writer.ready, D_ChannelConsumerRW),
        data(reader.data) {}

  static bool classof(const ConsumerRW *c) {
    return c->getType() == D_ChannelConsumerRW;
  }
};

/// In this case the user can change the valid and data signals, but has
/// ReadOnly access to the ready one.
struct ChannelProducerRW : public ProducerRW {
  Data &data;

  ChannelProducerRW(ChannelState &reader, ChannelState &writer)
      : ProducerRW(writer.valid, reader.ready, D_ChannelProducerRW),
        data(writer.data) {}

  static bool classof(const ProducerRW *c) {
    return c->getType() == D_ChannelProducerRW;
  }
};

//===----------------------------------------------------------------------===//
// Readers & writers
//===----------------------------------------------------------------------===//
/// The following structures are used to represent components that can be
/// either datafull or dataless. They store a pointer to Data and some
/// user-friendly functions.

/// A struct to represent consumer's data
struct ConsumerData {
  Data const *data = nullptr;
  unsigned dataWidth = 0;

  ConsumerData(ConsumerRW *ins) {
    if (auto *p = dyn_cast<ChannelConsumerRW>(ins)) {
      data = &p->data;
      dataWidth = data->bitwidth;
    } else
      data = nullptr;
  }

  bool hasValue() const { return data != nullptr; }
};

/// A struct to represent producer's data
struct ProducerData {
  Data *data = nullptr;
  unsigned dataWidth = 0;

  ProducerData(ProducerRW *outs) {
    if (auto *p = dyn_cast<ChannelProducerRW>(outs)) {
      data = &p->data;
    } else
      data = nullptr;
  }

  ProducerData &operator=(const ConsumerData &other) {
    if (data)
      *data = *other.data;
    return *this;
  }

  bool hasValue() const { return data != nullptr; }
};

//===----------------------------------------------------------------------===//
// Execution Model
//===----------------------------------------------------------------------===//
/// Classes to represent execution models
/// Base class

class ExecutionModel {

public:
  ExecutionModel(Operation *op) : op(op) {}
  // Execute the model: reset if isClkRisingEdge == true, clock iteration
  // otherwise
  virtual void reset() = 0;
  virtual void exec(bool isClkRisingEdge) = 0;
  virtual void printStates() = 0;

  virtual ~ExecutionModel() {}

protected:
  Operation *op;
};

/// Typed execution model
template <typename Op>
class OpExecutionModel : public ExecutionModel {
public:
  OpExecutionModel(Op op) : ExecutionModel(op.getOperation()) {}

protected:
  Op getOperation() { return cast<Op>(op); }

  // Get exact RW state type (ChannelProducerRW / ChannelConsumerRW /
  // ControlProducerRW / ControlConsumerRW)
  template <typename State>
  static State *getState(Value val, mlir::DenseMap<Value, RW *> &rws) {
    return static_cast<State *>(rws[val]);
  }

  // Just a temporary function to print models' states
  template <typename State, typename Data>
  void printValue(const std::string &name, State *ins, Data *insData) {
    llvm::outs() << name << ": ";
    if (insData) {
      APInt t = dataCast<APInt>(*insData);
      auto val = t.reverseBits().getZExtValue();
      for (unsigned i = 0; i < t.getBitWidth(); ++i) {
        llvm::outs() << (val & 1);
        val >>= 1;
      }
      llvm::outs() << " ";
    }

    llvm::outs() << ins->valid << " " << ins->ready << "\n";
  }
};

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//
/// Components required for the internal state.

class ForkSupport {
public:
  ForkSupport(unsigned size, unsigned datawidth = 0)
      : size(size), datawidth(datawidth) {
    transmitValue.resize(size, false);
    keepValue.resize(size, false);
    blockStopArray.resize(size, false);
  }

  void resetDataless(ConsumerRW *ins, std::vector<ProducerRW *> &outs) {
    for (unsigned i = 0; i < size; ++i) {
      transmitValue[i] = true;
      keepValue[i] = !outs[i]->ready;
      outs[i]->valid = ins->valid;
      blockStopArray[i] = !outs[i]->ready;
    }

    // or_n
    anyBlockStop = false;
    for (bool c : blockStopArray)
      anyBlockStop = anyBlockStop || c;

    ins->ready = !anyBlockStop;
    backpressure = ins->valid && anyBlockStop;
  }

  void execDataless(bool isClkRisingEdge, ConsumerRW *ins,
                    std::vector<ProducerRW *> &outs) {
    for (unsigned i = 0; i < outs.size(); ++i) {
      keepValue[i] = !outs[i]->ready && transmitValue[i];
      if (isClkRisingEdge)
        transmitValue[i] = keepValue[i] || !backpressure;
      outs[i]->valid = transmitValue[i] && ins->valid;
      blockStopArray[i] = keepValue[i];
    }

    // or_n
    anyBlockStop = false;
    for (bool c : blockStopArray)
      anyBlockStop = anyBlockStop || c;

    ins->ready = !anyBlockStop;
    backpressure = ins->valid && anyBlockStop;
  }

  void reset(ConsumerRW *ins, std::vector<ProducerRW *> &outs,
             const ConsumerData &insData, std::vector<ProducerData> &outsData) {
    resetDataless(ins, outs);
    for (auto &out : outsData)
      out = insData;
  }

  void exec(bool isClkRisingEdge, ConsumerRW *ins,
            std::vector<ProducerRW *> &outs, const ConsumerData &insData,
            std::vector<ProducerData> &outsData) {
    execDataless(isClkRisingEdge, ins, outs);
    for (auto &out : outsData)
      out = insData;
  }

private:
  unsigned size;
  // eager fork
  std::vector<bool> transmitValue, keepValue;
  // fork dataless
  std::vector<bool> blockStopArray;
  bool anyBlockStop = false, backpressure = false;

public:
  unsigned datawidth;
};

class JoinSupport {
public:
  JoinSupport(unsigned size) : size(size) {}

  void exec(std::vector<ConsumerRW *> &ins, ProducerRW *outs) {
    outs->valid = true;
    for (unsigned i = 0; i < size; ++i)
      outs->valid = outs->valid && ins[i]->valid;

    for (unsigned i = 0; i < size; ++i) {
      ins[i]->ready = outs->ready;
      for (unsigned j = 0; j < size; ++j)
        if (i != j)
          ins[i]->ready = ins[i]->ready && ins[j]->valid;
    }
  }

private:
  unsigned size;
};

// Not used at the moment
class DelayBuffer {
public:
  DelayBuffer(unsigned size = 32) : size(size) { regs.resize(size, false); }
  void reset(const bool &validIn, const bool &readyIn, bool &validOut) {
    for (unsigned i = 0; i < size; ++i) {
      if (i == 0) {
        regs[i] = validIn;
      } else if (i > 0) {
        regs[i] = false;
      }
    }
    validOut = regs[size - 1];
  }
  void exec(bool isClkRisingEdge, const bool &validIn, const bool &readyIn,
            bool &validOut) {
    for (unsigned i = 0; i < size; ++i) {
      if (i == 0) {
        if (isClkRisingEdge && readyIn)
          regs[i] = validIn;
      } else if (i > 0) {
        if (isClkRisingEdge && readyIn)
          regs[i] = regs[i - 1];
      }
    }
    validOut = regs[size - 1];
  }

private:
  unsigned size;
  std::vector<bool> regs;
};

class OEHBSupport {
public:
  OEHBSupport(unsigned datawidth = 0) : datawidth(datawidth) {}

  void resetDataless(ConsumerRW *ins, ProducerRW *outs) {
    outputValid = false;
    ins->ready = true;
    outs->valid = false;
  }

  void execDataless(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs) {
    if (isClkRisingEdge)
      outputValid = ins->valid || (outputValid && !outs->ready);

    ins->ready = !outputValid || outs->ready;
    outs->valid = outputValid;
  }

  void reset(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
             Data *outsData) {
    resetDataless(ins, outs);
    if (insData)
      *outsData = APInt(datawidth, 0);
    regEn = ins->ready && ins->valid;
  }

  void exec(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
            const Data *insData, Data *outsData) {
    execDataless(isClkRisingEdge, ins, outs);
    if (isClkRisingEdge && regEn && insData)
      *outsData = *insData;
    regEn = ins->ready && ins->valid;
  }

  unsigned datawidth;

private:
  // oehb dataless
  bool outputValid = false;
  // oehb datafull
  bool regEn = false;
};

class TEHBSupport {
public:
  TEHBSupport(unsigned datawidth = 0) : datawidth(datawidth) {}
  void resetDataless(ConsumerRW *ins, ProducerRW *outs) {
    outputValid = ins->valid;
    fullReg = false;
    ins->ready = true;
    outs->valid = ins->valid;
  }

  void execDataless(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs) {
    if (isClkRisingEdge)
      fullReg = outputValid && !outs->ready;
    outputValid = ins->valid || fullReg;
    ins->ready = !fullReg;
    outs->valid = outputValid;
  }

  void reset(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
             Data *outsData) {
    if (insData)
      resetDataFull(ins, outs, insData, outsData);
    else
      resetDataless(ins, outs);
  }

  void exec(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
            const Data *insData, Data *outsData) {
    if (insData)
      execDataFull(isClkRisingEdge, ins, outs, insData, outsData);
    else
      execDataless(isClkRisingEdge, ins, outs);
  }

  unsigned datawidth = 0;

private:
  // tehb dataless
  bool fullReg = false, outputValid = false;
  // tehb datafull
  bool regNotFull = false, regEnable = false;
  Data dataReg;

  void resetDataFull(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
                     Data *outsData) {
    regEnable = regNotFull && ins->valid && !outs->ready;

    // tehb dataless
    ConsumerRW c(ins->valid, regNotFull);
    resetDataless(&c, outs);

    // process datareg
    dataReg = APInt(datawidth, 0);

    if (regNotFull)
      *outsData = *insData;
    else
      *outsData = dataReg;

    ins->ready = regNotFull;
  };

  void execDataFull(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
                    const Data *insData, Data *outsData) {
    regEnable = regNotFull && ins->valid && !outs->ready;

    // tehb dataless
    ConsumerRW c(ins->valid, regNotFull);
    execDataless(isClkRisingEdge, &c, outs);

    // process datareg
    if (isClkRisingEdge && regEnable)
      dataReg = *insData;

    // process (regNotFull, dataReg, ins)
    if (regNotFull)
      *outsData = *insData;
    else
      *outsData = dataReg;

    ins->ready = regNotFull;
  }
};

//===----------------------------------------------------------------------===//
// Handshake
//===----------------------------------------------------------------------===//
/// Handshake components. Some of them can represent both datafull and dataless
/// options.

/// Example of a model that can be initialized with o without data.
class BranchModel : public OpExecutionModel<handshake::BranchOp> {
public:
  using OpExecutionModel<handshake::BranchOp>::OpExecutionModel;
  BranchModel(handshake::BranchOp branchOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::BranchOp>(branchOp), op(branchOp),
        // get the exact structure for the particular value
        ins(getState<ConsumerRW>(op.getOperand(), subset)),
        outs(getState<ProducerRW>(op.getResult(), subset)),
        // initialize data (nullptr if dataless)
        insData(ins), outsData(outs) {}

  void reset() override {
    outs->valid = ins->valid;
    ins->ready = outs->ready;
    outsData = insData;
  };

  void exec(bool isClkRisingEdge) override { reset(); }

  void printStates() override {
    printValue<ConsumerRW, const Data>("ins", ins, insData.data);
    printValue<ProducerRW, Data>("outs", outs, outsData.data);
  }

private:
  handshake::BranchOp op;

  // ports
  ConsumerRW *ins;
  ProducerRW *outs;

  ConsumerData insData;
  ProducerData outsData;
};

class CondBranchModel
    : public OpExecutionModel<handshake::ConditionalBranchOp> {
public:
  using OpExecutionModel<handshake::ConditionalBranchOp>::OpExecutionModel;
  CondBranchModel(handshake::ConditionalBranchOp condBranchOp,
                  mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::ConditionalBranchOp>(condBranchOp),
        op(condBranchOp),
        data(getState<ConsumerRW>(op.getDataOperand(), subset)),
        condition(
            getState<ChannelConsumerRW>(op.getConditionOperand(), subset)),
        trueOut(getState<ProducerRW>(op.getTrueResult(), subset)),
        falseOut(getState<ProducerRW>(op.getFalseResult(), subset)),
        dataData(data), trueOutData(trueOut), falseOutData(falseOut),
        condBrJoin(2) {}

  void reset() override {
    auto k = dataCast<APInt>(condition->data);

    auto cond = k.getBoolValue();
    // join
    std::vector<ConsumerRW *> insJoin = {data, condition};
    ProducerRW outsJoin(brInpValid,
                        (falseOut->ready && !cond) || (trueOut->ready && cond));
    condBrJoin.exec(insJoin, &outsJoin);

    trueOut->valid = cond && brInpValid;
    falseOut->valid = !cond && brInpValid;
    trueOutData = dataData;
    falseOutData = dataData;
  }

  void exec(bool isClkRisingEdge) override { reset(); }

  void printStates() override {
    printValue<ConsumerRW, const Data>("data", data, dataData.data);
    printValue<ChannelConsumerRW, const Data>("condition", condition,
                                              &condition->data);
    printValue<ProducerRW, Data>("trueOut", trueOut, trueOutData.data);
    printValue<ProducerRW, Data>("falseOut", falseOut, falseOutData.data);
  }

private:
  handshake::ConditionalBranchOp op;

  // ports
  ConsumerRW *data;
  ChannelConsumerRW *condition;
  ProducerRW *trueOut, *falseOut;

  ConsumerData dataData;
  ProducerData trueOutData, falseOutData;

  // cond br dataless
  bool brInpValid = false;

  // internal components
  JoinSupport condBrJoin;
};

class ConstantModel : public OpExecutionModel<handshake::ConstantOp> {
public:
  using OpExecutionModel<handshake::ConstantOp>::OpExecutionModel;
  ConstantModel(handshake::ConstantOp constOp,
                mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::ConstantOp>(constOp), op(constOp),
        value(dyn_cast<mlir::IntegerAttr>(op.getValue()).getValue()) {
    ctrl = getState<ControlConsumerRW>(op.getCtrl(), subset);
    outs = getState<ChannelProducerRW>(op.getResult(), subset);
  }

  void reset() override {
    outs->data = value;
    outs->valid = ctrl->valid;
    ctrl->ready = outs->ready;
  }

  void exec(bool isClkRisingEdge) override { reset(); }

  void printStates() override {
    llvm::outs() << "ctrl: " << ctrl->valid << " " << ctrl->ready << "\n";
    printValue<ChannelProducerRW, Data>("outs", outs, &outs->data);
  }

private:
  handshake::ConstantOp op;

  // ports
  Data value;
  ControlConsumerRW *ctrl;
  ChannelProducerRW *outs;
};

class ControlMergeModel : public OpExecutionModel<handshake::ControlMergeOp> {
public:
  using OpExecutionModel<handshake::ControlMergeOp>::OpExecutionModel;
  ControlMergeModel(handshake::ControlMergeOp cMergeOp,
                    mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::ControlMergeOp>(cMergeOp), op(cMergeOp),
        size(op->getNumOperands()),
        indexWidth(op.getIndex().getType().getDataBitWidth()),
        cMergeTEHB(indexWidth), cMergeFork(2),
        outs(getState<ProducerRW>(op.getResult(), subset)),
        index(getState<ChannelProducerRW>(op.getIndex(), subset)),
        outsData(outs), insTEHB(dataAvailable, tehbOutReady),
        outsTEHB(tehbOutValid, readyToFork),
        insFork(tehbOutValid, readyToFork) {
    for (auto oper : op->getOperands())
      ins.push_back(getState<ConsumerRW>(oper, subset));

    outsFork = {outs, index};

    for (unsigned i = 0; i < size; ++i)
      insData.emplace_back(ins[i]);

    cMergeFork.datawidth = outsData.dataWidth;
  }

  void reset() override {
    indexTEHB = APInt(indexWidth, 0);

    // process (ins_valid)
    for (unsigned i = 0; i < size; ++i)
      if (ins[i]->valid) {
        indexTEHB = APInt(indexWidth, i);
        break;
      }

    // mergeNotehbDataless
    dataAvailable = false;
    for (auto &in : ins)
      dataAvailable = dataAvailable || in->valid;

    // tehb
    cMergeTEHB.reset(&insTEHB, &outsTEHB, &indexTEHB, &index->data);

    // mergeNotehbDataless
    for (auto &in : ins)
      in->ready = tehbOutReady;

    // fork dataless
    cMergeFork.resetDataless(&insFork, outsFork);

    outsData = insData[dataCast<APInt>(index->data).getZExtValue()];
  }

  void exec(bool isClkRisingEdge) override {
    indexTEHB = APInt(indexWidth, 0);

    // process (ins_valid)
    for (unsigned i = 0; i < size; ++i)
      if (ins[i]->valid) {
        indexTEHB = APInt(indexWidth, i);
        break;
      }

    // mergeNotehbDataless
    dataAvailable = false;
    for (auto &in : ins)
      dataAvailable = dataAvailable || in->valid;

    // tehb
    cMergeTEHB.exec(isClkRisingEdge, &insTEHB, &outsTEHB, &indexTEHB,
                    &index->data);

    // mergeNotehbDataless
    for (auto &in : ins)
      in->ready = tehbOutReady;

    // fork dataless
    cMergeFork.execDataless(isClkRisingEdge, &insFork, outsFork);

    outsData = insData[dataCast<APInt>(index->data).getZExtValue()];
  }

  void printStates() override {
    for (unsigned i = 0; i < size; ++i)
      printValue<ConsumerRW, const Data>("ins", ins[i], insData[i].data);
    printValue<ProducerRW, Data>("outs", outs, outsData.data);
    printValue<ChannelProducerRW, Data>("index", index, &index->data);
  }

private:
  handshake::ControlMergeOp op;

  // parameters
  unsigned size, indexWidth;

  // internal components
  TEHBSupport cMergeTEHB;
  ForkSupport cMergeFork;

  // ports
  std::vector<ConsumerRW *> ins;
  ProducerRW *outs;
  ChannelProducerRW *index;

  std::vector<ConsumerData> insData;
  ProducerData outsData;

  // control merge dataless
  Data indexTEHB = APInt(indexWidth, 0);
  bool dataAvailable = false, readyToFork = false, tehbOutValid = false,
       tehbOutReady = false;
  ConsumerRW insTEHB;
  ProducerRW outsTEHB;
  ConsumerRW insFork;
  std::vector<ProducerRW *> outsFork;
};

class ForkModel : public OpExecutionModel<handshake::ForkOp> {
public:
  using OpExecutionModel<handshake::ForkOp>::OpExecutionModel;
  ForkModel(handshake::ForkOp forkOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::ForkOp>(forkOp), op(forkOp),
        size(op->getNumResults()), forkSupport(size),
        ins(getState<ConsumerRW>(op.getOperand(), subset)), insData(ins) {
    for (unsigned i = 0; i < size; ++i)
      outs.push_back(getState<ProducerRW>(op->getResult(i), subset));

    for (unsigned i = 0; i < size; ++i)
      outsData.emplace_back(outs[i]);

    forkSupport.datawidth = insData.dataWidth;
  }

  void reset() override { forkSupport.reset(ins, outs, insData, outsData); }

  void exec(bool isClkRisingEdge) override {
    forkSupport.exec(isClkRisingEdge, ins, outs, insData, outsData);
  }

  void printStates() override {
    printValue<ConsumerRW, const Data>("ins", ins, insData.data);
    for (unsigned i = 0; i < size; ++i)
      printValue<ProducerRW, Data>("outs", outs[i], outsData[i].data);
  }

private:
  handshake::ForkOp op;

  // parameters
  unsigned size;
  ForkSupport forkSupport;

  // ports
  ConsumerRW *ins;
  std::vector<ProducerRW *> outs;

  ConsumerData insData;
  std::vector<ProducerData> outsData;
};

// Never used but let it be
class JoinModel : public OpExecutionModel<handshake::JoinOp> {
public:
  using OpExecutionModel<handshake::JoinOp>::OpExecutionModel;
  JoinModel(handshake::JoinOp joinOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::JoinOp>(joinOp), op(joinOp),
        join(op->getNumOperands()) {
    for (auto oper : op->getOperands())
      ins.push_back(getState<ConsumerRW>(oper, subset));
    outs = getState<ProducerRW>(op.getResult(), subset);
  }

  void reset() override { join.exec(ins, outs); }

  void exec(bool isClkRisingEdge) override { reset(); }

  void printStates() override {
    for (auto *in : ins)
      llvm::outs() << "Ins: " << in->valid << " " << in->ready << "\n";
    llvm::outs() << "Outs: " << outs->valid << " " << outs->ready << "\n";
  }

private:
  handshake::JoinOp op;

  // ports
  std::vector<ConsumerRW *> ins;
  ProducerRW *outs;

  // internal components
  JoinSupport join;
};

class LazyForkModel : public OpExecutionModel<handshake::LazyForkOp> {
public:
  using OpExecutionModel<handshake::LazyForkOp>::OpExecutionModel;
  LazyForkModel(handshake::LazyForkOp lazyForkOp,
                mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::LazyForkOp>(lazyForkOp), op(lazyForkOp),
        size(op->getNumResults()),
        ins(getState<ConsumerRW>(op.getOperand(), subset)), insData(ins) {

    for (unsigned i = 0; i < size; ++i)
      outs.push_back(getState<ProducerRW>(op->getResult(i), subset));

    for (unsigned i = 0; i < size; ++i)
      outsData.emplace_back(outs[i]);
  }

  void reset() override {
    ins->ready = true;
    for (unsigned i = 0; i < size; ++i) {
      bool tempReady = true;
      for (unsigned j = 0; j < size; ++j)
        if (i != j)
          tempReady = tempReady && outs[j]->ready;

      ins->ready = ins->ready && outs[i]->ready;
      outs[i]->valid = ins->valid && tempReady;
    }
    for (unsigned i = 0; i < size; ++i)
      outsData[i] = insData;
  }

  void exec(bool isClkRisingEdge) override { reset(); }

  void printStates() override {
    printValue<ConsumerRW, const Data>("ins", ins, insData.data);
    for (unsigned i = 0; i < size; ++i)
      printValue<ProducerRW, Data>("outs", outs[i], outsData[i].data);
  }

private:
  handshake::ForkOp op;

  // parameters
  unsigned size;

  // ports
  ConsumerRW *ins;
  std::vector<ProducerRW *> outs;

  ConsumerData insData;
  std::vector<ProducerData> outsData;
};

class MergeModel : public OpExecutionModel<handshake::MergeOp> {
public:
  using OpExecutionModel<handshake::MergeOp>::OpExecutionModel;
  MergeModel(handshake::MergeOp mergeOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::MergeOp>(mergeOp), op(mergeOp),
        size(op->getNumOperands()),
        outs(getState<ProducerRW>(op.getResult(), subset)), outsData(outs),
        insTEHB(tehbValid, tehbReady) {
    for (unsigned i = 0; i < size; ++i)
      ins.push_back(getState<ConsumerRW>(op->getOperand(i), subset));

    for (unsigned i = 0; i < size; ++i)
      insData.emplace_back(ins[i]);
    mergeTEHB.datawidth = outsData.dataWidth;
  }

  void reset() override {
    if (outsData.hasValue())
      resetDataFull(ins, outs, insData, outsData);
    else
      resetDataless(ins, outs);
  }

  void exec(bool isClkRisingEdge) override {
    if (outsData.hasValue())
      execDataFull(isClkRisingEdge, ins, outs, insData, outsData);
    else
      execDataless(isClkRisingEdge, ins, outs);
  }

  void printStates() override {
    for (unsigned i = 0; i < size; ++i)
      printValue<ConsumerRW, const Data>("ins", ins[i], insData[i].data);
    printValue<ProducerRW, Data>("outs", outs, outsData.data);
  }

private:
  handshake::MergeOp op;

  // parameters
  unsigned size;

  // ports
  std::vector<ConsumerRW *> ins;
  ProducerRW *outs;

  std::vector<ConsumerData> insData;
  ProducerData outsData;

  // merge dataless
  bool tehbValid = false, tehbReady = false;
  ConsumerRW insTEHB;
  // merge datafull
  Data tehbDataIn;

  // internal components
  TEHBSupport mergeTEHB;

  void resetDataless(std::vector<ConsumerRW *> &ins, ProducerRW *outs) {
    tehbValid = false;
    for (unsigned i = 0; i < size; ++i)
      tehbValid = tehbValid || ins[i]->valid;

    for (unsigned i = 0; i < size; ++i)
      ins[i]->ready = tehbReady;

    mergeTEHB.resetDataless(&insTEHB, outs);
  }

  void execDataless(bool isClkRisingEdge, std::vector<ConsumerRW *> &ins,
                    ProducerRW *outs) {
    tehbValid = false;
    for (unsigned i = 0; i < size; ++i)
      tehbValid = tehbValid || ins[i]->valid;

    for (unsigned i = 0; i < size; ++i)
      ins[i]->ready = tehbReady;

    mergeTEHB.execDataless(isClkRisingEdge, &insTEHB, outs);
  }

  void resetDataFull(std::vector<ConsumerRW *> &ins, ProducerRW *outs,
                     std::vector<ConsumerData> &insData,
                     ProducerData &outsData) {
    tehbValid = false;
    tehbDataIn = *insData[0].data;
    for (unsigned i = 0; i < size; ++i) {
      if (ins[i]->valid) {
        tehbValid = true;
        tehbDataIn = *insData[i].data;
        break;
      }
    }

    for (unsigned i = 0; i < size; ++i)
      ins[i]->ready = tehbReady;

    mergeTEHB.reset(&insTEHB, outs, &tehbDataIn, outsData.data);
  }

  void execDataFull(bool isClkRisingEdge, std::vector<ConsumerRW *> &ins,
                    ProducerRW *outs, std::vector<ConsumerData> &insData,
                    ProducerData &outsData) {
    tehbValid = false;
    tehbDataIn = *insData[0].data;
    for (unsigned i = 0; i < size; ++i) {
      if (ins[i]->valid) {
        tehbValid = true;
        tehbDataIn = *insData[i].data;
        break;
      }
    }
    for (unsigned i = 0; i < size; ++i)
      ins[i]->ready = tehbReady;

    mergeTEHB.exec(isClkRisingEdge, &insTEHB, outs, &tehbDataIn, outsData.data);
  }
};

class MuxModel : public OpExecutionModel<handshake::MuxOp> {
public:
  using OpExecutionModel<handshake::MuxOp>::OpExecutionModel;
  MuxModel(handshake::MuxOp muxOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::MuxOp>(muxOp), op(muxOp),
        size(op.getDataOperands().size()),
        selectWidth(op.getSelectOperand().getType().getDataBitWidth()),
        outs(getState<ProducerRW>(op.getResult(), subset)), outsData(outs),
        insTEHB(tehbValid, tehbReady) {
    for (auto oper : op.getDataOperands())
      ins.push_back(getState<ChannelConsumerRW>(oper, subset));
    index = getState<ChannelConsumerRW>(op.getSelectOperand(), subset);
    outs = getState<ProducerRW>(op.getDataResult(), subset);

    for (unsigned i = 0; i < size; ++i)
      insData.emplace_back(ins[i]);
    muxTEHB.datawidth = outsData.dataWidth;
  }

  void reset() override {
    indexNum = dataCast<APInt>(index->data).getZExtValue();
    if (outsData.hasValue())
      resetDataFull(ins, index, indexNum, outs, insData, outsData);
    else
      resetDataless(ins, index, indexNum, outs);
  }

  void exec(bool isClkRisingEdge) override {
    indexNum = dataCast<APInt>(index->data).getZExtValue();
    if (outsData.hasValue())
      execDataFull(isClkRisingEdge, ins, index, indexNum, outs, insData,
                   outsData);
    else
      execDataless(isClkRisingEdge, ins, index, indexNum, outs);
  }

  void printStates() override {
    for (unsigned i = 0; i < size; ++i)
      printValue<ConsumerRW, const Data>("ins", ins[i], insData[i].data);
    printValue<ProducerRW, Data>("outs", outs, outsData.data);
    printValue<ChannelConsumerRW, const Data>("index", index, &index->data);
  }

private:
  handshake::MuxOp op;

  // parameters
  unsigned size, selectWidth;
  unsigned indexNum = 0;

  // ports
  std::vector<ConsumerRW *> ins;
  ChannelConsumerRW *index;
  ProducerRW *outs;

  std::vector<ConsumerData> insData;
  ProducerData outsData;

  // merge dataless
  bool tehbValid = false, tehbReady = false;
  ConsumerRW insTEHB;
  // merge datafull
  Data tehbDataIn;

  // internal components
  TEHBSupport muxTEHB;

  void resetDataless(std::vector<ConsumerRW *> &ins, ConsumerRW *index,
                     unsigned indexNum, ProducerRW *outs) {
    bool selectedDataValid = false, indexEqual = false;
    for (unsigned i = 0; i < size; ++i) {
      indexEqual = (i == indexNum);

      if (indexEqual && index->valid && ins[i]->valid)
        selectedDataValid = true;
      ins[i]->ready =
          (indexEqual && index->valid && ins[i]->valid && tehbReady) ||
          !ins[i]->valid;
    }
    index->ready = !index->valid || (selectedDataValid && tehbReady);
    tehbValid = selectedDataValid;
    muxTEHB.resetDataless(&insTEHB, outs);
  }

  void execDataless(bool isClkRisingEdge, std::vector<ConsumerRW *> &ins,
                    ConsumerRW *index, unsigned indexNum, ProducerRW *outs) {
    bool selectedDataValid = false, indexEqual = false;
    for (unsigned i = 0; i < size; ++i) {
      indexEqual = (i == indexNum);

      if (indexEqual && index->valid && ins[i]->valid)
        selectedDataValid = true;
      ins[i]->ready =
          (indexEqual && index->valid && ins[i]->valid && tehbReady) ||
          !ins[i]->valid;
    }
    index->ready = !index->valid || (selectedDataValid && tehbReady);
    tehbValid = selectedDataValid;
    muxTEHB.execDataless(isClkRisingEdge, &insTEHB, outs);
  }

  void resetDataFull(std::vector<ConsumerRW *> &ins, ConsumerRW *index,
                     unsigned indexNum, ProducerRW *outs,
                     std::vector<ConsumerData> &insData,
                     ProducerData &outsData) {
    bool selectedDataValid = false, indexEqual = false;
    Data selectedData = *insData[0].data;
    for (unsigned i = 0; i < size; ++i) {
      indexEqual = (i == indexNum);

      if (indexEqual && index->valid && ins[i]->valid) {
        selectedDataValid = true;
        selectedData = *insData[i].data;
      }
      ins[i]->ready =
          (indexEqual && index->valid && ins[i]->valid && tehbReady) ||
          !ins[i]->valid;
    }
    index->ready = !index->valid || (selectedDataValid && tehbReady);
    tehbValid = selectedDataValid;
    tehbDataIn = selectedData;

    muxTEHB.reset(&insTEHB, outs, &tehbDataIn, outsData.data);
  }

  void execDataFull(bool isClkRisingEdge, std::vector<ConsumerRW *> &ins,
                    ConsumerRW *index, unsigned indexNum, ProducerRW *outs,
                    std::vector<ConsumerData> &insData,
                    ProducerData &outsData) {
    bool selectedDataValid = false, indexEqual = false;
    Data selectedData = *insData[0].data;
    for (unsigned i = 0; i < size; ++i) {
      indexEqual = (i == indexNum);

      if (indexEqual && index->valid && ins[i]->valid) {
        selectedDataValid = true;
        selectedData = *insData[i].data;
      }
      ins[i]->ready =
          (indexEqual && index->valid && ins[i]->valid && tehbReady) ||
          !ins[i]->valid;
    }
    index->ready = !index->valid || (selectedDataValid && tehbReady);
    tehbValid = selectedDataValid;
    tehbDataIn = selectedData;

    muxTEHB.exec(isClkRisingEdge, &insTEHB, outs, &tehbDataIn, outsData.data);
  }
};

class OEHBModel : public OpExecutionModel<handshake::BufferOp> {
public:
  using OpExecutionModel<handshake::BufferOp>::OpExecutionModel;

  OEHBModel(handshake::BufferOp oehbOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::BufferOp>(oehbOp), op(oehbOp),
        ins(getState<ConsumerRW>(op.getOperand(), subset)),
        outs(getState<ProducerRW>(op.getResult(), subset)), insData(ins),
        outsData(outs), oehbDl() {
    oehbDl.datawidth = insData.dataWidth;
  }

  void reset() override {
    oehbDl.reset(ins, outs, insData.data, outsData.data);
  }

  void exec(bool isClkRisingEdge) override {
    oehbDl.exec(isClkRisingEdge, ins, outs, insData.data, outsData.data);
  }

  void printStates() override {
    printValue<ConsumerRW, const Data>("ins", ins, insData.data);
    printValue<ProducerRW, Data>("outs", outs, outsData.data);
  }

private:
  handshake::BufferOp op;

  // ports
  ConsumerRW *ins;
  ProducerRW *outs;

  ConsumerData insData;
  ProducerData outsData;

  // internal components
  OEHBSupport oehbDl;
};

class SinkModel : public OpExecutionModel<handshake::SinkOp> {
public:
  using OpExecutionModel<handshake::SinkOp>::OpExecutionModel;
  SinkModel(handshake::SinkOp sinkOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::SinkOp>(sinkOp), op(sinkOp),
        ins(getState<ConsumerRW>(op.getOperand(), subset)), insData(ins) {}

  void reset() override { ins->ready = true; }

  void exec(bool isClkRisingEdge) override { reset(); }

  void printStates() override {
    printValue<ConsumerRW, const Data>("ins", ins, insData.data);
  }

private:
  handshake::SinkOp op;

  // ports
  ConsumerRW *ins;
  ConsumerData insData;
};

class SourceModel : public OpExecutionModel<handshake::SourceOp> {
public:
  using OpExecutionModel<handshake::SourceOp>::OpExecutionModel;
  SourceModel(handshake::SourceOp sourceOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::SourceOp>(sourceOp), op(sourceOp) {
    outs = getState<ProducerRW>(op.getResult(), subset);
  }

  void reset() override { outs->valid = true; }

  void exec(bool isClkRisingEdge) override { reset(); }

  void printStates() override {
    llvm::outs() << "Outs: " << outs->valid << " " << outs->ready << "\n";
  }

private:
  handshake::SourceOp op;

  // ports
  ProducerRW *outs;
};

class TEHBModel : public OpExecutionModel<handshake::BufferOp> {
public:
  using OpExecutionModel<handshake::BufferOp>::OpExecutionModel;

  TEHBModel(handshake::BufferOp tehbOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::BufferOp>(tehbOp), op(tehbOp),
        ins(getState<ConsumerRW>(op.getOperand(), subset)),
        outs(getState<ProducerRW>(op.getResult(), subset)), insData(ins),
        outsData(outs), returnTEHB() {

    returnTEHB.datawidth = insData.dataWidth;
  }

  void reset() override {
    returnTEHB.reset(ins, outs, insData.data, outsData.data);
  }

  void exec(bool isClkRisingEdge) override {
    returnTEHB.exec(isClkRisingEdge, ins, outs, insData.data, outsData.data);
  }

  void printStates() override {
    printValue<ConsumerRW, const Data>("ins", ins, insData.data);
    printValue<ProducerRW, Data>("outs", outs, outsData.data);
  }

private:
  handshake::BufferOp op;

  // ports
  ConsumerRW *ins;
  ProducerRW *outs;

  ConsumerData insData;
  ProducerData outsData;

  // internal components
  TEHBSupport returnTEHB;
};

class EndMemlessModel : public OpExecutionModel<handshake::EndOp> {
public:
  using OpExecutionModel<handshake::EndOp>::OpExecutionModel;
  EndMemlessModel(handshake::EndOp endOp, mlir::DenseMap<Value, RW *> &subset,
                  bool &resValid, const bool &resReady, Data &resData)
      : OpExecutionModel<handshake::EndOp>(endOp), op(endOp),
        outs(resValid, resReady), outsData(resData) {
    ins = getState<ChannelConsumerRW>(op.getInputs().front(), subset);

    auto temp =
        cast<handshake::ChannelType>(endOp.getInputs().front().getType());
    bitwidth = temp.getDataBitWidth();
  }

  void reset() override {
    outs.valid = ins->valid;
    ins->ready = ins->valid && outs.ready;
    outsData = ins->data;
  }

  void exec(bool isClkRisingEdge) override { reset(); }

  void printStates() override {
    printValue<ConsumerRW, const Data>("ins", ins, &ins->data);
  }

private:
  handshake::EndOp op;
  // parameters
  unsigned bitwidth;

  // ports
  ChannelConsumerRW *ins;
  ProducerRW outs;
  Data &outsData;
};

//===----------------------------------------------------------------------===//
// Arithmetic
//===----------------------------------------------------------------------===//
/// Arithmetic and generic components

class TruncIModel : public OpExecutionModel<handshake::TruncIOp> {
public:
  using OpExecutionModel<handshake::TruncIOp>::OpExecutionModel;
  TruncIModel(handshake::TruncIOp trunciOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::TruncIOp>(trunciOp), op(trunciOp),
        inputWidth(op.getIn().getType().getDataBitWidth()),
        outputWidth(op.getResult().getType().getDataBitWidth()) {
    ins = getState<ChannelConsumerRW>(op.getOperand(), subset);
    outs = getState<ChannelProducerRW>(op.getResult(), subset);
  }

  void reset() override {
    const APInt k = dataCast<const APInt>(ins->data);
    outs->data = k.trunc(outputWidth);
    outs->valid = ins->valid;
    ins->ready = !ins->valid || (ins->valid && outs->ready);
  }

  void exec(bool isClkRisingEdge) override { reset(); }

  void printStates() override {
    printValue<ConsumerRW, const Data>("ins", ins, &ins->data);
    printValue<ProducerRW, Data>("outs", outs, &outs->data);
  }

private:
  handshake::TruncIOp op;

  // parameters
  unsigned inputWidth;
  unsigned outputWidth;

  // ports
  ChannelConsumerRW *ins;
  ChannelProducerRW *outs;
};

using UnaryCompFunc = std::function<Data(const Data &, unsigned)>;

/// Class to represent NEGF, NOT, EXTSI, EXTUI
template <typename Op>
class GenericUnaryOpModel : public OpExecutionModel<Op> {
public:
  using OpExecutionModel<Op>::OpExecutionModel;
  GenericUnaryOpModel(Op op, mlir::DenseMap<Value, RW *> &subset,
                      const UnaryCompFunc &callback)
      : OpExecutionModel<Op>(op), op(op),
        outputWidth(op.getResult().getType().getDataBitWidth()),
        callback(callback) {
    ins = OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
        op.getOperand(), subset);
    outs = OpExecutionModel<Op>::template getState<ChannelProducerRW>(
        op.getResult(), subset);
  }

  void reset() override {
    outs->data = callback(ins->data, outputWidth);
    outs->valid = ins->valid;
    ins->ready = outs->ready;
  }

  void exec(bool isClkRisingEdge) override { reset(); }

  void printStates() override {
    OpExecutionModel<Op>::template printValue<ConsumerRW, const Data>(
        "ins", ins, &ins->data);
    OpExecutionModel<Op>::template printValue<ProducerRW, Data>("outs", outs,
                                                                &outs->data);
  }

private:
  Op op;

  // parameters
  unsigned outputWidth;
  UnaryCompFunc callback;

  // ports
  ChannelConsumerRW *ins;
  ChannelProducerRW *outs;
};

/// Mutual component for binary operations: AddI, AndI, CmpI, MulI, OrI,
/// ShlI, ShrsI, ShruI, SubI, XorI...
using BinaryCompFunc = std::function<Data(const Data &, const Data &)>;
template <typename Op>
class GenericBinaryOpModel : public OpExecutionModel<Op> {
public:
  using OpExecutionModel<Op>::OpExecutionModel;
  GenericBinaryOpModel(Op op, mlir::DenseMap<Value, RW *> &subset,
                       const BinaryCompFunc &callback, unsigned latency = 0)
      : OpExecutionModel<Op>(op), op(op), callback(callback), latency(latency),
        bitwidth(cast<handshake::ChannelType>(op.getResult().getType())
                     .getDataBitWidth()),

        lhs(OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
            op.getLhs(), subset)),
        rhs(OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
            op.getRhs(), subset)),
        result(OpExecutionModel<Op>::template getState<ChannelProducerRW>(
            op.getResult(), subset)),
        outsJoin(joinValid, result->ready), binJoin(2) {
    insJoin = {lhs, rhs};
  }

  void reset() override {
    binJoin.exec(insJoin, result);
    result->data = callback(lhs->data, rhs->data);
  }

  void exec(bool isClkRisingEdge) override {
    if (!latency)
      reset();
    else {
      binJoin.exec(insJoin, &outsJoin);

      if (isClkRisingEdge) {
        if (lhs->valid && rhs->valid && !hasData) {
          // collect
          hasData = true;
          counter = 1;
          tempData = callback(lhs->data, rhs->data);
          result->valid = false;
        } else if (hasData && counter < latency - 1) {
          // wait
          ++counter;
          if (counter == latency - 1)
            result->data = tempData;
        } else if (hasData && counter >= latency - 1 && !result->valid) {
          result->data = tempData;
          result->valid = true;
        } else if (!lhs->valid && !rhs->valid && hasData && result->valid &&
                   result->ready) {
          result->valid = false;
          hasData = false;
          counter = 0;
        }
      }
    }
  }

  void printStates() override {
    OpExecutionModel<Op>::template printValue<ConsumerRW, const Data>(
        "lhs", lhs, &lhs->data);
    OpExecutionModel<Op>::template printValue<ConsumerRW, const Data>(
        "rhs", rhs, &rhs->data);
    OpExecutionModel<Op>::template printValue<ProducerRW, Data>(
        "result", result, &result->data);
  }

private:
  Op op;

  // parameters
  BinaryCompFunc callback;
  unsigned latency = 0;
  unsigned bitwidth;

  // ports
  ChannelConsumerRW *lhs, *rhs;
  ChannelProducerRW *result;

  // latency parameters
  std::vector<ConsumerRW *> insJoin;
  ProducerRW outsJoin;
  bool joinValid = false;
  unsigned counter = 0;
  bool hasData = false;
  Data tempData = APInt(bitwidth, 0);

  // internal components
  JoinSupport binJoin;
};

//===----------------------------------------------------------------------===//
// Simulator
//===----------------------------------------------------------------------===//

class Simulator {
public:
  Simulator(handshake::FuncOp funcOp) : funcOp(funcOp) {
    // Iterate through all the values of the circuit
    for (BlockArgument arg : funcOp.getArguments())
      associateState(arg, funcOp, funcOp->getLoc());

    for (Operation &op : funcOp.getOps())
      for (auto res : op.getResults())
        associateState(res, &op, op.getLoc());

    // register models for all ops of the funcOp
    for (Operation &op : funcOp.getOps())
      associateModel(&op);

    reset();
  }

  void reset() {
    while (true) {
      for (auto &[op, model] : opModels)
        model->reset();

      bool isFin = true;
      // Check if states have changed
      for (auto [val, state] : updaters)
        isFin = isFin && state->check();
      if (isFin)
        break;

      // Update oldStates
      for (auto [val, state] : updaters)
        state->update();
    }
    for (auto [val, state] : updaters)
      state->update();
  }

  void simulate(llvm::ArrayRef<std::string> inputArg) {
    // Counter of channels
    unsigned channelCount = 0;
    // All values that match channels
    llvm::SmallVector<Value> channelArgs;
    for (BlockArgument arg : funcOp.getArguments())
      if (isa<handshake::ChannelType>(arg.getType())) {
        ++channelCount;
        channelArgs.push_back(arg);
      }

    // Check if the number of data values is divisible by the number of
    // ChannelStates
    assert(inputArg.size() % channelCount == 0 &&
           "The amount of input data values must be divisible by the number of "
           "the input channels!");

    // Iterate through all the collected channel values...
    size_t k = 0;
    for (auto val : channelArgs) {
      auto channel = cast<TypedValue<handshake::ChannelType>>(val);
      auto *channelArg = static_cast<ChannelProducerRW *>(producerViews[val]);
      // ...and update the corresponding data fields
      llvm::TypeSwitch<mlir::Type>(channel.getType().getDataType())
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
      updaters[val]->update();
    }

    // Set all inputs valid to true
    for (BlockArgument arg : funcOp.getArguments())
      updaters[arg]->setValid();

    auto returnOp = *funcOp.getOps<handshake::EndOp>().begin();
    auto *res = static_cast<ChannelConsumerRW *>(
        consumerViews[&returnOp->getOpOperand(0)]);

    unsigned eqStateCount = 0;
    iterNum = 0;
    // Outer loop: for clock cycles
    // Stops when return.input->valid becomes true or limit of cycles is reached
    while (true) {
      ++iterNum;
      //  True only once on each clkRisingEdge
      bool isClock = true;
      // Inner loop: for signal propagation within one clkRisingEdge
      // Stops when there's no change in valueStates
      while (true) {
        // Execute each model
        for (auto &[op, model] : opModels)
          model->exec(isClock);

        bool isFin = true;
        // Check if states have changed
        for (auto [val, state] : updaters)
          isFin = isFin && state->check();
        if (isFin)
          break;
        // Update oldStates
        for (auto [val, state] : updaters)
          state->update();
        isClock = false;
      }

      if (res->valid) {
        for (auto [val, state] : updaters)
          state->resetValid();
        break;
      }

      bool isFin = true;
      // Check if states have changed
      for (auto [val, state] : updaters)
        isFin = isFin && state->check();

      if (isFin) {
        ++eqStateCount;
        if (eqStateCount >= cyclesLimit)
          break;
      }

      // At the end of each cycle reset each input's valid signal to false if
      // the corresponding ready was true
      for (BlockArgument arg : funcOp.getArguments())
        updaters[arg]->resetValid();
    }
  }

  // Just a temporary function to print the results of the simulation to
  // standart output
  void printResults() {
    llvm::outs() << "Results\n";

    llvm::outs() << resValid << " " << resReady << " "
                 << dataCast<APInt>(resData) << "\n";
    llvm::outs() << "Number of iterations: " << iterNum << "\n";
  }

  // The temporary function
  void printModelStates() {
    std::map<std::string, ExecutionModel *> names;
    for (auto &[op, model] : opModels) {
      auto j = op->getAttr("handshake.name");
      std::string str;
      llvm::raw_string_ostream output(str);
      j.print(output, true);
      names.insert({str, model});
    }

    for (auto &[name, model] : names) {
      llvm::outs() << "=================== ===================\n";
      llvm::outs() << name << "\n";
      model->printStates();
    }
  }

  ~Simulator() {
    for (auto [_, model] : opModels)
      delete model;
    for (auto [_, state] : oldValuesStates)
      delete state;
    for (auto [_, state] : newValuesStates)
      delete state;
    for (auto [_, state] : updaters)
      delete state;
    for (auto [_, rw] : consumerViews)
      delete rw;
    for (auto [_, rw] : producerViews)
      delete rw;
  }

private:
  // Maybe need it some day, let it be the part of simulator class
  handshake::FuncOp funcOp;
  // Results of the simulation
  bool resValid = false, resReady = true;
  Data resData;
  // Number of iterations during the simulation
  unsigned iterNum = 0;
  // Map for execution models
  mlir::DenseMap<Operation *, ExecutionModel *> opModels;
  // Map the stores RW API classes
  // mlir::DenseMap<std::pair<Value, Operation *>, RW *> rws;
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
  // Set the number of the iterations for the simulator to execute before force
  // break
  unsigned cyclesLimit = 100;
  /// Maps all value uses to their *consumer*'s RW object.
  mlir::DenseMap<OpOperand *, ConsumerRW *> consumerViews;
  /// Maps all operation results (OpResult) and block arguments
  /// (BlockArgument) to their *producer*'s RW object.
  mlir::DenseMap<Value, ProducerRW *> producerViews;

  // Register the Model inside opNodels
  template <typename Model, typename Op, typename... Args>
  void registerModel(Op op, Args &&...modelArgs) {
    mlir::DenseMap<Value, RW *> subset;
    auto opOperands = op->getOpOperands();
    auto results = op->getResults();
    for (auto &opOp : opOperands)
      subset.insert({opOp.get(), consumerViews[&opOp]});

    for (auto res : results)
      subset.insert({res, producerViews[res]});

    ExecutionModel *model =
        new Model(op, subset, std::forward<Args>(modelArgs)...);
    opModels.insert({op, model});
  }

  // Determine the concrete Model type
  void associateModel(Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<handshake::SinkOp>([&](handshake::SinkOp sinkOp) {
          registerModel<SinkModel, handshake::SinkOp>(sinkOp);
        })
        .Case<handshake::ForkOp>([&](handshake::ForkOp forkOp) {
          registerModel<ForkModel, handshake::ForkOp>(forkOp);
        })
        .Case<handshake::ConstantOp>([&](handshake::ConstantOp constOp) {
          registerModel<ConstantModel, handshake::ConstantOp>(constOp);
        })
        .Case<handshake::ExtSIOp>([&](handshake::ExtSIOp extsiOp) {
          UnaryCompFunc callback = [](const Data &lhs, unsigned outWidth) {
            auto temp = dataCast<APInt>(lhs).getSExtValue();
            APInt ext(outWidth, temp, true);
            return ext;
          };

          registerModel<GenericUnaryOpModel<handshake::ExtSIOp>>(extsiOp,
                                                                 callback);
        })
        .Case<handshake::ShLIOp>([&](handshake::ShLIOp shliOp) {
          BinaryCompFunc callback = [](Data lhs, Data rhs) {
            APInt shli = dataCast<APInt>(lhs) << dataCast<APInt>(rhs);
            return shli;
          };
          registerModel<GenericBinaryOpModel<handshake::ShLIOp>>(shliOp,
                                                                 callback, 0);
        })
        .Case<handshake::TruncIOp>([&](handshake::TruncIOp trunciOp) {
          registerModel<TruncIModel, handshake::TruncIOp>(trunciOp);
        })
        .Case<handshake::ConditionalBranchOp>(
            [&](handshake::ConditionalBranchOp cBranchOp) {
              registerModel<CondBranchModel, handshake::ConditionalBranchOp>(
                  cBranchOp);
            })

        .Case<handshake::CmpIOp>([&](handshake::CmpIOp cmpiOp) {
          BinaryCompFunc callback;

          switch (cmpiOp.getPredicate()) {
          case handshake::CmpIPredicate::eq:
            callback = [](Data lhs, Data rhs) {
              APInt comp(1, dataCast<APInt>(lhs).getRawData() ==
                                dataCast<APInt>(rhs).getRawData());
              return comp;
            };
            break;
          case handshake::CmpIPredicate::ne:
            callback = [](Data lhs, Data rhs) {
              APInt comp(1, dataCast<APInt>(lhs).getRawData() !=
                                dataCast<APInt>(rhs).getRawData());
              return comp;
            };
            break;
          case handshake::CmpIPredicate::uge:
            callback = [](Data lhs, Data rhs) {
              APInt comp(1, dataCast<APInt>(lhs).getZExtValue() >=
                                dataCast<APInt>(rhs).getZExtValue());
              return comp;
            };
            break;
          case handshake::CmpIPredicate::sge:
            callback = [](Data lhs, Data rhs) {
              APInt comp(1, dataCast<APInt>(lhs).getSExtValue() >=
                                dataCast<APInt>(rhs).getSExtValue());
              return comp;
            };
            break;
          case handshake::CmpIPredicate::ugt:
            callback = [](Data lhs, Data rhs) {
              APInt comp(1, dataCast<APInt>(lhs).getZExtValue() >
                                dataCast<APInt>(rhs).getZExtValue());
              return comp;
            };
            break;
          case handshake::CmpIPredicate::sgt:
            callback = [](Data lhs, Data rhs) {
              APInt comp(1, dataCast<APInt>(lhs).getSExtValue() >
                                dataCast<APInt>(rhs).getSExtValue());
              return comp;
            };
            break;
          case handshake::CmpIPredicate::ule:
            callback = [](Data lhs, Data rhs) {
              APInt comp(1, dataCast<APInt>(lhs).getZExtValue() <=
                                dataCast<APInt>(rhs).getZExtValue());
              return comp;
            };
            break;
          case handshake::CmpIPredicate::sle:
            callback = [](Data lhs, Data rhs) {
              APInt comp(1, dataCast<APInt>(lhs).getSExtValue() <=
                                dataCast<APInt>(rhs).getSExtValue());
              return comp;
            };
            break;
          case handshake::CmpIPredicate::ult:
            callback = [](Data lhs, Data rhs) {
              APInt comp(1, dataCast<APInt>(lhs).getZExtValue() <
                                dataCast<APInt>(rhs).getZExtValue());
              return comp;
            };
            break;
          case handshake::CmpIPredicate::slt:
            callback = [](Data lhs, Data rhs) {
              APInt comp(1, dataCast<APInt>(lhs).getSExtValue() <
                                dataCast<APInt>(rhs).getSExtValue());
              return comp;
            };
            break;
          }

          registerModel<GenericBinaryOpModel<handshake::CmpIOp>>(cmpiOp,
                                                                 callback, 0);
        })
        .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
          registerModel<MuxModel, handshake::MuxOp>(muxOp);
        })
        .Case<handshake::ControlMergeOp>(
            [&](handshake::ControlMergeOp cMergeOp) {
              registerModel<ControlMergeModel, handshake::ControlMergeOp>(
                  cMergeOp);
            })
        .Case<handshake::SourceOp>([&](handshake::SourceOp sourceOp) {
          registerModel<SourceModel, handshake::SourceOp>(sourceOp);
        })
        .Case<handshake::MulIOp>([&](handshake::MulIOp muliOp) {
          BinaryCompFunc callback = [](Data lhs, Data rhs) {
            APInt mul = dataCast<APInt>(lhs) * dataCast<APInt>(rhs);
            return mul;
          };
          registerModel<GenericBinaryOpModel<handshake::MulIOp>>(muliOp,
                                                                 callback, 4);
        })
        .Case<handshake::BufferOp>([&](handshake::BufferOp bufferOp) {
          auto params =
              bufferOp->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
          auto optTiming =
              params.getNamed(handshake::BufferOp::TIMING_ATTR_NAME);
          if (auto timing =
                  dyn_cast<handshake::TimingAttr>(optTiming->getValue())) {
            auto info = timing.getInfo();
            if (info == handshake::TimingInfo::oehb())
              registerModel<OEHBModel, handshake::BufferOp>(bufferOp);
            if (info == handshake::TimingInfo::tehb())
              registerModel<TEHBModel, handshake::BufferOp>(bufferOp);
          }
        })
        .Case<handshake::EndOp>([&](handshake::EndOp endOp) {
          registerModel<EndMemlessModel, handshake::EndOp>(endOp, resValid,
                                                           resReady, resData);
        })
        .Case<handshake::AddIOp>([&](handshake::AddIOp addiOp) {
          BinaryCompFunc callback = [](Data lhs, Data rhs) {
            APInt add = dataCast<APInt>(lhs) + dataCast<APInt>(rhs);
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

    State *oldState = new State(typedVal);
    State *newState = new State(typedVal);
    Updater *upd = new Updater(*oldState, *newState);
    Producer *producerRW = new Producer(*oldState, *newState);
    Consumer *consumerRW = new Consumer(*oldState, *newState);

    oldValuesStates.insert({val, oldState});
    newValuesStates.insert({val, newState});
    updaters.insert({val, upd});
    producerViews.insert({val, producerRW});

    for (auto &opOp : val.getUses())
      consumerViews.insert({&opOp, consumerRW});
  }

  // Determine if the state belongs to channel or control
  void associateState(Value val, Operation *producerOp, Location loc) {
    llvm::TypeSwitch<mlir::Type>(val.getType())
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

  sim.simulate(inputArgs);
  sim.printResults();
}