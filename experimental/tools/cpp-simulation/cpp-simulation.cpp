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
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"

#include "dynamatic/Transforms/HandshakeMaterialize.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

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

    llvm::TypeSwitch<Type>(channel.getType().getDataType())
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
public:
  bool &valid;
  const bool &ready;

protected:
  enum ProducerDescendants { D_ChannelProducerRW, D_ControlProducerRW };

public:
  ProducerDescendants getType() const { return prod; }
  ProducerRW(ProducerDescendants p, bool &valid, bool &ready)
      : valid(valid), ready(ready), prod(p) {}

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

  ConsumerRW(ConsumerDescendants c, bool &valid, bool &ready)
      : valid(valid), ready(ready), cons(c) {}
  virtual ~ConsumerRW() {};

private:
  const ConsumerDescendants cons;
};

////--- Control (no data)

/// In this case the user can change the ready signal, but has
/// ReadOnly access to the valid one.
struct ControlConsumerRW : public ConsumerRW {
  ControlConsumerRW(ControlState &reader, ControlState &writer)
      : ConsumerRW(D_ControlConsumerRW, reader.valid, writer.ready) {}

  static bool classof(const ConsumerRW *c) {
    return c->getType() == D_ControlConsumerRW;
  }
};

/// In this case the user can change the valid signal, but has
/// ReadOnly access to the ready one.
struct ControlProducerRW : public ProducerRW {
  ControlProducerRW(ControlState &reader, ControlState &writer)
      : ProducerRW(D_ControlProducerRW, writer.valid, reader.ready) {}

  static bool classof(const ProducerRW *c) {
    return c->getType() == D_ControlProducerRW;
  }
};

////--- Channel (data)

/// In this case the user can change the valid signal, but has ReadOnly access
/// to the valid and data ones.
struct ChannelConsumerRW : public ConsumerRW {
  const Any &data;

  ChannelConsumerRW(ChannelState &reader, ChannelState &writer)
      : ConsumerRW(D_ChannelConsumerRW, reader.valid, writer.ready),
        data(reader.data) {}

  static bool classof(const ConsumerRW *c) {
    return c->getType() == D_ChannelConsumerRW;
  }
};

/// In this case the user can change the valid and data signals, but has
/// ReadOnly access to the ready one.
struct ChannelProducerRW : public ProducerRW {
  Any &data;

  ChannelProducerRW(ChannelState &reader, ChannelState &writer)
      : ProducerRW(D_ChannelProducerRW, writer.valid, reader.ready),
        data(writer.data) {}

  static bool classof(const ProducerRW *c) {
    return c->getType() == D_ChannelProducerRW;
  }
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
  static State *getState(Value val, mlir::DenseMap<Value, RW *> &rws) {
    return static_cast<State *>(rws[val]);
  }
};

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//
/// Components required for the internal state.

class ForkSupport {
public:
  ForkSupport() = default;

  void resetDataless(bool insValid, const std::vector<bool> &outsReady,
                     bool &insReady,
                     std::vector<std::reference_wrapper<bool>> &outsValid) {
    for (bool outR : outsReady) {
      transmitValue.push_back(true);
      keepValue.push_back(!outR);
      outsValid.emplace_back(insValid);
      blockStopArray.push_back(!outR);
    }
    anyBlockStop = false;
    for (bool c : blockStopArray)
      anyBlockStop = anyBlockStop || c;
    insReady = !anyBlockStop;
    backpressure = insValid && anyBlockStop;
  }

  void execDataless(bool isClkRisingEdge, bool insValid,
                    const std::vector<bool> &outsReady, bool &insReady,
                    std::vector<std::reference_wrapper<bool>> &outsValid) {

    for (unsigned i = 0; i < outsReady.size(); ++i) {
      keepValue[i] = !outsReady[i] && transmitValue[i];
      if (isClkRisingEdge)
        transmitValue[i] = keepValue[i] || !backpressure;
      outsValid[i].get() = transmitValue[i] && insValid;
      blockStopArray[i] = keepValue[i];
    }
    anyBlockStop = false;
    for (bool c : blockStopArray)
      anyBlockStop = anyBlockStop || c;
    insReady = !anyBlockStop;
    backpressure = insValid && anyBlockStop;
  }
  void resetDataFull(bool insValid, const std::vector<bool> &outsReady,
                     bool &insReady,
                     std::vector<std::reference_wrapper<bool>> &outsValid,
                     std::vector<Any *> &outsData, const Any &insData) {
    resetDataless(insValid, outsReady, insReady, outsValid);
    for (auto &out : outsData)
      *out = insData;
  };

  void execDataFull(bool isClkRisingEdge, bool insValid,
                    const std::vector<bool> &outsReady, bool &insReady,
                    std::vector<std::reference_wrapper<bool>> &outsValid,
                    std::vector<Any *> &outsData, const Any &insData) {
    execDataless(isClkRisingEdge, insValid, outsReady, insReady, outsValid);
    for (auto &out : outsData)
      *out = insData;
  }

private:
  // eager fork
  std::vector<bool> transmitValue, keepValue;
  // fork
  std::vector<bool> blockStopArray;
  bool anyBlockStop = false, backpressure = false;
};

class Join {
public:
  Join(unsigned size) : size(size) {}

  void exec(const std::vector<bool> &insValid, bool outsReady,
            std::vector<std::reference_wrapper<bool>> &insReady,
            bool &outsValid) {
    outsValid = true;
    for (unsigned i = 0; i < size; ++i)
      outsValid = outsValid && insValid[i];

    for (unsigned i = 0; i < size; ++i) {
      insReady[i].get() = outsReady;
      for (unsigned j = 0; j < size; ++j)
        if (i != j)
          insReady[i].get() = insReady[i].get() && insValid[j];
    }
  }

private:
  unsigned size;
};

class OEHBSupport {
public:
  OEHBSupport() = default;

  void resetDataless(bool insValid, bool outsReady, bool &insReady,
                     bool &outsValid) {
    outputValid = false;
    insReady = true;
    outsValid = outputValid;
  }
  void execDataless(bool isClkRisingEdge, bool insValid, bool outsReady,
                    bool &insReady, bool &outsValid) {
    if (isClkRisingEdge)
      outputValid = insValid || (outputValid && !outsReady);
    insReady = !outputValid || outsReady;
    outsValid = outputValid;
  }

  void resetDataFull(bool insValid, bool outsReady, bool &insReady,
                     bool &outsValid, Any &outsData, const Any &insData) {
    resetDataless(insValid, outsReady, inputReady, outsValid);
    outsData = APInt(llvm::any_cast<APInt>(insData).getBitWidth(), 0);
    insReady = inputReady;
    regEn = inputReady && insValid;
  };

  void execDataFull(bool isClkRisingEdge, bool insValid, bool outsReady,
                    bool &insReady, bool &outsValid, Any &outsData,
                    const Any &insData) {
    execDataless(isClkRisingEdge, insValid, outsReady, inputReady, outsValid);
    if (isClkRisingEdge && regEn)
      outsData = insData;
    insReady = inputReady;
    regEn = inputReady && insValid;
  }

private:
  bool outputValid = false, regEn = false, inputReady = false;
};

class TEHBSupport {
public:
  TEHBSupport() = default;
  void resetDataless(bool insValid, bool outsReady, bool &insReady,
                     bool &outsValid) {
    outputValid = insValid;
    fullReg = false;
    insReady = true;
    outsValid = outputValid;
  }
  void execDataless(bool isClkRisingEdge, bool insValid, bool outsReady,
                    bool &insReady, bool &outsValid) {
    if (isClkRisingEdge)
      fullReg = outputValid && !outsReady;
    outputValid = insValid || fullReg;
    insReady = !fullReg;
    outsValid = outputValid;
  }

  void resetDataFull(bool insValid, bool outsReady, bool &insReady,
                     bool &outsValid, Any &outsData, const Any &insData) {

    resetDataless(insValid, outsReady, regNotFull, outsValid);
    regEnable = regNotFull && insValid && !outsReady;
    dataReg = 0;

    if (regNotFull)
      outsData = insData;
    else
      outsData = dataReg;

    insReady = regNotFull;
  };

  void execDataFull(bool isClkRisingEdge, bool insValid, bool outsReady,
                    bool &insReady, bool &outsValid, Any &outsData,
                    const Any &insData) {
    execDataless(isClkRisingEdge, insValid, outsReady, regNotFull, outsValid);
    regEnable = regNotFull && insValid && !outsReady;

    if (isClkRisingEdge && regEnable)
      dataReg = insData;

    if (regNotFull)
      outsData = insData;
    else
      outsData = dataReg;

    insReady = regNotFull;
  }

private:
  bool fullReg = false, outputValid = false;
  bool regNotFull = false, regEnable = false;
  Any dataReg;
};

//===----------------------------------------------------------------------===//
// Handshake
//===----------------------------------------------------------------------===//

class BranchModel : public OpExecutionModel<handshake::BranchOp> {
public:
  using OpExecutionModel<handshake::BranchOp>::OpExecutionModel;
  BranchModel(handshake::BranchOp branchOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::BranchOp>(branchOp), op(branchOp) {
    ins = getState<ConsumerRW>(op.getOperand(), subset);
    outs = getState<ProducerRW>(op.getResult(), subset);
    if (auto *p = dyn_cast<ChannelProducerRW>(outs)) {
      outsData = &p->data;
    } else {
      outsData = nullptr;
    }
    if (auto *p = dyn_cast<ChannelConsumerRW>(ins)) {
      insData = &p->data;
    } else {
      insData = nullptr;
    }
  }
  void reset() override {
    outs->valid = ins->valid;
    ins->ready = outs->ready;
    if (insData)
      *outsData = *insData;
  };
  void exec(bool isClkRisingEdge) override { reset(); }

private:
  handshake::BranchOp op;
  ConsumerRW *ins;
  ProducerRW *outs;
  Any const *insData = nullptr;
  Any *outsData = nullptr;
};

class CondBranchModel
    : public OpExecutionModel<handshake::ConditionalBranchOp> {
public:
  using OpExecutionModel<handshake::ConditionalBranchOp>::OpExecutionModel;
  CondBranchModel(handshake::ConditionalBranchOp condBranchOp,
                  mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::ConditionalBranchOp>(condBranchOp),
        op(condBranchOp), condBrJoin(2) {
    data = getState<ConsumerRW>(op.getDataOperand(), subset);
    condition = getState<ChannelConsumerRW>(op.getConditionOperand(), subset);
    trueOut = getState<ProducerRW>(op.getTrueResult(), subset);
    falseOut = getState<ProducerRW>(op.getFalseResult(), subset);

    if (auto *p = dyn_cast<ChannelProducerRW>(trueOut)) {
      trueOutData = &p->data;
    } else {
      trueOutData = nullptr;
    }
    if (auto *p = dyn_cast<ChannelProducerRW>(falseOut)) {
      falseOutData = &p->data;
    } else {
      falseOutData = nullptr;
    }
    if (auto *p = dyn_cast<ChannelConsumerRW>(data)) {
      dataData = &p->data;
    } else {
      dataData = nullptr;
    }
  }
  void reset() override {
    std::vector<bool> insValid{data->valid, condition->valid};
    bool cond = llvm::any_cast<APInt>(condition->data).getBoolValue();
    std::vector<std::reference_wrapper<bool>> insReady{data->ready,
                                                       condition->ready};
    condBrJoin.exec(insValid,
                    (falseOut->ready && !cond) || (trueOut->ready && cond),
                    insReady, brInpValid);
    trueOut->valid = cond && brInpValid;
    falseOut->valid = !cond && brInpValid;
    if (dataData) {
      *trueOutData = *dataData;
      *falseOutData = *dataData;
    }
  };
  void exec(bool isClkRisingEdge) override { reset(); }

private:
  handshake::ConditionalBranchOp op;
  ChannelConsumerRW *condition;
  ConsumerRW *data;
  ProducerRW *trueOut, *falseOut;

  Any const *dataData = nullptr;
  Any *trueOutData = nullptr, *falseOutData = nullptr;

  bool brInpValid = false;
  Join condBrJoin;
};

class ControlMergeModel : public OpExecutionModel<handshake::ControlMergeOp> {
public:
  using OpExecutionModel<handshake::ControlMergeOp>::OpExecutionModel;
  ControlMergeModel(handshake::ControlMergeOp cMergeOp,
                    mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::ControlMergeOp>(cMergeOp), op(cMergeOp),
        size(op->getNumOperands()),
        indexWidth(op.getIndex().getType().getDataBitWidth()), cMergeTEHB() {
    for (auto oper : op->getOperands())
      ins.push_back(getState<ConsumerRW>(oper, subset));

    outs = getState<ProducerRW>(op.getResult(), subset);
    index = getState<ChannelProducerRW>(op.getIndex(), subset);

    if (auto *p = dyn_cast<ChannelProducerRW>(outs)) {
      outsData = &p->data;
    } else {
      outsData = nullptr;
    }

    for (unsigned i = 0; i < size; ++i)
      if (auto *p = dyn_cast<ChannelConsumerRW>(ins[i])) {
        insData.push_back(&p->data);
      } else {
        insData.push_back(nullptr);
      }
  }

  void reset() override {
    indexTEHB = APInt(indexWidth, 0);
    for (unsigned i = 0; i < size; ++i)
      if (ins[i]->valid) {
        indexTEHB = i;
        break;
      }

    // mergeNotehbDataless
    dataAvailable = false;
    for (auto &in : ins) {
      dataAvailable = dataAvailable || in->valid;
    }

    if (indexTEHB.has_value())
      cMergeTEHB.resetDataFull(dataAvailable, readyToFork, tehbOutReady,
                               tehbOutValid, index->data, indexTEHB);
    else
      cMergeTEHB.resetDataless(dataAvailable, readyToFork, tehbOutReady,
                               tehbOutValid);
    for (auto &in : ins) {
      in->ready = tehbOutReady;
    }
    std::vector<bool> outsReady{outs->ready, index->ready};
    std::vector<std::reference_wrapper<bool>> outsValid{outs->valid,
                                                        index->valid};

    cMergeFork.resetDataless(tehbOutValid, outsReady, readyToFork, outsValid);
    auto ind = any_cast<APInt>(index->data).getZExtValue();
    if (insData[ind])
      *outsData = *insData[ind];
  };
  void exec(bool isClkRisingEdge) override {
    indexTEHB = APInt(indexWidth, 0);
    for (unsigned i = 0; i < size; ++i)
      if (ins[i]->valid) {
        indexTEHB = i;
        break;
      }

    // mergeNotehbDataless
    dataAvailable = false;
    for (auto &in : ins) {
      dataAvailable = dataAvailable || in->valid;
    }

    if (indexTEHB.has_value())
      cMergeTEHB.execDataFull(isClkRisingEdge, dataAvailable, readyToFork,
                              tehbOutReady, tehbOutValid, index->data,
                              indexTEHB);
    else
      cMergeTEHB.execDataless(isClkRisingEdge, dataAvailable, readyToFork,
                              tehbOutReady, tehbOutValid);
    for (auto &in : ins) {
      in->ready = tehbOutReady;
    }
    std::vector<bool> outsReady{outs->ready, index->ready};
    std::vector<std::reference_wrapper<bool>> outsValid{outs->valid,
                                                        index->valid};

    cMergeFork.execDataless(isClkRisingEdge, tehbOutValid, outsReady,
                            readyToFork, outsValid);
    auto ind = any_cast<APInt>(index->data).getZExtValue();
    if (insData[ind])
      *outsData = *insData[ind];
  }

private:
  handshake::ControlMergeOp op;
  unsigned size, indexWidth;
  TEHBSupport cMergeTEHB;
  ForkSupport cMergeFork;

  std::vector<ConsumerRW *> ins;
  ProducerRW *outs;
  ChannelProducerRW *index;

  std::vector<Any const *> insData;
  Any *outsData = nullptr;

  Any indexTEHB = APInt(indexWidth, 0);
  bool dataAvailable = false, readyToFork = false, tehbOutValid = false,
       tehbOutReady = false;
};

class ConstantModel : public OpExecutionModel<handshake::ConstantOp> {
public:
  using OpExecutionModel<handshake::ConstantOp>::OpExecutionModel;
  ConstantModel(handshake::ConstantOp constOp,
                mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::ConstantOp>(constOp), op(constOp),
        dataWidth(op.getResult().getType().getDataBitWidth()),
        value(dyn_cast<mlir::IntegerAttr>(op.getValue()).getValue()) {
    ctrl = getState<ControlConsumerRW>(op.getCtrl(), subset);
    outs = getState<ChannelProducerRW>(op.getResult(), subset);
  }
  void reset() override {
    outs->data = value;
    outs->valid = ctrl->valid;
    ctrl->ready = outs->ready;
  };
  void exec(bool isClkRisingEdge) override { reset(); }

private:
  handshake::ConstantOp op;
  unsigned dataWidth;
  Any value;
  ControlConsumerRW *ctrl;
  ChannelProducerRW *outs;
};

class ForkModel : public OpExecutionModel<handshake::ForkOp> {
public:
  using OpExecutionModel<handshake::ForkOp>::OpExecutionModel;
  ForkModel(handshake::ForkOp forkOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::ForkOp>(forkOp), op(forkOp),
        size(op->getNumResults()) {
    ins = getState<ConsumerRW>(op.getOperand(), subset);
    for (unsigned i = 0; i < size; ++i)
      outs.push_back(getState<ProducerRW>(op->getResult(i), subset));

    if (auto *p = dyn_cast<ChannelConsumerRW>(ins)) {
      insData = &p->data;
    } else {
      insData = nullptr;
    }

    for (unsigned i = 0; i < size; ++i)
      if (auto *p = dyn_cast<ChannelProducerRW>(outs[i])) {
        outsData.push_back(&p->data);
      } else {
        outsData.push_back(nullptr);
      }
  }
  void reset() override {
    std::vector<bool> outsReady;
    std::vector<std::reference_wrapper<bool>> outsValid;
    for (auto &b : outs) {
      outsReady.push_back(b->ready);
      outsValid.emplace_back(b->valid);
    }
    if (insData)
      forkSupport.resetDataFull(ins->valid, outsReady, ins->ready, outsValid,
                                outsData, *insData);
    else
      forkSupport.resetDataless(ins->valid, outsReady, ins->ready, outsValid);
  };
  void exec(bool isClkRisingEdge) override {
    std::vector<bool> outsReady;
    std::vector<std::reference_wrapper<bool>> outsValid;
    for (auto &b : outs) {
      outsReady.push_back(b->ready);
      outsValid.emplace_back(b->valid);
    }
    if (insData)
      forkSupport.execDataFull(isClkRisingEdge, ins->valid, outsReady,
                               ins->ready, outsValid, outsData, *insData);
    else
      forkSupport.execDataless(isClkRisingEdge, ins->valid, outsReady,
                               ins->ready, outsValid);
  }

private:
  handshake::ForkOp op;
  unsigned size;
  ForkSupport forkSupport;

  ConsumerRW *ins;
  std::vector<ProducerRW *> outs;

  Any const *insData = nullptr;
  std::vector<Any *> outsData;
};

class LazyForkModel : public OpExecutionModel<handshake::LazyForkOp> {
public:
  using OpExecutionModel<handshake::LazyForkOp>::OpExecutionModel;
  LazyForkModel(handshake::LazyForkOp lazyForkOp,
                mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::LazyForkOp>(lazyForkOp), op(lazyForkOp),
        size(op->getNumResults()) {
    ins = getState<ConsumerRW>(op.getOperand(), subset);
    for (unsigned i = 0; i < size; ++i)
      outs.push_back(getState<ProducerRW>(op->getResult(i), subset));
    if (auto *p = dyn_cast<ChannelConsumerRW>(ins)) {
      insData = &p->data;
    } else {
      insData = nullptr;
    }

    for (unsigned i = 0; i < size; ++i)
      if (auto *p = dyn_cast<ChannelProducerRW>(outs[i])) {
        outsData.push_back(&p->data);
      } else {
        outsData.push_back(nullptr);
      }
  }
  void reset() override {
    ins->ready = true;
    for (unsigned i = 0; i < size; ++i) {
      bool tempReady = true;
      for (unsigned j = 0; j < size; ++j) {
        if (i != j)
          tempReady = tempReady && outs[j]->ready;
      }
      ins->ready = ins->ready && outs[i]->ready;
      outs[i]->valid = ins->valid && tempReady;
    }
    for (unsigned i = 0; i < size; ++i)
      if (outsData[i])
        *outsData[i] = *insData;
  };
  void exec(bool isClkRisingEdge) override { reset(); }

private:
  handshake::ForkOp op;
  unsigned size;

  ConsumerRW *ins;
  std::vector<ProducerRW *> outs;

  Any const *insData = nullptr;
  std::vector<Any *> outsData;
};

class MergeModel : public OpExecutionModel<handshake::MergeOp> {
public:
  using OpExecutionModel<handshake::MergeOp>::OpExecutionModel;
  MergeModel(handshake::MergeOp mergeOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::MergeOp>(mergeOp), op(mergeOp),
        size(op->getNumOperands()) {
    for (unsigned i = 0; i < size; ++i)
      ins.push_back(getState<ConsumerRW>(op->getOperand(i), subset));
    outs = getState<ProducerRW>(op.getResult(), subset);

    if (auto *p = dyn_cast<ChannelProducerRW>(outs)) {
      outsData = &p->data;
    } else {
      outsData = nullptr;
    }

    for (unsigned i = 0; i < size; ++i)
      if (auto *p = dyn_cast<ChannelConsumerRW>(ins[i])) {
        insData.push_back(&p->data);
      } else {
        insData.push_back(nullptr);
      }
  }

  void reset() override {
    tehbValid = false;
    if (insData[0])
      tehbDataIn = *insData[0];

    for (unsigned i = 0; i < size; ++i) {
      if (ins[i]->valid) {
        tehbValid = true;
        tehbDataIn = *insData[i];
        break;
      }
    }
    if (outsData)
      mergeTEHB.resetDataFull(tehbValid, outs->ready, tehbReady, outs->valid,
                              *outsData, tehbDataIn);
    else
      mergeTEHB.resetDataless(tehbValid, outs->ready, tehbReady, outs->valid);

    for (unsigned i = 0; i < size; ++i)
      ins[i]->ready = tehbReady;
  };

  void exec(bool isClkRisingEdge) override {
    tehbValid = false;
    if (insData[0])
      tehbDataIn = *insData[0];

    for (unsigned i = 0; i < size; ++i) {
      if (ins[i]->valid) {
        tehbValid = true;
        if (insData[i])
          tehbDataIn = *insData[i];
        break;
      }
    }

    if (outsData)
      mergeTEHB.execDataFull(isClkRisingEdge, tehbValid, outs->ready, tehbReady,
                             outs->valid, *outsData, tehbDataIn);
    else
      mergeTEHB.execDataless(isClkRisingEdge, tehbValid, outs->ready, tehbReady,
                             outs->valid);

    for (unsigned i = 0; i < size; ++i)
      ins[i]->ready = tehbReady;
  }

private:
  handshake::MergeOp op;
  unsigned size;
  bool tehbValid = false, tehbReady = false;
  Any tehbDataIn;

  TEHBSupport mergeTEHB;

  std::vector<ConsumerRW *> ins;
  ProducerRW *outs;

  std::vector<Any const *> insData;
  Any *outsData = nullptr;
};

class MuxModel : public OpExecutionModel<handshake::MuxOp> {
public:
  using OpExecutionModel<handshake::MuxOp>::OpExecutionModel;
  MuxModel(handshake::MuxOp muxOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::MuxOp>(muxOp), op(muxOp),
        size(op.getDataOperands().size()),
        selectWidth(op.getSelectOperand().getType().getDataBitWidth()),
        muxTEHB() {
    for (auto oper : op.getDataOperands())
      ins.push_back(getState<ChannelConsumerRW>(oper, subset));
    index = getState<ChannelConsumerRW>(op.getSelectOperand(), subset);
    outs = getState<ProducerRW>(op.getDataResult(), subset);

    if (auto *p = dyn_cast<ChannelProducerRW>(outs)) {
      outsData = &p->data;
    } else {
      outsData = nullptr;
    }

    for (unsigned i = 0; i < size; ++i)
      if (auto *p = dyn_cast<ChannelConsumerRW>(ins[i])) {
        insData.push_back(&p->data);
      } else {
        insData.push_back(nullptr);
      }
  }
  void reset() override {
    if (outsData)
      muxTEHB.resetDataFull(tehbInsValid, outs->ready, tehbInsReady,
                            outs->valid, *outsData, tehbIns);
    else
      muxTEHB.resetDataless(tehbInsValid, outs->ready, tehbInsReady,
                            outs->valid);
    chooseInput();
  };
  void exec(bool isClkRisingEdge) override {
    if (outsData)
      muxTEHB.execDataFull(isClkRisingEdge, tehbInsValid, outs->ready,
                           tehbInsReady, outs->valid, *outsData, tehbIns);
    else
      muxTEHB.execDataless(isClkRisingEdge, tehbInsValid, outs->ready,
                           tehbInsReady, outs->valid);
    chooseInput();
  }

private:
  handshake::MuxOp op;
  unsigned size, selectWidth;
  TEHBSupport muxTEHB;

  std::vector<ConsumerRW *> ins;
  ChannelConsumerRW *index;
  ProducerRW *outs;

  std::vector<Any const *> insData;
  Any *outsData = nullptr;

  Any tehbIns = {};
  bool tehbInsValid = false, tehbInsReady = false;

  void chooseInput() {
    Any selectedData;
    bool indexEqual = false, selectedDataValid = false;
    for (unsigned i = 0; i < size; ++i) {
      indexEqual = llvm::any_cast<APInt>(index->data) == i;
      if (indexEqual && index->valid && ins[i]->valid) {
        if (insData[i])
          selectedData = *insData[i];
        selectedDataValid = true;
      }
      ins[i]->ready =
          (indexEqual && index->valid && ins[i]->valid && tehbInsReady) ||
          (!ins[i]->valid);
    }
    index->ready = !index->valid || (selectedDataValid && tehbInsReady);
    if (selectedData.has_value())
      tehbIns = selectedData;
    tehbInsValid = selectedDataValid;
  }
};

class OEHBModel : public OpExecutionModel<handshake::BufferOp> {
public:
  using OpExecutionModel<handshake::BufferOp>::OpExecutionModel;

  OEHBModel(handshake::BufferOp oehbOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::BufferOp>(oehbOp), op(oehbOp), oehbDl() {
    ins = getState<ConsumerRW>(op.getOperand(), subset);
    outs = getState<ProducerRW>(op.getResult(), subset);

    if (auto *p = dyn_cast<ChannelProducerRW>(outs)) {
      outsData = &p->data;
    } else {
      outsData = nullptr;
    }
    if (auto *p = dyn_cast<ChannelConsumerRW>(ins)) {
      insData = &p->data;
    } else {
      insData = nullptr;
    }
  }
  void reset() override {
    if (insData)
      oehbDl.resetDataFull(ins->valid, outs->ready, ins->ready, outs->valid,
                           *outsData, *insData);
    else
      oehbDl.resetDataless(ins->valid, outs->ready, ins->ready, outs->valid);
  };
  void exec(bool isClkRisingEdge) override {
    if (insData)
      oehbDl.execDataFull(isClkRisingEdge, ins->valid, outs->ready, ins->ready,
                          outs->valid, *outsData, *insData);
    else
      oehbDl.execDataless(isClkRisingEdge, ins->valid, outs->ready, ins->ready,
                          outs->valid);
  }

private:
  handshake::BufferOp op;
  OEHBSupport oehbDl;

  ConsumerRW *ins;
  ProducerRW *outs;

  Any const *insData = nullptr;
  Any *outsData = nullptr;
};

class SinkModel : public OpExecutionModel<handshake::SinkOp> {
public:
  using OpExecutionModel<handshake::SinkOp>::OpExecutionModel;
  SinkModel(handshake::SinkOp sinkOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::SinkOp>(sinkOp), op(sinkOp) {
    ins = getState<ConsumerRW>(op.getOperand(), subset);
  }
  void reset() override { ins->ready = true; };
  void exec(bool isClkRisingEdge) override { reset(); }

private:
  handshake::SinkOp op;
  ConsumerRW *ins;
};

class SourceModel : public OpExecutionModel<handshake::SourceOp> {
public:
  using OpExecutionModel<handshake::SourceOp>::OpExecutionModel;
  SourceModel(handshake::SourceOp sourceOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::SourceOp>(sourceOp), op(sourceOp) {
    outs = getState<ProducerRW>(op.getResult(), subset);
  }
  void reset() override { outs->valid = true; };
  void exec(bool isClkRisingEdge) override { reset(); }

private:
  handshake::SourceOp op;
  ProducerRW *outs;
};

class TEHBModel : public OpExecutionModel<handshake::BufferOp> {
public:
  using OpExecutionModel<handshake::BufferOp>::OpExecutionModel;

  TEHBModel(handshake::BufferOp returnOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::BufferOp>(returnOp), op(returnOp),
        returnTEHB() {

    ins = getState<ConsumerRW>(op.getOperand(), subset);
    outs = getState<ProducerRW>(op.getResult(), subset);
    if (auto *p = dyn_cast<ChannelProducerRW>(outs)) {
      outsData = &p->data;
    } else {
      outsData = nullptr;
    }
    if (auto *p = dyn_cast<ChannelConsumerRW>(ins)) {
      insData = &p->data;
    } else {
      insData = nullptr;
    }
  }
  void reset() override {
    if (outsData)
      returnTEHB.resetDataFull(ins->valid, outs->ready, ins->ready, outs->valid,
                               *outsData, *insData);
    else
      returnTEHB.resetDataless(ins->valid, outs->ready, ins->ready,
                               outs->valid);
  };

  void exec(bool isClkRisingEdge) override {
    if (outsData)
      returnTEHB.execDataFull(isClkRisingEdge, ins->valid, outs->ready,
                              ins->ready, outs->valid, *outsData, *insData);
    else
      returnTEHB.execDataless(isClkRisingEdge, ins->valid, outs->ready,
                              ins->ready, outs->valid);
  }

private:
  handshake::BufferOp op;

  ConsumerRW *ins;
  ProducerRW *outs;

  Any const *insData = nullptr;
  Any *outsData = nullptr;

  TEHBSupport returnTEHB;
};

/// In fact, there's no need in a specific class for End - it just sets
/// the ready for the circuit on reset (e.g. once). To be refactored later
class EndMemlessModel : public OpExecutionModel<handshake::EndOp> {
public:
  using OpExecutionModel<handshake::EndOp>::OpExecutionModel;
  EndMemlessModel(handshake::EndOp endOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::EndOp>(endOp), op(endOp) {
    ins = getState<ChannelConsumerRW>(op.getInputs().front(), subset);

    auto temp =
        cast<handshake::ChannelType>(endOp.getInputs().front().getType());
    bitwidth = temp.getDataBitWidth();
  }
  void reset() override { ins->ready = true; };
  void exec(bool isClkRisingEdge) override {}

private:
  handshake::EndOp op;
  unsigned bitwidth;

  ChannelConsumerRW *ins;
};

// negf - not
class TruncIModel : public OpExecutionModel<handshake::TruncIOp> {
public:
  using OpExecutionModel<handshake::TruncIOp>::OpExecutionModel;
  TruncIModel(handshake::TruncIOp trunciOp, mlir::DenseMap<Value, RW *> &subset)
      : OpExecutionModel<handshake::TruncIOp>(trunciOp), op(trunciOp),
        outputWidth(op.getResult().getType().getDataBitWidth()) {
    ins = getState<ChannelConsumerRW>(op.getOperand(), subset);
    outs = getState<ChannelProducerRW>(op.getResult(), subset);
  }
  void reset() override {
    APInt insData =
        llvm::any_cast<APInt>(ins->data).getBitsSetFrom(outputWidth, 0);
    outs->data = insData;
    outs->valid = ins->valid;
    ins->ready = !ins->valid || (ins->valid && outs->ready);
  };
  void exec(bool isClkRisingEdge) override { reset(); }

private:
  handshake::TruncIOp op;
  unsigned outputWidth;

  ChannelConsumerRW *ins;
  ChannelProducerRW *outs;
};

using UnaryCompFunc = std::function<llvm::Any(const llvm::Any &, unsigned)>;

// NEGF, NOT, EXTSI, EXTUI
template <typename Op>
class GenericUnaryOpModel : public OpExecutionModel<Op> {
public:
  using OpExecutionModel<Op>::OpExecutionModel;
  GenericUnaryOpModel(Op op, mlir::DenseMap<Value, RW *> &subset,
                      const UnaryCompFunc &callback)
      : OpExecutionModel<Op>(op), op(op), callback(callback),
        outputWidth(op.getResult().getType().getDataBitWidth()) {
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
  void exec(bool isClkRisingEdge) override {
    outs->data = callback(ins->data, outputWidth);
    outs->valid = ins->valid;
    ins->ready = outs->ready;
  }

private:
  Op op;
  UnaryCompFunc callback;
  unsigned outputWidth;

  ChannelConsumerRW *ins;
  ChannelProducerRW *outs;
};

// Mutual component for binary operations: AddI, AndI, CmpI, MulI, OrI,
// ShlI, ShrsI, ShruI, SubI, XorI
using BinaryCompFunc =
    std::function<llvm::Any(const llvm::Any &, const llvm::Any &)>;
template <typename Op>
class GenericBinaryOpModel : public OpExecutionModel<Op> {
public:
  using OpExecutionModel<Op>::OpExecutionModel;
  GenericBinaryOpModel(Op op, mlir::DenseMap<Value, RW *> &subset,
                       const BinaryCompFunc &callback, unsigned latency)
      : OpExecutionModel<Op>(op), op(op), callback(callback), latency(latency),
        bitwidth(cast<handshake::ChannelType>(op.getResult().getType())
                     .getDataBitWidth()),
        binJoin(2),
        lhs(OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
            op.getLhs(), subset)),
        rhs(OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
            op.getRhs(), subset)),
        result(OpExecutionModel<Op>::template getState<ChannelProducerRW>(
            op.getResult(), subset)) {}
  void reset() override {
    std::vector<bool> insValid{lhs->valid, rhs->valid};
    std::vector<std::reference_wrapper<bool>> insReady{lhs->ready, rhs->ready};
    binJoin.exec(insValid, result->ready, insReady, result->valid);
    resData = APInt(bitwidth, 0);
    hasData = false;
    counter = 0;
  };
  void exec(bool isClkRisingEdge) override {
    std::vector<bool> insValid{lhs->valid, rhs->valid};
    std::vector<std::reference_wrapper<bool>> insReady{lhs->ready, rhs->ready};
    binJoin.exec(insValid, result->ready, insReady, joinValid);

    // increase counter if the component stores data
    if (hasData && isClkRisingEdge) {
      ++counter;
    }

    // read data
    if (lhs->ready && rhs->ready && lhs->valid && rhs->valid && !hasData) {
      resData = any_cast<APInt>(callback(lhs->data, rhs->data));
      hasData = true;
      ++counter;
    }

    // ready to transfer data
    if (counter > latency && hasData) {
      result->valid = true;
      result->data = resData;
      // reset counter if the consumer is ready to transfer
      if (result->ready) {
        counter = 0;
        resData = APInt(bitwidth, 0);
        hasData = false;
      }
    }
  }

private:
  Op op;
  BinaryCompFunc callback;
  unsigned latency = 0;
  unsigned bitwidth;
  Join binJoin;
  ChannelConsumerRW *lhs, *rhs;
  ChannelProducerRW *result;

  APInt resData = APInt(bitwidth, 0);
  unsigned counter = 0;
  bool hasData = false;
  bool joinValid = false;
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
  }

  void reset() {
    // Collect all the states
    for (auto &[op, model] : opModels) {
      llvm::outs() << "Reset = " << op->getAttr("handshake.name") << "\n";
      model->reset();
    }

    // update old values with collected ones
    for (auto [val, state] : updaters)
      state->update();
    llvm::outs() << "======= reset =========\n";
    printValuesStates();
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
      llvm::TypeSwitch<Type>(channel.getType().getDataType())
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

    auto returnOp = *funcOp.getOps<handshake::BufferOp>().begin();
    // auto *channelArg = static_cast<ChannelProducerRW
    // *>(producerViews[returnOp.getOutputs().front()]);

    auto *res =
        static_cast<ChannelProducerRW *>(producerViews[returnOp.getResult()]);
    /*
        for (unsigned temp = 0; temp < 10; temp++) {
          bool isClock = true;
          llvm::outs() << "Clock " << temp << "\n";
          llvm::outs() << "Before:\n";
          printValuesStates();
          for (int i = 0; i < 6; ++i) {
            llvm::outs() << "Propagate\n";
            // Execute each model
            for (auto &[op, model] : opModels)
              model->exec(isClock);

            for (auto [val, state] : updaters)
              state->update();
            isClock = false;
            printValuesStates();
            llvm::outs() << "++++++++++++++\n";
          }
          llvm::outs() << "After:\n";
          printValuesStates();
          llvm::outs() << "----------------\n";
          for (auto [val, state] : updaters)
            state->resetValid();
        }
        */

    unsigned eqStateCount = 0;
    unsigned temp = 0;
    // Outer loop: for clock cycles
    // Stops when return.input->valid becomes true or limit of cycles is reached
    while (true) {
      llvm::outs() << "Clock " << ++temp << "\n At the beginning\n";

      printValuesStates();
      //  True only once on each clkRisingEdge
      bool isClock = true;
      // Inner loop: for signal propagation within one clkRisingEdge
      // Stops when there's no change in valueStates
      while (true) {
        llvm::outs() << "[[[[[[[[[[[  Propagate  ]]]]]]]]]]]]\n";
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
        printValuesStates();
        llvm::outs() << "++++++++++++++\n";
      }
      llvm::outs() << "At the end\n";
      printValuesStates();

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
      for (auto [val, state] : updaters)
        state->resetValid();
    }
  }

  // Just a temporary function to print the results of the simulation to
  // standart output
  void printResults() {
    auto returnOp = *funcOp.getOps<handshake::BufferOp>().begin();
    llvm::outs() << "Result:\n";
    for (auto res : returnOp->getResults()) {

      auto *state = static_cast<ChannelProducerRW *>(producerViews[res]);
      if (state->data.has_value())
        llvm::outs() << llvm::any_cast<APInt>(state->data);
      llvm::outs() << "\n";
    }
  }

  // Just a temporary function to print the current valueStates to
  // standart output
  void printValuesStates() {
    for (auto x : newValuesStates) {
      auto val = x.first;
      auto *state = x.second;
      llvm::outs() << val << ":\n";
      llvm::TypeSwitch<Type>(val.getType())
          .Case<handshake::ChannelType>(
              [&](handshake::ChannelType channelType) {
                auto *cl = static_cast<ChannelState *>(state);
                llvm::outs() << cl->valid << " " << cl->ready << " ";
                if (cl->data.has_value())
                  llvm::outs() << llvm::any_cast<APInt>(cl->data);
                llvm::outs() << "\n";
              })
          .Case<handshake::ControlType>(
              [&](handshake::ControlType controlType) {
                auto *cl = static_cast<ControlState *>(state);
                llvm::outs() << cl->valid << " " << cl->ready << "\n";
              })
          .Default([&](auto) {
            llvm::outs() << "Value " << val
                         << " has unsupported type, we should probably "
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
    for (auto &opOp : opOperands) {
      subset.insert({opOp.get(), consumerViews[&opOp]});
    }
    for (auto res : results) {
      subset.insert({res, producerViews[res]});
    }

    ExecutionModel *model =
        new Model(op, subset, std::forward<Args>(modelArgs)...);
    opModels.insert({op, model});
  }

  // Determine the concrete Model type
  void associateModel(Operation *op) {
    llvm::outs() << op->getAttr("handshake.name") << "\n";
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
          UnaryCompFunc callback = [](const llvm::Any &lhs, unsigned outWidth) {
            auto temp = llvm::any_cast<APInt>(lhs).getSExtValue();
            APInt ext(temp, outWidth);
            return ext;
          };

          registerModel<GenericUnaryOpModel<handshake::ExtSIOp>>(extsiOp,
                                                                 callback);
        })
        .Case<handshake::ShLIOp>([&](handshake::ShLIOp shliOp) {
          BinaryCompFunc callback = [](llvm::Any lhs, llvm::Any rhs) {
            APInt shli = any_cast<APInt>(lhs) << any_cast<APInt>(rhs);
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
            callback = [](llvm::Any lhs, llvm::Any rhs) {
              bool comp = any_cast<APInt>(lhs).getRawData() ==
                          any_cast<APInt>(rhs).getRawData();
              return comp;
            };
            break;
          case handshake::CmpIPredicate::ne:
            callback = [](llvm::Any lhs, llvm::Any rhs) {
              bool comp = any_cast<APInt>(lhs).getRawData() !=
                          any_cast<APInt>(rhs).getRawData();
              return comp;
            };
            break;
          case handshake::CmpIPredicate::uge:
            callback = [](llvm::Any lhs, llvm::Any rhs) {
              bool comp = any_cast<APInt>(lhs).getZExtValue() >=
                          any_cast<APInt>(rhs).getZExtValue();
              return comp;
            };
            break;
          case handshake::CmpIPredicate::sge:
            callback = [](llvm::Any lhs, llvm::Any rhs) {
              bool comp = any_cast<APInt>(lhs).getSExtValue() >=
                          any_cast<APInt>(rhs).getSExtValue();
              return comp;
            };
            break;
          case handshake::CmpIPredicate::ugt:
            callback = [](llvm::Any lhs, llvm::Any rhs) {
              bool comp = any_cast<APInt>(lhs).getZExtValue() >
                          any_cast<APInt>(rhs).getZExtValue();
              return comp;
            };
            break;
          case handshake::CmpIPredicate::sgt:
            callback = [](llvm::Any lhs, llvm::Any rhs) {
              bool comp = any_cast<APInt>(lhs).getSExtValue() >
                          any_cast<APInt>(rhs).getSExtValue();
              return comp;
            };
            break;
          case handshake::CmpIPredicate::ule:
            callback = [](llvm::Any lhs, llvm::Any rhs) {
              bool comp = any_cast<APInt>(lhs).getZExtValue() <=
                          any_cast<APInt>(rhs).getZExtValue();
              return comp;
            };
            break;
          case handshake::CmpIPredicate::sle:
            callback = [](llvm::Any lhs, llvm::Any rhs) {
              bool comp = any_cast<APInt>(lhs).getSExtValue() <=
                          any_cast<APInt>(rhs).getSExtValue();
              return comp;
            };
            break;
          case handshake::CmpIPredicate::ult:
            callback = [](llvm::Any lhs, llvm::Any rhs) {
              bool comp = any_cast<APInt>(lhs).getZExtValue() <
                          any_cast<APInt>(rhs).getZExtValue();
              return comp;
            };
            break;
          case handshake::CmpIPredicate::slt:
            callback = [](llvm::Any lhs, llvm::Any rhs) {
              bool comp = any_cast<APInt>(lhs).getSExtValue() <
                          any_cast<APInt>(rhs).getSExtValue();
              return comp;
            };
            break;
          }

          registerModel<GenericBinaryOpModel<handshake::CmpIOp>>(cmpiOp,
                                                                 callback, 0);
        })
        /*
        .Case<handshake::BufferOp>([&](handshake::BufferOp oehbOp) {
          registerModel<OEHBModel, handshake::BufferOp>(oehbOp);
        })
        */
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
          BinaryCompFunc callback = [](llvm::Any lhs, llvm::Any rhs) {
            APInt mul = any_cast<APInt>(lhs) * any_cast<APInt>(rhs);
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
    llvm::TypeSwitch<Type>(val.getType())
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

  sim.reset();
  // sim.simulate(inputArgs);
  // sim.printResults();
}
