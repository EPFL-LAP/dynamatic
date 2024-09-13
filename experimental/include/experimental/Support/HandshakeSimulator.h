#ifndef EXPERIMENTAL_SUPPORT_HANDSHAKE_SIMULATOR_H
#define EXPERIMENTAL_SUPPORT_HANDSHAKE_SIMULATOR_H
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <string>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

namespace dynamatic {
namespace experimental {

// The wrapper for llvm::Any that lets the user check if the value was changed
// It's required for updaters to stop the internal loop
struct Data {
  llvm::Any data;
  llvm::hash_code hash;
  unsigned bitwidth;

  Data() = default;

  Data(const APInt &value);

  Data(const APFloat &value);

  Data &operator=(const APInt &value);

  Data &operator=(const APFloat &value);

  bool hasValue() const;
};

//===----------------------------------------------------------------------===//
// States
//===----------------------------------------------------------------------===//
/// Classes that store states.
/// The simulator defines 2 maps - newValueStates and oldValueStates - to
/// represent the previous (used during the signals' collection to prevent "data
/// race conditions") and current (where we collect new values to). The map
/// updaters (inside the Simulator class as well), in fact, containing
/// references to corresponding pair of new and old valueStates, assign the
/// newly collected values from newValueStates to oldValueStates when the
/// collection is finished.

// The state in circuit's value states
class ValueState {
public:
  ValueState(Value val);

  virtual ~ValueState() = default;

protected:
  Value val;
};

/// The state, templated with concrete hasndshake value type
template <typename Ty>
class TypedValueState : public ValueState {
public:
  TypedValueState(TypedValue<Ty> val);
};

class ChannelState : public TypedValueState<handshake::ChannelType> {
  // Give the corresponding RW API an access to data members
  friend struct ChannelConsumerRW;
  friend struct ChannelProducerRW;
  // Give the corresponding Updater an access to data members
  friend class ChannelUpdater;

public:
  using TypedValueState<handshake::ChannelType>::TypedValueState;

  ChannelState(TypedValue<handshake::ChannelType> channel);

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

  ControlState(TypedValue<handshake::ControlType> control);

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
  DoubleUpdater(State &oldState, State &newState);
};

/// Update valueStates with "Channel type": valid, ready, data
class ChannelUpdater : public DoubleUpdater<ChannelState> {
public:
  ChannelUpdater(ChannelState &oldState, ChannelState &newState);
  bool check() override;
  void setValid() override;
  void resetValid() override;
  void update() override;
};

/// Update valueStates with "Control type": valid, ready
class ControlUpdater : public DoubleUpdater<ControlState> {
public:
  ControlUpdater(ControlState &oldState, ControlState &newState);
  bool check() override;
  void setValid() override;
  void resetValid() override;
  void update() override;
};

//===----------------------------------------------------------------------===//
// Readers & writers
//===----------------------------------------------------------------------===//

/// Classes with references to oldValueStates and newValueStates to represent
/// readers/writers API (that is, opened the user's access). The map
/// <value, ProducerRW*> producerViews (inside the Simulator class) stores such
/// API's for values, data & valid signals of which can be changed. The map
/// <OpOperand*, ConsumerRW*> consumerViews contains pointers to "consumer
/// views" for all uses of the particular value. Keys in these maps are not the
/// same because multiple users may share the same value. The case when there're
/// several OpOperands connecting 2 components within one value is considered to
/// be a just one OpOperand.

/// Base RW
class RW {
public:
  RW() = default;
  virtual ~RW() = default;
};

class ProducerRW : public RW {
protected:
  enum ProducerDescendants { D_ChannelProducerRW, D_ControlProducerRW };

public:
  bool &valid;
  const bool &ready;

public:
  ProducerDescendants getType() const;

  ProducerRW(bool &valid, const bool &ready,
             ProducerDescendants p = D_ControlProducerRW);

  ProducerRW(ProducerRW &p);

  virtual ~ProducerRW() = default;

private:
  // LLVM RTTI is used here for convenient conversion
  const ProducerDescendants prod;
};

class ConsumerRW : public RW {
public:
  const bool &valid;
  bool &ready;

protected:
  enum ConsumerDescendants { D_ChannelConsumerRW, D_ControlConsumerRW };

public:
  ConsumerDescendants getType() const;

  ConsumerRW(const bool &valid, bool &ready,
             ConsumerDescendants c = D_ControlConsumerRW);

  ConsumerRW(ConsumerRW &c);

  virtual ~ConsumerRW() = default;

private:
  // LLVM RTTI is used here for convenient conversion
  const ConsumerDescendants cons;
};

////--- Control (no data)

/// In this case the user can change the ready signal, but has
/// ReadOnly access to the valid one.
struct ControlConsumerRW : public ConsumerRW {
  ControlConsumerRW(ControlState &reader, ControlState &writer);

  static bool classof(const ConsumerRW *c);
};

/// In this case the user can change the valid signal, but has
/// ReadOnly access to the ready one.
struct ControlProducerRW : public ProducerRW {
  ControlProducerRW(ControlState &reader, ControlState &writer);

  static bool classof(const ProducerRW *c);
};

////--- Channel (data)

/// In this case the user can change the valid signal, but has ReadOnly access
/// to the valid and data ones.
struct ChannelConsumerRW : public ConsumerRW {
  const Data &data;

  ChannelConsumerRW(ChannelState &reader, ChannelState &writer);

  static bool classof(const ConsumerRW *c);
};

/// In this case the user can change the valid and data signals, but has
/// ReadOnly access to the ready one.
struct ChannelProducerRW : public ProducerRW {
  Data &data;

  ChannelProducerRW(ChannelState &reader, ChannelState &writer);

  static bool classof(const ProducerRW *c);
};

//===----------------------------------------------------------------------===//
// Datafull & Dataless
//===----------------------------------------------------------------------===//
/// The following structures are used to represent components that can be
/// either datafull or dataless. They store a pointer to Data and maybe some
/// userful members and functions (e.g. hasValue() or the width).

/// A struct to represent consumer's data
struct ConsumerData {
  Data const *data = nullptr;
  unsigned dataWidth = 0;

  ConsumerData(ConsumerRW *ins);

  bool hasValue() const;
};

/// A struct to represent producer's data
struct ProducerData {
  Data *data = nullptr;
  unsigned dataWidth = 0;

  ProducerData(ProducerRW *outs);

  ProducerData &operator=(const ConsumerData &value);

  bool hasValue() const;
};

//===----------------------------------------------------------------------===//
// Execution Model
//===----------------------------------------------------------------------===//
/// Classes to represent execution models

/// Base class
class ExecutionModel {

public:
  ExecutionModel(Operation *op);
  virtual void reset() = 0;
  virtual void exec(bool isClkRisingEdge) = 0;
  virtual void printStates() = 0;

  virtual ~ExecutionModel() = default;

protected:
  Operation *op;
};

/// Typed execution model
template <typename Op>
class OpExecutionModel : public ExecutionModel {
public:
  OpExecutionModel(Op op);

protected:
  Op getOperation();

  // Get exact RW state type (ChannelProducerRW / ChannelConsumerRW /
  // ControlProducerRW / ControlConsumerRW)
  template <typename State>
  static State *getState(Value val, mlir::DenseMap<Value, RW *> &rws);

  // Just a temporary function to print models' states
  template <typename State, typename Data>
  void printValue(const std::string &name, State *ins, Data *insData);
};

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//
/// Components required for the internal state.

class Antotokens {
public:
  Antotokens() = default;

  void reset(const bool &pvalid1, const bool &pvalid0, bool &kill1, bool &kill0,
             const bool &generateAt1, const bool &generateAt0, bool &stopValid);

  void exec(bool isClkRisingEdge, const bool &pvalid1, const bool &pvalid0,
            bool &kill1, bool &kill0, const bool &generateAt1,
            const bool &generateAt0, bool &stopValid);

private:
  bool regIn0 = false, regIn1 = false, regOut0 = false, regOut1 = false;
};

class ForkSupport {
public:
  ForkSupport(unsigned size, unsigned datawidth = 0);

  void resetDataless(ConsumerRW *ins, std::vector<ProducerRW *> &outs);

  void execDataless(bool isClkRisingEdge, ConsumerRW *ins,
                    std::vector<ProducerRW *> &outs);
  void reset(ConsumerRW *ins, std::vector<ProducerRW *> &outs,
             const ConsumerData &insData, std::vector<ProducerData> &outsData);

  void exec(bool isClkRisingEdge, ConsumerRW *ins,
            std::vector<ProducerRW *> &outs, const ConsumerData &insData,
            std::vector<ProducerData> &outsData);

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
  JoinSupport(unsigned size);

  void exec(std::vector<ConsumerRW *> &ins, ProducerRW *outs);

private:
  unsigned size;
};

class OEHBSupport {
public:
  OEHBSupport(unsigned datawidth = 0);

  void resetDataless(ConsumerRW *ins, ProducerRW *outs);

  void execDataless(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs);

  void reset(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
             Data *outsData);

  void exec(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
            const Data *insData, Data *outsData);

  unsigned datawidth;

private:
  // oehb dataless
  bool outputValid = false;
  // oehb datafull
  bool regEn = false;
};

class TEHBSupport {
public:
  TEHBSupport(unsigned datawidth = 0);
  void resetDataless(ConsumerRW *ins, ProducerRW *outs);
  void execDataless(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs);

  void reset(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
             Data *outsData);
  void exec(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
            const Data *insData, Data *outsData);
  unsigned datawidth = 0;

private:
  // tehb dataless
  bool fullReg = false, outputValid = false;
  // tehb datafull
  bool regNotFull = false, regEnable = false;
  Data dataReg;

  void resetDataFull(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
                     Data *outsData);

  void execDataFull(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
                    const Data *insData, Data *outsData);
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
  BranchModel(handshake::BranchOp branchOp,
              mlir::DenseMap<Value, RW *> &subset);
  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
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
                  mlir::DenseMap<Value, RW *> &subset);

  void reset() override;
  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
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
                mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  Data value;
  ControlConsumerRW *ctrl;
  ChannelProducerRW *outs;
};

class ControlMergeModel : public OpExecutionModel<handshake::ControlMergeOp> {
public:
  using OpExecutionModel<handshake::ControlMergeOp>::OpExecutionModel;
  ControlMergeModel(handshake::ControlMergeOp cMergeOp,
                    mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
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
  ForkModel(handshake::ForkOp forkOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
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
  JoinModel(handshake::JoinOp joinOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
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
                mlir::DenseMap<Value, RW *> &subset);

  void reset() override;
  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
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
  MergeModel(handshake::MergeOp mergeOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
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

  void execDataless();

  void execDataFull();
};

class MuxModel : public OpExecutionModel<handshake::MuxOp> {
public:
  using OpExecutionModel<handshake::MuxOp>::OpExecutionModel;
  MuxModel(handshake::MuxOp muxOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
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

  void execDataless();

  void execDataFull();
};

class OEHBModel : public OpExecutionModel<handshake::BufferOp> {
public:
  using OpExecutionModel<handshake::BufferOp>::OpExecutionModel;

  OEHBModel(handshake::BufferOp oehbOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
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
  SinkModel(handshake::SinkOp sinkOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  ConsumerRW *ins;
  ConsumerData insData;
};

class SourceModel : public OpExecutionModel<handshake::SourceOp> {
public:
  using OpExecutionModel<handshake::SourceOp>::OpExecutionModel;
  SourceModel(handshake::SourceOp sourceOp,
              mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  ProducerRW *outs;
};

class TEHBModel : public OpExecutionModel<handshake::BufferOp> {
public:
  using OpExecutionModel<handshake::BufferOp>::OpExecutionModel;

  TEHBModel(handshake::BufferOp tehbOp, mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  ConsumerRW *ins;
  ProducerRW *outs;

  ConsumerData insData;
  ProducerData outsData;

  // internal components
  TEHBSupport returnTEHB;
};

class EndModel : public OpExecutionModel<handshake::EndOp> {
public:
  using OpExecutionModel<handshake::EndOp>::OpExecutionModel;
  EndModel(handshake::EndOp endOp, mlir::DenseMap<Value, RW *> &subset,
           bool &resValid, const bool &resReady, Data &resData);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
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
  TruncIModel(handshake::TruncIOp trunciOp,
              mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // parameters
  unsigned outputWidth;

  // ports
  ChannelConsumerRW *ins;
  ChannelProducerRW *outs;
};

class SelectModel : public OpExecutionModel<handshake::SelectOp> {
public:
  using OpExecutionModel<handshake::SelectOp>::OpExecutionModel;
  SelectModel(handshake::SelectOp selectOp,
              mlir::DenseMap<Value, RW *> &subset);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // ports
  ChannelConsumerRW *condition, *trueValue, *falseValue;
  ChannelProducerRW *result;

  bool ee = false, validInternal = false, kill0 = false, kill1 = false,
       antitokenStop = false, g0 = false, g1 = false;
  // internal components
  Antotokens anti;

  void selectExec();
};

using UnaryCompFunc = std::function<Data(const Data &, unsigned)>;

/// Class to represent ExtSI, ExtUI, Negf, Not
template <typename Op>
class GenericUnaryOpModel : public OpExecutionModel<Op> {
public:
  using OpExecutionModel<Op>::OpExecutionModel;
  GenericUnaryOpModel(Op op, mlir::DenseMap<Value, RW *> &subset,
                      const UnaryCompFunc &callback);
  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
  // parameters
  unsigned outputWidth;
  UnaryCompFunc callback;

  // ports
  ChannelConsumerRW *ins;
  ChannelProducerRW *outs;
};

using BinaryCompFunc = std::function<Data(const Data &, const Data &)>;

/// Mutual component for binary operations: AddF, AddI, AndI, CmpF, CmpI, DivF,
/// DivSI, DivUI, Maximumf, Minimumf, MulF, MulI, OrI, ShlI, ShrSI, ShrUI, SubF,
/// SubI, XorI
template <typename Op>
class GenericBinaryOpModel : public OpExecutionModel<Op> {
public:
  using OpExecutionModel<Op>::OpExecutionModel;
  GenericBinaryOpModel(Op op, mlir::DenseMap<Value, RW *> &subset,
                       const BinaryCompFunc &callback, unsigned latency = 0);

  void reset() override;

  void exec(bool isClkRisingEdge) override;

  void printStates() override;

private:
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
  Simulator(handshake::FuncOp funcOp, unsigned cyclesLimit = 100);

  void reset();

  void simulate(llvm::ArrayRef<std::string> inputArgs);

  // Just a temporary function to print the results of the simulation to
  // standart output
  void printResults();

  // A temporary function
  void printModelStates();

  ~Simulator();

private:
  // Maybe need it some day, let it be the part of simulator class
  handshake::FuncOp funcOp;
  // Results of the simulation
  bool resValid = false, resReady = true;
  Data resData;
  // End operation to extract results
  handshake::EndOp endOp;
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
  void registerModel(Op op, Args &&...modelArgs);

  // Determine the concrete Model type
  void associateModel(Operation *op);

  // Register the State state where needed (oldValueStates, newValueStates,
  // updaters, rws).
  template <typename State, typename Updater, typename Producer,
            typename Consumer, typename Ty>
  void registerState(Value val, Operation *producerOp, Ty type);

  // Determine if the state belongs to channel or control
  void associateState(Value val, Operation *producerOp, Location loc);
};

} // namespace experimental
} // namespace dynamatic

#endif