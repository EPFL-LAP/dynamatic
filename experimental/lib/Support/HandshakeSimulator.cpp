#include "experimental/Support/HandshakeSimulator.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
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
#include "llvm/Support/ErrorHandling.h"
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

using namespace dynamatic::experimental;

Data::Data(const APInt &value) {
  data = value;
  hash = llvm::hash_value(value);
  bitwidth = value.getBitWidth();
}

Data::Data(const APFloat &value) {
  data = value;
  hash = llvm::hash_value(value);
  bitwidth = value.getSizeInBits(value.getSemantics());
}

Data &Data::operator=(const APInt &value) {
  this->data = value;
  hash = llvm::hash_value(value);
  bitwidth = value.getBitWidth();
  return *this;
}

Data &Data::operator=(const APFloat &value) {
  this->data = value;
  hash = llvm::hash_value(value);
  Data::bitwidth = value.getSizeInBits(value.getSemantics());
  return *this;
}

bool Data::hasValue() const { return data.has_value(); }

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

ValueState::ValueState(Value val) : val(val) {}

template <typename Ty>
TypedValueState<Ty>::TypedValueState(TypedValue<Ty> val) : ValueState(val) {}

ChannelState::ChannelState(TypedValue<handshake::ChannelType> channel)
    : TypedValueState<handshake::ChannelType>(channel) {

  llvm::TypeSwitch<mlir::Type>(channel.getType().getDataType())
      .Case<IntegerType>([&](IntegerType intType) {
        data =
            APInt(std::max(1, (int)intType.getWidth()), 0, intType.isSigned());
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

ControlState::ControlState(TypedValue<handshake::ControlType> control)
    : TypedValueState<handshake::ControlType>(control) {}

template <typename State>
DoubleUpdater<State>::DoubleUpdater(State &oldState, State &newState)
    : oldState(oldState), newState(newState) {}

ChannelUpdater::ChannelUpdater(ChannelState &oldState, ChannelState &newState)
    : DoubleUpdater<ChannelState>(oldState, newState) {}

bool ChannelUpdater::check() {
  return newState.valid == oldState.valid && newState.ready == oldState.ready &&
         newState.data.hash == oldState.data.hash;
}

void ChannelUpdater::setValid() {
  newState.valid = true;
  update();
}

void ChannelUpdater::resetValid() {
  if (newState.ready) {
    newState.valid = false;
  }
  update();
}

void ChannelUpdater::update() {
  oldState.valid = newState.valid;
  oldState.ready = newState.ready;
  oldState.data = newState.data;
}

ControlUpdater::ControlUpdater(ControlState &oldState, ControlState &newState)
    : DoubleUpdater<ControlState>(oldState, newState) {}

bool ControlUpdater::check() {
  return newState.valid == oldState.valid && newState.ready == oldState.ready;
}

void ControlUpdater::setValid() {
  newState.valid = true;
  update();
}

void ControlUpdater::resetValid() {
  if (newState.ready)
    newState.valid = false;
  update();
}

void ControlUpdater::update() {
  oldState.valid = newState.valid;
  oldState.ready = newState.ready;
}

ProducerRW::ProducerDescendants ProducerRW::getType() const { return prod; }

ProducerRW::ProducerRW(bool &valid, const bool &ready, ProducerDescendants p)
    : valid(valid), ready(ready), prod(p) {}

ProducerRW::ProducerRW(ProducerRW &p)
    : valid(p.valid), ready(p.ready), prod(p.prod) {}

ConsumerRW::ConsumerDescendants ConsumerRW::getType() const { return cons; }

ConsumerRW::ConsumerRW(const bool &valid, bool &ready, ConsumerDescendants c)
    : valid(valid), ready(ready), cons(c) {}

ConsumerRW::ConsumerRW(ConsumerRW &c)
    : valid(c.valid), ready(c.ready), cons(c.cons) {}

ControlConsumerRW::ControlConsumerRW(ControlState &reader, ControlState &writer)
    : ConsumerRW(reader.valid, writer.ready, D_ControlConsumerRW) {}

bool ControlConsumerRW::classof(const ConsumerRW *c) {
  return c->getType() == D_ControlConsumerRW;
}

ControlProducerRW::ControlProducerRW(ControlState &reader, ControlState &writer)
    : ProducerRW(writer.valid, reader.ready, D_ControlProducerRW) {}

bool ControlProducerRW::classof(const ProducerRW *c) {
  return c->getType() == D_ControlProducerRW;
}

ChannelConsumerRW::ChannelConsumerRW(ChannelState &reader, ChannelState &writer)
    : ConsumerRW(reader.valid, writer.ready, D_ChannelConsumerRW),
      data(reader.data) {}

bool ChannelConsumerRW::classof(const ConsumerRW *c) {
  return c->getType() == D_ChannelConsumerRW;
}

ChannelProducerRW::ChannelProducerRW(ChannelState &reader, ChannelState &writer)
    : ProducerRW(writer.valid, reader.ready, D_ChannelProducerRW),
      data(writer.data) {}

bool ChannelProducerRW::classof(const ProducerRW *c) {
  return c->getType() == D_ChannelProducerRW;
}

ConsumerData::ConsumerData(ConsumerRW *ins) {
  if (auto *p = dyn_cast<ChannelConsumerRW>(ins)) {
    data = &p->data;
    dataWidth = data->bitwidth;
  } else
    data = nullptr;
}

bool ConsumerData::hasValue() const { return data != nullptr; }

ProducerData::ProducerData(ProducerRW *outs) {
  if (auto *p = dyn_cast<ChannelProducerRW>(outs)) {
    data = &p->data;
  } else
    data = nullptr;
}

ProducerData &ProducerData::operator=(const ConsumerData &value) {
  if (data)
    *data = *value.data;
  return *this;
}

bool ProducerData::hasValue() const { return data != nullptr; }

ExecutionModel::ExecutionModel(Operation *op) : op(op) {}

template <typename Op>
OpExecutionModel<Op>::OpExecutionModel(Op op)
    : ExecutionModel(op.getOperation()) {}

template <typename Op>
Op OpExecutionModel<Op>::getOperation() {
  return cast<Op>(op);
}

template <typename Op>
template <typename State>
State *OpExecutionModel<Op>::getState(Value val,
                                      mlir::DenseMap<Value, RW *> &rws) {
  return static_cast<State *>(rws[val]);
}

template <typename Op>
template <typename State, typename Data>
void OpExecutionModel<Op>::printValue(const std::string &name, State *ins,
                                      Data *insData) {
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

void Antotokens::reset(const bool &pvalid1, const bool &pvalid0, bool &kill1,
                       bool &kill0, const bool &generateAt1,
                       const bool &generateAt0, bool &stopValid) {
  regOut0 = false;
  regOut1 = false;

  regIn0 = !pvalid0 && (generateAt0 || regOut0);
  regIn1 = !pvalid1 && (generateAt1 || regOut1);

  stopValid = regOut0 || regOut1;

  kill0 = generateAt0 || regOut0;
  kill1 = generateAt1 || regOut1;
}

void Antotokens::exec(bool isClkRisingEdge, const bool &pvalid1,
                      const bool &pvalid0, bool &kill1, bool &kill0,
                      const bool &generateAt1, const bool &generateAt0,
                      bool &stopValid) {
  if (isClkRisingEdge) {
    regOut0 = regIn0;
    regOut1 = regIn1;
  }
  regIn0 = !pvalid0 && (generateAt0 || regOut0);
  regIn1 = !pvalid1 && (generateAt1 || regOut1);

  stopValid = regOut0 || regOut1;

  kill0 = generateAt0 || regOut0;
  kill1 = generateAt1 || regOut1;
}

ForkSupport::ForkSupport(unsigned size, unsigned datawidth)
    : size(size), datawidth(datawidth) {
  transmitValue.resize(size, false);
  keepValue.resize(size, false);
  blockStopArray.resize(size, false);
}

void ForkSupport::resetDataless(ConsumerRW *ins,
                                std::vector<ProducerRW *> &outs) {
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

void ForkSupport::execDataless(bool isClkRisingEdge, ConsumerRW *ins,
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

void ForkSupport::reset(ConsumerRW *ins, std::vector<ProducerRW *> &outs,
                        const ConsumerData &insData,
                        std::vector<ProducerData> &outsData) {
  resetDataless(ins, outs);
  for (auto &out : outsData)
    out = insData;
}

void ForkSupport::exec(bool isClkRisingEdge, ConsumerRW *ins,
                       std::vector<ProducerRW *> &outs,
                       const ConsumerData &insData,
                       std::vector<ProducerData> &outsData) {
  execDataless(isClkRisingEdge, ins, outs);
  for (auto &out : outsData)
    out = insData;
}

JoinSupport::JoinSupport(unsigned size) : size(size) {}

void JoinSupport::exec(std::vector<ConsumerRW *> &ins, ProducerRW *outs) {
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

OEHBSupport::OEHBSupport(unsigned datawidth) : datawidth(datawidth) {}

void OEHBSupport::resetDataless(ConsumerRW *ins, ProducerRW *outs) {
  outputValid = false;
  ins->ready = true;
  outs->valid = false;
}

void OEHBSupport::execDataless(bool isClkRisingEdge, ConsumerRW *ins,
                               ProducerRW *outs) {
  if (isClkRisingEdge)
    outputValid = ins->valid || (outputValid && !outs->ready);

  ins->ready = !outputValid || outs->ready;
  outs->valid = outputValid;
}

void OEHBSupport::reset(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
                        Data *outsData) {
  resetDataless(ins, outs);
  if (insData)
    *outsData = APInt(datawidth, 0);
  regEn = ins->ready && ins->valid;
}

void OEHBSupport::exec(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
                       const Data *insData, Data *outsData) {
  execDataless(isClkRisingEdge, ins, outs);
  if (isClkRisingEdge && regEn && insData)
    *outsData = *insData;
  regEn = ins->ready && ins->valid;
}

TEHBSupport::TEHBSupport(unsigned datawidth) : datawidth(datawidth) {}

void TEHBSupport::resetDataless(ConsumerRW *ins, ProducerRW *outs) {
  outputValid = ins->valid;
  fullReg = false;
  ins->ready = true;
  outs->valid = ins->valid;
}

void TEHBSupport::execDataless(bool isClkRisingEdge, ConsumerRW *ins,
                               ProducerRW *outs) {
  if (isClkRisingEdge)
    fullReg = outputValid && !outs->ready;
  outputValid = ins->valid || fullReg;
  ins->ready = !fullReg;
  outs->valid = outputValid;
}

void TEHBSupport::reset(ConsumerRW *ins, ProducerRW *outs, const Data *insData,
                        Data *outsData) {
  if (insData)
    resetDataFull(ins, outs, insData, outsData);
  else
    resetDataless(ins, outs);
}

void TEHBSupport::exec(bool isClkRisingEdge, ConsumerRW *ins, ProducerRW *outs,
                       const Data *insData, Data *outsData) {
  if (insData)
    execDataFull(isClkRisingEdge, ins, outs, insData, outsData);
  else
    execDataless(isClkRisingEdge, ins, outs);
}

void TEHBSupport::resetDataFull(ConsumerRW *ins, ProducerRW *outs,
                                const Data *insData, Data *outsData) {
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

void TEHBSupport::execDataFull(bool isClkRisingEdge, ConsumerRW *ins,
                               ProducerRW *outs, const Data *insData,
                               Data *outsData) {
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

BranchModel::BranchModel(handshake::BranchOp branchOp,
                         mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::BranchOp>(branchOp),
      // get the exact structure for the particular value
      ins(getState<ConsumerRW>(branchOp.getOperand(), subset)),
      outs(getState<ProducerRW>(branchOp.getResult(), subset)),
      // initialize data (nullptr if dataless)
      insData(ins), outsData(outs) {}

void BranchModel::reset() {
  outs->valid = ins->valid;
  ins->ready = outs->ready;
  outsData = insData;
};

void BranchModel::exec(bool isClkRisingEdge) { reset(); }

void BranchModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
}

CondBranchModel::CondBranchModel(handshake::ConditionalBranchOp condBranchOp,
                                 mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::ConditionalBranchOp>(condBranchOp),
      data(getState<ConsumerRW>(condBranchOp.getDataOperand(), subset)),
      condition(getState<ChannelConsumerRW>(condBranchOp.getConditionOperand(),
                                            subset)),
      trueOut(getState<ProducerRW>(condBranchOp.getTrueResult(), subset)),
      falseOut(getState<ProducerRW>(condBranchOp.getFalseResult(), subset)),
      dataData(data), trueOutData(trueOut), falseOutData(falseOut),
      condBrJoin(2) {}

void CondBranchModel::reset() {
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

void CondBranchModel::exec(bool isClkRisingEdge) { reset(); }

void CondBranchModel::printStates() {
  printValue<ConsumerRW, const Data>("data", data, dataData.data);
  printValue<ChannelConsumerRW, const Data>("condition", condition,
                                            &condition->data);
  printValue<ProducerRW, Data>("trueOut", trueOut, trueOutData.data);
  printValue<ProducerRW, Data>("falseOut", falseOut, falseOutData.data);
}

ConstantModel::ConstantModel(handshake::ConstantOp constOp,
                             mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::ConstantOp>(constOp),
      value(dyn_cast<mlir::IntegerAttr>(constOp.getValue()).getValue()),
      ctrl(getState<ControlConsumerRW>(constOp.getCtrl(), subset)),
      outs(getState<ChannelProducerRW>(constOp.getResult(), subset)) {}

void ConstantModel::reset() {
  outs->data = value;
  outs->valid = ctrl->valid;
  ctrl->ready = outs->ready;
}

void ConstantModel::exec(bool isClkRisingEdge) { reset(); }

void ConstantModel::printStates() {
  llvm::outs() << "ctrl: " << ctrl->valid << " " << ctrl->ready << "\n";
  printValue<ChannelProducerRW, Data>("outs", outs, &outs->data);
}

ControlMergeModel::ControlMergeModel(handshake::ControlMergeOp cMergeOp,
                                     mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::ControlMergeOp>(cMergeOp),
      size(op->getNumOperands()),
      indexWidth(cMergeOp.getIndex().getType().getDataBitWidth()),
      cMergeTEHB(indexWidth), cMergeFork(2),
      outs(getState<ProducerRW>(cMergeOp.getResult(), subset)),
      index(getState<ChannelProducerRW>(cMergeOp.getIndex(), subset)),
      outsData(outs), insTEHB(dataAvailable, tehbOutReady),
      outsTEHB(tehbOutValid, readyToFork), insFork(tehbOutValid, readyToFork) {
  for (auto oper : cMergeOp->getOperands())
    ins.push_back(getState<ConsumerRW>(oper, subset));

  outsFork = {outs, index};

  for (unsigned i = 0; i < size; ++i)
    insData.emplace_back(ins[i]);

  cMergeFork.datawidth = outsData.dataWidth;
}

void ControlMergeModel::reset() {
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

void ControlMergeModel::exec(bool isClkRisingEdge) {
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

void ControlMergeModel::printStates() {
  for (unsigned i = 0; i < size; ++i)
    printValue<ConsumerRW, const Data>("ins", ins[i], insData[i].data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
  printValue<ChannelProducerRW, Data>("index", index, &index->data);
}

ForkModel::ForkModel(handshake::ForkOp forkOp,
                     mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::ForkOp>(forkOp), size(op->getNumResults()),
      forkSupport(size), ins(getState<ConsumerRW>(forkOp.getOperand(), subset)),
      insData(ins) {
  for (unsigned i = 0; i < size; ++i)
    outs.push_back(getState<ProducerRW>(forkOp->getResult(i), subset));

  for (unsigned i = 0; i < size; ++i)
    outsData.emplace_back(outs[i]);

  forkSupport.datawidth = insData.dataWidth;
}

void ForkModel::reset() { forkSupport.reset(ins, outs, insData, outsData); }

void ForkModel::exec(bool isClkRisingEdge) {
  forkSupport.exec(isClkRisingEdge, ins, outs, insData, outsData);
}

void ForkModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
  for (unsigned i = 0; i < size; ++i)
    printValue<ProducerRW, Data>("outs", outs[i], outsData[i].data);
}

JoinModel::JoinModel(handshake::JoinOp joinOp,
                     mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::JoinOp>(joinOp),
      outs(getState<ProducerRW>(joinOp.getResult(), subset)),
      join(joinOp->getNumOperands()) {
  for (auto oper : joinOp->getOperands())
    ins.push_back(getState<ConsumerRW>(oper, subset));
}

void JoinModel::reset() { join.exec(ins, outs); }

void JoinModel::exec(bool isClkRisingEdge) { reset(); }

void JoinModel::printStates() {
  for (auto *in : ins)
    llvm::outs() << "Ins: " << in->valid << " " << in->ready << "\n";
  llvm::outs() << "Outs: " << outs->valid << " " << outs->ready << "\n";
}

LazyForkModel::LazyForkModel(handshake::LazyForkOp lazyForkOp,
                             mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::LazyForkOp>(lazyForkOp),
      size(lazyForkOp->getNumResults()),
      ins(getState<ConsumerRW>(lazyForkOp.getOperand(), subset)), insData(ins) {

  for (unsigned i = 0; i < size; ++i)
    outs.push_back(getState<ProducerRW>(lazyForkOp->getResult(i), subset));

  for (unsigned i = 0; i < size; ++i)
    outsData.emplace_back(outs[i]);
}

void LazyForkModel::reset() {
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

void LazyForkModel::exec(bool isClkRisingEdge) { reset(); }

void LazyForkModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
  for (unsigned i = 0; i < size; ++i)
    printValue<ProducerRW, Data>("outs", outs[i], outsData[i].data);
}

MergeModel::MergeModel(handshake::MergeOp mergeOp,
                       mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::MergeOp>(mergeOp),
      size(mergeOp->getNumOperands()),
      outs(getState<ProducerRW>(mergeOp.getResult(), subset)), outsData(outs),
      insTEHB(tehbValid, tehbReady) {
  for (unsigned i = 0; i < size; ++i)
    ins.push_back(getState<ConsumerRW>(mergeOp->getOperand(i), subset));
  for (unsigned i = 0; i < size; ++i)
    insData.emplace_back(ins[i]);

  mergeTEHB.datawidth = outsData.dataWidth;
}

void MergeModel::reset() {
  if (outsData.hasValue()) {
    execDataFull();
    mergeTEHB.reset(&insTEHB, outs, &tehbDataIn, outsData.data);
  } else {
    execDataless();
    mergeTEHB.resetDataless(&insTEHB, outs);
  }
}

void MergeModel::exec(bool isClkRisingEdge) {
  if (outsData.hasValue()) {
    execDataFull();
    mergeTEHB.exec(isClkRisingEdge, &insTEHB, outs, &tehbDataIn, outsData.data);
  } else {
    execDataless();
    mergeTEHB.execDataless(isClkRisingEdge, &insTEHB, outs);
  }
}

void MergeModel::printStates() {
  for (unsigned i = 0; i < size; ++i)
    printValue<ConsumerRW, const Data>("ins", ins[i], insData[i].data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
}

void MergeModel::execDataless() {
  tehbValid = false;
  for (unsigned i = 0; i < size; ++i)
    tehbValid = tehbValid || ins[i]->valid;

  for (unsigned i = 0; i < size; ++i)
    ins[i]->ready = tehbReady;
}

void MergeModel::execDataFull() {
  tehbValid = false;
  tehbDataIn = *insData[0].data;
  for (unsigned i = 0; i < size; ++i)
    if (ins[i]->valid) {
      tehbValid = true;
      tehbDataIn = *insData[i].data;
      break;
    }

  for (unsigned i = 0; i < size; ++i)
    ins[i]->ready = tehbReady;
}

MuxModel::MuxModel(handshake::MuxOp muxOp, mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::MuxOp>(muxOp),
      size(muxOp.getDataOperands().size()),
      selectWidth(muxOp.getSelectOperand().getType().getDataBitWidth()),
      index(getState<ChannelConsumerRW>(muxOp.getSelectOperand(), subset)),
      outs(getState<ProducerRW>(muxOp.getResult(), subset)), outsData(outs),
      insTEHB(tehbValid, tehbReady) {
  for (auto oper : muxOp.getDataOperands())
    ins.push_back(getState<ChannelConsumerRW>(oper, subset));
  for (unsigned i = 0; i < size; ++i)
    insData.emplace_back(ins[i]);

  muxTEHB.datawidth = outsData.dataWidth;
}

void MuxModel::reset() {
  indexNum = dataCast<APInt>(index->data).getZExtValue();
  if (outsData.hasValue()) {
    execDataFull();
    muxTEHB.reset(&insTEHB, outs, &tehbDataIn, outsData.data);
  } else {
    execDataless();
    muxTEHB.resetDataless(&insTEHB, outs);
  }
}

void MuxModel::exec(bool isClkRisingEdge) {
  indexNum = dataCast<APInt>(index->data).getZExtValue();
  if (outsData.hasValue()) {
    execDataFull();
    muxTEHB.exec(isClkRisingEdge, &insTEHB, outs, &tehbDataIn, outsData.data);
  } else {
    execDataless();
    muxTEHB.execDataless(isClkRisingEdge, &insTEHB, outs);
  }
}

void MuxModel::printStates() {
  for (unsigned i = 0; i < size; ++i)
    printValue<ConsumerRW, const Data>("ins", ins[i], insData[i].data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
  printValue<ChannelConsumerRW, const Data>("index", index, &index->data);
}

void MuxModel::execDataless() {
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
}

void MuxModel::execDataFull() {
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
}

OEHBModel::OEHBModel(handshake::BufferOp oehbOp,
                     mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::BufferOp>(oehbOp),
      ins(getState<ConsumerRW>(oehbOp.getOperand(), subset)),
      outs(getState<ProducerRW>(oehbOp.getResult(), subset)), insData(ins),
      outsData(outs), oehbDl(insData.dataWidth) {}

void OEHBModel::reset() {
  oehbDl.reset(ins, outs, insData.data, outsData.data);
}

void OEHBModel::exec(bool isClkRisingEdge) {
  oehbDl.exec(isClkRisingEdge, ins, outs, insData.data, outsData.data);
}

void OEHBModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
}

SinkModel::SinkModel(handshake::SinkOp sinkOp,
                     mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::SinkOp>(sinkOp),
      ins(getState<ConsumerRW>(sinkOp.getOperand(), subset)), insData(ins) {}

void SinkModel::reset() { ins->ready = true; }

void SinkModel::exec(bool isClkRisingEdge) { reset(); }

void SinkModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
}

SourceModel::SourceModel(handshake::SourceOp sourceOp,
                         mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::SourceOp>(sourceOp),
      outs(getState<ProducerRW>(sourceOp.getResult(), subset)) {}

void SourceModel::reset() { outs->valid = true; }

void SourceModel::exec(bool isClkRisingEdge) { reset(); }

void SourceModel::printStates() {
  llvm::outs() << "Outs: " << outs->valid << " " << outs->ready << "\n";
}

TEHBModel::TEHBModel(handshake::BufferOp tehbOp,
                     mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::BufferOp>(tehbOp),
      ins(getState<ConsumerRW>(tehbOp.getOperand(), subset)),
      outs(getState<ProducerRW>(tehbOp.getResult(), subset)), insData(ins),
      outsData(outs), returnTEHB(insData.dataWidth) {}

void TEHBModel::reset() {
  returnTEHB.reset(ins, outs, insData.data, outsData.data);
}

void TEHBModel::exec(bool isClkRisingEdge) {
  returnTEHB.exec(isClkRisingEdge, ins, outs, insData.data, outsData.data);
}

void TEHBModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, insData.data);
  printValue<ProducerRW, Data>("outs", outs, outsData.data);
}

EndModel::EndModel(handshake::EndOp endOp, mlir::DenseMap<Value, RW *> &subset,
                   bool &resValid, const bool &resReady, Data &resData)
    : OpExecutionModel<handshake::EndOp>(endOp),
      ins(getState<ChannelConsumerRW>(endOp.getInputs().front(), subset)),
      outs(resValid, resReady), outsData(resData) {}

void EndModel::reset() {
  outs.valid = ins->valid;
  ins->ready = ins->valid && outs.ready;
  outsData = ins->data;
}

void EndModel::exec(bool isClkRisingEdge) { reset(); }

void EndModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, &ins->data);
  printValue<ProducerRW, Data>("outs", &outs, &outsData);
}

//===----------------------------------------------------------------------===//
// Arithmetic
//===----------------------------------------------------------------------===//
/// Arithmetic and generic components

TruncIModel::TruncIModel(handshake::TruncIOp trunciOp,
                         mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::TruncIOp>(trunciOp),
      outputWidth(trunciOp.getResult().getType().getDataBitWidth()),
      ins(getState<ChannelConsumerRW>(trunciOp.getOperand(), subset)),
      outs(getState<ChannelProducerRW>(trunciOp.getResult(), subset)) {}

void TruncIModel::reset() {
  outs->data = dataCast<const APInt>(ins->data).trunc(outputWidth);
  outs->valid = ins->valid;
  ins->ready = !ins->valid || (ins->valid && outs->ready);
}

void TruncIModel::exec(bool isClkRisingEdge) { reset(); }

void TruncIModel::printStates() {
  printValue<ConsumerRW, const Data>("ins", ins, &ins->data);
  printValue<ProducerRW, Data>("outs", outs, &outs->data);
}

SelectModel::SelectModel(handshake::SelectOp selectOp,
                         mlir::DenseMap<Value, RW *> &subset)
    : OpExecutionModel<handshake::SelectOp>(selectOp),
      condition(getState<ChannelConsumerRW>(selectOp.getCondition(), subset)),
      trueValue(getState<ChannelConsumerRW>(selectOp.getTrueValue(), subset)),
      falseValue(getState<ChannelConsumerRW>(selectOp.getFalseValue(), subset)),
      result(getState<ChannelProducerRW>(selectOp.getResult(), subset)) {}

void SelectModel::reset() {
  selectExec();
  anti.reset(falseValue->valid, trueValue->valid, kill1, kill0, g1, g0,
             antitokenStop);
}

void SelectModel::exec(bool isClkRisingEdge) {
  selectExec();
  anti.exec(isClkRisingEdge, falseValue->valid, trueValue->valid, kill1, kill0,
            g1, g0, antitokenStop);
}

void SelectModel::printStates() {
  printValue<ConsumerRW, const Data>("condition", condition, &condition->data);
  printValue<ConsumerRW, const Data>("trueValue", trueValue, &trueValue->data);
  printValue<ConsumerRW, const Data>("falseValue", falseValue,
                                     &falseValue->data);
  printValue<ProducerRW, Data>("result", result, &result->data);
}

void SelectModel::selectExec() {
  auto cond = dataCast<APInt>(condition->data).getBoolValue();

  ee = condition->valid &&
       ((!cond && falseValue->valid) || (cond && trueValue->valid));
  validInternal = ee && !antitokenStop;
  g0 = !trueValue->valid && validInternal && result->ready;
  g1 = !falseValue->valid && validInternal && result->ready;

  result->valid = validInternal;
  trueValue->ready =
      !trueValue->valid || (validInternal && result->ready) || kill0;
  falseValue->ready =
      !falseValue->valid || (validInternal && result->ready) || kill1;
  condition->ready = !condition->valid || (validInternal && result->ready);

  if (cond)
    result->data = trueValue->data;
  else
    result->data = falseValue->data;
}

template <typename Op>
GenericUnaryOpModel<Op>::GenericUnaryOpModel(
    Op op, mlir::DenseMap<Value, RW *> &subset, const UnaryCompFunc &callback)
    : OpExecutionModel<Op>(op),
      outputWidth(op.getResult().getType().getDataBitWidth()),
      callback(callback),
      ins(OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
          op.getOperand(), subset)),
      outs(OpExecutionModel<Op>::template getState<ChannelProducerRW>(
          op.getResult(), subset)) {}

template <typename Op>
void GenericUnaryOpModel<Op>::reset() {
  outs->data = callback(ins->data, outputWidth);
  outs->valid = ins->valid;
  ins->ready = outs->ready;
}

template <typename Op>
void GenericUnaryOpModel<Op>::exec(bool isClkRisingEdge) {
  reset();
}

template <typename Op>
void GenericUnaryOpModel<Op>::printStates() {
  OpExecutionModel<Op>::template printValue<ConsumerRW, const Data>("ins", ins,
                                                                    &ins->data);
  OpExecutionModel<Op>::template printValue<ProducerRW, Data>("outs", outs,
                                                              &outs->data);
}

template <typename Op>
::GenericBinaryOpModel<Op>::GenericBinaryOpModel(
    Op op, mlir::DenseMap<Value, RW *> &subset, const BinaryCompFunc &callback,
    unsigned latency)
    : OpExecutionModel<Op>(op), callback(callback), latency(latency),
      bitwidth(cast<handshake::ChannelType>(op.getResult().getType())
                   .getDataBitWidth()),
      lhs(OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
          op.getLhs(), subset)),
      rhs(OpExecutionModel<Op>::template getState<ChannelConsumerRW>(
          op.getRhs(), subset)),
      result(OpExecutionModel<Op>::template getState<ChannelProducerRW>(
          op.getResult(), subset)),
      insJoin({lhs, rhs}), outsJoin(joinValid, result->ready), binJoin(2) {}

template <typename Op>
void ::GenericBinaryOpModel<Op>::reset() {
  binJoin.exec(insJoin, result);
  result->data = callback(lhs->data, rhs->data);
}

template <typename Op>
void ::GenericBinaryOpModel<Op>::exec(bool isClkRisingEdge) {
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

template <typename Op>
void ::GenericBinaryOpModel<Op>::printStates() {
  OpExecutionModel<Op>::template printValue<ConsumerRW, const Data>("lhs", lhs,
                                                                    &lhs->data);
  OpExecutionModel<Op>::template printValue<ConsumerRW, const Data>("rhs", rhs,
                                                                    &rhs->data);
  OpExecutionModel<Op>::template printValue<ProducerRW, Data>("result", result,
                                                              &result->data);
}

//===----------------------------------------------------------------------===//
// Simulator
//===----------------------------------------------------------------------===//

Simulator::Simulator(handshake::FuncOp funcOp, unsigned cyclesLimit)
    : funcOp(funcOp), cyclesLimit(cyclesLimit) {
  // Iterate through all the values of the circuit
  for (BlockArgument arg : funcOp.getArguments())
    associateState(arg, funcOp, funcOp->getLoc());

  for (Operation &op : funcOp.getOps())
    for (auto res : op.getResults())
      associateState(res, &op, op.getLoc());

  // register models for all ops of the funcOp
  for (Operation &op : funcOp.getOps())
    associateModel(&op);
  endOp = *funcOp.getOps<handshake::EndOp>().begin();

  reset();
}

void Simulator::reset() {
  // Loop until the models' states stop changing
  while (true) {
    // Reset all models
    for (auto &[op, model] : opModels)
      model->reset();
    // Check if states have changed
    bool isFin = true;
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

void Simulator::simulate(llvm::ArrayRef<std::string> inputArgs) {
  /// First, process simulator's current inputs. The user is only expected to
  /// enter data values.

  // Counter of channels to check if it corresponds the obtained ones
  unsigned channelCount = 0;
  // The vector representing values that match channels
  llvm::SmallVector<Value> channelArgs;
  for (BlockArgument arg : funcOp.getArguments())
    if (isa<handshake::ChannelType>(arg.getType())) {
      ++channelCount;
      channelArgs.push_back(arg);
    }

  // Check if the number of data values is divisible by the number of
  // ChannelStates
  if (channelCount == 0 || inputArgs.size() % channelCount != 0) {
    llvm::errs() << "The amount of input data values must be divisible by "
                    "the number of the input channels!";
    exit(1);
  }

  // Iterate through all the collected channel values...
  size_t inputCount = 0;
  for (auto val : channelArgs) {
    auto channel = cast<TypedValue<handshake::ChannelType>>(val);
    auto *channelArg = static_cast<ChannelProducerRW *>(producerViews[val]);
    // ...and update the corresponding data fields
    llvm::TypeSwitch<mlir::Type>(channel.getType().getDataType())
        .Case<IntegerType>([&](IntegerType intType) {
          auto argNumVal = APInt(intType.getWidth(), inputArgs[inputCount], 10);
          channelArg->data = argNumVal;
        })
        .Case<FloatType>([&](FloatType floatType) {
          auto argNumVal =
              APFloat(floatType.getFloatSemantics(), inputArgs[inputCount]);
          channelArg->data = argNumVal;
        })
        .Default([&](auto) {
          emitError(channel.getLoc())
              << "Unsuported date type " << channel.getType()
              << ", we111 should probably report an error and stop";
        });
    ++inputCount;
    updaters[val]->update();
  }

  // Set all inputs' valid to true
  for (BlockArgument arg : funcOp.getArguments())
    updaters[arg]->setValid();

  /// Second, locate the results

  // Pointer to the struct containing results

  auto *res =
      static_cast<ChannelConsumerRW *>(consumerViews[&endOp->getOpOperand(0)]);

  /// Third, iterate with a clock

  // The variable that counts the number of consecutive iterations with the
  // same models' states
  unsigned eqStateCount = 0;
  // The total number of clk cycles
  iterNum = 0;

  // The outer loop: for clock cycles
  // It stops when simulator's output is valid or the limit of cycles is
  // reached
  while (true) {
    ++iterNum;
    // True only once on each clkRisingEdge
    bool isClock = true;
    // The inner loop: for signal propagation within one clkRisingEdge
    // It stops when there's no more change in valueStates
    while (true) {
      // Execute each model
      for (auto &[op, model] : opModels)
        model->exec(isClock);
      // Check if states have changed
      bool isFin = true;
      for (auto [val, state] : updaters)
        isFin = isFin && state->check();
      if (isFin)
        break;
      // Update oldStates
      for (auto [val, state] : updaters)
        state->update();
      isClock = false;
    }

    // If the simulator's result is valid, the simulation can be finished
    if (res->valid) {
      for (auto [val, state] : updaters)
        state->resetValid();
      break;
    }

    // Check if states have changed
    bool isFin = true;
    for (auto [val, state] : updaters)
      isFin = isFin && state->check();
    // Stop iterating if the limit of cycles is reached. In general this means
    // a deadlock
    if (isFin) {
      ++eqStateCount;
      if (eqStateCount >= cyclesLimit)
        break;
    } else
      eqStateCount = 0;

    // At the end of each cycle reset each input's valid signal to false if
    // the corresponding ready was true (The arguments have already been read)
    for (BlockArgument arg : funcOp.getArguments())
      updaters[arg]->resetValid();
  }
}

// Just a temporary function to print the results of the simulation to
// standart output
void Simulator::printResults() {
  llvm::outs() << "Results\n";
  llvm::outs() << resValid << " " << resReady << " ";
  SmallVector<char> outData;

  auto channel = cast<TypedValue<handshake::ChannelType>>(endOp->getOperand(0));
  llvm::TypeSwitch<mlir::Type>(channel.getType().getDataType())
      .Case<IntegerType>([&](IntegerType intType) {
        llvm::outs() << dataCast<APInt>(resData) << "\n";
      })
      .Case<FloatType>([&](FloatType floatType) {
        dataCast<APFloat>(resData).toString(outData);
        llvm::outs() << outData << "\n";
      })
      .Default([&](auto) {
        emitError(channel.getLoc())
            << "Unsuported date type " << channel.getType()
            << ", we111 should probably report an error and stop";
      });

  llvm::outs() << "Number of iterations: " << iterNum << "\n";
}

// A temporary function
void Simulator::printModelStates() {
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

Simulator::~Simulator() {
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

template <typename Model, typename Op, typename... Args>
void Simulator::registerModel(Op op, Args &&...modelArgs) {
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
void Simulator::associateModel(Operation *op) {
  llvm::TypeSwitch<Operation *>(op)
      // handshake
      .Case<handshake::BranchOp>([&](handshake::BranchOp branchOp) {
        registerModel<BranchModel, handshake::BranchOp>(branchOp);
      })
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp cBranchOp) {
            registerModel<CondBranchModel, handshake::ConditionalBranchOp>(
                cBranchOp);
          })
      .Case<handshake::ConstantOp>([&](handshake::ConstantOp constOp) {
        registerModel<ConstantModel, handshake::ConstantOp>(constOp);
      })
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp cMergeOp) {
        registerModel<ControlMergeModel, handshake::ControlMergeOp>(cMergeOp);
      })
      .Case<handshake::ForkOp>([&](handshake::ForkOp forkOp) {
        registerModel<ForkModel, handshake::ForkOp>(forkOp);
      })
      .Case<handshake::JoinOp>([&](handshake::JoinOp joinOp) {
        registerModel<JoinModel, handshake::JoinOp>(joinOp);
      })
      .Case<handshake::LazyForkOp>([&](handshake::LazyForkOp lForkOp) {
        registerModel<LazyForkModel, handshake::LazyForkOp>(lForkOp);
      })
      .Case<handshake::MergeOp>([&](handshake::MergeOp mergeOp) {
        registerModel<MergeModel, handshake::MergeOp>(mergeOp);
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        registerModel<MuxModel, handshake::MuxOp>(muxOp);
      })
      .Case<handshake::NotOp>([&](handshake::NotOp notOp) {
        UnaryCompFunc callback = [](const Data &lhs, unsigned outWidth) {
          auto temp = dataCast<APInt>(lhs);
          return ~temp;
        };
        registerModel<GenericUnaryOpModel<handshake::NotOp>>(notOp, callback);
      })
      .Case<handshake::BufferOp>([&](handshake::BufferOp bufferOp) {
        auto params =
            bufferOp->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
        auto optTiming = params.getNamed(handshake::BufferOp::TIMING_ATTR_NAME);
        if (auto timing =
                dyn_cast<handshake::TimingAttr>(optTiming->getValue())) {
          auto info = timing.getInfo();
          if (info == handshake::TimingInfo::break_dv())
            registerModel<OEHBModel, handshake::BufferOp>(bufferOp);
          if (info == handshake::TimingInfo::break_r())
            registerModel<TEHBModel, handshake::BufferOp>(bufferOp);
        }
      })
      .Case<handshake::SinkOp>([&](handshake::SinkOp sinkOp) {
        registerModel<SinkModel, handshake::SinkOp>(sinkOp);
      })
      .Case<handshake::SourceOp>([&](handshake::SourceOp sourceOp) {
        registerModel<SourceModel, handshake::SourceOp>(sourceOp);
      })
      // arithmetic
      .Case<handshake::AddFOp>([&](handshake::AddFOp addfOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          APFloat res = dataCast<APFloat>(lhs) + dataCast<APFloat>(rhs);
          return res;
        };
        registerModel<GenericBinaryOpModel<handshake::AddFOp>>(addfOp, callback,
                                                               9);
      })
      .Case<handshake::AddIOp>([&](handshake::AddIOp addiOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          APInt res = dataCast<APInt>(lhs) + dataCast<APInt>(rhs);
          return res;
        };
        registerModel<GenericBinaryOpModel<handshake::AddIOp>>(addiOp, callback,
                                                               0);
      })
      .Case<handshake::AndIOp>([&](handshake::AndIOp andiOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          APInt x = dataCast<APInt>(lhs) & dataCast<APInt>(rhs);
          return x;
        };
        registerModel<GenericBinaryOpModel<handshake::AndIOp>>(andiOp, callback,
                                                               0);
      })
      .Case<handshake::CmpFOp>([&](handshake::CmpFOp cmpfOp) {
        BinaryCompFunc callback;

        switch (cmpfOp.getPredicate()) {
        case handshake::CmpFPredicate::OEQ:
        case handshake::CmpFPredicate::UEQ:
          callback = [](Data lhs, Data rhs) {
            APInt comp(1, dataCast<APFloat>(lhs).bitwiseIsEqual(
                              dataCast<APFloat>(rhs)));
            return comp;
          };
          break;
        case handshake::CmpFPredicate::ONE:
        case handshake::CmpFPredicate::UNE:
          callback = [](Data lhs, Data rhs) {
            APInt comp(1, !dataCast<APFloat>(lhs).bitwiseIsEqual(
                              dataCast<APFloat>(rhs)));
            return comp;
          };
          break;
        case handshake::CmpFPredicate::OLE:
        case handshake::CmpFPredicate::ULE:
          callback = [](Data lhs, Data rhs) {
            APInt comp(1, dataCast<APFloat>(lhs) <= dataCast<APFloat>(rhs));
            return comp;
          };
          break;
        case handshake::CmpFPredicate::OLT:
        case handshake::CmpFPredicate::ULT:
          callback = [](Data lhs, Data rhs) {
            APInt comp(1, dataCast<APFloat>(lhs) < dataCast<APFloat>(rhs));
            return comp;
          };
          break;
        case handshake::CmpFPredicate::OGE:
        case handshake::CmpFPredicate::UGE:
          callback = [](Data lhs, Data rhs) {
            APInt comp(1, dataCast<APFloat>(lhs) >= dataCast<APFloat>(rhs));
            return comp;
          };
          break;
        case handshake::CmpFPredicate::OGT:
        case handshake::CmpFPredicate::UGT:
          callback = [](Data lhs, Data rhs) {
            APInt comp(1, dataCast<APFloat>(lhs) > dataCast<APFloat>(rhs));
            return comp;
          };
          break;
        case handshake::CmpFPredicate::ORD:
        case handshake::CmpFPredicate::UNO:
        case handshake::CmpFPredicate::AlwaysFalse:
          callback = [](const Data &lhs, const Data &rhs) {
            APInt comp(1, false);
            return comp;
          };
          break;
        case handshake::CmpFPredicate::AlwaysTrue:
          callback = [](const Data &lhs, const Data &rhs) {
            APInt comp(1, true);
            return comp;
          };
          break;
        }

        registerModel<GenericBinaryOpModel<handshake::CmpFOp>>(cmpfOp, callback,
                                                               4);
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

        registerModel<GenericBinaryOpModel<handshake::CmpIOp>>(cmpiOp, callback,
                                                               0);
      })
      .Case<handshake::DivFOp>([&](handshake::DivFOp divfOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          auto res = dataCast<APFloat>(lhs) / dataCast<APFloat>(rhs);
          return res;
        };
        registerModel<GenericBinaryOpModel<handshake::DivFOp>>(divfOp, callback,
                                                               29);
      })
      .Case<handshake::DivSIOp>([&](handshake::DivSIOp divsIOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          auto res = dataCast<APInt>(lhs).sdiv(dataCast<APInt>(rhs));
          return res;
        };
        registerModel<GenericBinaryOpModel<handshake::DivSIOp>>(divsIOp,
                                                                callback, 36);
      })
      .Case<handshake::DivUIOp>([&](handshake::DivUIOp divuIOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          auto res = dataCast<APInt>(lhs).udiv(dataCast<APInt>(rhs));
          return res;
        };
        registerModel<GenericBinaryOpModel<handshake::DivUIOp>>(divuIOp,
                                                                callback, 36);
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
      .Case<handshake::ExtUIOp>([&](handshake::ExtUIOp extuiOp) {
        UnaryCompFunc callback = [](const Data &lhs, unsigned outWidth) {
          auto temp = dataCast<APInt>(lhs).getZExtValue();
          APInt ext(outWidth, temp, true);
          return ext;
        };

        registerModel<GenericUnaryOpModel<handshake::ExtUIOp>>(extuiOp,
                                                               callback);
      })
      .Case<handshake::MaximumFOp>([&](handshake::MaximumFOp maxOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          auto res = maxnum(dataCast<APFloat>(lhs), dataCast<APFloat>(rhs));
          return res;
        };
        registerModel<GenericBinaryOpModel<handshake::MaximumFOp>>(maxOp,
                                                                   callback, 2);
      })
      .Case<handshake::MinimumFOp>([&](handshake::MinimumFOp minOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          auto res = minnum(dataCast<APFloat>(lhs), dataCast<APFloat>(rhs));
          return res;
        };
        registerModel<GenericBinaryOpModel<handshake::MinimumFOp>>(minOp,
                                                                   callback, 2);
      })
      .Case<handshake::MulFOp>([&](handshake::MulFOp mulfOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          auto mul = dataCast<APFloat>(lhs) * dataCast<APFloat>(rhs);
          return mul;
        };
        registerModel<GenericBinaryOpModel<handshake::MulFOp>>(mulfOp, callback,
                                                               4);
      })
      .Case<handshake::MulIOp>([&](handshake::MulIOp muliOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          APInt mul = dataCast<APInt>(lhs) * dataCast<APInt>(rhs);
          return mul;
        };
        registerModel<GenericBinaryOpModel<handshake::MulIOp>>(muliOp, callback,
                                                               4);
      })
      .Case<handshake::NegFOp>([&](handshake::NegFOp negfOp) {
        UnaryCompFunc callback = [](const Data &lhs, unsigned outWidth) {
          auto temp = dataCast<APFloat>(lhs);
          return neg(temp);
        };
        registerModel<GenericUnaryOpModel<handshake::NegFOp>>(negfOp, callback);
      })
      .Case<handshake::OrIOp>([&](handshake::OrIOp orIOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          APInt res = dataCast<APInt>(lhs) | dataCast<APInt>(rhs);
          return res;
        };
        registerModel<GenericBinaryOpModel<handshake::OrIOp>>(orIOp, callback,
                                                              0);
      })
      .Case<handshake::SelectOp>([&](handshake::SelectOp selectOp) {
        registerModel<SelectModel, handshake::SelectOp>(selectOp);
      })
      .Case<handshake::ShLIOp>([&](handshake::ShLIOp shliOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          APInt shli = dataCast<APInt>(lhs) << dataCast<APInt>(rhs);
          return shli;
        };
        registerModel<GenericBinaryOpModel<handshake::ShLIOp>>(shliOp, callback,
                                                               0);
      })
      .Case<handshake::ShRSIOp>([&](handshake::ShRSIOp shrSIOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          APInt res = dataCast<APInt>(lhs).ashr(dataCast<APInt>(rhs));
          return res;
        };
        registerModel<GenericBinaryOpModel<handshake::ShRSIOp>>(shrSIOp,
                                                                callback, 0);
      })
      .Case<handshake::ShRUIOp>([&](handshake::ShRUIOp shrUIOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          APInt res = dataCast<APInt>(lhs).lshr(dataCast<APInt>(rhs));
          return res;
        };
        registerModel<GenericBinaryOpModel<handshake::ShRUIOp>>(shrUIOp,
                                                                callback, 0);
      })
      .Case<handshake::SubFOp>([&](handshake::SubFOp subFOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          auto res = dataCast<APFloat>(lhs) - dataCast<APFloat>(rhs);
          return res;
        };
        registerModel<GenericBinaryOpModel<handshake::SubFOp>>(subFOp, callback,
                                                               9);
      })
      .Case<handshake::SubIOp>([&](handshake::SubIOp subiOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          APInt res = dataCast<APInt>(lhs) - dataCast<APInt>(rhs);
          return res;
        };
        registerModel<GenericBinaryOpModel<handshake::SubIOp>>(subiOp, callback,
                                                               0);
      })
      .Case<handshake::TruncIOp>([&](handshake::TruncIOp trunciOp) {
        registerModel<TruncIModel, handshake::TruncIOp>(trunciOp);
      })
      .Case<handshake::XOrIOp>([&](handshake::XOrIOp xorIOp) {
        BinaryCompFunc callback = [](Data lhs, Data rhs) {
          APInt x = dataCast<APInt>(lhs) ^ dataCast<APInt>(rhs);
          return x;
        };
        registerModel<GenericBinaryOpModel<handshake::XOrIOp>>(xorIOp, callback,
                                                               0);
      })
      // end
      .Case<handshake::EndOp>([&](handshake::EndOp endOp) {
        registerModel<EndModel, handshake::EndOp>(endOp, resValid, resReady,
                                                  resData);
      })
      .Default([&](auto) {
        emitError(op->getLoc()) << "Operation " << op
                                << " has unsupported type, we should probably "
                                   "report an error and stop";
      });
}

// Register the State state where needed (oldValueStates, newValueStates,
// updaters, rws).
template <typename State, typename Updater, typename Producer,
          typename Consumer, typename Ty>
void Simulator::registerState(Value val, Operation *producerOp, Ty type) {
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
void Simulator::associateState(Value val, Operation *producerOp, Location loc) {
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