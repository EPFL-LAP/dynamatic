#include "experimental/Support/FlowExpression.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace dynamatic {
namespace handshake {

// --------------------
// --- FlowVariable ---
// --------------------
FlowVariable::FlowVariable(const IndexChannelAnalysis &indexChannels,
                           ChannelLambda channel) {
  *this = FlowVariable(Variants(channel));
  if (auto numValues = indexChannels.getIndexChannelValues(channel.channel)) {
    indexTokenConstraint = *numValues;
  }
}

FlowVariable FlowVariable::nextInternal() const {
  if (auto *lambda = std::get_if<ChannelLambda>(&variable)) {
    for (auto &use : lambda->channel.getUses()) {
      Operation *op = use.getOwner();
      FlowVariable ret(InternalLambda(op, 0));
      ret.indexTokenConstraint = indexTokenConstraint;
      return ret;
    }
  }
  auto *lambda = std::get_if<InternalLambda>(&variable);
  assert(lambda && "next internal can only be used on internal variables");
  InternalLambda nextLambda = *lambda;
  nextLambda.index += 1;
  FlowVariable next = *this;
  next.variable = nextLambda;
  return next;
}
FlowVariable FlowVariable::setTrackedTokens(size_t x) const {
  FlowVariable p = *this;
  if (!p.isIndex()) {
    llvm::report_fatal_error(llvm::formatv(
        "Attempting to set tracked token of non-index variable {0}",
        p.getDebugName()));
  }
  if (p.indexTokenConstraint->trackedValue) {
    llvm::report_fatal_error(
        llvm::formatv("Attempting to set tracked token that has already been "
                      "set of variable {0}",
                      p.getDebugName()));
  }
  p.indexTokenConstraint->trackedValue = x;
  return p;
}

std::shared_ptr<InternalStateNamer> FlowVariable::getAnnotater() const {
  auto *namer = std::get_if<std::shared_ptr<InternalStateNamer>>(&variable);
  if (!namer)
    return nullptr;

  auto &state = *namer;
  if (indexTokenConstraint && indexTokenConstraint->trackedValue) {
    return state->tryConstrain(*(indexTokenConstraint->trackedValue));
  }
  return state;
}

std::string FlowVariable::getDebugName() const {
  std::string ret = "";
  if (auto *namer =
          std::get_if<std::shared_ptr<InternalStateNamer>>(&variable)) {
    ret += (*namer)->getSMVName();
  }
  if (auto *channel = std::get_if<ChannelLambda>(&variable)) {
    if (auto *op = channel->channel.getDefiningOp()) {
      ret += getUniqueName(op);
      for (auto [i, ch] : llvm::enumerate(op->getResults())) {
        if (ch == channel->channel) {
          ret += llvm::formatv(".out{0}", i);
          break;
        }
      }
    } else {
      for (auto &opop : channel->channel.getUses()) {
        ret += llvm::formatv("{0}.in{1}", getUniqueName(opop.getOwner()),
                             opop.getOperandNumber());
        break;
      }
    }
  }
  if (auto *internal = std::get_if<InternalLambda>(&variable)) {
    ret +=
        llvm::formatv("{0}.#{1}", getUniqueName(internal->op), internal->index);
  }

  if (indexTokenConstraint) {
    if (indexTokenConstraint->trackedValue) {
      ret += llvm::formatv("(={0})", *(indexTokenConstraint->trackedValue));
    } else {
      ret += llvm::formatv("(=x)");
    }
  }
  return ret;
}

// ----------------------
// --- FlowExpression ---
// ----------------------
FlowExpression::FlowExpression(const FlowVariable &v) {
  if (v.isIndex()) {
    if (v.indexTokenConstraint->trackedValue) {
      terms[v] = 1;
      return;
    }
    // If plusAndMinus, separate into plus and minus parts
    for (size_t i = 0; i < v.indexTokenConstraint->numValues; ++i) {
      terms[v.setTrackedTokens(i)] = 1;
    }
  } else {
    terms[v] = 1;
  }
};

llvm::json::Value FlowExpression::toJSON() const {
  std::vector<llvm::json::Value> jsonTerms{};
  for (auto &[key, value] : terms) {
    auto *namer =
        std::get_if<std::shared_ptr<InternalStateNamer>>(&key.variable);
    assert(namer);
    std::optional<llvm::json::Value> constraintJson;
    if (key.indexTokenConstraint) {
      constraintJson = key.indexTokenConstraint->toJSON();
    } else {
      constraintJson = nullptr;
    }
    // int pm = key.pm;
    jsonTerms.emplace_back(
        llvm::json::Object({{STATE_LIT, (*namer)->toJSON()},
                            {COEFFICIENT_LIT, value},
                            {CONSTRAINT_LIT, constraintJson}}));
  }
  return llvm::json::Array(jsonTerms);
}

FlowExpression FlowExpression::fromJSON(const llvm::json::Value &value,
                                        llvm::json::Path path) {
  FlowExpression expr;
  const llvm::json::Array *arr = value.getAsArray();
  assert(arr && "FlowExpression JSON is not an array");
  for (const llvm::json::Value &termJSON : *arr) {
    const llvm::json::Object *obj = termJSON.getAsObject();
    assert(obj && "FlowExpression term JSON not an object");
    const llvm::json::Value *state = obj->get(STATE_LIT);
    if (!state) {
      llvm::report_fatal_error(
          "FlowExpression term JSON does not contain STATE_LIT");
    }
    std::shared_ptr<InternalStateNamer> namer =
        InternalStateNamer::fromJSON(*state, path);
    FlowVariable var(namer);
    int coef;
    llvm::json::ObjectMapper mapper(termJSON, path);
    if (!mapper || !mapper.map(COEFFICIENT_LIT, coef)) {
      llvm::report_fatal_error(
          "FlowExpression term JSON does not contain COEFFICIENT_LIT");
    }
    const llvm::json::Value *constraint = obj->get(CONSTRAINT_LIT);
    assert(constraint && "FlowExpression does not contain CONSTRAINT_LIT");
    if (auto n = constraint->getAsNull()) {
      assert(!var.indexTokenConstraint);
    } else {
      var.indexTokenConstraint = IndexTracker::fromJSON(*constraint, path);
    }

    expr.terms[var] = coef;
  }
  return expr;
}

std::string FlowExpression::debug() const {
  std::string ret;
  for (auto &[var, coef] : terms) {
    if (coef == 0) {
      ret += "0 * ";
    } else if (coef == 1) {
      ret += "+ ";
    } else if (coef == -1) {
      ret += "- ";
    } else {
      ret += llvm::formatv("{0} * ", coef);
    }
    ret += var.getDebugName();
    ret += "  ";
  }
  ret += "\n";
  return ret;
}

FlowExpression operator-(FlowExpression expr) {
  for (auto &[key, value] : expr.terms) {
    expr.terms[key] = -value;
  }
  return expr;
}

FlowExpression operator+(FlowExpression left, const FlowExpression &right) {
  for (auto &[key, value] : right.terms) {
    left.terms[key] += value;
  }
  return left;
}

FlowExpression operator*(int coef, FlowExpression expr) {
  for (auto &[key, value] : expr.terms) {
    expr.terms[key] *= coef;
  }
  return expr;
}

FlowExpression operator-(FlowExpression left, const FlowExpression &right) {
  for (auto &[key, value] : right.terms) {
    left.terms[key] -= value;
  }
  return left;
}

void operator+=(FlowExpression &left, const FlowExpression &right) {
  for (auto &[key, value] : right.terms) {
    left.terms[key] += value;
  }
}

void operator-=(FlowExpression &left, const FlowExpression &right) {
  for (auto &[key, value] : right.terms) {
    left.terms[key] -= value;
  }
}

// -----------------------------
// --- FlowEquationExtractor ---
// -----------------------------
LogicalResult FlowEquationExtractor::extractAll(ModuleOp modOp) {
  // Store the result so that all operations can be handled, rather than exiting
  // after first error
  // Upon failed annotation, do `res = failure();` and simply continue
  // annotating
  LogicalResult res = success();
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (isa<AddIOp, CmpIOp, ConstantOp, ExtUIOp, ExtSIOp, MulIOp, SubIOp,
              TruncIOp>(op)) {
        if (failed(extractArithmeticJoinOp(op))) {
          res = failure();
        }
      } else if (auto endOp = dyn_cast<EndOp>(op)) {
        if (failed(extractEndOp(endOp))) {
          res = failure();
        }
      } else if (isa<SinkOp, SourceOp>(op)) {
        // These operations do not place any constraint on incoming/outgoing
        // lambda channels, and can safely be ignored
      } else if (auto forkOp = dyn_cast<ForkOp>(op)) {
        if (failed(extractForkOp(forkOp))) {
          res = failure();
        }
      } else if (auto muxOp = dyn_cast<MuxOp>(op)) {
        if (failed(extractMuxOp(muxOp))) {
          res = failure();
        }
      } else if (auto bufferOp = dyn_cast<BufferOp>(op)) {
        if (failed(extractBufferOp(bufferOp))) {
          res = failure();
        }
      } else if (auto branchOp = dyn_cast<ConditionalBranchOp>(op)) {
        if (failed(extractBranchOp(branchOp))) {
          res = failure();
        }
      } else if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
        if (failed(extractLoadOp(loadOp))) {
          res = failure();
        }
      } else if (auto storeOp = dyn_cast<handshake::StoreOp>(op)) {
        if (failed(extractStoreOp(storeOp))) {
          res = failure();
        }
      } else if (auto cmergeOp = dyn_cast<handshake::ControlMergeOp>(op)) {
        if (failed(extractControlMergeOp(cmergeOp))) {
          res = failure();
        }
      } else if (auto memCon = dyn_cast<handshake::MemoryControllerOp>(op)) {
        if (failed(extractMemoryControllerOp(memCon))) {
          res = failure();
        }
      } else {
        op.emitError("Not handled yet!");
        res = failure();
      }
    }
  }
  return res;
}

// Extracts the slot equation from the corresponding variables, while preserving
// index constraints if present:
// lambda_in = lambda_out + slot
LogicalResult FlowEquationExtractor::extractSlotEquation(
    const FlowVariable &in, const FlowVariable &out, const FlowVariable &slot) {
  if (in.isIndex()) {
    // All variables should have the same number of possible index values, as
    // the slot does not modify the token
    if (!out.isIndex() || in.indexTokenConstraint->numValues !=
                              out.indexTokenConstraint->numValues) {
      llvm::errs()
          << "input and output of slot have index token constraint mismatch\n";
      return failure();
    }
    if (!slot.isIndex() || in.indexTokenConstraint->numValues !=
                               slot.indexTokenConstraint->numValues) {
      llvm::errs() << "input and slot have index token constraint mismatch\n";
      return failure();
    }
    // The slot equation is annotated for each possible index:
    // lambda_in(0) = lambda_out(0) + slot(0)
    // lambda_in(1) = lambda_out(1) + slot(1)
    // lambda_in(2) = lambda_out(2) + slot(2)
    // ...
    // where lambda_in(0) denotes the number of tokens with value 0 that have
    // propagated through `in`
    for (size_t i = 0; i < in.indexTokenConstraint->numValues; ++i) {
      equations.push_back(in.setTrackedTokens(i) - out.setTrackedTokens(i) -
                          slot.setTrackedTokens(i));
    }
  } else {
    equations.push_back(in - out - slot);
  }
  return success();
}

// Extracts the eager sent equation from the corresponding variables, while
// preserving index constraints if present:
// lambda_in + slot = lambda_out
LogicalResult FlowEquationExtractor::extractEagerSentEquation(
    const FlowVariable &in, const FlowVariable &out, const FlowVariable &sent) {
  if (in.isIndex()) {
    // All variables should have the same number of possible index values, as
    // the slot does not modify the token
    if (!out.isIndex() || in.indexTokenConstraint->numValues !=
                              out.indexTokenConstraint->numValues) {
      llvm::errs() << "input and output of eager sent state have index token "
                      "constraint mismatch";
      return failure();
    }
    if (!sent.isIndex() || in.indexTokenConstraint->numValues !=
                               sent.indexTokenConstraint->numValues) {
      llvm::errs()
          << "input and eager sent state have index token constraint mismatch";
      return failure();
    }
    // The slot equation is annotated for each possible index:
    // lambda_in(0) + slot(0) = lambda_out(0)
    // lambda_in(1) + slot(1) = lambda_out(1)
    // lambda_in(2) + slot(2) = lambda_out(2)
    // ...
    // where lambda_in(0) denotes the number of tokens with value 0 that have
    // propagated through `in`
    for (size_t i = 0; i < in.indexTokenConstraint->numValues; ++i) {
      equations.push_back(in.setTrackedTokens(i) - out.setTrackedTokens(i) +
                          sent.setTrackedTokens(i));
    }
  } else {
    equations.push_back(in - out + sent);
  }
  return success();
}

// Handles any arithmetic operation with a single output and any number of
// inputs, and potentially a pipeline for operations with latency. All inputs
// together propagate each token to the output, so:
// lambda_in_1 = lambda_in_2 = ... = joinResult
// joinResult = lambda_out + #slots_in_pipeline
LogicalResult FlowEquationExtractor::extractArithmeticJoinOp(Operation &op) {
  assert(op.getResults().size() == 1);
  FlowVariable out(indexChannelAnalysis, ChannelLambda(op.getResults()[0]));
  FlowVariable joinResult = out;
  if (auto latencyOp = dyn_cast<handshake::LatencyInterface>(op)) {
    FlowVariable inner(InternalLambda(&op, 0));
    joinResult = inner;
    // extractPipeline annotates the equations:
    // joinResult_i = joinResult_(i+1) + latencySlot_i
    // and sets `inner` to joinResult_
    if (failed(extractPipeline(latencyOp, inner))) {
      return failure();
    }
    equations.push_back(out - inner);
  }
  for (auto inChannel : op.getOperands()) {
    FlowVariable in(indexChannelAnalysis, ChannelLambda(inChannel));
    equations.push_back(joinResult - in);
  }
  return success();
}

LogicalResult
FlowEquationExtractor::extractBranchOp(ConditionalBranchOp branchOp) {
  FlowVariable dataVar(indexChannelAnalysis,
                       ChannelLambda(branchOp.getDataOperand()));
  FlowVariable trueVar(indexChannelAnalysis,
                       ChannelLambda(branchOp.getTrueResult()));
  FlowVariable falseVar(indexChannelAnalysis,
                        ChannelLambda(branchOp.getFalseResult()));
  FlowVariable condition(indexChannelAnalysis,
                         ChannelLambda(branchOp.getConditionOperand()));
  if (!condition.isIndex()) {
    branchOp.emitError("branch op condition should be an index");
    return failure();
  }
  if (condition.indexTokenConstraint->numValues != 2) {
    branchOp.emitError("branch op condition should have two possible values");
    return failure();
  }
  // The number of tokens going across the false result is equal to the
  // number of tokens=0 received at the condition input:
  // lambda_condition(0) = lambda_false
  // lambda_condition(1) = lambda_true
  equations.push_back(falseVar - condition.setTrackedTokens(0));
  equations.push_back(trueVar - condition.setTrackedTokens(1));

  // Data and condition come together:
  // lambda_data = lambda_condition
  equations.push_back(dataVar - condition);
  return success();
}

LogicalResult FlowEquationExtractor::extractBufferOp(BufferOp bufferOp) {
  FlowVariable in(indexChannelAnalysis, ChannelLambda(bufferOp.getOperand()));
  FlowVariable out(indexChannelAnalysis, ChannelLambda(bufferOp.getResult()));
  for (auto &slotNamer : bufferOp.getInternalSlotStateNamers()) {
    // For any slot: in -> slot_i -> next, where `next` is simply the next
    // internal channel after `in`. The normal slot equation is annotated:
    // in = next + slot
    // Then, `in` is set to `next`
    std::shared_ptr<InternalStateNamer> sharedNamer =
        std::make_shared<BufferSlotFullNamer>(slotNamer);
    FlowVariable slot(sharedNamer);
    FlowVariable next = in.nextInternal();
    if (failed(extractSlotEquation(in, next, slot))) {
      return failure();
    }
    in = next;
  }
  // `in` now is the last internal channel after all the slots. It is connected
  // to the output.
  equations.push_back(in - out);
  return success();
}

LogicalResult
FlowEquationExtractor::extractControlMergeOp(ControlMergeOp cmergeOp) {
  // Control Merge Op:
  //
  //    |         |         |
  //    |in1      |in2      |in3
  //    |         |         |
  //    ---  CMERGE OP  -----
  //          |       |
  //      data|       |index
  //    (CTRL)|       |(2 bits)
  //          |       |      `dataIntermediate` / `indexIntermediate`
  //          ----|----
  //             ---
  //             | |slot
  //             ---
  //              |
  //          ---------
  //          |       |      `dataChannel` / `indexChannel`
  //         ---     ---
  // dataSent| |     | |indexSent
  //         ---     ---
  //          |       |
  // dataOut  |       | indexOut
  //
  //
  // Notes:
  // 1. The data contained in the slot is simply the index, as the data is a
  // control signal that does not contain any data.
  // 2. The index signal is 2 bits as, in this example, there are three inputs,
  // and two bits are required to represent an index to three values.
  size_t numInputs = cmergeOp.getDataOperands().size();

  FlowVariable indexIntermediate(InternalLambda(cmergeOp, 0));
  indexIntermediate.indexTokenConstraint = IndexTracker(numInputs);

  for (auto [i, channel] : llvm::enumerate(cmergeOp.getDataOperands())) {
    FlowVariable channelVar(indexChannelAnalysis, ChannelLambda(channel));
    equations.push_back(channelVar - indexIntermediate.setTrackedTokens(i));
  }

  auto slots = cmergeOp.getInternalSlotStateNamers();
  std::shared_ptr<InternalStateNamer> slotNamer =
      std::make_shared<BufferSlotFullNamer>(slots[0]);
  FlowVariable slot(slotNamer);
  slot.indexTokenConstraint = IndexTracker(numInputs);

  FlowVariable indexChannel = indexIntermediate.nextInternal();
  if (failed(extractSlotEquation(indexIntermediate, indexChannel, slot))) {
    return failure();
  }
  FlowVariable dataIntermediate = indexChannel.nextInternal();
  dataIntermediate.indexTokenConstraint.reset();

  // Same number of tokens arrive at dataIntermediate and indexIntermediate
  equations.push_back(dataIntermediate - indexIntermediate);

  FlowVariable dataChannel = dataIntermediate.nextInternal();
  if (failed(extractSlotEquation(dataIntermediate, dataChannel, slot))) {
    return failure();
  }

  auto sentNamers = cmergeOp.getInternalSentStateNamers();

  std::shared_ptr<InternalStateNamer> dataNamer =
      std::make_shared<EagerForkSentNamer>(sentNamers[0]);
  std::shared_ptr<InternalStateNamer> indexNamer =
      std::make_shared<EagerForkSentNamer>(sentNamers[1]);
  FlowVariable dataSent(dataNamer);
  FlowVariable indexSent(indexNamer);
  indexSent.indexTokenConstraint = IndexTracker(numInputs);

  auto outputs = cmergeOp.getResults();
  FlowVariable dataOut(indexChannelAnalysis, ChannelLambda(outputs[0]));
  FlowVariable indexOut(indexChannelAnalysis, ChannelLambda(outputs[1]));

  if (failed(extractEagerSentEquation(indexChannel, indexOut, indexSent))) {
    return failure();
  }
  if (failed(extractEagerSentEquation(dataChannel, dataOut, dataSent))) {
    return failure();
  }
  return success();
}

LogicalResult FlowEquationExtractor::extractEndOp(EndOp endOp) {
  // All inputs of the end op propagate the same number of tokens:
  // lambda_in_1 = lambda_in_2 = ...
  FlowVariable out(InternalLambda(endOp, 0));
  for (auto inChannel : endOp.getInputs()) {
    FlowVariable in(indexChannelAnalysis, ChannelLambda(inChannel));
    equations.push_back(out - in);
  }
  return success();
}

LogicalResult FlowEquationExtractor::extractForkOp(ForkOp forkOp) {
  // The input is propagated to all outputs according to eager fork rules
  FlowVariable in(indexChannelAnalysis, ChannelLambda(forkOp.getOperand()));
  auto namers = forkOp.getInternalSentStateNamers();
  for (auto [i, outChannel] : llvm::enumerate(forkOp.getResult())) {
    FlowVariable out(indexChannelAnalysis, ChannelLambda(outChannel));
    std::shared_ptr<InternalStateNamer> sentNamer =
        std::make_shared<EagerForkSentNamer>(namers[i]);
    FlowVariable sent(sentNamer);
    sent.indexTokenConstraint = in.indexTokenConstraint;
    if (failed(extractEagerSentEquation(in, out, sent))) {
      return failure();
    }
  }
  return success();
}

LogicalResult FlowEquationExtractor::extractLoadOp(LoadOp loadOp) {
  // addrInput = addrOutput + addrSlot
  // addr_input -> addr_slot ----> MC ----->
  // dataInput = dataOutput + dataSlot
  // data_input -> data_slot -> data_output
  auto slots = loadOp.getInternalSlotStateNamers();
  std::shared_ptr<InternalStateNamer> addrNamer =
      std::make_shared<BufferSlotFullNamer>(slots[0]);
  FlowVariable addrSlot(addrNamer);
  FlowVariable addrInput(indexChannelAnalysis,
                         ChannelLambda(loadOp.getAddress()));
  FlowVariable addrOutput(indexChannelAnalysis,
                          ChannelLambda(loadOp.getAddressResult()));

  equations.push_back(addrInput - addrOutput - addrSlot);

  std::shared_ptr<InternalStateNamer> dataNamer =
      std::make_shared<BufferSlotFullNamer>(slots[1]);
  FlowVariable dataSlot(dataNamer);
  FlowVariable dataInput(indexChannelAnalysis, ChannelLambda(loadOp.getData()));
  FlowVariable dataOutput(indexChannelAnalysis,
                          ChannelLambda(loadOp.getDataResult()));

  equations.push_back(dataInput - dataOutput - dataSlot);

  return success();
}

LogicalResult
FlowEquationExtractor::extractMemoryControllerOp(MemoryControllerOp memCon) {
  // LoadPort         MemoryController
  //
  //           ----->   addrIn
  //                        |
  //                    ---------
  //                    | slot  |
  //                    ---------
  //                        |
  //           <-----   dataOut
  size_t nLoads = memCon.getNumLoadPorts();
  for (size_t loadIndex = 0; loadIndex < nLoads; ++loadIndex) {
    if (auto load = memCon.getLoadPort(loadIndex)) {
      unsigned operandIndex = load->getAddrInputIndex();
      unsigned resultIndex = load->getDataOutputIndex();
      FlowVariable addrIn(indexChannelAnalysis,
                          ChannelLambda(memCon.getOperands()[operandIndex]));
      FlowVariable dataOut(indexChannelAnalysis,
                           ChannelLambda(memCon.getResults()[resultIndex]));
      auto slotNamer = memCon.getLoadPortSlotNamer(loadIndex);
      std::shared_ptr<InternalStateNamer> sharedNamer =
          std::make_shared<MemoryControllerSlotNamer>(slotNamer);
      FlowVariable slot(sharedNamer);
      if (failed(extractSlotEquation(addrIn, dataOut, slot))) {
        return failure();
      }
    }
  }
  // For store ports, a next goal could be to transmit a control token
  // when the store operation has finished.
  return success();
}

LogicalResult FlowEquationExtractor::extractMuxOp(MuxOp muxOp) {
  // A mux op propagates a token from the data input selected by the select
  // input to the output:
  // lambda_select = lambda_out
  // lambda_select(0) = lambda_data_0
  // lambda_select(1) = lambda_data_1
  FlowVariable out(indexChannelAnalysis, ChannelLambda(muxOp.getResult()));
  FlowVariable selectVar(indexChannelAnalysis,
                         ChannelLambda(muxOp.getSelectOperand()));
  if (!selectVar.isIndex()) {
    muxOp.emitWarning("muxOp select input should always be index");
    FlowExpression dataEq = -selectVar;
    for (auto operand : muxOp.getDataOperands()) {
      FlowVariable chVar(indexChannelAnalysis, ChannelLambda(operand));
      dataEq += chVar;
    }
    equations.push_back(dataEq);
    return success();
  }
  auto dataOperands = muxOp.getDataOperands();

  if (selectVar.indexTokenConstraint->numValues != dataOperands.size()) {
    muxOp.emitError(
        "index channel analysis does not match number of mux inputs");
    return failure();
  }

  // lambda_select = lambda_out
  equations.push_back(selectVar - out);

  // lambda_select(i) = lambda_data_i
  for (auto [i, operand] : llvm::enumerate(dataOperands)) {
    FlowVariable data(indexChannelAnalysis, ChannelLambda(operand));
    equations.push_back(selectVar.setTrackedTokens(i) - data);
  }
  return success();
}

LogicalResult FlowEquationExtractor::extractPipeline(LatencyInterface latencyOp,
                                                     FlowVariable &internal) {
  // Annotates equation for each pipeline slot, and changes `internal` to be an
  // internal channel after the slots:
  // lambda_internal_0 = lambda_internal_1 + pipeline_full_0
  // lambda_internal_1 = lambda_internal_2 + pipeline_full_1
  // ...
  // lambda_internal_n-1 = lambda_internal_n + pipeline_full_n-1
  //
  // Finally, `internal` = lambda_internal_n
  for (auto &pipelineSlot : latencyOp.getPipelineSlots()) {
    std::shared_ptr<InternalStateNamer> namer =
        std::make_shared<PipelineSlotNamer>(pipelineSlot);
    FlowVariable full(namer);

    FlowVariable before = internal;
    FlowVariable after = before.nextInternal();
    if (before.isIndex()) {
      latencyOp.emitError("Pipeline slot's data cannot be accessed, so it "
                          "cannot be constrained");
      return failure();
    }

    if (failed(extractSlotEquation(before, after, full))) {
      return failure();
    }
    internal = after;
  }
  return success();
}

LogicalResult FlowEquationExtractor::extractStoreOp(StoreOp storeOp) {
  // Both data and address are simply forwarded along their channels
  FlowVariable dataInput(indexChannelAnalysis,
                         ChannelLambda(storeOp.getData()));
  FlowVariable dataOutput(indexChannelAnalysis,
                          ChannelLambda(storeOp.getDataResult()));

  equations.push_back(dataInput - dataOutput);

  FlowVariable addrInput(indexChannelAnalysis,
                         ChannelLambda(storeOp.getAddress()));
  FlowVariable addrOutput(indexChannelAnalysis,
                          ChannelLambda(storeOp.getAddressResult()));

  equations.push_back(addrInput - addrOutput);

  return success();
}

// ------------------
// --- FlowSystem ---
// ------------------
FlowExpression FlowSystem::getRowAsExpression(size_t row) const {
  FlowExpression ret;
  for (size_t col = 0; col < registry.size(); ++col) {
    int coef = matrix(row, col);
    if (coef != 0) {
      ret += coef * registry.getVar(col);
    }
  }
  return ret;
}

FlowSystem::FlowSystem(const std::vector<FlowExpression> &exprs) {
  // give lower indices to variables that cannot be annotated
  for (auto &expr : exprs) {
    for (auto &[key, value] : expr.terms) {
      // skip variables that can be annotated in SMV
      if (key.getAnnotater() != nullptr) {
        continue;
      }
      // PlusAndMinus variables should never be inserted, as the DSL will
      // insert them as two separate variables
      assert(!key.indexTokenConstraint ||
             key.indexTokenConstraint->trackedValue);
      registry.addVariable(key);
    }
  }
  nLambdas = registry.size();
  // annotate remaining variables
  for (auto &expr : exprs) {
    for (auto &[key, value] : expr.terms) {
      registry.addVariable(key);
    }
  }

  // matrix with one row per equation, and column per variable
  matrix = MatIntZero(exprs.size(), registry.size());

  // insert equations into the matrix
  for (auto [row, expr] : llvm::enumerate(exprs)) {
    for (auto &[key, value] : expr.terms) {
      unsigned index = registry.getIndex(key);
      matrix(row, index) = (int)value;
    }
  }
}

} // namespace handshake
} // namespace dynamatic
