//===- HandshakeAnnotateProperties.cpp - Property annotation ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --handshake-annotate-properties pass.
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/FormalPropertyAnnotation/HandshakeAnnotateProperties.h"
#include "dynamatic/Analysis/IndexChannelAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LinearAlgebra/Gaussian.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/FormalProperty.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <ostream>
#include <unordered_set>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Analysis/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_HANDSHAKEANNOTATEPROPERTIES
#include "experimental/Analysis/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

namespace {

struct HandshakeAnnotatePropertiesPass
    : public dynamatic::experimental::impl::HandshakeAnnotatePropertiesBase<
          HandshakeAnnotatePropertiesPass> {

  using HandshakeAnnotatePropertiesBase::HandshakeAnnotatePropertiesBase;

  void runDynamaticPass() override;

private:
  unsigned int uid;
  json::Array propertyTable;

  LogicalResult annotateAbsenceOfBackpressure(ModuleOp modOp);
  LogicalResult annotateValidEquivalence(ModuleOp modOp);
  LogicalResult annotateValidEquivalenceBetweenOps(Operation &op1,
                                                   Operation &op2);
  LogicalResult annotateEagerForkNotAllOutputSent(ModuleOp modOp);
  LogicalResult
  annotateCopiedSlotsRec(std::unordered_set<std::string> &visitedSet,
                         handshake::EagerForkLikeOpInterface &originFork,
                         Operation &curOp);
  LogicalResult annotateCopiedSlots(Operation &op);
  LogicalResult annotateCopiedSlotsOfAllForks(ModuleOp modOp);
  LogicalResult annotateReconvergentPathFlow(ModuleOp modOp);
};

bool isChannelToBeChecked(OpResult res) {
  // The channel connected to EndOp, MemoryControllerOp, and LSQOp don't appear
  // in the properties database for the following reasons:
  // - EndOp: the operation doesn't exist in the output model; the property
  //   creation is still possible but requires to get the names of the model's
  //   I/O signals (not implemented yet)
  // - MemeoryControllerOp and LSQOp: only load and stores can be connected to
  //   these Ops, therefore we cannot rigidify their channels with the
  //   ReadyRemoverOp and ValidMergerOp
  if (isa<handshake::EndOp, handshake::MemoryControllerOp, handshake::LSQOp>(
          res.getOwner()))
    return false;

  return std::all_of(
      res.getUsers().begin(), res.getUsers().end(), [](auto *user) {
        return !isa<handshake::EndOp, handshake::MemoryControllerOp,
                    handshake::LSQOp>(*user);
      });
}
} // namespace

LogicalResult
HandshakeAnnotatePropertiesPass::annotateValidEquivalenceBetweenOps(
    Operation &op1, Operation &op2) {
  for (auto [i, res1] : llvm::enumerate(op1.getResults()))
    for (auto [j, res2] : llvm::enumerate(op2.getResults())) {
      // equivalence is symmetrical so it needs to be checked only once for
      // each pair of signals when the Ops are the same
      if ((&op1 != &op2 || i < j) && isChannelToBeChecked(res1) &&
          isChannelToBeChecked(res2)) {
        ValidEquivalence p(uid, FormalProperty::TAG::OPT, res1, res2);

        propertyTable.push_back(p.toJSON());
        uid++;
      }
    }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateValidEquivalence(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (auto &op : funcOp.getOps()) {
      if (failed(annotateValidEquivalenceBetweenOps(op, op))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateAbsenceOfBackpressure(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      for (auto [resIndex, res] : llvm::enumerate(op.getResults()))
        if (isChannelToBeChecked(res)) {

          AbsenceOfBackpressure p(uid, FormalProperty::TAG::OPT, res);

          propertyTable.push_back(p.toJSON());
          uid++;
        }
    }
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateEagerForkNotAllOutputSent(
    ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (auto forkOp = dyn_cast<handshake::EagerForkLikeOpInterface>(op)) {
        EagerForkNotAllOutputSent p(uid, FormalProperty::TAG::INVAR, forkOp);

        propertyTable.push_back(p.toJSON());
        uid++;
      }
    }
  }
  return success();
}

LogicalResult HandshakeAnnotatePropertiesPass::annotateCopiedSlotsRec(
    std::unordered_set<std::string> &visitedSet,
    handshake::EagerForkLikeOpInterface &originFork, Operation &curOp) {

  // If this operation has been visited, there is nothing to do
  std::string id = getUniqueName(&curOp).str();
  if (auto iter = visitedSet.find(id); iter != visitedSet.end()) {
    return success();
  }
  visitedSet.insert(id);

  // If this operation contains a slot, the copied slot has been found and can
  // be annotated
  if (auto bufferOp = dyn_cast<handshake::BufferLikeOpInterface>(curOp)) {
    CopiedSlotsOfActiveForkAreFull p(uid, FormalProperty::TAG::INVAR, bufferOp,
                                     originFork);
    propertyTable.push_back(p.toJSON());
    uid++;
    return success();
  }

  if (auto mergeOp = dyn_cast<handshake::MergeLikeOpInterface>(curOp)) {
    // TODO: Which of the previous paths should be followed?
    return success();
  }

  // Only JoinLikeOps or single-operand ops are remaining, but ideally a
  // dyn_cast would happen for either case
  for (auto value : curOp.getOperands()) {
    Operation *prevOpPtr = value.getDefiningOp();
    if (prevOpPtr == nullptr)
      // if there is no defining op, the value must be a constant, and does not
      // need to be annotated
      continue;
    Operation &prevOp = *prevOpPtr;
    if (failed(annotateCopiedSlotsRec(visitedSet, originFork, prevOp))) {
      return failure();
    }
  }

  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateCopiedSlots(Operation &op) {
  std::unordered_set<std::string> visitedSet = {};
  if (auto forkOp = dyn_cast<handshake::EagerForkLikeOpInterface>(op)) {
    return annotateCopiedSlotsRec(visitedSet, forkOp, op);
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateCopiedSlotsOfAllForks(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (failed(annotateCopiedSlots(op)))
        return failure();
    }
  }
  return success();
}

namespace dynamatic {
// Used to assign dense indices to FlowVariables based on a list of
// FlowExpression, i.e. indices 0 to n-1 are used for n variables, while keeping
// lambda variables with low indices to ensure they are eliminated first within
// the row-echelon form
struct FlowEquationsMatrix {
  std::unordered_map<FlowVariable, size_t> varToIndex;
  std::vector<FlowVariable> indexToVar;
  size_t nLambdas;
  MatIntType matrix;

  size_t size() const { return indexToVar.size(); }
  void verify() {
    assert(varToIndex.size() == indexToVar.size());
    for (size_t i = 0; i < indexToVar.size(); ++i) {
      FlowVariable &a = indexToVar[i];
      size_t j = varToIndex[a];
      assert(i == j);
    }
    for (auto &[key, value] : varToIndex) {
      assert(indexToVar[value] == key);
    }
  }

  FlowExpression getRowAsExpression(size_t row) const {
    FlowExpression ret;
    for (size_t col = 0; col < indexToVar.size(); ++col) {
      int coef = matrix(row, col);
      if (coef != 0) {
        ret += coef * indexToVar[col];
      }
    }
    return ret;
  }

  FlowEquationsMatrix() = default;
  FlowEquationsMatrix(const std::vector<FlowExpression> &exprs) {
    size_t index = 0;
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
        if (varToIndex.count(key) == 0) {
          varToIndex[key] = index;
          ++index;
          indexToVar.push_back(key);
        }
      }
    }
    nLambdas = index;
    // annotate remaining variables
    for (auto &expr : exprs) {
      for (auto &[key, value] : expr.terms) {
        if (varToIndex.count(key) == 0) {
          varToIndex[key] = index;
          ++index;
          indexToVar.push_back(key);
        }
      }
    }

    // matrix with one row per equation, and column per variable
    matrix = MatIntZero(exprs.size(), size());

    // insert equations into the matrix
    for (auto [row, expr] : llvm::enumerate(exprs)) {
      for (auto &[key, value] : expr.terms) {
        unsigned index = varToIndex[key];
        matrix(row, index) = (int)value;
      }
    }
  }
};

struct EquationExtractor {
  std::vector<FlowExpression> equations;
  const IndexChannelAnalysis &indexChannelAnalysis;

  EquationExtractor(ModuleOp modOp, const IndexChannelAnalysis &ica)
      : equations(), indexChannelAnalysis(ica) {
    extractAll(modOp);
  }
  void extractAll(ModuleOp modOp);
  void extractControlMergeOp(ControlMergeOp cmergeOp);
  void extractMergeLikeOp(MergeLikeOpInterface mergeOp, FlowVariable &after);
  void extractMuxOpExtra(MuxOp muxOp, FlowVariable &after);
  void extractJoinOp(Operation &op, FlowVariable &after);
  void extractPipeline(LatencyInterface op, FlowVariable &i2);
  void extractBufferLikeOp(BufferLikeOpInterface bufferOp, FlowVariable &exit);
  void extractEagerFork(EagerForkLikeOpInterface forkOp, FlowVariable &before);
  void extractBranchOp(ConditionalBranchOp branchOp, FlowVariable &before);
  void extractLazyFork(Operation &op, FlowVariable &before);
};

void EquationExtractor::extractControlMergeOp(ControlMergeOp cmergeOp) {
  size_t numInputs = cmergeOp.getDataOperands().size();

  FlowVariable x0(InternalLambda(cmergeOp, 0));
  x0.indexTokenConstraint = IndexTracker(numInputs);

  for (auto [i, channel] : llvm::enumerate(cmergeOp.getDataOperands())) {
    FlowVariable channelVar(indexChannelAnalysis, ChannelLambda(channel));
    equations.push_back(channelVar - x0.setTrackedTokens(i));
  }

  auto slots = cmergeOp.getInternalSlotStateNamers();
  std::shared_ptr<InternalStateNamer> slotNamer =
      std::make_shared<BufferSlotFullNamer>(slots[0]);
  FlowVariable slot(slotNamer);
  slot.indexTokenConstraint = IndexTracker(numInputs);

  FlowVariable indexChannel = x0.nextInternal();
  for (size_t i = 0; i < numInputs; ++i) {
    equations.push_back(x0.setTrackedTokens(i) -
                        indexChannel.setTrackedTokens(i) -
                        slot.setTrackedTokens(i));
  }
  FlowVariable dataChannel = indexChannel.nextInternal();
  dataChannel.indexTokenConstraint.reset();
  equations.push_back(x0 - dataChannel - slot);

  auto sentNamers = cmergeOp.getInternalSentStateNamers();

  std::shared_ptr<InternalStateNamer> dataNamer =
      std::make_shared<EagerForkSentNamer>(sentNamers[0]);
  std::shared_ptr<InternalStateNamer> indexNamer =
      std::make_shared<EagerForkSentNamer>(sentNamers[1]);
  FlowVariable dataSent(dataNamer);
  FlowVariable indexSent(indexNamer);
  indexSent.indexTokenConstraint = IndexTracker(numInputs);

  auto outputs = cmergeOp.getResults();
  FlowVariable data(indexChannelAnalysis, ChannelLambda(outputs[0]));
  FlowVariable index(indexChannelAnalysis, ChannelLambda(outputs[1]));

  for (size_t i = 0; i < numInputs; ++i) {
    equations.push_back(index.setTrackedTokens(i) -
                        indexSent.setTrackedTokens(i) -
                        indexChannel.setTrackedTokens(i));
  }
  equations.push_back(data - dataSent - dataChannel);
}

void EquationExtractor::extractMergeLikeOp(MergeLikeOpInterface mergeOp,
                                           FlowVariable &after) {
  FlowExpression mergeEq = -after;
  std::vector<FlowExpression> constrainedEqs;
  size_t indexValue;
  bool foundIndexed = false;
  bool foundUnindexed = false;
  auto channels = mergeOp.getDataOperands();
  for (auto channel : channels) {
    FlowVariable ch(indexChannelAnalysis, ChannelLambda(channel));
    if (ch.isIndex()) {
      if (!foundIndexed) {
        indexValue = ch.indexTokenConstraint->numValues;
        constrainedEqs.resize(indexValue);
      }
      foundIndexed = true;
      assert(indexValue == ch.indexTokenConstraint->numValues &&
             "found differing index");
      for (size_t i = 0; i < indexValue; ++i) {
        constrainedEqs[i] += ch.setTrackedTokens(i);
      }
    } else {
      foundUnindexed = true;
      assert(foundIndexed == false);
      mergeEq += ch;
    }
  }
  assert(!(foundIndexed && foundUnindexed) &&
         "some index channels and some normal channels");
  if (foundIndexed) {
    after.indexTokenConstraint = IndexTracker(indexValue);
    for (size_t i = 0; i < indexValue; ++i) {
      constrainedEqs[i] -= after.setTrackedTokens(i);
      equations.push_back(constrainedEqs[i]);
    }
  } else {
    equations.push_back(mergeEq);
  }
}

void EquationExtractor::extractMuxOpExtra(MuxOp muxOp, FlowVariable &after) {
  // mux : select input has same as output lambda, data inputs act like
  Value select = muxOp.getSelectOperand();
  FlowVariable selectVar(indexChannelAnalysis, ChannelLambda(select));
  if (selectVar.isIndex()) {
    auto dataOperands = muxOp.getDataOperands();
    assert(selectVar.indexTokenConstraint->numValues == dataOperands.size());
    for (auto [i, operand] : llvm::enumerate(dataOperands)) {
      FlowVariable var(indexChannelAnalysis, ChannelLambda(operand));
      equations.push_back(selectVar.setTrackedTokens(i) - var);
    }
  } else {
    assert(false && "muxOp select var should always be index");
    FlowExpression dataEq = -after;
    for (auto operand : muxOp.getDataOperands()) {
      FlowVariable chVar(indexChannelAnalysis, ChannelLambda(operand));
      dataEq += chVar;
    }
    equations.push_back(dataEq);
  }
}

void EquationExtractor::extractJoinOp(Operation &op, FlowVariable &after) {
  auto channels = op.getOperands();
  if (channels.size() == 1) {
    // Only 1 input channel
    auto channel = channels[0];
    FlowVariable chVar(indexChannelAnalysis, ChannelLambda(channel));
    // If input is +-, then intermediate channel is as well
    after.indexTokenConstraint = chVar.indexTokenConstraint;
    if (chVar.isIndex()) {
      for (size_t i = 0; i < chVar.indexTokenConstraint->numValues; ++i) {
        equations.push_back(chVar.setTrackedTokens(i) -
                            after.setTrackedTokens(i));
      }
    } else {
      equations.push_back(chVar - after);
    }
  } else {
    for (auto channel : channels) {
      FlowVariable chVar(indexChannelAnalysis, ChannelLambda(channel));
      equations.push_back(chVar - after);
    }
  }
}

void EquationExtractor::extractPipeline(LatencyInterface latencyOp,
                                        FlowVariable &i2) {
  // Annotates equation for each pipeline slot, and changes i2 to be the
  // internal channel after the slots
  for (auto &pipelineSlot : latencyOp.getPipelineSlots()) {
    std::shared_ptr<InternalStateNamer> namer =
        std::make_shared<PipelineSlotNamer>(pipelineSlot);
    FlowVariable full(namer);

    FlowVariable before = i2;
    FlowVariable after = before.nextInternal();
    assert(!before.isIndex() && "Pipeline slot's data cannot be "
                                "accessed, so it cannot be constrained");

    equations.push_back(before - full - after);
    i2 = after;
  }
}

void EquationExtractor::extractBufferLikeOp(BufferLikeOpInterface bufferOp,
                                            FlowVariable &exit) {
  // Annotates equation for each slot, and changes exit to be the internal
  // channel after the slots
  for (auto &slotFull : bufferOp.getInternalSlotStateNamers()) {
    std::shared_ptr<InternalStateNamer> namer =
        std::make_shared<BufferSlotFullNamer>(slotFull);
    FlowVariable full(namer);

    FlowVariable before = exit;
    FlowVariable after = before.nextInternal();
    if (before.isIndex()) {
      assert(after.isIndex());
      FlowVariable constrainedFull = full;
      size_t numValues = before.indexTokenConstraint->numValues;
      constrainedFull.indexTokenConstraint = IndexTracker(numValues);
      // The slot contains one of the indices - never any other value
      equations.push_back(full - constrainedFull);
      for (size_t i = 0; i < numValues; ++i) {
        equations.push_back(before.setTrackedTokens(i) -
                            constrainedFull.setTrackedTokens(i) -
                            after.setTrackedTokens(i));
      }
    } else {
      equations.push_back(before - full - after);
    }
    exit = after;
  }
}

void EquationExtractor::extractEagerFork(EagerForkLikeOpInterface forkOp,
                                         FlowVariable &before) {
  for (auto [i, sentVariable] :
       llvm::enumerate(forkOp.getInternalSentStateNamers())) {
    std::shared_ptr<InternalStateNamer> namer =
        std::make_shared<EagerForkSentNamer>(sentVariable);
    FlowVariable sent(namer);
    FlowVariable result(indexChannelAnalysis,
                        ChannelLambda(forkOp->getResults()[i]));
    if (before.isIndex()) {
      assert(result.isIndex());
      sent.indexTokenConstraint =
          IndexTracker(before.indexTokenConstraint->numValues);
      // before + sent - result = 0 for every token value
      for (size_t i = 0; i < before.indexTokenConstraint->numValues; ++i) {
        equations.push_back(before.setTrackedTokens(i) +
                            sent.setTrackedTokens(i) -
                            result.setTrackedTokens(i));
      }
    } else {
      equations.push_back(before + sent - result);
    }
  }
}

void EquationExtractor::extractBranchOp(ConditionalBranchOp branchOp,
                                        FlowVariable &before) {
  FlowVariable trueVar(indexChannelAnalysis,
                       ChannelLambda(branchOp.getTrueResult()));
  FlowVariable falseVar(indexChannelAnalysis,
                        ChannelLambda(branchOp.getFalseResult()));
  FlowVariable condition(indexChannelAnalysis,
                         ChannelLambda(branchOp.getConditionOperand()));
  assert(condition.isIndex() && "branch op condition should be an index");
  assert(condition.indexTokenConstraint->numValues == 2);
  // The number of tokens going across the false result is equal to the
  // number of tokens=0 received at the condition input
  equations.push_back(falseVar - condition.setTrackedTokens(0));
  equations.push_back(trueVar - condition.setTrackedTokens(1));
}

void EquationExtractor::extractLazyFork(Operation &op, FlowVariable &before) {
  // lazy fork: all outputs have same tokens in as out
  for (auto [i, channel] : llvm::enumerate(op.getResults())) {
    FlowVariable result(indexChannelAnalysis, ChannelLambda(channel));
    if (before.isIndex() && result.isIndex()) {
      for (size_t i = 0; i < before.indexTokenConstraint->numValues; ++i) {
        equations.push_back(before.setTrackedTokens(i) -
                            result.setTrackedTokens(i));
      }
    } else {
      equations.push_back(before - result);
    }
  }
}

void EquationExtractor::extractAll(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      // A general structure for an operation is assumed:
      // in1, in2, ... -> Join/Merge/Mux        -> entry channel
      // entry channel -> arithmetic operation? -> i1
      // i1            -> pipeline slots?       -> i2
      // i2            -> slots?                -> exit
      // exit channel  -> Fork/Branch           -> out1, out2, ...
      //
      // Some operations do not follow this structure, and should be handled
      // separately to avoid making false assumptions.
      if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
        continue;
      }
      if (auto storeOp = dyn_cast<handshake::StoreOp>(op)) {
        continue;
      }
      if (auto controllerOp = dyn_cast<handshake::MemoryControllerOp>(op)) {
        continue;
      }
      if (auto cmergeOp = dyn_cast<handshake::ControlMergeOp>(op)) {
        extractControlMergeOp(cmergeOp);
        continue;
      }

      // in1, in2, ... -> Join/Merge/Mux -> entry channel
      FlowVariable entry = InternalLambda(&op, 0);
      if (auto mergeOp = dyn_cast<handshake::MergeLikeOpInterface>(op)) {
        extractMergeLikeOp(mergeOp, entry);
        if (auto muxOp = dyn_cast<handshake::MuxOp>(op)) {
          extractMuxOpExtra(muxOp, entry);
        }
      } else {
        extractJoinOp(op, entry);
      }

      // entry channel -> arithmetic operation? -> i1
      FlowVariable i1 = entry;
      if (auto arithOp = dyn_cast<handshake::ArithOpInterface>(op)) {
        // Arithmetic operations modify the channel - unless further analysis is
        // done, information about the carried token is lost
        if (entry.isIndex()) {
          i1 = entry.nextInternal();
          i1.indexTokenConstraint.reset();
          equations.push_back(entry - i1);
        }
      }

      // Annotate latency-induced slots
      // i1            -> pipeline slots?       -> i2
      FlowVariable i2 = i1;
      if (auto latencyOp = dyn_cast<handshake::LatencyInterface>(op)) {
        extractPipeline(latencyOp, i2);
      }

      // Annotate buffer slots
      // i2            -> slots?                -> exit
      FlowVariable exit = i2;
      if (auto bufferOp = dyn_cast<handshake::BufferLikeOpInterface>(op)) {
        extractBufferLikeOp(bufferOp, exit);
      }

      if (auto forkOp = dyn_cast<handshake::EagerForkLikeOpInterface>(op)) {
        extractEagerFork(forkOp, exit);
      } else if (auto branchOp = dyn_cast<handshake::ConditionalBranchOp>(op)) {
        extractBranchOp(branchOp, exit);
      } else {
        extractLazyFork(op, exit);
      }
    }
  }
}
} // namespace dynamatic

LogicalResult
HandshakeAnnotatePropertiesPass::annotateReconvergentPathFlow(ModuleOp modOp) {
  auto &indexChannelAnalysis = getAnalysis<dynamatic::IndexChannelAnalysis>();

  // The equations are represented by a FlowExpression that is equal to zero
  EquationExtractor extractor(modOp, indexChannelAnalysis);

  // Map all variables used in `equations` to an index in the matrix
  FlowEquationsMatrix indices(extractor.equations);
  MatIntType &matrix = indices.matrix;
  indices.verify();

  // bring to row-echelon form
  gaussianElimination(matrix);

  size_t rows = matrix.size1();
  for (size_t row = 0; row < rows; ++row) {
    bool canAnnotate = true;
    for (size_t col = 0; col < indices.nLambdas; ++col) {
      if (matrix(row, col) != 0) {
        canAnnotate = false;
        break;
      }
    }

    if (!canAnnotate) {
      continue;
    }

    FlowExpression expr = indices.getRowAsExpression(row);
    if (expr.terms.size() == 0) {
      continue;
    }
    ReconvergentPathFlow p(uid, FormalProperty::TAG::OPT);
    p.addEquation(expr);
    if (p.getEquations().size() > 0) {
      uid++;
      propertyTable.push_back(p.toJSON());
    }
  }
  return success();
}

void HandshakeAnnotatePropertiesPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  if (failed(annotateAbsenceOfBackpressure(modOp)))
    return signalPassFailure();
  if (failed(annotateValidEquivalence(modOp)))
    return signalPassFailure();
  if (annotateInvariants) {
    if (failed(annotateEagerForkNotAllOutputSent(modOp)))
      return signalPassFailure();
    if (failed(annotateCopiedSlotsOfAllForks(modOp)))
      return signalPassFailure();
    if (failed(annotateReconvergentPathFlow(modOp)))
      return signalPassFailure();
  }

  llvm::json::Value jsonVal(std::move(propertyTable));

  std::error_code EC;
  llvm::raw_fd_ostream jsonOut(jsonPath, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return;

  jsonOut << formatv("{0:2}", jsonVal);
}
