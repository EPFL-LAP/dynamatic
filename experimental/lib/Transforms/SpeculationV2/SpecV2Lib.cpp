#include "JSONImporter.h"
#include "PreSpecV2Gamma.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/MaterializationUtil/MaterializationUtil.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include <fstream>

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculationv2;

LogicalResult replaceBranchesWithPassers(FuncOp &funcOp, unsigned bb) {
  // Find one ConditionBranchOp in the BB
  ConditionalBranchOp referenceBranch = nullptr;
  for (auto candidate : funcOp.getOps<ConditionalBranchOp>()) {
    auto condBB = getLogicBB(candidate);
    if (condBB && *condBB == bb) {
      referenceBranch = candidate;
      break;
    }
  }

  if (!referenceBranch)
    return funcOp.emitError("Could not find ConditionalBranchOp in the BB: " +
                            std::to_string(bb));

  Value condition = referenceBranch.getConditionOperand();

  // Build a NotOp to invert the condition
  OpBuilder builder(funcOp->getContext());
  builder.setInsertionPoint(referenceBranch);
  NotOp invertCondition = builder.create<NotOp>(condition.getLoc(), condition);
  setBB(invertCondition, bb);

  // Replace all branches in the BB with passers
  for (auto branch :
       llvm::make_early_inc_range(funcOp.getOps<ConditionalBranchOp>())) {
    if (getLogicBB(branch) != bb)
      continue;

    // The condition must be the same (ignoring the difference of the fork
    // outputs)
    if (!equalsIndirectly(condition, branch.getConditionOperand()))
      return branch.emitError("Branch condition does not match the condition "
                              "of the reference branch");

    Value data = branch.getDataOperand();

    builder.setInsertionPoint(branch);

    if (auto sink = dyn_cast<SinkOp>(getUniqueUser(branch.getTrueResult()))) {
      sink->erase();
    } else {
      // Build a passer for the trueResult
      PasserOp trueResultPasser =
          builder.create<PasserOp>(branch.getLoc(), data, condition);
      setBB(trueResultPasser, bb);
      branch.getTrueResult().replaceAllUsesWith(trueResultPasser.getResult());
    }

    if (auto sink = dyn_cast<SinkOp>(getUniqueUser(branch.getFalseResult()))) {
      sink->erase();
    } else {
      // Build a passer for the falseResult
      // The passer ctrl is inverted condition.
      PasserOp falseResultPasser = builder.create<PasserOp>(
          branch.getLoc(), data, invertCondition.getResult());
      setBB(falseResultPasser, bb);
      branch.getFalseResult().replaceAllUsesWith(falseResultPasser.getResult());
    }

    // Erase the branch
    branch->erase();
  }

  if (invertCondition.getResult().use_empty()) {
    // If the inverted condition is not used, erase it.
    invertCondition->erase();
  }

  return success();
}

bool isSourced(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return false;

  // Heuristic to stop the traversal earlier.
  if (isa<handshake::MuxOp>(definingOp))
    return false;

  if (isa<SourceOp>(value.getDefiningOp()))
    return true;

  // If all operands of the defining operation are sourced, the value is also
  // sourced.
  return llvm::all_of(value.getDefiningOp()->getOperands(),
                      [](Value v) { return isSourced(v); });
}

llvm::SmallVector<Value> getEffectiveOperands(Operation *op) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // For LoadOp, only the data result is effective for rewriting
    return {loadOp.getAddress()};
  }
  return llvm::to_vector(op->getOperands());
}

/// If op is LoadOp, excludes results going to MemoryControllerOp.
llvm::SmallVector<Value> getEffectiveResults(Operation *op) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // For LoadOp, only the data result is effective for rewriting
    return {loadOp.getDataResult()};
  }
  if (auto cmergeOp = dyn_cast<handshake::ControlMergeOp>(op)) {
    // We move suppressors past 1-input CMerge, which exists in
    // non-canonicalized circuits.
    return {cmergeOp.getResult()};
  }
  // Unlike the operands, to_vector doesn't work
  llvm::SmallVector<Value> results;
  for (OpResult result : op->getResults()) {
    results.push_back(result);
  }
  return results;
}

LogicalResult movePassersDownPM(Operation *pmOp) {
  OpBuilder builder(pmOp->getContext());
  builder.setInsertionPoint(pmOp);

  Location loc = builder.getUnknownLoc();
  Value ctrl = nullptr;

  // Remove PasserOp from each effective operand of the PM unit.
  for (Value operand : getEffectiveOperands(pmOp)) {
    Operation *definingOp = operand.getDefiningOp();
    if (auto passer = dyn_cast<PasserOp>(definingOp)) {
      loc = passer->getLoc();
      ctrl = passer.getCtrl();

      // The operand must be materialized to perform the motion correctly.
      assertMaterialization(operand);

      // Remove the defining PasserOp operation.
      passer.getResult().replaceAllUsesWith(passer.getData());
      passer->erase();
    } else {
      // If the operand is sourced, it doesn't need to be defined by OpT.
      if (isSourced(operand))
        continue;
      return pmOp->emitError(
          "Expected all operands to be defined by the PasserOp");
    }
  }

  // Add new PasserOp for each effective result of the PM unit.
  for (Value result : getEffectiveResults(pmOp)) {
    assertMaterialization(result);

    PasserOp newPasser = builder.create<PasserOp>(loc, result, ctrl);
    inheritBB(pmOp, newPasser);

    result.replaceAllUsesExcept(newPasser.getResult(), newPasser);
  }

  return success();
}

bool isEligibleForPasserMotionOverPM(PasserOp passerOp) {
  Value passerControl = passerOp.getCtrl();

  Operation *targetOp = getUniqueUser(passerOp.getResult());

  // If the targetOp is not a PM unit, return false.
  if (!isa<ArithOpInterface, NotOp, ForkOp, LazyForkOp, BufferOp, LoadOp,
           BranchOp>(targetOp)) {
    if (!isa<MergeOp, ControlMergeOp>(targetOp) ||
        targetOp->getNumOperands() != 1)
      return false;
  }

  // Iterate over operands of the targetOp to decide the eligibility for
  // motion.
  for (Value operand : getEffectiveOperands(targetOp)) {
    if (auto passerOp = dyn_cast<PasserOp>(operand.getDefiningOp())) {
      // If this passerOp is controlled by different control from the specified
      // one, not eligible.
      if (!equalsIndirectly(passerControl, passerOp.getCtrl()))
        return false;
    } else if (!isSourced(operand)) {
      // Each operand must be defined by a passer, except when it is driven by a
      // source op.
      return false;
    }
  }

  return true;
}

void performPasserMotionPastPM(PasserOp passerOp,
                               DenseSet<PasserOp> &frontiers) {
  Value passerControl = passerOp.getCtrl();
  OpBuilder builder(passerOp->getContext());
  builder.setInsertionPoint(passerOp);

  Operation *targetOp = getUniqueUser(passerOp.getResult());

  // Remove passers from the frontiers
  for (Value operand : getEffectiveOperands(targetOp)) {
    if (auto passerOp = dyn_cast<PasserOp>(operand.getDefiningOp())) {
      frontiers.erase(passerOp);
    }
  }

  // Perform the motion
  auto motionResult = movePassersDownPM(targetOp);

  if (failed(motionResult)) {
    targetOp->emitError("Failed to perform motion for PasserOp");
    llvm_unreachable("SpeculationV2 algorithm failed");
  }

  // Add new passers to the frontiers
  for (auto result : getEffectiveResults(targetOp)) {
    auto newPasser = cast<PasserOp>(getUniqueUser(result));
    frontiers.insert(newPasser);
    // Materialize the result of the new passer for further rewriting.
    materializeValue(newPasser.getResult());
  }

  materializeValue(passerControl);
}

DenseMap<unsigned, unsigned> unifyBBs(ArrayRef<unsigned> loopBBs,
                                      FuncOp funcOp) {
  DenseMap<unsigned, unsigned> bbMap;
  unsigned minBB = *std::min_element(loopBBs.begin(), loopBBs.end());
  funcOp.walk([&](Operation *op) {
    auto bbOrNull = getLogicBB(op);
    if (!bbOrNull.has_value())
      return;

    unsigned bb = bbOrNull.value();
    if (!bbMap.contains(bb)) {
      if (std::find(loopBBs.begin(), loopBBs.end(), bb) != loopBBs.end()) {
        bbMap[bb] = minBB;
      } else {
        unsigned d = 0;
        for (auto loopBB : loopBBs) {
          if (loopBB == minBB)
            continue;
          if (loopBB < bb)
            d++;
        }
        bbMap[bb] = bb - d;
      }
    }

    setBB(op, bbMap[bb]);
  });

  return bbMap;
}

void recalculateMCBlocks(FuncOp funcOp) {
  DenseSet<int32_t> bbs;
  OpBuilder builder(funcOp->getContext());

  for (auto mc :
       llvm::make_early_inc_range(funcOp.getOps<MemoryControllerOp>())) {
    bbs.clear();
    for (auto oprd : mc->getOperands()) {
      if (isa<ControlType>(oprd.getType()))
        continue;
      if (oprd.getDefiningOp()) {
        if (auto bbOrNull = getLogicBB(oprd.getDefiningOp())) {
          bbs.insert(bbOrNull.value());
        }
      }
    }
    for (auto res : mc->getResults()) {
      if (isa<ControlType>(res.getType()))
        continue;
      for (auto *user : res.getUsers()) {
        if (auto bbOrNull = getLogicBB(user)) {
          bbs.insert(bbOrNull.value());
        }
      }
    }
    auto i32Attr = builder.getI32ArrayAttr(llvm::to_vector(bbs));
    mc.setConnectedBlocksAttr(i32Attr);
  }
}

bool tryErasePasser(PasserOp passer) {
  Value result = passer.getResult();

  if (result.use_empty()) {
    passer->erase();
    return true;
  }
  assert(result.hasOneUse());
  Operation *user = getUniqueUser(result);
  if (isa<SinkOp>(user)) {
    user->erase();
    passer->erase();
    return true;
  }
  if (auto childPasser = dyn_cast<PasserOp>(user)) {
    if (tryErasePasser(childPasser)) {
      passer->erase();
      return true;
    }
  }
  return false;
}

std::optional<ControlMergeOp> getConfluencePoint(Value value) {
  Operation *user = getUniqueUser(value);
  if (auto cmerge = dyn_cast<ControlMergeOp>(user)) {
    if (cmerge.getNumOperands() == 1) {
      assert(isa<SinkOp>(getUniqueUser(cmerge.getIndex())));
      return getConfluencePoint(cmerge.getResult());
    }
    return cmerge;
  }
  if (auto branch = dyn_cast<BranchOp>(user)) {
    return getConfluencePoint(branch.getResult());
  }
  if (auto fork = dyn_cast<ForkOp>(user)) {
    for (auto res : fork->getResults()) {
      auto confluence = getConfluencePoint(res);
      if (confluence.has_value())
        return confluence;
    }
  }
  return std::nullopt;
}

void introduceGSAMux(FuncOp &funcOp, unsigned branchBB) {
  OpBuilder builder(funcOp->getContext());
  for (auto branchOp : funcOp.getOps<ConditionalBranchOp>()) {
    if (getLogicBB(branchOp) != branchBB)
      continue;
    if (!branchOp.getDataOperand().getType().isa<ControlType>())
      continue;

    Value trueValue = branchOp.getTrueResult();
    auto confluenceCMerge = getConfluencePoint(trueValue);
    assert(confluenceCMerge.has_value());
    if (getUniqueUser(trueValue) != *confluenceCMerge &&
        getLogicBB(getUniqueUser(trueValue)) !=
            getLogicBB(
                confluenceCMerge->getDataOperands()[1].getDefiningOp())) {
      llvm::report_fatal_error("Invalid structure");
    }

    Value condition = branchOp.getConditionOperand();

    builder.setInsertionPoint(confluenceCMerge.value());
    MuxOp newMux = builder.create<MuxOp>(
        builder.getUnknownLoc(), confluenceCMerge->getResult().getType(),
        condition,
        ArrayRef<Value>{confluenceCMerge->getDataOperands()[0],
                        confluenceCMerge->getDataOperands()[1]});
    inheritBB(confluenceCMerge.value(), newMux);

    confluenceCMerge->getResult().replaceAllUsesWith(newMux.getResult());
    confluenceCMerge->getIndex().replaceAllUsesWith(condition);
    confluenceCMerge->erase();

    materializeValue(condition);
  }
}
