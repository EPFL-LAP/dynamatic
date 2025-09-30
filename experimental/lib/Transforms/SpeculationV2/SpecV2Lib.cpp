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
#include "llvm/Support/raw_ostream.h"
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

bool isEligibleForPasserMotionOverPM(PasserOp passerOp, bool reason = false) {
  Value passerControl = passerOp.getCtrl();

  Operation *targetOp = getUniqueUser(passerOp.getResult());

  // If the targetOp is not a PM unit, return false.
  if (!isa<ArithOpInterface, NotOp, ForkOp, LazyForkOp, BufferOp, LoadOp,
           BranchOp>(targetOp)) {
    if (!isa<MergeOp, ControlMergeOp>(targetOp) ||
        targetOp->getNumOperands() != 1) {
      if (reason)
        llvm::errs() << "Target op is not a PM unit\n";
      return false;
    }
  }

  // Iterate over operands of the targetOp to decide the eligibility for
  // motion.
  for (Value operand : getEffectiveOperands(targetOp)) {
    if (auto passerOp = dyn_cast<PasserOp>(operand.getDefiningOp())) {
      // If this passerOp is controlled by different control from the specified
      // one, not eligible.
      if (!equalsIndirectly(passerControl, passerOp.getCtrl())) {
        if (reason)
          llvm::errs() << "Passer ctrl mismatch\n";
        return false;
      }
    } else if (!isSourced(operand)) {
      // Each operand must be defined by a passer, except when it is driven by a
      // source op.
      if (reason)
        llvm::errs() << "Operand not from passer or source\n";
      return false;
    }
  }

  return true;
}

void performPasserMotionPastPM(PasserOp passerOp,
                               DenseSet<PasserOp> &frontiers) {
  Value passerControl = getForkTop(passerOp.getCtrl());
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
    newPasser->setAttr("specv2_frontier", builder.getBoolAttr(false));
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

bool hasBranch(FuncOp &funcOp, unsigned bb) {
  for (auto branch : funcOp.getOps<ConditionalBranchOp>()) {
    auto brBB = getLogicBB(branch);
    if (brBB && *brBB == bb)
      return true;
  }
  return false;
}

bool isInsideLoop(Value value, ArrayRef<unsigned> loopBBs) {
  // Might not be materialized
  Operation *user = *value.getUsers().begin();
  while (isa<ExtSIOp, TruncIOp, ForkOp>(user)) {
    user = *user->getUsers().begin();
  }
  auto outputBBOrNull = getLogicBB(user);
  if (!outputBBOrNull.has_value()) {
    // Connected to outside the loop.
    return false;
  }
  return llvm::find(loopBBs, outputBBOrNull.value()) != loopBBs.end();
}

Value calculateLoopCondition(FuncOp &funcOp, ArrayRef<unsigned> exitBBs,
                             ArrayRef<unsigned> loopBBs) {
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPoint(funcOp.getBodyBlock(),
                            funcOp.getBodyBlock()->begin());

  Value condition = nullptr;
  for (size_t i = 0; i < exitBBs.size(); i++) {
    unsigned bb = exitBBs[i];
    auto passers = funcOp.getOps<PasserOp>();
    auto passer = llvm::find_if(passers, [&](PasserOp passer) {
      if (getLogicBB(passer) != bb)
        return false;

      // Use the polarity of the passer connected inside the loop
      return isInsideLoop(passer.getResult(), loopBBs);
    });
    if (passer != passers.end()) {
      // Add the condition to loop conditions
      if (condition == nullptr) {
        // Simply use the condition
        condition = (*passer).getCtrl();
      } else {
        // TODO: consider the basic block
        SourceOp src = builder.create<SourceOp>(builder.getUnknownLoc());
        setBB(src, bb);
        ConstantOp cst = builder.create<ConstantOp>(
            builder.getUnknownLoc(),
            IntegerAttr::get(builder.getIntegerType(1), 0), src);
        setBB(cst, bb);
        MuxOp mux = builder.create<MuxOp>(
            builder.getUnknownLoc(), condition.getType(), condition,
            ArrayRef<Value>{cst.getResult(), (*passer).getCtrl()});
        setBB(mux, bb);
        mux->setAttr("specv2_loop_cond_mux", builder.getBoolAttr(true));
        condition = mux.getResult();
      }
    } else {
      llvm::errs() << "didn't find passer for bb " << bb << "\n";
      llvm_unreachable("");
    }
  }

  return condition;
}

LogicalResult updateLoopHeader(FuncOp &funcOp, ArrayRef<unsigned> bbs,
                               Value loopCondition) {
  OpBuilder builder(funcOp->getContext());
  builder.setInsertionPoint(funcOp.getBodyBlock(),
                            funcOp.getBodyBlock()->begin());
  unsigned headBB = bbs[0];

  // Find control merge in the loop head BB.
  ControlMergeOp cmergeOp = nullptr;
  for (auto cmergeCandidate :
       llvm::make_early_inc_range(funcOp.getOps<handshake::ControlMergeOp>())) {
    if (getLogicBB(cmergeCandidate) != headBB)
      continue;
    if (cmergeCandidate->hasAttr("specv1_adaptor_inner_loop"))
      continue;

    if (cmergeOp)
      return funcOp.emitError(
          "Multiple ControlMergeOps found in the loop head BB");

    cmergeOp = cmergeCandidate;
  }

  // Only support basic blocks with two predecessors.
  assert(cmergeOp.getDataOperands().size() == 2 &&
         "The loop head BB must have exactly two predecessors (you can run "
         "gate binarization pass)");

  // The backedge must be the second operand. If it is the first operand, we
  // need to swap operands.
  bool needsSwapping;
  Operation *definingOp0 = cmergeOp.getDataOperands()[0].getDefiningOp();
  if (definingOp0) {
    if (auto nonspec = dyn_cast<NonSpecOp>(definingOp0)) {
      definingOp0 = nonspec.getOperand().getDefiningOp();
    }
  }
  Operation *definingOp1 = cmergeOp.getDataOperands()[1].getDefiningOp();
  if (definingOp1) {
    if (auto nonspec = dyn_cast<NonSpecOp>(definingOp1)) {
      definingOp1 = nonspec.getOperand().getDefiningOp();
    }
  }
  Value entry, backedge;
  if (definingOp0 && llvm::find(bbs, getLogicBB(definingOp0)) != bbs.end()) {
    needsSwapping = true;
    entry = cmergeOp.getDataOperands()[1];
    backedge = cmergeOp.getDataOperands()[0];
  } else if (definingOp1 &&
             llvm::find(bbs, getLogicBB(definingOp1)) != bbs.end()) {
    needsSwapping = false;
    entry = cmergeOp.getDataOperands()[0];
    backedge = cmergeOp.getDataOperands()[1];
  } else {
    return cmergeOp.emitError(
        "Expected one of the operands to be defined in the loop tail BB");
  }

  // Before replacing CMerge with Mux, update existing Muxes
  for (auto muxOp : funcOp.getOps<handshake::MuxOp>()) {
    if (getLogicBB(muxOp) != headBB)
      continue;
    if (muxOp->hasAttr("specv2_loop_cond_mux"))
      continue;
    if (muxOp->hasAttr("specv1_adaptor_inner_loop"))
      continue;

    assert(muxOp.getDataOperands().size() == 2);

    // Build an InitOp[False] for each MuxOp
    builder.setInsertionPoint(muxOp);
    InitOp initOp =
        builder.create<InitOp>(loopCondition.getLoc(), loopCondition, 0);
    setBB(initOp, headBB);

    // Update the select operand
    muxOp.getSelectOperandMutable()[0].set(initOp.getResult());

    if (needsSwapping) {
      // Swap operands
      Value entry = muxOp.getDataOperands()[1];
      muxOp.getDataOperandsMutable()[1].set(muxOp.getDataOperands()[0]);
      muxOp.getDataOperandsMutable()[0].set(entry);
    }
  }

  // Build an InitOp[False] for CMerge-replacing Mux
  InitOp initOp =
      builder.create<InitOp>(loopCondition.getLoc(), loopCondition, 0);
  setBB(initOp, headBB);

  // Build a MuxOp to replace the CMergeOp
  // Use the result of the init as the selector.
  builder.setInsertionPoint(cmergeOp);
  MuxOp muxOp = builder.create<MuxOp>(
      cmergeOp.getLoc(), cmergeOp.getResult().getType(),
      /*selector=*/initOp.getResult(), llvm::ArrayRef{entry, backedge});
  setBB(muxOp, headBB);
  cmergeOp.getResult().replaceAllUsesWith(muxOp.getResult());

  // Erase CMerge (and possibly connected fork)
  eraseMaterializedOperation(cmergeOp);

  return success();
}

Operation *getEffectiveUser(Value value) {
  Operation *user = getUniqueUser(value);
  if (isa<ExtSIOp, TruncIOp, ForkOp>(user)) {
    return getEffectiveUser(user->getResult(0));
  }
  return user;
}

bool isExitingBBWithBranch(FuncOp funcOp, unsigned bb,
                           ArrayRef<unsigned> loopBBs) {
  for (auto branch : funcOp.getOps<ConditionalBranchOp>()) {
    auto brBB = getLogicBB(branch);
    if (brBB && *brBB == bb) {
      auto trueBranchBB = getLogicBB(getEffectiveUser(branch.getTrueResult()));
      auto falseBranchBB =
          getLogicBB(getEffectiveUser(branch.getFalseResult()));
      if (!trueBranchBB.has_value() || !falseBranchBB.has_value())
        return true;
      return llvm::find(loopBBs, trueBranchBB.value()) == loopBBs.end() ||
             llvm::find(loopBBs, falseBranchBB.value()) == loopBBs.end();
    }
  }
  // No branch
  return false;
}
