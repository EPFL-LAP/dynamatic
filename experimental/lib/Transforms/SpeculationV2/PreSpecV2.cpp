#include "PreSpecV2.h"
#include "JSONImporter.h"
#include "MaterializationUtil.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
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

namespace dynamatic {
namespace experimental {
namespace speculationv2 {

// Implement the base class and auto-generated create functions.
// Must be called from the .cpp file to avoid multiple definitions
#define GEN_PASS_DEF_PRESPECV2
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct PreSpecV2Pass
    : public dynamatic::experimental::speculationv2::impl::PreSpecV2Base<
          PreSpecV2Pass> {
  using PreSpecV2Base<PreSpecV2Pass>::PreSpecV2Base;
  void runDynamaticPass() override;
};

static bool hasBranch(FuncOp &funcOp, unsigned bb) {
  for (auto branch : funcOp.getOps<ConditionalBranchOp>()) {
    auto brBB = getLogicBB(branch);
    if (brBB && *brBB == bb)
      return true;
  }
  return false;
}

/// Replaces all branches in the specified BB with passers, and returns the
/// passer control values for trueValue and falseValue.
/// Potentially can be applied to branches inside loops (i.e. PMSC), or to those
/// not at the loop's bottom in cases with multiple loop exits.
static LogicalResult replaceBranchesWithPassers(FuncOp &funcOp, unsigned bb) {
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

    // Build a passer for the trueResult
    PasserOp trueResultPasser =
        builder.create<PasserOp>(branch.getLoc(), data, condition);
    setBB(trueResultPasser, bb);
    branch.getTrueResult().replaceAllUsesWith(trueResultPasser.getResult());

    // Build a passer for the falseResult
    // The passer ctrl is inverted condition.
    PasserOp falseResultPasser = builder.create<PasserOp>(
        branch.getLoc(), data, invertCondition.getResult());
    setBB(falseResultPasser, bb);
    branch.getFalseResult().replaceAllUsesWith(falseResultPasser.getResult());

    // Erase the branch
    branch->erase();
  }

  return success();
}

/// Calculate the loop condition fed by Init ops.
/// Currently the implementation is based on the BBs specified in the json, and
/// the internal control flow is unsupported.
/// TODO: integrate this with the GSA implementation for fast token delivery.
static Value calculateLoopCondition(FuncOp &funcOp, ArrayRef<unsigned> bbs) {
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPoint(funcOp.getBodyBlock(),
                            funcOp.getBodyBlock()->begin());

  unsigned headBB = bbs[0];
  Value condition = nullptr;
  for (unsigned bb : bbs) {
    auto passers = funcOp.getOps<PasserOp>();
    auto passer = llvm::find_if(passers, [&](PasserOp passer) {
      if (getLogicBB(passer) != bb)
        return false;

      // Use the polarity of the passer connected inside the loop
      auto outputBBOrNull = getLogicBB(getUniqueUser(passer.getResult()));
      if (!outputBBOrNull.has_value()) {
        // Connected to outside the loop.
        return false;
      }
      return llvm::find(bbs, outputBBOrNull.value()) != bbs.end();
    });
    if (passer != passers.end()) {
      // Add the condition to loop conditions
      if (condition == nullptr) {
        // Simply use the condition
        condition = (*passer).getCtrl();
      } else {
        // TODO: consider the basic block
        SourceOp src = builder.create<SourceOp>(builder.getUnknownLoc());
        setBB(src, headBB);
        ConstantOp cst = builder.create<ConstantOp>(
            builder.getUnknownLoc(),
            IntegerAttr::get(builder.getIntegerType(1), 0), src);
        setBB(cst, headBB);
        MuxOp mux = builder.create<MuxOp>(
            builder.getUnknownLoc(), condition.getType(), condition,
            ArrayRef<Value>{cst.getResult(), (*passer).getCtrl()});
        setBB(mux, headBB);
        condition = mux.getResult();
      }
    } else {
      llvm::errs() << "didn't find passer for bb " << bb << "\n";
    }
  }

  return condition;
}

/// Replace the CMerge-controlled loop header with Init[False]-controlled one.
static LogicalResult updateLoopHeader(FuncOp &funcOp, ArrayRef<unsigned> bbs,
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
  Operation *definingOp1 = cmergeOp.getDataOperands()[1].getDefiningOp();
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

  // Build an InitOp[False] for each MuxOp
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

void PreSpecV2Pass::runDynamaticPass() {
  // Parse json (jsonPath is a member variable handled by tablegen)
  auto bbOrFailure = readFromJSON(jsonPath);
  if (failed(bbOrFailure))
    return signalPassFailure();

  auto [loopBBs] = bbOrFailure.value();

  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  // Replace branches with passers
  for (unsigned exitBB : loopBBs) {
    if (!hasBranch(funcOp, exitBB))
      continue;

    if (failed(replaceBranchesWithPassers(funcOp, exitBB)))
      return signalPassFailure();
  }

  Value loopCondition = calculateLoopCondition(funcOp, loopBBs);

  // Update the loop header (CMerge -> Init)
  if (failed(updateLoopHeader(funcOp, loopBBs, loopCondition)))
    return signalPassFailure();
}
