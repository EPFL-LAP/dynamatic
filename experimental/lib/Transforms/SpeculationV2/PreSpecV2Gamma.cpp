#include "PreSpecV2Gamma.h"
#include "JSONImporter.h"
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

namespace dynamatic {
namespace experimental {
namespace speculationv2 {

// Implement the base class and auto-generated create functions.
// Must be called from the .cpp file to avoid multiple definitions
#define GEN_PASS_DEF_PRESPECV2GAMMA
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct PreSpecV2GammaPass
    : public dynamatic::experimental::speculationv2::impl::PreSpecV2GammaBase<
          PreSpecV2GammaPass> {
  using PreSpecV2GammaBase<PreSpecV2GammaPass>::PreSpecV2GammaBase;
  void runDynamaticPass() override;
};

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

static void introduceGSA(FuncOp &funcOp, unsigned branchBB) {
  OpBuilder builder(funcOp->getContext());
  for (auto branchOp : funcOp.getOps<ConditionalBranchOp>()) {
    if (getLogicBB(branchOp) != branchBB)
      continue;
    if (!branchOp.getDataOperand().getType().isa<ControlType>())
      continue;

    ControlMergeOp trueCMerge =
        cast<ControlMergeOp>(getUniqueUser(branchOp.getTrueResult()));
    ControlMergeOp falseCMerge =
        cast<ControlMergeOp>(getUniqueUser(branchOp.getFalseResult()));

    assert(isa<SinkOp>(getUniqueUser(trueCMerge.getIndex())));
    assert(isa<SinkOp>(getUniqueUser(falseCMerge.getIndex())));

    BranchOp trueBranch;
    BranchOp falseBranch;
    for (Operation *user :
         iterateOverPossiblyIndirectUsers(trueCMerge.getResult())) {
      if (auto br = dyn_cast<BranchOp>(user)) {
        trueBranch = br;
        break;
      }
    }
    for (Operation *user :
         iterateOverPossiblyIndirectUsers(falseCMerge.getResult())) {
      if (auto br = dyn_cast<BranchOp>(user)) {
        falseBranch = br;
        break;
      }
    }
    assert(trueBranch && falseBranch);

    ControlMergeOp confluenceCMerge =
        cast<ControlMergeOp>(getUniqueUser(trueBranch.getResult()));
    assert(confluenceCMerge.getDataOperands()[0] == falseBranch.getResult());
    assert(confluenceCMerge.getDataOperands()[1] == trueBranch.getResult());

    Value condition = branchOp.getConditionOperand();

    builder.setInsertionPoint(confluenceCMerge);
    MuxOp newMux = builder.create<MuxOp>(
        builder.getUnknownLoc(), confluenceCMerge.getResult().getType(),
        condition,
        ArrayRef<Value>{confluenceCMerge.getDataOperands()[0],
                        confluenceCMerge.getDataOperands()[1]});
    inheritBB(confluenceCMerge, newMux);

    confluenceCMerge.getResult().replaceAllUsesWith(newMux.getResult());
    confluenceCMerge.getIndex().replaceAllUsesWith(condition);
    confluenceCMerge->erase();
  }
}

void PreSpecV2GammaPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  introduceGSA(funcOp, branchBB);

  if (failed(replaceBranchesWithPassers(funcOp, branchBB))) {
    funcOp.emitError("Failed to replace branches in BB 1 with passers");
    return signalPassFailure();
  }
}
