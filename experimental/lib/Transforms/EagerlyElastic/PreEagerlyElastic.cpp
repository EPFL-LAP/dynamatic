#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

using namespace dynamatic;
using namespace mlir;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_PREEAGERLYELASTIC
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

struct PreEagerlyElasticPass
    : public dynamatic::experimental::impl::PreEagerlyElasticBase<
          PreEagerlyElasticPass> {
  using PreEagerlyElasticBase::PreEagerlyElasticBase;

  void runOnOperation() override;

private:
  SmallVector<handshake::ConditionalBranchOp> prepareSuppressors(handshake::FuncOp funcOp);

  bool isEligibleForSuppressorMotion(handshake::ConditionalBranchOp branchOp);

  SmallVector<handshake::ConditionalBranchOp> performSuppressorMotion(
    handshake::ConditionalBranchOp branchOp);
};

/// TODO: function description
SmallVector<handshake::ConditionalBranchOp>
PreEagerlyElasticPass::prepareSuppressors(handshake::FuncOp funcOp) {
  SmallVector<handshake::ConditionalBranchOp> branchesToErase;
  SmallVector<handshake::ConditionalBranchOp> suppressors;

  // iterate over all basic blocks
  for (Block &block : funcOp.getBody()) {
    // iterate over all operations inside the current bb
    for (Operation &op : block) {
      // check if the operation is a conditional branch
      auto branchOp = dyn_cast<handshake::ConditionalBranchOp>(&op);
      if (!branchOp || !branchOp->hasAttr("ftd.skip"))
        continue;

      // a suppressor has to eliminate the token on a true signal
      // if the true result has no uses it is a sink, as desired
      if (branchOp.getTrueResult().use_empty()) {
        suppressors.push_back(branchOp);
        continue;
      }

      OpBuilder builder(branchOp);
      builder.setInsertionPoint(branchOp); // TODO: necessary?

      // create inverted condition
      Value condition = branchOp.getConditionOperand();
      Location loc = branchOp.getLoc();
      auto invertedCondition =
          builder.create<handshake::NotIOp>(loc, condition);

      // add bb attribute
      auto bbAttr = branchOp->getAttr("handshake.bb");
      invertedCondition->setAttr(builder.getStringAttr("handshake.bb"), bbAttr);

      // the false result is a sink -> switch results with inverted condition
      if (branchOp.getFalseResult().use_empty()) {
        // update the branch condition and all downstream uses
        branchOp.getConditionOperandMutable().assign(invertedCondition);
        branchOp.getTrueResult().replaceAllUsesWith(branchOp.getFalseResult());

        suppressors.push_back(branchOp);
        continue;
      }

      // convert suppressors without sinks into two suppressors
      Value data = branchOp.getDataOperand();

      // create suppressor for true path with inverted condition
      auto suppressorA = builder.create<handshake::ConditionalBranchOp>(
          loc, invertedCondition, data);
      suppressorA->setAttr("ftd.skip", branchOp->getAttr("ftd.skip"));
      suppressorA->setAttr("handshake.bb", bbAttr);

      // create suppressor for false path with normal condition
      auto suppressorB =
          builder.create<handshake::ConditionalBranchOp>(loc, condition, data);
      suppressorB->setAttr("ftd.skip", branchOp->getAttr("ftd.skip"));
      suppressorB->setAttr("handshake.bb", bbAttr);

      // rewire the downstream consumers to point to our new split branches
      branchOp.getTrueResult().replaceAllUsesWith(suppressorA.getFalseResult());
      branchOp.getFalseResult().replaceAllUsesWith(
          suppressorB.getFalseResult());

      suppressors.push_back(suppressorA);
      suppressors.push_back(suppressorB);

      // erase the original combined branch
      branchesToErase.push_back(branchOp);
    }
  }
  // safely erase after loop is done
  for (handshake::ConditionalBranchOp deadOp : branchesToErase) {
    deadOp->erase();
  }

  return suppressors;
}

/// Checks if a suppressor (BranchOp) can be pushed past its downstream
/// operation in Rewrite A from the paper
bool PreEagerlyElasticPass::isEligibleForSuppressorMotion(
    handshake::ConditionalBranchOp branchOp) {
  // get the non-suppressed data path (false result)
  Value dataPath = branchOp.getFalseResult();

  // data path must have exactly one consumer to move past it cleanly
  if (!dataPath.hasOneUse())
    return false;

  Operation *targetOp = *dataPath.user_begin();

  // verify the targetOp is a PM unit
  // TODO: what about loads and LSQs?
  if (!isa<handshake::ArithOpInterface, handshake::NotIOp, handshake::ForkOp,
           handshake::LazyForkOp, handshake::BufferOp, handshake::LoadOp,
           handshake::BranchOp>(
          targetOp) ||
      ((isa<handshake::MergeOp, handshake::ControlMergeOp>(targetOp)) &&
       targetOp->getNumOperands() != 1)) {
    return false; // reject anything that isn't a PM unit or a 1-input merge
  }

  // ensure all other inputs to the target op match this suppressor's condition
  // because otherwise we cannot push the branch past the operation due to
  // synchronization issues
  Value currentCond = branchOp.getConditionOperand();
  for (Value operand : targetOp->getOperands()) {
    if (operand == dataPath)
      continue;

    if (auto siblingBranch = dyn_cast_or_null<handshake::ConditionalBranchOp>(
            operand.getDefiningOp())) {
      if (siblingBranch.getConditionOperand() != currentCond)
        return false;
    } else {
      return false;
    }
  }

  return true;
}

/// actually moves the suppressors TODO: better description
SmallVector<handshake::ConditionalBranchOp>
PreEagerlyElasticPass::performSuppressorMotion(
    handshake::ConditionalBranchOp branchOp) {

  // identify the operation we want to move past
  Value dataPath = branchOp.getFalseResult();
  Operation *targetOp = *dataPath.user_begin();

  OpBuilder builder(targetOp);
  Location loc = targetOp->getLoc();
  auto bbAttr = targetOp->getAttr("handshake.bb");

  // bypass this branch
  targetOp->replaceUsesOfWith(dataPath, branchOp.getDataOperand());

  // cleanup: save condition and delete the old branch
  Value condition = branchOp.getConditionOperand();
  branchOp->erase();

  // place the new suppressors after the targetOp
  builder.setInsertionPointAfter(targetOp);
  SmallVector<handshake::ConditionalBranchOp> newSuppressors;
  for (Value result : targetOp->getResults()) {
    auto newBranch =
        builder.create<handshake::ConditionalBranchOp>(loc, condition, result);

    // copy the attributes
    newBranch->setAttr(builder.getStringAttr("handshake.bb"), bbAttr);

    // reroute downstream consumers to look at the new branch's FalseResult
    result.replaceAllUsesExcept(newBranch.getFalseResult(), newBranch);
    newSuppressors.push_back(newBranch);
  }

  // return the newly created downstream branches
  return newSuppressors;
}

void PreEagerlyElasticPass::runOnOperation() {
  ModuleOp modOp = getOperation();

  handshake::FuncOp funcOp =
      cast<handshake::FuncOp>(&modOp.getBodyRegion().front().front());
  assert(funcOp && "No funcOp found!");

  // identify and prepare suppressors and return a list of all of them
  SmallVector<handshake::ConditionalBranchOp> suppressors = prepareSuppressors(funcOp);

  // Rewrite A:
  // Loop over all suppressors
  // Move them as far down as possible:
  // Stop at stores, loads with LSQs, and Muxes (anything else?)
  SmallVector<handshake::ConditionalBranchOp> frontier = suppressors;
  while (!frontier.empty()) {
    auto branchOp = frontier.pop_back_val();

    // if it cannot be moved further down, stop
    if (!isEligibleForSuppressorMotion(branchOp)) {
      llvm::errs() << "not eligible...\n";
      continue;
    }
    llvm::errs() << "eligible!\n";
    SmallVector<handshake::ConditionalBranchOp> newSuppressors =
        performSuppressorMotion(branchOp);

    // append new supps to frontier to be processed later
    frontier.append(newSuppressors.begin(), newSuppressors.end());
  }
}
