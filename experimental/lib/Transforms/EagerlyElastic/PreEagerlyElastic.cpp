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
  SmallVector<handshake::ConditionalBranchOp>
  prepareSuppressors(handshake::FuncOp funcOp);

  bool isEligibleForSuppressorMotion(handshake::ConditionalBranchOp branchOp);

  void
  performSuppressorMotion(handshake::ConditionalBranchOp branchOp,
                          DenseSet<handshake::ConditionalBranchOp> &frontier);

  Value getForkTop(Value value, bool &isInverted);
  bool isSourced(Value value);
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
      if (!branchOp)
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
      suppressorA->setAttr("handshake.bb", bbAttr);

      // create suppressor for false path with normal condition
      auto suppressorB =
          builder.create<handshake::ConditionalBranchOp>(loc, condition, data);
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

// TODO: write description
Value PreEagerlyElasticPass::getForkTop(Value value, bool &isInverted) {
  Operation *defOp = value.getDefiningOp();
  // look through logical inversions
  if (auto notOp = dyn_cast<handshake::NotIOp>(defOp)) {
    isInverted = !isInverted;
    return getForkTop(notOp.getOperand(), isInverted);
  }
  // look through standard handshake forks
  if (auto fork = dyn_cast<handshake::ForkOp>(defOp)) {
    return getForkTop(fork.getOperand(), isInverted);
  }
  // look through handshake buffers
  if (auto buf = dyn_cast<handshake::BufferOp>(defOp)) {
    return getForkTop(buf.getOperand(), isInverted);
  }
  return value;
}

bool PreEagerlyElasticPass::isSourced(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return false;

  // Heuristic to stop the traversal earlier.
  if (isa<handshake::MuxOp>(definingOp))
    return false;

  if (isa<handshake::SourceOp>(value.getDefiningOp()))
    return true;

  // If all operands of the defining operation are sourced, the value is also
  // sourced.
  return llvm::all_of(value.getDefiningOp()->getOperands(),
                      [this](Value v) { return isSourced(v); });
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
  if (!isa<handshake::ArithOpInterface, handshake::NotIOp, handshake::ForkOp,
           handshake::LazyForkOp, handshake::BufferOp, handshake::LoadOp,
           handshake::BranchOp>(targetOp) ||
      ((isa<handshake::MergeOp, handshake::ControlMergeOp>(targetOp)) &&
       targetOp->getNumOperands() != 1)) {
    return false; // reject anything that isn't a PM unit or a 1-input merge
  }

  // ensure all other inputs to the target op match this suppressor's condition
  // because otherwise we cannot push the branch past the operation due to
  // synchronization issues
  Value currentCond = branchOp.getConditionOperand();
  for (Value operand : targetOp->getOperands()) {
    if (auto siblingBranch =
            dyn_cast<handshake::ConditionalBranchOp>(operand.getDefiningOp())) {
      // Check condition matching indirectly (accounting for
      // forks/buffer/notops)
      bool currentInverted = false, siblingInverted = false;
      Value currentRoot = getForkTop(currentCond, currentInverted);
      Value siblingRoot =
          getForkTop(siblingBranch.getConditionOperand(), siblingInverted);

      // They must originate from the same wire AND have the exact same polarity
      if (currentRoot != siblingRoot || currentInverted != siblingInverted) {
        llvm::errs() << "Passer ctrl mismatch\n";
        llvm::errs() << "For branchop: " << branchOp->getAttr("handshake.name")
                     << '\n';
        return false;
      }

    } else if (!isSourced(operand)) {
      llvm::errs() << "Operand not from passer or source\n";
      return false;
    }
  }

  return true;
}

/// actually moves the suppressors TODO: better description
void PreEagerlyElasticPass::performSuppressorMotion(
    handshake::ConditionalBranchOp branchOp,
    DenseSet<handshake::ConditionalBranchOp> &frontier) {

  // identify the operation we want to move past
  Value dataPath = branchOp.getFalseResult();
  Operation *targetOp = *dataPath.user_begin();

  Location loc = targetOp->getLoc();
  auto bbAttr = targetOp->getAttr("handshake.bb");
  Value condition = branchOp.getConditionOperand();

  // erase all old suppressors feeding into the target operation
  for (Value operand : targetOp->getOperands()) {
    if (auto incomingBranch =
            dyn_cast<handshake::ConditionalBranchOp>(operand.getDefiningOp())) {
      frontier.erase(incomingBranch);
      targetOp->replaceUsesOfWith(operand, incomingBranch.getDataOperand());
      incomingBranch->erase();
    }
  }

  // place the new suppressors after the targetOp
  OpBuilder builder(targetOp);
  builder.setInsertionPointAfter(targetOp); // necessary?

  // place the new suppressors after the targetOp
  for (Value result : targetOp->getResults()) {
    auto newBranch =
        builder.create<handshake::ConditionalBranchOp>(loc, condition, result);

    // copy the bb attribute
    newBranch->setAttr(builder.getStringAttr("handshake.bb"), bbAttr);

    // reroute downstream consumers to look at the new branch's FalseResult
    result.replaceAllUsesExcept(newBranch.getFalseResult(), newBranch);
    frontier.insert(newBranch);
  }
}

void PreEagerlyElasticPass::runOnOperation() {
  ModuleOp modOp = getOperation();

  handshake::FuncOp funcOp =
      cast<handshake::FuncOp>(&modOp.getBodyRegion().front().front());
  assert(funcOp && "No funcOp found!");

  // identify and prepare suppressors and return a list of all of them
  llvm::errs() << "start prepareSuppressors\n";
  SmallVector<handshake::ConditionalBranchOp> suppressors =
      prepareSuppressors(funcOp);
  llvm::errs() << "end prepareSuppressors\n";

  // Rewrite A
  DenseSet<handshake::ConditionalBranchOp> frontier;
  frontier.insert(suppressors.begin(), suppressors.end());

  bool frontierUpdated;
  do {
    frontierUpdated = false;

    for (auto branchOp : frontier) {
      if (isEligibleForSuppressorMotion(branchOp)) {
        llvm::errs() << "is eligible\n";

        performSuppressorMotion(branchOp, frontier);

        frontierUpdated = true;
        // frontier was mutated, break and restart the loop
        break;
      }
      llvm::errs() << "not eligible\n";
    }
  } while (frontierUpdated);
}
