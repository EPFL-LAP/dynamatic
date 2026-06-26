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

  void rewriteD_suppOnA(handshake::MuxOp dataMux,
                        handshake::ConditionalBranchOp branchOp,
                        DenseSet<handshake::ConditionalBranchOp> &frontier);
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
    if (Operation *defOp = operand.getDefiningOp()) {
      if (auto incomingBranch =
              dyn_cast<handshake::ConditionalBranchOp>(defOp)) {
        frontier.erase(incomingBranch);
        targetOp->replaceUsesOfWith(operand, incomingBranch.getDataOperand());
        incomingBranch->erase();
      }
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

void PreEagerlyElasticPass::rewriteD_suppOnA(
    handshake::MuxOp dataMux, handshake::ConditionalBranchOp branchOp,
    DenseSet<handshake::ConditionalBranchOp> &frontier) {

  Location loc = dataMux->getLoc();
  auto bbAttr = dataMux->getAttr("handshake.bb");
  llvm::errs() << dataMux->getAttr("handshake.name") << '\n';

  // find the condition signal C
  Value conditionC = branchOp.getConditionOperand();
  if (auto notOp = dyn_cast<handshake::NotIOp>(conditionC.getDefiningOp())) {
    // if it was converted, we want it before the Not
    // should we always go into this if?
    conditionC = notOp.getOperand();
  }

  // find the original InitOp already attached to the Mux's select operand
  Value originalSelect = dataMux.getSelectOperand();
  auto existingInitOp =
      dyn_cast<handshake::InitOp>(originalSelect.getDefiningOp());
  if (!existingInitOp) {
    // this should never happen?
    llvm::errs() << "Error: Expected an existing InitOp attached to the Mux "
                    "select lines.\n";
    return;
  }

  // build the top speculative loop control structure
  OpBuilder builder(dataMux);
  builder.setInsertionPoint(dataMux); // is this done automatically?

  // create the constant true generator
  auto trueSrc = builder.create<handshake::SourceOp>(loc);
  auto trueCst = builder.create<handshake::ConstantOp>(
      loc, conditionC.getType(), builder.getBoolAttr(true),
      trueSrc.getResult());
  llvm::errs() << "constant true generator generated\n";

  // create the new condition-generating mux
  // takes condition C on true and constant TRUE on false
  SmallVector<Value> condMuxInputs = {conditionC, trueCst.getResult()};
  auto condMux = builder.create<handshake::MuxOp>(loc, conditionC.getType(),
                                                  conditionC, condMuxInputs);
  llvm::errs() << "mux generation successful\n";

  // create the fork and wire output 0 back to the condition mmux via an init
  // and output 1 to the original data mux via the already existing init
  auto controlFork =
      builder.create<handshake::ForkOp>(loc, condMux.getResult(), 3);
  auto loopInitF = builder.create<handshake::InitOp>(
      loc, controlFork.getResults()[0].getType(), controlFork.getResults()[0]);
  condMux.getSelectOperandMutable()[0].set(loopInitF.getResult());
  existingInitOp.getOperandMutable().assign(controlFork.getResults()[1]);
  llvm::errs() << "fork generated\n";

  // assign the basic block attributes from the dataMux to all other ops
  trueSrc->setAttr("handshake.bb", bbAttr);
  trueCst->setAttr("handshake.bb", bbAttr);
  condMux->setAttr("handshake.bb", bbAttr);
  controlFork->setAttr("handshake.bb", bbAttr);
  loopInitF->setAttr("handshake.bb", bbAttr);

  // rewire the NotIOp's input to the new control fork result
  Value branchCond = branchOp.getConditionOperand();
  if (auto notOp = dyn_cast<handshake::NotIOp>(branchCond.getDefiningOp())) {
    notOp.getOperandMutable().assign(controlFork.getResults()[2]);
  } else {
    llvm::errs() << "There was no NOT connected to the suppressor\n";
  }
  // move suppressor past the mux
  performSuppressorMotion(branchOp, frontier);
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

  llvm::errs() << "finish rewrite A!\n";

  /* llvm::errs() << "=== IR AFTER REWRITE A ===\n";
  modOp.dump();
  llvm::errs() << "==============================================\n"; */

  // Do Rewrite D exactly once on the first MuxOp you find
  // this needs to be fixed, as rewrite D can only be done on loop entries
  for (auto branchOp : frontier) {
    if (auto mux = dyn_cast<handshake::MuxOp>(
            *branchOp.getFalseResult().user_begin())) {
      llvm::errs() << "branchOp for rewrite D: "
                   << branchOp->getAttr("handshake.name") << '\n';
      // check which path the suppressor is connected to
      Value pathA = mux.getDataOperands()[0];
      Value pathB = mux.getDataOperands()[1];
      if (pathA == branchOp.getFalseResult()) {
        llvm::errs() << "supp connected to path A\n";
        rewriteD_suppOnA(mux, branchOp, frontier);
      } else if (pathB == branchOp.getFalseResult()) {
        llvm::errs() << "supp connected to path B\n";
        // rewriteD_suppOnB(mux, branchOp, frontier);
      }
      break;
    }
  }
}
