#include "dynamatic/Analysis/NameAnalysis.h"
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
#define GEN_PASS_DEF_EAGERLYELASTICAD
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

struct EagerlyElasticADPass
    : public dynamatic::experimental::impl::EagerlyElasticADBase<
          EagerlyElasticADPass> {
  using EagerlyElasticADBase::EagerlyElasticADBase;

  void runOnOperation() override;

private:
  template <typename... Args>
  void setupMetadata(Attribute bbAttr, NameAnalysis &namer, Args... ops);

  SmallVector<handshake::ConditionalBranchOp>
  prepareSuppressors(handshake::FuncOp funcOp, NameAnalysis &namer);

  bool isEligibleForSuppressorMotion(handshake::ConditionalBranchOp branchOp,
                                     Operation *targetOp);

  void
  performSuppressorMotion(handshake::ConditionalBranchOp branchOp,
                          DenseSet<handshake::ConditionalBranchOp> &frontier,
                          NameAnalysis &namer, int DRewrite = 0);

  Value getForkTop(Value value, bool &isInverted);
  bool isSourced(Value value);

  void rewriteA(DenseSet<handshake::ConditionalBranchOp> &frontier,
                NameAnalysis &namer);

  void applyRewriteD(handshake::MuxOp dataMux,
                     handshake::ConditionalBranchOp branchOp,
                     handshake::InitOp initOp,
                     DenseSet<handshake::ConditionalBranchOp> &frontier,
                     NameAnalysis &namer);

  void rewriteD(DenseSet<handshake::ConditionalBranchOp> &frontier,
                NameAnalysis &namer);
};

/// Helper function to add the bbAttr and name to new operations. You can either
/// pass one operation or multiple ones
template <typename... Args>
void EagerlyElasticADPass::setupMetadata(Attribute bbAttr, NameAnalysis &namer,
                                         Args... ops) {
  // Create a braced initializer list to unpack and process each operation
  (..., [&]() {
    if (ops) {
      if (bbAttr)
        ops->setAttr("handshake.bb", bbAttr);
      namer.setName(ops);
    }
  }());
}

/// Identifies conditional branches and converts them into suppressors which
/// eliminate their token on the True Result path and returns a vector
/// containing all suppressors.
/// 1. Branches whose True Result is a sink are not changed
/// 2. Branches whose False Result is a sink are inverted
/// 3. Branches without sinks are split into two suppressors using an inverted
/// and the normal condition respectively.
SmallVector<handshake::ConditionalBranchOp>
EagerlyElasticADPass::prepareSuppressors(handshake::FuncOp funcOp,
                                         NameAnalysis &namer) {
  // vector with the branches that have been replaced by new suppressors
  SmallVector<handshake::ConditionalBranchOp> branchesToErase;
  // final vector containing all suppressors of the function
  SmallVector<handshake::ConditionalBranchOp> suppressors;

  // iterate over all operations in the function to find the condbranchops
  for (auto branchOp : funcOp.getOps<handshake::ConditionalBranchOp>()) {

    // a suppressor has to eliminate the token on a true signal
    // if the true result has no uses it is a sink, as desired
    if (branchOp.getTrueResult().use_empty()) {
      suppressors.push_back(branchOp);
      continue;
    }

    OpBuilder builder(branchOp);
    auto bbAttr = branchOp->getAttr("handshake.bb");

    // create inverted condition
    Value condition = branchOp.getConditionOperand();
    Location loc = branchOp.getLoc();
    auto invertedCondition = builder.create<handshake::NotIOp>(loc, condition);
    setupMetadata(bbAttr, namer, invertedCondition);

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

    // create suppressors for true path with inverted condition and false path
    // with normal condition
    auto suppressorA = builder.create<handshake::ConditionalBranchOp>(
        loc, invertedCondition, data);
    auto suppressorB =
        builder.create<handshake::ConditionalBranchOp>(loc, condition, data);

    setupMetadata(bbAttr, namer, suppressorA, suppressorB);

    // rewire the downstream consumers to point to our new split branches
    branchOp.getTrueResult().replaceAllUsesWith(suppressorA.getFalseResult());
    branchOp.getFalseResult().replaceAllUsesWith(suppressorB.getFalseResult());

    suppressors.push_back(suppressorA);
    suppressors.push_back(suppressorB);

    // erase the original combined branch
    branchesToErase.push_back(branchOp);
  }
  // safely erase after loop is done
  for (handshake::ConditionalBranchOp deadOp : branchesToErase) {
    deadOp->erase();
  }

  return suppressors;
}

/// Recursively trace a value back to its source, traversing through
/// non-blocking structural ops such as forks, buffers, and logical inversions.
Value EagerlyElasticADPass::getForkTop(Value value, bool &isInverted) {
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

/// Determines whether a value originates from a constant source
bool EagerlyElasticADPass::isSourced(Value value) {
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
/// operation (for Rewrite A)
bool EagerlyElasticADPass::isEligibleForSuppressorMotion(
    handshake::ConditionalBranchOp branchOp, Operation *targetOp) {

  StringRef opName = targetOp->getName().getStringRef();

  // verify the targetOp is a PM unit
  if (!isa<handshake::ArithOpInterface, handshake::ForkOp, handshake::NotIOp,
           handshake::LazyForkOp, handshake::BufferOp, handshake::LoadOp,
           handshake::BranchOp>(targetOp) ||
      ((isa<handshake::MergeOp, handshake::ControlMergeOp>(targetOp)) &&
       targetOp->getNumOperands() != 1)) {

    llvm::errs() << opName << " is not a pm unit or a 1-input merge\n";
    return false; // reject anything that isn't a PM unit or a 1-input merge
  }

  // the other input to the loadOp is coming from the memory controller meaning
  // we can always move past a loadOp
  if (isa<handshake::LoadOp>(targetOp)) {
    return true;
  }

  // only move past NotOps if it's not part of a suppressor condition
  if (isa<handshake::NotIOp>(targetOp) &&
      llvm::any_of(targetOp->getResult(0).getUsers(), [](Operation *user) {
        return isa<handshake::ConditionalBranchOp>(user);
      })) {
    return false;
  }

  // ensure all other inputs to the target op match this suppressor's condition
  // because otherwise we cannot push the branch past the operation due to
  // synchronization issues
  Value currentCond = branchOp.getConditionOperand();
  for (Value operand : targetOp->getOperands()) {
    // if we have a branch it must have the same condition
    if (auto siblingBranch =
            dyn_cast<handshake::ConditionalBranchOp>(operand.getDefiningOp())) {
      // check whether condition matches indirectly
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

/// Moves the suppressors past the following operation.
void EagerlyElasticADPass::performSuppressorMotion(
    handshake::ConditionalBranchOp branchOp,
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer,
    int DRewrite) {

  // identify the operation we want to move past
  Value dataPath = branchOp.getFalseResult();
  Operation *targetOp = *dataPath.user_begin();

  // ensure that targetOp and the new branch are in the same bb as the old
  // branch. skip for load operations due to the memory controller
  auto bbAttr = targetOp->getAttr("handshake.bb");
  if (!isa<handshake::LoadOp>(targetOp) &&
      targetOp->getAttr("handshake.bb") != bbAttr) {
    targetOp->moveAfter(branchOp);
    bbAttr = branchOp->getAttr("handshake.bb");
    targetOp->setAttr("handshake.bb", bbAttr);
  }

  Location loc = targetOp->getLoc();
  Value condition = branchOp.getConditionOperand();

  // erase all old suppressors feeding into the target operation
  for (Value operand : targetOp->getOperands()) {
    if (Operation *defOp = operand.getDefiningOp()) {
      if (auto incomingBranch =
              dyn_cast<handshake::ConditionalBranchOp>(defOp)) {
        // for Rewrite D, move only the suppressor connected to A of the Mux
        if (incomingBranch != branchOp && DRewrite) {
          continue;
        }
        // erase the branch
        frontier.erase(incomingBranch);
        targetOp->replaceUsesOfWith(operand, incomingBranch.getDataOperand());
        incomingBranch->erase();
      }
    }
  }

  // place the new suppressors after the targetOp
  OpBuilder builder(targetOp);
  builder.setInsertionPointAfter(targetOp);

  // place the new suppressors on every result of the targetOp
  for (Value result : targetOp->getResults()) {
    // If this is a LoadOp, skip index 0 (the address output going to the memory
    // controller)
    if (isa<handshake::LoadOp>(targetOp) &&
        llvm::cast<mlir::OpResult>(result).getResultNumber() == 0) {
      continue;
    }

    auto newBranch =
        builder.create<handshake::ConditionalBranchOp>(loc, condition, result);
    setupMetadata(bbAttr, namer, newBranch);

    // reroute downstream consumers to look at the new branch's FalseResult
    result.replaceAllUsesExcept(newBranch.getFalseResult(), newBranch);
    frontier.insert(newBranch);
  }
}

/// Apply rewrite A as often as possible
/// This function loops over all conditional branches (the suppressors). For
/// every branch, it looks at the operations connected to its FalseResult to
/// check whether the branch can be moved past. Because moving these branches
/// adds and removes branches in our tracking set `frontier`, the function
/// restarts the loop after every change.
void EagerlyElasticADPass::rewriteA(
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer) {

  bool frontierUpdated;
  do {
    frontierUpdated = false;

    for (auto branchOp : frontier) {
      Value dataPath = branchOp.getFalseResult();
      Operation *eligibleTarget = nullptr;

      // search for any consumer of the suppressor that is eligible
      for (Operation *user : dataPath.getUsers()) {
        if (isEligibleForSuppressorMotion(branchOp, user)) {
          eligibleTarget = user;
          break;
        }
      }

      if (eligibleTarget) {
        llvm::errs() << branchOp->getAttr("handshake.name") << '\n';
        frontierUpdated = true;

        // if there are multiple uses, isolate the target by inserting a fork
        if (!dataPath.hasOneUse()) {
          OpBuilder builder(branchOp);
          builder.setInsertionPointAfter(branchOp);
          auto forkOp =
              builder.create<handshake::ForkOp>(branchOp.getLoc(), dataPath, 2);
          forkOp->setAttr("handshake.bb", branchOp->getAttr("handshake.bb"));
          namer.setName(forkOp);

          eligibleTarget->replaceUsesOfWith(dataPath, forkOp.getResults()[0]);
          dataPath.replaceAllUsesExcept(forkOp.getResults()[1], forkOp);
          llvm::errs() << "added fork\n";
          // break to move this branch past the fork in a later step
          break;
        }

        llvm::errs() << "is eligible\n";
        performSuppressorMotion(branchOp, frontier, namer);

        // frontier was mutated, break and restart the loop
        break;
      }
      llvm::errs() << "not eligible\n";
    }
  } while (frontierUpdated);
  llvm::errs() << "finish rewrite A!\n";
}

/// Perform RewriteD from the EagerlyElastic Paper
void EagerlyElasticADPass::applyRewriteD(
    handshake::MuxOp dataMux, handshake::ConditionalBranchOp branchOp,
    handshake::InitOp initOp,
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer) {

  Location loc = dataMux->getLoc();
  auto bbAttr = dataMux->getAttr("handshake.bb");
  llvm::errs() << dataMux->getAttr("handshake.name") << '\n';

  // find the condition signal C
  auto notOp = dyn_cast<handshake::NotIOp>(
      branchOp.getConditionOperand().getDefiningOp());
  if (!notOp)
    return;
  Value conditionC = notOp.getOperand();

  // build the top speculative loop control structure
  OpBuilder builder(dataMux);

  // completely replace the pink circuit with a RepeatingInitOp
  auto repeatingInit = builder.create<RepeatingInitOp>(loc, conditionC, 1);
  setupMetadata(bbAttr, namer, repeatingInit);
  Value specOutput = repeatingInit.getResult();

  // connect the repeatingInit to the loop init and the suppressor
  initOp.getOperandMutable().assign(specOutput);

  /*
  // create the constant true generator
  auto trueSrc = builder.create<handshake::SourceOp>(loc);
  auto trueCst = builder.create<handshake::ConstantOp>(
      loc, conditionC.getType(), builder.getBoolAttr(true),
      trueSrc.getResult());

  // create the new condition-generating mux
  // takes condition C on true and constant TRUE on false
  SmallVector<Value> condMuxInputs = {trueCst.getResult(), conditionC};
  auto condMux = builder.create<handshake::MuxOp>(loc, conditionC.getType(),
                                                  conditionC, condMuxInputs);

  // create the fork and wire output 0 back to the condition mux via an init
  // and output 1 to the original data mux via the already existing init
  auto controlFork =
      builder.create<handshake::ForkOp>(loc, condMux.getResult(), 3);
  auto loopInitF = builder.create<handshake::InitOp>(
      loc, controlFork.getResults()[0].getType(), controlFork.getResults()[0]);
  condMux.getSelectOperandMutable()[0].set(loopInitF.getResult());
  initOp.getOperandMutable().assign(controlFork.getResults()[1]);
  llvm::errs() << "fork generated\n";

  // assign the bb attribute from the dataMux to all new ops and name them
  setupMetadata(bbAttr, namer, trueSrc, trueCst, condMux, controlFork,
                loopInitF);
  */

  // rewire the NotIOp's input to the new control fork result
  notOp.getOperandMutable().assign(specOutput);

  // move suppressor past the mux
  performSuppressorMotion(branchOp, frontier, namer, 1);
  llvm::errs() << "successfully performed Rewrite D\n";
}

// Identify all branches before loop muxes and apply Rewrite D once
void EagerlyElasticADPass::rewriteD(
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer) {
  bool frontierUpdated;
  do {
    frontierUpdated = false;

    for (auto branchOp : frontier) {
      if (auto mux = dyn_cast<handshake::MuxOp>(
              *branchOp.getFalseResult().user_begin())) {

        // identify loops (mux connected to init) and make sure the suppressor
        // connected to the mux does not go anywhere else
        auto init =
            dyn_cast<handshake::InitOp>(mux.getSelectOperand().getDefiningOp());
        if (init && branchOp.getFalseResult().hasOneUse()) {
          llvm::errs() << "branchOp for rewrite D: "
                       << branchOp->getAttr("handshake.name") << '\n';

          // check whether the suppressor is connected to path A of the mux
          // and make sure there is no future fork in between
          if (mux.getDataOperands()[1] == branchOp.getFalseResult()) {
            applyRewriteD(mux, branchOp, init, frontier, namer);
          } else {
            llvm::errs()
                << "suppressor not connected to path A of the Mux !\n ";
            continue;
          }

          frontierUpdated = true;
          // frontier was mutated, break and restart the loop
          break;
        }
      }
    }
  } while (frontierUpdated);
}

void EagerlyElasticADPass::runOnOperation() {
  ModuleOp modOp = getOperation();
  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  handshake::FuncOp funcOp =
      cast<handshake::FuncOp>(&modOp.getBodyRegion().front().front());
  assert(funcOp && "No funcOp found!");

  // identify and prepare suppressors and return a list of all of them
  auto suppressors = prepareSuppressors(funcOp, namer);
  DenseSet<handshake::ConditionalBranchOp> frontier;
  frontier.insert(suppressors.begin(), suppressors.end());

  // apply rewrites: A, (D, A)^n
  rewriteA(frontier, namer);

  for (unsigned i = 0; i < numRewriteD; i++) {
    rewriteD(frontier, namer);
    rewriteA(frontier, namer);
  }

  /*   llvm::errs() << "=== IR AFTER REWRITES A & D ===\n";
    modOp.dump();
    llvm::errs() << "==============================================\n"; */
}
