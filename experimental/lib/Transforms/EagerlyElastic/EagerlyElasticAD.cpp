#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

// NOTE: The code wrapped in LLVM_DEBUG(...) is executed when
// - Dynamatic is built in debug mode
// - dynamatic-opt is called with `--debug` or `--debug-only=<DEBUG_TYPE>`.
#define DEBUG_TYPE "eagerly-elastic"

using namespace dynamatic;
using namespace mlir;

enum class BypassResult : bool { Ineligible = false, Eligible = true };

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
  void setHandshakeAttrs(Attribute bbAttr, NameAnalysis &namer,
                         ArrayRef<Operation *> ops);

  DenseSet<handshake::ConditionalBranchOp>
  prepareSuppressors(handshake::FuncOp funcOp, NameAnalysis &namer);

  BypassResult isEligibleForBypass(handshake::ConditionalBranchOp branchOp,
                                   Operation *targetOp);

  void moveSuppressorPastOp(handshake::ConditionalBranchOp branchOp,
                            Operation *targetOp,
                            DenseSet<handshake::ConditionalBranchOp> &frontier,
                            NameAnalysis &namer, int DRewrite = 0);

  bool checkConditionsMatch(Value valA, Value valB, bool expectSamePolarity);
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

/// Helper function to add the bbAttr and name to new operations.
void EagerlyElasticADPass::setHandshakeAttrs(Attribute bbAttr,
                                             NameAnalysis &namer,
                                             ArrayRef<Operation *> ops) {
  for (Operation *op : ops) {
    assert(op);
    if (bbAttr)
      op->setAttr("handshake.bb", bbAttr);
    namer.setName(op);
  }
}

/// Identifies conditional branches and converts them into suppressors which
/// eliminate their token on the True Result path and returns a vector
/// containing all suppressors.
/// 1. Branches whose True Result is a sink are not changed
/// 2. Branches whose False Result is a sink are inverted
/// 3. Branches without sinks are split into two suppressors using an inverted
/// and the normal condition respectively.
DenseSet<handshake::ConditionalBranchOp>
EagerlyElasticADPass::prepareSuppressors(handshake::FuncOp funcOp,
                                         NameAnalysis &namer) {

  // final vector containing all suppressors of the function
  DenseSet<handshake::ConditionalBranchOp> suppressors;

  // iterate over all operations in the function to find the condbranchops
  for (auto branchOp : funcOp.getOps<handshake::ConditionalBranchOp>()) {
    suppressors.insert(branchOp);

    // a suppressor has to eliminate the token on a true signal
    // if the true result has no uses it is a sink, as desired
    if (branchOp.getTrueResult().use_empty()) {
      continue;
    }

    OpBuilder builder(branchOp);
    auto bbAttr = branchOp->getAttr("handshake.bb");

    // create inverted condition
    Value condition = branchOp.getConditionOperand();
    Location loc = branchOp.getLoc();
    auto invertedCondition = builder.create<handshake::NotIOp>(loc, condition);
    // Tag the NotIOp so we know it belongs to a suppressor condition
    invertedCondition->setAttr("is_suppressor_not", builder.getUnitAttr());
    setHandshakeAttrs(bbAttr, namer, {invertedCondition});

    // the false result is a sink -> switch results with inverted condition
    if (branchOp.getFalseResult().use_empty()) {
      // update the branch condition and all downstream uses
      branchOp.getConditionOperandMutable().assign(invertedCondition);
      branchOp.getTrueResult().replaceAllUsesWith(branchOp.getFalseResult());
      continue;
    }

    // convert suppressors without sinks into two suppressors
    Value data = branchOp.getDataOperand();

    // create a suppressor for true path with inverted condition
    auto suppressorInverted = builder.create<handshake::ConditionalBranchOp>(
        loc, invertedCondition, data);
    setHandshakeAttrs(bbAttr, namer, {suppressorInverted});

    // rewire the true path downstream consumers to the inverted suppressor
    branchOp.getTrueResult().replaceAllUsesWith(
        suppressorInverted.getFalseResult());
    suppressors.insert(suppressorInverted);
  }
  return suppressors;
}

/// Checks whether two condition values originate from the same root source.
bool EagerlyElasticADPass::checkConditionsMatch(Value valA, Value valB,
                                                bool expectSamePolarity) {
  bool invA = false, invB = false;

  // Trace valA back through NotIOps
  while (auto notOp =
             dyn_cast_or_null<handshake::NotIOp>(valA.getDefiningOp())) {
    invA = !invA;
    valA = notOp.getOperand();
  }

  // Trace valB back through NotIOps
  while (auto notOp =
             dyn_cast_or_null<handshake::NotIOp>(valB.getDefiningOp())) {
    invB = !invB;
    valB = notOp.getOperand();
  }

  // They must originate from the exact same root wire
  if (valA != valB)
    return false;

  return expectSamePolarity ? (invA == invB) : (invA != invB);
}

/// Recursive function to determine whether a value originates from a constant
/// source.
bool EagerlyElasticADPass::isSourced(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return false;

  // No constant source possible here
  if (isa<handshake::MuxOp>(definingOp) or
      isa<handshake::RepeatingInitOp>(definingOp))
    return false;

  if (isa<handshake::SourceOp>(value.getDefiningOp()))
    return true;

  // If all operands of the defining operation are sourced, the value is also
  // sourced.
  return llvm::all_of(value.getDefiningOp()->getOperands(),
                      [this](Value v) { return isSourced(v); });
}

/// Checks if a suppressor (BranchOp) can be pushed past its downstream
/// operation (for Rewrite A). Eligible operations are pure and matched and need
/// to have the same condition on all their input operands or a source.
BypassResult EagerlyElasticADPass::isEligibleForBypass(
    handshake::ConditionalBranchOp branchOp, Operation *targetOp) {

  // verify the targetOp is a PM unit
  if (!isa<handshake::ArithOpInterface, handshake::ForkOp, handshake::NotIOp,
           handshake::LazyForkOp, handshake::BufferOp, handshake::LoadOp,
           handshake::BranchOp>(targetOp) ||
      ((isa<handshake::MergeOp, handshake::ControlMergeOp>(targetOp)) &&
       targetOp->getNumOperands() != 1)) {
    return BypassResult::Ineligible; // reject anything that isn't a PM unit or
                                     // a 1-input merge
  }

  // loadOps receive address control independently, the can always be bypassed
  if (isa<handshake::LoadOp>(targetOp)) {
    return BypassResult::Eligible;
  }

  // only move past NotOps if they aren't part of a suppressor condition
  if (targetOp->hasAttr("is_suppressor_not")) {
    return BypassResult::Ineligible;
  }

  // ensure all other inputs to the target op match this suppressor's condition
  Value currentCond = branchOp.getConditionOperand();
  for (Value operand : targetOp->getOperands()) {
    // if we have a branch it must have the same condition
    if (auto siblingBranch = dyn_cast_or_null<handshake::ConditionalBranchOp>(
            operand.getDefiningOp())) {
      if (siblingBranch == branchOp)
        continue;
      // if the branch has more than one use, a fork needs to be created first
      if (!siblingBranch.getFalseResult().hasOneUse()) {
        return BypassResult::Ineligible;
      }
      // check whether condition matches indirectly
      if (!checkConditionsMatch(currentCond,
                                siblingBranch.getConditionOperand(), true)) {
        return BypassResult::Ineligible;
      }
    } else if (!isSourced(operand)) {
      return BypassResult::Ineligible;
    }
  }
  return BypassResult::Eligible;
}

/// Move the suppressors past the following operation, targetOp. Erase all the
/// suppressors going into targetOp and create new suppressors on every output
/// of targetOp.
void EagerlyElasticADPass::moveSuppressorPastOp(
    handshake::ConditionalBranchOp branchOp, Operation *targetOp,
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer,
    int DRewrite) {

  llvm::errs() << "Moving past: " << targetOp->getAttr("handshake.name")
               << '\n';

  Location loc = targetOp->getLoc();
  Value condition = branchOp.getConditionOperand();

  // rewire the suppressor
  for (OpOperand &use : targetOp->getOpOperands()) {
    auto incomingBranch = dyn_cast_or_null<handshake::ConditionalBranchOp>(
        use.get().getDefiningOp());
    if (!incomingBranch)
      continue;

    // for Rewrite D, move only the suppressor connected to A of the Mux
    if (incomingBranch != branchOp && DRewrite) {
      continue;
    }

    // rewire the target operand directly to the suppressor's input data
    use.set(incomingBranch.getDataOperand());

    // check if the suppressor has any remaining downstream uses
    if (incomingBranch.getFalseResult().use_empty()) {
      frontier.erase(incomingBranch);
      incomingBranch->erase();
    }
  }

  OpBuilder builder(targetOp);
  builder.setInsertionPointAfter(targetOp);

  // place the new suppressors on every result of the targetOp
  for (Value result : targetOp->getResults()) {
    // If this is a LoadOp, skip index 0 (the address output going to the MC)
    if (isa<handshake::LoadOp>(targetOp) &&
        llvm::cast<mlir::OpResult>(result).getResultNumber() == 0) {
      continue;
    }

    auto newBranch =
        builder.create<handshake::ConditionalBranchOp>(loc, condition, result);
    setHandshakeAttrs(targetOp->getAttr("handshake.bb"), namer, {newBranch});

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
        if (isEligibleForBypass(branchOp, user) == BypassResult::Eligible) {
          eligibleTarget = user;
          break;
        }
      }

      if (eligibleTarget) {
        frontierUpdated = true;
        moveSuppressorPastOp(branchOp, eligibleTarget, frontier, namer);
        // frontier was mutated, break and restart the loop
        break;
      }
    }
  } while (frontierUpdated);
}

/// Apply Rewrite D once by connecting a repeatingInitOp to the circuit and then
/// moving the suppressor past the mux.
void EagerlyElasticADPass::applyRewriteD(
    handshake::MuxOp dataMux, handshake::ConditionalBranchOp branchOp,
    handshake::InitOp initOp,
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer) {

  Location loc = dataMux->getLoc();
  auto bbAttr = dataMux->getAttr("handshake.bb");
  llvm::errs() << "Perform Rewrite D for: "
               << dataMux->getAttr("handshake.name") << '\n';
  Value conditionC = initOp.getOperand();

  // build the additional control structure for the rewrite
  OpBuilder builder(dataMux);

  auto repeatingInit =
      builder.create<handshake::RepeatingInitOp>(loc, conditionC, 1);
  setHandshakeAttrs(bbAttr, namer, {repeatingInit});
  Value specOutput = repeatingInit.getResult();

  // connect the repeatingInit to the loop init
  initOp.getOperandMutable().assign(specOutput);

  // update the suppressor's condition - must be inverted relative to the init
  auto notOp = dyn_cast_or_null<handshake::NotIOp>(
      branchOp.getConditionOperand().getDefiningOp());
  if (notOp && notOp.getResult().hasOneUse())
    notOp.getOperandMutable().assign(specOutput);
  else { // create a new isolated NotIOp
    builder.setInsertionPoint(notOp ? notOp : branchOp);
    auto newNotOp =
        builder.create<handshake::NotIOp>(branchOp.getLoc(), specOutput);
    setHandshakeAttrs(bbAttr, namer, {newNotOp});
    newNotOp->setAttr("is_suppressor_not", builder.getUnitAttr());
    branchOp.getConditionOperandMutable().assign(newNotOp.getResult());
  }

  // move suppressor past the mux
  moveSuppressorPastOp(branchOp, dataMux, frontier, namer, 1);
}

// Identify all branches before loop muxes and apply Rewrite D on them.
void EagerlyElasticADPass::rewriteD(
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer) {

  SmallVector<handshake::ConditionalBranchOp> initialFrontier(frontier.begin(),
                                                              frontier.end());
  for (auto branchOp : initialFrontier) {
    for (auto *nextOp : branchOp.getFalseResult().getUsers()) {
      if (auto mux = dyn_cast<handshake::MuxOp>(nextOp)) {
        // identify loops (mux connected to init)
        auto init =
            dyn_cast_or_null<handshake::InitOp>(mux.getSelectOperand().getDefiningOp());

        if (init) {
          if (!checkConditionsMatch(branchOp.getConditionOperand(),
                                    init.getOperand(), false)) {
            continue;
          }
          // check whether the suppressor is connected to path A of the mux
          if (mux.getDataOperands()[1] == branchOp.getFalseResult()) {
            applyRewriteD(mux, branchOp, init, frontier, namer);
            // break;
          } else
            continue;
        }
      }
    }
  }
}

void EagerlyElasticADPass::runOnOperation() {
  ModuleOp modOp = getOperation();
  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  handshake::FuncOp funcOp =
      cast<handshake::FuncOp>(&modOp.getBodyRegion().front().front());
  assert(funcOp && "No funcOp found!");

  // identify and prepare suppressors and return a list of all of them
  auto frontier = prepareSuppressors(funcOp, namer);

  // apply rewrites: A, (D, A)^n
  rewriteA(frontier, namer);

  for (unsigned i = 0; i < 1; i++) {
    rewriteD(frontier, namer);
    rewriteA(frontier, namer);
  }
}
