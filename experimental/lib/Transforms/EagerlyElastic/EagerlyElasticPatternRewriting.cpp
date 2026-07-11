#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// NOTE: The code wrapped in LLVM_DEBUG(...) is executed when
// - Dynamatic is built in debug mode
// - dynamatic-opt is called with `--debug` or `--debug-only=<DEBUG_TYPE>`.
#define DEBUG_TYPE "eagerly-elastic-pattern-rewriter"

using namespace dynamatic;
using namespace mlir;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_EAGERLYELASTICPATTERNREWRITING
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

struct EagerlyElasticPatternRewritingPass
    : public dynamatic::experimental::impl::EagerlyElasticPatternRewritingBase<
          EagerlyElasticPatternRewritingPass> {
  using EagerlyElasticPatternRewritingBase::EagerlyElasticPatternRewritingBase;

  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

template <typename... Args>
void setupMetadata(Attribute bbAttr, NameAnalysis &namer, Args... ops) {
  (..., [&]() {
    if (ops) {
      if (bbAttr)
        ops->setAttr("handshake.bb", bbAttr);
      namer.setName(ops);
    }
  }());
}

Value getForkTop(Value value, bool &isInverted) {
  Operation *defOp = value.getDefiningOp();
  if (auto notOp = dyn_cast_or_null<handshake::NotIOp>(defOp)) {
    isInverted = !isInverted;
    return getForkTop(notOp.getOperand(), isInverted);
  }
  if (auto fork = dyn_cast_or_null<handshake::ForkOp>(defOp)) {
    return getForkTop(fork.getOperand(), isInverted);
  }
  if (auto buf = dyn_cast_or_null<handshake::BufferOp>(defOp)) {
    return getForkTop(buf.getOperand(), isInverted);
  }
  return value;
}

bool isSourced(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return false;

  if (isa<handshake::MuxOp>(definingOp) ||
      isa<handshake::RepeatingInitOp>(definingOp))
    return false;

  if (isa<handshake::SourceOp>(definingOp))
    return true;

  return llvm::all_of(definingOp->getOperands(),
                      [](Value v) { return isSourced(v); });
}

/// Identifies conditional branches and converts them into suppressors which
/// eliminate their token on the True Result path.
/// 1. Branches whose True Result is a sink are not changed
/// 2. Branches whose False Result is a sink are inverted
/// 3. Branches without sinks are split into two suppressors using an inverted
/// and the normal condition respectively.
void prepareSuppressors(handshake::FuncOp funcOp, NameAnalysis &namer) {
  // vector with the branches that have been replaced by new suppressors
  SmallVector<handshake::ConditionalBranchOp> branchesToErase;

  // iterate over all operations in the function to find the condbranchops
  for (auto branchOp : funcOp.getOps<handshake::ConditionalBranchOp>()) {

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
    setupMetadata(bbAttr, namer, invertedCondition);

    // the false result is a sink -> switch results with inverted condition
    if (branchOp.getFalseResult().use_empty()) {
      // update the branch condition and all downstream uses
      branchOp.getConditionOperandMutable().assign(invertedCondition);
      branchOp.getTrueResult().replaceAllUsesWith(branchOp.getFalseResult());
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

    // make sure the original combined branch is erased later
    branchesToErase.push_back(branchOp);
  }

  // safely erase all obsolete conditional branches after loop is done
  for (handshake::ConditionalBranchOp deadOp : branchesToErase)
    deadOp->erase();
}

/// Checks if a suppressor (BranchOp) can be pushed past its downstream
/// operation (for Rewrite A). Eligible operations are pure and matched and need
/// to have the same condition on all their input operands or a source.
bool isEligibleForSuppressorMotion(handshake::ConditionalBranchOp branchOp,
                                   Operation *targetOp) {

  // can be pushed past a Mux but not during rewrite A - there is an additional
  // check in the rewriteA() function
  if (isa<handshake::MuxOp>(targetOp))
    return true;

  // verify the targetOp is a PM unit
  if (!isa<handshake::ArithOpInterface, handshake::ForkOp, handshake::NotIOp,
           handshake::LazyForkOp, handshake::BufferOp, handshake::LoadOp,
           handshake::BranchOp>(targetOp) ||
      ((isa<handshake::MergeOp, handshake::ControlMergeOp>(targetOp)) &&
       targetOp->getNumOperands() != 1)) {
    return false; // reject anything that isn't a PM unit or a 1-input merge
  }

  // the other input to the loadOp is coming from the memory controller meaning
  // we can always move past a loadOp
  if (isa<handshake::LoadOp>(targetOp)) {
    return true;
  }

  // only move past NotOps if it's not part of a suppressor condition
  if (isa<handshake::NotIOp>(targetOp)) {
    auto users = targetOp->getResult(0).getUsers();
    // If there are no users, return false
    if (users.empty())
      return false;

    // check if any of the users match the specific operations
    for (Operation *user : users) {
      if (llvm::isa<handshake::ConditionalBranchOp, handshake::RepeatingInitOp>(
              user)) {
        return false;
      }
    }
  }

  // ensure all other inputs to the target op match this suppressor's condition
  Value currentCond = branchOp.getConditionOperand();
  for (Value operand : targetOp->getOperands()) {
    // if we have a branch it must have the same condition
    if (auto siblingBranch =
            dyn_cast<handshake::ConditionalBranchOp>(operand.getDefiningOp())) {
      // if the branch has more than one use, a fork needs to be created first
      if (!siblingBranch.getFalseResult().hasOneUse()) {
        return false;
      }
      // check whether condition matches indirectly
      bool currentInverted = false, siblingInverted = false;
      Value currentRoot = getForkTop(currentCond, currentInverted);
      Value siblingRoot =
          getForkTop(siblingBranch.getConditionOperand(), siblingInverted);

      // They must originate from the same wire AND have the exact same polarity
      if (currentRoot != siblingRoot || currentInverted != siblingInverted) {
        return false;
      }
    } else if (!isSourced(operand)) {
      return false;
    }
  }
  return true;
}

struct RewriteAPattern
    : public OpRewritePattern<handshake::ConditionalBranchOp> {
  NameAnalysis &namer;
  RewriteAPattern(MLIRContext *ctx, NameAnalysis &namer)
      : OpRewritePattern<handshake::ConditionalBranchOp>(ctx), namer(namer) {}

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp branchOp,
                                PatternRewriter &rewriter) const override {
    Value dataPath = branchOp.getFalseResult();
    Operation *eligibleTarget = nullptr;

    // search for any consumer of the suppressor that is eligible
    for (Operation *user : dataPath.getUsers()) {
      if (isEligibleForSuppressorMotion(branchOp, user)) {
        eligibleTarget = user;
        break;
      }
    }

    if (!eligibleTarget)
      return failure();

    llvm::errs() << "eligible Target: ";
    eligibleTarget->dump();

    // if there are multiple uses, isolate the target by inserting a fork
    if (!dataPath.hasOneUse()) {
      rewriter.setInsertionPointAfter(branchOp);
      llvm::SmallVector<OpOperand *> usesToReplace;
      for (OpOperand &use : dataPath.getUses())
        usesToReplace.push_back(&use);

      unsigned numUses = usesToReplace.size();
      auto forkOp = rewriter.create<handshake::ForkOp>(branchOp.getLoc(),
                                                       dataPath, numUses);
      setupMetadata(branchOp->getAttr("handshake.bb"), namer, forkOp);

      for (unsigned i = 0; i < numUses; ++i) {
        rewriter.updateRootInPlace(usesToReplace[i]->getOwner(), [&]() {
          usesToReplace[i]->set(forkOp.getResults()[i]);
        });
      }
      return success();
    }

    // if the mux is the only consumer of the branch, skip
    if (auto mux = dyn_cast<handshake::MuxOp>(*dataPath.user_begin())) {
      return failure();
    }

    // ensure that targetOp and the new branch are in the same bb as the old
    // branch. skip for load operations due to the memory controller
    auto bbAttr = eligibleTarget->getAttr("handshake.bb");
    if (!isa<handshake::LoadOp>(eligibleTarget) &&
        branchOp->getAttr("handshake.bb") != bbAttr) {
      // Move target operation to follow right behind the current branch
      eligibleTarget->moveAfter(branchOp);
      // Re-tag the basic block attribute safely
      bbAttr = branchOp->getAttr("handshake.bb");
      rewriter.updateRootInPlace(eligibleTarget, [&]() {
        eligibleTarget->setAttr("handshake.bb", bbAttr);
      });
    }

    llvm::errs() << "bbattr changed to: " << bbAttr << '\n';

    Location loc = eligibleTarget->getLoc();
    Value condition = branchOp.getConditionOperand();

    // erase all old suppressors feeding into eligibletarget
    for (Value operand : eligibleTarget->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        if (auto incomingBranch =
                dyn_cast<handshake::ConditionalBranchOp>(defOp)) {
          rewriter.updateRootInPlace(eligibleTarget, [&]() {
            eligibleTarget->replaceUsesOfWith(operand,
                                              incomingBranch.getDataOperand());
          });
          rewriter.eraseOp(incomingBranch);
        }
      }
    }

    rewriter.setInsertionPointAfter(eligibleTarget);

    // create new replacement suppressors downstream of eligibletarget
    for (Value result : eligibleTarget->getResults()) {
      // If this is a LoadOp, skip index 0 (the address output going to the MC)
      if (isa<handshake::LoadOp>(eligibleTarget) &&
          cast<OpResult>(result).getResultNumber() == 0)
        continue;

      // new branches created exactly after eligibleTarget
      auto newBranch = rewriter.create<handshake::ConditionalBranchOp>(
          loc, condition, result);
      setupMetadata(bbAttr, namer, newBranch);
      newBranch->setAttr("eagerly.suppressor", rewriter.getUnitAttr());

      rewriter.replaceAllUsesExcept(result, newBranch.getFalseResult(),
                                    newBranch);
    }
    return success();
  }
};

void EagerlyElasticPatternRewritingPass::runOnOperation() {
  ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  handshake::FuncOp funcOp =
      cast<handshake::FuncOp>(&modOp.getBodyRegion().front().front());
  assert(funcOp && "No funcOp found!");

  // prepare suppressors
  prepareSuppressors(funcOp, namer);

  RewritePatternSet patternA(ctx);
  patternA.add<RewriteAPattern>(ctx, namer);

  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patternA))))
    return signalPassFailure();
}
