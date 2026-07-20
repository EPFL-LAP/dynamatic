#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "experimental/Transforms/EagerlyElastic/EagerlyElasticLib.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

// NOTE: The code wrapped in LLVM_DEBUG(...) is executed when
// - Dynamatic is built in debug mode
// - dynamatic-opt is called with `--debug` or `--debug-only=<DEBUG_TYPE>`.
#define DEBUG_TYPE "eagerly-elastic"

using namespace dynamatic;
using namespace mlir;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_EAGERLYELASTICALL
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

enum class RewriteStrategy { RewriteB, RewriteC, RewriteE, RewriteG, RewriteH };

struct EagerlyElasticAllPass
    : public dynamatic::experimental::impl::EagerlyElasticAllBase<
          EagerlyElasticAllPass> {
  using EagerlyElasticAllBase::EagerlyElasticAllBase;

  void runOnOperation() override;

private:
  SmallVector<handshake::ConditionalBranchOp>
  prepareSuppressors(handshake::FuncOp funcOp, NameAnalysis &namer);

  void rewriteA(DenseSet<handshake::ConditionalBranchOp> &frontier,
                NameAnalysis &namer);

  void rewriteD(DenseSet<handshake::ConditionalBranchOp> &frontier,
                NameAnalysis &namer);

  void checkRewriteF(DenseSet<handshake::ConditionalBranchOp> &frontier,
                     NameAnalysis &namer);

  void applyMuxRewrites(DenseSet<handshake::ConditionalBranchOp> &frontier,
                        NameAnalysis &namer, RewriteStrategy rewrite);
};

/// Identifies conditional branches and converts them into suppressors which
/// eliminate their token on the True Result path and returns a vector
/// containing all suppressors.
/// 1. Branches whose True Result is a sink are not changed
/// 2. Branches whose False Result is a sink are inverted
/// 3. Branches without sinks are split into two suppressors using an inverted
/// and the normal condition respectively.
SmallVector<handshake::ConditionalBranchOp>
EagerlyElasticAllPass::prepareSuppressors(handshake::FuncOp funcOp,
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

    // make sure the original combined branch is erased later
    branchesToErase.push_back(branchOp);
  }

  // safely erase all obsolete conditional branches after loop is done
  for (handshake::ConditionalBranchOp deadOp : branchesToErase) {
    deadOp->erase();
  }

  return suppressors;
}

/// Apply rewrite A as often as possible
/// This function loops over all conditional branches (the suppressors). For
/// every branch, it looks at the operations connected to its FalseResult to
/// check whether the branch can be moved past. Because moving these branches
/// adds and removes branches in our tracking set `frontier`, the function
/// restarts the loop after every change.
void EagerlyElasticAllPass::rewriteA(
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
          // llvm::errs() << "eligible: " << user->getAttr("handshake.name") << '\n';
          eligibleTarget = user;
          break;
        }
      }

      if (eligibleTarget) {
        frontierUpdated = true;

        // if there are multiple uses, isolate the target by inserting a fork
        if (!dataPath.hasOneUse()) {
          OpBuilder builder(branchOp);
          builder.setInsertionPointAfter(branchOp);

           // create the fork
          auto forkOp =
              builder.create<handshake::ForkOp>(branchOp.getLoc(), dataPath, 2);
          setupMetadata(branchOp->getAttr("handshake.bb"), namer, forkOp);

          // separate the single target operand from all other downstream operands
          llvm::SmallVector<OpOperand *, 4> remainingUses;
          OpOperand *targetUse = nullptr;

          for (OpOperand &use : dataPath.getUses()) {
            if (use.getOwner() == forkOp) {
              continue;
            }
            if (use.getOwner() == eligibleTarget) {
              targetUse = &use;
            } else {
              remainingUses.push_back(&use);
            }
          }

          if (targetUse) {
            targetUse->set(forkOp.getResults()[0]);
          }

          // Connect #1 of the fork to all other remaining downstream users
          for (OpOperand *use : remainingUses) {
            use->set(forkOp.getResults()[1]);
          }

          // break to move this branch past the fork in a later step
          break;
        }
        // if the mux is the only consumer of the branch, skip
        if (auto mux = dyn_cast<handshake::MuxOp>(*dataPath.user_begin())) {
          frontierUpdated = false;
          continue;
        }

        performSuppressorMotion(branchOp, frontier, namer);

        // frontier was mutated, break and restart the loop
        break;
      }
    }
  } while (frontierUpdated);
}

// Identify all branches before loop muxes and apply Rewrite D on them.
void EagerlyElasticAllPass::rewriteD(
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer) {

  SmallVector<handshake::ConditionalBranchOp> initialFrontier(frontier.begin(),
                                                              frontier.end());
  for (auto branchOp : initialFrontier) {
    if (auto mux = dyn_cast<handshake::MuxOp>(*branchOp.getFalseResult().user_begin())) {
      if (llvm::isa<dynamatic::handshake::ControlType>(mux.getResult().getType()))
        return;
      // identify loops (mux connected to init)
      auto init =
          dyn_cast<handshake::InitOp>(mux.getSelectOperand().getDefiningOp());
      if (init) {
        // check whether the suppressor is connected to path A of the mux
        if (mux.getDataOperands()[1] == branchOp.getFalseResult()) {
          applyRewriteD(mux, branchOp, init, frontier, namer);
        } else
          continue;
      }
    }
  }
}

void EagerlyElasticAllPass::applyMuxRewrites(
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer,
    RewriteStrategy rewrite) {
  bool frontierUpdated;
  do {
    frontierUpdated = false;

    for (auto branchOp : frontier) {
      Value dataPath = branchOp.getFalseResult();

      // check whether the branch is connected to a mux
      auto mux = dyn_cast<handshake::MuxOp>(*dataPath.user_begin());
      if (!mux)
        continue;

        // check if the branch is connected to the true result of the mux
      if (mux.getDataOperands()[1] != dataPath) {
        if (rewrite == RewriteStrategy::RewriteE) {
          llvm::errs() << "check inverted rewriteE\n";
          // check whether a constant false is connected
          auto cnstFalse = dyn_cast_or_null<handshake::ConstantOp>(mux.getDataOperands()[1].getDefiningOp());
          if (!cnstFalse) continue;
          auto boolAttr = dyn_cast_or_null<BoolAttr>(cnstFalse.getValue());
          if (!boolAttr || boolAttr.getValue() != false) continue;

          // check whether the branch and the mux have the same condition
          bool muxCondInverted = false;
          bool branchCondInverted = false;
          Value muxCondSrc = getForkTop(mux.getSelectOperand(), muxCondInverted);
          Value branchCondSrc =
              getForkTop(branchOp.getConditionOperand(), branchCondInverted);

          if (muxCondInverted == branchCondInverted && muxCondSrc == branchCondSrc) {
            applyRewriteE(mux, branchOp, frontier, namer, 1);
            frontierUpdated = true;
            break;
          }
        }
        continue;
      }

      llvm::errs() << "checking rewrites for: " << mux->getAttr("handshake.name") << '\n';

      // check whether the necessary operations for the different rewrites exist
      auto falseBranch = dyn_cast_or_null<handshake::ConditionalBranchOp>(
          mux.getDataOperands()[0].getDefiningOp());
      auto initOp =
          dyn_cast_or_null<handshake::InitOp>(mux.getSelectOperand().getDefiningOp());

      // check for rewrite H
      if (falseBranch && initOp && rewrite == RewriteStrategy::RewriteH) {
        // Both suppressors must share the same control source 'C'
        bool leftCondInverted = false;
        bool rightCondInverted = false;
        Value leftCondSrc =
            getForkTop(branchOp.getConditionOperand(), leftCondInverted);
        Value rightCondSrc =
            getForkTop(falseBranch.getConditionOperand(), rightCondInverted);

        // Verify they originate from the same control signal and share the same
        // inversion
        if (leftCondSrc == rightCondSrc &&
            leftCondInverted == rightCondInverted) {
        }

        auto notOp =
            dyn_cast<handshake::NotIOp>(initOp.getOperand().getDefiningOp());
        if (notOp && notOp.getOperand().getDefiningOp() == initOp) {
          applyRewriteH(mux, branchOp, initOp, frontier, namer);
          frontierUpdated = true;
          break;
        }
      }

      // trace both conditions back to their source, tracking logical
      // inversions
      bool muxCondInverted = false;
      bool branchCondInverted = false;
      Value muxCondSrc = getForkTop(mux.getSelectOperand(), muxCondInverted);
      Value branchCondSrc =
          getForkTop(branchOp.getConditionOperand(), branchCondInverted);

      // verify the conditions match but are logically inverted
      if (muxCondSrc != branchCondSrc ||
          (muxCondInverted == branchCondInverted)) {
        llvm::errs() << "condition not matching / logically inverted\n";
        continue;
      }
      
      llvm::errs() << "start looking at rewrite B and G\n";
      if (rewrite == RewriteStrategy::RewriteB ||
          rewrite == RewriteStrategy::RewriteG) {
        
        if (!falseBranch)
          continue;
        llvm::errs() << "two supps above the mux, apply rewrite B/G\n";
        
        // check that falsebranch condition = mux condition
        bool falseBranchCondInverted = false;
        Value falseBranchCondSrc = getForkTop(falseBranch.getConditionOperand(),
                                              falseBranchCondInverted);

        if (falseBranchCondSrc != muxCondSrc ||
            (falseBranchCondInverted != muxCondInverted)) {
          continue;
        }

        if (rewrite == RewriteStrategy::RewriteG) {
          auto topSuppressorA = dyn_cast<handshake::ConditionalBranchOp>(
              branchOp.getDataOperand().getDefiningOp());
          auto topSuppressorB = dyn_cast<handshake::ConditionalBranchOp>(
              falseBranch.getDataOperand().getDefiningOp());
          auto topSuppressorC = dyn_cast<handshake::ConditionalBranchOp>(
              mux.getSelectOperand().getDefiningOp());

          // all lines need to have a suppressor
          if (!topSuppressorA || !topSuppressorB || !topSuppressorC)
            continue;

          bool condAInverted = false, condBInverted = false,
               condCInverted = false;

          Value condSrcA =
              getForkTop(topSuppressorA.getConditionOperand(), condAInverted);
          Value condSrcB =
              getForkTop(topSuppressorB.getConditionOperand(), condBInverted);
          Value condSrcC =
              getForkTop(topSuppressorC.getConditionOperand(), condCInverted);

          // verify all three trace back to the same source
          if (condSrcA == condSrcB && condSrcB == condSrcC &&
              condAInverted == condBInverted &&
              condBInverted == condCInverted) {
            // applyRewriteG(mux, );
            frontierUpdated = true;
            break;
          }
        }

        applyRewriteB(mux, branchOp, falseBranch, frontier, namer);
        frontierUpdated = true;
        break; // Break the inner loop to restart with the updated frontier

      } else if (rewrite == RewriteStrategy::RewriteE) {
        // check whether a constant false is connected
        llvm::errs() << "rewriteE\n";
        auto cnstFalse = dyn_cast_or_null<handshake::ConstantOp>(mux.getDataOperands()[0].getDefiningOp());
        if (!cnstFalse) continue;
        auto boolAttr = dyn_cast_or_null<BoolAttr>(cnstFalse.getValue());
        if (!boolAttr || boolAttr.getValue() != false) continue;
        llvm::errs() << "start applyRewriteE\n";
        applyRewriteE(mux, branchOp, frontier, namer);
        frontierUpdated = true;
        break;
      } else if (rewrite == RewriteStrategy::RewriteC) {
        applyRewriteC(mux, branchOp, frontier, namer);
        frontierUpdated = true;
        break; // Break the inner loop to restart with the updated frontier
      }
    }
  } while (frontierUpdated);
}

void EagerlyElasticAllPass::checkRewriteF(
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer) {
  bool frontierUpdated;
  do {
    frontierUpdated = false;
    for (auto branchOp : frontier) {
      auto topSuppLeft = dyn_cast<handshake::ConditionalBranchOp>(
          branchOp.getDataOperand().getDefiningOp());
      if (!topSuppLeft)
        continue;

      // find the condition operand of the branch
      Value conditionC = branchOp.getConditionOperand();
      auto notOp = dyn_cast<handshake::NotIOp>(conditionC.getDefiningOp());
      if (notOp) {
        conditionC = notOp.getOperand();
      }

      auto topSuppRight =
          dyn_cast<handshake::ConditionalBranchOp>(conditionC.getDefiningOp());
      if (!topSuppRight)
        continue;

      // verify that the top suppressors are controlled by the same condition
      bool condLeftInverted = false;
      bool condRightInverted = false;
      Value conditionBLeft =
          getForkTop(topSuppLeft.getConditionOperand(), condLeftInverted);
      Value conditionBRight =
          getForkTop(topSuppRight.getConditionOperand(), condRightInverted);

      // they must share the same source value and have matching polarities
      if (conditionBLeft != conditionBRight ||
          condLeftInverted != condRightInverted)
        continue;

      applyRewriteF(branchOp, topSuppLeft, topSuppRight, frontier, namer);
      frontierUpdated = true;
      break;
    }
  } while (frontierUpdated);
}

void EagerlyElasticAllPass::runOnOperation() {
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
  applyMuxRewrites(frontier, namer, RewriteStrategy::RewriteE);
  llvm::errs() << "rewrite E done\n";
  // checkRewriteF(frontier, namer);

  for (unsigned i = 0; i < 1; i++) {
    llvm::errs() << "start rewrite D\n";
    rewriteD(frontier, namer);
    applyMuxRewrites(frontier, namer, RewriteStrategy::RewriteG);
    rewriteA(frontier, namer);
  }

  for (unsigned i = 0; i<1; i++) {
    applyMuxRewrites(frontier, namer, RewriteStrategy::RewriteB);
    applyMuxRewrites(frontier, namer, RewriteStrategy::RewriteG);
    rewriteA(frontier, namer);
  }
}
