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

enum class RewriteStrategy {
  RewriteA,
  RewriteB,
  RewriteC,
  RewriteD,
  RewriteE,
  RewriteF,
  RewriteG,
  RewriteH
};

struct EagerlyElasticAllPass
    : public dynamatic::experimental::impl::EagerlyElasticAllBase<
          EagerlyElasticAllPass> {
  using EagerlyElasticAllBase::EagerlyElasticAllBase;

  void runOnOperation() override;

private:
  DenseSet<handshake::ConditionalBranchOp>
  prepareSuppressors(handshake::FuncOp funcOp, NameAnalysis &namer);

  void applyMuxRewrites(DenseSet<handshake::ConditionalBranchOp> &frontier,
                        NameAnalysis &namer, RewriteStrategy rewrite);

  void applyRewriteXAsOftenAsPossible(
      DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer,
      RewriteStrategy rewrite);

  void applyRewriteXOnce(DenseSet<handshake::ConditionalBranchOp> &frontier,
                         NameAnalysis &namer, RewriteStrategy rewrite);

  void movePastFunctionBlock(
      DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &name,
      ModuleOp modOp);
};

/// Identifies conditional branches and converts them into suppressors which
/// eliminate their token on the True Result path and returns a vector
/// containing all suppressors.
/// 1. Branches whose True Result is a sink are not changed
/// 2. Branches whose False Result is a sink are inverted
/// 3. Branches without sinks are split into two suppressors using an inverted
/// and the normal condition respectively.
DenseSet<handshake::ConditionalBranchOp>
EagerlyElasticAllPass::prepareSuppressors(handshake::FuncOp funcOp,
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

/// Apply Rewrites A, E, F, G, H
void EagerlyElasticAllPass::applyRewriteXAsOftenAsPossible(
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer,
    RewriteStrategy rewrite) {
  bool frontierUpdated;
  do {
    frontierUpdated = false;

    for (auto branchOp : frontier) {
      for (auto *user : branchOp.getFalseResult().getUsers()) {
        auto mux = dyn_cast<handshake::MuxOp>(user);
        switch (rewrite) {
        case RewriteStrategy::RewriteA: {
          if (isEligibleForBypass(branchOp, user) == BypassResult::Ineligible) {
            continue;
          }
          frontierUpdated = true;
          moveSuppressorPastOp(branchOp, user, frontier, namer);
          break;
        }

        case RewriteStrategy::RewriteE: {
          if (!mux)
            continue;

          bool inverted = false;
          Value muxFalse;
          if (mux.getDataOperands()[0] == branchOp.getFalseResult()) {
            muxFalse = mux.getDataOperands()[1];
            inverted = true;
          } else {
            muxFalse = mux.getDataOperands()[0];
          }

          auto cnstFalse =
              dyn_cast_or_null<handshake::ConstantOp>(muxFalse.getDefiningOp());
          if (!cnstFalse)
            continue;
          auto boolAttr = dyn_cast_or_null<BoolAttr>(cnstFalse.getValue());
          if (!boolAttr || boolAttr.getValue() != false)
            continue;

          if (!checkConditionsMatch(mux.getSelectOperand(),
                                    branchOp.getConditionOperand(), inverted))
            continue;

          applyRewriteE(mux, branchOp, frontier, namer, inverted);
          frontierUpdated = true;
          break;
        }

        case RewriteStrategy::RewriteF: {
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

          auto topSuppRight = dyn_cast<handshake::ConditionalBranchOp>(
              conditionC.getDefiningOp());
          if (!topSuppRight)
            continue;

          if (!checkConditionsMatch(topSuppLeft.getConditionOperand(),
                                    topSuppRight.getConditionOperand(), true)) {
            continue;
          }

          applyRewriteF(branchOp, topSuppLeft, topSuppRight, frontier, namer);
          frontierUpdated = true;
          break;
        }

        case RewriteStrategy::RewriteG: {
          // make sure mux exists and we have the TrueBranch
          if (!mux || mux.getDataOperands()[1] != branchOp.getFalseResult())
            continue;

          // verify that the conditions match but inverted
          if (!checkConditionsMatch(mux.getSelectOperand(),
                                    branchOp.getConditionOperand(), false))
            continue;

          auto falseBranch = dyn_cast_or_null<handshake::ConditionalBranchOp>(
              mux.getDataOperands()[0].getDefiningOp());
          if (!falseBranch)
            continue;

          // verify that the mux and falseBranch have the exact same condition
          if (!checkConditionsMatch(mux.getSelectOperand(),
                                    falseBranch.getConditionOperand(), true))
            continue;

          auto topSuppressorA = dyn_cast<handshake::ConditionalBranchOp>(
              branchOp.getDataOperand().getDefiningOp());
          auto topSuppressorB = dyn_cast<handshake::ConditionalBranchOp>(
              falseBranch.getDataOperand().getDefiningOp());
          auto topSuppressorC = dyn_cast<handshake::ConditionalBranchOp>(
              mux.getSelectOperand().getDefiningOp());

          // all lines need to have a suppressor
          if (!topSuppressorA || !topSuppressorB || !topSuppressorC)
            continue;

          // verify all three trace back to the same source
          if (!checkConditionsMatch(topSuppressorA.getConditionOperand(),
                                    topSuppressorB.getConditionOperand(), true))
            continue;
          if (!checkConditionsMatch(topSuppressorC.getConditionOperand(),
                                    topSuppressorB.getConditionOperand(), true))
            continue;

          applyRewriteG(mux, branchOp, falseBranch, topSuppressorA,
                        topSuppressorB, topSuppressorC, frontier, namer);
          frontierUpdated = true;

          break;
        }

        case RewriteStrategy::RewriteH: {
          if (!mux)
            continue;

          auto falseBranch = dyn_cast_or_null<handshake::ConditionalBranchOp>(
              mux.getDataOperands()[0].getDefiningOp());
          auto initOp = dyn_cast_or_null<handshake::InitOp>(
              mux.getSelectOperand().getDefiningOp());

          if (!falseBranch || !initOp)
            continue;

          if (checkConditionsMatch(branchOp.getConditionOperand(),
                                   falseBranch.getConditionOperand(), true)) {
            auto notOp = dyn_cast<handshake::NotIOp>(
                initOp.getOperand().getDefiningOp());
            if (notOp && notOp.getOperand().getDefiningOp() == initOp) {
              applyRewriteH(mux, branchOp, initOp, frontier, namer);
              frontierUpdated = true;
            }
          }
          break;
        }

        default:
          llvm::errs() << static_cast<int>(rewrite)
                       << " should not be applied as often as possible\n";
          break;
        }
        // frontier was mutated, break and restart the loop
        if (frontierUpdated)
          break;
      }
      if (frontierUpdated)
        break;
    }
  } while (frontierUpdated);
}

/// Apply Rewrites B, C, D
void EagerlyElasticAllPass::applyRewriteXOnce(
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer,
    RewriteStrategy rewrite) {
  SmallVector<handshake::ConditionalBranchOp> initialFrontier(frontier.begin(),
                                                              frontier.end());

  for (auto branchOp : initialFrontier) {
    for (auto *user : branchOp.getFalseResult().getUsers()) {
      if (auto mux = dyn_cast<handshake::MuxOp>(user)) {

        switch (rewrite) {
        case RewriteStrategy::RewriteB: {
          if (!checkConditionsMatch(branchOp.getConditionOperand(),
                                    mux.getSelectOperand(), false))
            continue;

          auto falseBranch = dyn_cast_or_null<handshake::ConditionalBranchOp>(
              mux.getDataOperands()[0].getDefiningOp());
          if (!falseBranch)
            continue;

          if (!checkConditionsMatch(mux.getSelectOperand(),
                                    falseBranch.getConditionOperand(), true))
            continue;

          applyRewriteB(mux, branchOp, falseBranch, frontier, namer);
          return;
        }

        case RewriteStrategy::RewriteC: {
          if (!checkConditionsMatch(branchOp.getConditionOperand(),
                                    mux.getSelectOperand(), false))
            continue;

          applyRewriteC(mux, branchOp, frontier, namer);
          return;
        }

        case RewriteStrategy::RewriteD: {
          // identify loops (mux connected to init)
          auto init = dyn_cast_or_null<handshake::InitOp>(
              mux.getSelectOperand().getDefiningOp());

          if (init) {
            if (!checkConditionsMatch(branchOp.getConditionOperand(),
                                      init.getOperand(), false)) {
              continue;
            }

            // check whether the suppressor is connected to path A of the mux
            if (mux.getDataOperands()[1] == branchOp.getFalseResult()) {
              auto nameAttr = mux->getAttrOfType<mlir::StringAttr>("handshake.name");
              if (nameAttr && nameAttr.getValue() == "mux1") {
                applyRewriteD(mux, branchOp, init, frontier, namer);
                // return;
              }
            }
          }
          break;
        }

        default:
          llvm::errs() << static_cast<int>(rewrite)
                       << " should not be applied only once\n";
          break;
        }
      }
    }
  }
}

void EagerlyElasticAllPass::movePastFunctionBlock(
      DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer,
      ModuleOp modOp) {

  SmallVector<handshake::ConditionalBranchOp> initialFrontier(frontier.begin(),
                                                              frontier.end());

  auto infoDict = modOp->getAttrOfType<DictionaryAttr>("handshake.subatomic_region_info");
  if (!infoDict) return;

  auto subloopBbAttr = infoDict.getAs<IntegerAttr>("subloop_bb");
  auto storeOpsArray = infoDict.getAs<ArrayAttr>("store_ops");
  auto entryOpsArray = infoDict.getAs<ArrayAttr>("entry_ops");
  if (!entryOpsArray || !subloopBbAttr) return;

  for (auto branchOp : initialFrontier) {
    for (auto *user : branchOp.getFalseResult().getUsers()) {
      auto mux = dyn_cast<handshake::MuxOp>(user);
      if (!mux) continue;

      // check that we wouldn't apply another rewrite
      if (mux.getDataOperands()[1] == branchOp.getFalseResult()) continue;

      // check that this mux is part of bb2
      auto bbAttr = mux->getAttrOfType<IntegerAttr>("handshake.bb");
      if (!bbAttr || bbAttr.getInt() != subloopBbAttr.getInt()) 
        continue;

      llvm::errs() << "start moving past entire bb2 function!\n";

      Value condition = branchOp.getConditionOperand();

      // place a suppressor in front of all identified entry ops
      for (Attribute attr : entryOpsArray) {
        auto entryDict = cast<DictionaryAttr>(attr);
        StringRef opName = entryDict.getAs<StringAttr>("op").getValue();
        int64_t operandIdx = entryDict.getAs<IntegerAttr>("operand_idx").getInt();

        // Find the target operation by handshake.name
        Operation *targetOp = namer.getOp(opName);
        if (!targetOp) continue;

        OpOperand &targetOperand = targetOp->getOpOperand(operandIdx);
        Value incomingData = targetOperand.get();

        // Insert suppressor (ConditionalBranchOp) before targetOp's operand
        OpBuilder builder(targetOp);
        Location loc = targetOp->getLoc();

        auto newBranch = builder.create<handshake::ConditionalBranchOp>(
            loc, condition, incomingData);
        setHandshakeAttrs(targetOp->getAttr("handshake.bb"), namer, {newBranch});

        // Rewire operand to consume the false outcome of the new suppressor
        targetOperand.set(newBranch.getFalseResult());
        frontier.insert(newBranch);
      }

      llvm::errs() << "placed all new suppressors\n";

      // if there's a store in the region, add a suppressor in front of it
      if (storeOpsArray && !storeOpsArray.empty()) {
        Value conditionInBB2 = nullptr;
        for (Operation &op : mux->getBlock()->getOperations()) {
          if (auto targetMux = dyn_cast<handshake::MuxOp>(&op)) {
          
            // Check if one of its data inputs comes from the pseudo-constant
            for (unsigned i = 0; i < targetMux.getDataOperands().size(); ++i) {
              Value operand = targetMux.getDataOperands()[i];
              if (Operation *defOp = operand.getDefiningOp()) {
                if (defOp->hasAttr("handshake.pseudo_cond")) {
                  // rewire the mux: replace the pseudo-constant with the real condition
                  targetMux.setOperand(i+1, condition);
                  conditionInBB2 = targetMux.getResult();
                  break;
                }
              }
            }
          }
          if (conditionInBB2) break;
        }

        if (!conditionInBB2) {
          llvm::errs() << "Could not find pseudo-constant Mux in bb2!\n";
          return;
        }

        for (Attribute attr : storeOpsArray) {
          StringRef storeName = cast<StringAttr>(attr).getValue();

          // Look up the store operation by its handshake.name
          Operation *storeOp = namer.getOp(storeName);
          if (!storeOp) continue;

          OpOperand &dataOperand = storeOp->getOpOperand(1);
          Value incomingData = dataOperand.get();

          if (incomingData.getDefiningOp<handshake::ConditionalBranchOp>())
            continue;

          // create the suppressor using the token channel from the pseudo constant
          OpBuilder builder(storeOp);
          auto newBranch = builder.create<handshake::ConditionalBranchOp>(
              storeOp->getLoc(), conditionInBB2, incomingData);
          setHandshakeAttrs(storeOp->getAttr("handshake.bb"), namer,
                            {newBranch});

          dataOperand.set(newBranch.getFalseResult());
          frontier.insert(newBranch);
        } 
      }

      // rewire the old branchOp
      for (OpOperand &use : mux->getOpOperands()) {
        if (use.get() == branchOp.getFalseResult()) {
          use.set(branchOp.getDataOperand());
        }
      }

      llvm::errs() << "rewiring done\n";

      // if the branchOp has no uses anymore, delete it
      if (branchOp.getFalseResult().use_empty()) {
        llvm::errs() << "erase!\n";
        frontier.erase(branchOp);
        branchOp.erase();
      }

      return;
    }
  }
}

void EagerlyElasticAllPass::runOnOperation() {
  ModuleOp modOp = getOperation();
  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  handshake::FuncOp funcOp =
      cast<handshake::FuncOp>(&modOp.getBodyRegion().front().front());
  assert(funcOp && "No funcOp found!");

  // identify and prepare suppressors and return a list of all of them
  auto frontier = prepareSuppressors(funcOp, namer);

  // apply rewrites: A*, E*, F*
  applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);
  applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteE);
  applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteF);

  for (unsigned i = 0; i < 1; i++) {
    applyRewriteXOnce(frontier, namer, RewriteStrategy::RewriteD);
    applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteG);
    applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);
  }

  movePastFunctionBlock(frontier, namer, modOp);
  // applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);

  for (unsigned i = 0; i < 0; i++) {
    applyRewriteXOnce(frontier, namer, RewriteStrategy::RewriteB);
    applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);
    applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteH);
  }
}
