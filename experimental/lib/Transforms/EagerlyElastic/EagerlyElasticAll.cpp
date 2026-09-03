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
  RewriteD2,
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

    if (branchOp->hasAttr("store_suppressor")) {
      llvm::errs() << "skipping store suppressor\n";
      // continue;
    }

    OpBuilder builder(branchOp);
    auto bbAttr = branchOp->getAttr(HANDSHAKEBB);

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
        // continue;
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
              if (nameAttr && nameAttr.getValue() == "mux3") { // mux18
                applyRewriteD(mux, branchOp, init, frontier, namer);
                // return;
              }
              if (nameAttr && nameAttr.getValue() == "mux4") { // mux18
                applyRewriteD(mux, branchOp, init, frontier, namer);
                // return;
              }
              if (nameAttr && nameAttr.getValue() == "mux7") { // mux18
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
      } else if (auto merge = dyn_cast<handshake::ControlMergeOp>(user)) {
        auto nameAttr = merge->getAttrOfType<mlir::StringAttr>("handshake.name");
        if (nameAttr && nameAttr.getValue() == "control_merge1" && rewrite == RewriteStrategy::RewriteD2) { // mux18
          applyRewriteDMerged(merge, branchOp, frontier, namer);
          return;
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

  auto infoArray = modOp->getAttrOfType<ArrayAttr>(SUBLOOP_INFO_ATTR);
  if (!infoArray || infoArray.empty()) return;

  for (auto branchOp : initialFrontier) {
    for (auto *user : branchOp.getFalseResult().getUsers()) {
      DictionaryAttr matchedRegionDict = nullptr;
      auto mux = dyn_cast<handshake::MuxOp>(user);
      auto targetBranch = dyn_cast<handshake::ConditionalBranchOp>(user);
      int targetHeaderBB = 0;
      int muxBB;

      if (mux) {
        // check that we wouldn't apply another rewrite
        if (mux.getDataOperands()[1] == branchOp.getFalseResult()) continue;

        // check that it's an actual loop mux
        if (!isa<handshake::InitOp>(mux.getSelectOperand().getDefiningOp())) continue;
        llvm::errs() << "survived loop\n";
        // check that this branch is a header and find matching entry
        muxBB = mux->getAttrOfType<IntegerAttr>(HANDSHAKEBB).getInt();
        for (Attribute attr : infoArray) {
          auto dict = dyn_cast<DictionaryAttr>(attr);
          auto successorBBsArray = dict.getAs<ArrayAttr>("successor_bbs");
          if (!successorBBsArray || successorBBsArray.size() != 1) continue;

          // Check only the first successor entry in the array
          int firstSuccBB = cast<IntegerAttr>(successorBBsArray[0]).getInt();
          if (muxBB && firstSuccBB == muxBB) {
            matchedRegionDict = dict;
            break;
          }
        }
      } else if (targetBranch) {
        // verify the targetBranch is marked with subloop_header_bb
        auto subloopHeaderBBAttr = targetBranch->getAttrOfType<IntegerAttr>("subloop_header_bb");
        if (!subloopHeaderBBAttr)
          continue;

        // ensure branchOp is NOT the condition operand of targetBranch
        Value targetCond = targetBranch.getConditionOperand();
        if (targetCond.getDefiningOp() == branchOp)
          continue;

        // find matching subloop entry
        targetHeaderBB = subloopHeaderBBAttr.getInt();
        for (Attribute attr : infoArray) {
          auto dict = dyn_cast<DictionaryAttr>(attr);
          if (!dict) continue;
          auto headerBBAttr = dict.getAs<IntegerAttr>("header_bb");
          if (headerBBAttr && headerBBAttr.getInt() == targetHeaderBB) {
            matchedRegionDict = dict;
            break;
          }
        }
        llvm::errs() << "branchOp: ";
        branchOp.dump();
        llvm::errs() << "targetBranch: ";
        targetBranch.dump();
      }

      if (!matchedRegionDict) continue;
      llvm::errs() << "found matching region dict!\n";

      // check that the frontier doesn't get broken
      bool frontierValid = true;
      llvm::DenseSet<handshake::ConditionalBranchOp> branchesToDelete = {branchOp};
      if (mux) {
      Block *currentBlock = mux->getBlock();
      for (Operation &op : *currentBlock) {
        auto otherMux = dyn_cast<handshake::MuxOp>(&op);
        if (!otherMux || otherMux == mux)
          continue;
        
        if (otherMux->getAttrOfType<mlir::IntegerAttr>("handshake.bb").getInt() != muxBB)
          continue;

        otherMux.dump();

        // Check input for this Mux (not the loop backedge, that would be 1)
        Value operand = otherMux.getDataOperands()[0];
        operand.dump();

        // is it a sourced value
        if (dyn_cast<handshake::ConstantOp>(operand.getDefiningOp())) {
          llvm::errs() << "comes from constant\n";
          continue;
        }

        Operation *defOp = operand.getDefiningOp();
        if (!defOp) {
          frontierValid = false;
          break;
        }          

        // is it a sibling conditional branch with a matching condition
        if (auto siblingBranch = dyn_cast<handshake::ConditionalBranchOp>(defOp)) {
          Value currentCond = branchOp.getConditionOperand();
          if (checkConditionsMatch(currentCond, 
                                      siblingBranch.getConditionOperand(), true)) {
            branchesToDelete.insert(siblingBranch);
            llvm::errs() << "conditions match\n";
            continue;
          } else {
            auto nameAttr = siblingBranch->getAttrOfType<mlir::StringAttr>("handshake.name");
              if (nameAttr && nameAttr.getValue() =="cond_br4") { // mux18
                branchesToDelete.insert(siblingBranch);
              }
          }
        }
        frontierValid = false;
        // break;
      
      } 
      } else {
        // check whether all branches with the correct subloop_header_bb attribute have a cond_br
        // in front of them with the same condition as the branchOp. They have to be deleted later
        Value currentCond = branchOp.getConditionOperand();
        for (auto otherBranch : initialFrontier) {
          // make sure the branch I'm looking at has a subloop_header_bb attribute of correct no.
          auto headerBBAttr = otherBranch->getAttrOfType<IntegerAttr>("subloop_header_bb");
          if (!headerBBAttr || headerBBAttr.getInt() != targetHeaderBB)
            continue;

          Value condVal = otherBranch.getConditionOperand();
          Operation *condDefOp = condVal.getDefiningOp();

          // step past a NotIOp if present
          if (auto notOp = dyn_cast_or_null<handshake::NotIOp>(condDefOp)) {
            condVal = notOp.getOperand();
            condDefOp = condVal.getDefiningOp();
          }

          Value dataVal = otherBranch.getDataOperand();

          for (Operation *defOp : {condDefOp, dataVal.getDefiningOp()}) {
            if (auto siblingBranch = dyn_cast_or_null<handshake::ConditionalBranchOp>(
                    defOp)) {
              if (siblingBranch == branchOp)
                continue;

              // check whether condition matches
              if (!checkConditionsMatch(currentCond,
                                        siblingBranch.getConditionOperand(), true)) {
                llvm::errs() << "conditions do not match...\n";
                otherBranch.dump();
                frontierValid = false;
                break;
              }
              branchesToDelete.insert(siblingBranch);
            } else if (!isSourced(dataVal)) {
              // if (!isa<handshake::ConstantOp>(dataVal.getDefiningOp())) {
              llvm::errs() << "not sourced\n";
              otherBranch.dump();
              frontierValid = false;
              break; // }
            }
          }
          if (!frontierValid) break;
        }
      }

      if (!frontierValid) {
        llvm::errs() << "frontier invalid!\n";
        // continue;
      } 

      llvm::errs() << "branches to delete:\n";
      for (auto branch : branchesToDelete) {
        branch.dump();
      }

      auto stores = matchedRegionDict.getAs<BoolAttr>(STORES);
      auto entryOpsArray = matchedRegionDict.getAs<ArrayAttr>(ENTRY_OPS);
      auto headerBB = matchedRegionDict.getAs<IntegerAttr>(HEADER_BB);
      Value condition = branchOp.getConditionOperand();
      llvm::errs() << "start moving past subblocks after headerbb" << headerBB.getInt() << "\n";

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

        // Insert suppressor before targetOp's operand
        OpBuilder builder(targetOp);
        Location loc = targetOp->getLoc();

        auto newBranch = builder.create<handshake::ConditionalBranchOp>(
            loc, condition, incomingData);
        setHandshakeAttrs(targetOp->getAttr(HANDSHAKEBB), namer, {newBranch});
        llvm::errs() << "placed an entryOp after:\n";
        incomingData.dump();

        // Rewire operand to consume the false outcome of the new suppressor
        targetOperand.set(newBranch.getFalseResult());
        frontier.insert(newBranch);
      }

      // handle stores
      if (stores && stores.getValue()) {
        std::string pseudoName = "pseudo_cond" + std::to_string(headerBB.getInt());
        auto pseudoConstant = dyn_cast_or_null<handshake::ConstantOp>(namer.getOp(pseudoName));
        if (!pseudoConstant || pseudoConstant.getResult().use_empty()) {
          llvm::errs() << "ABORT NO PSEUDOCONSTANT??\n";
          continue;
        }

        for (OpOperand &targetOperand : llvm::make_early_inc_range(pseudoConstant.getResult().getUses())) {
          // This is MuxOp on Pass 1, but an AndIOp on Pass 2, 3, etc.
          Operation *consumerOp = targetOperand.getOwner(); 
          Value currentInput = targetOperand.get();

          // insert the new AndIOp right before the consumer
          /* OpBuilder builder(consumerOp);
          auto notOp = builder.create<handshake::NotIOp>(consumerOp->getLoc(), condition);
          auto andOp = builder.create<handshake::AndIOp>(
              consumerOp->getLoc(), currentInput, notOp.getResult());
          setHandshakeAttrs(consumerOp->getAttr(HANDSHAKEBB), namer, {andOp, notOp}); */

          // insert the new Or right before the consumer
          OpBuilder builder(consumerOp);
          auto orOp = builder.create<handshake::OrIOp>(
              consumerOp->getLoc(), currentInput, condition);
          setHandshakeAttrs(consumerOp->getAttr(HANDSHAKEBB), namer, {orOp});
          llvm::errs() << "placed or\n";
          condition.dump();

          targetOperand.set(orOp.getResult());
        }
      } else { llvm::errs() << "no stores\n";}

      llvm::errs() << "size: " << branchesToDelete.size() << '\n';
      // rewire the original muxes to bypass the old branchOps
      for (auto muxBranchOp : branchesToDelete) {
        Value falseResult = muxBranchOp.getFalseResult();
        Value dataOperand = muxBranchOp.getDataOperand();
        auto branchBB = muxBranchOp->getAttrOfType<mlir::IntegerAttr>("handshake.bb");

        // find and update all downstream Muxes consuming this specific branch
        for (OpOperand &use : llvm::make_early_inc_range(falseResult.getUses())) {
          // TODO: remove this fix
          /* use.set(dataOperand);
          continue; */
          Operation *owner = use.getOwner();

          if (auto targetMux = dyn_cast<handshake::MuxOp>(use.getOwner())) {
            auto targetBB = targetMux->getAttrOfType<mlir::IntegerAttr>("handshake.bb");
            if (branchBB && targetBB && branchBB.getInt() != targetBB.getInt()) {
              llvm::errs() << "replaced mux\n";
              use.set(dataOperand);
            }
          }

          if (auto branch = dyn_cast<handshake::ConditionalBranchOp>(use.getOwner())) {
            if (auto headerBBAttr = branch->getAttrOfType<IntegerAttr>("subloop_header_bb")) {
              if (headerBBAttr.getInt() == headerBB.getInt()) {
                llvm::errs() << "replaced use because attribute\n";
                use.set(dataOperand); 
              }
            }
          }

          // Check if owner is a Not operation and its result feeds a matching ConditionalBranchOp
          if (isa<handshake::NotIOp>(owner)) {
            bool feedsMatchingBranch = false;
            for (Operation *notUser : owner->getResult(0).getUsers()) {
              if (auto branch = dyn_cast<handshake::ConditionalBranchOp>(notUser)) {
                if (auto headerBBAttr = branch->getAttrOfType<IntegerAttr>("subloop_header_bb")) {
                  if (headerBBAttr.getInt() == headerBB.getInt()) {
                    feedsMatchingBranch = true;
                    break;
                  }
                }
              }
            }
            // If the NOT op feeds a marked branch, rewire the NOT op's input operand
            if (feedsMatchingBranch) {
              llvm::errs() << "replaced use because feedsMatchingBranch\n";
              use.set(dataOperand);
            }
          }
        }        

        // if the branchOp has no uses anymore, delete it
        if (falseResult.use_empty()) {
          llvm::errs() << "deleted branch\n";
          frontier.erase(muxBranchOp);
          muxBranchOp.erase();
        }
      }
      llvm::errs() << "finished movePastFunctionBlock\n";
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
  markMultiSuccessorHeaderBranches(frontier, modOp);

  // apply rewrites: A*, E*, F*
  applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);
  applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteE);
  // applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteF);

  for (unsigned i = 0; i < 1; i++) {
    applyRewriteXOnce(frontier, namer, RewriteStrategy::RewriteD);
    // applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteG);
    applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);
  }
  applyRewriteXOnce(frontier, namer, RewriteStrategy::RewriteD2);
  applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);

  // movePastFunctionBlock(frontier, namer, modOp);
  // applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);
  // applyRewriteXOnce(frontier, namer, RewriteStrategy::RewriteD);
  // applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);
  // movePastFunctionBlock(frontier, namer, modOp);

  for (unsigned i = 0; i < 0; i++) {
    applyRewriteXOnce(frontier, namer, RewriteStrategy::RewriteB);
    applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);
    applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteH);
  }
}
