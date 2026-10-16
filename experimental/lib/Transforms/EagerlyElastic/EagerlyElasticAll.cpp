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

  void applyRewritesForRegion(
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer,
    uint32_t expectedBB);

  void movePastFunctionBlock(
      DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &name,
      ModuleOp modOp, DenseMap<int64_t, Value> blockEntryControlTokens);

  void removeControlNetwork(handshake::FuncOp funcOp);
  DenseMap<int64_t, Value> rewireControlNetwork(handshake::FuncOp funcOp, dynamatic::NameAnalysis &namer);
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

    SmallVector<handshake::ConditionalBranchOp> sortedFrontier(frontier.begin(), frontier.end());
    llvm::sort(sortedFrontier, [](handshake::ConditionalBranchOp a, handshake::ConditionalBranchOp b) {
      StringRef nameA, nameB;
      if (auto attrA = a->getAttrOfType<StringAttr>("handshake.name"))
        nameA = attrA.getValue();
      if (auto attrB = b->getAttrOfType<StringAttr>("handshake.name"))
        nameB = attrB.getValue();
      return nameA < nameB;
    });

    for (auto branchOp : sortedFrontier) {
      for (auto *user : branchOp.getFalseResult().getUsers()) {
        auto mux = dyn_cast<handshake::MuxOp>(user);
        switch (rewrite) {
        case RewriteStrategy::RewriteA: {
          if (isEligibleForBypass(branchOp, user) == BypassResult::Ineligible) {
            if (regionBased || !isa<handshake::RegionInputOp>(user))
              continue;
          }
          auto nameAttr = user->getAttrOfType<mlir::StringAttr>("handshake.name");
          if (nameAttr && nameAttr.getValue() == "constant21") { // mux18
            continue;
          }
          if (nameAttr && nameAttr.getValue() == "constant22") { // mux18
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
              applyRewriteD(mux, branchOp, init, frontier, namer);
              return;
              /* if (nameAttr && nameAttr.getValue() == "mux5") { // mux18
                applyRewriteD(mux, branchOp, init, frontier, namer);
                // return;
              } */
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

void EagerlyElasticAllPass::applyRewritesForRegion(
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer,
    uint32_t expectedBB) {
  SmallVector<handshake::ConditionalBranchOp> initialFrontier(frontier.begin(),
                                                              frontier.end());

  struct MuxCandidate {
    handshake::MuxOp mux;
    handshake::ConditionalBranchOp branchOp;
    handshake::InitOp init;
    std::string name;
  };

  SmallVector<MuxCandidate> candidates;
  DenseSet<handshake::MuxOp> seenMuxes;

  for (auto branchOp : initialFrontier) {
    for (auto *user : branchOp.getFalseResult().getUsers()) {
      if (auto mux = dyn_cast<handshake::MuxOp>(user)) {
        // Skip this mux if it has already been rewritten.
        if (seenMuxes.contains(mux))
          continue;

        // identify loops (mux connected to init)
        auto init = dyn_cast_or_null<handshake::InitOp>(
            mux.getSelectOperand().getDefiningOp());

        if (!init) continue;

        if (!checkConditionsMatch(branchOp.getConditionOperand(),
                                  init.getOperand(), false)) {
          continue;
        }

        // check whether the suppressor is connected to path A of the mux
        if (mux.getDataOperands()[1] == branchOp.getFalseResult()) {
          auto bbAttr =
              mux->getAttrOfType<mlir::IntegerAttr>("handshake.bb");
          if (bbAttr && bbAttr.getValue() == expectedBB) {
            std::string muxName = mux->getAttrOfType<StringAttr>("handshake.name").getValue().str();
            candidates.push_back({mux, branchOp, init, muxName});
          }
        }
      }
    }
  }
  // Sort only the final list of candidates alphabetically by their handshake.name
  llvm::sort(candidates, [](const MuxCandidate &a, const MuxCandidate &b) {
    return a.name < b.name;
  });

  // Apply the rewrites in the sorted name order
  for (const auto &candidate : candidates) {
    applyRewriteD(candidate.mux, candidate.branchOp, candidate.init, frontier, namer);
  }
}

void EagerlyElasticAllPass::movePastFunctionBlock(
      DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer,
      ModuleOp modOp, DenseMap<int64_t, Value> blockEntryControlTokens) {

  SmallVector<handshake::ConditionalBranchOp> initialFrontier(frontier.begin(),
                                                              frontier.end());

  auto infoArray = modOp->getAttrOfType<ArrayAttr>(SUBREG_INFO_ATTR);
  if (!infoArray || infoArray.empty()) return;

  for (auto branchOp : initialFrontier) {
    for (auto *user : branchOp.getFalseResult().getUsers()) {
      DictionaryAttr matchedRegionDict = nullptr;
      auto mux = dyn_cast<handshake::MuxOp>(user);
      auto inputMarker = dyn_cast<handshake::RegionInputOp>(user);
      int targetHeaderBB = 0;
      int muxBB;

      // ------- Check whether we are at the start of an eligible region -------
      if (mux) {
        // check that we wouldn't apply another rewrite
        if (mux.getDataOperands()[1] == branchOp.getFalseResult()) continue;

        // check that it's an actual loop mux
        if (!isa<handshake::InitOp>(mux.getSelectOperand().getDefiningOp())) continue;

        // check that this branch is a header and find matching entry
        muxBB = mux->getAttrOfType<IntegerAttr>(HANDSHAKEBB).getInt();
        for (Attribute attr : infoArray) {
          auto dict = dyn_cast<DictionaryAttr>(attr);
          auto successorBBsArray = dict.getAs<ArrayAttr>(SUCCESSOR_BBS);
          if (!successorBBsArray || successorBBsArray.size() != 1) continue;

          // Check only the first successor entry in the array
          int firstSuccBB = cast<IntegerAttr>(successorBBsArray[0]).getInt();
          if (muxBB && firstSuccBB == muxBB) {
            matchedRegionDict = dict;
            llvm::errs() << "found mux in bb: " << muxBB << '\n';
            break;
          }
        }
      } else if (inputMarker) {
        // verify the marker has a handshake.bb attribute
        auto markerBBAttr = inputMarker->getAttrOfType<IntegerAttr>("handshake.bb");
        if (!markerBBAttr)
          continue;

        // find matching subregion entry
        for (Attribute attr : infoArray) {
          auto dict = dyn_cast<DictionaryAttr>(attr);
          if (!dict) continue;
          auto entryBBAttr = dict.getAs<IntegerAttr>(ENTRY_BB);
          if (!entryBBAttr) continue;
          if (entryBBAttr.getValue() == markerBBAttr.getValue()) {
            matchedRegionDict = dict;
            targetHeaderBB = markerBBAttr.getInt();
            llvm::errs() << "found inputmarker\n";
            break;
          }
        }
      }

      if (!matchedRegionDict) continue;
      llvm::errs() << "found matching region dict!\n";
      llvm::errs() << "branchOp: ";
      branchOp.dump();

      // ------- Check that the frontier doesn't get broken -------
      bool frontierValid = true;
      llvm::DenseSet<handshake::ConditionalBranchOp> branchesToDelete = {branchOp};
      
      // collect all incomingValues that also go into the next block
      SmallVector<Value> incomingValues;
      Block *currentBlock = mux ? mux->getBlock() : inputMarker->getBlock();

      if (mux) {
        for (Operation &op : *currentBlock) {
          if (&op == mux || !isa<handshake::MuxOp>(&op))
            continue;

          // only look at muxes / cmerges in this basic block
          auto bbAttr = op.getAttrOfType<IntegerAttr>(HANDSHAKEBB);
          if (!bbAttr || bbAttr.getInt() != muxBB) {
            continue;
          }

          // check input for this Mux (not the loop backedge, that would be 1)
          if (auto muxOp = dyn_cast<handshake::MuxOp>(&op))
            incomingValues.push_back(muxOp.getDataOperands()[0]);
        }
      } else {
        for (auto markerOp : currentBlock->getOps<handshake::RegionInputOp>()) {
          // only look at markers in this basic block
          auto bbAttr = markerOp->getAttrOfType<IntegerAttr>(HANDSHAKEBB);
          if (!bbAttr || bbAttr.getInt() != targetHeaderBB)
            continue;

          incomingValues.push_back(markerOp->getOperand(0));
        }
      }

      // check that all incomingValues have a suppressor in front
      // of them with the same condition as branchOp (or are sourced/valid).
      for (Value operand : incomingValues) {
        Operation *defOp = operand.getDefiningOp();
        
        // is it a sibling conditional branch with a matching condition
        if (auto siblingBranch = dyn_cast<handshake::ConditionalBranchOp>(defOp)) {
          // check whether condition matches
          if (!checkConditionsMatch(branchOp.getConditionOperand(), 
                                      siblingBranch.getConditionOperand(), true)) {
            llvm::errs() << "conditions do not match...\n";
            siblingBranch.dump();
            frontierValid = false;
            break;
          } 
          branchesToDelete.insert(siblingBranch);
        } else if (!isSourced(operand)) {
          llvm::errs() << "not sourced\n";
          if (defOp) defOp->dump();
          frontierValid = false;
          break;
        }
      }

      if (!frontierValid) {
        llvm::errs() << "frontier invalid!\n";
        continue;
      } 

      llvm::errs() << branchesToDelete.size() << " branches to delete:\n";
      for (auto branch : branchesToDelete) {
        branch.dump();
      }

      SmallVector<handshake::ConditionalBranchOp> sortedBranches(branchesToDelete.begin(), branchesToDelete.end());
      llvm::sort(sortedBranches, [](handshake::ConditionalBranchOp a, handshake::ConditionalBranchOp b) {
        StringRef nameA, nameB;
        if (auto attrA = a->getAttrOfType<StringAttr>("handshake.name"))
          nameA = attrA.getValue();
        if (auto attrB = b->getAttrOfType<StringAttr>("handshake.name"))
          nameB = attrB.getValue();
        return nameA < nameB;
      });
      

      // ------- Start Actual Moving Past! --------
      auto stores = matchedRegionDict.getAs<BoolAttr>(STORES);
      auto entryOpsArray = matchedRegionDict.getAs<ArrayAttr>(ENTRY_OPS);
      auto entryBB = matchedRegionDict.getAs<IntegerAttr>(ENTRY_BB);
      Value condition = sortedBranches[0].getConditionOperand();
      llvm::errs() << "start moving past subblocks after entryBB" << entryBB.getInt() << "\n";

      // --- place a suppressor in front of all identified entry ops ---
      for (Attribute attr : entryOpsArray) {
        auto entryDict = cast<DictionaryAttr>(attr);
        StringRef opName = entryDict.getAs<StringAttr>("op").getValue();
        int64_t operandIdx = entryDict.getAs<IntegerAttr>("operand_idx").getInt();

        Value incomingData;
        Attribute bbAttr;
        OpBuilder builder(branchOp);

        if (opName == "fake_control") {
          llvm::errs() << "exitBB is: " << operandIdx << '\n';
          incomingData = blockEntryControlTokens.lookup(operandIdx);
          if (!incomingData) continue;
          bbAttr = builder.getUI32IntegerAttr(operandIdx);
          builder.setInsertionPointAfterValue(incomingData);
        } else {

          // Find the target operation by handshake.name
          Operation *targetOp = namer.getOp(opName);
          if (!targetOp) continue;

          if (isa<handshake::StoreOp>(targetOp)) {
            // memref::StoreOp had: 0 = data, 1 = memref, 2 = address
            // handshake::StoreOp has: 0 = address, 1 = data
            operandIdx = (operandIdx == 0) ? 1 : 0;
          }

          OpOperand &targetOperand = targetOp->getOpOperand(operandIdx);
          incomingData = targetOperand.get();

          // If incomingData comes from a cond_br_predicate, step one operation above
          if (auto predBr = incomingData.getDefiningOp<handshake::ConditionalBranchOp>()) {
            if (auto defName = predBr->getAttrOfType<StringAttr>("handshake.name")) {
              if (defName.getValue().starts_with("cond_br_predicate")) {
                incomingData = predBr.getDataOperand();
              }
            }
          }

          bbAttr = targetOp->getAttr(HANDSHAKEBB);
          builder.setInsertionPoint(targetOp);
        }

        // Insert suppressor
        auto newBranch = builder.create<handshake::ConditionalBranchOp>(
            incomingData.getLoc(), condition, incomingData);
        setHandshakeAttrs(bbAttr, namer, {newBranch});
        llvm::errs() << "placed an entryOp after:\n";
        incomingData.dump();

        // Replace ALL downstream uses of incomingData (except the newBranch itself!)
        // so that the later fork materialization pass places the fork after newBranch.
        incomingData.replaceAllUsesExcept(newBranch.getFalseResult(), newBranch);
        frontier.insert(newBranch);
      }

      // --- rewire the condition for the suppressors in front of stores ---
      if (stores && stores.getValue()) {
        // find the corresponding pseudo constant
        std::string pseudoName = "pseudo_cond" + std::to_string(entryBB.getInt());
        auto pseudoConstant = dyn_cast_or_null<handshake::ConstantOp>(namer.getOp(pseudoName));
        if (!pseudoConstant || pseudoConstant.getResult().use_empty()) {
          llvm::errs() << "ABORT NO PSEUDOCONSTANT??\n";
          continue;
        }

        // create a new OrIOp directly after the pseudoconstant
        OpBuilder builder(pseudoConstant);
        builder.setInsertionPointAfter(pseudoConstant);

        Value currentVal = pseudoConstant.getResult();

        // create the OrIOp combining the constant with the condition
        auto orOp = builder.create<handshake::OrIOp>(
            pseudoConstant.getLoc(), currentVal, condition);
        setHandshakeAttrs(pseudoConstant->getAttr(HANDSHAKEBB), namer, {orOp});

        // rewire all existing uses of pseudoConstant to the OrIOp result
        currentVal.replaceAllUsesExcept(orOp.getResult(), orOp);
      } else { llvm::errs() << "no stores\n";}

      // --- remove the suppressors that have just been bypassed ---
      
      // rewire the original muxes to bypass the old branchOps      
      for (auto branchToDelete : sortedBranches) {
        Value falseResult = branchToDelete.getFalseResult();
        Value dataOperand = branchToDelete.getDataOperand();

        // iterate over every branchToDelete's falseResult uses
        for (OpOperand &use : llvm::make_early_inc_range(falseResult.getUses())) {
          Operation *owner = use.getOwner();

          // check if the consumer is a MuxOp or CMergeOp located inside 
          // the same bb as the original mux. Only necessary for mux blocks
          if (isa<handshake::MuxOp>(owner)) { // removed cmergeop
            auto targetBB = owner->getAttrOfType<mlir::IntegerAttr>(HANDSHAKEBB);
            if (targetBB && muxBB && targetBB.getInt() == muxBB) {
              llvm::errs() << "Bypassing suppressor directly into mux/cmerge\n";
              use.set(dataOperand);
              continue;
            }
          }

          // check if the consumer is a RegionInputOp in the entryBB
          if (auto inputMarker = dyn_cast<handshake::RegionInputOp>(owner)) {
            auto markerBB = inputMarker->getAttrOfType<IntegerAttr>(HANDSHAKEBB);
            if (markerBB && markerBB.getInt() == entryBB.getInt()) {
              llvm::errs() << "Bypassing suppressor directly into RegionInputOp\n";
              use.set(dataOperand);
              continue;
            }
          }
        }        

        // if the branchOp has no uses anymore, delete it
        if (falseResult.use_empty()) {
          llvm::errs() << "deleted branch\n";
          frontier.erase(branchToDelete);
          branchToDelete.erase();
        }
      }
      llvm::errs() << "finished movePastFunctionBlock\n";
      return;
    }
  }
}

/// Removes the isolated basic block control-token network (ControlMerges, 
/// control-carrying Branches, and their associated dead constants/inverters)
/// while leaving data paths, memory controllers, and sources intact.
void EagerlyElasticAllPass::removeControlNetwork(handshake::FuncOp funcOp) {
  // 1. Clean up constants triggered by control values.
  for (auto cstOp : llvm::make_early_inc_range(funcOp.getOps<handshake::ConstantOp>())) {
    if (llvm::isa<handshake::ControlType>(cstOp.getOperand().getType())) {
      if (cstOp.use_empty()) {
        cstOp.erase();
      }
    }
  }

  // Track potential dead condition drivers (e.g., suppressor noti ops)
  llvm::SmallVector<Operation *, 16> candidateDeadDrivers;

  // 2. Erase conditional branches whose data operand is a control token.
  for (auto condBr : llvm::make_early_inc_range(funcOp.getOps<handshake::ConditionalBranchOp>())) {
    if (llvm::isa<handshake::ControlType>(condBr.getDataOperand().getType())) {
      if (Operation *condDef = condBr.getConditionOperand().getDefiningOp())
        candidateDeadDrivers.push_back(condDef);
      condBr->dropAllUses();
      condBr.erase();
    }
  }

  // 3. Erase unconditional control branches.
  for (auto br : llvm::make_early_inc_range(funcOp.getOps<handshake::BranchOp>())) {
    if (llvm::isa<handshake::ControlType>(br.getOperand().getType())) {
      br->dropAllUses();
      br.erase();
    }
  }

  // 4. Erase all control merges.
  // With step 1-3 finished, their operands and results have no remaining users.
  for (auto cmerge : llvm::make_early_inc_range(funcOp.getOps<handshake::ControlMergeOp>())) {
    // If anything still holds a reference, drop uses to prevent assertion failure
    cmerge.getResult().dropAllUses();
    cmerge.getIndex().dropAllUses();
    cmerge.erase();
  }

  // 5. Clean up dangling predicate inverters (noti) whose only purpose was
  // to feed the control branch conditions
  for (Operation *op : candidateDeadDrivers) {
    if (op && op->use_empty()) {
      op->erase();
    }
  }
}

DenseMap<int64_t, Value> EagerlyElasticAllPass::rewireControlNetwork(
    handshake::FuncOp funcOp, dynamatic::NameAnalysis &namer) {
  llvm::errs() << "rewire control network\n";

  MLIRContext *ctx = funcOp.getContext();
  auto ctrlType = handshake::ControlType::get(ctx);
  DenseSet<Operation *> opsToErase;
  DenseMap<int64_t, Value> blockEntryControlTokens;

  // 1. Convert all BlockControlOp results (and the entry "control_constant")
  //    from !handshake.channel<i32> to !handshake.control<>, and propagate
  //    that type through all connected MuxOps and ConditionalBranchOps.
  SmallVector<Value> worklist;
  for (auto ctrlOp : funcOp.getOps<handshake::BlockControlOp>()) {
    Value inToken = ctrlOp.getOperand(0);

    // If the input is the entry "control_constant", bypass the constant
    // and use its handshake::SourceOp (: <>) directly!
    if (auto cstOp = inToken.getDefiningOp<handshake::ConstantOp>()) {
      Value srcCtrl = cstOp.getCtrl(); // Already !handshake.control<>
      ctrlOp.setOperand(0, srcCtrl);
      opsToErase.insert(cstOp);
    }

    ctrlOp->getResult(0).setType(ctrlType);
    worklist.push_back(ctrlOp.getResult());
  }

  // Propagate !handshake.control<> through the mux / cond_br steering network
  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    for (Operation *user : val.getUsers()) {
      if (auto muxOp = dyn_cast<handshake::MuxOp>(user)) {
        if (muxOp.getResult().getType() != ctrlType) {
          muxOp.getResult().setType(ctrlType);
          worklist.push_back(muxOp.getResult());
        }
      } else if (auto condBr = dyn_cast<handshake::ConditionalBranchOp>(user)) {
        // Only update if this token is the data operand (operand 1), not condition
        if (condBr.getDataOperand() == val &&
            condBr.getTrueResult().getType() != ctrlType) {
          condBr.getTrueResult().setType(ctrlType);
          condBr.getFalseResult().setType(ctrlType);
          worklist.push_back(condBr.getTrueResult());
          worklist.push_back(condBr.getFalseResult());
        }
      } else if (auto regionInput = dyn_cast<handshake::RegionInputOp>(user)) {
        // Propagate the control type through the RegionInputOp
        if (regionInput.getResult().getType() != ctrlType) {
          regionInput.getIns().setType(ctrlType);     // Update input channel type
          regionInput.getResult().setType(ctrlType);  // Update result channel type
          worklist.push_back(regionInput.getResult());
        }
      }
    }
  }

  Value finalExitToken = nullptr;

  // 2. Process and bypass each handshake::BlockControlOp
  for (auto ctrlOp : funcOp.getOps<handshake::BlockControlOp>()) {
    auto bbAttr = ctrlOp->getAttrOfType<IntegerAttr>(HANDSHAKEBB);
    Value inToken = ctrlOp.getOperand(0);

    // Save the incoming control value for this basic block
    blockEntryControlTokens[bbAttr.getValue().getZExtValue()] = inToken;

    // Check if this BlockControlOp feeds the final exit ConditionalBranchOp
    // (the one whose falseResult is unused because there is no outer loop).
    for (Operation *user : ctrlOp.getResult().getUsers()) {
      if (auto condBr = dyn_cast<handshake::ConditionalBranchOp>(user)) {
        if (condBr.getDataOperand() == ctrlOp.getResult() &&
            condBr.getFalseResult().use_empty()) {
          finalExitToken = condBr.getFalseResult();
        }
      }
    }

    // Find all stores in that basic block
    for (auto storeOp : funcOp.getOps<handshake::StoreOp>()) {
      if (storeOp->getAttr(HANDSHAKEBB) != bbAttr)
        continue;

      auto addrBranch =
          storeOp.getAddress().getDefiningOp<handshake::ConditionalBranchOp>();
      Value conditionVal =
          addrBranch ? addrBranch.getConditionOperand() : nullptr;

      for (Operation *user : storeOp->getUsers()) {
        auto mcOp = dyn_cast<handshake::MemoryControllerOp>(user);
        if (!mcOp || mcOp.getInputs().size() < 3)
          continue;

        OpOperand &storeCtrlSlot = mcOp.getInputsMutable()[0];
        auto oldCst = storeCtrlSlot.get().getDefiningOp<handshake::ConstantOp>();
        if (!oldCst)
          continue;

        // Keep oldCst! It takes a !handshake.control<> trigger and outputs
        // the 1 : i32 constant that the MemoryController's ctrl input expects.
        if (conditionVal) {
          OpBuilder builder(oldCst);
          auto ctrlBranch = builder.create<handshake::ConditionalBranchOp>(
              oldCst->getLoc(), conditionVal, inToken);
          setHandshakeAttrs(bbAttr, namer, {ctrlBranch});
          oldCst.getCtrlMutable().assign(ctrlBranch.getFalseResult());
        } else {
          oldCst.getCtrlMutable().assign(inToken);
        }
        break;
      }
    }

    // Forward the input token to all consumers of BlockControlOp's result
    ctrlOp.getResult().replaceAllUsesWith(inToken);

    // Collect BlockControlOp for cleanup
    opsToErase.insert(ctrlOp);
  }

  // 3. Route the final exit token (now !handshake.control<>) into memEnd
  if (finalExitToken) {
    for (auto mcOp : funcOp.getOps<handshake::MemoryControllerOp>()) {
      OpOperand &memEndSlot = mcOp->getOpOperands().back();
      memEndSlot.set(finalExitToken);
    }
  }

  // 4. Clean up all matched operations
  for (Operation *op : opsToErase) {
    op->dropAllUses();
    op->erase();
  }

  return blockEntryControlTokens;
}


void EagerlyElasticAllPass::runOnOperation() {
  ModuleOp modOp = getOperation();
  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  handshake::FuncOp funcOp =
      cast<handshake::FuncOp>(&modOp.getBodyRegion().front().front());
  assert(funcOp && "No funcOp found!");

  // handle control network
  removeControlNetwork(funcOp);
  auto blockEntryCtrlTokens = rewireControlNetwork(funcOp, namer);

  // identify and prepare suppressors and return a list of all of them
  auto frontier = prepareSuppressors(funcOp, namer);
  // markMultiSuccessorHeaderBranches(frontier, modOp);

  applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);

  applyRewritesForRegion(frontier, namer, 2);
  applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);
  
  if (regionBased) {
    llvm::errs() << "regionbased enabled\n";
    movePastFunctionBlock(frontier, namer, modOp, blockEntryCtrlTokens);
  } else {
    llvm::errs() << "no region based\n";
  }
  applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);

  // Erase all handshake::RegionInputOp markers and bypass them directly
  for (auto markerOp : llvm::make_early_inc_range(
          funcOp.getOps<handshake::RegionInputOp>())) {
    Value incomingChannel = markerOp.getIns();
    
    // Forward the incoming channel to all operations consuming the marker's output
    markerOp.getOuts().replaceAllUsesWith(incomingChannel);
    
    // Erase the marker from the IR
    markerOp.erase();
  }
}