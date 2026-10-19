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

  void applyRewriteXAsOftenAsPossible(
      DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer,
      RewriteStrategy rewrite);

  void applyRewriteXOnce(DenseSet<handshake::ConditionalBranchOp> &frontier,
                         NameAnalysis &namer, RewriteStrategy rewrite);

  void
  applyRewritesForRegion(DenseSet<handshake::ConditionalBranchOp> &frontier,
                         NameAnalysis &namer, uint32_t expectedBB);

  void movePastFunctionBlock(DenseSet<handshake::ConditionalBranchOp> &frontier,
                             NameAnalysis &name, ModuleOp modOp,
                             DenseMap<int64_t, Value> blockEntryControlTokens);

  void removeControlNetwork(handshake::FuncOp funcOp);
  DenseMap<int64_t, Value> rewireControlNetwork(handshake::FuncOp funcOp,
                                                dynamatic::NameAnalysis &namer);
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

    SmallVector<handshake::ConditionalBranchOp> sortedFrontier(frontier.begin(),
                                                               frontier.end());
    llvm::sort(sortedFrontier, [](handshake::ConditionalBranchOp a,
                                  handshake::ConditionalBranchOp b) {
      StringRef nameA, nameB;
      if (auto attrA = a->getAttrOfType<StringAttr>(HANDSHAKENAME))
        nameA = attrA.getValue();
      if (auto attrB = b->getAttrOfType<StringAttr>(HANDSHAKENAME))
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
          if (!init)
            continue;

          if (!checkConditionsMatch(branchOp.getConditionOperand(),
                                    init.getOperand(), false))
            continue;

          // check whether the suppressor is connected to path A of the mux
          if (mux.getDataOperands()[1] != branchOp.getFalseResult())
            continue;

          applyRewriteD(mux, branchOp, init, frontier, namer);
          return;
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

        if (!init)
          continue;

        if (!checkConditionsMatch(branchOp.getConditionOperand(),
                                  init.getOperand(), false))
          continue;

        // check whether the suppressor is connected to path A of the mux
        if (mux.getDataOperands()[1] == branchOp.getFalseResult()) {
          auto bbAttr = mux->getAttrOfType<mlir::IntegerAttr>(HANDSHAKEBB);
          if (bbAttr && bbAttr.getValue() == expectedBB) {
            std::string muxName =
                mux->getAttrOfType<StringAttr>(HANDSHAKENAME).getValue().str();
            candidates.push_back({mux, branchOp, init, muxName});
            seenMuxes.insert(mux);
          }
        }
      }
    }
  }
  // Sort the final list of candidates alphabetically by their handshake.name
  llvm::sort(candidates, [](const MuxCandidate &a, const MuxCandidate &b) {
    return a.name < b.name;
  });

  // Apply the rewrites in the sorted name order
  for (const auto &candidate : candidates) {
    applyRewriteD(candidate.mux, candidate.branchOp, candidate.init, frontier,
                  namer);
  }
}

void EagerlyElasticAllPass::movePastFunctionBlock(
    DenseSet<handshake::ConditionalBranchOp> &frontier, NameAnalysis &namer,
    ModuleOp modOp, DenseMap<int64_t, Value> blockEntryControlTokens) {

  SmallVector<handshake::ConditionalBranchOp> initialFrontier(frontier.begin(),
                                                              frontier.end());

  auto infoArray = modOp->getAttrOfType<ArrayAttr>(SUBREG_INFO_ATTR);
  if (!infoArray || infoArray.empty())
    return;

  for (auto branchOp : initialFrontier) {
    for (auto *user : branchOp.getFalseResult().getUsers()) {

      // verify that we are at the start of an eligible region
      DictionaryAttr matchedRegionDict =
          findMatchingRegion(branchOp, user, infoArray);
      if (!matchedRegionDict)
        continue;
      llvm::errs() << "Found matching region dict!\n";

      // verify that the frontier doesn't get broken
      DenseSet<handshake::ConditionalBranchOp> branchesToDelete = {branchOp};
      if (!collectAndValidateFrontier(branchOp, user, branchesToDelete)) {
        llvm::errs() << "frontier invalid!\n";
        continue;
      }
      llvm::errs() << branchesToDelete.size() << " branches to delete\n";

      // sort the frontier branches by name to get determinism
      SmallVector<handshake::ConditionalBranchOp> sortedBranches(
          branchesToDelete.begin(), branchesToDelete.end());
      llvm::sort(sortedBranches, [](handshake::ConditionalBranchOp a,
                                    handshake::ConditionalBranchOp b) {
        StringRef nameA, nameB;
        if (auto attrA = a->getAttrOfType<StringAttr>(HANDSHAKENAME))
          nameA = attrA.getValue();
        if (auto attrB = b->getAttrOfType<StringAttr>(HANDSHAKENAME))
          nameB = attrB.getValue();
        return nameA < nameB;
      });

      // move the suppressors past the region to its exit operations
      auto stores = matchedRegionDict.getAs<BoolAttr>(STORES);
      auto entryOpsArray = matchedRegionDict.getAs<ArrayAttr>(ENTRY_OPS);
      auto entryBB = matchedRegionDict.getAs<IntegerAttr>(ENTRY_BB).getInt();
      Value condition = sortedBranches[0].getConditionOperand();
      llvm::errs() << "start moving past blocks after bb" << entryBB << "\n";

      OpBuilder builder(branchOp);
      insertExitSuppressors(entryOpsArray, condition, builder, namer,
                            blockEntryControlTokens, frontier);

      // update store predicate conditions inside the region, if any
      if (stores && stores.getValue()) {
        // find the corresponding pseudo constant
        std::string pseudoName = "pseudo_cond" + std::to_string(entryBB);
        auto pseudoConstant =
            dyn_cast_or_null<handshake::ConstantOp>(namer.getOp(pseudoName));
        if (!pseudoConstant || pseudoConstant.getResult().use_empty())
          continue;

        // create a new OrIOp directly after the pseudoconstant
        OpBuilder builder(pseudoConstant);
        builder.setInsertionPointAfter(pseudoConstant);
        Value currentVal = pseudoConstant.getResult();

        // create the OrIOp combining the constant with the condition
        auto orOp = builder.create<handshake::OrIOp>(pseudoConstant.getLoc(),
                                                     currentVal, condition);
        setHandshakeAttrs(pseudoConstant->getAttr(HANDSHAKEBB), namer, {orOp});

        // rewire all existing uses of pseudoConstant to the OrIOp result
        currentVal.replaceAllUsesExcept(orOp.getResult(), orOp);
      }

      // bypass and remove the old suppressors at the region entry
      bypassAndEraseOldBranches(sortedBranches, user, frontier);

      llvm::errs() << "finished movePastFunctionBlock\n";
      return;
    }
  }
}

/// Removes the isolated basic block control-token network (ControlMerges,
/// control-carrying branches, and their associated dead constants/inverters)
/// while leaving data paths, memory controllers, and sources intact.
void EagerlyElasticAllPass::removeControlNetwork(handshake::FuncOp funcOp) {
  // Clean up constants triggered by control values.
  for (auto cstOp :
       llvm::make_early_inc_range(funcOp.getOps<handshake::ConstantOp>())) {
    if (llvm::isa<handshake::ControlType>(cstOp.getOperand().getType())) {
      if (cstOp.use_empty()) {
        cstOp.erase();
      }
    }
  }

  // Track potential dead condition drivers (e.g., suppressor noti ops)
  llvm::SmallVector<Operation *, 16> candidateDeadDrivers;

  // Erase conditional branches whose data operand is a control token.
  for (auto condBr : llvm::make_early_inc_range(
           funcOp.getOps<handshake::ConditionalBranchOp>())) {
    if (llvm::isa<handshake::ControlType>(condBr.getDataOperand().getType())) {
      if (Operation *condDef = condBr.getConditionOperand().getDefiningOp())
        candidateDeadDrivers.push_back(condDef);
      condBr->dropAllUses();
      condBr.erase();
    }
  }

  // Erase unconditional control branches.
  for (auto br :
       llvm::make_early_inc_range(funcOp.getOps<handshake::BranchOp>())) {
    if (llvm::isa<handshake::ControlType>(br.getOperand().getType())) {
      br->dropAllUses();
      br.erase();
    }
  }

  // Erase all control merges, they should have no remaining users
  for (auto cmerge :
       llvm::make_early_inc_range(funcOp.getOps<handshake::ControlMergeOp>())) {
    // If anything still holds a reference, drop uses to prevent assertion
    cmerge.getResult().dropAllUses();
    cmerge.getIndex().dropAllUses();
    cmerge.erase();
  }

  // Clean up any dangling predicate inverters (e.g. noti)
  for (Operation *op : candidateDeadDrivers) {
    if (op && op->use_empty()) {
      op->erase();
    }
  }
}

DenseMap<int64_t, Value>
EagerlyElasticAllPass::rewireControlNetwork(handshake::FuncOp funcOp,
                                            dynamatic::NameAnalysis &namer) {

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
        // Only update if this token is the data operand (op. 1), not condition
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
          regionInput.getIns().setType(ctrlType); // Update input channel type
          regionInput.getResult().setType(
              ctrlType); // Update result channel type
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
    for (Operation *user : ctrlOp.getResult().getUsers()) {
      if (auto condBr = dyn_cast<handshake::ConditionalBranchOp>(user)) {
        if (condBr.getDataOperand() == ctrlOp.getResult() &&
            condBr.getFalseResult().use_empty()) {
          finalExitToken = condBr.getFalseResult();
        }
      }
    }

    // Check for stores and connect to the constant from the memory controller
    // to the inToken of the corresponding BlockControlOp (same bb)
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
        auto oldCst =
            storeCtrlSlot.get().getDefiningOp<handshake::ConstantOp>();
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

  applyRewriteXAsOftenAsPossible(frontier, namer, RewriteStrategy::RewriteA);
  // applyRewriteXOnce(frontier, namer, RewriteStrategy::RewriteD);

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
  for (auto markerOp :
       llvm::make_early_inc_range(funcOp.getOps<handshake::RegionInputOp>())) {
    Value incomingChannel = markerOp.getIns();

    // Forward the incoming channel to all operations consuming the marker's
    // output
    markerOp.getOuts().replaceAllUsesWith(incomingChannel);

    // Erase the marker from the IR
    markerOp.erase();
  }
}
