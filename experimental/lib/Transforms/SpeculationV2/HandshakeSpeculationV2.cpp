#include "HandshakeSpeculationV2.h"
#include "JSONImporter.h"
#include "SpecV2Lib.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/MaterializationUtil/MaterializationUtil.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <fstream>

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculationv2;

namespace dynamatic {
namespace experimental {
namespace speculationv2 {

// Implement the base class and auto-generated create functions.
// Must be called from the .cpp file to avoid multiple definitions
#define GEN_PASS_DEF_HANDSHAKESPECULATIONV2
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct HandshakeSpeculationV2Pass
    : public dynamatic::experimental::speculationv2::impl::
          HandshakeSpeculationV2Base<HandshakeSpeculationV2Pass> {
  using HandshakeSpeculationV2Base<
      HandshakeSpeculationV2Pass>::HandshakeSpeculationV2Base;
  void runDynamaticPass() override;
};

/// Returns if the circuit is eligible for MuxPasserSwap.
static bool isEligibleForMuxPasserSwap(MuxOp muxOp) {
  // Ensure the rewritten subcircuit structure.
  Operation *backedgeDefiningOp = muxOp.getDataOperands()[1].getDefiningOp();
  if (!isa<PasserOp>(backedgeDefiningOp)) {
    llvm::errs() << "data1 is not passer\n";
    return false;
  }
  auto passerOp = cast<PasserOp>(backedgeDefiningOp);

  Operation *selectorDefiningOp = muxOp.getSelectOperand().getDefiningOp();
  if (!isa<InitOp>(selectorDefiningOp)) {
    llvm::errs() << "mux select is not init\n";
    return false;
  }
  auto initOp = cast<InitOp>(selectorDefiningOp);

  if (!equalsIndirectly(passerOp.getCtrl(), initOp.getOperand())) {
    llvm::errs() << "mux select and passer ctrl not equal\n";
    return false;
  }

  // TODO: Verify token counts

  return true;
}

static bool isMuxPasserToAndEligible(MuxOp muxOp) {
  auto *data0 = muxOp.getDataOperands()[0].getDefiningOp();
  auto *data1 = muxOp.getDataOperands()[1].getDefiningOp();

  if (!isa<ConstantOp>(data0)) {
    llvm::errs() << "data0 is not constant\n";
    return false;
  }
  auto constOp = cast<ConstantOp>(data0);
  if (constOp.getValue().cast<IntegerAttr>().getValue() != 0) {
    llvm::errs() << "data0 is not constant 0\n";
    return false;
  }
  auto *constDefiningOp = constOp.getOperand().getDefiningOp();
  if (!isa<SourceOp>(constDefiningOp)) {
    llvm::errs() << "data0 is not from source\n";
    return false;
  }

  if (!isa<PasserOp>(data1)) {
    llvm::errs() << "data1 is not passer\n";
    return false;
  }
  auto passerOp = cast<PasserOp>(data1);

  if (!equalsIndirectly(passerOp.getCtrl(), muxOp.getSelectOperand())) {
    llvm::errs() << "mux select and passer ctrl not equal\n";
    return false;
  }

  return true;
}
static AndIOp muxPasserToAnd(MuxOp muxOp) {
  OpBuilder builder(muxOp.getContext());
  builder.setInsertionPoint(muxOp);

  auto passerOp = cast<PasserOp>(muxOp.getDataOperands()[1].getDefiningOp());
  auto constOp = cast<ConstantOp>(muxOp.getDataOperands()[0].getDefiningOp());
  auto sourceOp = cast<SourceOp>(constOp.getOperand().getDefiningOp());

  AndIOp andIOp = builder.create<AndIOp>(
      muxOp->getLoc(), muxOp.getResult().getType(),
      ArrayRef<Value>{passerOp.getCtrl(), passerOp.getData()});
  inheritBB(muxOp, andIOp);

  muxOp.getResult().replaceAllUsesWith(andIOp.getResult());

  // Erase the old ops
  eraseMaterializedOperation(muxOp);
  eraseMaterializedOperation(passerOp);
  eraseMaterializedOperation(constOp);
  eraseMaterializedOperation(sourceOp);

  materializeValue(andIOp.getLhs());
  materializeValue(andIOp.getRhs());
  materializeValue(andIOp.getResult());

  return andIOp;
}

/// Performs the MuxPasserSwap, swapping the MuxOp and PasserOp, and updating
/// the select operand and control of the MuxOp and PasserOp.
/// Returns the PasserOp that was swapped.
static PasserOp performMuxPasserSwap(MuxOp muxOp) {
  OpBuilder builder(muxOp.getContext());
  builder.setInsertionPoint(muxOp);

  auto passerOp =
      dyn_cast<PasserOp>(muxOp.getDataOperands()[1].getDefiningOp());

  // Materialization is required for swapping
  assertMaterialization(passerOp.getResult());

  // Swap mux and passer
  muxOp.getDataOperandsMutable()[1].set(passerOp.getData());
  passerOp.getDataMutable()[0].set(muxOp.getResult());
  muxOp.getResult().replaceAllUsesExcept(passerOp.getResult(), passerOp);

  InitOp init = cast<InitOp>(muxOp.getSelectOperand().getDefiningOp());
  assertMaterialization(init.getOperand());
  // Add repeating inits
  SpecV2RepeatingInitOp ri1 = builder.create<SpecV2RepeatingInitOp>(
      muxOp->getLoc(), init.getOperand(), 1);
  inheritBB(init, ri1);
  init.getOperandMutable()[0].set(ri1.getResult());

  assertMaterialization(passerOp.getCtrl());
  SpecV2RepeatingInitOp ri2 = builder.create<SpecV2RepeatingInitOp>(
      muxOp->getLoc(), passerOp.getCtrl(), 1);
  inheritBB(passerOp, ri2);
  passerOp.getCtrlMutable()[0].set(ri2.getResult());

  // Materialize passer's result for further rewriting.
  materializeValue(passerOp.getResult());

  return passerOp;
}

static SpecV2RepeatingInitOp moveRepeatingInitsUp(Value fedValue) {
  fedValue = getForkTop(fedValue);
  materializeValue(fedValue);

  Operation *uniqueUser = getUniqueUser(fedValue);
  if (auto ri = dyn_cast<SpecV2RepeatingInitOp>(uniqueUser)) {
    // Single use. No need to move repeating inits.
    return ri;
  }

  OpBuilder builder(fedValue.getContext());
  builder.setInsertionPointAfterValue(fedValue);

  ForkOp fork = cast<ForkOp>(uniqueUser);

  auto newRI =
      builder.create<SpecV2RepeatingInitOp>(fedValue.getLoc(), fedValue, 1);
  inheritBB(fedValue.getDefiningOp(), newRI);
  fedValue.replaceAllUsesExcept(newRI.getResult(), newRI);

  for (auto res : fork->getResults()) {
    if (auto ri = dyn_cast<SpecV2RepeatingInitOp>(getUniqueUser(res))) {
      ri.getResult().replaceAllUsesWith(newRI.getResult());
      ri->erase();
    } else {
      res.replaceAllUsesWith(fedValue);
    }
  }

  fork->erase();

  materializeValue(fedValue);
  materializeValue(newRI.getResult());

  return newRI;
}

/// Returns if the simplification of 3 passers is possible.
static bool isPasserSimplifiable(PasserOp ctrlDefiningPasser) {
  // Ensure the structure
  Operation *bottomOp = getUniqueUser(ctrlDefiningPasser.getResult());
  if (!isa<PasserOp>(bottomOp)) {
    llvm::errs() << "Bottom passer not found\n";
    return false;
  }
  auto bottomPasser = cast<PasserOp>(bottomOp);
  if (ctrlDefiningPasser.getResult() != bottomPasser.getCtrl()) {
    llvm::errs() << "Ctrl mismatch\n";
    return false;
  }

  Operation *topOp = bottomPasser.getData().getDefiningOp();
  if (!isa<PasserOp>(topOp)) {
    llvm::errs() << "Top passer not found\n";
    return false;
  }
  auto topPasser = cast<PasserOp>(topOp);

  // Confirm the context
  if (!equalsIndirectly(ctrlDefiningPasser.getCtrl(), topPasser.getCtrl())) {
    llvm::errs() << "Ctrl mismatch\n";
    return false;
  }

  return true;
}

/// Simplify 3 passers into a single one.
static AndIOp simplifyPasser(PasserOp ctrlDefiningPasser) {
  OpBuilder builder(ctrlDefiningPasser.getContext());
  builder.setInsertionPoint(ctrlDefiningPasser);

  AndIOp andOp = builder.create<AndIOp>(builder.getUnknownLoc(),
                                        ctrlDefiningPasser.getData(),
                                        ctrlDefiningPasser.getCtrl());
  inheritBB(ctrlDefiningPasser, andOp);

  auto bottomPasser =
      cast<PasserOp>(getUniqueUser(ctrlDefiningPasser.getResult()));
  auto topPasser = cast<PasserOp>(bottomPasser.getData().getDefiningOp());

  bottomPasser.getCtrlMutable()[0].set(andOp.getResult());
  bottomPasser.getDataMutable()[0].set(topPasser.getData());

  topPasser->erase();
  ctrlDefiningPasser->erase();

  materializeValue(andOp.getLhs());
  materializeValue(andOp.getRhs());
  materializeValue(andOp.getResult());

  return andOp;
}

static AndIOp moveAndIUp(AndIOp candidateAndI) {
  Value lhs = getForkTop(candidateAndI.getLhs());
  Value rhs = getForkTop(candidateAndI.getRhs());

  materializeValue(lhs);

  Operation *lhsUniqueUser = getUniqueUser(lhs);

  if (!isa<ForkOp>(lhsUniqueUser)) {
    return cast<AndIOp>(lhsUniqueUser);
  }

  ForkOp lhsFork = cast<ForkOp>(lhsUniqueUser);

  OpBuilder builder(lhs.getContext());
  builder.setInsertionPointAfterValue(rhs);

  AndIOp newAndI = builder.create<AndIOp>(lhs.getLoc(), lhs, rhs);
  inheritBB(lhsFork, newAndI);

  newAndI->setAttrs(candidateAndI->getAttrs());

  for (auto res : lhsFork->getResults()) {
    bool rewritten = false;
    if (auto andI = dyn_cast<AndIOp>(getUniqueUser(res))) {
      if ((andI.getLhs() == res && equalsIndirectly(andI.getRhs(), rhs)) ||
          (andI.getRhs() == res && equalsIndirectly(andI.getLhs(), rhs))) {
        andI.getResult().replaceAllUsesWith(newAndI.getResult());
        andI->erase();
        rewritten = true;
      }
    }
    if (!rewritten) {
      res.replaceAllUsesWith(lhs);
    }
  }

  materializeValue(lhs);
  materializeValue(rhs);
  materializeValue(newAndI.getResult());

  return newAndI;
}

void HandshakeSpeculationV2Pass::runDynamaticPass() {
  // Parse json (jsonPath is a member variable handled by tablegen)
  auto bbOrFailure = readFromJSON(jsonPath);
  if (failed(bbOrFailure))
    return signalPassFailure();

  auto [loopBBs] = bbOrFailure.value();

  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();
  OpBuilder builder(funcOp->getContext());

  DenseSet<PasserOp> frontiers;
  // n is a member variable handled by tablegen.
  // Storing repeating inits for post-processing.
  SmallVector<SpecV2RepeatingInitOp> repeatingInits(n);

  if (!disableInitialMotion) {
    frontiers.clear();

    for (auto passer : funcOp.getOps<PasserOp>()) {
      if (llvm::find(loopBBs, getLogicBB(passer)) == loopBBs.end())
        continue;
      frontiers.insert(passer);
    }

    bool frontiersUpdated;
    do {
      frontiersUpdated = false;
      for (auto passerOp : frontiers) {
        if (llvm::find(loopBBs, getLogicBB(getUniqueUser(
                                    passerOp.getResult()))) == loopBBs.end()) {
          // The passer is exiting the loop. Do not move it.
          // (Exit passer motion is performed later if enabled)
          continue;
        }

        if (isEligibleForPasserMotionOverPM(passerOp)) {
          performPasserMotionPastPM(passerOp, frontiers);
          frontiersUpdated = true;
          // If frontiers are updated, the iterator is outdated.
          // Break and restart the loop.
          break;
        }
      }
      // If no frontiers were updated, we can stop.
    } while (frontiersUpdated);

    do {
      frontiersUpdated = false;
      for (auto passer : frontiers) {
        if (isPasserSimplifiable(passer)) {
          auto bottomPasser = cast<PasserOp>(getUniqueUser(passer.getResult()));
          auto topPasser =
              cast<PasserOp>(bottomPasser.getData().getDefiningOp());
          frontiers.erase(bottomPasser);
          frontiers.erase(topPasser);
          frontiers.erase(passer);
          AndIOp andIOp = simplifyPasser(passer);
          frontiers.insert(cast<PasserOp>(getUniqueUser(andIOp.getResult())));
          moveAndIUp(andIOp);
          frontiersUpdated = true;
          break;
        }
      }
    } while (frontiersUpdated);

    bool muxUpdated;
    do {
      muxUpdated = false;
      for (auto muxOp : funcOp.getOps<MuxOp>()) {
        if (llvm::find(loopBBs, getLogicBB(muxOp)) == loopBBs.end())
          continue;

        if (isMuxPasserToAndEligible(muxOp)) {
          AndIOp andIOp = muxPasserToAnd(muxOp);
          moveAndIUp(andIOp);
          muxUpdated = true;
          break;
        }
      }
    } while (muxUpdated);
  }

  // Repeatedly move passers past Muxes and PMSC.
  for (unsigned i = 0; i < n; i++) {
    frontiers.clear();

    // Perform MuxPasserSwap for each Mux
    for (auto muxOp :
         llvm::make_early_inc_range(funcOp.getOps<handshake::MuxOp>())) {
      if (getLogicBB(muxOp) != loopBBs.front())
        continue;

      if (!isEligibleForMuxPasserSwap(muxOp)) {
        muxOp.emitWarning(
            "MuxOp is not eligible for Passer swap, exiting the pass.");
        return signalPassFailure();
      }

      auto passerOp = performMuxPasserSwap(muxOp);

      // Add the moved passer to the frontiers
      frontiers.insert(passerOp);
    }

    Value riResult = frontiers.begin()->getCtrl();
    auto newRI = moveRepeatingInitsUp(riResult.getDefiningOp()->getOperand(0));
    if (i == 0) {
      newRI->setAttr("specv2_top_ri", builder.getBoolAttr(true));
    }

    repeatingInits[i] = newRI;

    // Repeatedly move passers inside PMSC.
    bool frontiersUpdated;
    do {
      frontiersUpdated = false;
      for (auto passerOp : frontiers) {
        if (isEligibleForPasserMotionOverPM(passerOp)) {
          performPasserMotionPastPM(passerOp, frontiers);
          frontiersUpdated = true;
          // If frontiers are updated, the iterator is outdated.
          // Break and restart the loop.
          break;
        }
      }
      // If no frontiers were updated, we can stop.
    } while (frontiersUpdated);
  }

  if (n >= 2) {
    // // Reduce the passer chain by introducing interpolator op and performing
    // // induction.

    SmallVector<PasserOp> bottomPassers;
    for (Operation *user :
         iterateOverPossiblyIndirectUsers(repeatingInits[0].getResult())) {
      if (auto passer = dyn_cast<PasserOp>(user)) {
        bottomPassers.push_back(passer);
      }
    }

    SmallVector<AndIOp> andChain;
    for (unsigned i = 1; i < n; i++) {
      bool firstPasser = true;
      for (auto bottomPasser : bottomPassers) {
        auto topPasser = cast<PasserOp>(bottomPasser.getData().getDefiningOp());
        builder.setInsertionPoint(topPasser);

        // Place AND temporarily for buffering
        auto tmpAnd =
            builder.create<AndIOp>(builder.getUnknownLoc(),
                                   bottomPasser.getCtrl(), topPasser.getCtrl());
        inheritBB(topPasser, tmpAnd);
        tmpAnd->setAttr("specv2_tmp_and", builder.getBoolAttr(true));

        if (firstPasser) {
          andChain.push_back(tmpAnd);
          firstPasser = false;
        }

        bottomPasser.getCtrlMutable()[0].set(tmpAnd.getResult());

        topPasser.getResult().replaceAllUsesWith(topPasser.getData());
        topPasser->erase();
      }
    }

    for (auto andOp : andChain) {
      moveAndIUp(andOp);
    }
  }

  // Simplify the exit passers.
  SmallVector<PasserOp> ctrlDefiningPassers;
  for (auto passer : funcOp.getOps<PasserOp>()) {
    if (isPasserSimplifiable(passer)) {
      ctrlDefiningPassers.push_back(passer);
    }
  }
  frontiers.clear();
  for (auto passer : ctrlDefiningPassers) {
    AndIOp andIOp = simplifyPasser(passer);
    if (exitEagerEval) {
      PasserOp exitPasser = cast<PasserOp>(getUniqueUser(andIOp.getResult()));
      frontiers.insert(exitPasser);
    }
    moveAndIUp(andIOp);
  }
  if (exitEagerEval) {
    llvm::errs() << "Performing exit eager evaluation\n";
    bool frontiersUpdated;
    do {
      frontiersUpdated = false;
      for (auto passerOp : frontiers) {
        if (isEligibleForPasserMotionOverPM(passerOp)) {
          performPasserMotionPastPM(passerOp, frontiers);
          frontiersUpdated = true;
          // If frontiers are updated, the iterator is outdated.
          // Break and restart the loop.
          break;
        }
      }
      // If no frontiers were updated, we can stop.
    } while (frontiersUpdated);
  }

  DenseMap<unsigned, unsigned> bbMap = unifyBBs(loopBBs, funcOp);
  if (bbMapping != "") {
    // Convert bbMap to a json file
    std::ofstream csvFile(bbMapping);
    csvFile << "before,after\n";
    for (const auto &entry : bbMap) {
      csvFile << entry.first << "," << entry.second << "\n";
    }
    csvFile.close();
  }

  recalculateMCBlocks(funcOp);
}
