#include "SpecV2Gamma.h"
#include "JSONImporter.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/MaterializationUtil/MaterializationUtil.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
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
#define GEN_PASS_DEF_SPECV2GAMMA
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct SpecV2GammaPass
    : public dynamatic::experimental::speculationv2::impl::SpecV2GammaBase<
          SpecV2GammaPass> {
  using SpecV2GammaBase<SpecV2GammaPass>::SpecV2GammaBase;
  void runDynamaticPass() override;
};

/// Returns if the value is driven by a SourceOp
static bool isSourced(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return false;

  // Heuristic to stop the traversal earlier.
  if (isa<handshake::MuxOp>(definingOp))
    return false;

  if (isa<SourceOp>(value.getDefiningOp()))
    return true;

  // If all operands of the defining operation are sourced, the value is also
  // sourced.
  return llvm::all_of(value.getDefiningOp()->getOperands(),
                      [](Value v) { return isSourced(v); });
}

/// If op is LoadOp, excludes operands coming from MemoryControllerOp.
static llvm::SmallVector<Value> getEffectiveOperands(Operation *op) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // For LoadOp, only the data result is effective for rewriting
    return {loadOp.getAddress()};
  }
  return llvm::to_vector(op->getOperands());
}

/// If op is LoadOp, excludes results going to MemoryControllerOp.
static llvm::SmallVector<Value> getEffectiveResults(Operation *op) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    // For LoadOp, only the data result is effective for rewriting
    return {loadOp.getDataResult()};
  }
  // Unlike the operands, to_vector doesn't work
  llvm::SmallVector<Value> results;
  for (OpResult result : op->getResults()) {
    results.push_back(result);
  }
  return results;
}

static LogicalResult movePassersDownPM(Operation *pmOp) {
  OpBuilder builder(pmOp->getContext());
  builder.setInsertionPoint(pmOp);

  Location loc = builder.getUnknownLoc();
  Value ctrl = nullptr;

  // Remove PasserOp from each effective operand of the PM unit.
  for (Value operand : getEffectiveOperands(pmOp)) {
    Operation *definingOp = operand.getDefiningOp();
    if (auto passer = dyn_cast<PasserOp>(definingOp)) {
      loc = passer->getLoc();
      ctrl = passer.getCtrl();

      // The operand must be materialized to perform the motion correctly.
      assertMaterialization(operand);

      // Remove the defining PasserOp operation.
      passer.getResult().replaceAllUsesWith(passer.getData());
      passer->erase();
    } else {
      // If the operand is sourced, it doesn't need to be defined by OpT.
      if (isSourced(operand))
        continue;
      return pmOp->emitError(
          "Expected all operands to be defined by the PasserOp");
    }
  }

  // Add new PasserOp for each effective result of the PM unit.
  for (Value result : getEffectiveResults(pmOp)) {
    assertMaterialization(result);

    PasserOp newPasser = builder.create<PasserOp>(loc, result, ctrl);
    inheritBB(pmOp, newPasser);

    result.replaceAllUsesExcept(newPasser.getResult(), newPasser);
  }

  return success();
}

/// Returns if the specified PasserOp is eligible for motion past a PM unit.
static bool isEligibleForPasserMotionOverPM(PasserOp passerOp) {
  Value passerControl = passerOp.getCtrl();

  Operation *targetOp = getUniqueUser(passerOp.getResult());

  // If the targetOp is not a PM unit, return false.
  if (!isa<ArithOpInterface, NotOp, ForkOp, LazyForkOp, BufferOp, LoadOp,
           BranchOp>(targetOp)) {
    if (!isa<MergeOp, ControlMergeOp>(targetOp) ||
        targetOp->getNumOperands() != 1)
      return false;
  }

  // Iterate over operands of the targetOp to decide the eligibility for
  // motion.
  for (Value operand : getEffectiveOperands(targetOp)) {
    if (auto passerOp = dyn_cast<PasserOp>(operand.getDefiningOp())) {
      // If this passerOp is controlled by different control from the specified
      // one, not eligible.
      if (!equalsIndirectly(passerControl, passerOp.getCtrl()))
        return false;
    } else if (!isSourced(operand)) {
      // Each operand must be defined by a passer, except when it is driven by a
      // source op.
      return false;
    }
  }

  return true;
}

static DenseMap<unsigned, unsigned> unifyBBs(ArrayRef<unsigned> loopBBs,
                                             FuncOp funcOp) {
  DenseMap<unsigned, unsigned> bbMap;
  unsigned minBB = *std::min_element(loopBBs.begin(), loopBBs.end());
  funcOp.walk([&](Operation *op) {
    auto bbOrNull = getLogicBB(op);
    if (!bbOrNull.has_value())
      return;

    unsigned bb = bbOrNull.value();
    if (!bbMap.contains(bb)) {
      if (std::find(loopBBs.begin(), loopBBs.end(), bb) != loopBBs.end()) {
        bbMap[bb] = minBB;
      } else {
        unsigned d = 0;
        for (auto loopBB : loopBBs) {
          if (loopBB == minBB)
            continue;
          if (loopBB < bb)
            d++;
        }
        bbMap[bb] = bb - d;
      }
    }

    setBB(op, bbMap[bb]);
  });

  return bbMap;
}

/// Move the specified passer past a PM unit.
static void performPasserMotionPastPM(PasserOp passerOp,
                                      DenseSet<PasserOp> &frontiers) {
  Value passerControl = passerOp.getCtrl();
  OpBuilder builder(passerOp->getContext());
  builder.setInsertionPoint(passerOp);

  Operation *targetOp = getUniqueUser(passerOp.getResult());

  // Remove passers from the frontiers
  for (Value operand : getEffectiveOperands(targetOp)) {
    if (auto passerOp = dyn_cast<PasserOp>(operand.getDefiningOp())) {
      frontiers.erase(passerOp);
    }
  }

  // Perform the motion
  auto motionResult = movePassersDownPM(targetOp);

  if (failed(motionResult)) {
    targetOp->emitError("Failed to perform motion for PasserOp");
    llvm_unreachable("SpeculationV2 algorithm failed");
  }

  // Add new passers to the frontiers
  for (auto result : getEffectiveResults(targetOp)) {
    auto newPasser = cast<PasserOp>(getUniqueUser(result));
    frontiers.insert(newPasser);
    // Materialize the result of the new passer for further rewriting.
    materializeValue(newPasser.getResult());
  }

  materializeValue(passerControl);
}

static void recalculateMCBlocks(FuncOp funcOp) {
  DenseSet<int32_t> bbs;
  OpBuilder builder(funcOp->getContext());

  for (auto mc :
       llvm::make_early_inc_range(funcOp.getOps<MemoryControllerOp>())) {
    bbs.clear();
    for (auto oprd : mc->getOperands()) {
      if (isa<ControlType>(oprd.getType()))
        continue;
      if (oprd.getDefiningOp()) {
        if (auto bbOrNull = getLogicBB(oprd.getDefiningOp())) {
          bbs.insert(bbOrNull.value());
        }
      }
    }
    for (auto res : mc->getResults()) {
      if (isa<ControlType>(res.getType()))
        continue;
      for (auto *user : res.getUsers()) {
        if (auto bbOrNull = getLogicBB(user)) {
          bbs.insert(bbOrNull.value());
        }
      }
    }
    auto i32Attr = builder.getI32ArrayAttr(llvm::to_vector(bbs));
    mc.setConnectedBlocksAttr(i32Attr);
  }
}

static bool tryErasePasser(PasserOp passer) {
  Value result = passer.getResult();

  if (result.use_empty()) {
    passer->erase();
    return true;
  }
  assert(result.hasOneUse());
  Operation *user = getUniqueUser(result);
  if (isa<SinkOp>(user)) {
    user->erase();
    passer->erase();
    return true;
  }
  if (auto childPasser = dyn_cast<PasserOp>(user)) {
    if (tryErasePasser(childPasser)) {
      passer->erase();
      return true;
    }
  }
  return false;
}

void SpecV2GammaPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  DenseSet<PasserOp> frontiers;
  for (auto passer : funcOp.getOps<PasserOp>()) {
    frontiers.insert(passer);
  }

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

  OpBuilder builder(funcOp->getContext());
  if (oneSided) {
    unsigned side = 0; // 0: false side, 1: true side
    MuxOp candidate;
    for (auto frontier : frontiers) {
      getUniqueUser(frontier.getResult())->dump();
      if (auto mux = dyn_cast<MuxOp>(getUniqueUser(frontier.getResult()))) {
        candidate = mux;
        break;
      }
    }
    assert(candidate && "No Mux found at the frontier");
    Value index = getForkTop(candidate.getSelectOperand());
    Value cond = index;
    if (side == 0) {
      NotOp notOp;
      for (Operation *user : iterateOverPossiblyIndirectUsers(cond)) {
        if (auto n = dyn_cast<NotOp>(user)) {
          notOp = n;
          break;
        }
      }
      assert(notOp && "No NotOp found for the Mux condition");
      cond = notOp.getResult();
    }
    builder.setInsertionPointAfterValue(cond);
    auto ri =
        builder.create<SpecV2RepeatingInitOp>(builder.getUnknownLoc(), cond, 1);
    inheritBB(cond.getDefiningOp(), ri);
    Value newIndex = ri.getResult();
    if (side == 0) {
      NotOp newNotOp = builder.create<NotOp>(builder.getUnknownLoc(), newIndex);
      inheritBB(ri, newNotOp);
      newIndex = newNotOp.getResult();
    }
    for (Operation *user : iterateOverPossiblyIndirectUsers(index)) {
      if (auto mux = dyn_cast<MuxOp>(user)) {
        mux.getSelectOperandMutable()[0].set(newIndex);
      }
    }

    auto src = builder.create<SourceOp>(builder.getUnknownLoc());
    inheritBB(ri, src);

    auto cst = builder.create<ConstantOp>(
        builder.getUnknownLoc(), IntegerAttr::get(builder.getIntegerType(1), 1),
        src.getResult());
    inheritBB(ri, cst);

    MuxOp newCondGenMux;
    if (side == 0) {
      newCondGenMux = builder.create<MuxOp>(
          builder.getUnknownLoc(), cond.getType(), newIndex,
          ArrayRef<Value>{cond, cst.getResult()});
    } else {
      newCondGenMux = builder.create<MuxOp>(
          builder.getUnknownLoc(), cond.getType(), newIndex,
          ArrayRef<Value>{cst.getResult(), cond});
    }
    inheritBB(ri, newCondGenMux);

    for (Operation *user : iterateOverPossiblyIndirectUsers(newIndex)) {
      if (auto mux = dyn_cast<MuxOp>(user)) {
        if (mux == newCondGenMux)
          continue;
        Operation *dataDefOp = mux.getDataOperands()[side].getDefiningOp();
        if (auto passer = dyn_cast<PasserOp>(dataDefOp)) {
          if (equalsIndirectly(passer.getCtrl(), cond)) {
            auto newPasser = builder.create<PasserOp>(
                builder.getUnknownLoc(), mux.getResult(),
                newCondGenMux.getResult());
            inheritBB(mux, newPasser);
            mux.getResult().replaceAllUsesExcept(newPasser.getResult(),
                                                 newPasser);
            passer.getResult().replaceAllUsesWith(passer.getData());
            frontiers.insert(newPasser);
            frontiers.erase(passer);
            passer->erase();
          } else {
            llvm::errs() << "mux dataT passer ctrl not equal to cond\n";
          }
        } else {
          llvm::errs() << "mux dataT not passer\n";
        }
      }
    }

    materializeValue(newCondGenMux.getResult());

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

  // Erase unused PasserOps
  for (auto passerOp : llvm::make_early_inc_range(funcOp.getOps<PasserOp>())) {
    tryErasePasser(passerOp);
  }

  SmallVector<unsigned> loopBBs;
  for (unsigned bb = branchBB; bb <= mergeBB; bb++) {
    loopBBs.push_back(bb);
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
