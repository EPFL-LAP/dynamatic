#include "SpecV2GammaKmp.h"
#include "JSONImporter.h"
#include "SpecV2Lib.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
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
#define GEN_PASS_DEF_SPECV2GAMMAKMP
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct SpecV2GammaKmpPass
    : public dynamatic::experimental::speculationv2::impl::SpecV2GammaKmpBase<
          SpecV2GammaKmpPass> {
  using SpecV2GammaKmpBase<SpecV2GammaKmpPass>::SpecV2GammaKmpBase;
  void runDynamaticPass() override;
};

void SpecV2GammaKmpPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  if (stepsUntil >= 1) {
    DenseSet<PasserOp> frontiers;
    for (auto passer : funcOp.getOps<PasserOp>()) {
      if (emulatePrediction) {
        Operation *ctrlDefOp = getIndirectDefiningOp(passer.getCtrl());
        if (isa<NotOp>(ctrlDefOp)) {
          if (prioritizedSide != 0)
            continue;
        } else {
          if (prioritizedSide != 1)
            continue;
        }
      }
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

    if (stepsUntil >= 2) {
      OpBuilder builder(funcOp->getContext());
      MuxOp candidate;
      for (auto frontier : frontiers) {
        if (auto mux = dyn_cast<MuxOp>(getUniqueUser(frontier.getResult()))) {
          candidate = mux;
          break;
        }
      }
      assert(candidate && "No Mux found at the frontier");
      Value index = getForkTop(candidate.getSelectOperand());

      if (oneSided) {
        Value cond = index;
        if (prioritizedSide == 0) {
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
        auto ri = builder.create<SpecV2RepeatingInitOp>(builder.getUnknownLoc(),
                                                        cond, 1);
        inheritBB(cond.getDefiningOp(), ri);
        Value newIndex = ri.getResult();
        if (prioritizedSide == 0) {
          NotOp newNotOp =
              builder.create<NotOp>(builder.getUnknownLoc(), newIndex);
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
            builder.getUnknownLoc(),
            IntegerAttr::get(builder.getIntegerType(1), 1), src.getResult());
        inheritBB(ri, cst);

        MuxOp newCondGenMux;
        if (prioritizedSide == 0) {
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
            Operation *dataDefOp =
                mux.getDataOperands()[prioritizedSide].getDefiningOp();
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
      } else {
        builder.setInsertionPointAfterValue(index);
        Value newIndex;

        BufferOp rBufOp = builder.create<BufferOp>(
            builder.getUnknownLoc(), /*tmp*/ index, 1,
            dynamatic::handshake::BufferType::ONE_SLOT_BREAK_R);
        inheritBB(candidate, rBufOp);

        InitOp init = builder.create<InitOp>(builder.getUnknownLoc(),
                                             rBufOp.getResult(), 0);
        inheritBB(candidate, init);

        NotOp notOp =
            builder.create<NotOp>(builder.getUnknownLoc(), init.getResult());
        inheritBB(init, notOp);
        rBufOp.getOperandMutable()[0].set(notOp.getResult());

        if (prioritizedSide == 0) {
          newIndex = init.getResult();
        } else {
          newIndex = notOp.getResult();
        }

        NotOp inverseIndex;
        for (Operation *user : iterateOverPossiblyIndirectUsers(index)) {
          if (auto n = dyn_cast<NotOp>(user)) {
            inverseIndex = n;
            break;
          }
        }
        assert(inverseIndex && "No NotOp found for the Mux condition");
        MuxOp newCondGenMux = builder.create<MuxOp>(
            builder.getUnknownLoc(), index.getType(), newIndex,
            ArrayRef<Value>{inverseIndex.getResult(), index});
        inheritBB(init, newCondGenMux);

        for (Operation *user : iterateOverPossiblyIndirectUsers(index)) {
          if (auto mux = dyn_cast<MuxOp>(user)) {
            if (mux == newCondGenMux)
              continue;

            mux.getSelectOperandMutable()[0].set(newIndex);
            PasserOp newPasser = builder.create<PasserOp>(
                builder.getUnknownLoc(), mux.getResult(),
                newCondGenMux.getResult());
            inheritBB(mux, newPasser);
            mux.getResult().replaceAllUsesExcept(newPasser.getResult(),
                                                 newPasser);
            frontiers.insert(newPasser);

            mux.getDataOperands()[1].getDefiningOp()->dump();
            PasserOp dataTPasser =
                cast<PasserOp>(mux.getDataOperands()[1].getDefiningOp());
            dataTPasser.getResult().replaceAllUsesWith(dataTPasser.getData());
            frontiers.erase(dataTPasser);
            dataTPasser->erase();

            mux.getDataOperands()[0].getDefiningOp()->dump();
            PasserOp dataFPasser =
                cast<PasserOp>(mux.getDataOperands()[0].getDefiningOp());
            dataFPasser.getResult().replaceAllUsesWith(dataFPasser.getData());
            frontiers.erase(dataFPasser);
            dataFPasser->erase();
          }
        }

        materializeValue(newCondGenMux.getResult());
      }

      if (stepsUntil >= 3) {
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

        // Erase unused PasserOps
        for (auto passerOp :
             llvm::make_early_inc_range(funcOp.getOps<PasserOp>())) {
          tryErasePasser(passerOp);
        }
      }
    }
  }

  // to prepare for the buffering, we cut the path
  unsigned gsaId = 0;
  OpBuilder builder(funcOp->getContext());
  for (auto mux : llvm::make_early_inc_range(funcOp.getOps<MuxOp>())) {
    if (mux->hasAttr("specv2_gsa_mux")) {
      unsigned nonPriSide = (prioritizedSide == 0) ? 1 : 0;
      builder.setInsertionPoint(mux);
      SinkOp sink = builder.create<SinkOp>(builder.getUnknownLoc(),
                                           mux.getDataOperands()[nonPriSide]);
      inheritBB(mux, sink);
      sink->setAttr("specv2_gsa_mux_nonpri",
                    builder.getIntegerAttr(builder.getIntegerType(32), gsaId));
      // sink->setAttr("specv2_gsa_side",
      //               builder.getIntegerAttr(builder.getIntegerType(32),
      //                                      prioritizedSide));
      PasserOp tmpPasser = builder.create<PasserOp>(
          builder.getUnknownLoc(), mux.getDataOperands()[prioritizedSide],
          mux.getSelectOperand());
      inheritBB(mux, tmpPasser);
      mux.getResult().replaceAllUsesWith(tmpPasser.getResult());
      tmpPasser->setAttr(
          "specv2_gsa_mux_tmp",
          builder.getIntegerAttr(builder.getIntegerType(32), gsaId));
      gsaId++;
      mux->erase();
    }
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
