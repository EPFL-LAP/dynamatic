#include "SpecV2GammaPostBuffering.h"
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
#define GEN_PASS_DEF_SPECV2GAMMAPOSTBUFFERING
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct SpecV2GammaPostBufferingPass
    : public dynamatic::experimental::speculationv2::impl::
          SpecV2GammaPostBufferingBase<SpecV2GammaPostBufferingPass> {
  using SpecV2GammaPostBufferingBase<
      SpecV2GammaPostBufferingPass>::SpecV2GammaPostBufferingBase;
  void runDynamaticPass() override;
};

void SpecV2GammaPostBufferingPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  OpBuilder builder(funcOp->getContext());
  for (auto passer : llvm::make_early_inc_range(funcOp.getOps<PasserOp>())) {
    if (passer->hasAttr("specv2_gsa_mux_tmp")) {
      auto gsaIdAttr = passer->getAttrOfType<IntegerAttr>("specv2_gsa_mux_tmp");
      if (!gsaIdAttr) {
        passer.emitError("specv2_gsa_mux_tmp attribute is not an integer");
        return signalPassFailure();
      }
      unsigned gsaId = gsaIdAttr.getValue().getZExtValue();
      SinkOp sink;
      bool found = false;
      for (auto sink_ : llvm::make_early_inc_range(funcOp.getOps<SinkOp>())) {
        if (sink_->hasAttr("specv2_gsa_mux_nonpri")) {
          auto sinkGsaIdAttr =
              sink_->getAttrOfType<IntegerAttr>("specv2_gsa_mux_nonpri");
          if (!sinkGsaIdAttr) {
            sink_.emitError(
                "specv2_gsa_mux_nonpri attribute is not an integer");
            return signalPassFailure();
          }
          unsigned sinkGsaId = sinkGsaIdAttr.getValue().getZExtValue();
          if (sinkGsaId == gsaId) {
            sink = sink_;
            found = true;
            break;
          }
        }
      }
      if (!found) {
        passer.emitError("Could not find matching sink for the passer");
        return signalPassFailure();
      }
      builder.setInsertionPoint(passer);
      BufferOp breakR = builder.create<BufferOp>(
          builder.getUnknownLoc(), sink.getOperand(), 1,
          dynamatic::handshake::BufferType::ONE_SLOT_BREAK_R);
      inheritBB(passer, breakR);
      BufferOp breakDV = builder.create<BufferOp>(
          builder.getUnknownLoc(), breakR.getResult(), 1,
          dynamatic::handshake::BufferType::ONE_SLOT_BREAK_DV);
      inheritBB(passer, breakDV);
      MuxOp newMux;
      if (prioritizedSide == 0) {
        newMux = builder.create<MuxOp>(
            builder.getUnknownLoc(), passer.getData().getType(),
            passer.getCtrl(),
            ArrayRef<Value>{passer.getData(), breakDV.getResult()});
      } else {
        newMux = builder.create<MuxOp>(
            builder.getUnknownLoc(), passer.getData().getType(),
            passer.getCtrl(),
            ArrayRef<Value>{breakDV.getResult(), passer.getData()});
      }
      inheritBB(passer, newMux);
      newMux->setAttr("specv2_gsa_mux", builder.getUnitAttr());
      passer.getResult().replaceAllUsesWith(newMux.getResult());
      passer->erase();
      sink->erase();
    }
  }
}
