#include "PreSpecV2GammaRewriteA.h"
#include "JSONImporter.h"
#include "SpecV2Lib.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

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
#define GEN_PASS_DEF_PRESPECV2GAMMAREWRITEA
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct PreSpecV2GammaRewriteAPass
    : public dynamatic::experimental::speculationv2::impl::
          PreSpecV2GammaRewriteABase<PreSpecV2GammaRewriteAPass> {
  using PreSpecV2GammaRewriteABase<PreSpecV2GammaRewriteAPass>::PreSpecV2GammaRewriteABase;
  void runDynamaticPass() override;
};

void PreSpecV2GammaRewriteAPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  // introduceGSAMux(funcOp, branchBB);

  OpBuilder builder(funcOp->getContext());
  // for (auto mux : funcOp.getOps<MuxOp>()) {
  //   if (getLogicBB(mux) != mergeBB)
  //     continue;

  //   mux->setAttr("specv2_loop_cond_mux", builder.getBoolAttr(true));
  //   mux->setAttr("specv2_gsa_mux", builder.getUnitAttr());
  // }

  if (failed(replaceBranchesWithPassers(funcOp, branchBB))) {
    funcOp.emitError("Failed to replace branches in BB 1 with passers");
    return signalPassFailure();
  }
}
