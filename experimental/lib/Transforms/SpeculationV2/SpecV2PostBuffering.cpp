#include "SpecV2PostBuffering.h"
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
#define GEN_PASS_DEF_SPECV2POSTBUFFERING
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct SpecV2PostBufferingPass
    : public dynamatic::experimental::speculationv2::impl::
          SpecV2PostBufferingBase<SpecV2PostBufferingPass> {
  using SpecV2PostBufferingBase<
      SpecV2PostBufferingPass>::SpecV2PostBufferingBase;
  void runDynamaticPass() override;
};

void SpecV2PostBufferingPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();
  OpBuilder builder(funcOp->getContext());

  for (auto andOp : llvm::make_early_inc_range(funcOp.getOps<AndIOp>())) {
    if (!andOp->hasAttr("specv2_tmp_and"))
      continue;
    builder.setInsertionPoint(andOp);

    auto src = builder.create<SourceOp>(builder.getUnknownLoc());
    inheritBB(andOp, src);

    auto cst = builder.create<ConstantOp>(
        builder.getUnknownLoc(), IntegerAttr::get(builder.getIntegerType(1), 0),
        src.getResult());
    inheritBB(andOp, cst);

    auto muxOp = builder.create<MuxOp>(
        builder.getUnknownLoc(), andOp.getLhs().getType(), andOp.getRhs(),
        ArrayRef<Value>{cst.getResult(), andOp.getLhs()});
    inheritBB(andOp, muxOp);

    andOp.getResult().replaceAllUsesWith(muxOp.getResult());
    andOp->erase();
  }
}
