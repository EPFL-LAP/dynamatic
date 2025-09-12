#include "SpecV2CutCondDep.h"
#include "JSONImporter.h"
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
#define GEN_PASS_DEF_SPECV2CUTCONDDEP
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct SpecV2CutCondDepPass
    : public dynamatic::experimental::speculationv2::impl::SpecV2CutCondDepBase<
          SpecV2CutCondDepPass> {
  using SpecV2CutCondDepBase<SpecV2CutCondDepPass>::SpecV2CutCondDepBase;
  void runDynamaticPass() override;
};

void SpecV2CutCondDepPass::runDynamaticPass() {
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

  unsigned headBB = loopBBs[0];

  OpBuilder builder(funcOp->getContext());

  for (auto mux : funcOp.getOps<MuxOp>()) {
    if (getLogicBB(mux) != headBB)
      continue;

    auto initOp = cast<InitOp>(mux.getSelectOperand().getDefiningOp());
    Value loopContinue = getForkTop(initOp.getOperand());
    builder.setInsertionPointAfterValue(loopContinue);

    SourceOp src = builder.create<SourceOp>(builder.getUnknownLoc());
    setBB(src, headBB);
    ConstantOp cst = builder.create<ConstantOp>(
        builder.getUnknownLoc(), IntegerAttr::get(builder.getIntegerType(1), 1),
        src);
    setBB(cst, headBB);

    loopContinue.replaceAllUsesWith(cst.getResult());
    break;
  }
}
