#include "HandshakeCutBBsByBuffers.h"
#include "JSONImporter.h"
#include "dynamatic/Dialect/Handshake/HandshakeEnums.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
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
#define GEN_PASS_DEF_HANDSHAKECUTBBSBYBUFFERS
#include "experimental/Transforms/Passes.h.inc"

} // namespace speculationv2
} // namespace experimental
} // namespace dynamatic

struct HandshakeCutBBsByBuffersPass
    : public dynamatic::experimental::speculationv2::impl::
          HandshakeCutBBsByBuffersBase<HandshakeCutBBsByBuffersPass> {
  using HandshakeCutBBsByBuffersBase<
      HandshakeCutBBsByBuffersPass>::HandshakeCutBBsByBuffersBase;
  void runDynamaticPass() override;
};

void HandshakeCutBBsByBuffersPass::runDynamaticPass() {
  MLIRContext &context = getContext();
  ModuleOp mod = getOperation();
  FuncOp func = *mod.getOps<FuncOp>().begin();

  llvm::SmallVector<Value> valuesToCut;
  func.walk([&](Operation *op) {
    if (getLogicBB(op) == postBB) {
      for (auto &operand : op->getOpOperands()) {
        Operation *defOp = operand.get().getDefiningOp();
        if (defOp && getLogicBB(defOp) == preBB)
          valuesToCut.push_back(operand.get());
      }
    }
  });

  OpBuilder builder(&context);
  for (Value val : valuesToCut) {
    builder.setInsertionPointAfterValue(val);
    BufferOp breakDV = builder.create<BufferOp>(builder.getUnknownLoc(), val, 1,
                                                BufferType::ONE_SLOT_BREAK_DV);
    setBB(breakDV, preBB);
    BufferOp breakR =
        builder.create<BufferOp>(builder.getUnknownLoc(), breakDV.getResult(),
                                 1, BufferType::ONE_SLOT_BREAK_R);
    setBB(breakR, preBB);
    val.replaceAllUsesExcept(breakR.getResult(), breakDV);
  }
}
