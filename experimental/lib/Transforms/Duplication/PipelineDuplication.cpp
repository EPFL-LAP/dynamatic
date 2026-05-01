// Include some other useful headers.
#include "dynamatic/Analysis/NameAnalysis.h" // needed
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/FormalProperty.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <ostream>

using namespace llvm;
using namespace dynamatic;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_PIPELINEDUPLICATION
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]
namespace {

struct PipelineDuplicationPass 
    : public dynamatic::experimental::impl::PipelineDuplicationBase<
    PipelineDuplicationPass> {

  using PipelineDuplicationBase::PipelineDuplicationBase;

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Find addf0 operation
    OpBuilder builder(ctx);
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    Operation *rawOp = namer.getOp("addf0");
    if (!rawOp) {
      llvm::errs() << "No operation named \"addf0\" exists\n";
      return signalPassFailure();
    }
    auto op = dyn_cast<mlir::arith::AddFOp>(rawOp);
    if (!op) return signalPassFailure();

    Location loc = op.getLoc();

    // Find the preceding MulFOp to extract constants
    auto mulfOp = op.getLhs().getDefiningOp<mlir::arith::MulFOp>();
    if (!mulfOp)
      return signalPassFailure();

    Value cnstNegTwo = mulfOp.getRhs();
    Value cnstFifteen = op.getRhs();

    // Set insertion point after the original add operation (necessary?)
    builder.setInsertionPointAfter(op);

    // Create new duplicated pipeline logic
    Value cnstFive = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getFloatAttr(builder.getF64Type(), 5.0));

    // is .getResult() needed?
    Value newMulf =
        builder.create<mlir::arith::MulFOp>(loc, cnstFive, cnstNegTwo);
    Value newAddf =
        builder.create<mlir::arith::AddFOp>(loc, newMulf, cnstFifteen);
    Value newTrunc =
        builder.create<mlir::arith::TruncFOp>(loc, builder.getF32Type(), newAddf);

    // Navigate the IR to find the store operation downstream
    for (auto *user : op.getResult().getUsers()) {
      if (auto truncOp = dyn_cast<mlir::arith::TruncFOp>(user)) {
        for (auto *truncUser : truncOp.getResult().getUsers()) {
          if (auto storeOp = dyn_cast<mlir::memref::StoreOp>(truncUser)) {
            Value sharedIndex = storeOp.getIndices()[0];
            Value targetMemref = storeOp.getMemref();

            // Create the new duplicated store branch
            auto newStore = builder.create<mlir::memref::StoreOp>(
                loc, newTrunc, targetMemref, sharedIndex);
            
            // Inherit Basic Block information
            inheritBB(storeOp, newStore);
            break;
          }
        }
      }
    }
  }
};
  
} // namespace
