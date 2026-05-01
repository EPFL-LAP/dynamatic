#include "experimental/Transforms/Duplication/DuplicationLogic.h"

// Include some other useful headers.
#include "dynamatic/Dialect/Handshake/HandshakeOps.h" // maybe use the other dialect?
#include "dynamatic/Support/CFG.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_DUPLICATIONLOGIC
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

namespace {
  struct AddConstantBranch : public OpRewritePattern<arith::AddfOp> {
    using OpRewritePattern<arith::AddfOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::AddfOp op, PatternRewriter &rewriter) const override {
      // Find the end of my pipeline with addf, to get the same constants
      auto nameAttr = op->getAttrOfType<StringAttr>("handshake.name");
      if (!nameAttr || nameAttr.getValue() != "addf0")
        return failure();

      if (op->hasAttr("processed")) // is this necessary?
        return failure();
    
      Location loc = op.getLoc();

      // get the other constants
      auto mulfOp = op.getLhs().getDefiningOp<arith::MulfOp>();
      if (!mulfOp) return failure();

      // constants -2.0 and 15.0 that are used for the other operations
      Value cnstNegTwo = mulfOp.getRhs();
      Value cnstFifteen = op.getRhs();

      // actually create the new values
      Value cnstFive = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(rewriter.getF64Type(), 5.0));

      Value newMulf = rewriter.create<arith::MulfOp>(loc, cstFive, cstNegTwo).getResult();
      Value newAddf = rewriter.create<arith::AddfOp>(loc, newMulf, cstFifteen).getResult();
      Value newTrunc = rewriter.create<arith::TruncfOp>(
        loc, rewriter.getF32Type(), newAddf).getResult();

      
      for (auto *user : op.getResult().getUsers()) {
      if (auto truncOp = dyn_cast<arith::TruncfOp>(user)) {
        for (auto *truncUser : truncOp.getResult().getUsers()) {
          if (auto storeOp = dyn_cast<memref::StoreOp>(truncUser)) {
            
            Value sharedIndex = storeOp.getIndices()[0]; 
            Value targetMemref = storeOp.getMemref();

            // Create the 4th branch store
            rewriter.create<memref::StoreOp>(loc, newTrunc, targetMemref, sharedIndex);
            break;
          }
        }
      }
      }
      op->setAttr("processed", rewriter.getUnitAttr());
      return success();
    }
  };


  // wrapper
  struct PipelineDuplicationPass 
      : public dynamatic::impl::PiplineDuplicationBase<PipelineDuplicationPass> {

    void runDynamaticPass() override {
      MLIRContext *ctx = &getContext();
      RewritePatternSet patterns{ctx};
      patterns.add<AddConstantBranch>(ctx);

      if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
        signalPassFailure();
    };
  };
  
} // namespace

/// Implementation of the pass constructor
std::unique_ptr<dynamatic::DynamaticPass> dynamatic::createPipelineDuplicationPass() {
  return std::make_unique<PipelineDuplicationPass>();
}
