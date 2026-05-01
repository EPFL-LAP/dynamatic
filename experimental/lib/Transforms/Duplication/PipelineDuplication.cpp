#include "experimental/Transforms/Duplication/DuplicationLogic.h"

// Include some other useful headers.
#include "dynamatic/Dialect/Handshake/HandshakeOps.h" // maybe use the other dialect?
#include "dynamatic/Support/CFG.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_HANDSHAKERIGIDIFICATION
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]
namespace {
  /*
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
  */


  // wrapper
  struct PipelineDuplicationPass 
      : public dynamatic::experimental::impl::PipelineDuplicationBase<
      PipelineDuplicationPass> {

    using PipelineDuplicationBase::PipelineDuplicationBase;

    void runDynamaticPass() override {
      mlir::ModuleOp mod = getOperation();
      MLIRContext *ctx = &getContext();

      RewritePatternSet patterns{ctx};
      // patterns.add<AddConstantBranch>(ctx);
      patterns.add<ReplaceMuxWithMerge>(ctx);


      mlir::GreedyRewriteConfig config;
      if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
        return signalPassFailure();
    };
  };
  
} // namespace


/// Rewrite pattern that will match on all muxes in the IR and replace each of
/// them with a merge taking the same inputs (except the `select` input which
/// merges do not have due to their undeterministic nature).
/// Code taken from the tutorial. Change back when the project order works and 
/// it compiles without errors
struct ReplaceMuxWithMerge : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {
    // Retrieve all mux inputs except the `select`
    ValueRange dataOperands = muxOp.getDataOperands();
    // Create a merge in the IR at the mux's position and with the same data
    // inputs (or operands, in MLIR jargon)
    handshake::MergeOp mergeOp =
        rewriter.create<handshake::MergeOp>(muxOp.getLoc(), dataOperands);
    // Make the merge part of the same basic block (BB) as the mux
    inheritBB(muxOp, mergeOp);
    // Retrieve the merge's output (or result, in MLIR jargon)
    Value mergeResult = mergeOp.getResult();
    // Replace usages of the mux's output with the new merge's output
    rewriter.replaceOp(muxOp, mergeResult);
    // Signal that the pattern succeeded in rewriting the mux
    return success();
  }
};
