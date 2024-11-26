/// Include the header we just created.
#include "experimental/Transforms/HandshakeRigidification.h"
#include "experimental/Support/RigidificationSupport.h"

/// Include some other useful headers.
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// #include "dynamatic/Analysis/NameAnalysis.h"
// #include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
// #include "dynamatic/Dialect/Handshake/HandshakeOps.h"
// #include "dynamatic/Support/CFG.h"
#include "experimental/Transforms/Passes.h.inc"
// #include "experimental/Transforms/HandshakePlaceBuffersCustom.h"

#include <fstream>

using namespace llvm;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::rigidification;

namespace {
/// Rewrite pattern that will match on all muxes in the IR and replace each of
/// them with a merge taking the same inputs (except the `select` input which
/// merges do not have due to their undeterministic nature).
// struct HandshakeRigidification : public OpRewritePattern<Operation> {
//   using OpRewritePattern<Operation>::OpRewritePattern;

//   LogicalResult matchAndRewrite(Operation muxOp,
//                                 PatternRewriter &rewriter) const override {

//     // rewriter.replaceOp(muxOp, muxOp);

//     return success();
//   }
// };

/// Simple driver for the pass that replaces all muxes with merges.
struct HandshakeRigidificationPass
    : public impl::HandshakeRigidificationBase<HandshakeRigidificationPass> {

  void runDynamaticPass() override {
    // Get the MLIR context for the current operation being transformed
    MLIRContext *ctx = &getContext();
    // Get the operation being transformed (the top level module)
    ModuleOp mod = getOperation();

    // for (handshake::FuncOp funcOp : mod.getOps<handshake::FuncOp>()) {
    //   auto name = funcOp->getName();
    //   llvm::errs() << name.getIdentifier().str() << "\n";
    //   if (failed(rigidifyChannel(funcOp, ctx)))
    //     return signalPassFailure();
    // }
    mod->walk([&](Operation *op) {
      auto name = op->getName();
      llvm::errs() << name.getIdentifier().str() << "\n";
      for (auto ch : op->getResults()) {
        Type opType = ch.getType();
        // TODO: add support for control channels
        if (llvm::dyn_cast<handshake::ChannelType>(opType)) {
          llvm::errs() << "   " << opType << "\n";
          rigidifyChannel(&ch, ctx);
        }
      }
    });

    llvm::errs() << "\n\n\n\n";

    mod->walk([&](Operation *op) {
      auto name = op->getName();
      llvm::errs() << name.getIdentifier().str() << "\n";
      for (auto ch : op->getResults()) {
        Type opType = ch.getType();
        // TODO: add support for control channels
        if (llvm::dyn_cast<handshake::ChannelType>(opType)) {
          llvm::errs() << "   " << opType << "\n";
          // rigidifyChannel(&ch, ctx);
        }
      }
    });

    // MLIRContext *ctx = &getContext();

    // // Define the set of rewrite patterns we want to apply to the IR
    // RewritePatternSet patterns(ctx);

    // patterns.add<HandshakeRigidification<handshake::MuxOp>>(ctx);

    // // Run a greedy pattern rewriter on the entire IR under the top-level
    // // module
    // // operation
    // mlir::GreedyRewriteConfig config;
    // if (failed(
    //         applyPatternsAndFoldGreedily(mod, std::move(patterns),
    //         config))) {
    //   // If the greedy pattern rewriter fails, the pass must also fail
    //   return signalPassFailure();
    // }
  };
};
} // namespace

/// Implementation of our pass constructor, which just returns an instance of
/// the `HandshakeMuxToMergePass` struct.
std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::rigidification::createHandshakeRigidification() {
  return std::make_unique<HandshakeRigidificationPass>();
}