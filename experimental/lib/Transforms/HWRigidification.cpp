/// Include the header we just created.
#include "experimental/Transforms/HWRigidification.h"
// #include "experimental/Support/RigidificationSupport.h"

/// Include some other useful headers.
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
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

LogicalResult rigidifyHWChannel(OpResult &channel, MLIRContext *ctx) {
  if (!(llvm::dyn_cast<handshake::ChannelType>(channel.getType())))
    return LogicalResult::failure();

  OpBuilder builder(ctx);
  builder.setInsertionPointAfter(channel.getDefiningOp());
  auto loc = channel.getLoc();

  //{argNames = ["ins", "clk", "rst"], instanceName = "buffer7", moduleName =
  //@handshake_buffer_3, parameters = [], resultNames = ["outs"]}

  mlir::Attribute strAttr1 = builder.getStringAttr("AAAAAAAAAAA");
  mlir::NamedAttribute namedAttr1 = builder.getNamedAttr("argNames", strAttr1);

  mlir::Attribute strAttr2 = builder.getStringAttr("rigidifier0");
  mlir::NamedAttribute namedAttr2 =
      builder.getNamedAttr("instanceName", strAttr2);

  mlir::Attribute strAttr3 = builder.getStringAttr("rigidifier");
  mlir::NamedAttribute namedAttr3 =
      builder.getNamedAttr("moduleName", strAttr3);

  mlir::Attribute strAttr4 = builder.getStringAttr("");
  mlir::NamedAttribute namedAttr4 =
      builder.getNamedAttr("parameters", strAttr4);

  mlir::Attribute strAttr5 = builder.getStringAttr("BBBBBBBBB");
  mlir::NamedAttribute namedAttr5 =
      builder.getNamedAttr("resultNames", strAttr5);

  llvm::SmallVector<mlir::NamedAttribute, 0> namedAttrVec = {
      namedAttr1, namedAttr2, namedAttr3, namedAttr4, namedAttr5};

  hw::InstanceOp rigOp = builder.create<hw::InstanceOp>(
      loc, channel.getType(), mlir::ValueRange(channel), namedAttrVec);

  for (auto res : rigOp.getResults()) {
    auto *use = &(*(channel.getUses().begin()));
    if (use->getOwner() != rigOp) {
      use->set(res);
    }
  }
  return LogicalResult::success();
}

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
struct HWRigidificationPass
    : public impl::HWRigidificationBase<HWRigidificationPass> {

  void runDynamaticPass() override {
    // Get the MLIR context for the current operation being transformed
    MLIRContext *ctx = &getContext();
    // Get the operation being transformed (the top level module)
    ModuleOp mod = getOperation();

    for (hw::HWModuleOp modOp : mod.getOps<hw::HWModuleOp>()) {
      if (failed(performRigidification(modOp, ctx)))
        return signalPassFailure();
    }
    // mod->walk([&](Operation *op) {
    //   auto name = op->getName();
    //   llvm::errs() << name.getIdentifier().str() << "\n";
    //   for (auto ch : op->getResults()) {
    //     bool is_mem = false;
    //     for (auto &use : llvm::make_early_inc_range(ch.getUses()))
    //       if (isa<handshake::LSQOp, handshake::MemoryControllerOp>(
    //               use.getOwner()))
    //         is_mem = true;
    //     Type resType = ch.getType();
    //     if (llvm::dyn_cast<handshake::ChannelType>(resType) && !is_mem) {
    //       rigidifyChannel(&ch, ctx);
    //     }
    //   }
    // });

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
  static LogicalResult performRigidification(hw::HWModuleOp modOp,
                                             MLIRContext *ctx) {
    for (Operation &op : modOp.getBodyBlock()->getOperations()) {
      llvm::errs() << op << "\n\n\n\n\n\n\n\n";
      for (auto res : op.getResults()) {
        Type opType = res.getType();
        if (llvm::dyn_cast<handshake::ChannelType>(opType) &&
            op.getAttrOfType<mlir::ArrayAttr>("resultNames"))
          rigidifyHWChannel(res, ctx);
      }
    }
    return success();
  }
};

} // namespace

/// Implementation of our pass constructor, which just returns an instance of
/// the `HandshakeMuxToMergePass` struct.
std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::rigidification::createHWRigidificationPass() {
  return std::make_unique<HWRigidificationPass>();
}
