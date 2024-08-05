//===- HandshakeReshapeChannels.cpp - Reshape channels' signals -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass uses a simple greedy pattern rewriter with a single rewrite pattern
// matching on all operations implementing the
// `handshake::ReshapableChannelsInterface` interface to simplify the layout of
// channels around operations that do not care for the content of extra signals
// (and optionally the data signal). `handshake::ReshapeOp` operations are
// inserted around such operations to allow the IR to go back-and-forth between
// complex channel forms and simple channel forms.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeReshapeChannels.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include <iterator>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

/// Determines if the type information produced by the interface can be reshaped
/// into some simpler form.
static bool isReshapable(std::pair<ChannelType, bool> &reshapeInfo) {
  if (!reshapeInfo.first)
    return false;

  auto signalReshapable = [&](const ExtraSignal &extra) {
    if (extra.downstream) {
      // If the data signal is also ignored then any downstream extra signal can
      // be merged into it, otherwise only an extra downstream signal not named
      // like the merged one can be merged
      return reshapeInfo.second || extra.name != ReshapeOp::MERGED_DOWN_NAME;
    }
    return extra.name != ReshapeOp::MERGED_UP_NAME;
  };
  return llvm::any_of(reshapeInfo.first.getExtraSignals(), signalReshapable);
}

namespace {

struct ReshapeChannels
    : OpInterfaceRewritePattern<handshake::ReshapableChannelsInterface> {

  ReshapeChannels(NameAnalysis &namer, MLIRContext *ctx)
      : OpInterfaceRewritePattern<handshake::ReshapableChannelsInterface>(ctx),
        namer(namer) {}

  LogicalResult matchAndRewrite(handshake::ReshapableChannelsInterface op,
                                PatternRewriter &rewriter) const override {
    auto reshapeInfo = op.getReshapableChannelType();
    if (!isReshapable(reshapeInfo))
      return failure();
    ChannelType reshapeType = reshapeInfo.first;
    bool reshapeData = reshapeInfo.second;

    // Derive new operand types; those whose type is identical to the reference
    // type provided by the interface and have extra signals can be reshaped
    SmallVector<Value> newOperands;
    Type mergedType = nullptr;
    llvm::transform(
        llvm::enumerate(op->getOperands()), std::back_inserter(newOperands),
        [&](auto indexedOprd) -> Value {
          auto [idx, oprd] = indexedOprd;
          auto channelOprd = dyn_cast<TypedValue<ChannelType>>(oprd);
          if (!channelOprd || channelOprd.getType() != reshapeType)
            return oprd;

          auto reshapeOp = rewriter.create<handshake::ReshapeOp>(
              op->getLoc(), channelOprd,
              /*mergeDownstreamIntoData*/ reshapeData);
          mergedType = reshapeOp.getReshaped().getType();
          return reshapeOp.getReshaped();
        });
    if (!mergedType)
      return failure();

    // Derive new result types; those whose type is identical to the reference
    // type provided by the interface and have extra signals will need to be
    // reshaped later on
    SmallVector<Type> newResTypes;
    llvm::transform(llvm::enumerate(op->getResults()),
                    std::back_inserter(newResTypes),
                    [&](auto indexedRes) -> Type {
                      auto [idx, res] = indexedRes;
                      auto channelRes = dyn_cast<TypedValue<ChannelType>>(res);
                      if (!channelRes || channelRes.getType() != reshapeType)
                        return res.getType();
                      return mergedType;
                    });

    // Create a new version of the matched operation with modified operands and
    // result types
    StringAttr opName =
        StringAttr::get(rewriter.getContext(), op->getName().getStringRef());
    Operation *newOp =
        rewriter.create(op->getLoc(), opName, newOperands, newResTypes,
                        op->getAttrDictionary().getValue());

    // Derive values to replace the original operation with; those whose type
    // is different than before need to be reshaped back into the original type
    SmallVector<Value> newResults;
    llvm::transform(
        llvm::zip_equal(op->getResults(), newOp->getResults()),
        std::back_inserter(newResults), [&](auto oldAndNewRes) -> Value {
          auto [oldRes, newRes] = oldAndNewRes;
          if (oldRes.getType() == newRes.getType())
            return newRes;
          assert(oldRes.getType() == reshapeType &&
                 "non-reference type was changed");

          auto channelRes = dyn_cast<TypedValue<ChannelType>>(newRes);
          assert(channelRes && "channel result became non-channel result");
          auto reshapeOp = rewriter.create<handshake::ReshapeOp>(
              op->getLoc(), channelRes,
              /*splitDownstreamFromData*/ reshapeData, oldRes.getType());
          return reshapeOp.getReshaped();
        });

    namer.replaceOp(op, newOp);
    rewriter.replaceOp(op, newResults);
    return success();
  }

private:
  NameAnalysis &namer;
};

/// Simple driver for the channel-reshaping pass. It uses a greedy pattern
/// rewriter to reshape channels around channels implementing the
/// `ReshapableChannelsInterface` or `ReshapableChannelsInterface`
struct HandshakeReshapeChannelsPass
    : public dynamatic::impl::HandshakeReshapeChannelsBase<
          HandshakeReshapeChannelsPass> {

  void runDynamaticPass() override {
    mlir::ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns(ctx);
    patterns.add<ReshapeChannels>(getAnalysis<NameAnalysis>(), ctx);
    if (failed(applyPatternsAndFoldGreedily(mod, std::move(patterns), config)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeReshapeChannels() {
  return std::make_unique<HandshakeReshapeChannelsPass>();
}
