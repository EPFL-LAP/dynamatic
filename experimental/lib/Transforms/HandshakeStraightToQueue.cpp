//===- HandshakeStraightToQueue.cpp - Implement S2Q algorithm -*- C++ -*---===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass which allows to implement straight to the
// queue, a different way of allocating basic blocks in the LSQ, based on an
// ASAP approach rather than relying on the network of cmerges.
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/HandshakeStraightToQueue.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/HandshakeInferBasicBlocks.h"
#include "experimental/Support/CFGAnnotation.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::ftd;

namespace {

/// Tries to infer the basic block of untagged operations in a function.
struct FuncOpStraightToQueue : public OpConversionPattern<handshake::FuncOp> {

  FuncOpStraightToQueue(MLIRContext *ctx) : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(handshake::FuncOp funcOp, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::dbgs() << "> Running S2Q\n";
    rewriter.updateRootInPlace(funcOp, [&] {
      if (failed(cfg::restoreCfStructure(funcOp, rewriter)))
        return;

      funcOp.print(llvm::dbgs());

      if (failed(cfg::flattenFunction(funcOp, rewriter)))
        return;
    });
    return success();
  }
};

struct HandshakeStraightToQueuePass
    : public dynamatic::experimental::ftd::impl::HandshakeStraightToQueueBase<
          HandshakeStraightToQueuePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns{ctx};
    patterns.add<FuncOpStraightToQueue>(ctx);
    ConversionTarget target(*ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::ftd::createStraightToQueue() {
  return std::make_unique<HandshakeStraightToQueuePass>();
}
