//===- HandshakeInferBasicBlocks.cpp - Infer ops basic blocks ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The basic block inference pass is implemented as a single operation
// conversion pattern that iterates over all operations in a function repeatedly
// until no more inferences can be performed, at which point it succeeds.
//
// A local inference heuristic is applied on each operation eligible for
// inference. The locality of the heuristic may require the pass to run the
// inference logic on eligible operations multiple times in order to let
// inference results propagate incrementally to their immediate graph neighbors.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeInferBasicBlocks.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "dynamatic/Transforms/Passes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace mlir;
using namespace dynamatic;

/// Determines if the pass should attempt to infer the basic block of the
/// operation if it is missing.
static bool isLegalForInference(Operation *op) {
  return !isa<handshake::MemoryOpInterface, handshake::SinkOp>(op);
}

/// Iterates over all operations legal for inference that do not have a "bb"
/// attribute and tries to infer it.
static bool inferBasicBlocks(Operation *op, PatternRewriter &rewriter) {
  // Check whether we even need to run inference for the operation
  if (!isLegalForInference(op))
    return false;
  if (std::optional<unsigned> bb = getLogicBB(op); bb.has_value())
    return false;

  // Run the inference logic
  unsigned infBB;
  if (succeeded(inferLogicBB(op, infBB))) {
    op->setAttr(BB_ATTR, rewriter.getUI32IntegerAttr(infBB));
    return true;
  }
  return false;
}

LogicalResult dynamatic::inferLogicBB(Operation *op, unsigned &logicBB) {
  std::optional<unsigned> infBB;

  auto mergeInferredBB = [&](std::optional<unsigned> otherBB) -> LogicalResult {
    if (!otherBB.has_value() || (infBB.has_value() && *infBB != *otherBB)) {
      infBB = std::nullopt;
      return failure();
    }
    infBB = *otherBB;
    return success();
  };

  // First, try to infer the basic block of the current operation by looking at
  // its successors (i.e., users of its results). If they all belong to the same
  // basic block, then we can safely say that the current operation also belongs
  // to it
  for (OpResult res : op->getResults()) {
    bool conflict = false;
    for (Operation *user : res.getUsers())
      if (failed(mergeInferredBB(getLogicBB(user)))) {
        conflict = true;
        break;
      }
    if (conflict)
      break;
  }

  // If the successor analysis successfully inferred a basic block, return this
  // one; otherwise, run the predecessor analysis.
  if (infBB.has_value()) {
    logicBB = *infBB;
    return success();
  }

  // Second, try to infer the basic block of the current operation by looking at
  // its predecessors (i.e., producers of its operarands). If they all belong to
  // the same basic block, then we can safely say that the current operation
  // also belongs to it
  for (Value opr : op->getOperands()) {
    Operation *defOp = opr.getDefiningOp();
    std::optional<unsigned> oprBB = defOp ? getLogicBB(defOp) : ENTRY_BB;
    if (failed(mergeInferredBB(oprBB))) {
      return failure();
    }
  }

  if (infBB.has_value()) {
    logicBB = *infBB;
    return success();
  }
  return failure();
}

namespace {

/// Tries to infer the basic block of untagged operations in a function.
struct FuncOpInferBasicBlocks : public OpConversionPattern<handshake::FuncOp> {

  FuncOpInferBasicBlocks(MLIRContext *ctx) : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(handshake::FuncOp funcOp, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(funcOp, [&] {
      bool progress = false;
      do {
        progress = false;
        for (Operation &op : funcOp.getOps())
          progress |= inferBasicBlocks(&op, rewriter);
      } while (progress);
    });
    return success();
  }
};

/// Simple driver for basic block inference pass. Runs a partial conversion by
/// using a single operation conversion pattern on each handshake::FuncOp in the
/// module.
struct HandshakeInferBasicBlocksPass
    : public HandshakeInferBasicBlocksBase<HandshakeInferBasicBlocksPass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns{ctx};
    patterns.add<FuncOpInferBasicBlocks>(ctx);
    ConversionTarget target(*ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeInferBasicBlocksPass() {
  return std::make_unique<HandshakeInferBasicBlocksPass>();
}
