//===- HandshakeInferBasicBlocks.cpp - Infer ops basic blocks ---*- C++ -*-===//
//
// This file contains the implementation of the infer basic blocks pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeInferBasicBlocks.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Conversion/StandardToHandshakeFPGA18.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "dynamatic/Transforms/Passes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

static bool isLegalForInference(Operation *op) {
  return !isa<MemoryControllerOp>(op);
}

/// Iterates over all operations legal for inference that do not have a "bb"
/// attribute and tries to infer it.
static void inferBasicBlocks(handshake::FuncOp funcOp,
                             PatternRewriter &rewriter) {
  for (auto &op : funcOp.getOps())
    if (isLegalForInference(&op))
      if (auto bb = op.getAttrOfType<mlir::IntegerAttr>(BB_ATTR); !bb)
        if (auto infBB = inferOpBasicBlock(&op); infBB.has_value())
          op.setAttr(BB_ATTR, rewriter.getUI32IntegerAttr(infBB.value()));
}

std::optional<unsigned> dynamatic::inferOpBasicBlock(Operation *op) {
  assert(op->getParentOp() && isa<handshake::FuncOp>(op->getParentOp()) &&
         "operation must have a handshake::FuncOp as immediate parent");

  std::optional<unsigned> infBB;

  // For each operand of the operation, try to backtrack through parent
  // operations till we reach one with a known basic block or a function
  // argument
  for (auto operand : op->getOperands()) {
    Value val = operand;
    std::optional<unsigned> infOperandBB;
    do {
      auto defOp = val.getDefiningOp();

      if (!defOp)
        // Value originates from function argument i.e., from the first "block"
        // in the function
        infOperandBB = 0;
      else if (isa<handshake::BranchOp, handshake::ConditionalBranchOp>(defOp))
        // Don't backtrack through branches, which usually mark the end of a
        // block
        break;
      else if (auto opBB = defOp->getAttrOfType<mlir::IntegerAttr>(BB_ATTR);
               opBB)
        // We have backtracked to an operation with a known basic block, so
        // we can infer the original operation is from the same block
        infOperandBB = opBB.getValue().getZExtValue();
      else if (defOp->getNumOperands() == 1 &&
               !isa<handshake::MergeLikeOpInterface>(defOp))
        // Continue backtracking to the dataflow predecessor
        val = defOp->getOperand(0);
      else
        break;

    } while (!infOperandBB.has_value());

    // Determine whether the inferred basic block for the operand is compatible
    // with a potential previously inferred block
    if (!infOperandBB.has_value())
      return {};

    if (infBB.has_value()) {
      if (infBB.value() != infOperandBB.value())
        return {};
    } else
      infBB = infOperandBB.value();
  }

  return infBB;
}

namespace {

struct FuncOpInferBasicBlocks : public OpConversionPattern<handshake::FuncOp> {

  FuncOpInferBasicBlocks(MLIRContext *ctx) : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(handshake::FuncOp funcOp, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(funcOp,
                               [&] { inferBasicBlocks(funcOp, rewriter); });
    return success();
  }
};

struct HandshakeInferBasicBlocksPass
    : public HandshakeInferBasicBlocksBase<HandshakeInferBasicBlocksPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns{ctx};
    patterns.add<FuncOpInferBasicBlocks>(ctx);
    ConversionTarget target(*ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakeInferBasicBlocksPass() {
  return std::make_unique<HandshakeInferBasicBlocksPass>();
}
