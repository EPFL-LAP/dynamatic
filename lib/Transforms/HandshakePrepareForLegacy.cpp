//===- HandshakePrepareForLegacy.h - Prepare for legacy flow ----*- C++ -*-===//
//
// This file implements of a preprocessing step for handshake-level IR to make
// it compatible with the legacy Dynamatic flow (through DOT export). At the
// moment, this pass only turns unconditional branches (which legacy Dynamatic
// never generates) into conditional branches with a constant condition input
// and a sinked false output.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakePrepareForLegacy.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Conversion/StandardToHandshakeFPGA18.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "dynamatic/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

/// Creates a corresponding conditional branch for each unconditional branch.
/// The data input of corresponding branches is the same. A constant true
/// triggered by the given control signal is created to feed the conditional
/// branches' condition input.
static void createNewBranches(ArrayRef<handshake::BranchOp> branches,
                              Value ctrl, ConversionPatternRewriter &rewriter) {
  // Create constant source of true conditions
  rewriter.setInsertionPointAfterValue(ctrl);
  IntegerAttr cond = rewriter.getBoolAttr(true);
  Value trueCond = rewriter
                       .create<handshake::ConstantOp>(
                           ctrl.getLoc(), cond.getType(), cond, ctrl)
                       .getResult();

  // Insert a conditional branch for every unconditional branch and
  // replace the latter's result uses with the "true" result of the former
  for (auto br : branches) {
    rewriter.setInsertionPointAfter(br);
    auto cbranch = rewriter.create<handshake::ConditionalBranchOp>(
        br.getLoc(), trueCond, br->getOperand(0));
    br.getResult().replaceAllUsesWith(cbranch.getTrueResult());
  }
}

/// Converts all unconditional branches of a function into conditional branches
/// with a constant true condition input.
static void convertBranches(handshake::FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) {
  auto branches = funcOp.getOps<handshake::BranchOp>();
  if (branches.empty())
    return;

  auto handshakeBlocks = getHandshakeBlocks(funcOp);

  // Iterate over all identified handshake blocks to identify unconditional
  // branches and convert them
  SmallVector<handshake::BranchOp> branchesOutOfBlocks;
  for (auto &[blockID, ops] : handshakeBlocks.blocks) {

    // Identify all unconditional branches in the block, as well as a control
    // merge with dataless inputs if possible
    SmallVector<handshake::BranchOp> branches;
    handshake::ControlMergeOp cmerge = nullptr;
    for (auto op : ops)
      if (isa<handshake::BranchOp>(op))
        branches.push_back(dyn_cast<handshake::BranchOp>(op));
      else if (isa<handshake::ControlMergeOp>(op)) {
        auto blockCMerge = dyn_cast<handshake::ControlMergeOp>(op);
        if (!cmerge && blockCMerge.isControl())
          cmerge = blockCMerge;
      }

    if (branches.empty())
      continue;

    if (cmerge)
      // If we found a control merge with dataless inputs in the block, use its
      // result as control value for the new conditional branches
      createNewBranches(branches, cmerge.getResult(), rewriter);
    else if (blockID == 0)
      // If we are in the entry block, we can use the start input of the
      // function (last argument) as our control value
      createNewBranches(branches, funcOp.getArguments().back(), rewriter);
    else
      // If we did not find a control merge with dataless inputs in the
      // block, we'll simply create an endless source of true conditions
      // outside of all blocks to trigger the conditional branches
      llvm::copy(branches, std::back_inserter(branchesOutOfBlocks));
  }

  // Collect all unconditional branches that are out of blocks
  for (auto op : handshakeBlocks.outOfBlocks)
    if (isa<handshake::BranchOp>(op))
      branchesOutOfBlocks.push_back(dyn_cast<handshake::BranchOp>(op));

  if (!branchesOutOfBlocks.empty()) {
    // Create an endless source of control signals at the beginning of the
    // function for all unconditional branches that could not be matched with a
    // dataless control merge or the function's start control
    rewriter.setInsertionPointToStart(&funcOp.front());
    Value ctrl =
        rewriter
            .create<handshake::SourceOp>(funcOp.front().front().getLoc(),
                                         rewriter.getNoneType())
            .getResult();
    createNewBranches(branchesOutOfBlocks, ctrl, rewriter);
  }

  // Delete all unconditional branches
  for (auto op : funcOp.getOps<handshake::BranchOp>())
    rewriter.eraseOp(op);
}

namespace {

/// Converts simple branches to conditional branches with constant condition.
struct ConvertSimpleBranches : public OpConversionPattern<handshake::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(op, [&] { convertBranches(op, rewriter); });
    return success();
  }
};

/// Simple driver for prepare for legacy pass. Runs a partial conversion by
/// using a single operation conversion pattern on each handshake::FuncOp in the
/// module.
struct HandshakePrepareForLegacyPass
    : public HandshakePrepareForLegacyBase<HandshakePrepareForLegacyPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns{ctx};
    patterns.add<ConvertSimpleBranches>(ctx);
    ConversionTarget target(*ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakePrepareForLegacy() {
  return std::make_unique<HandshakePrepareForLegacyPass>();
}
