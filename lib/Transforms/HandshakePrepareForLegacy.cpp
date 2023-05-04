//===- HandshakePrepareForLegacy.h - Prepare for legacy flow ----*- C++ -*-===//
//
// This file implements a preprocessing step for handshake-level IR to make it
// compatible with the legacy Dynamatic flow (through DOT export).
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

/// Transfers the "bb" attribute from a source operation to a destination
/// operation if it exists. TODO: this should probably be a publicly visible
/// function, make it so when we have better support for "logical basic blocks"
static void inheritBB(Operation *src, Operation *dst) {
  if (auto bb = src->getAttr(BB_ATTR))
    dst->setAttr(BB_ATTR, bb);
}

/// Creates a corresponding conditional branch for each unconditional branch.
/// The data input of corresponding branches is the same. A constant true
/// triggered by the given control signal is created to feed the conditional
/// branches' condition input.
static void createNewBranches(ArrayRef<handshake::BranchOp> branches,
                              Value ctrl, ConversionPatternRewriter &rewriter) {
  // Create constant source of true conditions
  rewriter.setInsertionPointAfterValue(ctrl);
  IntegerAttr cond = rewriter.getBoolAttr(true);
  auto constOp = rewriter.create<handshake::ConstantOp>(
      ctrl.getLoc(), cond.getType(), cond, ctrl);

  // Try to set the bb attribute on the merge
  if (auto defOp = ctrl.getDefiningOp())
    inheritBB(defOp, constOp);
  else
    constOp->setAttr(BB_ATTR, rewriter.getUI32IntegerAttr(0));

  // Insert a conditional branch for every unconditional branch and replace the
  // latter's result uses with the "true" result of the former
  auto trueCond = constOp.getResult();
  for (auto br : branches) {
    rewriter.setInsertionPointAfter(br);
    auto cbranch = rewriter.create<handshake::ConditionalBranchOp>(
        br.getLoc(), trueCond, br->getOperand(0));
    inheritBB(br, cbranch);
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
  for (auto &[blockID, blockOps] : handshakeBlocks.blocks) {

    // Identify all unconditional branches in the block, as well as a control
    // merge with dataless inputs if possible
    SmallVector<handshake::BranchOp> blockBranchOps;
    handshake::ControlMergeOp cmerge = nullptr;
    for (auto op : blockOps)
      if (isa<handshake::BranchOp>(op))
        blockBranchOps.push_back(dyn_cast<handshake::BranchOp>(op));
      else if (isa<handshake::ControlMergeOp>(op)) {
        auto blockCMerge = dyn_cast<handshake::ControlMergeOp>(op);
        if (!cmerge && blockCMerge.isControl())
          cmerge = blockCMerge;
      }

    if (blockBranchOps.empty())
      continue;

    if (cmerge)
      // If we found a control merge with dataless inputs in the block, use its
      // result as control value for the new conditional branches
      createNewBranches(blockBranchOps, cmerge.getResult(), rewriter);
    else if (blockID == 0)
      // If we are in the entry block, we can use the start input of the
      // function (last argument) as our control value
      createNewBranches(blockBranchOps, funcOp.getArguments().back(), rewriter);
    else
      // If we did not find a control merge with dataless inputs in the
      // block, we'll simply create an endless source of true conditions
      // outside of all blocks to trigger the conditional branches
      llvm::copy(blockBranchOps, std::back_inserter(branchesOutOfBlocks));
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
  for (auto brOp : branches)
    rewriter.eraseOp(brOp);
}

namespace {

/// Custom conversion target for the pass that checks whether the IR is valid
/// for use in legacy Dynamatic.
class LegacyDynamatic : public ConversionTarget {
public:
  explicit LegacyDynamatic(MLIRContext &context) : ConversionTarget(context) {
    addLegalDialect<handshake::HandshakeDialect>();
    // Functions become legal after the ConvertSimpleBranches pattern has
    // matched them
    addDynamicallyLegalOp<handshake::FuncOp>(
        [&](const auto &op) { return convertedFuncOps.contains(op); });
    // CMerges are only valid when they have at least two operands in legacy
    // Dynamatic
    addDynamicallyLegalOp<handshake::ControlMergeOp>(
        [&](const auto &op) { return op->getNumOperands() >= 2; });
  }
  /// The set of function that have already been matched by the
  /// ConvertSimpleBranches pattern.
  SmallPtrSet<Operation *, 4> convertedFuncOps;
};

/// Converts simple branches to conditional branches with a constant condition
/// input and a sinked false output.
struct ConvertSimpleBranches : public OpConversionPattern<handshake::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertSimpleBranches(LegacyDynamatic &target, MLIRContext *ctx)
      : OpConversionPattern<handshake::FuncOp>(ctx), target(target) {}

  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(op, [&] { convertBranches(op, rewriter); });

    // Mark the function legal
    target.convertedFuncOps.insert(op);
    return success();
  }

private:
  /// Reference to the conversion target so that we can mark the operation legal
  /// once we are done converting branches.
  LegacyDynamatic &target;
};

/// Simplifies control merges into simple merges when they only have one operand
/// or when their index result is unused.
struct SimplifyControlMerges
    : public OpConversionPattern<handshake::ControlMergeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(handshake::ControlMergeOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto numOperands = op->getNumOperands();
    auto indexUnused = op.getIndex().use_empty();

    if (numOperands == 1 || indexUnused) {
      // A cmerge with one operands or an unused index result can be downgraded
      // to a simple merge with the same operands
      rewriter.setInsertionPoint(op);
      auto merge = rewriter.create<handshake::MergeOp>(op.getLoc(),
                                                       op.getDataOperands());
      inheritBB(op, merge);
      rewriter.updateRootInPlace(op, [&] {
        op.getResult().replaceAllUsesWith(merge.getResult());

        if (indexUnused)
          return;
        // When the index has users, we must replace it with a constant 0
        // triggered by the newly created merge

        // Create the attribute for the constant, whose type is derived from the
        // cmerge's index result (index or integer attribute)
        auto indexResType = op.getIndex().getType();
        TypedAttr constantAttr;
        if (isa<IndexType>(indexResType))
          constantAttr = rewriter.getIndexAttr(0);
        else
          constantAttr = rewriter.getIntegerAttr(indexResType, 0);

        // Create the constant and replace the cmerge's index result
        auto constantOp = rewriter.create<handshake::ConstantOp>(
            op.getLoc(), constantAttr.getType(), constantAttr,
            merge.getResult());
        inheritBB(op, constantOp);
        op.getIndex().replaceAllUsesWith(constantOp.getResult());
      });
      rewriter.eraseOp(op);
    }

    return success();
  }
};

/// Simple driver for prepare for legacy pass.
struct HandshakePrepareForLegacyPass
    : public HandshakePrepareForLegacyBase<HandshakePrepareForLegacyPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns{ctx};
    LegacyDynamatic target(*ctx);
    patterns.add<ConvertSimpleBranches>(target, ctx);
    patterns.add<SimplifyControlMerges>(ctx);

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
