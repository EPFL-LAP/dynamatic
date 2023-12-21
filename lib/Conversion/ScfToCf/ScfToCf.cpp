//===- ScfToCf.cpp - Lower scf ops to unstructured control flow -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --lower-scf-to-cf conversion pass. It is different from the
// upstream --convert-scf-to-cf pass in only one respect: it basically
// "overwrites" the for-lowering pattern from the upstream pass with a custom
// one which inserts unsigned integer comparisons in the IR whenever possible;
// this is in opposition to the for-lowering of the upstream pass, which always
// inserts signed comparisons.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/ScfToCf.h"
#include "dynamatic/Analysis/NumericAnalysis.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace dynamatic;

namespace {

/// Lower structured for loops into their unstructured form. Taken from MLIR's
/// --convert-scf-to-cf pass, but creates an unsigned comparison (ult) instead
/// of a signed one (lt) if the loop iterator can be proven to be always
/// positive.
struct ForLowering : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    Location loc = forOp.getLoc();

    // Compute loop bounds before branching to the condition.
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    if (!lowerBound || !upperBound)
      return failure();

    // Determine comparison predicate to use when lowering the loop. We can
    // insert an unsigned comparison only if the lower bound can be guaranteed
    // to be non-negative
    NumericAnalysis analysis;
    arith::CmpIPredicate pred = analysis.getRange(lowerBound).isPositive()
                                    ? arith::CmpIPredicate::ult
                                    : arith::CmpIPredicate::slt;

    // Start by splitting the block containing the 'scf.for' into two parts.
    // The part before will get the init code, the part after will be the end
    // point.
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPosition = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(initBlock, initPosition);

    // Use the first block of the loop body as the condition block since it is
    // the block that has the induction variable and loop-carried values as
    // arguments. Split out all operations from the first block into a new
    // block. Move all body blocks from the loop body region to the region
    // containing the loop.
    auto *conditionBlock = &forOp.getRegion().front();
    auto *firstBodyBlock =
        rewriter.splitBlock(conditionBlock, conditionBlock->begin());
    auto *lastBodyBlock = &forOp.getRegion().back();
    rewriter.inlineRegionBefore(forOp.getRegion(), endBlock);
    auto iv = conditionBlock->getArgument(0);

    // Append the induction variable stepping logic to the last body block and
    // branch back to the condition block. Loop-carried values are taken from
    // operands of the loop terminator.
    Operation *terminator = lastBodyBlock->getTerminator();
    rewriter.setInsertionPointToEnd(lastBodyBlock);
    auto step = forOp.getStep();
    auto stepped = rewriter.create<arith::AddIOp>(loc, iv, step).getResult();
    if (!stepped)
      return failure();

    SmallVector<Value, 8> loopCarried;
    loopCarried.push_back(stepped);
    loopCarried.append(terminator->operand_begin(), terminator->operand_end());
    rewriter.create<cf::BranchOp>(loc, conditionBlock, loopCarried);
    rewriter.eraseOp(terminator);

    // The initial values of loop-carried values is obtained from the operands
    // of the loop operation.
    SmallVector<Value, 8> destOperands;
    destOperands.push_back(lowerBound);
    auto iterOperands = forOp.getIterOperands();
    destOperands.append(iterOperands.begin(), iterOperands.end());
    rewriter.setInsertionPointToEnd(initBlock);
    rewriter.create<cf::BranchOp>(loc, conditionBlock, destOperands);

    // With the body block done, we can fill in the condition block.
    rewriter.setInsertionPointToEnd(conditionBlock);
    auto comparison = rewriter.create<arith::CmpIOp>(loc, pred, iv, upperBound);

    rewriter.create<cf::CondBranchOp>(loc, comparison, firstBodyBlock,
                                      ArrayRef<Value>(), endBlock,
                                      ArrayRef<Value>());
    // The result of the loop operation is the values of the condition block
    // arguments except the induction variable on the last iteration.
    rewriter.replaceOp(forOp, conditionBlock->getArguments().drop_front());
    return success();
  }
};

struct ScfToCfPass : public dynamatic::impl::ScfToCfBase<ScfToCfPass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp modOp = getOperation();

    // Set up rewrite patterns
    RewritePatternSet patterns{ctx};
    // Our for lowering is given a higher benefit than the one defined in MLIR,
    // so it will be matched first, essentially overriding the default one
    patterns.add<ForLowering>(ctx, /*benefit=*/2);
    populateSCFToControlFlowConversionPatterns(patterns);

    // Set up conversion target
    ConversionTarget target(*ctx);
    target.addIllegalOp<scf::ForOp, scf::IfOp, scf::ParallelOp, scf::WhileOp,
                        scf::ExecuteRegionOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(modOp, target, std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace

namespace dynamatic {
std::unique_ptr<dynamatic::DynamaticPass> createLowerScfToCf() {
  return std::make_unique<ScfToCfPass>();
}
} // namespace dynamatic
