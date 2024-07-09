//===- ScfSimpleIfToSelect.cpp - Transform if's into select's ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the scf-simple-if-to-select transformation pass using a single
// rewrite pattern that matches scf::IfOp operations. To be transformed, matched
// if operations must satisfy a number of structural constraints that allow us
// to rewrite them as an an arith::SelectOp and an arithmetic operation.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/ScfSimpleIfToSelect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

/// If the block is made up of two operations (op + yield), determines whether
/// it has the right structure and whether the first operation is supported by
/// the transformation. Returns a pointer to the first operation if the block's
/// structure is valid, nullptr otherwise.
static Operation *getArithOpIfValid(Block &block, Value yielded) {
  Operation &arithOp = *block.getOperations().begin();

  // Single result of the first operation should be the yielded one, it should
  // also be integer-like
  if (arithOp.getNumResults() != 1 || (yielded != arithOp.getResult(0)))
    return nullptr;

  if (isa<arith::AddIOp, arith::SubIOp>(&arithOp))
    return &arithOp;
  return nullptr;
}

/// Returns the single value yielded by one of the branches (the function
/// assumes there is at least one value yielded).
static Value getYieldedVal(Block &block) {
  scf::YieldOp yieldOp =
      dyn_cast<scf::YieldOp>(*(--block.getOperations().end()));
  assert(yieldOp && "expected last operation to be a yield");
  return yieldOp.getOperand(0);
}

namespace {

/// Converts eligible scf::IfOp into an arith::SelectOp and an arithmetic
/// operation, removing the associated control flow from the IR in the process.
struct ConvertIfToSelect : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (Value replaceWith = tryToConvert(ifOp, rewriter)) {
      rewriter.replaceOp(ifOp, replaceWith);
      return success();
    }
    return failure();
  }

private:
  /// Hoists an arithmetic operation out of one of the if's branches, then
  /// inserts a select operation (conditioned like the if) using its result and
  /// another value. Returns the select's result.
  Value hoistSingleArithOp(scf::IfOp ifOp, Operation *arithOp,
                           Value otherSelectVal, bool otherValIsFalse,
                           PatternRewriter &rewriter) const;

  /// Creates a select operation (conditioned like the if) using two provided
  /// values, and feeds the select's result to a two-operand arithmetic
  /// operation referenced by its name. Returns the arithmetic operation's
  /// result.
  Value createSelectThenArithOp(scf::IfOp ifOp, Value trueVal, Value falseVal,
                                StringRef arithOpName, Value otherArithVal,
                                bool otherValIsRhs,
                                PatternRewriter &rewriter) const;

  /// If both if branches contain a single arithmetic operation, checks whether
  /// they are eligible for conversion. If they are, convert them to the
  /// appropriate equivalent operation and returns the value to replace the if's
  /// result with.
  Value combineArithOps(scf::IfOp ifOp, Operation *thenOp, Operation *elseOp,
                        PatternRewriter &rewriter) const;

  /// Attempts to convert the if to a control-less sequence of operations. If it
  /// is possible, creates the equivalent operations above the if and returns
  /// the value to replace the if's result with; otherwise returns nullptr.
  Value tryToConvert(scf::IfOp ifOp, PatternRewriter &rewriter) const;
};
} // namespace

Value ConvertIfToSelect::hoistSingleArithOp(scf::IfOp ifOp, Operation *arithOp,
                                            Value otherSelectVal,
                                            bool otherValIsFalse,
                                            PatternRewriter &rewriter) const {
  Value lhs = arithOp->getOperand(0);
  Value rhs = arithOp->getOperand(1);

  // Hoist the arithmetic operation above the converted if
  Operation *clonedArithOp = rewriter.create(
      ifOp.getLoc(),
      StringAttr::get(getContext(), arithOp->getName().getStringRef()),
      {lhs, rhs}, {arithOp->getResult(0).getType()});

  Value trueVal = clonedArithOp->getResult(0);
  Value falseVal = otherSelectVal;
  if (!otherValIsFalse)
    std::swap(trueVal, falseVal);

  return rewriter
      .create<arith::SelectOp>(ifOp->getLoc(), ifOp.getCondition(), trueVal,
                               falseVal)
      .getResult();
};

Value ConvertIfToSelect::createSelectThenArithOp(
    scf::IfOp ifOp, Value trueVal, Value falseVal, StringRef arithOpName,
    Value otherArithVal, bool otherValIsRhs, PatternRewriter &rewriter) const {
  rewriter.setInsertionPoint(ifOp);

  arith::SelectOp selectOp = rewriter.create<arith::SelectOp>(
      ifOp->getLoc(), ifOp.getCondition(), trueVal, falseVal);
  Value lhs = selectOp.getResult();
  Value rhs = otherArithVal;
  if (!otherValIsRhs)
    std::swap(lhs, rhs);

  return rewriter
      .create(ifOp.getLoc(), StringAttr::get(getContext(), arithOpName),
              {lhs, rhs}, {ifOp->getResult(0).getType()})
      ->getResult(0);
}

Value ConvertIfToSelect::combineArithOps(scf::IfOp ifOp, Operation *thenOp,
                                         Operation *elseOp,
                                         PatternRewriter &rewriter) const {
  // Operations must be of the same type
  if (thenOp->getName() != elseOp->getName())
    return nullptr;

  StringRef arithOpName = thenOp->getName().getStringRef();
  Value thenLhs = thenOp->getOperand(0), thenRhs = thenOp->getOperand(1),
        elseLhs = elseOp->getOperand(0), elseRhs = elseOp->getOperand(1);

  /// Shortcut to call createSelectThenArithOp.
  auto convert = [&](Value trueVal, Value falseVal, Value otherArithVal,
                     bool otherValIsRhs) -> Value {
    return createSelectThenArithOp(ifOp, trueVal, falseVal, arithOpName,
                                   otherArithVal, otherValIsRhs, rewriter);
  };

  if (thenOp->hasTrait<OpTrait::IsCommutative>()) {
    // If the operation is commutative, we care about finding the same operand
    // in both operations in any position. We can let the logic fall-through to
    // the logic for non-commutative operations and only check for swapped
    // common values here
    if (thenLhs == elseRhs)
      return convert(thenRhs, elseLhs, thenLhs, false);
    if (thenRhs == elseLhs)
      return convert(thenLhs, elseRhs, thenRhs, true);
  }

  // If the operations is NOT commutative, we need to find the same operand
  // in the same position
  if (thenLhs == elseLhs)
    return convert(thenRhs, elseRhs, thenLhs, false);
  if (thenRhs == elseRhs)
    return convert(thenLhs, elseLhs, thenRhs, true);

  return nullptr;
}

Value ConvertIfToSelect::tryToConvert(scf::IfOp ifOp,
                                      PatternRewriter &rewriter) const {
  // We only support if's with a single integer-like result
  if (ifOp->getNumResults() != 1)
    return nullptr;
  Type resType = ifOp.getResult(0).getType();
  if (!isa<IntegerType, IndexType>(resType))
    return nullptr;

  // Make sure the then block has at most 2 operations
  Block &thenBlock = ifOp.getThenRegion().front();
  auto &thenOps = thenBlock.getOperations();
  unsigned numThenOps = std::distance(thenOps.begin(), thenOps.end());
  if (numThenOps > 2)
    return nullptr;
  Value thenYielded = getYieldedVal(thenBlock);

  // The then branch's structure must be valid for our transformation
  Operation *thenArithOp = nullptr;
  if (numThenOps == 2) {
    if (!(thenArithOp = getArithOpIfValid(thenBlock, thenYielded)))
      return nullptr;
  }

  // The if must have an else block since it returns a value, so the following
  // is safe. Make sure the else block has at most 2 operations
  Block &elseBlock = ifOp.getElseRegion().front();
  auto &elseOps = elseBlock.getOperations();
  unsigned numElseOps = std::distance(elseOps.begin(), elseOps.end());
  if (numElseOps > 2)
    return nullptr;
  Value elseYielded = getYieldedVal(elseBlock);

  if (numElseOps == 2) {
    // The else block is an operation followed by a yield
    Operation *elseArithOp = getArithOpIfValid(elseBlock, elseYielded);
    if (!elseArithOp)
      return nullptr;

    if (thenArithOp)
      // The then block has an arithmetic operation
      return combineArithOps(ifOp, thenArithOp, elseArithOp, rewriter);
    // The then block is just a yield
    return hoistSingleArithOp(ifOp, elseArithOp, thenYielded, false, rewriter);
  }

  // The else block is just a yield
  if (thenArithOp)
    return hoistSingleArithOp(ifOp, thenArithOp, elseYielded, true, rewriter);

  // If the then block is just a yield too, then the entire if is equivalent to
  // a select
  return rewriter.create<arith::SelectOp>(ifOp.getLoc(), ifOp.getCondition(),
                                          thenYielded, elseYielded);
}

namespace {

/// Simple greedy pattern rewriter driver for the if-to-select transformation
/// pass.
struct ScfSimpleIfToSelectPass
    : public dynamatic::impl::ScfSimpleIfToSelectBase<ScfSimpleIfToSelectPass> {

  void runDynamaticPass() override {
    auto *ctx = &getContext();
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    RewritePatternSet patterns{ctx};
    patterns.add<ConvertIfToSelect>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config)))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createScfSimpleIfToSelect() {
  return std::make_unique<ScfSimpleIfToSelectPass>();
}
