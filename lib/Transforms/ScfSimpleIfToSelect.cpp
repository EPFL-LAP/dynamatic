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
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

/// Holds information necessary to construct the flat operations equivalent to
/// an if that has the right structure for our transformation
struct MatchResult {
  /// The name of the arithmetic operation present in at least one branch.
  std::string opName;
  /// Is the value common between both branches the LHS?
  bool isCommonValLhs = false;
  /// The value used by both branches (i.e, the one that won't be part of the
  /// arith::SelectOp).
  Value commonVal = nullptr;
  /// The value from the "then" branch that will end up on the "true side" of
  /// the inserted arith::SelectOp.
  Value trueVal = nullptr;
  /// The value from the "else" branch that will end up on the "false side" of
  /// the inserted arith::SelectOp.
  Value falseVal = nullptr;
};

/// Converts eligible scf::IfOp into an arith::SelectOp and an arithmetic
/// operation, removing the associated control flow from the IR in the process.
struct ConvertIfToSelect : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    // Check whether the if has the right structure for our conversion
    MatchResult matchRes;
    if (!isLegalForConversion(ifOp, rewriter, matchRes))
      return failure();

    // Create the select operation
    rewriter.setInsertionPoint(ifOp);
    auto selectOp =
        rewriter.create<arith::SelectOp>(ifOp->getLoc(), ifOp.getCondition(),
                                         matchRes.trueVal, matchRes.falseVal);

    // Create the arithmetic operation from its name
    Value selRes = selectOp.getResult();
    Value lhs, rhs;
    if (matchRes.isCommonValLhs) {
      lhs = matchRes.commonVal;
      rhs = selRes;
    } else {
      lhs = selRes;
      rhs = matchRes.commonVal;
    }
    Operation *arithOp = rewriter.create(
        ifOp.getLoc(), StringAttr::get(getContext(), matchRes.opName),
        {lhs, rhs}, {ifOp.getResult(0).getType()});

    rewriter.replaceOp(ifOp, arithOp->getResult(0));
    return success();
  }

private:
  /// Determines whether the if has the right structure for our conversion. If
  /// it has, the function returns true and the result struct is filled with all
  /// the information necessary to construct the equivalent operations that will
  /// replace the if. Note that, in the latter case, the function may create
  /// constant operations in the IR next to the if to create "neutral elements"
  /// to select with in case one of the if's branches only yields a value.
  bool isLegalForConversion(scf::IfOp ifOp, PatternRewriter &rewriter,
                            MatchResult &res) const;
};
} // namespace

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

/// If both branches have an arithmetic operation, checks whether they are
/// compatible. They must be of the same type and share an operand (which may or
/// may not need to be in the same position, depending on the operation's
/// commutativity trait). If branches are compatible, fills up the result struct
/// with all relevant values (the common one, the true one, and the false one).
static LogicalResult handleOpInEachBranch(Operation *thenOp, Operation *elseOp,
                                          MatchResult &res) {
  // Operations must be of the same type
  if (thenOp->getName() != elseOp->getName())
    return failure();

  Value thenLhs = thenOp->getOperand(0), thenRhs = thenOp->getOperand(1),
        elseLhs = elseOp->getOperand(0), elseRhs = elseOp->getOperand(1);

  if (thenOp->hasTrait<OpTrait::IsCommutative>()) {
    // If the operation is commutative, we care about finding the same operand
    // in both operations in any position. We can let the logic fall-through to
    // the logic for non-commutative operations and only check for swapped
    // common values here
    if (thenLhs == elseRhs) {
      res.trueVal = thenRhs;
      res.falseVal = elseLhs;
      res.commonVal = thenLhs;
      return success();
    }
    if (thenRhs == elseLhs) {
      res.trueVal = thenLhs;
      res.falseVal = elseRhs;
      res.commonVal = thenRhs;
      return success();
    }
  }

  // If the operations is NOT commutative, we need to find the same operand
  // in the same position
  if (thenLhs == elseLhs) {
    res.isCommonValLhs = true;
    res.trueVal = thenRhs;
    res.falseVal = elseRhs;
    res.commonVal = thenLhs;
    return success();
  }
  if (thenRhs == elseRhs) {
    res.isCommonValLhs = false;
    res.trueVal = thenLhs;
    res.falseVal = elseLhs;
    res.commonVal = thenRhs;
    return success();
  }
  return failure();
}

bool ConvertIfToSelect::isLegalForConversion(scf::IfOp ifOp,
                                             PatternRewriter &rewriter,
                                             MatchResult &res) const {
  // We only support if's with a single integer-like result
  if (ifOp->getNumResults() != 1)
    return false;
  Type resType = ifOp.getResult(0).getType();
  if (!isa<IntegerType, IndexType>(resType))
    return false;

  // Make sure the then block has at most 2 operations
  Block &thenBlock = ifOp.getThenRegion().front();
  auto &thenOps = thenBlock.getOperations();
  unsigned numThenOps = std::distance(thenOps.begin(), thenOps.end());
  if (numThenOps > 2)
    return false;
  Value thenYielded = getYieldedVal(thenBlock);

  // The then branch's structure must be valid for our transformation
  Operation *thenArithOp = nullptr;
  if (numThenOps == 2) {
    if (!(thenArithOp = getArithOpIfValid(thenBlock, thenYielded)))
      return false;
    res.opName = thenArithOp->getName().getStringRef().str();
  }

  // The if must have an else block since it returns a value, so the following
  // is safe. Make sure the else block has at most 2 operations
  Block &elseBlock = ifOp.getElseRegion().front();
  auto &elseOps = elseBlock.getOperations();
  unsigned numElseOps = std::distance(elseOps.begin(), elseOps.end());
  if (numElseOps > 2)
    return false;
  Value elseYielded = getYieldedVal(elseBlock);

  // Creates an arithmetic constant with the given constant attribute and
  // returns its result.
  auto createConstant = [&](IntegerAttr cstAttr) -> Value {
    rewriter.setInsertionPoint(ifOp);
    return rewriter.create<arith::ConstantOp>(ifOp.getLoc(), cstAttr)
        .getResult();
  };

  // Creates an arithmetic constant that is the neutral value for the arithmetic
  // operation that was identified.
  auto getNeutralValue = [&]() -> Value {
    // All supported operations have a 0 neutral value for now
    return createConstant(IntegerAttr::get(resType, 0));
  };

  // Determines whether both branches are compatible for the transformation,
  // assuming that one is only a yield and the other one a supported operation
  // followed by a yield. If branches are compatible, fills up the result struct
  // with all relevant values (the common one, the true one, and the false one)
  // and succeeds, fails otherwise.
  auto handleOneEmptyBranch = [&](Operation *arithOp, Value otherYielded,
                                  bool thenIsYield) -> LogicalResult {
    Value lhs = arithOp->getOperand(0);
    Value rhs = arithOp->getOperand(1);

    // The yielded value must appear in the operation
    if (otherYielded != lhs && otherYielded != rhs)
      return failure();

    res.isCommonValLhs = otherYielded == lhs;
    if (thenIsYield) {
      res.trueVal = getNeutralValue();
      res.falseVal = res.isCommonValLhs ? rhs : lhs;
    } else {
      res.trueVal = res.isCommonValLhs ? rhs : lhs;
      res.falseVal = getNeutralValue();
    }
    res.commonVal = otherYielded;
    return success();
  };

  if (numElseOps == 2) {
    // The else block is an operation followed by a yield
    Operation *elseArithOp = getArithOpIfValid(elseBlock, elseYielded);
    if (!elseArithOp)
      return false;

    if (thenArithOp)
      // The then block has an arithmetic operation
      return succeeded(handleOpInEachBranch(thenArithOp, elseArithOp, res));
    // The then block is just a yield
    res.opName = elseArithOp->getName().getStringRef().str();
    return succeeded(handleOneEmptyBranch(elseArithOp, thenYielded, true));
  }

  // The else block is just a yield
  if (thenArithOp)
    // The then block has an arithmetic operation
    return succeeded(handleOneEmptyBranch(thenArithOp, elseYielded, false));

  // If the then block is just a yield too, then the if is equivalent to a
  // select at most. Fake an addition with 0 to flatten the if. The addition
  // will be canonicalized away automatically
  res.opName = arith::AddIOp::getOperationName().str();
  res.trueVal = thenYielded;
  res.falseVal = elseYielded;
  res.commonVal = createConstant(IntegerAttr::get(resType, 0));
  return true;
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
