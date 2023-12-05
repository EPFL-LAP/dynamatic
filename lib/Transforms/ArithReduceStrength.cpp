//===- ArithReduceStrength.cpp - Reduce stregnth of arith ops ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --arith-reduce-strength pass, which greedily applies rewrite
// patterns to arithmetic operations to reduce their strength, improving
// performance and/or area.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/ArithReduceStrength.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NumericAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <variant>

using namespace mlir;
using namespace dynamatic;
using namespace circt::handshake;

//===----------------------------------------------------------------------===//
// OpTree implementation
//===----------------------------------------------------------------------===//

OpTree::OpTree(OpType opType, OpTreeOperand left, OpTreeOperand right)
    : opType(opType), left(left), right(right),
      depth(std::max(getOperandDepth(left), getOperandDepth(right)) + 1),
      adderDepth(std::max(getOperandDepth(left), getOperandDepth(right)) +
                 ((opType == OpType::ADD || OpType::SUB) ? 1U : 0U)),
      numNodes(getOperandNumNodes(left) + getOperandNumNodes(right) + 1) {}

unsigned OpTree::getOperandDepth(OpTreeOperand &operand) {
  if (auto *tree = std::get_if<std::shared_ptr<OpTree>>(&operand))
    return (*tree)->depth;
  return 0U;
}

unsigned OpTree::getOperandAdderDepth(OpTreeOperand &operand) {
  if (auto *tree = std::get_if<std::shared_ptr<OpTree>>(&operand))
    return (*tree)->adderDepth;
  return 0U;
}

unsigned OpTree::getOperandNumNodes(OpTreeOperand &operand) {
  if (auto *tree = std::get_if<std::shared_ptr<OpTree>>(&operand))
    return (*tree)->numNodes;
  return 0U;
}

// NOLINTBEGIN(misc-no-recursion)
Value OpTree::buildTreeRecursive(
    Operation *op, PatternRewriter &rewriter,
    std::unordered_map<size_t, Value> &cstCache,
    std::unordered_map<std::shared_ptr<OpTree>, Value> &resultCache) {

  // Builds the tree corresponding to an operand of the current tree and returns
  // the value associated with it
  auto buildTreeOperand = [&](OpTreeOperand operand) -> Value {
    if (auto *tree = std::get_if<std::shared_ptr<OpTree>>(&operand)) {
      // Build the operand tree recursively, unless it was already computed
      if (auto valIt = resultCache.find(*tree); valIt != resultCache.end()) {
        return valIt->second;
      }
      Value res =
          (*tree)->buildTreeRecursive(op, rewriter, cstCache, resultCache);
      resultCache[*tree] = res;
      return res;
    }
    if (auto *value = std::get_if<size_t>(&operand)) {
      // Create a constant operation, unless an identical one was already
      // created
      if (auto valIt = cstCache.find(*value); valIt != cstCache.end()) {
        return valIt->second;
      }
      auto cstResult =
          rewriter
              .create<arith::ConstantOp>(
                  op->getLoc(),
                  rewriter.getIntegerAttr(op->getResult(0).getType(), *value))
              .getResult();
      cstCache[*value] = cstResult;
      return cstResult;
    }

    // Just return the value-type operand
    Value *val = std::get_if<Value>(&operand);
    assert(val && "variant type is illegal");
    return *val;
  };

  // Build the tree of each operand and combine their results by creating an
  // airthmetic operation of the correct type
  Value leftVal = buildTreeOperand(left);
  Value rightVal = buildTreeOperand(right);
  Value result;
  switch (opType) {
  case OpType::ADD:
    result = rewriter.create<arith::AddIOp>(op->getLoc(), leftVal, rightVal)
                 .getResult();
    break;
  case OpType::SUB:
    result = rewriter.create<arith::SubIOp>(op->getLoc(), leftVal, rightVal)
                 .getResult();
    break;
  case OpType::SHIFT_LEFT:
    result = rewriter.create<arith::ShLIOp>(op->getLoc(), leftVal, rightVal)
                 .getResult();
    break;
  }
  return result;
}
// NOLINTEND(misc-no-recursion)

Value OpTree::buildTree(Operation *op, PatternRewriter &rewriter) {
  std::unordered_map<size_t, Value> cstCache;
  std::unordered_map<std::shared_ptr<OpTree>, Value> resultCache;
  rewriter.setInsertionPoint(op);
  return buildTreeRecursive(op, rewriter, cstCache, resultCache);
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

/// Determines whether the defining operation of a value is a constant -1.
static bool isConstantNegOne(Value val) {
  if (auto cstOp = val.getDefiningOp<arith::ConstantOp>())
    if (auto cstAttr = dyn_cast<IntegerAttr>(cstOp.getValue()))
      return cstAttr.getValue().getSExtValue() == -1;
  return false;
}

/// Determines whether the argument is a multiplication with a constant -1.
static Value isMulTimesNegOne(arith::MulIOp mulOp) {
  // Check whether one of the two operands is a constant -1 value. If yes,
  // return the other operand
  auto mulOperands = mulOp->getOperands();
  auto mul0 = mulOperands[0], mul1 = mulOperands[1];
  if (isConstantNegOne(mul0))
    return mul1;
  if (isConstantNegOne(mul1))
    return mul0;
  return nullptr;
}

/// Returns an integer of the same width as the data type and whose binary
/// representation is all 1s.
static APInt getMaskAllOnes(Type dataType) {
  unsigned dataWidth;
  if (isa<IndexType>(dataType)) {
    dataWidth = IndexType::kInternalStorageBitWidth;
  } else {
    dataWidth = dataType.getIntOrFloatBitWidth();
  }
  SmallVector<uint64_t> bits;
  size_t numElems = (dataWidth >> 6) + 1;
  for (size_t i = 0; i < numElems; ++i)
    bits.push_back(0xFFFFFFFFFFFFFFFF);
  return APInt(dataWidth, bits);
}

namespace {

/// Replaces uses of the result of multiplications of the form x * (-1) with x
/// whenever possible. Currently, this can apply to uses within integer
/// additions and substractions, for which it is always possible to remove the
/// need for the result of the multiplication.
struct ReplaceMulNegOneUsers : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern<arith::MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp mulOp,
                                PatternRewriter &rewriter) const override {
    // Check whether the multiplication has the right form i.e., x * (-1)
    Value oprd = isMulTimesNegOne(mulOp);
    if (!oprd)
      return failure();

    // Iterate over uses of the multiplication results to see if any could
    // directly use the non-constant operand of the multiplication
    Value mulRes = mulOp.getResult();
    bool anyChange = false;
    Location loc = mulOp.getLoc();
    SmallVector<Operation *> mulUsers;
    llvm::copy(mulRes.getUsers(), std::back_inserter(mulUsers));

    for (Operation *user : mulUsers) {
      if (arith::AddIOp addOp = dyn_cast<arith::AddIOp>(user)) {
        // Additions get replaced with an equivalent substraction
        bool isLhs = mulRes == addOp.getLhs();
        Value newLhs = isLhs ? addOp.getRhs() : addOp.getLhs();
        rewriter.replaceOp(
            user,
            rewriter.create<arith::SubIOp>(loc, newLhs, oprd)->getResults());
        anyChange = true;
      } else if (arith::SubIOp subOp = dyn_cast<arith::SubIOp>(user)) {
        // Substractions are replaced with an equivalemt additiom and,
        // potentially, a sign flip (when the multiplication provides the RHS)
        if (mulRes == subOp.getRhs()) {
          rewriter.replaceOp(
              user, rewriter.create<arith::AddIOp>(loc, subOp.getLhs(), oprd)
                        ->getResults());
          anyChange = true;
        } else {
          arith::AddIOp addOp = rewriter.create<arith::AddIOp>(
              mulOp->getLoc(), oprd, subOp.getRhs());
          Value addRes = addOp.getResult();
          Type dataType = addRes.getType();

          // Flip the sign of the addition result. Create a XOR with an all-1
          // bitmask then add 1: -x = (x XOR 11...11) + 1

          // First create the constant mask
          IntegerAttr intAttr =
              rewriter.getIntegerAttr(dataType, getMaskAllOnes(dataType));
          arith::ConstantOp maskOp =
              rewriter.create<arith::ConstantOp>(loc, intAttr);

          // Then create the XOR between the first addition's result and the
          // mask, inverting the former's bits
          arith::XOrIOp xorOp =
              rewriter.create<arith::XOrIOp>(loc, addRes, maskOp.getResult());

          // Finally, add one to the XOR's output to get the negated version of
          // the first result and replace the initial operation
          arith::ConstantOp cstOneOp = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getIntegerAttr(dataType, 1));
          arith::AddIOp negAddOp = rewriter.create<arith::AddIOp>(
              loc, xorOp.getResult(), cstOneOp.getResult());
          rewriter.replaceOp(user, negAddOp->getResults());
          anyChange = true;
        }
      }
    }

    return success(anyChange);
  }
};

} // namespace

namespace {

/// Replaces multiplications with strictly positive constants into simpler
/// combinations of addition, substraction, and shift operations when possible.
struct MulReduceStrength : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern<arith::MulIOp>::OpRewritePattern;

  MulReduceStrength(unsigned maxAdderDepth, MLIRContext *ctx)
      : OpRewritePattern(ctx), maxAdderDepth(maxAdderDepth){};

  LogicalResult matchAndRewrite(arith::MulIOp mulOp,
                                PatternRewriter &rewriter) const override {
    // One of the two mul operands must be a positive constant
    std::optional<APInt> optCst;
    Value otherOperand;
    if (optCst = getPosConstantOperand(mulOp.getLhs()); optCst.has_value())
      otherOperand = mulOp.getRhs();
    else if (optCst = getPosConstantOperand(mulOp.getRhs()); optCst.has_value())
      otherOperand = mulOp.getLhs();
    if (!optCst.has_value())
      return failure();

    // Try to create a bitwise adder tree matching our constraints
    APInt cst = optCst.value();
    auto bitwise = getBitwiseAdderTree(cst, otherOperand);
    if (!bitwise)
      return failure();

    // Build the operation tree and replace the multipliaction with its result
    Value newMulResult = bitwise->buildTree(mulOp, rewriter);
    rewriter.replaceOp(mulOp, newMulResult);
    return success();
  }

private:
  /// Maximum number of adders that are allowed to be chained together.
  unsigned maxAdderDepth;

  /// Returns the list of positions where the binary representation of the input
  /// value has a 1-bit (the LSB has position 0, the MSB has position 63).
  /// Positions are stored in the vector in increasing order.
  SmallVector<size_t, 8> getOneBitPositions(uint64_t value) const;

  /// If the passed value was produced by an arithmetic constan holding a
  /// strictly positive integer, returns the corersponding APInt; otherwise
  /// returns an empty option.
  std::optional<APInt> getPosConstantOperand(Value mulOperand) const;

  /// Attempts to create an operation tree to replace the multiplication with by
  /// looking at all 1-bits in the constant value and creating a tree of adders
  /// that sums up shifted versions of the constant. Returns nullptr if no such
  /// tree could be derived given the performance constraints the rewrite
  /// pattern was created with.
  std::shared_ptr<OpTree> getBitwiseAdderTree(APInt &cst,
                                              Value mulOperand) const;
};

} // namespace

SmallVector<size_t, 8>
MulReduceStrength::getOneBitPositions(uint64_t value) const {
  SmallVector<size_t, 8> positions;
  for (size_t pos = 0; value != 0; ++pos, value >>= 1)
    if ((value & 1) != 0)
      positions.push_back(pos);
  return positions;
}

std::optional<APInt>
MulReduceStrength::getPosConstantOperand(Value mulOperand) const {
  if (Operation *op = mulOperand.getDefiningOp())
    if (arith::ConstantOp cstOp = dyn_cast<arith::ConstantOp>(op))
      if (IntegerAttr intAttr = dyn_cast<IntegerAttr>(cstOp.getValue()))
        if (auto cstValue = intAttr.getValue(); cstValue.isStrictlyPositive())
          return cstValue;
  return {};
}

std::shared_ptr<OpTree>
MulReduceStrength::getBitwiseAdderTree(APInt &cst, Value mulOperand) const {
  auto onePositions = getOneBitPositions(cst.getZExtValue());
  size_t numBits = onePositions.size();
  APInt numBitsAP(8, numBits);

  unsigned log2 = numBitsAP.logBase2();
  bool isPow2 = numBitsAP.isPowerOf2();

  // We can use a tree of adders if the number of leaf adders does not
  // exceed the maximum number allowed at the maximum depth
  unsigned treeDepth = isPow2 ? log2 : log2 + 1;
  if (treeDepth > maxAdderDepth)
    return nullptr;

  // Special case when the multiplication amounts to a single shift
  if (treeDepth == 0)
    return std::make_unique<OpTree>(OpTree(OpType::SHIFT_LEFT,
                                           OpTreeOperand(mulOperand),
                                           OpTreeOperand(onePositions[0])));

  // Compute the total number of adders necessary
  unsigned floorPow2 = (1 << log2);
  unsigned maxLeafAdders = (1 << (treeDepth - 1));
  unsigned numLeafAdders = isPow2 ? (numBits >> 1) : (numBits - floorPow2);
  assert(numLeafAdders > 0 && numLeafAdders <= maxLeafAdders &&
         "num leaves wrong");

  // Create a shift operation for each 1-bit in the constant (except for the
  // LSB, for which there no shift is necessary)
  TreeOperands shiftLeaves;
  for (size_t pos : onePositions) {
    if (pos != 0)
      shiftLeaves.push_back(std::make_shared<OpTree>(OpTree(
          OpType::SHIFT_LEFT, OpTreeOperand(mulOperand), OpTreeOperand(pos))));
    else
      shiftLeaves.push_back(OpTreeOperand(mulOperand));
  }

  // Create trees for the first (possibly incomplete) level of adders
  TreeOperands adderLeaves;
  for (size_t i = 0, j = 0; i < numLeafAdders; ++i, j += 2)
    adderLeaves.push_back(std::make_shared<OpTree>(
        OpTree(OpType::ADD, shiftLeaves[j], shiftLeaves[j + 1])));

  // If the first layer of adders is incomplete, add the leftover shift leaves
  // as the missing adders to get a "full first layer of adders"
  for (size_t i = (1 << numLeafAdders); i < shiftLeaves.size(); ++i)
    adderLeaves.push_back(std::move(shiftLeaves[i]));
  assert(adderLeaves.size() == maxLeafAdders && "size first level wrong");

  if (treeDepth == 1) {
    // We are already at the tree's root, return the top-level adder
    assert(adderLeaves.size() == 1 && "tree should be collapsed to one adder");
    std::shared_ptr<OpTree> *treeRoot =
        std::get_if<std::shared_ptr<OpTree>>(&adderLeaves[0]);
    assert(treeRoot && "root node doesn't have correct type");
    return std::move(*treeRoot);
  }

  // Go up each level of the tree till we reach the top. Use two pointers on
  // tree operands to avoid multiple vector allocations
  TreeOperands *current = &adderLeaves;
  TreeOperands nextData;
  TreeOperands *next = &nextData;
  for (size_t numAdders = 1 << (treeDepth - 2); numAdders > 0;
       numAdders >>= 1) {
    next->clear();
    for (size_t i = 0, j = 0; i < numAdders; ++i, j += 2)
      next->push_back(std::make_shared<OpTree>(
          OpTree(OpType::ADD, (*current)[j], (*current)[j + 1])));
    std::swap(current, next);
  }

  // We are at the tree's root, return the top-level adder
  assert(current->size() == 1 && "tree should be collapsed to one adder");
  std::shared_ptr<OpTree> *treeRoot =
      std::get_if<std::shared_ptr<OpTree>>(&(*current)[0]);
  assert(treeRoot && "root node doesn't have correct type");
  return std::move(*treeRoot);
}

namespace {
/// Promotes signed integer comparisons with provably positive operands into
/// corresponding unsigned integer comparisons. It is important to have explicit
/// unsigned comparisons as much as possible as it lets the bitwidth
/// optimization pass apply its critical bound optimization pattern, which
/// usually reduces the bitwidth of many operations significantly.
struct PromoteSignedCmp : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    arith::CmpIPredicate newPredicate;
    // Only operate on signed comparisons
    switch (cmpOp.getPredicate()) {
    case arith::CmpIPredicate::slt:
      newPredicate = arith::CmpIPredicate::ult;
      break;
    case arith::CmpIPredicate::sle:
      newPredicate = arith::CmpIPredicate::ule;
      break;
    case arith::CmpIPredicate::sgt:
      newPredicate = arith::CmpIPredicate::ugt;
      break;
    case arith::CmpIPredicate::sge:
      newPredicate = arith::CmpIPredicate::uge;
      break;
    default:
      return failure();
    }

    // Promote the signed comparison to an equivalent unsigned one if possible
    if (!isPromotionPossible(cmpOp))
      return failure();
    rewriter.updateRootInPlace(cmpOp,
                               [&]() { cmpOp.setPredicate(newPredicate); });
    return success();
  }

private:
  /// Determines whether it is possible to promote the comparison operation to
  /// an unsigned one by trying to prove that both of its operands are positive
  /// integers.
  bool isPromotionPossible(arith::CmpIOp cmpOp) const;
};
} // namespace

bool PromoteSignedCmp::isPromotionPossible(arith::CmpIOp cmpOp) const {
  NumericAnalysis analysis;
  return analysis.getRange(cmpOp.getLhs()).isPositive() &&
         analysis.getRange(cmpOp.getRhs()).isPositive();
}

namespace {
/// Simple greedy pattern rewrite driver for arithmetic strength reduction pass.
struct ArithReduceStrengthPass
    : public dynamatic::impl::ArithReduceStrengthBase<ArithReduceStrengthPass> {

  ArithReduceStrengthPass(unsigned maxAdderDepthMul) {
    this->maxAdderDepthMul = maxAdderDepthMul;
  }

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    RewritePatternSet patterns{ctx};
    patterns.add<ReplaceMulNegOneUsers, PromoteSignedCmp>(ctx);
    /// TODO: (RamirezLucas) Any provided value is somewhat arbitrary here.
    /// Ultimately, this should be driven by models of component delays (same
    /// as for buffer placement) as well as a general optimization strategy
    /// (area, performance, mixed)
    patterns.add<MulReduceStrength>(maxAdderDepthMul, ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config)))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createArithReduceStrength(unsigned maxAdderDepthMul) {
  return std::make_unique<ArithReduceStrengthPass>(maxAdderDepthMul);
}
