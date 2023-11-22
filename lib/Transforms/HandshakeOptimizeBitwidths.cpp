//===- HandshakeOptimizeBitwidths.cpp - Optimize channel widths -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a fairly standard bitwidth optimization pass using two set of
// rewrite patterns that are applied greedily and recursively on the IR until it
// converges. In addition to classical arithmetic optimizations presented, for
// example, in this paper
// (https://ieeexplore.ieee.org/abstract/document/959864), Handshake operations
// are also bitwidth-optimized according to their specific semantics. The end
// goal of the pass is to reduce the area taken up by the circuit modeled at the
// Handhshake level.
//
// Note on shift operation handling (forward and backward): the logic of
// truncating a value only to extend it again immediately may seem unnecessary,
// but it in fact allows the rest of the rewrite patterns to understand that
// a value fits on less bits than what the original value suggests. This is
// slightly convoluted but we are forced to do this like that since shift
// operations enforce that all their operands are of the same type. Ideally, we
// would have a Handshake version of shift operations that accept varrying
// bitwidths between its operands and result.
//
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeOptimizeBitwidths.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/HandshakeMinimizeCstWidth.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <functional>

using namespace mlir;
using namespace circt;
using namespace dynamatic;

namespace {
/// Extension type. When backtracing through extension operations, serves to
/// remember the type of any extension we may have encountered along the way.
/// Then, when modifying a value's bitwidth, serves to guide the determination
/// of which extension operation to use.
/// - UNKNOWN when no extension has been encountered / when a value's signedness
/// should determine its extension type.
/// - LOGICAL when only logical extensions have been encountered / when a value
/// should be logically extended.
/// - ARITHMETIC when only arithmaric extensions have been encountered / when a
/// value should be arithmetically extended.
/// - CONFLICT when both logical and arithmetic extensions have been encountered
/// when it's not possible to accurately determine what type of extension to
/// use for a value.
enum class ExtType { UNKNOWN, LOGICAL, ARITHMETIC, CONFLICT };

/// Shortcut for a value accompanied by its corresponding extension type.
using ExtValue = std::pair<Value, ExtType>;

/// Holds a set of operations that were already visisted during backtracking.
using VisitedOps = SmallPtrSet<Operation *, 4>;
} // namespace

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Determines whether the given type is able to be bitwidth optimized.
static inline bool isValidType(Type type) { return isa<IntegerType>(type); }

/// Retuns the ceiling of the logarithm in base 2 of the given value.
static inline unsigned getOptAddrWidth(unsigned value) {
  return APInt(APInt::APINT_BITS_PER_WORD, value).ceilLogBase2();
}

/// Backtracks through defining operations of the value as long as they are
/// extension operations. Returns the "minimal value", i.e., the potentially
/// different value that represents the same number as the originally provided
/// one but without all bits added by extension operations. During the forward
/// pass, the returned value gives an indication of how many bits of the
/// original value can be safely discarded. If an extension type is provided and
/// the function is able to backtrack through any extension operation, updates
/// the extension type with respect to the latter.
// NOLINTNEXTLINE(misc-no-recursion)
static Value getMinimalValue(Value val, ExtType *ext = nullptr) {
  // Ignore values whose type isn't optimizable
  Type type = val.getType();
  if (!isValidType(type))
    return val;

  // Only backtrack through values that were produced by extension operations
  while (Operation *defOp = val.getDefiningOp()) {
    if (!isa<arith::ExtSIOp, arith::ExtUIOp>(defOp))
      return val;

    // Update the extension type using the nature of the current extension
    // operation and the current type
    if (ext) {
      switch (*ext) {
      case ExtType::UNKNOWN:
        *ext =
            isa<arith::ExtSIOp>(defOp) ? ExtType::ARITHMETIC : ExtType::LOGICAL;
        break;
      case ExtType::LOGICAL:
        if (isa<arith::ExtSIOp>(defOp))
          *ext = ExtType::CONFLICT;
        break;
      case ExtType::ARITHMETIC:
        if (isa<arith::ExtUIOp>(defOp))
          *ext = ExtType::CONFLICT;
        break;
      default:
        break;
      }
    }
    // Backtrack through the extension operation
    val = defOp->getOperand(0);
  }

  return val;
}

// Backtracks through defining operations of the given value as long as they are
// "single data input data-forwarders" (i.e., Handshake operations which forward
// one their single "data input" to one of their outputs).
static Value backtrack(Value val) {
  VisitedOps visitedOps;
  while (Operation *defOp = val.getDefiningOp()) {
    // Stop when reaching an operation that was already backtracked through
    if (auto [_, isNewOp] = visitedOps.insert(defOp); !isNewOp)
      return val;

    if (isa<handshake::BufferOp, handshake::ForkOp, handshake::LazyForkOp,
            handshake::BranchOp>(defOp))
      val = defOp->getOperand(0);
    if (auto condOp = dyn_cast<handshake::ConditionalBranchOp>(defOp))
      val = condOp.getDataOperand();
    else if (auto mergeLikeOp =
                 dyn_cast<handshake::MergeLikeOpInterface>(defOp)) {
      if (auto dataOpr = mergeLikeOp.getDataOperands(); dataOpr.size() == 1)
        val = dataOpr[0];
    } else
      return val;
  }

  // Stop backtracking when reaching function arguments
  return val;
}

static Value backtrackToMinimalValue(Value val, ExtType *ext = nullptr) {
  Value newVal;
  while ((newVal = getMinimalValue(backtrack(val), ext)) != val)
    val = newVal;
  return newVal;
}

/// Returns the maximum number of bits that are used by any of the value's
/// users. If the value has no users, returns 0. During the backward pass, the
/// returned value gives an indication of how many high-significant bits can be
/// safely truncated away from the value during optimization.
static unsigned getUsefulResultWidth(Value val) {
  Type resType = val.getType();
  assert(isValidType(resType) && "value must be valid type");

  // Find the value use that discards the least amount of bits. This gives us
  // the amount of bits of the value that can be safely discarded
  std::optional<unsigned> maxWidth;
  for (Operation *user : val.getUsers()) {
    // Ignore sinks, since they do not actually use their operand for anything
    if (isa<handshake::SinkOp>(user))
      continue;
    if (!isa<arith::TruncIOp>(user))
      return resType.getIntOrFloatBitWidth();
    unsigned truncWidth = user->getResult(0).getType().getIntOrFloatBitWidth();
    maxWidth = std::max(maxWidth.value_or(0), truncWidth);
  }

  return maxWidth.value_or(0);
}

/// Produces a value that matches the content of the passed value but whose
/// bitwidth is modified to equal the target width. Inserts an extension or
/// truncation operation in the IR after the original value if necessary. If an
/// extension operation is required, the provided extension type determines
/// which type of extension operation is inserted. If the extension type is
/// unknown, the value's signedness  determines whether the extension should be
/// logical or arithmetic.
static Value modVal(ExtValue extVal, unsigned targetWidth,
                    PatternRewriter &rewriter) {
  auto &[val, ext] = extVal;
  Type type = val.getType();
  assert(isValidType(type) && "value must be valid type");

  // Return the original value when it already has the target width
  unsigned width = type.getIntOrFloatBitWidth();
  if (width == targetWidth)
    return val;

  // Otherwise, insert a bitwidth modification operation to create a value of
  // the target width
  Operation *newOp = nullptr;
  rewriter.setInsertionPointAfterValue(val);
  if (width < targetWidth) {
    if (ext == ExtType::CONFLICT) {
      // If the extension type is conflicting, just emit a warning and hope for
      // the best
      Operation *defOp = val.getDefiningOp();
      std::string origin;
      if (defOp)
        origin = "operation result";
      else {
        defOp = val.getParentBlock()->getParentOp();
        origin = "function argument";
      }
      defOp->emitWarning()
          << "Conflicting extension type given for " << origin
          << ", optimization result may change circuit semantics.";
    }
    if (ext == ExtType::LOGICAL ||
        (ext == ExtType::UNKNOWN && type.isUnsignedInteger()))
      newOp = rewriter.create<arith::ExtUIOp>(
          val.getLoc(), rewriter.getIntegerType(targetWidth), val);
    else
      newOp = rewriter.create<arith::ExtSIOp>(
          val.getLoc(), rewriter.getIntegerType(targetWidth), val);
  } else
    newOp = rewriter.create<arith::TruncIOp>(
        val.getLoc(), rewriter.getIntegerType(targetWidth), val);

  inheritBBFromValue(val, newOp);
  return newOp->getResult(0);
}

/// Recursive version of isOperandInCycle which includes an additional
/// parameter to keep track of which operations were already visited during
/// backtracking to avoid looping forever. See overload's documentation for more
/// details.
/// NOLINTNEXTLINE(misc-no-recursion)
static bool isOperandInCycle(Value val, OpResult res,
                             DenseSet<Value> &mergedValues,
                             VisitedOps &visitedOps) {
  // Stop when we've reached the result of the merge-like operation
  if (val == res)
    return true;

  // Stop when reaching function arguments
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return false;

  // Stop when reaching an operation that was already backtracked through
  if (auto [_, isNewOp] = visitedOps.insert(defOp); !isNewOp)
    return true;

  // Backtrack through operations that end up "forwarding" one of their
  // inputs to the output
  if (isa<handshake::BufferOp, handshake::ForkOp, handshake::LazyForkOp,
          handshake::BranchOp, arith::ExtSIOp, arith::ExtUIOp>(defOp))
    return isOperandInCycle(defOp->getOperand(0), res, mergedValues,
                            visitedOps);
  if (auto condOp = dyn_cast<handshake::ConditionalBranchOp>(defOp))
    return isOperandInCycle(condOp.getDataOperand(), res, mergedValues,
                            visitedOps);

  // NOLINTNEXTLINE(misc-no-recursion)
  auto recurseMergeLike = [&](ValueRange dataOperands) -> bool {
    bool oneOprInCycle = false;
    SmallVector<Value> mergeOperands;
    for (Value mergeLikeOpr : dataOperands) {
      VisitedOps nestedVisitedOps(visitedOps);
      if (isOperandInCycle(mergeLikeOpr, res, mergedValues, nestedVisitedOps))
        oneOprInCycle = true;
      else
        mergeOperands.push_back(mergeLikeOpr);
    }

    // If the merge-like operation is part of the cycle through one of its data
    // operands, add other data operands not part of the cycle to the merged
    // values
    if (oneOprInCycle)
      for (Value &outOfCycleOpr : mergeOperands)
        mergedValues.insert(outOfCycleOpr);
    return oneOprInCycle;
  };

  // Recursively explore data operands of merge-like operations to find cycles
  if (auto mergeLikeOp = dyn_cast<handshake::MergeLikeOpInterface>(defOp))
    return recurseMergeLike(mergeLikeOp.getDataOperands());
  if (auto selectOp = dyn_cast<arith::SelectOp>(defOp))
    return recurseMergeLike(
        ValueRange{selectOp.getTrueValue(), selectOp.getFalseValue()});

  return false;
}

/// Determines whether it is possible to backtrack the value to the result by
/// only going through defining Handshake operations that act as "data
/// forwarders" i.e, operations that forward one of their data inputs to one of
/// their outputs without modification. If yes, then we say the value and result
/// are in the same cycle and the function returns true; otherwise, the function
/// returns false. When the function returns true, mergedValues represents the
/// set of values that are fed inside the cycle through operands of merge-like
/// operations that are on the path between value and result. When the function
/// returns false, the value of mergedValues is undefined.
static bool isOperandInCycle(Value val, OpResult res,
                             DenseSet<Value> &mergedValues) {
  VisitedOps visitedOps;
  return isOperandInCycle(val, res, mergedValues, visitedOps);
}

/// Replaces an operation with two operands and one result of the same integer
/// or floating type with an operation of the same type but whose operands and
/// result bitwidths have been modified to match the provided optimized
/// bitwidth. Extension and truncation operations are inserted as necessary to
/// satisfy the IR and bitwidth constraints.
template <typename Op>
static void modArithOp(Op op, ExtValue lhs, ExtValue rhs, unsigned optWidth,
                       ExtType extRes, PatternRewriter &rewriter) {
  Type resType = op->getResult(0).getType();
  assert(isValidType(resType) && "result must have valid type");
  unsigned resWidth = resType.getIntOrFloatBitWidth();

  // Create a new operation as well as appropriate bitwidth
  // modification operations to keep the IR valid
  Value newLhs = modVal(lhs, optWidth, rewriter);
  Value newRhs = modVal(rhs, optWidth, rewriter);
  rewriter.setInsertionPoint(op);
  auto newOp = rewriter.create<Op>(op.getLoc(), newLhs, newRhs);
  Value newRes = modVal({newOp->getResult(0), extRes}, resWidth, rewriter);
  inheritBB(op, newOp);

  // Replace uses of the original operation's result with
  // the result of the optimized operation we just created
  rewriter.replaceOp(op, newRes);
}

//===----------------------------------------------------------------------===//
// Transfer functions for arith operations
//===----------------------------------------------------------------------===//

/// Transfer function for add/sub operations or alike.
static inline unsigned addWidth(unsigned lhs, unsigned rhs) {
  return std::max(lhs, rhs) + 1;
}

/// Transfer function for mul operations or alike.
static inline unsigned mulWidth(unsigned lhs, unsigned rhs) {
  return lhs + rhs;
}

/// Transfer function for div/rem operations or alike.
static inline unsigned divWidth(unsigned lhs, unsigned _) { return lhs + 1; }

/// Transfer function for and operations or alike.
static inline unsigned andWidth(unsigned lhs, unsigned rhs) {
  return std::min(lhs, rhs);
}

/// Transfer function for or/xor operations or alike.
static inline unsigned orWidth(unsigned lhs, unsigned rhs) {
  return std::max(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Configurations for data optimization of Handshake operations
//===----------------------------------------------------------------------===//

namespace {

/// Holds overridable methods called from the HandshakeOptData rewrite pattern
/// The template parameter of this class is meant to hold a Handshake operation
/// type. Subclassing this class allows to specify, for a specific operation
/// type, the operations/results that carry the data value whose bitwidth may be
/// optimized as well as to tweak the creation process of new operation
/// instances. The default configuration works for Handshake operations whose
/// operands and results all represent the data value (e.g., merge).
template <typename Op>
class OptDataConfig {
public:
  /// Constructs the configuration from the specific operation being
  /// transformed.
  OptDataConfig(Op op) : op(op){};

  /// Returns the list of operands that carry data. The method must return at
  /// least one operand. If multiple operands are returned, they must all have
  /// the same data type, which must also be shared by all results returned by
  /// getDataResults.
  virtual SmallVector<Value> getDataOperands() { return op->getOperands(); }

  /// Returns the list of results that carry data. The method must return at
  /// least one result. If multiple results are returned, they must all have
  /// the same data type, which must also be shared by all operands returned by
  /// getDataOperands.
  virtual SmallVector<Value> getDataResults() { return op->getResults(); }

  /// Determines the list of operands that will be given to the builder of the
  /// optimized operation from the optimized data width, extension type, and
  /// list of minimal data operands of the original operation. The vector given
  /// as last argument is filled with the new operands.
  virtual void getNewOperands(unsigned optWidth, ExtType ext,
                              SmallVector<Value> &minDataOperands,
                              PatternRewriter &rewriter,
                              SmallVector<Value> &newOperands) {
    llvm::transform(minDataOperands, std::back_inserter(newOperands),
                    [&](Value val) {
                      return modVal({val, ext}, optWidth, rewriter);
                    });
  }

  /// Determines the list of result types that will be given to the builder of
  /// the optimized operation. The dataType is the type shared by all data
  /// results. The vector given as last argument is filled with the new result
  /// types.
  virtual void getResultTypes(Type dataType, SmallVector<Type> &newResTypes) {
    for (size_t i = 0, numResults = op->getNumResults(); i < numResults; ++i)
      newResTypes.push_back(dataType);
  }

  /// Creates and returns the optimized operation from its result types and
  /// operands. The default builder for the operation must be available for the
  /// default implementation of this function.
  virtual Op createOp(SmallVector<Type> &newResTypes,
                      SmallVector<Value> &newOperands,
                      PatternRewriter &rewriter) {
    return rewriter.create<Op>(op.getLoc(), newResTypes, newOperands);
  }

  /// Determines the list of values that the original operation will be replaced
  /// with. These are the results of the newly inserted optimized operations
  /// whose bitwidth is modified to match those of the original operation. The
  /// width is the width that was shared by all data operands in the original
  /// operation. The vector given as last argument is filled with the new
  /// values.
  virtual void modResults(Op newOp, unsigned width, ExtType ext,
                          PatternRewriter &rewriter,
                          SmallVector<Value> &newResults) {
    llvm::transform(newOp->getResults(), std::back_inserter(newResults),
                    [&](OpResult res) {
                      return modVal({res, ext}, width, rewriter);
                    });
  }

  /// Default destructor declared virtual because of virtual methods.
  virtual ~OptDataConfig() = default;

protected:
  /// The operation currently being transformed.
  Op op;
};

/// Special configuration for control merges required because of the index
/// result which does not carry data.
class CMergeDataConfig : public OptDataConfig<handshake::ControlMergeOp> {
public:
  CMergeDataConfig(handshake::ControlMergeOp op) : OptDataConfig(op){};

  SmallVector<Value> getDataResults() override {
    return SmallVector<Value>{op.getResult()};
  }

  void getResultTypes(Type dataType, SmallVector<Type> &newResTypes) override {
    for (size_t i = 0, numResults = op->getNumResults() - 1; i < numResults;
         ++i)
      newResTypes.push_back(dataType);
    newResTypes.push_back(op.getIndex().getType());
  }

  void modResults(handshake::ControlMergeOp newOp, unsigned width, ExtType ext,
                  PatternRewriter &rewriter,
                  SmallVector<Value> &newResults) override {
    newResults.push_back(modVal({newOp.getResult(), ext}, width, rewriter));
    newResults.push_back(newOp.getIndex());
  }
};

/// Special configuration for muxes required because of the select operand
/// which does not carry data.
class MuxDataConfig : public OptDataConfig<handshake::MuxOp> {
public:
  MuxDataConfig(handshake::MuxOp op) : OptDataConfig(op){};

  SmallVector<Value> getDataOperands() override { return op.getDataOperands(); }

  void getNewOperands(unsigned optWidth, ExtType ext,
                      SmallVector<Value> &minDataOperands,
                      PatternRewriter &rewriter,
                      SmallVector<Value> &newOperands) override {
    newOperands.push_back(op.getSelectOperand());
    llvm::transform(minDataOperands, std::back_inserter(newOperands),
                    [&](Value val) {
                      return modVal({val, ext}, optWidth, rewriter);
                    });
  }
};

/// Special configuration for conditional branches required because of the
/// condition operand which does not carry data.
class CBranchDataConfig : public OptDataConfig<handshake::ConditionalBranchOp> {
public:
  CBranchDataConfig(handshake::ConditionalBranchOp op) : OptDataConfig(op){};

  SmallVector<Value> getDataOperands() override {
    return SmallVector<Value>{op.getDataOperand()};
  }

  void getNewOperands(unsigned optWidth, ExtType ext,
                      SmallVector<Value> &minDataOperands,
                      PatternRewriter &rewriter,
                      SmallVector<Value> &newOperands) override {
    newOperands.push_back(op.getConditionOperand());
    newOperands.push_back(
        modVal({minDataOperands[0], ext}, optWidth, rewriter));
  }
};

/// Special configuration for buffers required because of the buffer type
/// attribute and custom builder.
class BufferDataConfig : public OptDataConfig<handshake::BufferOp> {
public:
  BufferDataConfig(handshake::BufferOp op) : OptDataConfig(op){};

  SmallVector<Value> getDataOperands() override {
    return SmallVector<Value>{op.getOperand()};
  }

  void getNewOperands(unsigned optWidth, ExtType ext,
                      SmallVector<Value> &minDataOperands,
                      PatternRewriter &rewriter,
                      SmallVector<Value> &newOperands) override {
    newOperands.push_back(
        modVal({minDataOperands[0], ext}, optWidth, rewriter));
  }

  handshake::BufferOp createOp(SmallVector<Type> &newResTypes,
                               SmallVector<Value> &newOperands,
                               PatternRewriter &rewriter) override {
    return rewriter.create<handshake::BufferOp>(
        op.getLoc(), newOperands[0], op.getNumSlots(), op.getBufferType());
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Patterns for Handshake operations
//===----------------------------------------------------------------------===//

namespace {

/// Generic rewrite pattern for Handshake operations forwarding a "data value"
/// from their operand(s) to their result(s). The first template parameter is
/// meant to hold a Handshake operation type on which to apply the pattern,
/// while the second is meant to hold a subclass of OptDataConfig (or the class
/// itself) that specifies how the transformation may be performed on that
/// specific operation type. We use the latter as a way to reduce code
/// duplication, since a number of Handshake operations do not purely follow
/// this "data forwarding" behavior (e.g., they may have a separate operand,
/// like the index operand for muxes) yet their "data-carrying" operands/results
/// can be optimized in the same way as "pure data-forwarding" operations (e.g.,
/// merges). If possible, the pattern replaces the matched operation with one
/// whose bitwidth has been optimized.
///
/// The "data-carrying" operands/results are optimized in the standard way
/// during both the forward and backward passes. In forward mode, the largest
/// "minimal" data operand width is used to potentially reduce the bitwidth of
/// data results. In backward mode, the maximum number of bits used from any of
/// the data results drives a potential reduction in the number of bits in the
/// data operands.
template <typename Op, typename Cfg>
struct HandshakeOptData : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  HandshakeOptData(bool forward, MLIRContext *ctx)
      : OpRewritePattern<Op>(ctx), forward(forward) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Cfg cfg(op);
    SmallVector<Value> dataOperands = cfg.getDataOperands();
    SmallVector<Value> dataResults = cfg.getDataResults();
    assert(!dataOperands.empty() && "op must have at least one data operand");
    assert(!dataResults.empty() && "op must have at least one data result");

    Type dataType = dataResults[0].getType();
    if (!isValidType(dataType))
      return failure();

    // Get the operation's data operands actual widths
    SmallVector<Value> minDataOperands;
    ExtType ext = ExtType::UNKNOWN;
    llvm::transform(dataOperands, std::back_inserter(minDataOperands),
                    [&](Value val) { return getMinimalValue(val, &ext); });

    // Check whether we can reduce the bitwidth of the operation
    unsigned optWidth = 0;
    if (forward) {
      for (Value opr : minDataOperands)
        optWidth = std::max(optWidth, opr.getType().getIntOrFloatBitWidth());
    } else {
      for (Value res : dataResults)
        optWidth = std::max(optWidth, getUsefulResultWidth(res));
    }
    unsigned dataWidth = dataType.getIntOrFloatBitWidth();
    if (optWidth >= dataWidth)
      return failure();

    // Create a new operation as well as appropriate bitwidth modification
    // operations to keep the IR valid
    SmallVector<Value> newOperands, newResults;
    SmallVector<Type> newResTypes;
    cfg.getNewOperands(optWidth, ext, minDataOperands, rewriter, newOperands);
    cfg.getResultTypes(rewriter.getIntegerType(optWidth), newResTypes);
    rewriter.setInsertionPoint(op);
    Op newOp = cfg.createOp(newResTypes, newOperands, rewriter);
    inheritBB(op, newOp);
    cfg.modResults(newOp, dataWidth, ext, rewriter, newResults);

    // Replace uses of the original operation's results with the results of the
    // optimized operation we just created
    rewriter.replaceOp(op, newResults);
    return success();
  }

private:
  /// Indicates whether this pattern is part of the forward or backward pass.
  bool forward;
};

/// Template specialization of data optimization rewrite pattern for Handshake
/// operations that do not require a specific configuration.
template <typename Op>
using HandshakeOptDataNoCfg = HandshakeOptData<Op, OptDataConfig<Op>>;

/// Optimizes the bitwidth of muxes' select operand so that it can just support
/// the number of data operands. This pattern can be applied as part of a single
/// greedy rewriting pass and doesn't need to be part of the forward/backward
/// process.
struct HandshakeMuxSelect : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {
    // Compute the number of bits required to index into the mux data operands
    unsigned optWidth = getOptAddrWidth(muxOp.getDataOperands().size());

    // Check whether we can reduce the bitwidth of the operation
    Value selectOperand = muxOp.getSelectOperand();
    Type selectType = selectOperand.getType();
    assert(isa<IntegerType>(selectType) &&
           "select operand must have integer type");
    unsigned selectWidth = selectType.getIntOrFloatBitWidth();
    if (optWidth >= selectWidth)
      return failure();

    // Replace the select operand with one with optimize bitwidth
    Value newSelect =
        modVal({selectOperand, ExtType::LOGICAL}, optWidth, rewriter);
    rewriter.updateRootInPlace(
        muxOp, [&] { muxOp->replaceUsesOfWith(selectOperand, newSelect); });
    return success();
  }
};

/// Downgrades muxes with a single input into simpler yet equivalent merges.
struct DowngradeSingleInputMuxes : public OpRewritePattern<handshake::MuxOp> {
  using OpRewritePattern<handshake::MuxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MuxOp muxOp,
                                PatternRewriter &rewriter) const override {
    ValueRange dataOperands = muxOp.getDataOperands();
    if (dataOperands.size() != 1)
      return failure();

    rewriter.setInsertionPoint(muxOp);
    handshake::MergeOp mergeOp =
        rewriter.create<handshake::MergeOp>(muxOp.getLoc(), dataOperands);
    inheritBB(muxOp, mergeOp);
    rewriter.replaceOp(muxOp, mergeOp.getResult());
    return success();
  }
};

/// Optimizes the bitwidth of control merges' index result so that it can just
/// support the number of data operands. This pattern can be applied as part of
/// a single greedy rewriting pass and doesn't need to be part of the
/// forward/backward process.
struct HandshakeCMergeIndex
    : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {
    // Compute the number of bits required to index into the mux data operands
    unsigned optWidth = getOptAddrWidth(cmergeOp->getNumOperands());

    // Check whether we can reduce the bitwidth of the operation
    Value indexResult = cmergeOp.getIndex();
    Type indexType = indexResult.getType();
    assert(isa<IntegerType>(indexType) &&
           "select operand must have integer type");
    unsigned indexWidth = indexType.getIntOrFloatBitWidth();
    if (optWidth >= indexWidth)
      return failure();

    // Create a new control merge with whose index result is optimized
    SmallVector<Type, 2> resTypes;
    resTypes.push_back(cmergeOp.getDataType());
    resTypes.push_back(rewriter.getIntegerType(optWidth));
    rewriter.setInsertionPoint(cmergeOp);
    auto newOp = rewriter.create<handshake::ControlMergeOp>(
        cmergeOp.getLoc(), resTypes, cmergeOp.getDataOperands());
    inheritBB(cmergeOp, newOp);
    Value modIndex =
        modVal({newOp.getIndex(), ExtType::LOGICAL}, indexWidth, rewriter);
    rewriter.replaceOp(cmergeOp, ValueRange{newOp.getResult(), modIndex});
    return success();
  }
};

/// Downgrades control merges with a single input into simpler yet equivalent
/// merges. If necessary, inserts a sourced 0 constant to replace any real uses
/// of the index result of erased control merges.
struct DowngradeSingleInputControlMerges
    : public OpRewritePattern<handshake::ControlMergeOp> {
  using OpRewritePattern<handshake::ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ControlMergeOp cmergeOp,
                                PatternRewriter &rewriter) const override {
    ValueRange dataOperands = cmergeOp.getDataOperands();
    if (dataOperands.size() != 1)
      return failure();

    rewriter.setInsertionPoint(cmergeOp);
    Value indexRes = cmergeOp.getIndex();
    if (!indexRes.use_empty()) {
      // If the index has uses, create a sourced 0 constant to replace it
      handshake::SourceOp srcOp = rewriter.create<handshake::SourceOp>(
          cmergeOp->getLoc(), rewriter.getNoneType());
      inheritBB(cmergeOp, srcOp);

      // Build the attribute for the constant
      Type indexResType = indexRes.getType();
      handshake::ConstantOp cstOp = rewriter.create<handshake::ConstantOp>(
          cmergeOp.getLoc(), indexResType,
          rewriter.getIntegerAttr(indexResType, 0), srcOp.getResult());
      inheritBB(cmergeOp, cstOp);

      // Replace the cmerge's index result with a constant 0
      rewriter.replaceAllUsesWith(indexRes, cstOp.getResult());
    }

    handshake::MergeOp mergeOp =
        rewriter.create<handshake::MergeOp>(cmergeOp.getLoc(), dataOperands);
    inheritBB(cmergeOp, mergeOp);
    rewriter.replaceAllUsesWith(cmergeOp.getResult(), mergeOp.getResult());
    rewriter.eraseOp(cmergeOp);
    return success();
  }
};

/// Optimizes the bitwidth of memory controller's address-carrying channels so
/// that they can just support indexing into the memory region attached to the
/// controller. This pattern can be applied as part of a single
/// greedy rewriting pass and doesn't need to be part of the forward/backward
/// process.
struct HandshakeMCAddress
    : public OpRewritePattern<handshake::MemoryControllerOp> {
  using OpRewritePattern<handshake::MemoryControllerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::MemoryControllerOp mcOp,
                                PatternRewriter &rewriter) const override {
    unsigned optWidth =
        getOptAddrWidth(mcOp.getMemRef().getType().getDimSize(0));

    FuncMemoryPorts ports = mcOp.getPorts();
    if (ports.addrWidth == 0 || optWidth >= ports.addrWidth)
      return failure();

    ValueRange inputs = mcOp.getMemInputs();
    // Optimizes the bitwidth of the address channel currently being pointed to
    // by inputIdx, and increment inputIdx before returning the optimized value
    auto getOptAddrInput = [&](unsigned inputIdx) {
      return modVal({getMinimalValue(inputs[inputIdx]), ExtType::LOGICAL},
                    optWidth, rewriter);
    };

    // Iterate over memory controller inputs to create the new inputs and the
    // list of accesses
    SmallVector<Value> newInputs;
    for (GroupMemoryPorts &blockPorts : ports.groups) {
      // Handle eventual control input
      if (blockPorts.hasControl())
        newInputs.push_back(inputs[blockPorts.ctrlPort->getCtrlInputIndex()]);

      for (MemoryPort &port : blockPorts.accessPorts) {
        if (std::optional<LoadPort> loadPort = dyn_cast<LoadPort>(port)) {
          newInputs.push_back(getOptAddrInput(loadPort->getAddrInputIndex()));
        } else {
          std::optional<StorePort> storePort = dyn_cast<StorePort>(port);
          assert(storePort && "port must be load or store");
          newInputs.push_back(getOptAddrInput(storePort->getAddrInputIndex()));
          newInputs.push_back(inputs[storePort->getDataInputIndex()]);
        }
      }
    }

    // Replace the existing memory controller with the optimized one
    rewriter.setInsertionPoint(mcOp);
    auto newOp = rewriter.create<handshake::MemoryControllerOp>(
        mcOp.getLoc(), mcOp.getMemRef(), newInputs, mcOp.getMCBlocks(),
        mcOp.getPorts().getNumPorts<LoadPort, LSQLoadStorePort>());
    inheritBB(mcOp, newOp);
    rewriter.replaceOp(mcOp, newOp.getResults());
    return success();
  }
};

/// Optimizes the bitwidth of memory ports's address-carrying channels so that
/// they can just support indexing into the memory region these ports ultimately
/// talk to. The first template parameter is meant to be either
/// handshake::LoadOpInterface or handshake::StoreOpInterface. This pattern can
/// be applied as part of a single greedy rewriting pass and doesn't need to be
/// part of the forward/backward process.
template <typename Op>
struct HandshakeMemPortAddress : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op memOp,
                                PatternRewriter &rewriter) const override {
    // Check whether we can optimize the address bitwidth
    Value addrRes = memOp.getAddressResult();
    unsigned addrWidth = addrRes.getType().getIntOrFloatBitWidth();
    unsigned optWidth = getUsefulResultWidth(addrRes);
    if (optWidth >= addrWidth)
      return failure();

    Value newAddr =
        modVal({getMinimalValue(memOp.getAddress()), ExtType::LOGICAL},
               optWidth, rewriter);
    rewriter.setInsertionPoint(memOp);
    auto newOp = rewriter.create<Op>(memOp.getLoc(), newAddr.getType(),
                                     memOp.getDataResult().getType(), newAddr,
                                     memOp.getData());
    Value newAddrRes = modVal({newOp.getAddressResult(), ExtType::LOGICAL},
                              addrWidth, rewriter);
    inheritBB(memOp, newOp);
    rewriter.replaceOp(memOp, {newAddrRes, newOp.getDataResult()});
    return success();
  }
};

/// Shortcut for a value accompanied by its identified extension type.
using ExtValue = std::pair<Value, ExtType>;

/// Moves any extension operation feeding into a return operation past the
/// latter to optimize the bitwidth occupied by the return operation itself.
/// This is meant to be part of the forward pass.
/// NOTE: (lucas) This rewrite pattern is unused at the moment because applying
/// it sucessfully makes the return operation's verification function fail
/// (different return result types and enclosing function return types).
/// When we eventually formalize and rework our circuit's interfaces, this may
/// become useful, so here it stays.
struct HandshakeReturnFW
    : public OpRewritePattern<handshake::DynamaticReturnOp> {
  using OpRewritePattern<handshake::DynamaticReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::DynamaticReturnOp retOp,
                                PatternRewriter &rewriter) const override {

    // Try to move potential extension operations after the return
    SmallVector<Value> newOperands;
    SmallVector<ExtType> extTypes;
    bool changed = false;
    for (Value opr : retOp->getOperands()) {
      Type oprType = opr.getType();
      ExtType ext = ExtType::UNKNOWN;
      Value minVal = opr;
      if (isValidType(oprType)) {
        minVal = getMinimalValue(opr, &ext);
        changed |= minVal.getType().getIntOrFloatBitWidth() <
                   oprType.getIntOrFloatBitWidth();
      }
      newOperands.push_back(minVal);
      extTypes.push_back(ext);
    }

    // Check whether the transformation would change anything
    if (!changed)
      return failure();

    // Insert an optimized return operation that moves eventual value extensions
    // after itself
    rewriter.setInsertionPoint(retOp);
    auto newOp = rewriter.create<handshake::DynamaticReturnOp>(retOp->getLoc(),
                                                               newOperands);
    inheritBB(retOp, newOp);

    // Create required extension operations on the new return's results so that
    // the rest of the IR stays valid
    SmallVector<Value> newResults;
    for (auto [newRes, ogResType, ext] : llvm::zip_equal(
             newOp->getResults(), retOp->getResultTypes(), extTypes)) {
      if (isValidType(ogResType))
        newResults.push_back(
            modVal({newRes, ext}, ogResType.getIntOrFloatBitWidth(), rewriter));
      else
        newResults.push_back(newRes);
    }
    rewriter.replaceOp(retOp, newResults);
    return success();
  }
};

/// Optimizes the bitwidth of channels contained inside "forwarding cycles".
/// These are values that generally circulate between branch-like and merge-like
/// operations without modification (i.e., in a block that branches to itself).
/// These require special treatment to be optimized as the rest of the rewrite
/// patterns only look at the operation they are matched on when optimizing,
/// whereas this pattern attempts to backtracks through operands of merge-like
/// operations to identify whether it was produced by the operation itself. If
/// an operand is identified as being part of a cycle, all other out-of-cycle
/// merged values incoming to the cycle through merge-like operation operands
/// are considered to determine the optimized width that can be given to the
/// in-cycle operand.
///
/// The first template parameter is meant to be a merge-like operation i.e., a
/// Handhsake operation implementing the MergeLikeOpInterface trait on which to
/// apply the rewrite pattern. The second template parameter is meant to hold a
/// subclass of OptDataConfig (or the class itself) that specifies how the
/// transformation may be performed on that specific operation type.
template <typename Op, typename Cfg>
struct ForwardCycleOpt : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    // This pattern only works for merge-like operations with a valid data type
    auto mergeLikeOp =
        dyn_cast<handshake::MergeLikeOpInterface>((Operation *)op);
    if (!mergeLikeOp)
      return failure();
    OpResult dataRes = op->getResult(0);
    if (!isValidType(dataRes.getType()))
      return failure();

    // For each operand, determine whether it is in a forwarding cycle. If yes,
    // keep track of other values coming in the cycle through merge-like ops
    OperandRange dataOperands = mergeLikeOp.getDataOperands();
    SmallVector<bool> operandInCycle;
    DenseSet<Value> allMergedValues, mergedValues;
    for (auto opr : dataOperands) {
      mergedValues.clear();
      bool inCycle = isOperandInCycle(opr, dataRes, mergedValues);
      operandInCycle.push_back(inCycle);
      if (inCycle)
        for (Value &val : mergedValues)
          allMergedValues.insert(val);
      else
        allMergedValues.insert(opr);
    }

    // Determine the achievable optimized width for operands inside the cycle
    unsigned optWidth = 0;
    ExtType ext = ExtType::UNKNOWN;
    for (Value mergedVal : allMergedValues)
      optWidth = std::max(optWidth, backtrackToMinimalValue(mergedVal, &ext)
                                        .getType()
                                        .getIntOrFloatBitWidth());

    // Check whether we managed to optimize anything
    unsigned dataWidth = dataRes.getType().getIntOrFloatBitWidth();
    if (optWidth >= dataWidth)
      return failure();

    // Get the minimal valuue of all data operands
    SmallVector<Value> minDataOperands;
    for (auto opr : dataOperands)
      minDataOperands.push_back(getMinimalValue(opr));

    // Create a new operation as well as appropriate bitwidth modification
    // operations to keep the IR valid
    Cfg cfg(op);
    SmallVector<Value> newOperands, newResults;
    SmallVector<Type> newResTypes;
    cfg.getNewOperands(optWidth, ext, minDataOperands, rewriter, newOperands);
    cfg.getResultTypes(rewriter.getIntegerType(optWidth), newResTypes);
    rewriter.setInsertionPoint(op);
    Op newOp = cfg.createOp(newResTypes, newOperands, rewriter);
    inheritBB(op, newOp);
    cfg.modResults(newOp, dataWidth, ext, rewriter, newResults);

    // Replace uses of the original operation's results with the results of the
    // optimized operation we just created
    rewriter.replaceOp(op, newResults);
    return success();
  }
};

/// Template specialization of forward cycle optimization rewrite pattern for
/// Handshake operations that do not require a specific configuration.
template <typename Op>
using ForwardCycleOptNoCfg = ForwardCycleOpt<Op, OptDataConfig<Op>>;

} // namespace

//===----------------------------------------------------------------------===//
// Patterns for arith operations
//===----------------------------------------------------------------------===//

namespace {

/// Transfer function type for arithmetic operations with two operands and a
/// single result of the same type. Returns the result bitwidth required to
/// achieve the operation behavior given the two operands' respective bitwidths.
using FTransfer = std::function<unsigned(unsigned, unsigned)>;

/// Generic rewrite pattern for arith operations that have two operands and a
/// single result, all of the same type. The first template parameter is meant
/// to hold an arith operation satisfying such constraints. If possible, the
/// pattern replaces the matched operation with one whose bitwidth has been
/// optimized.
///
/// In forward mode, the pattern uses a transfer function to determine the
/// required result bitwidth based on the operands' respective "minimal"
/// bitwidth. In backward mode, the maximum number of bits used from the result
/// drives a potential reduction in the number of bits in the two operands.
template <typename Op>
struct ArithSingleType : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  ArithSingleType(bool forward, FTransfer fTransfer, MLIRContext *ctx)
      : OpRewritePattern<Op>(ctx), forward(forward),
        fTransfer(std::move(fTransfer)) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Type resType = op->getResult(0).getType();
    if (!isValidType(resType))
      return failure();

    // Check whether we can reduce the bitwidth of the operation
    ExtType extLhs = ExtType::UNKNOWN, extRhs = ExtType::UNKNOWN;
    Value minLhs = getMinimalValue(op->getOperand(0), &extLhs);
    Value minRhs = getMinimalValue(op->getOperand(1), &extRhs);
    unsigned optWidth;
    if (forward)
      optWidth = fTransfer(minLhs.getType().getIntOrFloatBitWidth(),
                           minRhs.getType().getIntOrFloatBitWidth());
    else
      optWidth = getUsefulResultWidth(op->getResult(0));
    unsigned resWidth = resType.getIntOrFloatBitWidth();
    if (optWidth >= resWidth)
      return failure();

    // Result extension is always logical for bitwise logical operations and
    // explicitly unsigned operations, othweise it is determined byt the
    // result's type
    ExtType extRes = ExtType::UNKNOWN;
    if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::DivUIOp,
            arith::RemUIOp>((Operation *)op))
      extRes = ExtType::LOGICAL;
    else
      extRes = ExtType::UNKNOWN;
    modArithOp(op, {minLhs, extLhs}, {minRhs, extRhs}, optWidth, extRes,
               rewriter);
    return success();
  }

private:
  /// Indicates whether this pattern is part of the forward or backward pass.
  bool forward;
  /// Transfer function used in forward mode.
  FTransfer fTransfer;
};

/// Optimizes the bitwidth of select operations using the same logic as in the
/// ArithSingleType pattern. The latter cannot be used directly since the select
/// operation has a third i1 operand to select which of the two others to
/// forward to the output.
struct ArithSelect : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  ArithSelect(bool forward, MLIRContext *ctx)
      : OpRewritePattern<arith::SelectOp>(ctx), forward(forward) {}

  LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                PatternRewriter &rewriter) const override {
    Type resType = selectOp.getResult().getType();
    if (!isValidType(resType))
      return failure();

    // Check whether we can reduce the bitwidth of the operation
    ExtType extLhs = ExtType::UNKNOWN, extRhs = ExtType::UNKNOWN;
    Value minLhs = getMinimalValue(selectOp.getTrueValue(), &extLhs);
    Value minRhs = getMinimalValue(selectOp.getFalseValue(), &extRhs);
    unsigned optWidth;
    if (forward)
      optWidth = std::max(minLhs.getType().getIntOrFloatBitWidth(),
                          minRhs.getType().getIntOrFloatBitWidth());
    else
      optWidth = getUsefulResultWidth(selectOp.getResult());
    unsigned resWidth = resType.getIntOrFloatBitWidth();
    if (optWidth >= resWidth)
      return failure();

    // Different operand extension types mean that we don't know how to extend
    // the operation's result, so it cannot be optimized
    if ((extLhs == ExtType::LOGICAL && extRhs == ExtType::ARITHMETIC) ||
        (extLhs == ExtType::ARITHMETIC && extRhs == ExtType::LOGICAL))
      return failure();

    // Create a new operation as well as appropriate bitwidth modification
    // operations to keep the IR valid
    Value newLhs = modVal({minLhs, extLhs}, optWidth, rewriter);
    Value newRhs = modVal({minRhs, extRhs}, optWidth, rewriter);
    rewriter.setInsertionPoint(selectOp);
    auto newOp = rewriter.create<arith::SelectOp>(
        selectOp.getLoc(), selectOp.getCondition(), newLhs, newRhs);
    Value newRes = modVal({newOp->getResult(0), extLhs}, resWidth, rewriter);
    inheritBB(selectOp, newOp);

    // Replace uses of the original operation's result with the result of the
    // optimized operation we just created
    rewriter.replaceOp(selectOp, newRes);
    return success();
  }

private:
  /// Indicates whether this pattern is part of the forward or backward pass.
  bool forward;
};

/// Optimizes the bitwidth of shift-type operations. The first template
/// parameter is meant to be either arith::ShLIOp, arith::ShRSIOp, or
/// arith::ShRUIOp. In both modes (forward and backward), the matched
/// operation's bitwidth may only be reduced when the data operand is shifted by
/// a known constant amount.
template <typename Op>
struct ArithShift : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  ArithShift(bool forward, MLIRContext *ctx)
      : OpRewritePattern<Op>(ctx), forward(forward) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Value toShift = op->getOperand(0);
    Value shiftBy = op->getOperand(1);
    ExtType extToShift = ExtType::UNKNOWN;
    Value minToShift = getMinimalValue(toShift, &extToShift);
    Value minShiftBy = backtrackToMinimalValue(shiftBy);
    bool isRightShift = isa<arith::ShRSIOp, arith::ShRUIOp>((Operation *)op);

    // Check whether we can reduce the bitwidth of the operation
    unsigned resWidth = op->getResult(0).getType().getIntOrFloatBitWidth();
    unsigned optWidth = resWidth;
    unsigned cstVal = 0;
    if (Operation *defOp = minShiftBy.getDefiningOp())
      if (auto cstOp = dyn_cast<handshake::ConstantOp>(defOp)) {
        cstVal = (unsigned)cast<IntegerAttr>(cstOp.getValue()).getInt();
        if (forward) {
          optWidth = minToShift.getType().getIntOrFloatBitWidth();
          if (!isRightShift)
            optWidth += cstVal;
        } else {
          optWidth = getUsefulResultWidth(op->getResult(0));
          if (isRightShift)
            optWidth += cstVal;
        }
      }
    if (optWidth >= resWidth)
      return failure();

    if (forward) {
      // Create a new operation as well as appropriate bitwidth modification
      // operations to keep the IR valid
      Value newToShift = modVal({minToShift, extToShift}, optWidth, rewriter);
      Value newShifyBy =
          modVal({minShiftBy, ExtType::LOGICAL}, optWidth, rewriter);
      rewriter.setInsertionPoint(op);
      auto newOp = rewriter.create<Op>(op.getLoc(), newToShift, newShifyBy);
      Value newRes = newOp->getResult(0);
      if (isRightShift)
        // In the case of a right shift, we first truncate the result of the
        // newly inserted shift operation to discard high-significance bits that
        // we know are 0s, then extend the result back to satisfy the users of
        // the original operation's result
        newRes = modVal({newRes, extToShift}, optWidth - cstVal, rewriter);
      Value modRes = modVal({newRes, extToShift}, resWidth, rewriter);
      inheritBB(op, newOp);

      // Replace uses of the original operation's result with the result of the
      // optimized operation we just created
      rewriter.replaceOp(op, modRes);
    } else {
      Value modToShift = minToShift;
      if (!isRightShift) {
        // In the case of a left shift, we first truncate the shifted integer to
        // discard high-significance bits that were discarded in the result,
        // then extend back to satisfy the users of the original integer
        unsigned requiredToShiftWidth = optWidth - std::min(cstVal, optWidth);
        modToShift =
            modVal({minToShift, extToShift}, requiredToShiftWidth, rewriter);
      }
      modArithOp(op, {modToShift, extToShift}, {minShiftBy, ExtType::LOGICAL},
                 optWidth, extToShift, rewriter);
    }
    return success();
  }

private:
  /// Indicates whether this pattern is part of the forward or backward pass.
  bool forward;
};

/// Optimizes the bitwidth of integer comparisons by looking at the respective
/// "minimal" value of their two operands. This is meant to be part of the
/// forward pass.
struct ArithCmpFW : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    // Check whether we can reduce the bitwidth of the operation
    Value lhs = cmpOp.getLhs();
    Value rhs = cmpOp.getRhs();
    ExtType extLhs = ExtType::UNKNOWN, extRhs = ExtType::UNKNOWN;
    Value minLhs = getMinimalValue(lhs, &extLhs);
    Value minRhs = getMinimalValue(rhs, &extRhs);
    unsigned optWidth = std::max(minLhs.getType().getIntOrFloatBitWidth(),
                                 minRhs.getType().getIntOrFloatBitWidth());
    unsigned actualWidth = lhs.getType().getIntOrFloatBitWidth();
    if (optWidth >= actualWidth)
      return failure();

    // Create a new operation as well as appropriate bitwidth modification
    // operations to keep the IR valid
    Value newLhs = modVal({minLhs, extLhs}, optWidth, rewriter);
    Value newRhs = modVal({minRhs, extRhs}, optWidth, rewriter);
    rewriter.setInsertionPoint(cmpOp);
    auto newOp = rewriter.create<arith::CmpIOp>(
        cmpOp.getLoc(), cmpOp.getPredicate(), newLhs, newRhs);
    inheritBB(cmpOp, newOp);

    // Replace uses of the original operation's result with the result of the
    // optimized operation we just created
    rewriter.replaceOp(cmpOp, newOp.getResult());
    return success();
  }
};

/// Removes truncation operations whose operand is produced by any sequence of
/// extension operations with the same type (logical or arithmetic).
struct ArithExtToTruncOpt : public OpRewritePattern<arith::TruncIOp> {
  using OpRewritePattern<arith::TruncIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncIOp truncOp,
                                PatternRewriter &rewriter) const override {
    // Operand must be produced by an extension operation
    ExtType extType = ExtType::UNKNOWN;
    Value minVal = getMinimalValue(truncOp.getOperand(), &extType);
    if (extType == ExtType::UNKNOWN || extType == ExtType::CONFLICT)
      return failure();

    unsigned finalWidth = truncOp.getResult().getType().getIntOrFloatBitWidth();
    if (finalWidth == minVal.getType().getIntOrFloatBitWidth())
      return failure();

    // Bypass all extensions and truncation operation and replace it with a
    // single bitwidth modification operation
    auto newExtRes = modVal({minVal, extType}, finalWidth, rewriter);
    rewriter.replaceOp(truncOp, {newExtRes});
    return success();
  }
};

/// Optimizes an IR pattern where a comparison between a number and a constant
/// is used to make a control flow decision. Depending on the branch outcome, it
/// is possible to truncate one of the Handshake::ConditionalBranchOp's output
/// to the bitwidth required by the constant involved in the comparison. This is
/// a pattern present in loops whose exist condition is a comparison with a
/// constant, and allows to reduce the bitwidth of the loop iterator in those
/// cases.
struct ArithBoundOpt : public OpRewritePattern<handshake::ConditionalBranchOp> {
  using OpRewritePattern<handshake::ConditionalBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ConditionalBranchOp condOp,
                                PatternRewriter &rewriter) const override {
    // The data type must be optimizable
    Value dataOperand = backtrackToMinimalValue(condOp.getDataOperand());
    if (!isValidType(dataOperand.getType()))
      return failure();

    // Find all comparison operations whose result is used in a logical and to
    // determine the condition operand and which have the data operand as one of
    // their inputs; then determine which comparison gives the tighest bound on
    // each branch outcome
    Value trueRes = condOp.getTrueResult(), falseRes = condOp.getFalseResult();
    std::optional<std::pair<unsigned, ExtType>> trueBranch, falseBranch;
    for (arith::CmpIOp cmpOp : getCmpOps(condOp.getConditionOperand())) {
      ExtType extLhs = ExtType::UNKNOWN, extRhs = ExtType::UNKNOWN;
      Value minLhs = backtrackToMinimalValue(cmpOp.getLhs(), &extLhs);
      Value minRhs = backtrackToMinimalValue(cmpOp.getRhs(), &extRhs);

      // One of the two comparison operands must be the data input
      unsigned width;
      bool isDataLhs;
      ExtType branchExt;
      if (dataOperand == minLhs) {
        width = minRhs.getType().getIntOrFloatBitWidth();
        isDataLhs = true;
        branchExt = extLhs;
      } else if (dataOperand == minRhs) {
        width = minLhs.getType().getIntOrFloatBitWidth();
        isDataLhs = false;
        branchExt = extRhs;
      } else
        continue;

      // Determine whether one of the branches can be optimized and by how much
      Value branch = getBranchToOptimize(condOp, cmpOp, isDataLhs);
      if (!branch)
        continue;
      if (isBoundTight(isDataLhs ? minRhs : minLhs))
        width = getRealOptWidth(cmpOp, width, isDataLhs);

      // Keep track of the best optimization opportunity found so far for the
      // branch
      if (branch == trueRes) {
        if (!trueBranch.has_value() || width < trueBranch.value().first)
          trueBranch = std::make_pair(width, branchExt);
      } else if (!falseBranch.has_value() || width < falseBranch.value().first)
        falseBranch = std::make_pair(width, branchExt);
    }

    // Optimize both branches if possible (in non-degenerate code, only one
    // branch should ever be optimized at a time, since a bound on one side
    // means no bound on the other side)
    rewriter.setInsertionPointAfter(condOp);
    bool anyOptPerformed = false;
    if (trueBranch.has_value())
      anyOptPerformed |= optBranchIfPossible(trueRes, trueBranch->first,
                                             trueBranch->second, rewriter);
    if (falseBranch.has_value())
      anyOptPerformed |= optBranchIfPossible(falseRes, falseBranch->first,
                                             falseBranch->second, rewriter);

    return success(anyOptPerformed);
  }

private:
  /// Returns the list of comparison operations involved in the computation of
  /// the given conditional value (which must have i1 type). All of the
  /// comparisons' respective result are ANDed to compute the given value.
  SmallVector<arith::CmpIOp> getCmpOps(Value condVal) const;

  /// Determines whether the bound that the data operand is compared with is
  /// tight, i.e. whether being strictly closer to 0 than it means we can
  /// represent the number using one less bit than the bound itself.
  bool isBoundTight(Value bound) const;

  /// Determines which branch may be optimized based on the nature of the
  /// comparison and the side of the data operand to the conditional branch.
  Value getBranchToOptimize(handshake::ConditionalBranchOp condOp,
                            arith::CmpIOp cmpOp, bool isDataLhs) const;

  /// Returns the real optimized bitwidth assuming that the bound against which
  /// the comparison is performed is provably tight. The real optimized bitwidth
  /// may be one less than the one passed as argument or identical.
  unsigned getRealOptWidth(arith::CmpIOp cmpOp, unsigned optWidth,
                           bool isDataLhs) const;

  /// Optimizes the branch output provided as argument to the given bitwidth is
  /// there is any benefit in doing so. Returns true if any optimization is
  /// performed; otherwise returns false;
  bool optBranchIfPossible(Value optBranch, unsigned optWidth, ExtType ext,
                           PatternRewriter &rewriter) const;
};

} // namespace

/// NOLINTNEXTLINE(misc-no-recursion)
SmallVector<arith::CmpIOp> ArithBoundOpt::getCmpOps(Value condVal) const {
  Value minVal = backtrackToMinimalValue(condVal);

  // Stop when reaching function arguments
  Operation *defOp = minVal.getDefiningOp();
  if (!defOp)
    return {};

  // If we have reached a comparison operation, return it
  if (arith::CmpIOp cmpOp = dyn_cast<arith::CmpIOp>(defOp))
    return {cmpOp};

  // If we have reached a logical and, backtrack through both its operands as it
  // means the branch condition will be more restrictive than the comparison
  // itself, which doesn't invalidate our optimization
  if (arith::AndIOp andOp = dyn_cast<arith::AndIOp>(defOp)) {
    SmallVector<arith::CmpIOp> cmpOps;
    llvm::copy(getCmpOps(andOp.getLhs()), std::back_inserter(cmpOps));
    llvm::copy(getCmpOps(andOp.getRhs()), std::back_inserter(cmpOps));
    return cmpOps;
  }

  return {};
}

bool ArithBoundOpt::isBoundTight(Value bound) const {
  // Bound must be a constant
  auto cstOp =
      dyn_cast_if_present<handshake::ConstantOp>(bound.getDefiningOp());
  if (!cstOp)
    return false;

  // Constant must have an integer attribute
  auto intAttr = cast<IntegerAttr>(cstOp.getValue());
  if (!intAttr)
    return false;

  // Check whether incrementing/decrementing the value toward 0 changes the
  // number of bits required to represent it.
  APInt val = intAttr.getValue();
  APInt centVal =
      val.isNegative()
          ? APInt(APInt::APINT_BITS_PER_WORD, val.getSExtValue() + 1)
          : APInt(APInt::APINT_BITS_PER_WORD, val.getZExtValue() - 1);
  return computeRequiredBitwidth(val) == computeRequiredBitwidth(centVal) + 1;
}

Value ArithBoundOpt::getBranchToOptimize(handshake::ConditionalBranchOp condOp,
                                         arith::CmpIOp cmpOp,
                                         bool isDataLhs) const {
  Value falseRes = condOp.getFalseResult(), trueRes = condOp.getTrueResult();
  switch (cmpOp.getPredicate()) {
  case arith::CmpIPredicate::eq:
    // x == BOUND
    return trueRes;
  case arith::CmpIPredicate::ne:
    // x != BOUND
    return falseRes;
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
    // x < BOUND / BOUND < x
    // x <= BOUND / BOUND <= x
    return isDataLhs ? trueRes : falseRes;
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::uge:
    // x > BOUND / BOUND > x
    // x >= BOUND / BOUND >= x
    return isDataLhs ? falseRes : trueRes;
  default:
    return nullptr;
  }
}

unsigned ArithBoundOpt::getRealOptWidth(arith::CmpIOp cmpOp, unsigned optWidth,
                                        bool isDataLhs) const {
  switch (cmpOp.getPredicate()) {
  case arith::CmpIPredicate::ult:
    // x < BOUND / BOUND < x
  case arith::CmpIPredicate::uge:
    // x >= BOUND / BOUND >= x
    return isDataLhs ? optWidth - 1 : optWidth;
  case arith::CmpIPredicate::ule:
    // x <= BOUND / BOUND <= x
  case arith::CmpIPredicate::ugt:
    // x > BOUND / BOUND > x
    return isDataLhs ? optWidth : optWidth - 1;
  default:
    return optWidth;
  }
}

bool ArithBoundOpt::optBranchIfPossible(Value optBranch, unsigned optWidth,
                                        ExtType ext,
                                        PatternRewriter &rewriter) const {
  // Check whether we will get any benefit from the optimization
  unsigned dataWidth = getUsefulResultWidth(optBranch);
  if (optWidth >= dataWidth)
    return false;

  // Insert a truncation operation and an extension between the result branch
  // to optimize and its users, to let the rest of the rewrite patterns know
  // that some bits of the value can be safely discarded
  Value truncVal = modVal({optBranch, ext}, optWidth, rewriter);
  Value extVal = modVal({truncVal, ext}, dataWidth, rewriter);
  rewriter.replaceAllUsesExcept(optBranch, extVal, truncVal.getDefiningOp());
  return true;
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

namespace {

/// Driver for the bitwidth optimization pass. After applying a set of patterns
/// on the entire module that do not benefit from the iterative process,
/// iteratively and greedily applies a set of forward rewrite patterns followed
/// by a set of backward rewrite patterns until the IR converges.
struct HandshakeOptimizeBitwidthsPass
    : public dynamatic::impl::HandshakeOptimizeBitwidthsBase<
          HandshakeOptimizeBitwidthsPass> {

  HandshakeOptimizeBitwidthsPass(bool legacy) { this->legacy = legacy; }

  void runDynamaticPass() override {
    auto *ctx = &getContext();
    mlir::ModuleOp modOp = getOperation();

    // Create greedy config for all optimization passes
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    // Some optimizations do not need to be applied iteratively. We include
    // patterns to downgrade control merges and muxes with useless indices into
    // simpler merges to avoid having i0 types in the IR. We downgrade instead
    // of erasing these operations entirely (which would be semantically
    // correct) because it is not this pass's job to perform this kind of
    // optimization, which down-the-line passes may be sensitive to.
    RewritePatternSet patterns{ctx};
    patterns.add<HandshakeMuxSelect, DowngradeSingleInputMuxes,
                 HandshakeCMergeIndex, DowngradeSingleInputControlMerges>(ctx);
    if (!legacy)
      patterns
          .add<HandshakeMCAddress, HandshakeMemPortAddress<handshake::MCLoadOp>,
               HandshakeMemPortAddress<handshake::LSQLoadOp>,
               HandshakeMemPortAddress<handshake::MCStoreOp>,
               HandshakeMemPortAddress<handshake::LSQStoreOp>>(ctx);
    if (failed(
            applyPatternsAndFoldGreedily(modOp, std::move(patterns), config)))
      return signalPassFailure();

    for (auto funcOp : modOp.getOps<handshake::FuncOp>()) {
      bool fwChanged, bwChanged;
      SmallVector<Operation *> ops;

      // Runs the forward or backward pass on the function
      auto applyPass = [&](bool forward, bool &changed) {
        changed = false;
        RewritePatternSet patterns{ctx};
        if (forward)
          addForwardPatterns(patterns);
        else
          addBackwardPatterns(patterns);
        ops.clear();
        llvm::transform(funcOp.getOps(), std::back_inserter(ops),
                        [&](Operation &op) { return &op; });
        return applyOpPatternsAndFold(ops, std::move(patterns), config,
                                      &changed);
      };

      // Apply the forward and backward pass continuously until the IR converges
      do
        if (failed(applyPass(true, fwChanged)) ||
            failed(applyPass(false, bwChanged)))
          return signalPassFailure();
      while (fwChanged || bwChanged);
    }
  }

private:
  /// Adds to the pattern set all patterns on arith operations that have both a
  /// forward and backward version.
  void addArithPatterns(RewritePatternSet &patterns, bool forward);

  /// Adds to the pattern set all patterns on Handshake operations that have
  /// both a forward and backward version.
  void addHandshakeDataPatterns(RewritePatternSet &patterns, bool forward);

  /// Adds all forward patterns to the pattern set.
  void addForwardPatterns(RewritePatternSet &fwPatterns);

  /// Adds all backward patterns to the pattern set.
  void addBackwardPatterns(RewritePatternSet &bwPatterns);
};

void HandshakeOptimizeBitwidthsPass::addArithPatterns(
    RewritePatternSet &patterns, bool forward) {
  MLIRContext *ctx = patterns.getContext();

  patterns.add<ArithSingleType<arith::AddIOp>, ArithSingleType<arith::SubIOp>>(
      forward, addWidth, ctx);
  patterns.add<ArithSingleType<arith::MulIOp>>(true, mulWidth, ctx);
  patterns.add<ArithSingleType<arith::AndIOp>>(true, andWidth, ctx);
  patterns.add<ArithSingleType<arith::OrIOp>, ArithSingleType<arith::XOrIOp>>(
      true, orWidth, ctx);
  patterns.add<ArithShift<arith::ShLIOp>, ArithShift<arith::ShRSIOp>,
               ArithShift<arith::ShRUIOp>, ArithSelect>(forward, ctx);
  patterns.add<ArithExtToTruncOpt>(ctx);
}

void HandshakeOptimizeBitwidthsPass::addHandshakeDataPatterns(
    RewritePatternSet &patterns, bool forward) {
  MLIRContext *ctx = patterns.getContext();

  patterns.add<HandshakeOptDataNoCfg<handshake::ForkOp>,
               HandshakeOptDataNoCfg<handshake::LazyForkOp>,
               HandshakeOptDataNoCfg<handshake::MergeOp>,
               HandshakeOptDataNoCfg<handshake::BranchOp>>(forward, ctx);
  patterns.add<HandshakeOptData<handshake::ControlMergeOp, CMergeDataConfig>>(
      forward, ctx);
  patterns.add<HandshakeOptData<handshake::MuxOp, MuxDataConfig>>(forward, ctx);
  patterns
      .add<HandshakeOptData<handshake::ConditionalBranchOp, CBranchDataConfig>>(
          forward, ctx);
  patterns.add<HandshakeOptData<handshake::BufferOp, BufferDataConfig>>(forward,
                                                                        ctx);
}

void HandshakeOptimizeBitwidthsPass::addForwardPatterns(
    RewritePatternSet &fwPatterns) {
  MLIRContext *ctx = fwPatterns.getContext();

  // Handshake operations
  addHandshakeDataPatterns(fwPatterns, true);
  fwPatterns.add<ForwardCycleOptNoCfg<handshake::MergeOp>,
                 ForwardCycleOpt<handshake::MuxOp, MuxDataConfig>,
                 ForwardCycleOpt<handshake::ControlMergeOp, CMergeDataConfig>>(
      ctx);

  // arith operations
  addArithPatterns(fwPatterns, true);
  fwPatterns
      .add<ArithSingleType<arith::DivUIOp>, ArithSingleType<arith::DivSIOp>,
           ArithSingleType<arith::RemUIOp>, ArithSingleType<arith::RemSIOp>>(
          true, divWidth, ctx);
  fwPatterns.add<ArithCmpFW, ArithBoundOpt>(ctx);
}

void HandshakeOptimizeBitwidthsPass::addBackwardPatterns(
    RewritePatternSet &bwPatterns) {
  addHandshakeDataPatterns(bwPatterns, false);
  addArithPatterns(bwPatterns, false);
}

} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeOptimizeBitwidths(bool legacy) {
  return std::make_unique<HandshakeOptimizeBitwidthsPass>(legacy);
}
