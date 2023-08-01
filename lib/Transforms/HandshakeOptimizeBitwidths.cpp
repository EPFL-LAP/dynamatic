//===- HandshakeOptimizeBitwidths.cpp - Optimize channel widths -*- C++ -*-===//
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
// truncing a value only to extend it again immediately may seem unnecessary,
// but it in fact allows the rest of the rewrite patterns to understand that
// a value fits on less bits than what the original value suggests. This is
// slightly convoluted but we are forced to do this like that since shift
// operations enforce that all their operands are of the same type. Ideally, we
// would have a Handshake version of shift operations that accept varrying
// bitwidths between its operands and result.
//
// Note on logical/arithmetic extension selection: the logic as is is incorrect
// and may lead to incorrect extensions in certain cases. For example, when
// moving an extension from an operation's operand to an operation's result we
// should keep track of the original extension type and use that same one after
// the operation. An orthogonal problem is that Polygeist doesn't seem to
// translate unsigned C/C++ variables to a ui<width> type in MLIR, instead using
// the signless i<width> type, which means we cannot know which values are
// logically unsigned and for which extensions should always be logical.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeOptimizeBitwidths.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <functional>

using namespace mlir;
using namespace circt;
using namespace dynamatic;

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
/// arithmetic extension operations. Returns the "minimal value", i.e., the
/// potentially different value that represents the same number as the
/// originally provided value but without all bits added by extension
/// operations. During the forward pass, the returned value gives an indication
/// of how many bits of the original value can be safely discarded.
static Value getMinimalValue(Value val) { // NOLINT(misc-no-recursion)
  Type type = val.getType();
  if (!isValidType(type))
    return val;

  Operation *defOp = val.getDefiningOp();
  if (!defOp || !isa<arith::ExtSIOp, arith::ExtUIOp>(defOp))
    return val;

  return getMinimalValue(defOp->getOperand(0));
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
/// extension operation is required, logicExt forces the use of a logical
/// extension operation, otherwise the signedness of the value is used to
/// determine whether the extension should be logical or arithmetic.
static Value modVal(Value val, unsigned targetWidth, bool logicExt,
                    PatternRewriter &rewriter) {
  Type type = val.getType();
  assert(isValidType(type) && "value must be valid type");

  unsigned width = type.getIntOrFloatBitWidth();
  Operation *newOp = nullptr;
  rewriter.setInsertionPointAfterValue(val);
  if (width < targetWidth) {
    if (logicExt || type.isUnsignedInteger())
      newOp = rewriter.create<arith::ExtUIOp>(
          val.getLoc(), rewriter.getIntegerType(targetWidth), val);
    else
      newOp = rewriter.create<arith::ExtSIOp>(
          val.getLoc(), rewriter.getIntegerType(targetWidth), val);
  } else if (width > targetWidth)
    newOp = rewriter.create<arith::TruncIOp>(
        val.getLoc(), rewriter.getIntegerType(targetWidth), val);
  if (newOp) {
    inheritBBFromValue(val, newOp);
    return newOp->getResult(0);
  }
  return val;
}

/// Replaces an operation with two operands and one result of the same integer
/// or floating type with an operation of the same type but whose operands and
/// result bitwidths have been modified to match the provided optimized
/// bitwidth. Extension and truncation operations are inserted as necessary to
/// satisfy the IR and bitwidth constraints. If an extension operation is
/// required, logicExt forces the use of a logical extension operation.
template <typename Op>
static void modArithOp(Op op, Value lhs, Value rhs, unsigned optWidth,
                       bool logicExt, PatternRewriter &rewriter) {
  Type resType = op->getResult(0).getType();
  assert(isValidType(resType) && "result must have valid type");
  unsigned resWidth = resType.getIntOrFloatBitWidth();

  // Create a new operation as well as appropriate bitwidth
  // modification operations to keep the IR valid
  Value newLhs = modVal(lhs, optWidth, logicExt, rewriter);
  Value newRhs = modVal(rhs, optWidth, logicExt, rewriter);
  rewriter.setInsertionPoint(op);
  auto newOp = rewriter.create<Op>(op.getLoc(), newLhs, newRhs);
  Value newRes = modVal(newOp->getResult(0), resWidth, logicExt, rewriter);
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

  /// Determines the list of operands that will be given to
  /// the builder of the optimized operation from the
  /// optimized data width and list of minimal data operands
  /// of the original operations. The vector given as last
  /// argument is filled with the new operands.
  virtual void getNewOperands(unsigned optWidth,
                              SmallVector<Value> &minDataOperands,
                              PatternRewriter &rewriter,
                              SmallVector<Value> &newOperands) {
    llvm::transform(
        minDataOperands, std::back_inserter(newOperands),
        [&](Value val) { return modVal(val, optWidth, false, rewriter); });
  }

  /// Determines the list of result types that will be given
  /// to the builder of the optimized operation. The
  /// dataType is the type shared by all data results. The
  /// vector given as last argument is filled with the new
  /// result types.
  virtual void getResultTypes(Type dataType, SmallVector<Type> &newResTypes) {
    for (size_t i = 0, numResults = op->getNumResults(); i < numResults; ++i)
      newResTypes.push_back(dataType);
  }

  /// Creates and returns the optimized operation from its
  /// result types and operands. The default builder for the
  /// operation must be available for the default
  /// implementation of this function.
  virtual Op createOp(SmallVector<Type> &newResTypes,
                      SmallVector<Value> &newOperands,
                      PatternRewriter &rewriter) {
    return rewriter.create<Op>(op.getLoc(), newResTypes, newOperands);
  }

  /// Determines the list of values that the original
  /// operation will be replaced with. These are the results
  /// of the newly inserted optimized operations whose
  /// bitwidth is modified to match those of the original
  /// operation. The width is the width that was shared by
  /// all data operands in the original operation. The
  /// vector given as last argument is filled with the new
  /// values.
  virtual void modResults(Op newOp, unsigned width, PatternRewriter &rewriter,
                          SmallVector<Value> &newResults) {
    llvm::transform(
        newOp->getResults(), std::back_inserter(newResults),
        [&](OpResult res) { return modVal(res, width, false, rewriter); });
  }

  /// Default destructor declared virtual because of virtual
  /// methods.
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

  void modResults(handshake::ControlMergeOp newOp, unsigned width,
                  PatternRewriter &rewriter,
                  SmallVector<Value> &newResults) override {
    newResults.push_back(modVal(newOp.getResult(), width, false, rewriter));
    newResults.push_back(newOp.getIndex());
  }
};

/// Special configuration for muxes required because of the select operand
/// which does not carry data.
class MuxDataConfig : public OptDataConfig<handshake::MuxOp> {
public:
  MuxDataConfig(handshake::MuxOp op) : OptDataConfig(op){};

  SmallVector<Value> getDataOperands() override { return op.getDataOperands(); }

  void getNewOperands(unsigned optWidth, SmallVector<Value> &minDataOperands,
                      PatternRewriter &rewriter,
                      SmallVector<Value> &newOperands) override {
    newOperands.push_back(op.getSelectOperand());
    llvm::transform(
        minDataOperands, std::back_inserter(newOperands),
        [&](Value val) { return modVal(val, optWidth, false, rewriter); });
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

  void getNewOperands(unsigned optWidth, SmallVector<Value> &minDataOperands,
                      PatternRewriter &rewriter,
                      SmallVector<Value> &newOperands) override {
    newOperands.push_back(op.getConditionOperand());
    newOperands.push_back(
        modVal(minDataOperands[0], optWidth, false, rewriter));
  }
};

/// Special configuration for buffers required because of the buffer type
/// attribute.
class BufferDataConfig : public OptDataConfig<handshake::BufferOp> {
public:
  BufferDataConfig(handshake::BufferOp op) : OptDataConfig(op){};

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
    llvm::transform(dataOperands, std::back_inserter(minDataOperands),
                    [&](Value val) { return getMinimalValue(val); });

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
    cfg.getNewOperands(optWidth, minDataOperands, rewriter, newOperands);
    cfg.getResultTypes(rewriter.getIntegerType(optWidth), newResTypes);
    rewriter.setInsertionPoint(op);
    Op newOp = cfg.createOp(newResTypes, newOperands, rewriter);
    inheritBB(op, newOp);
    cfg.modResults(newOp, dataWidth, rewriter, newResults);

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
    Value newSelect = modVal(selectOperand, optWidth, true, rewriter);
    rewriter.updateRootInPlace(
        muxOp, [&] { muxOp->replaceUsesOfWith(selectOperand, newSelect); });
    return success();
  }
};

/// Optimizes the bitwidth of control merges' index result so that it can just
/// support the number of data operands.This pattern can be applied as part of a
/// single greedy rewriting pass and doesn't need to be part of the
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
    Value modIndex = modVal(newOp.getIndex(), indexWidth, true, rewriter);
    rewriter.replaceOp(cmergeOp, ValueRange{newOp.getResult(), modIndex});
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
        getOptAddrWidth(mcOp.getMemref().getType().getDimSize(0));
    auto [_ctrlWidth, addrWidth, _dataWidth] = mcOp.getBitwidths();

    if (addrWidth == 0 || optWidth >= addrWidth)
      return failure();

    ValueRange inputs = mcOp.getInputs();
    size_t inputIdx = 0;

    // Optimizes the bitwidth of the address channel currently being pointed to
    // by inputIdx, and increment inputIdx before returning the optimized value
    auto getOptAddrInput = [&] {
      return modVal(getMinimalValue(inputs[inputIdx++]), optWidth, true,
                    rewriter);
    };

    // Iterate over memory controller inputs to create the new inputs and the
    // list of accesses
    SmallVector<Value> newInputs;
    SmallVector<SmallVector<AccessTypeEnum>> newAccesses;
    for (auto [blockIdx, accesses] : llvm::enumerate(mcOp.getAccesses())) {
      auto blockAccesses = cast<ArrayAttr>(accesses);
      if (mcOp.bbHasControl(blockIdx))
        newInputs.push_back(inputs[inputIdx++]); // Control channel

      newAccesses.push_back(SmallVector<AccessTypeEnum>());
      auto &newBlockAccesses = newAccesses[newAccesses.size() - 1];
      for (auto access : blockAccesses) {
        auto accessType =
            cast<handshake::AccessTypeEnumAttr>(access).getValue();
        newBlockAccesses.push_back(accessType);
        if (accessType == AccessTypeEnum::Load)
          newInputs.push_back(getOptAddrInput());
        else {
          newInputs.push_back(getOptAddrInput());
          newInputs.push_back(inputs[inputIdx++]); // Data channel
        }
      }
    }

    // Replace the existing memory controller with the optimized one
    rewriter.setInsertionPoint(mcOp);
    auto newOp = rewriter.create<handshake::MemoryControllerOp>(
        mcOp.getLoc(), mcOp.getMemref(), newInputs, newAccesses, mcOp.getId());
    inheritBB(mcOp, newOp);
    rewriter.replaceOp(mcOp, newOp.getResults());
    return success();
  }
};

/// Optimizes the bitwidth of memory ports's address-carrying channels so that
/// they can just support indexing into the memory region these ports ultimately
/// talk to. The first template parameter is meant to be either
/// handshake::DynamaticLoadOp or handshake::DynamaticStoreOp. This pattern can
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
        modVal(getMinimalValue(memOp.getAddress()), optWidth, true, rewriter);
    rewriter.setInsertionPoint(memOp);
    auto newOp = rewriter.create<Op>(memOp.getLoc(), newAddr.getType(),
                                     memOp.getDataResult().getType(), newAddr,
                                     memOp.getData());
    Value newAddrRes =
        modVal(newOp.getAddressResult(), addrWidth, true, rewriter);
    inheritBB(memOp, newOp);
    rewriter.replaceOp(memOp, {newAddrRes, newOp.getDataResult()});
    return success();
  }
};

/// Moves any extension operation feeding into a return operation past the
/// latter to optimize the bitwidth occupied by the return operation itself.
/// This is meant to be part of the forward pass.
struct HandshakeReturnFW : public OpRewritePattern<handshake::ReturnOp> {
  using OpRewritePattern<handshake::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ReturnOp retOp,
                                PatternRewriter &rewriter) const override {

    // Try to move potential extension operations after the return
    SmallVector<Value> newOperands;
    bool changed = false;
    auto tryToMinimize = [&](Value opr) {
      Type oprType = opr.getType();
      if (!isValidType(oprType))
        return opr;
      Value minVal = getMinimalValue(opr);
      changed |= minVal.getType().getIntOrFloatBitWidth() <
                 oprType.getIntOrFloatBitWidth();
      return minVal;
    };
    llvm::transform(retOp->getOperands(), std::back_inserter(newOperands),
                    tryToMinimize);

    // Check whether the transformation would change anything
    if (!changed)
      return failure();

    // Insert an optimized return operation that moves eventual value extensions
    // after itself
    rewriter.setInsertionPoint(retOp);
    auto newOp =
        rewriter.create<handshake::ReturnOp>(retOp->getLoc(), newOperands);
    SmallVector<Value> newResults;
    for (auto [newRes, ogResType] :
         llvm::zip_equal(newOp->getResults(), retOp->getResultTypes())) {
      if (!isValidType(ogResType))
        newResults.push_back(newRes);
      else
        newResults.push_back(
            modVal(newRes, ogResType.getIntOrFloatBitWidth(), false, rewriter));
    }
    inheritBB(retOp, newOp);
    rewriter.replaceOp(retOp, newResults);
    return success();
  }
};

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
    Value minLhs = getMinimalValue(op->getOperand(0));
    Value minRhs = getMinimalValue(op->getOperand(1));
    unsigned optWidth;
    if (forward)
      optWidth = fTransfer(minLhs.getType().getIntOrFloatBitWidth(),
                           minRhs.getType().getIntOrFloatBitWidth());
    else
      optWidth = getUsefulResultWidth(op->getResult(0));
    unsigned resWidth = resType.getIntOrFloatBitWidth();
    if (optWidth >= resWidth)
      return failure();

    // For bitwise logical operations, extension must also be logical
    bool logicExt =
        isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>((Operation *)op);
    modArithOp(op, minLhs, minRhs, optWidth, logicExt, rewriter);
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
    Value minLhs = getMinimalValue(selectOp.getTrueValue());
    Value minRhs = getMinimalValue(selectOp.getFalseValue());
    unsigned optWidth;
    if (forward)
      optWidth = std::max(minLhs.getType().getIntOrFloatBitWidth(),
                          minRhs.getType().getIntOrFloatBitWidth());
    else
      optWidth = getUsefulResultWidth(selectOp.getResult());
    unsigned resWidth = resType.getIntOrFloatBitWidth();
    if (optWidth >= resWidth)
      return failure();

    // Create a new operation as well as appropriate bitwidth modification
    // operations to keep the IR valid
    Value newLhs = modVal(minLhs, optWidth, false, rewriter);
    Value newRhs = modVal(minRhs, optWidth, false, rewriter);
    rewriter.setInsertionPoint(selectOp);
    auto newOp = rewriter.create<arith::SelectOp>(
        selectOp.getLoc(), selectOp.getCondition(), newLhs, newRhs);
    Value newRes = modVal(newOp->getResult(0), resWidth, false, rewriter);
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
    Value minToShift = getMinimalValue(toShift);
    Value minShiftBy = getMinimalValue(shiftBy);
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

    // For logical shifts, extension must also be logical
    bool logicExt = isa<arith::ShLIOp>(op) || isa<arith::ShRUIOp>(op);

    if (forward) {
      // Create a new operation as well as appropriate bitwidth modification
      // operations to keep the IR valid
      Value newLhs = modVal(minToShift, optWidth, logicExt, rewriter);
      Value newRhs = modVal(minShiftBy, optWidth, logicExt, rewriter);
      rewriter.setInsertionPoint(op);
      auto newOp = rewriter.create<Op>(op.getLoc(), newLhs, newRhs);
      Value newRes = newOp->getResult(0);
      if (isRightShift)
        // In the case of a right shift, we first truncate the result of the
        // newly inserted shift operation to discard high-significance bits that
        // we know are 0s, then extend the result back to satisfy the users of
        // the original operation's result
        newRes = modVal(newRes, optWidth - cstVal, logicExt, rewriter);
      Value modRes = modVal(newRes, resWidth, logicExt, rewriter);
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
            modVal(minToShift, requiredToShiftWidth, logicExt, rewriter);
      }
      modArithOp(op, modToShift, minShiftBy, optWidth, logicExt, rewriter);
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
    Value minLhs = getMinimalValue(lhs);
    Value minRhs = getMinimalValue(rhs);
    unsigned optWidth = std::max(minLhs.getType().getIntOrFloatBitWidth(),
                                 minRhs.getType().getIntOrFloatBitWidth());
    unsigned actualWidth = lhs.getType().getIntOrFloatBitWidth();
    if (optWidth >= actualWidth)
      return failure();

    // Create a new operation as well as appropriate bitwidth modification
    // operations to keep the IR valid
    Value newLhs = modVal(minLhs, optWidth, false, rewriter);
    Value newRhs = modVal(minRhs, optWidth, false, rewriter);
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

} // namespace

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

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::ModuleOp modOp = getOperation();

    // Create greedy config for all optimization passes
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    // Some optimizations do not need to be applied iteratively.
    RewritePatternSet patterns{ctx};
    patterns.add<HandshakeMuxSelect, HandshakeCMergeIndex, HandshakeMCAddress,
                 HandshakeMemPortAddress<handshake::DynamaticLoadOp>,
                 HandshakeMemPortAddress<handshake::DynamaticStoreOp>>(ctx);
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
  fwPatterns.add<HandshakeReturnFW>(ctx);

  // arith operations
  addArithPatterns(fwPatterns, true);
  fwPatterns
      .add<ArithSingleType<arith::DivUIOp>, ArithSingleType<arith::DivSIOp>,
           ArithSingleType<arith::RemUIOp>, ArithSingleType<arith::RemSIOp>>(
          true, divWidth, ctx);
  fwPatterns.add<ArithCmpFW>(ctx);
}

void HandshakeOptimizeBitwidthsPass::addBackwardPatterns(
    RewritePatternSet &bwPatterns) {
  addHandshakeDataPatterns(bwPatterns, false);
  addArithPatterns(bwPatterns, false);
}

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakeOptimizeBitwidths() {
  return std::make_unique<HandshakeOptimizeBitwidthsPass>();
}
