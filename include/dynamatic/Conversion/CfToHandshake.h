//===- CfToHandhsake.h - Convert func/cf to handhsake dialect ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --lower-cf-to-handshake conversion pass along with a
// helper class for performing the lowering.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_CF_TO_HANDSHAKE_H
#define DYNAMATIC_CONVERSION_CF_TO_HANDSHAKE_H

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/FuncMaximizeSSA.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace dynamatic {

/// Converts cf-level types to those expected by handshake-level IR. Right now
/// these are the sames but this will change as soon as the new type system is
/// integrated.
class CfToHandshakeTypeConverter : public TypeConverter {
public:
  CfToHandshakeTypeConverter();
};

/// Converts a func-level function into a handshake-level function. The function
/// signature gets an extra control-only argument to represent the starting
/// point of the control network. If the function did not return any result, a
/// control-only result is added to signal function completion. All of the
/// pattern's intermediate conversion steps are virtual, allowing other passes
/// to reuse part of the conversion while defining custom behavior.
class LowerFuncToHandshake : public DynOpConversionPattern<mlir::func::FuncOp> {
public:
  // Explicit constructors from base class, so that extended classes can benefit
  // from them. Without an explicit definition, the extended class cannot use
  // the constructor of a non-base class.
  LowerFuncToHandshake(NameAnalysis &namer, MLIRContext *ctx,
                       mlir::PatternBenefit benefit = 1)
      : DynOpConversionPattern<mlir::func::FuncOp>(namer, ctx, benefit){};

  LowerFuncToHandshake(NameAnalysis &namer, const TypeConverter &typeConverter,
                       MLIRContext *ctx, mlir::PatternBenefit benefit = 1)
      : DynOpConversionPattern<mlir::func::FuncOp>(namer, typeConverter, ctx,
                                                   benefit){};

  LogicalResult
  matchAndRewrite(mlir::func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

  /// Groups memory operations by interface and group for a given memory region.
  struct MemAccesses {
    /// Memory operations for a simple memory controller, grouped by
    /// originating basic block.
    llvm::MapVector<Block *, SmallVector<handshake::MemPortOpInterface>>
        mcPorts;
    /// Memory operations for an LSQ, grouped by belonging LSQ group.
    llvm::MapVector<unsigned, SmallVector<handshake::MemPortOpInterface>>
        lsqPorts;
    /// Function argument corresponding to the memory start signal for that
    /// interface.
    BlockArgument memStart;

    MemAccesses(BlockArgument memStart);
  };

  /// Stores a mapping between memory regions (identified by the function
  /// argument they correspond to) and the set of memory operations referencing
  /// them.
  using MemInterfacesInfo = llvm::MapVector<Value, MemAccesses>;

  /// Creates a Handshake-level equivalent to the matched func-level function,
  /// returning it on success. A `nullptr` return value indicates a failure.
  virtual FailureOr<handshake::FuncOp>
  lowerSignature(mlir::func::FuncOp funcOp,
                 ConversionPatternRewriter &rewriter) const;

  /// Produces the list of named attributes that the Handshake function's
  /// builder will be passed during signature lowering.
  virtual SmallVector<mlir::NamedAttribute>
  deriveNewAttributes(mlir::func::FuncOp funcOp) const;

  /// Adds merge-like operations after all block arguments within the region,
  /// then removes all block arguments and corresponding branch operands.
  virtual void
  addMergeOps(handshake::FuncOp funcOp, ConversionPatternRewriter &rewriter,
              DenseMap<BlockArgument, OpResult> &blockArgReplacements) const;

  /// Adds handshake-level branch-like operations before all cf-level
  /// branch-like terminators within the region. This needs to happen after
  /// merge-insertion because it also replaces data operands of merge-like
  /// operations with the result value(s) of inserted branch-like operations.
  virtual void addBranchOps(handshake::FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const;

  /// Identifies all memory interfaces and their associated operations in the
  /// function, converts all load/store-like operations by their handshake
  /// counterparts, and fills `memInfo` with information about which operations
  /// use which interface. The backedge builder is used to create temporary
  /// values for the data input to converted load ports. A flag for FTD is also
  /// introduced to tweak the rewriting process.
  virtual LogicalResult
  convertMemoryOps(handshake::FuncOp funcOp,
                   ConversionPatternRewriter &rewriter,
                   const DenseMap<Value, unsigned> &memrefIndices,
                   BackedgeBuilder &edgeBuilder, MemInterfacesInfo &memInfo,
                   bool isFtd = false) const;

  /// Verifies that LSQ groups derived from input IR annotations make sense
  /// (check for linear dominance property within each group and cross-group
  /// control signal compatibility). Then, instantiates all memory interfaces
  /// and connects them to their respective load/store operations. For each
  /// memory region:
  /// - A single `handshake::MemoryControllerOp` will be instantiated if all of
  /// its accesses indicate that they should connect to an MC.
  /// - A single `handshake::LSQOp` will be instantiated if none of
  /// its accesses indicate that they should connect to an LSQ.
  /// - Both a `handhsake::MemoryControllerOp` and `handhsake::LSQOp` will be
  /// instantiated if some but not all of its accesses indicate that they should
  /// connect to an LSQ.
  virtual LogicalResult
  verifyAndCreateMemInterfaces(handshake::FuncOp funcOp,
                               ConversionPatternRewriter &rewriter,
                               MemInterfacesInfo &memInfo) const;

  /// Sets an integer "bb" attribute on each operation to identify the basic
  /// block from which the operation originates in the std-level IR.
  virtual void idBasicBlocks(handshake::FuncOp funcOp,
                             ConversionPatternRewriter &rewriter) const;

  /// Creates the region's return network by sequentially moving all blocks'
  /// operations to the entry block, deleting all block terminators and
  /// non-entry blocks, merging the results of all return statements, and
  /// creating the region's end operation.
  virtual LogicalResult flattenAndTerminate(
      handshake::FuncOp funcOp, ConversionPatternRewriter &rewriter,
      const DenseMap<BlockArgument, OpResult> &blockArgReplacements) const;

  /// Returns the value representing the block's control signal.
  virtual Value getBlockControl(Block *block) const;

protected:
  /// Groups information to "rewire the IR" around a particular merge-like
  /// operation.
  struct MergeOpInfo {
    /// The original block argument that the merge-like operation "replaces".
    BlockArgument blockArg;
    /// The merge-like operation under consideration.
    handshake::MergeLikeOpInterface op = nullptr;
    /// Each vector entry represent a data operand to the merge as
    /// 1. the backedge that was inserted to temporarily represent it,
    /// 2. the predecessor block from which the data should come,
    /// 3. a boolean indicating whether this is the first operand to come from
    ///    the associated block
    SmallVector<std::tuple<Backedge, Block *, bool>, 2> operands;
    /// An optional index operand that needs to be resolved for mux-like
    /// operations.
    std::optional<Backedge> indexEdge{};

    /// Constructs from the block argument the future merge-like operation will
    /// replace.
    MergeOpInfo(BlockArgument blockArg) : blockArg(blockArg) {}
  };

  /// Inserts a merge-like operation in the IR for the provided block argument.
  /// Stores information about the merge-like operation in the last argument.
  void insertMerge(BlockArgument blockArg, ConversionPatternRewriter &rewriter,
                   BackedgeBuilder &edgeBuilder,
                   LowerFuncToHandshake::MergeOpInfo &iMerge) const;

  /// Checks whether the blocks in `opsPerBlock`'s keys exhibit a "linear
  /// dominance relationship" i.e., whether the execution of the "most dominant"
  /// block necessarily triggers the execution of all others in a deterministic
  /// order. This verification happens in linear time thanks to the cached
  /// dominator/dominated relationships in `dominations`. On success, stores the
  /// blocks' execution order in `dominanceOrder` ("most dominant" block first,
  /// then "second most dominant", etc.). Fails when the blocks do not exhibit
  /// that property.
  LogicalResult computeLinearDominance(
      DenseMap<Block *, DenseSet<Block *>> &dominations,
      llvm::MapVector<Block *, SmallVector<handshake::MemPortOpInterface>>
          &opsPerBlock,
      SmallVector<Block *> &dominanceOrder) const;
};

/// Strategy to use when putting the matched func-level function into maximal
/// SSA form.
class FuncSSAStrategy : public dynamatic::SSAMaximizationStrategy {
  /// Filters out block arguments of type MemRefType
  bool maximizeArgument(BlockArgument arg) override;
};

template <typename SrcOp, typename DstOp>
struct OneToOneConversion : public OpConversionPattern<SrcOp> {
public:
  using OpAdaptor = typename SrcOp::Adaptor;

  OneToOneConversion(NameAnalysis &namer, const TypeConverter &typeConverter,
                     MLIRContext *ctx)
      : OpConversionPattern<SrcOp>(typeConverter, ctx), namer(namer) {}

  LogicalResult
  matchAndRewrite(SrcOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

protected:
  /// Reference to the running pass's naming analysis.
  NameAnalysis &namer;
};

template <typename CastOp, typename ExtOp>
struct ConvertIndexCast : public OpConversionPattern<CastOp> {
public:
  using OpAdaptor = typename CastOp::Adaptor;

  ConvertIndexCast(NameAnalysis &namer, const TypeConverter &typeConverter,
                   MLIRContext *ctx)
      : OpConversionPattern<CastOp>(typeConverter, ctx), namer(namer) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

protected:
  /// Reference to the running pass's naming analysis.
  NameAnalysis &namer;
};

/// Converts each `func::CallOp` operation to an equivalent
/// `handshake::InstanceOp` operation.
struct ConvertCalls : public DynOpConversionPattern<mlir::func::CallOp> {
public:
  using DynOpConversionPattern<mlir::func::CallOp>::DynOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::func::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Convers arith-level constants to handshake-level constants. Constants are
/// triggered by a source if their successor is not a branch/return or memory
/// operation. Otherwise they are triggered by the control-only network.
struct ConvertConstants
    : public DynOpConversionPattern<mlir::arith::ConstantOp> {
public:
  using DynOpConversionPattern<mlir::arith::ConstantOp>::DynOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::arith::ConstantOp cstOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts undefined operations (LLVM::UndefOp) with a default "0" constant
/// triggered by the control merge of the block associated to the matched
/// operation.
struct ConvertUndefinedValues
    : public DynOpConversionPattern<mlir::LLVM::UndefOp> {
public:
  using DynOpConversionPattern<mlir::LLVM::UndefOp>::DynOpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::LLVM::UndefOp undefOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

#define GEN_PASS_DECL_CFTOHANDSHAKE
#define GEN_PASS_DEF_CFTOHANDSHAKE
#include "dynamatic/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createCfToHandshake();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_CF_TO_HANDSHAKE_H
