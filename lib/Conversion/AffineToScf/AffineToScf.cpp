//===- AffineToScf.cpp - Convert Affine to SCF/standard dialect -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a slight modification of the AffineToStandard conversion pass
// in MLIR that preserves the memory attributes of lowered AffineLoadOp's and
// AffineStoreOp's.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/AffineToScf.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Conversion/PassDetails.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/Handshake.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::affine;
using namespace circt;
using namespace circt::handshake;
using namespace dynamatic;

/// Given a range of values, emit the code that reduces them with "min" or "max"
/// depending on the provided comparison predicate.  The predicate defines which
/// comparison to perform, "lt" for "min", "gt" for "max" and is used for the
/// `cmpi` operation followed by the `select` operation:
///
///   %cond   = arith.cmpi "predicate" %v0, %v1
///   %result = select %cond, %v0, %v1
///
/// Multiple values are scanned in a linear sequence.  This creates a data
/// dependences that wouldn't exist in a tree reduction, but is easier to
/// recognize as a reduction by the subsequent passes.
static Value buildMinMaxReductionSeq(Location loc,
                                     arith::CmpIPredicate predicate,
                                     ValueRange values, OpBuilder &builder) {
  assert(!values.empty() && "empty min/max chain");

  auto valueIt = values.begin();
  Value value = *valueIt++;
  for (; valueIt != values.end(); ++valueIt) {
    auto cmpOp = builder.create<arith::CmpIOp>(loc, predicate, value, *valueIt);
    value = builder.create<arith::SelectOp>(loc, cmpOp.getResult(), value,
                                            *valueIt);
  }

  return value;
}

/// Emit instructions that correspond to computing the maximum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
static Value lowerAffineMapMax(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::sgt, *values,
                                   builder);
  return nullptr;
}

/// Emit instructions that correspond to computing the minimum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
static Value lowerAffineMapMin(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::slt, *values,
                                   builder);
  return nullptr;
}

namespace {

/// Affine yields ops are removed.
class AffineYieldOpLowering : public OpRewritePattern<AffineYieldOp> {
public:
  using OpRewritePattern<AffineYieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineYieldOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<scf::ParallelOp>(op->getParentOp())) {
      // scf.parallel does not yield any values via its terminator scf.yield but
      // models reductions differently using additional ops in its region.
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op);
      return success();
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands());
    return success();
  }
};

class AffineForLowering : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lowerBound = lowerAffineLowerBound(op, rewriter);
    Value upperBound = lowerAffineUpperBound(op, rewriter);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, op.getStep());
    auto scfForOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound,
                                                step, op.getIterOperands());
    rewriter.eraseBlock(scfForOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), scfForOp.getRegion(),
                                scfForOp.getRegion().end());
    rewriter.replaceOp(op, scfForOp.getResults());
    return success();
  }
};

/// Convert an `affine.parallel` (loop nest) operation into a `scf.parallel`
/// operation.
class AffineParallelLowering : public OpRewritePattern<AffineParallelOp> {
public:
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 8> steps;
    SmallVector<Value, 8> upperBoundTuple;
    SmallVector<Value, 8> lowerBoundTuple;
    SmallVector<Value, 8> identityVals;
    // Emit IR computing the lower and upper bound by expanding the map
    // expression.
    lowerBoundTuple.reserve(op.getNumDims());
    upperBoundTuple.reserve(op.getNumDims());
    for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
      Value lower = lowerAffineMapMax(rewriter, loc, op.getLowerBoundMap(i),
                                      op.getLowerBoundsOperands());
      if (!lower)
        return rewriter.notifyMatchFailure(op, "couldn't convert lower bounds");
      lowerBoundTuple.push_back(lower);

      Value upper = lowerAffineMapMin(rewriter, loc, op.getUpperBoundMap(i),
                                      op.getUpperBoundsOperands());
      if (!upper)
        return rewriter.notifyMatchFailure(op, "couldn't convert upper bounds");
      upperBoundTuple.push_back(upper);
    }
    steps.reserve(op.getSteps().size());
    for (int64_t step : op.getSteps())
      steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, step));

    // Get the terminator op.
    Operation *affineParOpTerminator = op.getBody()->getTerminator();
    scf::ParallelOp parOp;
    if (op.getResults().empty()) {
      // Case with no reduction operations/return values.
      parOp = rewriter.create<scf::ParallelOp>(loc, lowerBoundTuple,
                                               upperBoundTuple, steps,
                                               /*bodyBuilderFn=*/nullptr);
      rewriter.eraseBlock(parOp.getBody());
      rewriter.inlineRegionBefore(op.getRegion(), parOp.getRegion(),
                                  parOp.getRegion().end());
      rewriter.replaceOp(op, parOp.getResults());
      return success();
    }
    // Case with affine.parallel with reduction operations/return values.
    // scf.parallel handles the reduction operation differently unlike
    // affine.parallel.
    ArrayRef<Attribute> reductions = op.getReductions().getValue();
    for (auto pair : llvm::zip(reductions, op.getResultTypes())) {
      // For each of the reduction operations get the identity values for
      // initialization of the result values.
      Attribute reduction = std::get<0>(pair);
      Type resultType = std::get<1>(pair);
      std::optional<arith::AtomicRMWKind> reductionOp =
          arith::symbolizeAtomicRMWKind(
              static_cast<uint64_t>(reduction.cast<IntegerAttr>().getInt()));
      assert(reductionOp && "Reduction operation cannot be of None Type");
      arith::AtomicRMWKind reductionOpValue = *reductionOp;
      identityVals.push_back(
          arith::getIdentityValue(reductionOpValue, resultType, rewriter, loc));
    }
    parOp = rewriter.create<scf::ParallelOp>(
        loc, lowerBoundTuple, upperBoundTuple, steps, identityVals,
        /*bodyBuilderFn=*/nullptr);

    //  Copy the body of the affine.parallel op.
    rewriter.eraseBlock(parOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), parOp.getRegion(),
                                parOp.getRegion().end());
    assert(reductions.size() == affineParOpTerminator->getNumOperands() &&
           "Unequal number of reductions and operands.");
    for (unsigned i = 0, end = reductions.size(); i < end; i++) {
      // For each of the reduction operations get the respective mlir::Value.
      std::optional<arith::AtomicRMWKind> reductionOp =
          arith::symbolizeAtomicRMWKind(
              reductions[i].cast<IntegerAttr>().getInt());
      assert(reductionOp && "Reduction Operation cannot be of None Type");
      arith::AtomicRMWKind reductionOpValue = *reductionOp;
      rewriter.setInsertionPoint(&parOp.getBody()->back());
      auto reduceOp = rewriter.create<scf::ReduceOp>(
          loc, affineParOpTerminator->getOperand(i));
      rewriter.setInsertionPointToEnd(&reduceOp.getReductionOperator().front());
      Value reductionResult = arith::getReductionOp(
          reductionOpValue, rewriter, loc,
          reduceOp.getReductionOperator().front().getArgument(0),
          reduceOp.getReductionOperator().front().getArgument(1));
      rewriter.create<scf::ReduceReturnOp>(loc, reductionResult);
    }
    rewriter.replaceOp(op, parOp.getResults());
    return success();
  }
};

class AffineIfLowering : public OpRewritePattern<AffineIfOp> {
public:
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineIfOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Now we just have to handle the condition logic.
    auto integerSet = op.getIntegerSet();
    Value zeroConstant = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value, 8> operands(op.getOperands());
    auto operandsRef = llvm::ArrayRef(operands);

    // Calculate cond as a conjunction without short-circuiting.
    Value cond = nullptr;
    for (unsigned i = 0, e = integerSet.getNumConstraints(); i < e; ++i) {
      AffineExpr constraintExpr = integerSet.getConstraint(i);
      bool isEquality = integerSet.isEq(i);

      // Build and apply an affine expression
      auto numDims = integerSet.getNumDims();
      Value affResult = expandAffineExpr(rewriter, loc, constraintExpr,
                                         operandsRef.take_front(numDims),
                                         operandsRef.drop_front(numDims));
      if (!affResult)
        return failure();
      auto pred =
          isEquality ? arith::CmpIPredicate::eq : arith::CmpIPredicate::sge;
      Value cmpVal =
          rewriter.create<arith::CmpIOp>(loc, pred, affResult, zeroConstant);
      cond = cond
                 ? rewriter.create<arith::AndIOp>(loc, cond, cmpVal).getResult()
                 : cmpVal;
    }
    cond = cond ? cond
                : rewriter.create<arith::ConstantIntOp>(loc, /*value=*/1,
                                                        /*width=*/1);

    bool hasElseRegion = !op.getElseRegion().empty();
    auto ifOp = rewriter.create<scf::IfOp>(loc, op.getResultTypes(), cond,
                                           hasElseRegion);
    rewriter.inlineRegionBefore(op.getThenRegion(),
                                &ifOp.getThenRegion().back());
    rewriter.eraseBlock(&ifOp.getThenRegion().back());
    if (hasElseRegion) {
      rewriter.inlineRegionBefore(op.getElseRegion(),
                                  &ifOp.getElseRegion().back());
      rewriter.eraseBlock(&ifOp.getElseRegion().back());
    }

    // Replace the Affine IfOp finally.
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

/// Convert an "affine.apply" operation into a sequence of arithmetic
/// operations using the StandardOps dialect.
class AffineApplyLowering : public OpRewritePattern<AffineApplyOp> {
public:
  using OpRewritePattern<AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                        llvm::to_vector<8>(op.getOperands()));
    if (!maybeExpandedMap)
      return failure();
    rewriter.replaceOp(op, *maybeExpandedMap);
    return success();
  }
};

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  AffineLoadLowering(MemoryOpLowering &memOpLowering, MLIRContext *ctx)
      : OpRewritePattern(ctx), memOpLowering(memOpLowering){};

  LogicalResult matchAndRewrite(AffineLoadOp affineLoadOp,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(affineLoadOp.getMapOperands());
    auto resultOperands = expandAffineMap(rewriter, affineLoadOp.getLoc(),
                                          affineLoadOp.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Replace with simple load operation and keep correspondance between the
    // two operations
    memref::LoadOp loadOp = rewriter.replaceOpWithNewOp<memref::LoadOp>(
        affineLoadOp, affineLoadOp.getMemRef(), *resultOperands);
    memOpLowering.recordReplacement(affineLoadOp, loadOp);
    return success();
  }

private:
  /// Used to record the operation replacement (from an affine-level load to a
  /// memref-level load).
  MemoryOpLowering &memOpLowering;
};

/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which replaces
/// the original 'affine.store').
class AffineStoreLowering : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  AffineStoreLowering(MemoryOpLowering &memOpLowering, MLIRContext *ctx)
      : OpRewritePattern(ctx), memOpLowering(memOpLowering){};

  LogicalResult matchAndRewrite(AffineStoreOp affineStoreOp,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(affineStoreOp.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, affineStoreOp.getLoc(),
                        affineStoreOp.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Replace with simple store operation and keep correspondance between the
    // two operations
    memref::StoreOp storeOp = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        affineStoreOp, affineStoreOp.getValueToStore(),
        affineStoreOp.getMemRef(), *maybeExpandedMap);
    memOpLowering.recordReplacement(affineStoreOp, storeOp);
    return success();
  }

private:
  /// Used to record the operation replacement (from an affine-level store to a
  /// memref-level store).
  MemoryOpLowering &memOpLowering;
};

} // namespace

namespace {
class AffineToScfPass : public AffineToScfBase<AffineToScfPass> {
  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();
    MemoryOpLowering memOpLowering(getAnalysis<NameAnalysis>());

    RewritePatternSet patterns(ctx);
    patterns.add<AffineApplyLowering, AffineParallelLowering, AffineForLowering,
                 AffineIfLowering, AffineYieldOpLowering>(ctx);
    patterns.add<AffineLoadLowering, AffineStoreLowering>(memOpLowering, ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                           scf::SCFDialect, VectorDialect>();
    if (failed(applyPartialConversion(modOp, target, std::move(patterns))))
      signalPassFailure();

    // Change the name of destination memory acceses in all stored memory
    // dependencies to reflect the new access names
    memOpLowering.renameDependencies(modOp);
  }
};
} // namespace

/// Lowers If and For operations within a function into their lower level CFG
/// equivalent blocks.
std::unique_ptr<dynamatic::DynamaticPass> dynamatic::createAffineToScfPass() {
  return std::make_unique<AffineToScfPass>();
}
