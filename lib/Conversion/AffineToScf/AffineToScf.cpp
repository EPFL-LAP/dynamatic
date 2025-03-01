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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/DynamaticPass.h"
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
using namespace dynamatic;

namespace {

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  AffineLoadLowering(MemoryOpLowering &memOpLowering, MLIRContext *ctx)
      : OpRewritePattern(ctx, 2), memOpLowering(memOpLowering) {};

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
      : OpRewritePattern(ctx, 2), memOpLowering(memOpLowering) {};

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
class AffineToScfPass
    : public dynamatic::impl::AffineToScfBase<AffineToScfPass> {
  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();
    MemoryOpLowering memOpLowering(getAnalysis<NameAnalysis>());

    RewritePatternSet patterns(ctx);
    populateAffineToStdConversionPatterns(patterns);
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
