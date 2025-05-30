//===- FlattenMemRefROwMajor.cpp - MemRef flattening pass -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the MemRef flattening pass. It is closely modeled
// on the MemRef flattening pass from CIRCT bus uses row-major indexing to
// convert multidimensional load and store operations.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/FlattenMemRefRowMajor.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace dynamatic;

static inline bool isUniDimensional(MemRefType memref) {
  return memref.getShape().size() == 1;
}

/// Flatten indices in row-major style, making adjacent indices in the last
/// memref dimension be adjacent indices in the flattened memref.
static Value flattenIndices(ConversionPatternRewriter &rewriter, Location loc,
                            ValueRange indices, MemRefType memrefType) {
  assert(memrefType.hasStaticShape() && "expected statically shaped memref");
  auto numIndices = indices.size();

  if (numIndices == 0) {
    // Singleton memref (e.g. memref<i32>) - return 0
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0))
        .getResult();
  }

  if (numIndices == 1)
    // Memref is already unidimensional
    return indices.front();

  // An example of flattening a multi-dimension array in a row-major order.
  // Given an array: my_array[A][B][C][D];
  // we access it as my_array[i][j][k][l];
  // The flattened index is computed as:
  //
  // (B * C * D) * i + (C * D) * j + (D) * k + l

  // This variable will be the final value that drives the address input of
  // load/store operations.
  Value accumulatedArrayIndex = indices.back();

  // This holds the accumulated product of the previous dimensions.
  int64_t dimProduct = 1;

  // Iterate over indices to compute the final unidimensional index. We iterate
  // from the second last index to the first index
  for (int memIdx = numIndices - 2 /* i.e., the second last index */;
       memIdx >= 0; memIdx--) {
    Value partialIdx = indices[memIdx];
    // This multiplies the current index by the product of all the previous
    // dimensions. For example, in the example above, it would multiply
    // partialIdx by (B * C * D) for index "i", (C * D) for index "j", and (D)
    // for index "k".
    dimProduct *= memrefType.getShape()[memIdx + 1];

    // Multiply product by the current index operand
    if (llvm::isPowerOf2_64(dimProduct)) {
      auto constant =
          rewriter
              .create<arith::ConstantOp>(
                  loc, rewriter.getIndexAttr(llvm::Log2_64(dimProduct)))
              .getResult();
      partialIdx =
          rewriter.create<arith::ShLIOp>(loc, partialIdx, constant).getResult();
    } else {
      auto constant =
          rewriter
              .create<arith::ConstantOp>(loc, rewriter.getIndexAttr(dimProduct))
              .getResult();
      partialIdx =
          rewriter.create<arith::MulIOp>(loc, partialIdx, constant).getResult();
    }

    // Sum up with the prior lower dimension accessors
    auto sumOp =
        rewriter.create<arith::AddIOp>(loc, accumulatedArrayIndex, partialIdx);
    accumulatedArrayIndex = sumOp.getResult();
  }
  return accumulatedArrayIndex;
}

static bool hasMultiDimMemRef(ValueRange values) {
  return llvm::any_of(values, [](Value v) {
    auto memref = v.getType().dyn_cast<MemRefType>();
    if (!memref)
      return false;
    return !isUniDimensional(memref);
  });
}

namespace {

struct LoadOpConversion : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LoadOpConversion(MemoryOpLowering &memOpLowering, TypeConverter &converter,
                   MLIRContext *ctx)
      : OpConversionPattern(converter, ctx), memOpLowering(memOpLowering){};

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = loadOp.getMemRefType();
    if (isUniDimensional(type) || !type.hasStaticShape() ||
        /*Already converted?*/ loadOp.getIndices().size() == 1)
      return failure();
    Value finalIdx =
        flattenIndices(rewriter, loadOp.getLoc(), adaptor.getIndices(),
                       loadOp.getMemRefType());
    memref::LoadOp flatLoadOp = rewriter.replaceOpWithNewOp<memref::LoadOp>(
        loadOp, adaptor.getMemref(), SmallVector<Value>{finalIdx});
    memOpLowering.recordReplacement(loadOp, flatLoadOp);
    return success();
  }

private:
  /// Used to record the operation replacement.
  MemoryOpLowering &memOpLowering;
};

struct StoreOpConversion : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  StoreOpConversion(MemoryOpLowering &memOpLowering, TypeConverter &converter,
                    MLIRContext *ctx)
      : OpConversionPattern(converter, ctx), memOpLowering(memOpLowering){};

  LogicalResult
  matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = storeOp.getMemRefType();
    if (isUniDimensional(type) || !type.hasStaticShape() ||
        /*Already converted?*/ storeOp.getIndices().size() == 1)
      return failure();
    Value finalIdx =
        flattenIndices(rewriter, storeOp.getLoc(), adaptor.getIndices(),
                       storeOp.getMemRefType());
    memref::StoreOp flatStoreOp = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        storeOp, adaptor.getValue(), adaptor.getMemref(),
        SmallVector<Value>{finalIdx});
    memOpLowering.recordReplacement(storeOp, flatStoreOp);
    return success();
  }

private:
  /// Used to record the operation replacement.
  MemoryOpLowering &memOpLowering;
};

struct AllocOpConversion : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getType();
    if (isUniDimensional(type) || !type.hasStaticShape())
      return failure();
    MemRefType newType = MemRefType::get(
        SmallVector<int64_t>{type.getNumElements()}, type.getElementType());
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, newType);
    return success();
  }
};

// A generic pattern which will replace an op with a new op of the same type
// but using the adaptor (type converted) operands.
template <typename TOp>
struct OperandConversionPattern : public OpConversionPattern<TOp> {
  using OpConversionPattern<TOp>::OpConversionPattern;
  using OpAdaptor = typename TOp::Adaptor;
  LogicalResult
  matchAndRewrite(TOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TOp>(op, op->getResultTypes(),
                                     adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

// Cannot use OperandConversionPattern for branch op since the default builder
// doesn't provide a method for communicating block successors.
struct CondBranchOpConversion
    : public OpConversionPattern<mlir::cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, adaptor.getCondition(), adaptor.getTrueDestOperands(),
        adaptor.getFalseDestOperands(), op.getTrueDest(), op.getFalseDest());
    return success();
  }
};

// Rewrites a call op signature to flattened types. If rewriteFunctions is set,
// will also replace the callee with a private definition of the called
// function of the updated signature.
struct CallOpConversion : public OpConversionPattern<func::CallOp> {
  CallOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   bool rewriteFunctions = false)
      : OpConversionPattern(typeConverter, context),
        rewriteFunctions(rewriteFunctions) {}

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();
    auto newCallOp = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, adaptor.getCallee(), convResTypes, adaptor.getOperands());

    if (!rewriteFunctions)
      return success();

    // Override any definition corresponding to the updated signature.
    // It is up to users of this pass to define how these rewritten functions
    // are to be implemented.
    rewriter.setInsertionPoint(op->getParentOfType<func::FuncOp>());
    auto *calledFunction = dyn_cast<CallOpInterface>(*op).resolveCallable();
    FunctionType funcType = FunctionType::get(
        op.getContext(), newCallOp.getOperandTypes(), convResTypes);
    func::FuncOp newFuncOp;
    if (calledFunction)
      newFuncOp = rewriter.replaceOpWithNewOp<func::FuncOp>(
          calledFunction, op.getCallee(), funcType);
    else
      newFuncOp =
          rewriter.create<func::FuncOp>(op.getLoc(), op.getCallee(), funcType);
    newFuncOp.setVisibility(SymbolTable::Visibility::Private);

    return success();
  }

private:
  bool rewriteFunctions;
};

template <typename... TOp>
void addGenericLegalityConstraint(ConversionTarget &target) {
  (target.addDynamicallyLegalOp<TOp>([](TOp op) {
    return !hasMultiDimMemRef(op->getOperands()) &&
           !hasMultiDimMemRef(op->getResults());
  }),
   ...);
}

static void populateFlattenMemRefsLegality(ConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addDynamicallyLegalOp<memref::AllocOp>(
      [](memref::AllocOp op) { return isUniDimensional(op.getType()); });
  target.addDynamicallyLegalOp<memref::StoreOp>(
      [](memref::StoreOp op) { return op.getIndices().size() == 1; });
  target.addDynamicallyLegalOp<memref::LoadOp>(
      [](memref::LoadOp op) { return op.getIndices().size() == 1; });

  addGenericLegalityConstraint<mlir::cf::CondBranchOp, mlir::cf::BranchOp,
                               func::CallOp, func::ReturnOp, memref::DeallocOp,
                               memref::CopyOp>(target);

  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
    auto argsConverted = llvm::none_of(op.getBlocks(), [](auto &block) {
      return hasMultiDimMemRef(block.getArguments());
    });

    auto resultsConverted = llvm::all_of(op.getResultTypes(), [](Type type) {
      if (auto memref = type.dyn_cast<MemRefType>())
        return isUniDimensional(memref);
      return true;
    });

    return argsConverted && resultsConverted;
  });
}

static void populateTypeConversionPatterns(TypeConverter &typeConverter) {
  // Add default conversion for all types generically.
  typeConverter.addConversion([](Type type) { return type; });
  // Add specific conversion for memref types.
  typeConverter.addConversion([](MemRefType memref) {
    if (isUniDimensional(memref))
      return memref;
    return MemRefType::get(llvm::SmallVector<int64_t>{memref.getNumElements()},
                           memref.getElementType());
  });
}

struct FlattenMemRefRowMajorPass
    : public dynamatic::impl::FlattenMemRefRowMajorBase<
          FlattenMemRefRowMajorPass> {
public:
  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();
    TypeConverter typeConverter;
    MemoryOpLowering memOpLowering(getAnalysis<NameAnalysis>());
    populateTypeConversionPatterns(typeConverter);

    RewritePatternSet patterns(ctx);
    SetVector<StringRef> rewrittenCallees;
    patterns.add<AllocOpConversion, OperandConversionPattern<func::ReturnOp>,
                 OperandConversionPattern<memref::DeallocOp>,
                 CondBranchOpConversion,
                 OperandConversionPattern<memref::DeallocOp>,
                 OperandConversionPattern<memref::CopyOp>, CallOpConversion>(
        typeConverter, ctx);
    patterns.add<LoadOpConversion, StoreOpConversion>(memOpLowering,
                                                      typeConverter, ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    ConversionTarget target(*ctx);
    populateFlattenMemRefsLegality(target);

    if (failed(applyPartialConversion(modOp, target, std::move(patterns))))
      return signalPassFailure();

    // Change the name of destination memory acceses in all stored memory
    // dependencies to reflect the new access names
    memOpLowering.renameDependencies(modOp);
  }
};

} // namespace

namespace dynamatic {
std::unique_ptr<dynamatic::DynamaticPass> createFlattenMemRefRowMajorPass() {
  return std::make_unique<FlattenMemRefRowMajorPass>();
}
} // namespace dynamatic