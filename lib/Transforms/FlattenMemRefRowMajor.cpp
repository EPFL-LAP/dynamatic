//===- FlattenMemRefROwMajor.cpp - MemRef flattening pass -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the MemRef flattening pass. It is closely modeled
// on the MemRef flattening pass from CIRCT bus uses row-major indexing to
// convert multidimensional load and store operations.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/FlattenMemRefRowMajor.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace circt;
using namespace dynamatic;

static inline bool isUniDimensional(MemRefType memref) {
  return memref.getShape().size() == 1;
}

/// Flatten indices in row-major style, making adjacent indices in the last
/// memref dimension be adjacent indices in the flattened memref.
static Value flattenIndices(ConversionPatternRewriter &rewriter, Operation *op,
                            ValueRange indices, MemRefType memrefType) {
  assert(memrefType.hasStaticShape() && "expected statically shaped memref");
  Location loc = op->getLoc();
  auto numIndices = indices.size();

  if (numIndices == 0) {
    // Singleton memref (e.g. memref<i32>) - return 0
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0))
        .getResult();
  }

  if (numIndices == 1)
    // Memref is already unidimensional
    return indices.front();

  // Iterate over indices to compute the final unidimensional index
  Value finalIdx = indices.back();
  int64_t dimProduct = 1;
  for (size_t i = 0, e = numIndices - 1; i < e; ++i) {
    auto memIdx = numIndices - i - 2;
    Value partialIdx = indices[memIdx];
    dimProduct *= memrefType.getShape()[memIdx];

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
    auto sumOp = rewriter.create<arith::AddIOp>(loc, finalIdx, partialIdx);
    finalIdx = sumOp.getResult();
  }
  return finalIdx;
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

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getMemRefType();
    if (isUniDimensional(type) || !type.hasStaticShape() ||
        /*Already converted?*/ op.getIndices().size() == 1)
      return failure();
    Value finalIdx =
        flattenIndices(rewriter, op, adaptor.getIndices(), op.getMemRefType());
    memref::LoadOp newLoadOp = rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, adaptor.getMemref(), SmallVector<Value>{finalIdx});
    copyAttr<handshake::MemDependenceArrayAttr, handshake::MemInterfaceAttr>(
        op, newLoadOp);
    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getMemRefType();
    if (isUniDimensional(type) || !type.hasStaticShape() ||
        /*Already converted?*/ op.getIndices().size() == 1)
      return failure();
    Value finalIdx =
        flattenIndices(rewriter, op, adaptor.getIndices(), op.getMemRefType());
    memref::StoreOp newStoreOp = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, adaptor.getValue(), adaptor.getMemref(),
        SmallVector<Value>{finalIdx});
    copyAttr<handshake::MemDependenceArrayAttr, handshake::MemInterfaceAttr>(
        op, newStoreOp);
    return success();
  }
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
    : public FlattenMemRefRowMajorBase<FlattenMemRefRowMajorPass> {
public:
  void runDynamaticPass() override {

    auto *ctx = &getContext();
    TypeConverter typeConverter;
    populateTypeConversionPatterns(typeConverter);

    RewritePatternSet patterns(ctx);
    SetVector<StringRef> rewrittenCallees;
    patterns.add<LoadOpConversion, StoreOpConversion, AllocOpConversion,
                 OperandConversionPattern<func::ReturnOp>,
                 OperandConversionPattern<memref::DeallocOp>,
                 CondBranchOpConversion,
                 OperandConversionPattern<memref::DeallocOp>,
                 OperandConversionPattern<memref::CopyOp>, CallOpConversion>(
        typeConverter, ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    ConversionTarget target(*ctx);
    populateFlattenMemRefsLegality(target);

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed()) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace dynamatic {
std::unique_ptr<dynamatic::DynamaticPass> createFlattenMemRefRowMajorPass() {
  return std::make_unique<FlattenMemRefRowMajorPass>();
}
} // namespace dynamatic