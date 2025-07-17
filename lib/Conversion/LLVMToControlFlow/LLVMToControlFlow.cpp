#include "dynamatic/Conversion/LLVMToControlFlow.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang-c/CXString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"

#include "clang-c/Index.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <optional>

using namespace mlir;
using namespace dynamatic;

// Remark: the auto-generated code from tblgen have a "impl::" namespace (e.g.,
// impl::LLVMToControlFlowBase). This namespace declaration additionally puts
// that one under the dynamatic:: namespace. So, to correctly reference the
// generated base class of the pass, you need to do dynamtic::impl::...(name of
// the base class of the pass)
namespace dynamatic {
// Remarks on the boilerplate code here:
// - Base class definition: Import auto-generated base class definition (that we
// specified in the tblgen file) "LLVMToControlFlowBase" and put it under the
// dynamatic namespace (enabled by GEN_PASS_DEF_LLVMTOCONTROLFLOW).
// - The compiler complains that "dynamatic::impl::create<name of the pass>()"
// is undefined if you don't derive that base class (see LLVMToControlFlowPass
// below).
#define GEN_PASS_DEF_LLVMTOCONTROLFLOW
#include "dynamatic/Conversion/Passes.h.inc"
} // namespace dynamatic

/// \brief: The scalar types in the original C code. We could add our custom
/// types in the future.
enum BaseScalarType {
  Void,
  Bool,
  Int8,
  Int16,
  Int32,
  Int64,
  Float,
  Double,
  LongDouble,
  // Use this as a placeholder for all elaborated types. e.g., typedef, struct,
  // etc.
  Elaborated,
};

/// \brief: This struct stores the type of the original C code. It is used
/// instead of CXType because CXType manages raw pointers and we cannot keep
/// them safely.
///
/// \note: struct members:
/// - baseElemType: the type of the argument itself, or the type of the array
/// element.
/// - arrayDimensions: The size of each dimension of the constant sized array.
/// Note that Dynamatic does not allow dynamically sized arrays.
/// - isPassedByReference: not used now. But will be used to indicate whether
/// the value is passed by reference in the future (we can (and probably should)
/// use this to determine whether we return the value at the output).
struct ArgType {
  BaseScalarType baseElemType;
  std::vector<int64_t> arrayDimensions;
  bool isPassedByReference;

  Type getMlirType(OpBuilder &builder) const;
};

Type ArgType::getMlirType(OpBuilder &builder) const {
  Type baseMLIRElemType;
  switch (baseElemType) {
  case Void:
    baseMLIRElemType = builder.getNoneType();
    break;
  case Bool:
    baseMLIRElemType = builder.getI1Type();
    break;
  case Int8:
    baseMLIRElemType = builder.getI8Type();
    break;
  case Int16:
    baseMLIRElemType = builder.getI16Type();
    break;
  case Int32:
    baseMLIRElemType = builder.getI32Type();
    break;
  case Int64:
    baseMLIRElemType = builder.getI64Type();
    break;
  case Float:
    baseMLIRElemType = builder.getF32Type();
    break;
  case Double:
    baseMLIRElemType = builder.getF64Type();
    break;
  case LongDouble:
    baseMLIRElemType = builder.getF128Type();
    break;
  case Elaborated:
    assert(false && "Dynamatic currently cannot handle elaborated types (e.g., "
                    "struct, typedef, etc).");
    break;
  }
  if (arrayDimensions.empty()) {
    return baseMLIRElemType;
  }
  return MemRefType::get(llvm::ArrayRef<int64_t>(arrayDimensions),
                         baseMLIRElemType);
}

/// This function checks if clangType is a scalar type (e.g., int, float, ...,
/// anything that is not an array) and returns the corresponding enum "ArgType".
/// It return nothing if it is not a scalar type.
std::optional<BaseScalarType> processScalarType(CXType clangType) {
  switch (clangType.kind) {
  case CXType_Bool:
    return Bool;

  case CXType_Char_U:
  case CXType_UChar:
  case CXType_Char_S:
  case CXType_SChar:
    return Int8;

  case CXType_UShort:
  case CXType_Short:
    return Int16;

  case CXType_UInt:
  case CXType_Int:
    return Int32;

  case CXType_ULong:
  case CXType_ULongLong:
  case CXType_Long:
  case CXType_LongLong:
    return Int64;

  case CXType_Float:
    return Float;

  case CXType_Double:
    return Double;

  case CXType_LongDouble:
    return LongDouble;

  case CXType_Elaborated: {
    // Clang treats typedef or struct as elaborated types. Here, we try to
    // handle typedef of scalar types (e.g., int8_t in <stdint.h>).
    CXType canonical = clang_getCanonicalType(clangType);
    auto tryToParse = processScalarType(canonical);
    if (tryToParse.has_value()) {
      return tryToParse;
    }
    return Elaborated;
  }
  default:
    return std::nullopt;
  }
}

/// \brief: Construct a new ArgType from a CXType (which will not be
/// deallocated later).
///
/// \note: This type inference is on a **best-effort** basis: it returns a null
/// value when we don't know what the type is.
///
/// \note: For non-built-in types, Clang treat them as "Elaborated".
/// - For scalar types defined through typedef, we use clang_getCanonicalType to
/// retrieve the basic type.
/// - For user defined types (e.g., through struct), we mark them as Elaborate
/// and throw an error when we try to convert that function into func::FuncOp.
/// Furthermore, we don't want to throw an error on Elaborated types here,
/// because the included headers might contain those types (but they will not be
/// used anywhere later).
std::optional<ArgType> fromCXType(CXType type) {
  // Handle scalar type
  if (auto scalarType = processScalarType(type); scalarType.has_value()) {
    return ArgType{*scalarType, {}, false};
  }

  // Handle array type with constant sizes
  if (type.kind == CXType_ConstantArray) {
    std::vector<int64_t> arrayDimSizes;
    CXType arrayType = type;
    // Iterate through the array type to get the dimension sizes.
    while (arrayType.kind == CXType_ConstantArray) {
      arrayDimSizes.push_back(clang_getArraySize(arrayType));
      arrayType = clang_getArrayElementType(arrayType);
    }

    if (auto scalarType = processScalarType(arrayType);
        scalarType.has_value()) {
      return ArgType{scalarType.value(), arrayDimSizes, false};
    }

    /// \todo: One important thing to handle in the future is the arguments that
    /// are **passed by reference**. It is probably correct to promote them to
    /// the function return values.
  }

  /// \todo: Everything else is not handled yet.
  return std::nullopt;
}

using CFuncArgs = SmallVector<ArgType>;

/// \brief: Maps the function name to the set of MLIR types.
///
/// \example: int foo[1000] will be mapped to memref[1000xi32]
using FuncNameToCFuncArgsMap = std::map<std::string, CFuncArgs>;

/// \brief: This is used to store the string of the names of the C functions,
/// which is use to recover the actual parameter types when building the
/// func.funcOp.
static std::string getCursorSpelling(CXCursor cursor) {
  CXString cursorSpelling = clang_getCursorSpelling(cursor);
  std::string result = clang_getCString(cursorSpelling);

  clang_disposeString(cursorSpelling);
  return result;
}

/// \brief: Visits all parameter declarations at the cursor's level and saves
/// their respective name. CXClientData is expected to te a `FuncArgs*`.
static CXChildVisitResult visitParamDecl(CXCursor cursor, CXCursor parent,
                                         CXClientData argsPtr) {
  CXCursorKind cursorKind = clang_getCursorKind(cursor);
  if (cursorKind == CXCursor_ParmDecl) {
    CFuncArgs *args = reinterpret_cast<CFuncArgs *>(argsPtr);
    CXType type = clang_getCursorType(cursor);

    auto argType = fromCXType(type);
    if (argType.has_value()) {
      args->push_back(argType.value());
    }
    // else: Maybe instead of push nothing here, we should have a ArgType that
    // is specifically for "I don't know what it is?"

#ifdef LOG_CLANG_AST
    llvm::errs() << "Name " << getCursorSpelling(cursor) << "\n";
    llvm::errs() << "Type: " << clang_getCString(clang_getTypeSpelling(type))
                 << "\n";
    llvm::errs() << "Type kind: " << type.kind << "\n";
#endif
  }
  return CXChildVisit_Continue;
}

/// \brief: Visits all function declarations at the cursor's level; for each,
/// visits all their parameter declarations one level below in the AST.
/// CXClientData is expected to te a `FuncData*`.
static CXChildVisitResult visitFuncDecl(CXCursor cursor, CXCursor parent,
                                        CXClientData dataPtr) {
  CXCursorKind cursorKind = clang_getCursorKind(cursor);
  if (cursorKind == CXCursor_FunctionDecl) {
    FuncNameToCFuncArgsMap *data =
        reinterpret_cast<FuncNameToCFuncArgsMap *>(dataPtr);
    CFuncArgs &args = (*data)[getCursorSpelling(cursor)];
    if (args.empty())
      clang_visitChildren(cursor, visitParamDecl, &args);
  }
  return CXChildVisit_Continue;
}

/// \brief: for a givne function "funcName", get the corresponding types that
/// will be used in Dynamatic.
FuncNameToCFuncArgsMap inferArgTypes(const std::string &source,
                                     const std::string &includePath) {

  // NOTE: we need this to #include "dynamtic/Integration.h"
  std::string includeArg = "-I" + includePath;
  const char *args[] = {includeArg.c_str()};

  CXIndex index = clang_createIndex(0, 0);
  CXTranslationUnit unit = clang_parseTranslationUnit(
      index, source.c_str(), args, 0, nullptr, 0, CXTranslationUnit_None);
  if (unit == nullptr) {
    llvm::errs() << "Unable to parse translation unit\n";
  }
  CXCursor cursor = clang_getTranslationUnitCursor(unit);

  // Visit all functions in the C source and store the name of their arguments
  FuncNameToCFuncArgsMap data;
  clang_visitChildren(cursor, visitFuncDecl, &data);

  // Free all clang resources
  clang_disposeTranslationUnit(unit);
  clang_disposeIndex(index);

  return data;
}

namespace {

struct ConvertLLVMFuncOp : public OpConversionPattern<LLVM::LLVMFuncOp> {
  using OpConversionPattern<LLVM::LLVMFuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.isExternal())
      return failure();
    // Convert function type
    LLVM::LLVMFunctionType oldFuncType = op.getFunctionType();
    rewriter.setInsertionPoint(op);
    SmallVector<Type, 10> newTypes;
    auto newFuncType = rewriter.getFunctionType(oldFuncType.getParams(),
                                                oldFuncType.getReturnTypes());
    // Create new func::FuncOp
    auto newFunc =
        rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), newFuncType);

    // If function has a body, move it over (with argument remapping)
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());

    rewriter.eraseOp(op);

    return success();
  }
};

/// \brief: This struct rewrites all the pattern that connects GEP -> {Load1,
/// Load2, ..., Store1, Store2, ...}.
struct ConvertGEPToLoadStoreOpPattern
    : public OpConversionPattern<LLVM::GEPOp> {
  using OpConversionPattern<LLVM::GEPOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::GEPOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    if (!op || op.use_empty()) {
      return failure();
    }

    Value gepBasePtr = op.getBase();

    auto memrefType = gepBasePtr.getType().dyn_cast<MemRefType>();
    if (!memrefType) {
      // note: maybe we need to signal pass failure here if the GEP is not
      // connected to a memref.
      return failure();
    }

    // NOTE: Unlike LLVM::GEPOp, memref::LoadOp and memref::StoreOp expect
    // indices to be of IndexType. So here we cast the i32/i64 indices to
    // IndexType. This pattern will later be folded in the bitwidth optimization
    // pass.
    rewriter.setInsertionPoint(op);
    SmallVector<Value> indexValues;
    for (auto i64Value : op.getIndices()) {
      Value index = i64Value.dyn_cast<Value>();
      assert(index);
      auto idxCastOp = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), index);
      indexValues.push_back(idxCastOp);
    }

    // For each llvm::load or llvm::store connected to GEP, create a
    // memref::load or memref::store that takes the indices from indexValues

    for (Operation *op : op->getUsers()) {
      rewriter.setInsertionPoint(op);
      if (auto loadOp = dyn_cast<LLVM::LoadOp>(op)) {
        rewriter.replaceOpWithNewOp<memref::LoadOp>(
            loadOp, loadOp.getResult().getType(), gepBasePtr,
            ValueRange(indexValues));
      } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
        rewriter.replaceOpWithNewOp<memref::StoreOp>(
            storeOp, storeOp.getValue(), gepBasePtr, ValueRange(indexValues));
      } else {
        assert(false && "Unhandled connection GEPOp -> ???");
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename LLVMOpTy, typename ArithOpTy>
struct LLVMTOBinaryArithOpPattern : public OpConversionPattern<LLVMOpTy> {
  using OpConversionPattern<LLVMOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVMOpTy op,
                  typename OpConversionPattern<LLVMOpTy>::OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adapter.getLhs();
    Value rhs = adapter.getRhs();
    rewriter.replaceOpWithNewOp<ArithOpTy>(op, lhs, rhs);
    return success();
  }
};

template <typename LLVMUnaryOp, typename ArithUnaryOp>
struct LLVMToUnaryArithOpPattern : public OpConversionPattern<LLVMUnaryOp> {
  using OpConversionPattern<LLVMUnaryOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVMUnaryOp op,
                  typename OpConversionPattern<LLVMUnaryOp>::OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value operand = adapter.getArg();
    rewriter.replaceOpWithNewOp<ArithUnaryOp>(op, op.getRes().getType(),
                                              operand);
    return success();
  }
};

struct ICmpOpLowering : OpConversionPattern<LLVM::ICmpOp> {
  using OpConversionPattern<LLVM::ICmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ICmpOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    arith::CmpIPredicate pred;

    switch (op.getPredicate()) {
      // clang-format off
      case LLVM::ICmpPredicate::eq:  pred = arith::CmpIPredicate::eq; break;
      case LLVM::ICmpPredicate::ne:  pred = arith::CmpIPredicate::ne; break;
      case LLVM::ICmpPredicate::ugt: pred = arith::CmpIPredicate::ugt; break;
      case LLVM::ICmpPredicate::uge: pred = arith::CmpIPredicate::uge; break;
      case LLVM::ICmpPredicate::ult: pred = arith::CmpIPredicate::ult; break;
      case LLVM::ICmpPredicate::ule: pred = arith::CmpIPredicate::ule; break;
      case LLVM::ICmpPredicate::sgt: pred = arith::CmpIPredicate::sgt; break;
      case LLVM::ICmpPredicate::sge: pred = arith::CmpIPredicate::sge; break;
      case LLVM::ICmpPredicate::slt: pred = arith::CmpIPredicate::slt; break;
      case LLVM::ICmpPredicate::sle: pred = arith::CmpIPredicate::sle; break;
      // clang-format on
    }

    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, pred, op.getLhs(),
                                               op.getRhs());

    return success();
  }
};

struct FCmpOpLowering : OpConversionPattern<LLVM::FCmpOp> {
  using OpConversionPattern<LLVM::FCmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::FCmpOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    arith::CmpFPredicate pred;

    switch (op.getPredicate()) {
      // clang-format off
      case LLVM::FCmpPredicate::oeq: pred = arith::CmpFPredicate::OEQ; break;
      case LLVM::FCmpPredicate::ogt: pred = arith::CmpFPredicate::OGT; break;
      case LLVM::FCmpPredicate::oge: pred = arith::CmpFPredicate::OGE; break;
      case LLVM::FCmpPredicate::olt: pred = arith::CmpFPredicate::OLT; break;
      case LLVM::FCmpPredicate::ole: pred = arith::CmpFPredicate::OLE; break;
      case LLVM::FCmpPredicate::one: pred = arith::CmpFPredicate::ONE; break;
      case LLVM::FCmpPredicate::ord: pred = arith::CmpFPredicate::ORD; break;
      case LLVM::FCmpPredicate::ueq: pred = arith::CmpFPredicate::UEQ; break;
      case LLVM::FCmpPredicate::ugt: pred = arith::CmpFPredicate::UGT; break;
      case LLVM::FCmpPredicate::uge: pred = arith::CmpFPredicate::UGE; break;
      case LLVM::FCmpPredicate::ult: pred = arith::CmpFPredicate::ULT; break;
      case LLVM::FCmpPredicate::ule: pred = arith::CmpFPredicate::ULE; break;
      case LLVM::FCmpPredicate::une: pred = arith::CmpFPredicate::UNE; break;
      case LLVM::FCmpPredicate::uno: pred = arith::CmpFPredicate::UNO; break;
      default: return failure();
      // clang-format on
    }
    rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, pred, op.getLhs(),
                                               op.getRhs());

    return success();
  }
};

struct LLVMConstantOpLowering : OpConversionPattern<LLVM::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ConstantOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    auto valueAttr = op.getValue();

    // Handle only integer and float types
    if (auto intAttr = valueAttr.dyn_cast<IntegerAttr>()) {
      rewriter.setInsertionPoint(op);

      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, intAttr);
      return success();
    }

    if (auto floatAttr = valueAttr.dyn_cast<FloatAttr>()) {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, floatAttr);
      return success();
    }

    return failure();
  }
};

struct BranchOpLowering : OpConversionPattern<LLVM::BrOp> {
  using OpConversionPattern<LLVM::BrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::BrOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    // llvm.br has only one operand: the target block
    Block *targetBlock = op.getSuccessor();
    rewriter.setInsertionPoint(op);
    rewriter.create<cf::BranchOp>(op->getLoc(), op->getOperands(), targetBlock);
    rewriter.eraseOp(op);
    return success();
  }
};

struct CondBranchOpLowering : OpConversionPattern<LLVM::CondBrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::CondBrOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    Value condition = op.getCondition();
    Block *trueDest = op.getTrueDest();
    Block *falseDest = op.getFalseDest();

    // TODO: We need to look up the block argument of the branch target block to
    // see the branch argument.

    ValueRange trueOperands = op.getTrueDestOperands();
    ValueRange falseOperands = op.getFalseDestOperands();

    rewriter.setInsertionPoint(op);

    rewriter.create<cf::CondBranchOp>(op->getLoc(), condition, trueDest,
                                      trueOperands, falseDest, falseOperands);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LLVMSelectToArithSelectPattern : mlir::ConversionPattern {
  explicit LLVMSelectToArithSelectPattern(mlir::MLIRContext *context)
      : mlir::ConversionPattern(mlir::LLVM::SelectOp::getOperationName(), 1,
                                context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto selectOp = llvm::cast<mlir::LLVM::SelectOp>(op);

    // Check if the condition is i1 and the true/false values are of the same
    // type
    auto condType = selectOp.getCondition().getType();
    auto trueType = selectOp.getTrueValue().getType();
    auto falseType = selectOp.getFalseValue().getType();

    if (condType != rewriter.getI1Type())
      return rewriter.notifyMatchFailure(op, "condition is not i1");

    if (trueType != falseType)
      return rewriter.notifyMatchFailure(
          op, "true and false values must have the same type");

    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        op, trueType, selectOp.getCondition(), selectOp.getTrueValue(),
        selectOp.getFalseValue());

    return mlir::success();
  }
};

struct LLVMReturnOpLowering : public mlir::OpConversionPattern<LLVM::ReturnOp> {
  using OpConversionPattern<LLVM::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::LLVM::ReturnOp op, OpAdaptor adapter,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> operands(op.getOperands());

    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, operands);
    return mlir::success();
  }
};

} // namespace

// Naive conversion patterns (note that they don't handle any semantic details
// like rounding, etc..)

// clang-format off

// Integer arithmetic
using AddIOpLowering = LLVMTOBinaryArithOpPattern<LLVM::AddOp, arith::AddIOp>;
using SubIOpLowering = LLVMTOBinaryArithOpPattern<LLVM::SubOp, arith::SubIOp>;
using MulIOpLowering = LLVMTOBinaryArithOpPattern<LLVM::MulOp, arith::MulIOp>;
using DivSIOpLowering = LLVMTOBinaryArithOpPattern<LLVM::SDivOp, arith::DivSIOp>;
using DivUIOpLowering = LLVMTOBinaryArithOpPattern<LLVM::UDivOp, arith::DivUIOp>;
using RemSIOpLowering = LLVMTOBinaryArithOpPattern<LLVM::SRemOp, arith::RemSIOp>;
using RemUIOpLowering = LLVMTOBinaryArithOpPattern<LLVM::URemOp, arith::RemUIOp>;

// Floating point arithmetic
using AddFOpLowering = LLVMTOBinaryArithOpPattern<LLVM::FAddOp, arith::AddFOp>;
using SubFOpLowering = LLVMTOBinaryArithOpPattern<LLVM::FSubOp, arith::SubFOp>;
using MulFOpLowering = LLVMTOBinaryArithOpPattern<LLVM::FMulOp, arith::MulFOp>;
using DivFOpLowering = LLVMTOBinaryArithOpPattern<LLVM::FDivOp, arith::DivFOp>;
using RemFOpLowering = LLVMTOBinaryArithOpPattern<LLVM::FRemOp, arith::RemFOp>;

// Unary cast arithmetic
using SExtOpLowering = LLVMToUnaryArithOpPattern<LLVM::SExtOp, arith::ExtSIOp>;
using ZExtOpLowering = LLVMToUnaryArithOpPattern<LLVM::ZExtOp, arith::ExtUIOp>;
using FPExtOpLowering = LLVMToUnaryArithOpPattern<LLVM::FPExtOp, arith::ExtFOp>;
using TruncIOpLowering = LLVMToUnaryArithOpPattern<LLVM::TruncOp, arith::TruncIOp>;
using FPTruncOpLowering = LLVMToUnaryArithOpPattern<LLVM::FPTruncOp, arith::TruncFOp>;
using SIToFPOpLowering = LLVMToUnaryArithOpPattern<LLVM::SIToFPOp, arith::SIToFPOp>;
using UIToFPOpLowering = LLVMToUnaryArithOpPattern<LLVM::UIToFPOp, arith::UIToFPOp>;
using FPToSIOpLowering = LLVMToUnaryArithOpPattern<LLVM::FPToSIOp, arith::FPToSIOp>;
using FPToUIOpLowering = LLVMToUnaryArithOpPattern<LLVM::FPToUIOp, arith::FPToUIOp>;
// clang-format on

/// \brief: Read the function arguments in the C source code, and convert them
/// into the corresponding types in MLIR that we support.
SmallVector<Type> getFuncArgTypes(const std::string &funcName,
                                  FuncNameToCFuncArgsMap map,
                                  OpBuilder &rewriter) {
  SmallVector<Type> mlirArgTypes;
  for (const ArgType &clangType : map.at(funcName)) {
    mlirArgTypes.push_back(clangType.getMlirType(rewriter));
  }

  return mlirArgTypes;
}

namespace {
struct LLVMToControlFlowPass
    : public dynamatic::impl::LLVMToControlFlowBase<LLVMToControlFlowPass> {
  FuncNameToCFuncArgsMap nameToArgTypesMap;

  /// \note: Use the auto-generated construtors from tblgen
  using LLVMToControlFlowBase::LLVMToControlFlowBase;
  void runOnOperation() override;
};
} // namespace

/// \brief: Conversion steps:
/// 1. Function rewrite: Rewrite all LLVMFuncOps to func::FuncOps and move
/// all the blocks in the old functions into the new functions (this step
/// doesn't touch anything inside the functions).
/// 2. Fixing the function arguments: Convert all the raw types in LLVMIR to
/// more descriptive types. E.g., raw pointer to memref. The function argument
/// information is not available in the LLVM IR or the MLIR LLVMIR dialect, so
/// we use libclang to get these information from the original C code.
/// 3. Rewrite all the memory operations. LLVM uses GEP -> LoadOps/StoreOps,
/// since we use memref, we can convert all the GEP -> {many loads/stores} to
/// memref::load and memref::stores.
/// 4. Rewrite all the remaining operations that have trivial conversion
/// patterns.
///
/// \note: it is not clear if we can combine all the 4 steps in a single
/// applyPartialConversion.
///
/// \note: it is also not clear if we need to applyFullConversion.
void LLVMToControlFlowPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ModuleOp modOp = llvm::dyn_cast<ModuleOp>(getOperation());
  OpBuilder builder(ctx);

  // Setup conversion target
  ConversionTarget target(*ctx);

  // NOTE: somehow if I don't explicit mark the legal dialects, the inserted new
  // ops just simply gets dropped?!
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect, index::IndexDialect,
                         cf::ControlFlowDialect>();

  RewritePatternSet convertLLVMToFuncDialectPatterns(ctx);

  convertLLVMToFuncDialectPatterns.add<ConvertLLVMFuncOp>(ctx);

  if (failed(applyPartialConversion(
          modOp, target, std::move(convertLLVMToFuncDialectPatterns)))) {
    llvm::errs() << "LLVMFuncOp -> func::FuncOp conversion failed!\n";
    return signalPassFailure();
  }

  // Fixing the argument types of the funcOps using the information available
  // from the original C code.
  nameToArgTypesMap = inferArgTypes(source, dynamatic_path + "/include");
  for (auto funcOp : modOp.getOps<func::FuncOp>()) {
    if (funcOp.isExternal())
      continue;

    // Convert function type
    auto oldFuncType = funcOp.getFunctionType();

    assert(nameToArgTypesMap.at(funcOp.getSymName().str()).size() ==
               oldFuncType.getNumInputs() &&
           "Unmatched number of function arguments!");
    SmallVector<Type, 10> convertedInputs =
        getFuncArgTypes(funcOp.getSymName().str(), nameToArgTypesMap, builder);

    SmallVector<Type, 1> convertedResults;

    // The LLVM function returns llvm.void even if it has no results. So the
    // number of results must be 1.
    // Reference: https://mlir.llvm.org/docs/Dialects/LLVM/#function-types
    assert(oldFuncType.getNumResults() == 1);

    if (!oldFuncType.getResult(0).isa<LLVM::LLVMVoidType>()) {
      convertedResults.push_back(oldFuncType.getResult(0));
    }

    auto newType = builder.getFunctionType(convertedInputs, convertedResults);

    funcOp.setFunctionType(newType);
    for (auto [oldArg, newArg] : llvm::zip_equal(
             funcOp.getBody().front().getArguments(), convertedInputs)) {
      oldArg.setType(newArg);
    }
  }

  RewritePatternSet rewriteLoadStoreOperations(ctx);
  rewriteLoadStoreOperations.add<
      // clang-format off
      ConvertGEPToLoadStoreOpPattern
      // clang-format on
      >(ctx);
  if (failed(applyPartialConversion(modOp, target,
                                    std::move(rewriteLoadStoreOperations)))) {
    llvm::errs() << "Failed to convert GEP -> Load/Store patterns!\n";
    return signalPassFailure();
  }

  RewritePatternSet oneToOneConversionPatterns(ctx);
  oneToOneConversionPatterns.add<
      // clang-format off

      // Zero-ary operation:
      LLVMConstantOpLowering,
      
      // Unary operations
      SExtOpLowering,
      ZExtOpLowering,
      FPExtOpLowering,
      TruncIOpLowering,
      FPTruncOpLowering,
      SIToFPOpLowering,
      UIToFPOpLowering,
      FPToSIOpLowering,
      FPToUIOpLowering,

      // Binary operations:
      AddIOpLowering,
      SubIOpLowering,
      MulIOpLowering,
      DivSIOpLowering,
      DivUIOpLowering,
      RemSIOpLowering,
      RemUIOpLowering,
      AddFOpLowering,
      SubFOpLowering,
      MulFOpLowering,
      DivFOpLowering,
      RemFOpLowering,

      // Binary predicate comparisons:
      ICmpOpLowering,
      FCmpOpLowering,

      // Control flow operations:
      BranchOpLowering,
      CondBranchOpLowering,
      LLVMReturnOpLowering,

      LLVMSelectToArithSelectPattern
      // clang-format on
      >(ctx);

  if (failed(applyPartialConversion(modOp, target,
                                    std::move(oneToOneConversionPatterns)))) {
    llvm::errs() << "Failed to convert all remaining trivial patterns!\n";
    return signalPassFailure();
  }
}
