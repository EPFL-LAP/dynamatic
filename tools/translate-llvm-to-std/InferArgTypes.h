#pragma once

#include "dynamatic/Support/LLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang-c/CXString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"

using namespace mlir;

/// \brief: The scalar types in the original C code. We could add our custom
/// types in the future.
enum CXBuiltInScalarTypes {
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

/// Used to denote _BitInt(..)---a type introduced in C23.
struct BitIntType {
  unsigned bitWidth;
  bool isUnsigned;
};

using CXScalarType = std::variant<CXBuiltInScalarTypes, BitIntType>;

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
  CXScalarType baseElemType;
  std::vector<int64_t> arrayDimensions;
  bool isPassedByReference;

  mlir::Type getMlirType(OpBuilder &builder) const;
};

using CFuncArgs = SmallVector<ArgType>;

/// \brief: Maps the function name to the set of MLIR types.
///
/// \example: int foo[1000] will be mapped to memref[1000xi32]
using FuncNameToCFuncArgsMap = std::map<std::string, CFuncArgs>;

/// \brief: Read the function arguments in the C source code, and convert them
/// into the corresponding types in MLIR that we support.
SmallVector<mlir::Type> getFuncArgTypes(const std::string &funcName,
                                        FuncNameToCFuncArgsMap map,
                                        OpBuilder &builder);

/// \brief: for a given function "funcName", get the corresponding types that
/// will be used in Dynamatic.
FuncNameToCFuncArgsMap inferArgTypes(const std::string &source,
                                     const std::string &includePath);
