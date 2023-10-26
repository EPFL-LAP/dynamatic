//===- LLVM.h - Import and forward declare core LLVM types ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file forward declares and imports various common LLVM and MLIR datatypes
// that we want to use unqualified.
//
// Note that most of these are forward declared and then imported into the
// dynamatic namespace with using decls, rather than being #included. This is
// because we want clients to explicitly #include the files they need.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_LLVM_H
#define DYNAMATIC_SUPPORT_LLVM_H

// MLIR includes a lot of forward declarations of LLVM types, use them.
#include "mlir/Support/LLVM.h"

// Can not forward declare inline functions with default arguments, so we
// include the header directly.
#include "mlir/Support/LogicalResult.h"

// Import classes from the `mlir` namespace into the `dynamatic` namespace.  All
// of the following classes have been already forward declared and imported from
// `llvm` in to the `mlir` namespace. For classes with default template
// arguments, MLIR does not import the type directly, it creates a templated
// using statement. This is due to the limitiation that only one declaration of
// a type can have default arguments. For those types, it is important to import
// the MLIR version, and not the LLVM version. To keep things simple, all
// classes here should be imported from the `mlir` namespace, not the `llvm`
// namespace.

// NOLINTBEGIN(misc-unused-using-decls)

namespace dynamatic {
using mlir::APFloat;
using mlir::APInt;
using mlir::APSInt;
using mlir::ArrayRef;
using mlir::BitVector;
using mlir::cast;
using mlir::cast_or_null;
using mlir::DenseMap;
using mlir::DenseMapInfo;
using mlir::DenseSet;
using mlir::dyn_cast;
using mlir::dyn_cast_or_null;
using mlir::function_ref;
using mlir::isa;
using mlir::isa_and_nonnull;
using mlir::iterator_range;
using mlir::MutableArrayRef;
using mlir::PointerUnion;
using mlir::raw_ostream;
using mlir::SmallPtrSet;
using mlir::SmallPtrSetImpl;
using mlir::SmallString;
using mlir::SmallVector;
using mlir::SmallVectorImpl;
using mlir::StringLiteral;
using mlir::StringRef;
using mlir::StringSet;
using mlir::TinyPtrVector;
using mlir::Twine;
using mlir::TypeSwitch;
} // namespace dynamatic

// Forward declarations of LLVM classes to be imported into the dynamatic
// namespace.
namespace llvm {
template <typename KeyT, typename ValueT, unsigned InlineBuckets,
          typename KeyInfoT, typename BucketT>
class SmallDenseMap;
} // namespace llvm

// Import things we want into our namespace.
namespace dynamatic {
using llvm::SmallDenseMap;
} // namespace dynamatic

// Forward declarations of classes to be imported into the dynamatic namespace.
namespace mlir {
class ArrayAttr;
class AsmParser;
class AsmPrinter;
class Attribute;
class Block;
class BlockAndValueMapping;
class BlockArgument;
class BoolAttr;
class Builder;
class NamedAttrList;
class ConversionPattern;
class ConversionPatternRewriter;
class ConversionTarget;
class DenseElementsAttr;
class Diagnostic;
class Dialect;
class DialectAsmParser;
class DialectAsmPrinter;
class DictionaryAttr;
class ElementsAttr;
class FileLineColLoc;
class FlatSymbolRefAttr;
class FloatAttr;
class FunctionType;
class FusedLoc;
class ImplicitLocOpBuilder;
class IndexType;
class InFlightDiagnostic;
class IntegerAttr;
class IntegerType;
class Location;
class MemRefType;
class MLIRContext;
class ModuleOp;
class MutableOperandRange;
class NamedAttribute;
class NamedAttrList;
class NoneType;
class OpAsmDialectInterface;
class OpAsmParser;
class OpAsmPrinter;
class OpBuilder;
class OperandRange;
class Operation;
class OpFoldResult;
class OpOperand;
class OpResult;
template <typename OpTy>
class OwningOpRef;
class ParseResult;
class Pass;
class PatternRewriter;
class Region;
class RewritePatternSet;
class ShapedType;
class SplatElementsAttr;
class StringAttr;
class SymbolRefAttr;
class SymbolTable;
class SymbolTableCollection;
class TupleType;
class Type;
class TypeAttr;
class TypeConverter;
class TypeID;
class TypeRange;
class TypeStorage;
class UnitAttr;
class UnknownLoc;
class Value;
class ValueRange;
class VectorType;
class WalkResult;
enum class RegionKind;
struct CallInterfaceCallable;
struct LogicalResult;
struct MemRefAccess;
struct OperationState;
class OperationName;

template <typename T>
class FailureOr;
template <typename SourceOp>
class OpConversionPattern;
template <typename T>
class OperationPass;
template <typename SourceOp>
struct OpRewritePattern;

using DefaultTypeStorage = TypeStorage;
using OpAsmSetValueNameFn = function_ref<void(Value, StringRef)>;

namespace OpTrait {}

} // namespace mlir

// Import things we want into our namespace.
namespace dynamatic {
using mlir::ArrayAttr;
using mlir::AsmParser;
using mlir::AsmPrinter;
using mlir::Attribute;
using mlir::Block;
using mlir::BlockAndValueMapping;
using mlir::BlockArgument;
using mlir::BoolAttr;
using mlir::Builder;
using mlir::CallInterfaceCallable;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::DefaultTypeStorage;
using mlir::DenseElementsAttr;
using mlir::Diagnostic;
using mlir::Dialect;
using mlir::DialectAsmParser;
using mlir::DialectAsmPrinter;
using mlir::DictionaryAttr;
using mlir::ElementsAttr;
using mlir::failed;
using mlir::failure;
using mlir::FailureOr;
using mlir::FileLineColLoc;
using mlir::FlatSymbolRefAttr;
using mlir::FloatAttr;
using mlir::FunctionType;
using mlir::FusedLoc;
using mlir::ImplicitLocOpBuilder;
using mlir::IndexType;
using mlir::InFlightDiagnostic;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MemRefAccess;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::MutableOperandRange;
using mlir::NamedAttribute;
using mlir::NamedAttrList;
using mlir::NoneType;
using mlir::OpAsmDialectInterface;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OpAsmSetValueNameFn;
using mlir::OpBuilder;
using mlir::OpConversionPattern;
using mlir::OperandRange;
using mlir::Operation;
using mlir::OperationName;
using mlir::OperationPass;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::OpOperand;
using mlir::OpResult;
using mlir::OpRewritePattern;
using mlir::OwningOpRef;
using mlir::ParseResult;
using mlir::Pass;
using mlir::PatternRewriter;
using mlir::Region;
using mlir::RegionKind;
using mlir::RewritePatternSet;
using mlir::ShapedType;
using mlir::SplatElementsAttr;
using mlir::StringAttr;
using mlir::succeeded;
using mlir::success;
using mlir::SymbolRefAttr;
using mlir::SymbolTable;
using mlir::SymbolTableCollection;
using mlir::TupleType;
using mlir::Type;
using mlir::TypeAttr;
using mlir::TypeConverter;
using mlir::TypeID;
using mlir::TypeRange;
using mlir::TypeStorage;
using mlir::UnitAttr;
using mlir::UnknownLoc;
using mlir::Value;
using mlir::ValueRange;
using mlir::VectorType;
using mlir::WalkResult;
namespace OpTrait = mlir::OpTrait; // NOLINT(misc-unused-alias-decls)
} // namespace dynamatic

// NOLINTEND(misc-unused-using-decls)

#endif // DYNAMATIC_SUPPORT_LLVM_H
