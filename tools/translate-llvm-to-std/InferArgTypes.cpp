
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang-c/CXString.h"
#include "llvm/ADT/SmallVector.h"

#include "clang-c/Index.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <regex>

#include "InferArgTypes.h"

#define DEBUG_TYPE "infer-arg-types"

using namespace mlir;
using namespace dynamatic;

Type ArgType::getMlirType(OpBuilder &builder) const {
  Type baseMLIRElemType;

  if (std::holds_alternative<CXBuiltInScalarTypes>(baseElemType)) {
    switch (std::get<CXBuiltInScalarTypes>(baseElemType)) {
      // clang-format off
      case Void:       baseMLIRElemType = builder.getNoneType(); break;
      case Bool:       baseMLIRElemType = builder.getI1Type();   break;
      case Int8:       baseMLIRElemType = builder.getI8Type();   break;
      case Int16:      baseMLIRElemType = builder.getI16Type();  break;
      case Int32:      baseMLIRElemType = builder.getI32Type();  break;
      case Int64:      baseMLIRElemType = builder.getI64Type();  break;
      case Float:      baseMLIRElemType = builder.getF32Type();  break;
      case Double:     baseMLIRElemType = builder.getF64Type();  break;
      case LongDouble: baseMLIRElemType = builder.getF128Type(); break;
      // clang-format on
    case Elaborated:
      assert(false &&
             "Dynamatic currently cannot handle elaborated types (e.g., "
             "struct, typedef, etc).");
      break;
    }
  } else {
    baseMLIRElemType =
        builder.getIntegerType(std::get<BitIntType>(baseElemType).bitWidth);
  }

  if (arrayDimensions.empty()) {
    return baseMLIRElemType;
  }
  return MemRefType::get(llvm::ArrayRef<int64_t>(arrayDimensions),
                         baseMLIRElemType);
}

/// \brief: This function checks if clangType is a scalar type (e.g., int,
/// float, ..., anything that is not an array) and returns the corresponding
/// enum "ArgType". It return nothing if it is not a scalar type.
static std::optional<CXScalarType> processScalarType(CXType clangType) {
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

  case CXType_Unexposed: {
    // Newer C features are denoted as "CXType_Unexposed" in libclang
    // HACK: We use string regex to extract the integer type.
    std::string typeName = clang_getCString(clang_getTypeSpelling(clangType));
    std::regex re(R"(_BitInt\((\d+)\))");
    std::smatch match;
    if (std::regex_search(typeName, match, re)) {
      bool isUnsigned = false;
      unsigned width;
      if (typeName.find("unsigned") != std::string::npos) {
        isUnsigned = true;
      }
      width = std::stoi(match[1].str());
      return BitIntType{width, isUnsigned};
    }
    llvm_unreachable("Unhandled CXType_Unexposed type!");
    return std::nullopt;
  }
  default: {
    return std::nullopt;
  }
  }
}

/// \brief: Construct a new ArgType from a CXType (which will not be deallocated
/// later).
///
/// \note: This type inference is on a **best-effort** basis: it returns a null
/// value when we don't know what the type is.
///
/// \note: For non-built-in types, Clang marks them as "CXType_Elaborated".
/// - For scalar types defined through typedef, we use clang_getCanonicalType to
/// retrieve the basic type.
/// - For user defined types (e.g., through struct), we mark them as Elaborate
/// and throw an error when we try to convert that function into func::FuncOp.
/// Furthermore, we don't want to throw an error on Elaborated types here,
/// because the included headers might contain those types (but they will not be
/// used anywhere later).
static std::optional<ArgType> fromCXType(CXType type) {
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
  }
  // TODO: One important thing to handle in the future is the arguments that
  // are **passed by reference**. It is probably correct to promote them to
  // the function return values.
  //
  // TODO: Everything else is not handled yet.
  return std::nullopt;
}

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
    } else {
      llvm::errs() << "Warning - unable to parse " << getCursorSpelling(cursor)
                   << " with type "
                   << clang_getCString(clang_getTypeSpelling(type)) << "!\n";
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

SmallVector<Type> getFuncArgTypes(const std::string &funcName,
                                  FuncNameToCFuncArgsMap map,
                                  OpBuilder &builder) {
  SmallVector<Type> mlirArgTypes;
  for (const ArgType &clangType : map.at(funcName)) {
    mlirArgTypes.push_back(clangType.getMlirType(builder));
  }

  return mlirArgTypes;
}
