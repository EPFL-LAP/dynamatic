//===- FuncSetArgNames.cpp - Set argument names from C source ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --func-set-arg-names pass.
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/FuncSetArgNames.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang-c/Index.h"

using namespace mlir;
using namespace dynamatic;

using FuncArgs = SmallVector<std::string>;
using FuncData = llvm::StringMap<FuncArgs>;

/// Returns the cursor's spelling as a C++ string.
static std::string getCursorSpelling(CXCursor cursor) {
  CXString cursorSpelling = clang_getCursorSpelling(cursor);
  std::string result = clang_getCString(cursorSpelling);

  clang_disposeString(cursorSpelling);
  return result;
}

/// Visits all parameter declarations at the cursor's level and saves their
/// respective name. CXClientData is expected to te a `FuncArgs*`.
static CXChildVisitResult visitParamDecl(CXCursor cursor, CXCursor parent,
                                         CXClientData argsPtr) {
  CXCursorKind cursorKind = clang_getCursorKind(cursor);
  if (cursorKind == CXCursor_ParmDecl) {
    FuncArgs *args = reinterpret_cast<FuncArgs *>(argsPtr);
    args->push_back(getCursorSpelling(cursor));
  }
  return CXChildVisit_Continue;
}

/// Visits all function declarations at the cursor's level; for each, visits all
/// their parameter declarations one level below in the AST. CXClientData is
/// expected to te a `FuncData*`.
static CXChildVisitResult visitFuncDecl(CXCursor cursor, CXCursor parent,
                                        CXClientData dataPtr) {
  CXCursorKind cursorKind = clang_getCursorKind(cursor);
  if (cursorKind == CXCursor_FunctionDecl) {
    FuncData *data = reinterpret_cast<FuncData *>(dataPtr);
    FuncArgs &args = (*data)[getCursorSpelling(cursor)];
    if (args.empty())
      clang_visitChildren(cursor, visitParamDecl, &args);
  }
  return CXChildVisit_Continue;
}

namespace {

/// Simple pass driver that replaces the functions argument names of each
/// Handshake-level function in the IR.
struct FuncSetArgNamesPass
    : public dynamatic::experimental::impl::FuncSetArgNamesBase<
          FuncSetArgNamesPass> {

  FuncSetArgNamesPass(StringRef source) { this->source = source.str(); }

  void runDynamaticPass() override {
    // Open the source file with clang
    CXIndex index = clang_createIndex(0, 0);
    CXTranslationUnit unit = clang_parseTranslationUnit(
        index, source.c_str(), nullptr, 0, nullptr, 0, CXTranslationUnit_None);
    if (unit == nullptr) {
      llvm::errs() << "Unable to parse translation unit\n";
      return signalPassFailure();
    }
    CXCursor cursor = clang_getTranslationUnitCursor(unit);

    // Visit all functions in the C source and store the name of their arguments
    FuncData data;
    clang_visitChildren(cursor, visitFuncDecl, &data);

    // Free all clang resources
    clang_disposeTranslationUnit(unit);
    clang_disposeIndex(index);

    // Set argument names on func-level functions in the IR
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();
    auto nameAttrName = StringAttr::get(ctx, "handshake.arg_name");
    for (auto funcOp : modOp.getOps<func::FuncOp>()) {
      StringRef funcName = funcOp.getName();
      if (auto funcData = data.find(funcName); funcData != data.end()) {
        SmallVector<Attribute> args;
        llvm::transform(
            funcData->second, std::back_inserter(args), [&](StringRef argName) {
              auto argNameAttr = StringAttr::get(ctx, argName);
              auto namedArgNameAttr = NamedAttribute(nameAttrName, argNameAttr);
              return DictionaryAttr::get(ctx, {namedArgNameAttr});
            });
        funcOp.setArgAttrsAttr(ArrayAttr::get(&getContext(), args));
      } else {
        funcOp->emitError()
            << "Failed to find function in C source (" << source << ")";
        return signalPassFailure();
      }
    }
  };
};

} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::createFuncSetArgNames(StringRef source) {
  return std::make_unique<FuncSetArgNamesPass>(source);
}
