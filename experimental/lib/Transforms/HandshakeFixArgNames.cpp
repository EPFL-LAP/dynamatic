//===- HandshakeFixArgNames.cpp - Match argument names with C ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --handshake-fix-arg-names pass.
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/HandshakeFixArgNames.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <fstream>

using namespace mlir;
using namespace circt;
using namespace dynamatic;

namespace {

/// Simple pass driver that replaces the functions argument names of each
/// Handshake-level function in the IR.
struct HandshakeFixArgNamesPass
    : public dynamatic::experimental::impl::HandshakeFixArgNamesBase<
          HandshakeFixArgNamesPass> {

  HandshakeFixArgNamesPass(const std::string &source) { this->source = source; }

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();

    // Open the source file
    std::ifstream inputFile(source);
    std::stringstream ss;
    if (!inputFile.is_open()) {
      modOp->emitError() << "Failed to open C source";
      return signalPassFailure();
    }

    // Read the C source line-by-line into memory
    std::string line;
    while (std::getline(inputFile, line))
      ss << line;
    std::string srcTxt = ss.str();

    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>())
      if (failed(fixArgNames(funcOp, srcTxt)))
        return signalPassFailure();
  };

  /// Tries to identify the function declaration corresponding to the Handshake
  /// function inside the source file and replaces the Handshake function's
  /// argument names with what was parsed. Fantastically hacky, but appears to
  /// work in non-degenerate cases.
  LogicalResult fixArgNames(handshake::FuncOp funcOp, std::string &srcTxt);
};

} // namespace

LogicalResult HandshakeFixArgNamesPass::fixArgNames(handshake::FuncOp funcOp,
                                                    std::string &srcTxt) {
  std::string funName = funcOp.getName().str();
  size_t startIdx = 0;
  do {
    // Find the matching function in the source file
    size_t idx = srcTxt.find(funName, startIdx);
    if (idx == std::string::npos)
      return funcOp->emitError() << "Failed to find function in source";

    // The function name must be followed by an open paranthesis (with possible
    // spaces)
    size_t openIdx = idx + funName.size();
    for (size_t e = srcTxt.size(); openIdx < e; ++openIdx) {
      if (std::isspace(srcTxt[openIdx]))
        continue;
      if (srcTxt[openIdx] != '(')
        break;

      // We found the opening paranthesis, now find the next closing parenthesis
      // which indicates the end of arguments
      size_t closeIdx = srcTxt.find(")", openIdx);
      if (closeIdx == std::string::npos)
        break;

      std::string argList = srcTxt.substr(openIdx + 1, closeIdx - openIdx - 1);

      // Now split the argument list on commas to get each individual argument
      size_t pos;
      SmallVector<std::string> args;
      while ((pos = argList.find(",")) != std::string::npos) {
        args.push_back(argList.substr(0, pos));
        argList.erase(0, pos + 1);
      }
      args.push_back(argList);

      if (args.size() != funcOp.getNumArguments() - 1)
        break;

      // Now identify the name of each argument
      OpBuilder builder(&getContext());
      SmallVector<Attribute> argNames;
      for (StringRef argStr : args) {
        // Split the arg string into tokens; the last token contains the name
        std::istringstream iss(argStr.str());
        std::string tok;
        SmallVector<std::string> tokens;
        while (iss >> tok)
          tokens.push_back(tok);
        if (tokens.size() < 2)
          break;

        // Remove array brackets if necessary
        std::string name = tokens.back();
        if (size_t bracket = name.find("["); bracket != std::string::npos)
          name = name.substr(0, bracket);
        argNames.push_back(StringAttr::get(&getContext(), name));
      }
      // The last argument (added during cf-to-Handshake lowering) is always
      // called start
      argNames.push_back(StringAttr::get(&getContext(), "start"));

      // Replace the argument's name
      funcOp->setAttr("argNames", ArrayAttr::get(&getContext(), argNames));
      return success();
    }
    startIdx = openIdx;
  } while (true);

  return failure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::createHandshakeFixArgNames(const std::string &source) {
  return std::make_unique<HandshakeFixArgNamesPass>(source);
}
