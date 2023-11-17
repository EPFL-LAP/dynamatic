//===- GraphParser.cpp - Parse a dataflow graph -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Graph parsing.
//
//===----------------------------------------------------------------------===//

#include "GraphParser.h"
#include "CSVParser.h"
#include "DOTParser.h"
#include "MLIRMapper.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <fstream>
#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::visual_dataflow;

GraphParser::GraphParser(std::string filePath)
    : mFilePath(std::move(filePath)) {}

LogicalResult GraphParser::parse(Graph *graph) {
  std::ifstream file(mFilePath);
  if (!file.is_open()) {
    return failure();
  }

  std::string line;
  size_t lineIndex = 0;

  if (mFilePath.find(".csv") != std::string::npos) {
    while (std::getline(file, line)) {
      if (failed(processCSVLine(line, lineIndex, *graph)))
        return failure();
      lineIndex++;
    }
  } else if (mFilePath.find(".dot") != std::string::npos) {
    if (failed(processDOT(file, *graph)))
      return failure();
  } else if (mFilePath.find(".mlir") != std::string::npos) {
    auto fileOrErr = MemoryBuffer::getFileOrSTDIN(mFilePath.c_str());
    if (std::error_code error = fileOrErr.getError())
      return failure();

    MLIRContext context;
    context.loadDialect<func::FuncDialect, memref::MemRefDialect,
                        arith::ArithDialect, LLVM::LLVMDialect,
                        handshake::HandshakeDialect>();
    context.allowUnregisteredDialects();

    // Load the MLIR module
    SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module(
        mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
    if (!module)
      return failure();

    // Map the MLIR module to the graph
    MLIRMapper mapper(graph);
    if (failed(mapper.mapMLIR(*module)))
      return failure();
  }

  file.close();
  return success();
}
