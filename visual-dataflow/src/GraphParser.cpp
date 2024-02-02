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
#include "DOTReformat.h"
#include "Graph.h"
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
#include <cstdio>
#include <fstream>
#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::visual_dataflow;

GraphParser::GraphParser(Graph *graph) : mGraph(graph) {}

LogicalResult GraphParser::parse(std::string &filePath) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    return failure();
  }

  std::string line;
  size_t lineIndex = 0;
  CycleNb currCycle = 0;

  if (filePath.find(".csv") != std::string::npos) {
    while (std::getline(file, line)) {
      if (failed(processCSVLine(line, lineIndex, *mGraph, &currCycle)))
        return failure();
      lineIndex++;
    }
  } else if (filePath.find(".dot") != std::string::npos) {
    const std::string outputDotFile = "/tmp/graph.dot";

    if (failed(reformatDot(filePath, outputDotFile)))
      return failure();

    std::ifstream f;
    f.open(outputDotFile);

    if (failed(processDOT(f, *mGraph)))
      return failure();
  } else if (filePath.find(".mlir") != std::string::npos) {
    auto fileOrErr = MemoryBuffer::getFileOrSTDIN(filePath.c_str());
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
    MLIRMapper mapper(mGraph);
    if (failed(mapper.mapMLIR(*module)))
      return failure();
  }

  file.close();
  return success();
}
