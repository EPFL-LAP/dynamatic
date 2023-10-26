#include "GraphParser.h"
#include "CSVParser.h"
#include "DOTParser.h"
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
      // processLine(line, graph, lineIndex);
      lineIndex++;
    }
  } else if (mFilePath.find(".dot") != std::string::npos) {
    // processDOT(file, graph);
  }

  file.close();
  return success();
}