//===- constant.cpp - Fulfill a constant generator --------------*- C++ -*-===//
//
// Experimental tool that realizes a generator for constant component
// (handshake.constant)
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/EpochTracker.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemAlloc.h"
#include "llvm/Support/ReverseIteration.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/type_traits.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace llvm;

static const std::string CONSTANT_PATH =
    "experimental/data/vhdl/handshake/constant.vhd";

int main(int argc, char **argv) {
  // no value provided
  if (argc < 2) {
    llvm::errs() << "Too few arguments in generator call\n";
    return 1;
  }

  // read as file
  std::ifstream file;
  file.open(CONSTANT_PATH);

  if (!file.is_open()) {
    llvm::errs() << "Filepath is uncorrect\n";
    return 1;
  }

  // get the predicate name from command line options
  std::string predicateName(argv[1]);
  size_t num = std::atoi(predicateName.c_str());
  std::stringstream toHex;
  toHex << std::hex << num;
  std::stringstream buffer;
  std::string modText;
  buffer << file.rdbuf();
  std::string temp;
  // replace all #---# sequences
  while (buffer.good()) {
    std::getline(buffer, temp, '#');
    modText += temp;
    if (!buffer.good())
      break;

    std::getline(buffer, temp, '#');
    if (temp == "CST_VALUE")
      modText += "x\"" + toHex.str() + "\"";
    else if (temp == "CST_NAME")
      modText += toHex.str();
  }

  // print the result module text to std output
  llvm::outs() << modText;
  return 0;
}
