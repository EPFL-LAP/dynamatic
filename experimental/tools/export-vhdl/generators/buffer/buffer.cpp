//===- buffer.cpp - Fulfill a constant generator --------------*- C++ -*-===//
//
// Experimental tool that realizes a generator for buffer component
// (handshake.buffer)
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

static const std::string seq_path =
    "experimental/data/vhdl/handshake/buffer/seq.vhd";
static const std::string fifo_path =
    "experimental/data/vhdl/handshake/buffer/fifo.vhd";

int main(int argc, char **argv) {
  // no value provided
  if (argc < 2) {
    llvm::errs() << "Too few arguments in generator call\n";
    return 1;
  }
  std::ifstream file;

  // get the predicate name from command line options
  std::string predicateName(argv[1]);
  std::stringstream buffer;
  std::string modText;

  if (predicateName == "fifo") {
    // fifo
    file.open(fifo_path);
    if (!file.is_open()) {
      llvm::errs() << "Filepath is uncorrect\n";
      return 1;
    }
  } else if (predicateName == "seq") {
    // seq
    file.open(seq_path);
    if (!file.is_open()) {
      llvm::errs() << "Filepath is uncorrect\n";
      return 1;
    }
  } else {
    // wrong predicate
    llvm::errs() << "Wrong predicate\n";
    return 1;
  }

  buffer << file.rdbuf();
  modText = buffer.str();

  // print the result module text to std output
  llvm::outs() << modText;
  return 0;
}
