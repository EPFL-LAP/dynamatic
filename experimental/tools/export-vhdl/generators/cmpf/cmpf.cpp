//===- cmpf.cpp - Fulfill a cmpf generator ----------------------*- C++ -*-===//
//
// Experimental tool that realizes a generator for float comparison component
// (cmpf)
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

#define CMPF_PATH "experimental/data/vhdl/arithmetic/cmpf.vhd"

int main(int argc, char **argv) {
  // no predicate provided
  if (argc < 2) {
    llvm::errs() << "Too few arguments in generator call\n";
    return 1;
  }
  // map which stores corresponding data for predicates
  StringMap<std::string> data = {
      {"oeq", "00001"}, {"ogt", "00010"}, {"oge", "00011"}, {"olt", "00100"},
      {"ole", "00101"}, {"one", "00110"}, {"ord", "00111"}, {"ueq", "01000"},
      {"ugt", "01001"}, {"uge", "01010"}, {"ult", "01011"}, {"ule", "01100"},
      {"une", "01101"}, {"uno", "01110"}};

  // read as file
  std::ifstream file;
  file.open(CMPF_PATH);

  if (!file.is_open()) {
    llvm::errs() << "Filepath is uncorrect\n";
    return 1;
  }

  // get the predicate name from command line options
  std::string predicateName(argv[1]);
  std::stringstream buffer;
  std::string modText;
  auto it = data.find(predicateName);
  if (it != data.end()) {
    // oeq, ogt, oge, olt, ole, one, ord, ueq, ugt, uge, ult, ule, une, uno
    std::string temp;
    buffer << file.rdbuf();
    // replace all #---# sequences
    while (buffer.good()) {
      std::getline(buffer, temp, '#');
      modText += temp;
      if (!buffer.good())
        break;

      std::getline(buffer, temp, '#');
      if (temp == "PREDICATE")
        modText += it->getKey().str();
      else
        modText += it->getValue();
    }
  } else {
    // wrong predicate
    llvm::errs() << "Wrong predicate\n";
    return 1;
  }
  // print the result module text to std output
  llvm::outs() << modText;
  return 0;
}
