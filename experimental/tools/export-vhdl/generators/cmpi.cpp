//===- cmpi.cpp - Fulfill a cmpi generator ------*- C++ -*-===//
//
// Experimental tool that realizes a generator for integer comparison component
// (cmpi)
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/JSON.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
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
using namespace mlir;
using namespace circt;

int main(int argc, char **argv) {
  // no predicate provided
  if (argc < 2) {
    llvm::errs() << "Too few arguments in generator call\n";
    return 1;
  }
  // maps which store corresponding data for predicates
  StringMap<std::pair<std::string, std::string>> equality = {
      {"eq", {"one", "zero"}}, {"ne", {"zero", "one"}}};
  StringMap<std::string> comparison = {
      {"sge", ">="}, {"sgt", ">"}, {"sle", "<="}, {"slt", "<"},
      {"uge", ">="}, {"ugt", ">"}, {"ule", "<="}, {"ult", "<"}};

  // get the predicate name from command line options
  std::string predicateName(argv[1]);
  std::ifstream file;
  std::stringstream buffer;
  std::string modText;
  auto it = comparison.find(predicateName);
  if (it == comparison.end()) {
    auto jt = equality.find(predicateName);
    if (jt == equality.end()) {
      // wrong predicate
      llvm::errs() << "Wrong predicate\n";
      return 1;
    } else {
      // ne, eq
      file.open("experimental/data/vhdl/arithmetic/cmpi/equality.vhd");
      buffer << file.rdbuf();
      std::string temp{};
      // replace all #---# sequences
      while (buffer.good()) {
        std::getline(buffer, temp, '#');
        modText += temp;
        if (!buffer.good()) {
          break;
        }
        std::getline(buffer, temp, '#');
        if (temp == "NAME")
          modText += jt->getKey().str();
        else if (temp == "CONDTRUE")
          modText += jt->getValue().first;
        else
          modText += jt->getValue().second;
      }
    }
  } else {
    // sge, sgt, sle, slt, uge, ugt, ule, ult
    if (predicateName == "sge" || predicateName == "sgt" ||
        predicateName == "sle" || predicateName == "slt") {
      file.open("experimental/data/vhdl/arithmetic/cmpi/signed.vhd");
    } else
      file.open("experimental/data/vhdl/arithmetic/cmpi/unsigned.vhd");
    buffer << file.rdbuf();
    std::string temp{};
    // replace all #---# sequences
    while (buffer.good()) {
      std::getline(buffer, temp, '#');
      modText += temp;
      if (!buffer.good())
        break;

      std::getline(buffer, temp, '#');
      if (temp == "NAME")
        modText += it->getKey().str();
      else
        modText += it->getValue();
    }
  }
  // print the result module text to std output
  llvm::outs() << modText;
  return 0;
}