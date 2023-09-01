//===- cmpi.cpp - Fulfill a cmpi generator ----------------------*- C++ -*-===//
//
// Experimental tool that realizes a generator for integer comparison component
// (cmpi)
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

static const std::string EQUALITY_PATH =
    "experimental/data/vhdl/arithmetic/cmpi/equality.vhd";
static const std::string SIGNED_PATH =
    "experimental/data/vhdl/arithmetic/cmpi/signed.vhd";
static const std::string UNSIGNED_PATH =
    "experimental/data/vhdl/arithmetic/cmpi/unsigned.vhd";

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

  std::ifstream file;
  // get the predicate name from command line options
  std::string predicateName(argv[1]);
  std::stringstream buffer;
  std::string modText;
  auto it = comparison.find(predicateName);
  if (it == comparison.end()) {
    auto jt = equality.find(predicateName);
    if (jt != equality.end()) {
      // ne, eq
      file.open(EQUALITY_PATH);
      if (!file.is_open()) {
        llvm::errs() << "Filepath is uncorrect\n";
        return 1;
      }
      buffer << file.rdbuf();
      std::string temp;
      // replace all #---# sequences
      while (buffer.good()) {
        std::getline(buffer, temp, '#');
        modText += temp;
        if (!buffer.good()) {
          break;
        }
        std::getline(buffer, temp, '#');
        if (temp == "PREDICATE")
          modText += jt->getKey().str();
        else if (temp == "CONDTRUE")
          modText += jt->getValue().first;
        else
          modText += jt->getValue().second;
      }
    } else {
      // wrong predicate
      llvm::errs() << "Wrong predicate\n";
      return 1;
    }
  } else {
    // sge, sgt, sle, slt, uge, ugt, ule, ult
    if (predicateName == "sge" || predicateName == "sgt" ||
        predicateName == "sle" || predicateName == "slt") {
      file.open(SIGNED_PATH);
      if (!file.is_open()) {
        llvm::errs() << "Filepath is uncorrect\n";
        return 1;
      }
      buffer << file.rdbuf();
    } else {
      file.open(UNSIGNED_PATH);
      if (!file.is_open()) {
        llvm::errs() << "Filepath is uncorrect\n";
        return 1;
      }
      buffer << file.rdbuf();
    }
    std::string temp;
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
  }
  // print the result module text to std output
  llvm::outs() << modText;
  return 0;
}
