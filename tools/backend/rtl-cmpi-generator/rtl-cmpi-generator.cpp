//===- rtl-cmpi-generator.cpp - Generator for arith.cmpi --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL generator for the `arith.cmpi` MLIR operation. Generates the correct RTL
// based on the integer comparison predicate.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/RTL.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <fstream>
#include <map>

using namespace llvm;
using namespace mlir;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputRTLPath(cl::Positional, cl::Required,
                                         cl::desc("<input file>"),
                                         cl::cat(mainCategory));

static cl::opt<std::string> outputRTLPath(cl::Positional, cl::Required,
                                          cl::desc("<output file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> entityName(cl::Positional, cl::Required,
                                       cl::desc("<entity name>"),
                                       cl::cat(mainCategory));

static cl::opt<std::string>
    predicate(cl::Positional, cl::Required,
              cl::desc("<integer comparison predicate>"),
              cl::cat(mainCategory));

static cl::opt<std::string> hdlType(cl::Positional, cl::Required,
                                       cl::desc("<hdl type>"),
                                       cl::cat(mainCategory));

/// Returns the VHDL comparator corresponding to the comparison's predicate.
static StringRef getComparatorVHDL(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return "=";
  case arith::CmpIPredicate::ne:
    return "/=";
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return "<";
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return "<=";
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return ">";
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return ">=";
  }
}

static StringRef getComparatorVerilog(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return "==";
  case arith::CmpIPredicate::ne:
    return "!=";
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return "<";
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return "<=";
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return ">";
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return ">=";
  }
}

/// Returns the VHDL type modifier associated with the comparison's predicate.
static StringRef getModifierVHDL(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::ne:
    return "";
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::sge:
    return "signed";
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::uge:
    return "unsigned";
  }
}

/// Returns the Verilog type modifier associated with the comparison's predicate.
static StringRef getModifierVerilog(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::ne:
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::uge:
    return "";
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::sge:
    return "$signed";
  }
}


int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "RTL generator for the `arith.cmpi` MLIR operation. Generates the "
      "correct RTL based on the integer comparison predicate.");

  std::optional<arith::CmpIPredicate> pred =
      arith::symbolizeCmpIPredicate(predicate);
  if (!pred) {
    llvm::errs() << "Unknown integer comparison predicate \"" << predicate
                 << "\"\n";
    return 1;
  }

  // Open the input file
  std::ifstream inputFile(inputRTLPath);
  if (!inputFile.is_open()) {
    llvm::errs() << "Failed to open input file @ \"" << inputRTLPath << "\"\n";
    return 1;
  }

  // Open the output file
  std::ofstream outputFile(outputRTLPath);
  if (!outputFile.is_open()) {
    llvm::errs() << "Failed to open output file @ \"" << outputRTLPath
                 << "\"\n";
    return 1;
  }

  // Read the JSON content from the file and into a string
  std::string inputData;
  std::string line;
  while (std::getline(inputFile, line))
    inputData += line + "\n";

  // Record all replacements in a map
  std::map<std::string, std::string> replacementMap;
  replacementMap["ENTITY_NAME"] = entityName;
  
  if(hdlType == "vhdl") {
    replacementMap["COMPARATOR"] = getComparatorVHDL(*pred);
    replacementMap["MODIFIER"] = getModifierVHDL(*pred);
  }
  else if (hdlType == "verilog") {
    replacementMap["COMPARATOR"] = getComparatorVerilog(*pred);
    replacementMap["MODIFIER"] = getModifierVerilog(*pred);
  }
  else {
    llvm::errs() << "Unknown HDL type \"" << hdlType << "\"\n";
    return 1;
  }

  // Dump to the output file and return
  outputFile << dynamatic::replaceRegexes(inputData, replacementMap);
  return 0;
}
