//===- rtl-cmpf-generator.cpp - Generator for arith.cmpf --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL generator for the `arith.cmpf` MLIR operation. Generates the correct RTL
// based on the floating comparison predicate.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/RTL.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <fstream>
#include <map>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

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

/// Returns the ALU opcode corresponding to the comparison's predicate.
static StringRef getComparator(handshake::CmpFPredicate pred) {
  switch (pred) {
  case handshake::CmpFPredicate::OEQ:
    return "OEQ";
  case handshake::CmpFPredicate::OGT:
    return "OGT";
  case handshake::CmpFPredicate::OGE:
    return "OGE";
  case handshake::CmpFPredicate::OLT:
    return "OLT";
  case handshake::CmpFPredicate::OLE:
    return "OLE";
  case handshake::CmpFPredicate::ONE:
    return "ONE";
  case handshake::CmpFPredicate::ORD:
    return "ORD";
  case handshake::CmpFPredicate::UEQ:
    return "UEQ";
  case handshake::CmpFPredicate::UGT:
    return "UGT";
  case handshake::CmpFPredicate::UGE:
    return "UGE";
  case handshake::CmpFPredicate::ULT:
    return "ULT";
  case handshake::CmpFPredicate::ULE:
    return "ULE";
  case handshake::CmpFPredicate::UNE:
    return "UNE";
  case handshake::CmpFPredicate::UNO:
    return "UNO";
  default:
    return "";
  }
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "RTL generator for the `arith.cmpf` MLIR operation. Generates the "
      "correct RTL based on the floating comparison predicate.");

  std::optional<handshake::CmpFPredicate> pred =
      handshake::symbolizeCmpFPredicate(predicate);
  if (!pred) {
    llvm::errs() << "Unknown floating comparison predicate \"" << predicate
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
  StringRef cmp = getComparator(*pred);
  if (cmp.empty()) {
    llvm::errs() << "Floating comparison predicate \"" << predicate
                 << "\" is not supported\n";
    return 1;
  }
  replacementMap["COMPARATOR"] = cmp;

  // Dump to the output file and return
  outputFile << dynamatic::replaceRegexes(inputData, replacementMap);
  return 0;
}
