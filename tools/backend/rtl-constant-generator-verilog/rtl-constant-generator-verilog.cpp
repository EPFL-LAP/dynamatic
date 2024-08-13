//===- rtl-text-generator.cpp - Text-based RTL generator --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple generator for RTL components, which takes as input an RTL file,
// replaces user-provided strings (with regex support) within it, and dumps the
// result at a specified location.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/RTL/RTL.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <fstream>
#include <map>

using namespace llvm;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputRTLPath(cl::Positional, cl::Required,
                                         cl::desc("<input file>"),
                                         cl::cat(mainCategory));

static cl::opt<std::string> outputRTLPath(cl::Positional, cl::Required,
                                          cl::desc("<output file>"),
                                          cl::cat(mainCategory));

static cl::list<std::string> replacements(
    cl::Positional, cl::ZeroOrMore,
    cl::desc(
        "<text replacements, two-by-two (text to replace, then replacement)>"),
    cl::cat(mainCategory));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Simple generator for RTL components, which takes as input an RTL file, "
      "replaces user-provided string within it, and dumps the result at a "
      "specified location.");

  // It only makes sense to have an even number of "replacement" arguments
  // because they go two-by-two
  if ((replacements.size() & 1) != 0) {
    llvm::errs() << "Expected an even number of replacement parameters, got "
                 << replacements.size() << "\n";
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
  for (size_t i = 0, e = replacements.size(); i < e; i += 2) {
    if (replacements[i].compare("VALUE") == 0) {
      StringRef value = replacements[i + 1]; // verilog does not accept constant
                                             // values in string format
      replacementMap[replacements[i]] =
          std::to_string(value.size()) + "\'b" + value.data();
    } else {
      replacementMap[replacements[i]] = replacements[i + 1];
    }
  }

  // Dump to the output file and return
  outputFile << dynamatic::replaceRegexes(inputData, replacementMap);
  return 0;
}
