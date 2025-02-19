//===- elastic-miter.cpp - The elastic-miter driver -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO This file implements the elastic-miter tool, it creates an elastic miter
// circuit, which can later be used to formally verify equivalence of two
// handshake circuits.
//
//===----------------------------------------------------------------------===//

#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "dynamatic/InitAllDialects.h"
#include "llvm/Support/InitLLVM.h"

#include "../experimental/tools/elastic-miter-generator/CreateWrappers.h"
#include "../experimental/tools/elastic-miter-generator/ElasticMiterFabricGeneration.h"
#include "../experimental/tools/elastic-miter-generator/GetSequenceLength.h"
#include "../experimental/tools/elastic-miter-generator/SmvUtils.h"

namespace cl = llvm::cl;
using namespace mlir;
using namespace dynamatic::handshake;

// CLI Settings

static cl::OptionCategory mainCategory("elastic-miter Options");

static cl::opt<std::string>
    lhsFilenameArg("lhs", cl::Prefix, cl::Required,
                   cl::desc("Specify the left-hand side (LHS) input file"),
                   cl::cat(mainCategory));
static cl::opt<std::string>
    rhsFilenameArg("rhs", cl::Prefix, cl::Required,
                   cl::desc("Specify the right-hand side (RHS) input file"),
                   cl::cat(mainCategory));

static cl::opt<std::string> outputDirArg("o", cl::Prefix, cl::Required,
                                         cl::desc("Specify output directory"),
                                         cl::cat(mainCategory));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Checks the equivalence of two dynamic circuits in the handshake "
      "dialect. At the end it will output whether the circuits are "
      "latency-insensitive equivalent."
      "Takes two MLIR files as input. The files need to contain exactely one "
      "module each.\nEach module needs to contain exactely one "
      "handshake.func. "
      "\nThe resulting miter MLIR file and JSON config file are placed in "
      "the specified output directory.");

  // Register the supported dynamatic dialects and create a context
  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  MLIRContext context(registry);

  std::filesystem::path lhsPath = lhsFilenameArg.getValue();
  std::filesystem::path rhsPath = rhsFilenameArg.getValue();
  std::filesystem::path outputDir = outputDirArg.getValue();

  // Create the outputDir if it doesn't exist
  std::filesystem::create_directories(outputDir);

  // Find out needed number of tokens
  auto failOrLHSseqLen = dynamatic::experimental::getSequenceLength(
      context, outputDir / "lhs_reachability", lhsPath);
  if (failed(failOrLHSseqLen))
    return 1;

  // TODO remove
  llvm::outs() << "The LHS needs " << failOrLHSseqLen.value() << " tokens.\n";

  auto failOrRHSseqLen = dynamatic::experimental::getSequenceLength(
      context, outputDir / "rhs_reachability", rhsPath);
  if (failed(failOrRHSseqLen))
    return 1;

  // TODO remove
  llvm::outs() << "The RHS needs " << failOrRHSseqLen.value() << " tokens.\n";

  size_t n = std::max(failOrLHSseqLen.value(), failOrRHSseqLen.value());

  std::filesystem::path miterDir = outputDir / "miter";
  // Create the miterDir if it doesn't exist
  std::filesystem::create_directories(miterDir);

  // Create Miter module with needed N
  auto failOrPair = dynamatic::experimental::createMiterFabric(
      context, lhsPath, rhsPath, miterDir.string(), n);
  if (failed(failOrPair)) {
    llvm::errs() << "Failed to create elastic-miter module.\n";
    return 1;
  }
  auto [mlirPath, config] = failOrPair.value();

  auto failOrSmvPath =
      dynamatic::experimental::handshake2smv(mlirPath, miterDir, true);
  if (failed(failOrSmvPath)) {
    llvm::errs() << "Failed to convert miter module to SMV.\n";
    return 1;
  }
  auto smvPath = failOrSmvPath.value();

  // TODO ...
  std::filesystem::path wrapperPath = miterDir / "main.smv";

  // Create wrapper (main) for the elastic-miter
  // Currently handshake2smv only supports "model" as the model's name
  auto fail = dynamatic::experimental::createWrapper(wrapperPath, config,
                                                     "model", n, true);
  if (failed(fail))
    return 1;

  // Put the output of the CTLSPEC check into results.txt. Later we read from
  // that file to check whether all the CTL properties pass.
  std::filesystem::path resultTxtPath = miterDir / "result.txt";
  std::string miterCommand = "check_invar -s forward;\n"
                             "show_traces -a -p 4 -o ~/trace.xml;\n"
                             "check_ctlspec;\n";
  LogicalResult cmdFail = dynamatic::experimental::createCMDfile(
      miterDir / "prove.cmd", miterDir / "main.smv", miterCommand);
  if (failed(cmdFail))
    return 1;

  // Run equivalence checking
  dynamatic::experimental::runNuXmv(miterDir / "prove.cmd", resultTxtPath);

  bool equivalent = true;
  std::string line;
  std::ifstream result(resultTxtPath);
  while (getline(result, line)) {
    // TODO remove
    llvm::outs() << line << "\n";
    if (line.find("is false") != std::string::npos) {
      equivalent = false;
    }
  }
  result.close();
  // TODO print?

  exit(!equivalent);
}