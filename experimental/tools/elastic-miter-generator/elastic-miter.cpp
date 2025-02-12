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

#include "llvm/Support/InitLLVM.h"

// TODO rename to CreateWrappers.h
#include "../experimental/tools/elastic-miter-generator/CreateWrappers.h"
#include "../experimental/tools/elastic-miter-generator/ElasticMiterFabricGeneration.h"
#include "../experimental/tools/elastic-miter-generator/GetSequenceLength.h"

#include "dynamatic/InitAllDialects.h"

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

static cl::opt<std::string> outputDir("o", cl::Prefix, cl::Required,
                                      cl::desc("Specify output directory"),
                                      cl::cat(mainCategory));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "TODO Creates an elastic-miter module in the handshake dialect.\n"
      "Takes two MLIR files as input. The files need to contain exactely one "
      "module each.\nEach module needs to contain exactely one "
      "handshake.func. "
      "\nThe resulting miter MLIR file and JSON config file are placed in "
      "the "
      "specified output directory.");

  // Register the supported dynamatic dialects and create a context
  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  MLIRContext context(registry);

  // Find out needed N
  // TODO we should now pass the module instead of the file
  auto failOrLHSseqLen =
      dynamatic::experimental::getSequenceLength(context, lhsFilenameArg);
  if (failed(failOrLHSseqLen))
    return 1;

  llvm::outs() << "The LHS needs " << failOrLHSseqLen.value() << " tokens.\n ";

  auto failOrRHSseqLen =
      dynamatic::experimental::getSequenceLength(context, rhsFilenameArg);
  if (failed(failOrRHSseqLen))
    return 1;

  llvm::outs() << "The RHS needs " << failOrRHSseqLen.value() << " tokens.\n ";

  size_t n = std::max(failOrLHSseqLen.value(), failOrRHSseqLen.value());

  // Create Miter module with needed N
  // TODO wrap this function with the content of the old main function
  // TODO we should now pass the module instead of the file
  auto ret = dynamatic::experimental::createElasticMiter(
      context, lhsFilenameArg, rhsFilenameArg, n);
  if (failed(ret)) {
    llvm::errs() << "Failed to create elastic-miter module.\n";
    return 1;
  }
  auto [miterModule, json] = ret.value();

  // TODO this should be based on funcOp names
  std::string mlirFilename =
      "elastic_miter_" +
      std::filesystem::path(lhsFilenameArg.getValue()).stem().string() + "_" +
      std::filesystem::path(rhsFilenameArg.getValue()).stem().string() +
      ".mlir";
  // TODO also integrate in wrapper
  dynamatic::experimental::createFiles(outputDir, mlirFilename, miterModule,
                                       json);

  // TODO ...
  auto fail = dynamatic::experimental::handshake2smv(
      outputDir + "/" + mlirFilename, false);

  auto failOrWrapper = dynamatic::experimental::createMiterWrapper(n);
  if (failed(failOrWrapper))
    return 1;

  std::string wrapper = failOrWrapper.value();

  std::ofstream mainFile(outputDir + "/main.smv");
  mainFile << wrapper;
  mainFile.close();
  // TODO use this when creating prove.cmd
  std::string output = outputDir + "/result.txt";
  // Run equivalence checking
  dynamatic::experimental::runNuXmv(
      "experimental/tools/elastic-miter-generator/out/comp/prove.cmd",
      "/dev/null");

  // TODO check if it includes "is false"
  std::string line;
  std::ifstream result(output);
  while (getline(result, line)) {
    llvm::outs() << line << "\n";
  }
  result.close();

  // TODO ...
  exit(failed(success()));
}