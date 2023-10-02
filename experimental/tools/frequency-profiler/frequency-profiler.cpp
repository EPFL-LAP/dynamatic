//===- frequency-profiler.cpp - Profile std-level code ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tool inherited from CIRCT which executes a restricted form of the standard
// dialect and profile the IR by counting the number of transitions between
// basic blocks for a provided set of inputs. The tool prints transition
// frequencies between basic blocks on standard output, either in a CSV (for
// Dynamatic buffer placement) or DOT (for legacy Dynamatic buffer placement)
// format.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/StdProfiler.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

#include <cstdlib>
#include <fstream>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

/// Simulates a std-level function on a specific set of inputs.
mlir::LogicalResult simulate(func::FuncOp funcOp,
                             ArrayRef<std::string> inputArgs,
                             StdProfiler &prof);

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> fileArgs(
    "input-args-file", cl::Optional,
    cl::desc("If provided, the tool will fetch argument values from a file "
             "instead of from the command line"),
    cl::init(""), cl::cat(mainCategory));

static cl::opt<bool>
    printDot("print-dot", cl::Optional,
             cl::desc("If provided, the tool formats its output as a "
                      "legacy-compatible DOT instead of a CSV"),
             cl::init(false), cl::cat(mainCategory));

static cl::list<std::string> clArgs(cl::Positional, cl::desc("<input args>"),
                                    cl::ZeroOrMore, cl::cat(mainCategory));

static cl::opt<std::string>
    toplevelFunction("top-level-function", cl::Optional,
                     cl::desc("The top-level function to execute"),
                     cl::init("main"), cl::cat(mainCategory));

/// Reads argument from a file instead of from the command line.
static SmallVector<std::string> fetchArgsFromFile() {
  SmallVector<std::string> args;

  // Check that no argument were provided on the command line
  if (!clArgs.empty()) {
    errs() << "Input arguments file and command-line arguments were both "
              "provided, use one or the other";
    exit(1);
  }

  // Try to open the file containing arguments
  std::ifstream infile(fileArgs);
  if (!infile.is_open()) {
    errs() << "Failed to open arguments file at " << fileArgs;
    exit(1);
  }

  // Read file line by line (each line represents an argument value)
  std::string line;
  while (std::getline(infile, line))
    args.push_back(line);

  infile.close();
  return args;
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "MLIR Standard dialect runner\n\n"
      "This application executes a function in the given MLIR module\n"
      "Arguments to the function are passed on the command line and\n"
      "results are returned on stdout.\n"
      "Memref types are specified as a comma-separated list of values.\n");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    errs() << argv[0] << ": could not open input file '" << inputFileName
           << "': " << error.message() << "\n";
    return 1;
  }

  // Functions feeding into HLS tools might have attributes from high(er) level
  // dialects or parsers. Allow unregistered dialects to not fail in these
  // cases
  MLIRContext context;
  context.loadDialect<func::FuncDialect, memref::MemRefDialect,
                      LLVM::LLVMDialect>();
  context.allowUnregisteredDialects();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!module)
    return 1;

  // Try to find a function with the specified name
  auto funcOp = module->lookupSymbol<func::FuncOp>(toplevelFunction);
  if (!funcOp) {
    errs() << "Top-level function " << toplevelFunction << " not found!\n";
    return 1;
  }

  // Run the std-level simulator
  dynamatic::experimental::StdProfiler prof(funcOp);
  bool simFailed = false;
  if (fileArgs.empty())
    simFailed = failed(simulate(funcOp, clArgs, prof));
  else
    simFailed = failed(simulate(funcOp, fetchArgsFromFile(), prof));

  // Print statistics to stdout and return
  if (!simFailed)
    prof.writeStats(printDot);
  return int(simFailed);
}
