//===- handshake-simulator.cpp - Simulate Handshake-level code --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tool inherited from CIRCT which executes a restricted form of Handshake-level
// IR.
//
//===----------------------------------------------------------------------===//

#include "experimental/tools/handshake-simulator/Simulation.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/JSON.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"


#include <fstream>

#define DEFAULT_CONFIG_PATH                                                            \
  "../experimental/data/handshake-simulator-configuration.json"

using namespace llvm;
using namespace mlir;
using namespace circt;

static cl::OptionCategory mainCategory("Application options");
static cl::OptionCategory configCategory("Configuration options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"), cl::cat(mainCategory));

static cl::list<std::string> inputArgs(cl::Positional, cl::desc("<input args>"),
                                       cl::ZeroOrMore, cl::cat(mainCategory));

static cl::opt<std::string>
    toplevelFunction("top-level-function", cl::Optional,
                     cl::desc("The top-level function to execute"),
                     cl::init("main"), cl::cat(mainCategory));

static cl::list<std::string>
    modelConfiguration("change-model",
                       cl::desc("Change execution model function "
                                "(--change-model <op name> <struct name>).\n"
                                "This does not affect the configuration file."),
                       cl::multi_val(2), cl::ZeroOrMore, cl::Optional,
                       cl::cat(configCategory));

static cl::opt<std::string> 
    jsonSelection("config",
                  cl::desc("Change the configuration file path.\n"
                           "Must be a relative path."),
                  cl::Optional, cl::cat(configCategory));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  cl::ParseCommandLineOptions(
      argc, argv,
      "MLIR Standard dialect runner\n\n"
      "This application executes a function in the given MLIR module\n"
      "Arguments to the function are passed on the command line and\n"
      "results are returned on stdout.\n"
      "Memref types are specified as a comma-separated list of values.\n");

  // Change JSON path if needed
  std::string configPath = DEFAULT_CONFIG_PATH;
  if (jsonSelection.getNumOccurrences() == 1) {
    configPath = jsonSelection.getValue();
    errs() << " configuration file changed to " << configPath << "\n";

    // Return if the command is entered alone
    if (inputFileName.getNumOccurrences() == 0)
      return 1;
  }

  // Load JSON model configuration
  std::ifstream f;
  f.open(configPath); 

  std::stringstream buffer;
  buffer << f.rdbuf(); 
  std::string jsonStr = buffer.str(); 
  f.close();

  auto jsonConfig = llvm::json::parse(StringRef(jsonStr));

  if (!jsonConfig) {
    errs() << "Configuration JSON could not be parsed" << "\n";
    return 1;
  }

  if (!jsonConfig->getAsObject()) {
    errs() << "Configuration JSON is not a valid JSON" << "\n";
    return 1;
  }

  // JSON to map conversion
  llvm::StringMap<std::string> modelConfigMap;
  for (auto item : *jsonConfig->getAsObject()) {
    modelConfigMap.insert(std::make_pair(
      item.getFirst().str(),
      item.getSecond().getAsString().value().str()
    ));
  }

  // Change the model configuration if the command is entered,
  // without changing the JSON file
  int nbChangedPair = modelConfiguration.getNumOccurrences() * 2;
  if (nbChangedPair > 0) {
    for (int i = 0; i < nbChangedPair; i += 2) {
      std::string opToChange = modelConfiguration[i];
      std::string modelName = modelConfiguration[i + 1];
      modelConfigMap[opToChange] = modelName;
      errs() << opToChange << " execution model changed to '" << modelName 
             << "'\n";
    }
    return 1;
  }

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    errs() << argv[0] << ": could not open input file '" << inputFileName
           << "': " << error.message() << "\n";
    return 1;
  }

  // Load the MLIR module.
  mlir::MLIRContext context;
  context.loadDialect<func::FuncDialect, memref::MemRefDialect,
                      handshake::HandshakeDialect>();

  // functions feeding into HLS tools might have attributes from high(er) level
  // dialects or parsers. Allow unregistered dialects to not fail in these
  // cases.
  context.allowUnregisteredDialects();

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!module)
    return 1;

  mlir::Operation *mainP = module->lookupSymbol(toplevelFunction);
  // The toplevel function can accept any number of operands, and returns
  // any number of results.
  if (!mainP) {
    errs() << "Top-level function " << toplevelFunction << " not found!\n";
    return 1;
  }

  return dynamatic::experimental::simulate(toplevelFunction, inputArgs, module, 
                                           context, modelConfigMap);
}
