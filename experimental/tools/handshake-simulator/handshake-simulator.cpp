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

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/JSON.h"
#include "circt/Support/Version.h"
#include "experimental/tools/handshake-simulator/Simulation.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <fstream>

#define CONFIG_PATH                                                            \
  "../experimental/tools/handshake-simulator/model-configuration.json"

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
                                "(--change-model <OpName> <ExecFunction>)"),
                       cl::multi_val(2), cl::ZeroOrMore, cl::Optional,
                       cl::cat(configCategory));

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

  // Load JSON model configuration
  std::ifstream f;
  f.open(CONFIG_PATH); 

  std::stringstream buffer;
  buffer << f.rdbuf(); 
  std::string jsonStr = buffer.str(); 

  auto jsonConfig = llvm::json::parse(StringRef(jsonStr));

  if (!jsonConfig) {
    errs() << "Configuration JSON could not be parsed" << "\n";
    return 1;
  }

  if (!jsonConfig->getAsObject()) {
    errs() << "Configuration JSON is not a valid JSON" << "\n";
    return 1;
  }

  // Change the JSON model configuration if required
  if (modelConfiguration.getNumOccurrences() > 0) {
    llvm::json::Object* jsonObjectPtr = jsonConfig->getAsObject();
    std::string opToChange = modelConfiguration[0];
    std::string modelName = modelConfiguration[1];

    if (!jsonObjectPtr->getString(opToChange)) {
      errs() << opToChange << " : OP could not be found" << "\n";
      return 1;
    }
    
    std::error_code errorCode;
    llvm::raw_fd_ostream jsonOs(CONFIG_PATH, errorCode);
    if (errorCode) {
      errs() << "Configuration JSON could not be loaded : " 
             << errorCode.message() << "\n";
      return 1;
    }

    (*jsonObjectPtr)[opToChange] = modelName;
    jsonOs << json::Value(std::move(*jsonObjectPtr));
    jsonOs.close();
    errs() << opToChange << " execution model changed to '" << modelName 
           << "'\n";

    return 1;
  }

  auto file_or_err = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = file_or_err.getError()) {
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

  SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(*file_or_err), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<ModuleOp>(source_mgr, &context));
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
                                           context, jsonConfig.get());
}
