//===----------- rigidification-testbench.cpp -------------------*-C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file generates a testbench for formal verification.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/HW/HWDialect.h"
#include "dynamatic/Dialect/HW/HWOpInterfaces.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include <filesystem>
#include <fstream>
#include <string>

#include "dynamatic/InitAllDialects.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include "experimental/Support/CreateSmvFormalTestbench.h"

namespace cl = llvm::cl;
using namespace mlir;
using namespace dynamatic;

// CLI Settings

static cl::OptionCategory generalCategory("Testbench options");

static cl::opt<std::string>
    modelPathArg("i", cl::Prefix, cl::Required,
                 cl::desc("Path to the SMV model to check"));
static cl::opt<std::string> mlirPathArg("mlir", cl::Prefix, cl::Required,
                                        cl::desc("Path to the MLIR file"));

static cl::opt<std::string> kernelNameArg("name", cl::Prefix, cl::Required,
                                          cl::desc("Kernel name"));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      R"DELIM(Generates a testbench for formal verification. It
instantiates the module under test and provides valid inputs to it.

Usage Example:
testbench-generator -i integration-tests/fir/out/hdl/ --name fir -o
integration-tests/fir/out/hdl/ --mlir intregation-tests/fir/out/comp/hw.mlir/n)DELIM");

  // Register the supported dynamatic dialects and create a context
  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  MLIRContext context(registry);

  std::string kernelName = kernelNameArg.getValue();
  std::filesystem::path mlirPath = mlirPathArg.getValue();
  std::filesystem::path modelPath = modelPathArg.getValue();
  std::filesystem::path wrapperPath = modelPath / "main.smv";

  // Create the outputDir if it doesn't exist
  std::filesystem::create_directories(modelPath);

  OwningOpRef<ModuleOp> moduleRef =
      parseSourceFile<ModuleOp>(mlirPath.string(), &context);
  if (!moduleRef) {
    llvm::errs() << "Failed to load module.\n";
  }

  SmallVector<std::pair<std::string, mlir::Type>> args;
  SmallVector<std::pair<std::string, mlir::Type>> res;
  for (hw::HWModuleOp hwModOp : moduleRef.get().getOps<hw::HWModuleOp>()) {
    if (hwModOp.getSymName() == kernelName) {

      auto inTypes = hwModOp.getInputTypes();
      auto inNames = hwModOp.getInputNamesStr();
      auto outTypes = hwModOp.getOutputTypes();
      auto outNames = hwModOp.getOutputNamesStr();

      for (size_t i = 0; i < inTypes.size(); ++i) {
        args.emplace_back(inNames[i].str(), inTypes[i]);
      }

      for (size_t i = 0; i < outTypes.size(); ++i) {
        res.emplace_back(outNames[i].str(), outTypes[i]);
      }
    }
    const dynamatic::experimental::SmvTestbenchConfig smvConfig = {
        .arguments = args,
        .results = res,
        .modelSmvName = kernelName,
        .nrOfTokens = 1,
        .generateExactNrOfTokens = false,
        .syncOutput = true};

    std::string wrapper =
        dynamatic::experimental::createSmvFormalTestbench(smvConfig);
    std::ofstream mainFile(wrapperPath);
    mainFile << wrapper;
    mainFile.close();
  }

  exit(false);
}