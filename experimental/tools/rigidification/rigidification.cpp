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
#include "experimental/Support/SmvUtils.h"

namespace cl = llvm::cl;
using namespace mlir;
using namespace dynamatic;

// CLI Settings

static cl::OptionCategory generalCategory("Testbench options");

static cl::opt<std::string>
    mlirPathArg("mlir", cl::Prefix, cl::Required,
                cl::desc("Path to the MLIR file to rigidify"),
                cl::cat(generalCategory));

static cl::opt<std::string> kernelNameArg("name", cl::Prefix, cl::Required,
                                          cl::desc("Kernel name"),
                                          cl::cat(generalCategory));

static cl::opt<std::string> workDirArg("w", cl::Prefix, cl::Required,
                                       cl::desc("Specify work directory"),
                                       cl::cat(generalCategory));

void generateTestbench(const std::filesystem::path &mlirPath,
                       const std::filesystem::path &wrapperPath,
                       const std::string &kernelName) {
  // Register the supported dynamatic dialects and create a context
  DialectRegistry registry;
  dynamatic::registerAllDialects(registry);
  MLIRContext context(registry);

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
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Generates a testbench for formal verification."
      "It instantiates the module under test and provides valid"
      "inputs to it."
      "Usage Example:\n"
      "testbench-generator -i integration-tests/fir/out/hdl/ --name fir"
      "-o integration-tests/fir/out/hdl/"
      "--mlir intregation-tests/fir/out/comp/hw.mlir/n");

  std::string kernelName = kernelNameArg.getValue();
  std::filesystem::path mlirPath = mlirPathArg.getValue();
  std::filesystem::path workDir = workDirArg.getValue();

  std::filesystem::path modelPath = workDir / "model";
  std::filesystem::path propertyPath = workDir / "formal_properties.json";
  std::filesystem::path hwPath = workDir / "tmp_hw.mlir";
  std::filesystem::path nuxmvReport = workDir / "property.rpt";
  std::filesystem::path nuxmvCommand = workDir / "prove.cmd";

  std::filesystem::path wrapperPath = modelPath / "main.smv";
  std::filesystem::path modelFilePath = modelPath / (kernelName + ".smv");

  // Create the workDir if it doesn't exist
  std::filesystem::create_directories(workDir);

  // Annotate poperties
  std::string cmd =
      "bin/dynamatic-opt " + mlirPath.string() +
      " --handshake-annotate-properties=json-path=" + propertyPath.string();
  int ret = dynamatic::experimental::executeWithRedirect(cmd, "/dev/null");
  if (ret != 0) {
    llvm::errs() << "Failed to annnotate properties\n";
    exit(true);
  }

  // Lower to HW
  cmd = "bin/dynamatic-opt " + mlirPath.string() + " --lower-handshake-to-hw";
  ret = dynamatic::experimental::executeWithRedirect(cmd, hwPath.string());
  if (ret != 0) {
    llvm::errs() << "Failed to convert to HW\n";
    exit(true);
  }

  // Generate SMV
  std::filesystem::create_directories(modelPath);
  cmd = "bin/export-rtl " + hwPath.string() + " " + modelPath.string() +
        " data/rtl-config-smv.json --hdl smv --property-database " +
        propertyPath.string();
  ret = dynamatic::experimental::execute(cmd);
  if (ret != 0) {
    llvm::errs() << "Failed to export to SMV\n";
    exit(true);
  }

  // Generate the testbench
  generateTestbench(hwPath.string(), wrapperPath.string(), kernelName);

  // Run nuXmv
  cmd = "bash experimental/tools/rigidification/run_nuxmv.sh " +
        modelFilePath.string() + " " + wrapperPath.string() + " " +
        nuxmvReport.string() + " " + nuxmvCommand.string();
  ret = dynamatic::experimental::execute(cmd);
  if (ret != 0) {
    llvm::errs() << "Failed to run to NuXmv\n";
    exit(true);
  }

  // Parse the results
  cmd = "python experimental/tools/rigidification/parse_nuxmv_results.py " +
        propertyPath.string() + " " + nuxmvReport.string();
  ret = dynamatic::experimental::executeWithRedirect(cmd, "/dev/null");
  if (ret != 0) {
    llvm::errs() << "Failed to parse NuXmv results\n";
    exit(true);
  }

  // Apply rigidification
  cmd = "bin/dynamatic-opt " + mlirPath.string() +
        " --handshake-rigidification=json-path=" + propertyPath.string();
  ret = dynamatic::experimental::execute(cmd);
  if (ret != 0) {
    llvm::errs() << "Failed to apply rigidification\n";
    exit(true);
  }
  exit(false);
}