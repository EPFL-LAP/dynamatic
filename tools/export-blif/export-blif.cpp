//===- export-blif.cpp - BLIF exporter pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exports a BLIF file from a Synth circuit. The code iterates through the
// operations in the Synth circuit and generates the corresponding BLIF
// statements for latches, logic gates and constants.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/HW/HWDialect.h"
#include "dynamatic/Dialect/HW/HWOpInterfaces.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/RTL/RTL.h"
#include "dynamatic/Support/System.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "dynamatic/Transforms/BlifExporter/BlifExporterSupport.h"
#include "experimental/Support/FormalProperty.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>

using namespace mlir;
using namespace dynamatic;

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::synth;
using namespace dynamatic::hw;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputMlirFilename(cl::Positional, cl::Required,
                                              cl::desc("<input MLIR file>"),
                                              cl::cat(mainCategory));

static cl::opt<std::string> outputBlifFilename(cl::Positional, cl::Required,
                                               cl::desc("<output BLIF file>"),
                                               cl::cat(mainCategory));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Exports a BLIF file from a Synth circuit. The code iterates through the "
      "operations in the Synth circuit and generates the corresponding BLIF "
      "statements for latches, logic gates and constants.");

  auto inputMlirFileOrErr =
      MemoryBuffer::getFileOrSTDIN(inputMlirFilename.c_str());
  if (std::error_code error = inputMlirFileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '"
                 << inputMlirFilename << "': " << error.message() << "\n";
    return 1;
  }

  // Get the output folder path from the pass arguments
  std::string blifFilePath = outputBlifFilename.getValue();

  // Check that the output folder path is not empty
  if (blifFilePath.empty()) {
    llvm::errs()
        << "The output BLIF file path is empty. Please provide a valid "
           "output BLIF file path as an argument to the pass."
        << "\n";
    return 1;
  }

  // Open the output file for writing to append the generated blif content
  std::error_code ec;
  llvm::raw_fd_ostream outputFile(blifFilePath, ec);

  if (ec) {
    llvm::errs() << "Failed to open the output file: " << blifFilePath << " - "
                 << ec.message() << "\n";
    return 1;
  }

  // We need the HW and Synth dialects
  MLIRContext context;
  context.loadDialect<hw::HWDialect, synth::SynthDialect>();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*inputMlirFileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  // Iterate through the hwModuleOps in the module and export each of them as
  // a blif file in the output folder
  for (auto hwModuleOp : modOp->getOps<hw::HWModuleOp>()) {
    // Get the name of the hwModuleOp to use it as the name of the blif file
    StringRef moduleName = hwModuleOp.getName();

    BlifExporter blifExporter(hwModuleOp, outputFile);
    // Export the hwModuleOp as a blif file in the output folder
    if (failed(blifExporter.exportBlifCircuit())) {
      llvm::errs() << "Failed to export the hw module '" << moduleName
                   << "' to a blif file." << "\n";
      return 1;
    }
  }
  outputFile.flush();
}
