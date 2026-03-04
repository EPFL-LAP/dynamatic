//===- importer-blif.cpp - BLIF importer pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Imports a BLIF file into a Synth circuit. The code iterates through the
// operations in the BLIF file and generates the corresponding Synth
// operations for latches, logic gates and constants.
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
#include "dynamatic/Transforms/BlifImporter/BlifImporterSupport.h"
#include "experimental/Support/FormalProperty.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
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

static cl::opt<std::string> outputMlirFilename(cl::Positional, cl::Required,
                                               cl::desc("<output MLIR file>"),
                                               cl::cat(mainCategory));

static cl::opt<std::string> inputBlifFilename(cl::Positional, cl::Required,
                                              cl::desc("<input BLIF file>"),
                                              cl::cat(mainCategory));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Imports a BLIF file into a Synth circuit. The code iterates through the "
      "operations in the BLIF file and generates the corresponding Synth "
      "operations for latches, logic gates and constants.");

  auto inputBlifFileOrErr =
      MemoryBuffer::getFileOrSTDIN(inputBlifFilename.c_str());
  if (std::error_code error = inputBlifFileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '"
                 << inputBlifFilename << "': " << error.message() << "\n";
    return 1;
  }

  // Get the output folder path from the pass arguments
  std::string outputMlirFilePath = outputMlirFilename.getValue();

  // Check that the output folder path is not empty
  if (outputMlirFilePath.empty()) {
    llvm::errs()
        << "The output MLIR file path is empty. Please provide a valid "
           "output MLIR file path as an argument to the pass."
        << "\n";
    return 1;
  }

  // Open the output file for writing to append the generated blif content
  std::error_code ec;
  llvm::raw_fd_ostream outputMlirFile(outputMlirFilePath, ec);

  if (ec) {
    llvm::errs() << "Failed to open the output file: " << outputMlirFilePath
                 << " - " << ec.message() << "\n";
    return 1;
  }

  // We need the HW and Synth dialects
  mlir::MLIRContext context;
  context.loadDialect<hw::HWDialect, synth::SynthDialect>();

  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> modOp =
      mlir::ModuleOp::create(builder.getUnknownLoc());

  mlir::ModuleOp moduleOp = modOp.get();

  // Import the blif circuit and generate the corresponding synth circuit
  importBlifCircuit(moduleOp, inputBlifFilename);

  // Write the generated MLIR module to the output file
  if (failed(verify(moduleOp))) {
    llvm::errs() << "Module verification failed!\n";
    return 1;
  }
  moduleOp.print(outputMlirFile);
  outputMlirFile.flush();
}
