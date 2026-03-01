//===- BlifExporterPass.cpp - BLIF exporter pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the BlifExporterPass, which exports a Synth
// circuit to a BLIF file.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Transforms/BlifExporter/BlifExporterSupport.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include <fstream>

using namespace mlir;
using namespace dynamatic;

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_BLIFEXPORTER
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

namespace {
struct BlifExporterPass
    : public dynamatic::impl::BlifExporterBase<BlifExporterPass> {
public:
  using BlifExporterBase::BlifExporterBase;
  void runDynamaticPass() override {

    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    // Register the Synth and HW dialects which the pass depends on
    context->getOrLoadDialect<synth::SynthDialect>();
    context->getOrLoadDialect<hw::HWDialect>();

    OpBuilder builder(context);

    // Get the output folder path from the pass arguments
    std::string blifFilePath = this->filepath.getValue();

    // Check that the output folder path is not empty
    if (blifFilePath.empty()) {
      llvm::errs()
          << "The output BLIF file path is empty. Please provide a valid "
             "output BLIF file path as an argument to the pass."
          << "\n";
      return signalPassFailure();
    }

    // Open the output file for writing to append the generated blif content
    std::error_code ec;
    llvm::raw_fd_ostream outputFile(blifFilePath, ec);

    if (ec) {
      llvm::errs() << "Failed to open the output file: " << blifFilePath
                   << " - " << ec.message() << "\n";
      return signalPassFailure();
    }

    // Iterate through the hwModuleOps in the module and export each of them as
    // a blif file in the output folder
    for (auto hwModuleOp : moduleOp.getOps<hw::HWModuleOp>()) {
      // Get the name of the hwModuleOp to use it as the name of the blif file
      StringRef moduleName = hwModuleOp.getName();

      // Export the hwModuleOp as a blif file in the output folder
      if (failed(exportBlifCircuit(hwModuleOp, outputFile))) {
        llvm::errs() << "Failed to export the hw module '" << moduleName
                     << "' to a blif file." << "\n";
        return signalPassFailure();
      }
    }
    outputFile.flush();
  }
};

} // namespace
