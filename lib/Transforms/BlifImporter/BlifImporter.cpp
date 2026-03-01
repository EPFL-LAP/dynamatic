//===- BlifImporterPass.cpp - BLIF importer pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the BlifImporterPass, which imports a BLIF file
// and generates a corresponding Synth circuit.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Transforms/BlifImporter/BlifImporterSupport.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace dynamatic;

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_BLIFIMPORTER
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

namespace {
struct BlifImporterPass
    : public dynamatic::impl::BlifImporterBase<BlifImporterPass> {
public:
  using BlifImporterBase::BlifImporterBase;
  void runDynamaticPass() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    // Register the Synth and HW dialects which the pass depends on
    context->getOrLoadDialect<synth::SynthDialect>();
    context->getOrLoadDialect<hw::HWDialect>();

    OpBuilder builder(context);

    // Get the blif file path from the pass arguments
    StringRef blifFilePath = this->filepath.getValue();

    // Check that the blif file path is not empty
    if (blifFilePath.empty()) {
      llvm::errs() << "The blif file path is empty. Please provide a valid "
                      "blif file path as an argument to the pass."
                   << "\n";
      return signalPassFailure();
    }

    // Import the blif circuit and generate the corresponding synth circuit
    importBlifCircuit(moduleOp,
                      /*Location inside module operation*/ moduleOp->getLoc(),
                      blifFilePath);
  }
};

} // namespace
