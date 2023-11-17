//===- OperationNames.cpp - Canonicalize Handshake ops ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the pass that only deal with operation names.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/OperationNames.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"

using namespace circt;
using namespace dynamatic;

namespace {

/// Simple driver for the pass that names all operations in the IR.
struct NameAllOperationsPass
    : public dynamatic::impl::NameAllOperationsBase<NameAllOperationsPass> {

  void runOnOperation() override {
    NameAnalysis &analysis = getAnalysis<NameAnalysis>();
    if (!analysis.isAnalysisValid())
      return signalPassFailure();
    if (!analysis.areAllOpsNamed())
      analysis.nameAllUnnamedOps();
    markAnalysesPreserved<NameAnalysis>();
  };
};

/// Simple driver for the pass that removes all operation names from the IR.
struct RemoveOperationNamesPass
    : public dynamatic::impl::RemoveOperationNamesBase<
          RemoveOperationNamesPass> {

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      if (op->hasAttrOfType<handshake::NameAttr>(
              handshake::NameAttr::getMnemonic()))
        op->removeAttr(handshake::NameAttr::getMnemonic());
    });
  };
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createNameAllOperations() {
  return std::make_unique<NameAllOperationsPass>();
}
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createRemoveOperationNames() {
  return std::make_unique<RemoveOperationNamesPass>();
}
