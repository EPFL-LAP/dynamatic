//===- SpecAnnotatePaths.cpp - Annotate speculative paths -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --spec-annotate-paths pass. It is responsible for
// adding the attribute "speculative" to the operations in speculative paths.
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Speculation/SpecAnnotatePaths.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

static void annotateSpeculativeRegion(handshake::SpeculatorOp specOp);

static std::optional<bool> getSpeculativeAttr(Operation *op) {
  if (auto spec = op->getAttrOfType<mlir::BoolAttr>("speculative"))
    return spec.getValue();
  return {};
}

static void runAnnotateOnFuncOp(handshake::FuncOp funcOp) {
  funcOp->walk([](handshake::SpeculatorOp specOp) {
    annotateSpeculativeRegion(specOp);
  });
}

// Check if the given operation is annotated to be speculative
bool speculation::isSpeculative(mlir::Operation *op, bool runAnalysis) {
  // Run the analysis when the op is not annotated
  if (runAnalysis && !getSpeculativeAttr(op)) {
    handshake::FuncOp funcOp = op->getParentOfType<handshake::FuncOp>();
    assert(funcOp && "op should have parent function");
    runAnnotateOnFuncOp(funcOp);
  }
  std::optional<bool> spec = getSpeculativeAttr(op);
  return spec && spec.value();
}

bool speculation::isSpeculative(mlir::OpOperand &operand, bool runAnalysis) {
  Operation *op = operand.getOwner();
  return speculation::isSpeculative(op, runAnalysis);
}

bool speculation::isSpeculative(mlir::Value value, bool runAnalysis) {
  Operation *op = value.getDefiningOp();
  return speculation::isSpeculative(op, runAnalysis);
}

// DFS-like traversal in within the speculative region that marks every visited
// operation with the speculative=true attribute. The traversal ends where the
// speculative region ends, that is, at commit units.
static void markSpeculativeOpsRecursive(Operation *currOp,
                                        llvm::DenseSet<Operation *> visited) {
  // End traversal if currOp is already in visited set
  if (auto [_, isNewOp] = visited.insert(currOp); !isNewOp)
    return;

  BoolAttr specAttr = BoolAttr::get(currOp->getContext(), 1);
  currOp->setAttr("speculative", specAttr);
  // End traversal at Commit units
  if (not isa<handshake::SpecCommitOp>(currOp)) {
    for (Operation *succOp : currOp->getUsers()) {
      markSpeculativeOpsRecursive(succOp, visited);
    }
  }
}

static void annotateSpeculativeRegion(handshake::SpeculatorOp specOp) {
  // Create visited set
  llvm::DenseSet<Operation *> visited;
  // Traverse speculative region starting at the speculator
  markSpeculativeOpsRecursive(specOp, visited);
}

namespace {
struct SpecAnnotatePathsPass
    : public dynamatic::experimental::speculation::impl::SpecAnnotatePathsBase<
          SpecAnnotatePathsPass> {

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    modOp->walk([](handshake::SpeculatorOp specOp) {
      annotateSpeculativeRegion(specOp);
    });
  }
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculation::createSpecAnnotatePaths() {
  return std::make_unique<SpecAnnotatePathsPass>();
}
