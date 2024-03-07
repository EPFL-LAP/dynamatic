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

static void
markSpeculativeOperationsRecursive(Operation *currOp,
                                   llvm::DenseSet<Operation *> visited) {
  // End traversal if currOp is already in visited set
  if (auto [_, isNewOp] = visited.insert(currOp); !isNewOp)
    return;

  BoolAttr specAttr = BoolAttr::get(currOp->getContext(), 1);
  currOp->setAttr("speculative", specAttr);
  // End traversal at Commit units
  if (not isa<handshake::SpecCommitOp>(currOp)) {
    for (Operation *succOp : currOp->getUsers()) {
      markSpeculativeOperationsRecursive(succOp, visited);
    }
  }
}

static bool annotateSpeculativeRegion(handshake::SpeculatorOp specOp) {

  // Create visited set
  llvm::DenseSet<Operation *> visited;
  visited.insert(specOp);

  // Traverse speculative region starting at the speculator
  for (Operation *user : specOp.getDataOut().getUsers()) {
    markSpeculativeOperationsRecursive(user, visited);
  }

  // Traverse speculative region starting at save units
  for (Operation *user : specOp.getSaveCtrl().getUsers()) {
    markSpeculativeOperationsRecursive(user, visited);
  }
  return true;
}

namespace {
struct SpecAnnotatePathsPass
    : public dynamatic::experimental::speculation::impl::SpecAnnotatePathsBase<
          SpecAnnotatePathsPass> {

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    modOp->walk([](handshake::SpeculatorOp specOp) {
      llvm::outs() << "Found a speculator\n";
      annotateSpeculativeRegion(specOp);
    });
  }
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculation::createSpecAnnotatePaths() {
  return std::make_unique<SpecAnnotatePathsPass>();
}