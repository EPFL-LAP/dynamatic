//===- HandshakeSpeculation.cpp - Speculative Dataflows ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Placement of Speculation components to enable speculative execution.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/Speculation/HandshakeSpeculation.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseSet.h"
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace llvm::sys;
using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::speculation;

HandshakeSpeculationPass::HandshakeSpeculationPass(std::string unitPositions) {
  this->unitPositions = unitPositions;
}

template <typename T>
LogicalResult HandshakeSpeculationPass::placeUnits(Value ctrlSignal) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  for (const OpPlacement p : placements.getPlacements<T>()) {
    // Create and connect the new Operation
    builder.setInsertionPoint(p.dstOp);
    T newOp = builder.create<T>(p.dstOp->getLoc(), p.srcOpResult, ctrlSignal);

    // Connect the new Operation to dstOp
    p.srcOpResult.replaceAllUsesExcept(newOp.getResult(), newOp);
  }

  return success();
}

LogicalResult HandshakeSpeculationPass::placeSpeculator() {
  MLIRContext *ctx = &getContext();

  OpPlacement place = placements.getSpeculatorPlacement();

  OpBuilder builder(ctx);
  builder.setInsertionPoint(place.dstOp);

  specOp = builder.create<handshake::SpeculatorOp>(place.dstOp->getLoc(),
                                                   place.srcOpResult);

  // Replace uses of the orginal source operation's result with the speculator's
  // result, except in the speculator's operands (otherwise this would create a
  // self-loop from the speculator to the speculator)
  place.srcOpResult.replaceAllUsesExcept(specOp->getResult(0), specOp);

  return success();
}

void HandshakeSpeculationPass::runDynamaticPass() {
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  if (failed(SpeculationPlacements::readFromJSON(
          this->unitPositions, this->placements, nameAnalysis)))
    return signalPassFailure();

  if (failed(placeSpeculator()))
    return signalPassFailure();

  // Place Save operations
  if (failed(placeUnits<handshake::SpecSaveOp>(this->specOp->getResult(1))))
    return signalPassFailure();

  // Place Commit operations and the Commit control path
  if (failed(placeUnits<handshake::SpecCommitOp>(this->specOp->getResult(2))))
    return signalPassFailure();

  // Place SaveCommit operations and the SaveCommit control path
  if (failed(
          placeUnits<handshake::SpecSaveCommitOp>(this->specOp->getResult(3))))
    return signalPassFailure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::speculation::createHandshakeSpeculation(std::string unitPositions) {
  return std::make_unique<HandshakeSpeculationPass>(unitPositions);
}
