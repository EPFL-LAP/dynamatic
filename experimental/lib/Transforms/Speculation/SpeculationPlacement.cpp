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

#include "experimental/Transforms/Speculation/SpeculationPlacement.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include <fstream>
#include <map>
#include <string>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;

// SpeculationPlacements Methods

void SpeculationPlacements::setSpeculator(OpOperand &dstOpOperand) {
  this->speculator = &dstOpOperand;
}

void SpeculationPlacements::addSave(OpOperand &dstOpOperand) {
  this->saves.insert(&dstOpOperand);
}

void SpeculationPlacements::addCommit(OpOperand &dstOpOperand) {
  this->commits.insert(&dstOpOperand);
}

void SpeculationPlacements::addSaveCommit(OpOperand &dstOpOperand) {
  this->saveCommits.insert(&dstOpOperand);
}

bool SpeculationPlacements::containsCommit(OpOperand &dstOpOperand) {
  return this->commits.contains(&dstOpOperand);
}

bool SpeculationPlacements::containsSave(OpOperand &dstOpOperand) {
  return this->saves.contains(&dstOpOperand);
}

bool SpeculationPlacements::containsSaveCommit(OpOperand &dstOpOperand) {
  return this->saveCommits.contains(&dstOpOperand);
}

void SpeculationPlacements::eraseSave(OpOperand &dstOpOperand) {
  this->saves.erase(&dstOpOperand);
}

void SpeculationPlacements::eraseCommit(OpOperand &dstOpOperand) {
  this->commits.erase(&dstOpOperand);
}

OpOperand &SpeculationPlacements::getSpeculatorPlacement() {
  return *this->speculator;
}

template <>
const llvm::DenseSet<OpOperand *> &
SpeculationPlacements::getPlacements<handshake::SpecSaveOp>() {
  return this->saves;
}

template <>
const llvm::DenseSet<OpOperand *> &
SpeculationPlacements::getPlacements<handshake::SpecCommitOp>() {
  return this->commits;
}

template <>
const llvm::DenseSet<OpOperand *> &
SpeculationPlacements::getPlacements<handshake::SpecSaveCommitOp>() {
  return this->saveCommits;
}


LogicalResult
SpeculationPlacements::readFromAttribute(mlir::ModuleOp modOp,
                                         SpeculationPlacements &placements) {
  llvm::SmallVector<mlir::Operation *, 2> markedOps;
  modOp.walk([&](mlir::Operation *op) {
    if (op->hasAttr("dynamatic.speculate"))
      markedOps.push_back(op);
  });
  if (markedOps.empty()) {
    modOp.emitError() << "no op carries the `dynamatic.speculate` attribute";
    return failure();
  }
  if (markedOps.size() > 1) {
    modOp.emitError() << "more than one op carries the `dynamatic.speculate` "
                         "attribute; only one speculator is supported";
    return failure();
  }
  mlir::Operation *producer = markedOps.front();

  auto dictAttr = producer->getAttrOfType<mlir::DictionaryAttr>(
      "dynamatic.speculate");
  if (!dictAttr) {
    producer->emitError()
        << "`dynamatic.speculate` must be a DictionaryAttr";
    return failure();
  }

  auto maxPredAttr =
      dictAttr.getAs<mlir::IntegerAttr>("max_predictions");
  if (!maxPredAttr) {
    producer->emitError()
        << "`dynamatic.speculate` is missing `max_predictions`";
    return failure();
  }
  unsigned maxPred =
      static_cast<unsigned>(maxPredAttr.getValue().getZExtValue());

  // Which result of the producer the marker was attached to. Recorded as a
  // sibling `dynamatic.speculate.result_idx` attribute by
  // ConsumeProducerOutputAttrMarker.
  auto idxAttr = producer->getAttrOfType<mlir::IntegerAttr>(
      "dynamatic.speculate.result_idx");
  if (!idxAttr) {
    producer->emitError() << "op carrying `dynamatic.speculate` is missing "
                             "the sibling `dynamatic.speculate.result_idx` "
                             "attribute";
    return failure();
  }
  unsigned resultIdx =
      static_cast<unsigned>(idxAttr.getValue().getLimitedValue());
  if (resultIdx >= producer->getNumResults()) {
    producer->emitError() << "`dynamatic.speculate.result_idx` " << resultIdx
                          << " is out of range for producer with "
                          << producer->getNumResults() << " result(s)";
    return failure();
  }
  mlir::Value res = producer->getResult(resultIdx);
  if (!res.hasOneUse()) {
    producer->emitError() << "`dynamatic.speculate` producer's result must "
                             "have exactly one use (the speculator cut "
                             "point)";
    return failure();
  }
  mlir::OpOperand &dstOpOperand = *res.getUses().begin();

  placements.setSpeculator(dstOpOperand);
  placements.setSpeculatorFifoDepth(maxPred);
  placements.setSaveCommitsFifoDepth(maxPred);
  return success();
}

unsigned int SpeculationPlacements::getSpeculatorFifoDepth() {
  return this->speculatorFifoDepth;
}

void SpeculationPlacements::setSpeculatorFifoDepth(unsigned int depth) {
  this->speculatorFifoDepth = depth;
}

unsigned int SpeculationPlacements::getSaveCommitsFifoDepth() {
  return this->saveCommitsFifoDepth;
}

void SpeculationPlacements::setSaveCommitsFifoDepth(unsigned int depth) {
  this->saveCommitsFifoDepth = depth;
}
