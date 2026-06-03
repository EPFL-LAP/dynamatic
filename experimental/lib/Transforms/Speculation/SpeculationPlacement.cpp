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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

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

LogicalResult
SpeculationPlacements::readFromAttribute(mlir::ModuleOp modOp,
                                         SpeculationPlacements &placements) {
  // small vector to store
  // the ops with a speculation attribute
  llvm::SmallVector<mlir::Operation *, 2> speculateOnOps;

  // walk over op in the ir and store it
  // if it has the attr
  modOp.walk([&](mlir::Operation *op) {
    if (op->hasAttr("dynamatic.speculate"))
      speculateOnOps.push_back(op);
  });

  // if no op found, speculation pass fails
  if (speculateOnOps.empty()) {
    modOp.emitError() << "no op carries the `dynamatic.speculate` attribute";
    return failure();
  }

  // if more than one op found, speculation pass fails
  if (speculateOnOps.size() > 1) {
    modOp.emitError() << "more than one op carries the `dynamatic.speculate` "
                         "attribute; only one speculator is supported";
    return failure();
  }

  // get the op to speculate on
  mlir::Operation *producer = speculateOnOps.front();

  // get the dictionary attribute
  // with the options of how to speculate
  auto dictAttr =
      producer->getAttrOfType<mlir::DictionaryAttr>("dynamatic.speculate");

  // enforce that the attribute is the right type
  if (!dictAttr) {
    producer->emitError() << "`dynamatic.speculate` must be a DictionaryAttr";
    return failure();
  }

  // get the max number of predictions
  auto maxPredAttr = dictAttr.getAs<mlir::IntegerAttr>("max_predictions");
  if (!maxPredAttr) {
    producer->emitError()
        << "`dynamatic.speculate` is missing `max_predictions`";
    return failure();
  }
  // convert from IntegerAttr to an APInt to a ui64_t to an unsigned
  unsigned maxPred =
      static_cast<unsigned>(maxPredAttr.getValue().getLimitedValue());

  // Get which result the speculate attibute applies to
  auto idxAttr = producer->getAttrOfType<mlir::IntegerAttr>(
      "dynamatic.speculate.result_idx");
  if (!idxAttr) {
    producer->emitError() << "Op containing `dynamatic.speculate` attribute "
                             "did not contain a "
                             "`dynamatic.speculate.result_idx` attribute";
    return failure();
  }

  // convert from IntegerAttr to an APInt to a ui64_t to an unsigned
  unsigned resultIdx =
      static_cast<unsigned>(idxAttr.getValue().getLimitedValue());

  // if there is no result corresponding to the index
  // placement fails
  if (resultIdx >= producer->getNumResults()) {
    producer->emitError() << "`dynamatic.speculate.result_idx` " << resultIdx
                          << " is out of range for producer with "
                          << producer->getNumResults() << " result(s)";
    return failure();
  }
  // get the actual result
  mlir::Value res = producer->getResult(resultIdx);

  // IR must be setup to use forks
  // before placing running speculation pass
  if (!res.hasOneUse()) {
    producer->emitError() << "`dynamatic.speculate` producer's result must "
                             "have exactly one use (the speculator cut point)";
    return failure();
  }

  // get the one use of the result
  // as an input to the consumer
  mlir::OpOperand &dstOpOperand = *res.getUses().begin();

  // populate the placements option struct
  placements.setSpeculator(dstOpOperand);
  placements.setSpeculatorFifoDepth(maxPred);
  placements.setSaveCommitsFifoDepth(maxPred);
  return success();
}
