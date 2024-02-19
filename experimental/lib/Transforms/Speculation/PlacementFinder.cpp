//===- PlacementFinder.cpp - Automatic speculation units finder -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the class and methods for automatic finding of
// speculative units positions.
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/Speculation/PlacementFinder.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/Logging.h"
#include "experimental/Transforms/Speculation/SpeculationPlacement.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculation;

PlacementFinder::PlacementFinder(SpeculationPlacements &placements)
    : placements(placements) {
  OpPlacement specPos = placements.getSpeculatorPlacement();
  assert(specPos.dstOp != nullptr && "Speculator position is undefined");
}

void PlacementFinder::clearPlacements() {
  OpPlacement specPos = placements.getSpeculatorPlacement();
  this->placements = SpeculationPlacements(specPos.srcOpResult, specPos.dstOp);
}

LogicalResult PlacementFinder::findSavePositions() {
  // Implementation for findSavePositions
  return success();
}

LogicalResult PlacementFinder::findCommitPositions() {
  // Implementation for findCommitPositions
  return success();
}

LogicalResult PlacementFinder::findSaveCommitPositions() {
  // Implementation for findSaveCommitPositions
  return success();
}

LogicalResult PlacementFinder::findPlacements() {
  // Clear the data structure
  // clearPlacements();

  if (failed(findSavePositions()))
    return failure();

  if (failed(findCommitPositions()))
    return failure();

  if (failed(findSaveCommitPositions()))
    return failure();

  return success();
}