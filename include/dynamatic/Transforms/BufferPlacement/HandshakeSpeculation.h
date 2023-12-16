//===- HandshakeSpeculation.h - Speculation units placement -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-speculation pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_SPECULATION_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_SPECULATION_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace dynamatic {
namespace buffer {

struct Placement {
  StringRef srcOpName;
  StringRef dstOpName;

  // Constructor from a C++ pair
  Placement(std::pair<StringRef, StringRef> p = {"", ""})
      : srcOpName(p.first), dstOpName(p.second) {}
};

typedef std::vector<Placement> PlacementList;

// For the map structure we need to use std::string instead of StringRef
typedef std::multimap<std::string, std::string> PlacementMap;

class SpeculationPlacements {
private:
  Placement speculator;
  PlacementMap saves;
  PlacementMap commits;
  PlacementMap saveCommits;
  PlacementMap synchronizers;

public:
  // Initializer with source and destination operations for the Speculator
  SpeculationPlacements(StringRef srcOpName, StringRef dstOpName) {
    speculator = Placement({srcOpName, dstOpName});
  }

  // Explicitly set the speculator position
  void setSpeculator(StringRef srcOpName, StringRef dstOpName);

  // Add the position of a Save operation
  void addSave(StringRef srcOpName, StringRef dstOpName);

  // Add the position of a Save operation
  void addCommit(StringRef srcOpName, StringRef dstOpName);

  // Add the position of a Save operation
  void addSaveCommit(StringRef srcOpName, StringRef dstOpName);

  // Scan for consecutive Save and Commit operations, replace by a SaveCommit
  // void mergeConsecutiveSaveCommits();

  // Check if there is a commit from srcOp to dstOp
  bool containsCommit(StringRef srcOpName, StringRef dstOpName);

  // Remove a commit from the commit placement map
  void eraseCommit(StringRef srcOpName, StringRef dstOpName);

  // Get the Placement instance that specifies the Speculator position
  Placement getSpeculatorPlacement();

  // Get a vector of the existing Save operation placements
  PlacementList getSavePlacements();

  // Get a vector of the existing Commit operation placements
  PlacementList getCommitPlacements();

  // Get a vector of the existing SaveCommit operation placements
  PlacementList getSaveCommitPlacements();
};

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeSpeculation(StringRef srcOpName = "", StringRef dstOpName = "");

#define GEN_PASS_DECL_HANDSHAKESPECULATION
#define GEN_PASS_DEF_HANDSHAKESPECULATION
#include "dynamatic/Transforms/Passes.h.inc"

/// Public pass driver for the speculation placement pass.
struct HandshakeSpeculationPass
    : public dynamatic::buffer::impl::HandshakeSpeculationBase<
          HandshakeSpeculationPass> {
  HandshakeSpeculationPass(StringRef srcOpName, StringRef dstOpName);

  void runDynamaticPass() override;

protected:
  // Get the placement of the operations
  LogicalResult findOpPlacement(SpeculationPlacements &specPlacement);

  // Driver to place the Speculation Operations
  LogicalResult placeSpecOperations(SpeculationPlacements &specPlacement);

  // Place the operation handshake::SpeculatorOp in between src and dst.
  LogicalResult placeSpeculator(const Placement &specPlacement,
                                circt::handshake::SpeculatorOp *specOp);

  // Place the operation specified in T with the control signal ctrlSignal
  template <typename T>
  LogicalResult placeUnits(const PlacementList &placements, Value ctrlSignal);

  // Wrapper around placeCommitsTraversal to prepare and invoke the placement
  LogicalResult placeCommits(SpeculationPlacements &placements,
                             circt::handshake::SpeculatorOp *specOp);

  // Recursive function to traverse the IR in a DFS fashion and place
  // commits. The already placed commits are erased from SpeculationPlacements
  bool placeCommitsTraversal(std::unordered_set<std::string> &visited,
                             Value ctrlSignal, Operation *currOp,
                             SpeculationPlacements &place);

  // Place the SaveCommit operations and the control path
  LogicalResult placeSaveCommits(const PlacementList &specPlacement,
                                 circt::handshake::SpeculatorOp *specOp);
};

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_SPECULATION_H
