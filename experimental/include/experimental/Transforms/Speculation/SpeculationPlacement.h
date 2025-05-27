//===- SpeculationPlacement.h - Speculation units placement -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the data structures needed for speculation algorithms
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SPECULATION_PLACEMENT_H
#define DYNAMATIC_TRANSFORMS_SPECULATION_PLACEMENT_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace dynamatic {
namespace experimental {
namespace speculation {

struct PlacementOperand {
  std::string opName;
  unsigned opIdx;
};

class SpeculationPlacements {
private:
  OpOperand *speculator;
  llvm::DenseSet<OpOperand *> saves;
  llvm::DenseSet<OpOperand *> commits;
  llvm::DenseSet<OpOperand *> saveCommits;

  unsigned int speculatorFifoDepth;
  unsigned int saveCommitsFifoDepth;

public:
  /// Empty constructor
  SpeculationPlacements() = default;

  /// Initializer with operand specifying the speculator position
  SpeculationPlacements(OpOperand &speculatorPosition)
      : speculator(&speculatorPosition){};

  /// Set the speculator operations positions according to a JSON file
  static LogicalResult readFromJSON(const std::string &jsonPath,
                                    SpeculationPlacements &place,
                                    NameAnalysis &nameAnalysis);

  /// Explicitly set the speculator position
  void setSpeculator(OpOperand &dstOpOperand);

  /// Add the position of a Save operation
  void addSave(OpOperand &dstOpOperand);

  /// Add the position of a Commit operation
  void addCommit(OpOperand &dstOpOperand);

  /// Add the position of a SaveCommit operation
  void addSaveCommit(OpOperand &dstOpOperand);

  /// Check if there is a save in the given OpOperand edge
  bool containsSave(OpOperand &dstOpOperand);

  /// Check if there is a commit in the given OpOperand edge
  bool containsCommit(OpOperand &dstOpOperand);

  /// Check if there is a save-commit in the given OpOperand edge
  bool containsSaveCommit(OpOperand &dstOpOperand);

  /// Remove a commit (edge) from the commit placement map
  void eraseCommit(OpOperand &dstOpOperand);

  /// Remove a save (edge) from the save placement map
  void eraseSave(OpOperand &dstOpOperand);

  /// Get the Placement instance that specifies the Speculator position
  OpOperand &getSpeculatorPlacement();

  /// Get a set of the existing operation placements
  template <typename T>
  const llvm::DenseSet<OpOperand *> &getPlacements();

  unsigned int getSpeculatorFifoDepth();
  void setSpeculatorFifoDepth(unsigned int depth);
  unsigned int getSaveCommitsFifoDepth();
  void setSaveCommitsFifoDepth(unsigned int depth);
};

} // namespace speculation
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SPECULATION_PLACEMENT_H
