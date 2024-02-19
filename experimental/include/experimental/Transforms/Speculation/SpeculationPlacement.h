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

struct OpPlacement {
  Value srcOpResult;
  Operation *dstOp;

  bool operator==(const OpPlacement &other) const;

  struct Hash {
    std::size_t operator()(const OpPlacement &p) const;
  };
};

using PlacementList = std::unordered_set<OpPlacement, OpPlacement::Hash>;

class SpeculationPlacements {
private:
  OpPlacement speculator;
  PlacementList saves;
  PlacementList commits;
  PlacementList saveCommits;

public:
  /// Empty constructor
  SpeculationPlacements() = default;

  /// Initializer with source and destination operations for the Speculator
  SpeculationPlacements(Value srcOpResult, Operation *dstOp)
      : speculator{srcOpResult, dstOp} {};

  /// Set the speculator operations positions according to a JSON file
  static LogicalResult readFromJSON(const std::string &jsonPath,
                                    SpeculationPlacements &place,
                                    NameAnalysis &nameAnalysis);

  /// Explicitly set the speculator position
  void setSpeculator(Value srcOpResult, Operation *dstOp);

  /// Add the position of a Save operation
  void addSave(Value srcOpResult, Operation *dstOp);

  /// Add the position of a Save operation
  void addCommit(Value srcOpResult, Operation *dstOp);

  /// Add the position of a Save operation
  void addSaveCommit(Value srcOpResult, Operation *dstOp);

  /// Check if there is a commit from srcOp to dstOp
  bool containsCommit(Value srcOpResult, Operation *dstOp);

  /// Check if there is a save from srcOp to dstOp
  bool containsSave(Value srcOpResult, Operation *dstOp);

  /// Remove a commit from the commit placement map
  void eraseCommit(Value srcOpResult, Operation *dstOp);

  /// Remove a save from the save placement map
  void eraseSave(Value srcOpResult, Operation *dstOp);

  /// Get the Placement instance that specifies the Speculator position
  OpPlacement getSpeculatorPlacement();

  /// Get a set of the existing operation placements
  template <typename T>
  const PlacementList &getPlacements();
};

} // namespace speculation
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SPECULATION_PLACEMENT_H
