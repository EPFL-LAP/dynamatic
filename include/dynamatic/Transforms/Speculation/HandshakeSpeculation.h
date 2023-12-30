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

#ifndef DYNAMATIC_TRANSFORMS_SPECULATION_PASS_H
#define DYNAMATIC_TRANSFORMS_SPECULATION_PASS_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/Speculation/SpeculationPlacement.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace dynamatic {
namespace speculation {

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeSpeculation(std::string unitPositions = "");

#define GEN_PASS_DECL_HANDSHAKESPECULATION
#define GEN_PASS_DEF_HANDSHAKESPECULATION
#include "dynamatic/Transforms/Passes.h.inc"

/// Public pass driver for the speculation placement pass.
struct HandshakeSpeculationPass
    : public dynamatic::speculation::impl::HandshakeSpeculationBase<
          HandshakeSpeculationPass> {
  HandshakeSpeculationPass(std::string unitPositions = "");

  void runDynamaticPass() override;

private:
  SpeculationPlacements placements;
  Operation *specOp;

protected:
  // Get the placement of the operations
  LogicalResult findOpPlacement();

  // Scan the IR from specOp and store the SaveOp positions in placements
  LogicalResult findSavePlacements(Value startValue);

  // Driver to place the Speculation Operations
  LogicalResult placeAllOperations();

  // Place the operation handshake::SpeculatorOp in between src and dst.
  LogicalResult placeSpeculator();

  // Place the operation specified in T with the control signal ctrlSignal
  template <typename T>
  LogicalResult placeUnits(Value ctrlSignal);

  // Wrapper around placeCommitsTraversal to prepare and invoke the placement
  LogicalResult placeCommits();

  // Create the control path for commit units by replicating branches in the IR
  bool routeCommitControlTraversal(llvm::DenseSet<Operation *> visited,
                                   Value ctrlSignal, Operation *currOp);

  // Place the SaveCommit operations and the control path
  LogicalResult placeSaveCommits();
};

} // namespace speculation
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_SPECULATION_PASS_H