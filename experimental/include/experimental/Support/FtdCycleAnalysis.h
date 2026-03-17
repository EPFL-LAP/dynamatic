//===- FtdCycleAnalysis.h - Local Control Flow Graph Utils ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structures and utilities for analyzing the cyclic
// structure of Local Control Flow Graphs (LocalCFG) used in the Fast Token
// Delivery (FTD) algorithm. It supports:
// 1. Hierarchical loop analysis (LoopScope).
// 2. Identification of back-edges at different nesting levels.
// 3. Extraction of acyclic subgraphs (Layered CFG) for BDD construction.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_FTDCYCLEANALYSIS_H
#define EXPERIMENTAL_SUPPORT_FTDCYCLEANALYSIS_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace dynamatic {
namespace experimental {
namespace ftd {

/// Represents a reconstructed local CFG extracted from the original CFG to
/// represent the producer-consumer relationship.
struct LocalCFG {
  // The MLIR region representing the local subgraph.
  Region *region = nullptr;
  // Mapping: block in local graph -> original block.
  DenseMap<Block *, Block *> origMap;
  // The producer block in the local CFG.
  Block *newProd = nullptr;
  // The consumer block in the local CFG.
  Block *newCons = nullptr;
  // A replicated block used for self-loop delivery (Producer == Consumer).
  Block *secondVisitBB = nullptr;
  // A unique sink (exit) block to which all terminal paths lead.
  Block *sinkBB = nullptr;
  // Topological order of the reconstructed region.
  SmallVector<Block *, 8> topoOrder;
  // Temporary parent operation that owns the region.
  Operation *containerOp = nullptr;

  ~LocalCFG() = default;
};

/// Represents a single level of loop nesting within the local CFG.
/// It forms a tree structure where Level 0 is the top-level acyclic function
/// scope.
struct LoopScope {
  /// The nesting level. 0 for the top-level scope, 1 for top-level loops, etc.
  unsigned level = 0;

  /// The header block of this loop scope. For Level 0, this is the graph entry.
  Block *header = nullptr;

  /// List of latch blocks that jump back to the header in this specific loop.
  SmallVector<Block *> latches;

  /// A set of ALL back-edges contained within this scope, including those
  /// belonging to nested sub-loops. Used to prune deep back-edges during DAG
  /// extraction. Format: pair<Source, Destination>.
  DenseSet<std::pair<Block *, Block *>> allBackEdges;

  /// A set of all blocks contained within this scope (including sub-loops).
  DenseSet<Block *> allBlocksInclusive;

  /// List of immediate sub-loops nested within this scope.
  SmallVector<std::unique_ptr<LoopScope>> subLoops;

  /// Pointer to the parent scope. Null for the top-level scope.
  LoopScope *parent = nullptr;

  /// Pointer to the MLIR LoopInfo object. Null for the top-level scope.
  mlir::CFGLoop *loopInfo = nullptr;
};

/// Manages the cyclic analysis of a LocalCFG. It builds the LoopScope hierarchy
/// and provides utilities to extract acyclic, layered subgraphs for FTD
/// analysis.
class CyclicGraphManager {
public:
  /// Constructs the manager and immediately performs topological analysis to
  /// build the scope tree.
  /// \param lcfg The local control flow graph to analyze.
  explicit CyclicGraphManager(LocalCFG &lcfg);

  /// Analyzes the topology of the LocalCFG, identifying loops and building the
  /// LoopScope hierarchy (TopLevel -> SubLoops).
  void analyzeTopology();

  /// Returns the nesting level of a given block.
  /// \param bb The block to query.
  /// \return The nesting level (0 for top-level).
  unsigned getNestingLevel(Block *bb) const;

  /// Returns the root of the LoopScope tree (Level 0).
  LoopScope *getTopLevelScope() const { return topLevelScope.get(); }

  /// Extracts a normalized, acyclic LocalCFG for a specific LoopScope.
  /// This process involves:
  /// 1. cloning blocks within the scope.
  /// 2. redirecting current-level back-edges to Sink (False).
  /// 3. pruning deep-level back-edges (invalid paths).
  /// 4. redirecting loop exits to a True terminal.
  /// \param scope The loop scope to extract.
  /// \param builder The OpBuilder used to create the new graph operations.
  /// \return A unique_ptr to the newly created acyclic LocalCFG.
  std::unique_ptr<LocalCFG> extractLayeredCFG(const LoopScope *scope,
                                              OpBuilder &builder);

private:
  /// Reference to the underlying LocalCFG being analyzed.
  LocalCFG &lcfg;

  /// Dominator tree analysis utility.
  mlir::DominanceInfo domInfo;

  /// Loop analysis utility based on the dominator tree.
  mlir::CFGLoopInfo loopInfo;

  /// Mapping from blocks to their nesting level.
  DenseMap<Block *, unsigned> blockLevelMap;

  /// The root of the scope hierarchy (Level 0).
  std::unique_ptr<LoopScope> topLevelScope;

  /// Recursively builds the LoopScope tree starting from a given MLIR loop.
  /// \param loop The current MLIR loop being analyzed.
  /// \param level The current nesting level.
  /// \return A unique_ptr to the constructed LoopScope.
  std::unique_ptr<LoopScope> buildScopeRecursive(mlir::CFGLoop *loop,
                                                 unsigned level);
};

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_FTDCYCLEANALYSIS_H
