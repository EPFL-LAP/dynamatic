//===- FtdSuppression.h - Suppression Infrastructure ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public interface for suppression: building the circuitry that discards a
// producer's data token on the executions where its consumer will not use it.
//
// The entry point is insertDirectSuppression; the remaining declarations are
// the data structures and building blocks it relies on (local CFG and decision
// graph types, loop analysis, Boolean-condition-to-circuit conversion, token
// distribution, and cyclic demotion). See FtdSuppression.cpp for how these fit
// together.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_FTDSUPPRESSION_H
#define EXPERIMENTAL_SUPPORT_FTDSUPPRESSION_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <memory>
#include <vector>

namespace dynamatic {
namespace experimental {
namespace ftd {

// ===--------------------------------------------------------------------=== //
// LocalCFG — Reconstructed subgraph for producer-consumer analysis
// ===--------------------------------------------------------------------=== //

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

// ===--------------------------------------------------------------------=== //
// LoopScope — Hierarchical loop nesting
// ===--------------------------------------------------------------------=== //

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

// ===--------------------------------------------------------------------=== //
// CyclicGraphManager — Loop analysis and layered CFG extraction
// ===--------------------------------------------------------------------=== //

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

// ===--------------------------------------------------------------------=== //
// Distribution data structures
// ===--------------------------------------------------------------------=== //

/// One branch decision along a path: condition variable `var` resolved to
/// `value` (which way the branch went).
struct PathStep {
  std::string var;
  bool value;

  bool operator==(const PathStep &other) const {
    return var == other.var && value == other.value;
  }
  bool operator!=(const PathStep &other) const { return !(*this == other); }
};

/// A path through the control flow, given as the branch decisions taken.
using PathContext = std::vector<PathStep>;

/// A request for the token of condition variable `varName` on the path `path`.
/// Used while routing condition tokens to the muxes that consume them.
struct VariableRequirement {
  std::string varName;
  PathContext path;
};

/// Maps a variable name to its available versions across different paths.
/// Each entry stores the path context where the value is valid.
struct SignalRegistry {
  std::map<std::string, std::vector<std::pair<PathContext, Value>>> map;

  /// Registers a physical signal available at a specific path context.
  void registerSignal(StringRef var, const PathContext &path, Value val);

  /// Finds the best signal source using Longest Prefix Match.
  /// Returns the value defined in the deepest matching path context.
  Value lookup(StringRef var, const PathContext &queryPath);
};

// ===--------------------------------------------------------------------=== //
// CyclicDemotionHelper — Cyclic demotion logic for suppression
// ===--------------------------------------------------------------------=== //

/// Helper struct encapsulating cyclic demotion logic for reuse across
/// different suppression contexts (direct suppression and MU gate exit).
struct CyclicDemotionHelper {
  // Context supplied by the caller (the suppression driver):
  mlir::OpBuilder &builder;
  MLIRContext *ctx;
  const BlockIndexing &bi;
  CyclicGraphManager &cyclicMgr; // source of loop scopes and layered CFGs
  DenseMap<Block *, Block *>
      &origToFullDG; // original block -> decision-graph block
  Location loc;
  Block *insertBlock; // basic block the emitted ops are tagged to
  DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands;
  ShadowCFG *shadow;
  // Caches one demoted value per (condition variable, target level) so the
  // same demotion circuit is not built twice.
  std::map<std::pair<std::string, unsigned>, Value> demotionCache;

  CyclicDemotionHelper(
      mlir::OpBuilder &builder, MLIRContext *ctx, const BlockIndexing &bi,
      CyclicGraphManager &cyclicMgr, DenseMap<Block *, Block *> &origToFullDG,
      Location loc, Block *insertBlock,
      DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
      ShadowCFG *shadow = nullptr);

  /// Get the nesting level of a condition variable.
  unsigned getVarNativeLevel(const std::string &var);

  /// Find the LoopScope at a given level that contains a block.
  LoopScope *findScopeForBlock(Block *origIRBlock, unsigned level);

  /// Demotes a value from fromLevel to fromLevel-1 using the layered CFG
  /// of the LoopScope at fromLevel that contains origBlock.
  Value demoteOneLevel(Value currentValue, Block *origBlock,
                       unsigned fromLevel);

  /// Returns the value of a condition variable demoted to the target level.
  /// Uses caching to avoid duplicate circuits.
  Value getValueAtLevel(const std::string &varName, unsigned targetLevel);

  /// Pre-register demoted values for high-level variables in a decision
  /// graph into the given registry.
  template <typename DGType>
  void preRegisterDemotedValues(DGType &dg, SignalRegistry &registry) {
    for (Block *b : dg->topoOrder) {
      Block *origBlock = dg->origMap.lookup(b);
      if (!origBlock)
        continue;
      std::string var = bi.getBlockCondition(origBlock);
      if (var.empty())
        continue;
      unsigned native = getVarNativeLevel(var);
      if (native > 0) {
        Value demoted = getValueAtLevel(var, 0);
        if (demoted)
          registry.registerSignal(var, {}, demoted);
      }
    }
  }
};

// ===--------------------------------------------------------------------=== //
// Graph construction functions
// ===--------------------------------------------------------------------=== //

/// Build a local control-flow subgraph (LocalCFG) between a producer and
/// consumer. The subgraph is reconstructed as a region with unique entry
/// (producer) and exit (sink).
std::unique_ptr<LocalCFG> buildLocalCFGRegion(OpBuilder &builder,
                                              Block *origProd, Block *origCons,
                                              const BlockIndexing &bi);

/// Constructs a NEW LocalCFG that represents the Decision Graph from a
/// raw LocalCFG, retaining only the blocks in `dependencies` plus the
/// consumer and sink.
std::unique_ptr<LocalCFG> buildDecisionGraph(
    const LocalCFG &rawGraph, const DenseSet<Block *> &dependencies,
    const DenseMap<Block *, bool> &muxConstraints = DenseMap<Block *, bool>());

// ===--------------------------------------------------------------------=== //
// Path enumeration and expression generation
// ===--------------------------------------------------------------------=== //

/// Enumerates all paths from producer to consumer in a LocalCFG and returns
/// the minimized SOP boolean expression representing the reaching condition.
boolean::BoolExpression *enumeratePaths(const LocalCFG &lcfg,
                                        const BlockIndexing &bi,
                                        const DenseSet<Block *> &controlDeps);

// ===--------------------------------------------------------------------=== //
// Circuit construction (BDD → hardware)
// ===--------------------------------------------------------------------=== //

/// Recursively converts a BDD to a Mux Tree circuit.
Value bddToCircuit(
    mlir::OpBuilder &builder, boolean::BDD *bdd, Block *block,
    SignalRegistry &registry, PathContext currentPath, const BlockIndexing &bi,
    DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
    ShadowCFG *shadow = nullptr, IntegerAttr forcedBBAttr = {});

/// Main entry point of the token distribution logic.
void buildDistributionNetwork(
    mlir::OpBuilder &builder, const LocalCFG &lcfg, const BlockIndexing &bi,
    SignalRegistry &registry,
    DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands,
    ShadowCFG *shadow = nullptr);

/// Compute the topological rank of original blocks from a LocalCFG.
/// Maps each original block (via origMap) to its index in topoOrder.
DenseMap<Block *, unsigned> computeTopoRank(const LocalCFG &lcfg);

/// Convert a boolean expression to a hardware mux-tree circuit.
/// Handles the full pipeline: minimize → topo-sort cofactors → buildBDD →
/// bddToCircuit. `varRank` maps original CFG blocks to their topological rank
/// (determines BDD variable ordering).
Value expressionToCircuit(
    OpBuilder &builder, boolean::BoolExpression *expr,
    const DenseMap<Block *, unsigned> &varRank, Block *insertBlock,
    SignalRegistry &registry, const BlockIndexing &bi,
    DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands = nullptr,
    ShadowCFG *shadow = nullptr, IntegerAttr forcedBBAttr = {});

/// Convenience overload: computes varRank from a LocalCFG's topoOrder.
Value expressionToCircuit(
    OpBuilder &builder, boolean::BoolExpression *expr,
    const LocalCFG &rankSource, Block *insertBlock, SignalRegistry &registry,
    const BlockIndexing &bi,
    DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands = nullptr,
    ShadowCFG *shadow = nullptr, IntegerAttr forcedBBAttr = {});

/// Compute the loop backedge condition for a self-loop at loopHeader.
/// Runs the full suppression pipeline for a header-to-header path:
///   buildLocalCFGRegion → CDA → CyclicGraphManager → CyclicDemotionHelper →
///   buildDistributionNetwork → enumeratePaths → expressionToCircuit.
/// Returns the condition Value that is true when the loop iterates back.
Value computeLoopBackedgeCondition(
    OpBuilder &builder, Block *loopHeader, Block *insertBlock,
    const BlockIndexing &bi,
    DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands = nullptr,
    ShadowCFG *shadow = nullptr);

// ===--------------------------------------------------------------------=== //
// Top-level suppression entry point
// ===--------------------------------------------------------------------=== //

/// Apply the suppression algorithm for a non-loop producer-consumer pair.
/// This is the main entry point for building suppression circuits.
void insertDirectSuppression(mlir::OpBuilder &builder,
                             handshake::FuncOp &funcOp, Operation *consumer,
                             Value connection, ShadowCFG &shadow);

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_FTDSUPPRESSION_H
