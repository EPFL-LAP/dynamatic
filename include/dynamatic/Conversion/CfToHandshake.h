//===- CfToHandhsake.h - Convert func/cf to handhsake dialect ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --lower-cf-to-handshake conversion pass along with a
// helper class for performing the lowering.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_CF_TO_HANDSHAKE_H
#define DYNAMATIC_CONVERSION_CF_TO_HANDSHAKE_H

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Transforms/DialectConversion.h"
#include <set>

namespace dynamatic {
// Represents a memory dependeency bteween 2 blocks: the producer basic block
// prodBb, and the consumer basic block consBb isBackward is used to indicate if
// the producer and the consumer are in a loop
struct ProdConsMemDep {
  Block *prodBb;
  Block *consBb;
  bool isBackward;

  ProdConsMemDep(Block *prod, Block *cons, bool backward)
      : prodBb(prod), consBb(cons), isBackward(backward) {}
};

// A group represents all operations belonging to the same basic block bb
struct Group {
  Block *bb;
  std::set<Group> preds;
  std::set<Group> succs;

  Group(Block *b) : bb(b) {}

  bool operator<(const Group &other) const { return bb < other.bb; }
};

// Structure that stores loop information of a Block.
struct BlockLoopInfo {
  mlir::CFGLoop *loop = nullptr;
  bool isHeader = false;
  bool isExit = false;
  bool isLatch = false;
};

/// This class is strongly inspired by CIRCT's own `HandshakeLowering` class. It
/// provides all the conversion steps necessary to concert a func-level function
/// into a matching handshake-level function.
class HandshakeLowering {
public:
  /// Groups memory operations by interface and group for a given memory region.
  struct MemAccesses {
    /// Memory operations for a simple memory controller, grouped by
    /// originating basic block.
    llvm::MapVector<Block *, SmallVector<Operation *>> mcPorts;
    /// Memory operations for an LSQ, grouped by belonging LSQ group.
    llvm::MapVector<unsigned, SmallVector<Operation *>> lsqPorts;
  };

  /// Groups information to "rewire the IR" around a particular merge-like
  /// operation.
  struct MergeOpInfo {
    /// The merge-like operation under consideration.
    handshake::MergeLikeOpInterface mergeLikeOp;
    /// The original block argument that the merge-like operation "replaces".
    BlockArgument blockArg;
    /// All data operands to the merge-like operation that need to be resolved
    /// during branch insertion.
    SmallVector<Backedge> dataEdges;
    /// An optional index operand that needs to be resolved for mux-like
    /// operations.
    std::optional<Backedge> indexEdge{};
  };

  /// Groups information to rewire the IR around merge-like operations by owning
  /// basic block (which must still exist).
  using BlockOps = DenseMap<Block *, std::vector<MergeOpInfo>>;

  /// Stores a mapping between memory regions (identified by the function
  /// argument they correspond to) and the set of memory operations referencing
  /// them.
  using MemInterfacesInfo = llvm::MapVector<Value, MemAccesses>;

  /// Constructor simply takes the region being lowered and a reference to the
  /// top-level name analysis.
  explicit HandshakeLowering(Region &region, NameAnalysis &nameAnalysis)
      : region(region), nameAnalysis(nameAnalysis) {}

  /// Creates the control-only network by adding a control-only argument to the
  /// region's entry block and forwarding it through all basic blocks.
  LogicalResult createControlNetwork(ConversionPatternRewriter &rewriter);

  /// Adds merge-like operations after all block arguments within the region,
  /// then removes all block arguments and corresponding branch operands. This
  /// always succeeds.
  LogicalResult addMergeOps(ConversionPatternRewriter &rewriter);

  /// Adds handshake-level branch-like operations before all cf-level
  /// branch-like terminators within the region. This needs to happen after
  /// merge-insertion because it also replaces data operands of merge-like
  /// operations with the result value(s) of inserted branch-like operations.
  /// This always succeeds.
  LogicalResult addBranchOps(ConversionPatternRewriter &rewriter);

  /// Identifies all memory interfaces and their associated operations in the
  /// function, replaces all load/store-like operations by their handshake
  /// counterparts, and fills `memInfo` with information about which operations
  /// use which interface.
  LogicalResult replaceMemoryOps(ConversionPatternRewriter &rewriter,
                                 MemInterfacesInfo &memInfo);

  /// Verifies that LSQ groups derived from input IR annotations make sense
  /// (check for linear dominance property within each group and cross-group
  /// control signal compatibility). Then, instantiates all memory interfaces
  /// and connects them to their respective load/store operations. For each
  /// memory region:
  /// - A single `handshake::MemoryControllerOp` will be instantiated if all of
  /// its accesses indicate that they should connect to an MC.
  /// - A single `handshake::LSQOp` will be instantiated if none of
  /// its accesses indicate that they should connect to an LSQ.
  /// - Both a `handhsake::MemoryControllerOp` and `handhsake::LSQOp` will be
  /// instantiated if some but not all of its accesses indicate that they should
  /// connect to an LSQ.
  LogicalResult
  verifyAndCreateMemInterfaces(ConversionPatternRewriter &rewriter,
                               MemInterfacesInfo &memInfo);

  /// Converts each `func::CallOp` operation to an equivalent
  /// `handshake::InstanceOp` operation. This always succeeds.
  LogicalResult convertCalls(ConversionPatternRewriter &rewriter);

  /// Connect constants to the rest of the circuit. Constants are triggered by a
  /// source if their successor is not a branch/return or memory operation.
  /// Otherwise they are triggered by the control-only network.
  LogicalResult connectConstants(ConversionPatternRewriter &rewriter);

  /// Replaces undefined operations (mlir::LLVM::UndefOp) with a default "0"
  /// constant triggered by the enclosing block's control merge.
  LogicalResult replaceUndefinedValues(ConversionPatternRewriter &rewriter);

  /// Sets an integer "bb" attribute on each operation to identify the basic
  /// block from which the operation originates in the std-level IR.
  LogicalResult idBasicBlocks(ConversionPatternRewriter &rewriter);

  /// Creates the region's return network by sequentially moving all blocks'
  /// operations to the entry block, replacing func::ReturnOp's with
  /// handshake::ReturnOp's, deleting all block terminators and non-entry
  /// blocks, merging the results of all return statements, and creating the
  /// region's end operation./// This class is strongly inspired by CIRCT's own
  /// `HandshakeLowering` class. It
  /// provides all the conversion steps necessary to concert a func-level
  /// function into a matching handshake-level function.
  LogicalResult createReturnNetwork(ConversionPatternRewriter &rewriter);

  /// Returns the entry control value for operations contained within this
  /// block.
  Value getBlockEntryControl(Block *block) const {
    auto it = blockControls.find(block);
    assert(it != blockControls.end() &&
           "No block entry control value registerred for this block!");
    return it->second;
  }

  /// Set the control value of a basic block.
  void setBlockEntryControl(Block *block, Value v) {
    blockControls[block] = v;
  };

  /// Returns a reference to the region being lowered.
  Region &getRegion() { return region; }

  //----------Construction of Allocation Network----------

  // interfaces dataflow circuits with LSQs
  LogicalResult addSmartControlForLSQ(ConversionPatternRewriter &rewriter,
                                      MemInterfacesInfo &memInfo);

  bool sameLoop(Block *source, Block *dest);

  // identify all the memory dependencies between the predecessors of an LSQ.
  // This
  // is the first step towards making memory deps explicit
  void identifyMemDeps(SmallVector<Operation *> &operations,
                       std::vector<ProdConsMemDep> &allMemDeps);

  // build a dependence graph betweeen the groups
  void constructGroupsGraph(const SmallVector<Operation *> &operations,
                            std::vector<ProdConsMemDep> &allMemDeps,
                            std::set<Group> &groups);

  LogicalResult print(ConversionPatternRewriter &rewriter);

protected:
  /// The region being lowered.
  Region &region;
  /// Start point of the control-only network
  BlockArgument startCtrl;

  /// Inserts a merge-like operation in the IR for the block argument and
  /// returns information necessary to rewire the IR around the new operation
  /// once all merges have been inserted. A control-merge is inserted for
  /// control-only (data-less) arguments. For other types of arguments, a
  /// non-deterministic merge is inserted for blocks with 0 or a single
  /// predecessor while a mux is inserted for blocks with multiple predecessors.
  MergeOpInfo insertMerge(BlockArgument blockArg, BackedgeBuilder &edgeBuilder,
                          ConversionPatternRewriter &rewriter);

private:
  /// Associates basic blocks of the region being lowered to their respective
  /// control value.
  DenseMap<Block *, Value> blockControls;
  /// Name analysis to name new memory operations as they are created and keep
  /// reference accesses in memory dependencies consistent.
  NameAnalysis &nameAnalysis;
};

/// Pointer to function lowering a region using a conversion pattern rewriter.
using RegionLoweringFunc =
    llvm::function_ref<LogicalResult(Region &, ConversionPatternRewriter &)>;

/// Partially lowers a region using a provided lowering function.
LogicalResult partiallyLowerRegion(const RegionLoweringFunc &loweringFunc,
                                   Region &region);

/// Runs a partial lowering method on an instance of the class the method
/// belongs to. We need two variadic template parameters because arguments
/// provided to this function may be slightly differeprod_cons_mem_depnt but
/// convertible to the arguments expected by the partial lowering method.
/// Success status is forwarded from the partial lowering method.
template <typename T, typename... TArgs1, typename... TArgs2>
static LogicalResult runPartialLowering(
    T &instance,
    LogicalResult (T::*memberFunc)(ConversionPatternRewriter &, TArgs2...),
    TArgs1 &...args) {
  return partiallyLowerRegion(
      [&](Region &, ConversionPatternRewriter &rewriter) -> LogicalResult {
        return (instance.*memberFunc)(rewriter, args...);
      },
      instance.getRegion());
}

#define GEN_PASS_DECL_CFTOHANDSHAKE
#define GEN_PASS_DEF_CFTOHANDSHAKE
#include "dynamatic/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createCfToHandshake();

// Function that runs loop analysis on the funcOp Region.
DenseMap<Block *, BlockLoopInfo> findLoopDetails(mlir::CFGLoopInfo &li,
                                                 Region &funcReg);

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_CF_TO_HANDSHAKE_H
