//===- FtdSupport.h - FTD conversion support -------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares utility functions and data structures shared across the FTD
// algorithm. Includes CFG analysis helpers, type utilities, annotation
// constants, and the ShadowCFG bridge between the original CFG and the
// flattened handshake IR.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_HANDSHAKE_EXPERIMENTAL_SUPPORT_FTDSUPPORT_H
#define DYNAMATIC_HANDSHAKE_EXPERIMENTAL_SUPPORT_FTDSUPPORT_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

// ===--------------------------------------------------------------------=== //
// Annotation constants used throughout the FTD algorithm.
// ===--------------------------------------------------------------------=== //

/// Annotation to use in the IR when an operation needs to be skipped by the FTD
/// algorithm.
constexpr llvm::StringLiteral FTD_OP_TO_SKIP("ftd.skip");
/// Annotation to identify muxes inserted with the `addGsaGates`
/// functionalities.
constexpr llvm::StringLiteral FTD_EXPLICIT_MU("ftd.MU");
constexpr llvm::StringLiteral FTD_EXPLICIT_GAMMA("ftd.GAMMA");
/// Temporary annotation to be used with merges created with the
/// `createPhiNetwork` functionality, which will then be converted into muxes.
constexpr llvm::StringLiteral NEW_PHI("nphi");
/// Annotation to use for initial merges and initial false constants.
constexpr llvm::StringLiteral FTD_INIT_MERGE("ftd.imerge");
/// Annotation to use for regeneration multiplexers.
constexpr llvm::StringLiteral FTD_REGEN("ftd.regen");
/// Annotation for condition variable placeholders.
constexpr llvm::StringLiteral FTD_COND_VAR("ftd.cvar");

// ===--------------------------------------------------------------------=== //
// ShadowCFG
// ===--------------------------------------------------------------------=== //

/// A temporary shadow of the original CFG, built after CfToHandshake
/// conversion flattens everything.  Encapsulates the shadow Region
/// (with real CF terminators) plus a condition-value map that bridges
/// shadow analysis to real handshake Values.
///
/// All analysis infrastructure (BlockIndexing, CFGLoopInfo, path
/// enumeration, dominance) operates on getRegion() as if the original
/// CFG were still alive.  The only thing the shadow cannot provide
/// natively is the real handshake condition Value for each cond_br
/// block — getCondition() provides that.
struct ShadowCFG {
  mlir::func::FuncOp shadowFunc;
  llvm::DenseMap<unsigned, mlir::Value> conditionMap;

  mlir::Region &getRegion() { return shadowFunc.getBody(); }

  mlir::Block *getBlock(unsigned bbIdx) {
    for (auto [i, blk] : llvm::enumerate(getRegion()))
      if (i == bbIdx)
        return &blk;
    llvm_unreachable("BB index out of range in shadow CFG");
  }

  unsigned getBlockIndex(mlir::Block *block) {
    for (auto [i, blk] : llvm::enumerate(getRegion()))
      if (&blk == block)
        return i;
    llvm_unreachable("Block not found in shadow CFG");
  }

  /// Get the real handshake condition Value for the cond_br in block bbIdx.
  /// Returns nullptr if the block had an unconditional branch.
  mlir::Value getCondition(unsigned bbIdx) {
    auto it = conditionMap.find(bbIdx);
    return (it != conditionMap.end()) ? it->second : nullptr;
  }

  mlir::Value getCondition(mlir::Block *block) {
    return getCondition(getBlockIndex(block));
  }

  void destroy() {
    if (shadowFunc)
      shadowFunc.erase();
  }
};

// ===--------------------------------------------------------------------=== //
// BlockIndexing
// ===--------------------------------------------------------------------=== //

/// Class to associate an index to each block, so that if block Bi dominates
/// block Bj then i < j. While this is guaranteed by the MLIR CFG construction,
/// it cannot really be given for granted, thus it is more convenient to have a
/// custom indexing.
class BlockIndexing {

  /// Map to store the connection between indices and blocks.
  DenseMap<unsigned, Block *> indexToBlock;

  /// Map to store the connection between blocks and indices.
  DenseMap<Block *, unsigned> blockToIndex;

public:
  /// Build the map out of a region.
  BlockIndexing(mlir::Region &region);

  /// Get a block out of an index.
  std::optional<Block *> getBlockFromIndex(unsigned index) const;

  /// Get a block out of a string condition in the format `cX` where X is a
  /// number.
  std::optional<Block *> getBlockFromCondition(StringRef condition) const;

  /// Get the index of a block.
  std::optional<unsigned> getIndexFromBlock(Block *bb) const;

  /// Return true if the index of bb1 is greater than the index of bb2.
  bool isGreater(Block *bb1, Block *bb2) const;

  /// Return true if the index of bb1 is smaller than the index of bb2.
  bool isLess(Block *bb1, Block *bb2) const;

  /// Given a block whose name is `^BBN` (where N is an integer) return a string
  /// in the format `cN`, used to identify the condition which allows the block
  /// to be executed. The adopted index is retrieved from the BlockIndexing.
  std::string getBlockCondition(Block *block) const;
};

// ===--------------------------------------------------------------------=== //
// CFG Analysis Utilities
// ===--------------------------------------------------------------------=== //

/// Checks if the source and destination are in a loop
/// (including any of their ancestor loops).
bool isSameLoopBlocks(Block *source, Block *dest, const mlir::CFGLoopInfo &li);

/// Checks whether the loop of `source` is the same as, or nested inside,
/// the loop of `dest`. Returns true if `source` lives in
/// `dest`'s loop or in any inner loop of it.
bool isSameOrInnerLoopBlocks(Block *source, Block *dest,
                             const mlir::CFGLoopInfo &li);

/// Gets all the paths from block `start` to block `end` using a DFS search.
/// If `blockToTraverse` is non null, then we want the paths having
/// `blockToTraverse` in the path; filters paths that do not contain blocks in
/// `blocksToAvoid`.
std::vector<std::vector<Block *>>
findAllPaths(Block *start, Block *end, const BlockIndexing &bi,
             Block *blockToTraverse = nullptr,
             ArrayRef<Block *> blocksToAvoid = {});

/// Given a sequence of block, find a boolean expression defining the conditions
/// for which the path is traversed. If one edge is unconditional, then no
/// condition is added; otherwise, the condition for the conditional branch is
/// added (either direct or negated). The list of blocks whose condition is
/// considered is saved in `blockIndexSet`. If `ignoreDeps` is false, then a
/// condition is added only if the source block was in the set `deps`.
boolean::BoolExpression *
getPathExpression(ArrayRef<Block *> path, DenseSet<unsigned> &blockIndexSet,
                  const BlockIndexing &bi,
                  const DenseSet<Block *> &deps = DenseSet<Block *>(),
                  bool ignoreDeps = true);

/// A lightweight DFS to check if 'end' is reachable from 'start'.
bool isReachable(Block *start, Block *end);

// ===--------------------------------------------------------------------=== //
// Type Utilities
// ===--------------------------------------------------------------------=== //

/// Return the channelified version of the input type.
Type channelifyType(Type type);

/// Get an array of `size` elements all identical to the channelified type.
SmallVector<Type> getListTypes(Type inputType, unsigned size = 2);

// ===--------------------------------------------------------------------=== //
// IR Attribute Utilities
// ===--------------------------------------------------------------------=== //

/// Compute the positional index of `block` in its parent region and set
/// the handshake.bb attribute on `op`.
void setBBAttr(Operation *op, Block *block, OpBuilder &builder);

/// Set the handshake.bb attribute on `op` from an existing attribute.
void setBBAttr(Operation *op, IntegerAttr bbAttr);

/// Set the handshake.bb attribute on `op`, preferring `bbAttr` if available,
/// otherwise computing from `block`.
void setBBAttrWithFallback(Operation *op, IntegerAttr bbAttr, Block *block,
                           OpBuilder &builder);

/// Build a `handshake.bb` IntegerAttr (32-bit unsigned) for `bbIdx`.
IntegerAttr getBBIndexAttr(MLIRContext *ctx, unsigned bbIdx);

/// Get or create a SourceOp placeholder in `condBlock` representing the
/// condition of that block's terminator. Reuses existing placeholder if one
/// exists. Tagged with FTD_COND_VAR and FTD_OP_TO_SKIP so that FTD skips it.
Value getOrCreateCondPlaceholder(Block *condBlock, OpBuilder &builder);

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_HANDSHAKE_EXPERIMENTAL_SUPPORT_FTDSUPPORT_H
