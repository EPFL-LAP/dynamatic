//===- CFG.h - CFG-related analysis and helpers -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains helpers for CFG-style analysis in Handshake functions. These help in
// identifying properties of the cf-level CFG within Handshake-level IR.
//
// In particular, this file offers ways to interact with the concept of "logical
// basic blocks" in Handshake functions. These are not basic blocks in the MLIR
// sense since Handshake function only have a single block. They instead map to
// the original basic blocks that the cf-level IR possessed before conversion to
// Handshake-level. Any Handshake operation may optionally have a "bb" integer
// attribute indicating the basic block it logically belonged to at the
// cf-level.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_CFG_H
#define DYNAMATIC_SUPPORT_CFG_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

/// Operation attribute to identify the basic block the operation originated
/// from in the std-level IR.
const std::string BB_ATTR = "bb";

/// ID of entry basic block of every Handshake function.
const unsigned ENTRY_BB = 0;

/// This struct groups the operations of a handshake::FuncOp in "blocks" based
/// on the "bb" attribute potentially attached to each operation.
struct LogicBBs {
  /// Maps each block ID to the operations (in program order) that are tagged
  /// with it.
  mlir::DenseMap<unsigned, mlir::SmallVector<mlir::Operation *>> blocks;
  /// List of operations (in program order) that do not belong to any block.
  mlir::SmallVector<mlir::Operation *> outOfBlocks;
};

/// Groups the operations of a function into "blocks" based on the "bb"
/// attribute of each operation.
LogicBBs getLogicBBs(circt::handshake::FuncOp funcOp);

/// If the source operation belongs to a logical BB, makes the destination
/// operation part of the same BB and returns true; otherwise return false.
bool inheritBB(Operation *srcOp, Operation *dstOp);

/// If the source value is the result of an operation that belongs to a logical
/// BB, makes the destination operation part of the same BB and returns true;
/// otherwise return false. If the source value is a block argument, make the
/// destination operation part of the entry BB.
bool inheritBBFromValue(Value val, Operation *dstOp);

/// Thin wrapper around an attribute access to the "bb" attribute.
std::optional<unsigned> getLogicBB(Operation *op);

/// A pair of BB IDs to represent the blocks that a channel connects. In case of
/// an inner channel, these blocks may be identical.
struct BBEndpoints {
  // The source/predecessor basic block.
  unsigned srcBB;
  // The destination/successor basic block.
  unsigned dstBB;
};

/// Gets the basic block endpoints of a channel (represented as an MLIR value
/// accompanied by one of its users). These are the blocks which operations
/// "near" the value belong to (the source block which is reached by
/// backtracking through the value's def-use chain and the destination block
/// which is reached by following the value's uses). On successful
/// identification of these blocks, the function returns true and the block
/// endpoints are set, otherwise the function returns false.
bool getBBEndpoints(Value val, Operation *user, BBEndpoints &endpoints);

/// Determines the basic block endpoints of a value which must have a single
/// user (see documentation of overriden function for more details).
bool getBBEndpoints(Value val, BBEndpoints &endpoints);

/// Determines whether the value is a backedge i.e., whether the channel
/// corresponding to the value is located between a branch-like operation and a
/// merge-like operation, where the merge-like operation happens semantically
/// "before" the branch-like operation. This function can only correctly
/// identify backedges if the circuit's branches and merges are associated to
/// basic blocks (otherwise it will always return false). `user` must be one of
/// `val`'s users.
bool isBackedge(Value val, Operation *user, BBEndpoints *endpoints = nullptr);

/// Determines whether the value is a backedge. The value must have a single
/// user (the function will assert if that is not the case).
bool isBackedge(Value val, BBEndpoints *endpoints = nullptr);

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_CFG_H
