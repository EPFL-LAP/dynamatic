//===- TraversalUtils.h - Paths of tokens through a circuit -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Describes where a token can go when it passes through a unit of a Handshake
/// circuit, and walks the circuit along these flow edges. The walks find the
/// units that a token can reach and collect the slots along the paths a token
/// can take, as needed by the invariants of the formal properties.
///
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ANALYSIS_FORMALPROPERTYANNOTATION_UTILS_TRAVERSALUTILS_H
#define EXPERIMENTAL_ANALYSIS_FORMALPROPERTYANNOTATION_UTILS_TRAVERSALUTILS_H

#include "dynamatic/Dialect/Handshake/HandshakeOpInternalStateNamer.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"
#include <vector>

namespace dynamatic {

/// The results of the owner of `operand` through which a token arriving on
/// `operand` can leave the unit. A token leaves no result of a unit that
/// consumes it or passes it outside of the circuit (e.g., a sink or a store).
/// Reports a fatal error, with an error on the unit, if the flow through the
/// unit is not modeled.
SmallVector<OpResult> getFlowSuccessors(OpOperand &operand);

/// The operands of the owner of `result` whose tokens can leave the unit
/// through `result`. This is the inverse of `getFlowSuccessors`.
SmallVector<OpOperand *> getFlowPredecessors(OpResult result);

/// The units that a token on any of the `starts` channels can reach by going
/// forward along flow edges. A unit for which `isBoundary` holds is reached,
/// but the walk does not continue past it. If `isFollowed` is set, the walk
/// only goes along the channels for which it holds (e.g., the channels of a
/// subgraph), including the starts.
llvm::SetVector<Operation *>
reachForward(ArrayRef<Value> starts, function_ref<bool(Operation *)> isBoundary,
             function_ref<bool(Value)> isFollowed = nullptr);

/// The units from which a token can reach any of the `starts` channels, found
/// by going backward along flow edges. A unit for which `isBoundary` holds is
/// reached, but the walk does not continue past it. The walk also stops at
/// the arguments of the function. If `isFollowed` is set, the walk only goes
/// along the channels for which it holds, including the starts.
llvm::SetVector<Operation *>
reachBackward(ArrayRef<Value> starts,
              function_ref<bool(Operation *)> isBoundary,
              function_ref<bool(Value)> isFollowed = nullptr);

/// A path that a token can take, described by the slots it goes through.
struct SlotPath {
  /// The sent states of the eager fork outputs that the path takes before its
  /// first slot.
  std::vector<handshake::EagerForkSentNamer> leadingSents;
  /// The slots on the path, in the order of the path. Each slot is effective:
  /// its copied sents are the sent states of the eager fork outputs that the
  /// path takes after the slot and before the next one, so that the slot only
  /// counts if its token has not yet been copied along the path.
  std::vector<handshake::EffectiveSlotNamer> slots;
  /// The operand at which the path ends.
  OpOperand *end = nullptr;
};

/// What `findSlotPaths` does when a path reaches an operand.
enum class PathStep {
  /// Continue the path through the unit.
  Traverse,
  /// End the path at the operand.
  End,
  /// Discard the path.
  Drop,
  /// Report a fatal error, as the path should not reach the unit.
  Unexpected
};

/// The paths that a token on `start` can take, from `start` to the operands at
/// which `step` ends them. The paths begin with `initialSlots`, e.g., the slot
/// of the unit producing `start`. A path never goes through the same unit
/// twice. The paths are listed in depth-first order, where the last result of
/// a unit is followed first. Reports a fatal error, with an error on the unit,
/// if `step` is `Unexpected` or the flow through a traversed unit is not
/// modeled.
std::vector<SlotPath>
findSlotPaths(Value start,
              std::vector<handshake::EffectiveSlotNamer> initialSlots,
              function_ref<PathStep(OpOperand &)> step);

} // namespace dynamatic

#endif // EXPERIMENTAL_ANALYSIS_FORMALPROPERTYANNOTATION_UTILS_TRAVERSALUTILS_H
