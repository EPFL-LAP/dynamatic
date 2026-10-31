//===- AnnotatedCfg.h - Annotated CFGs for operand routing 2 ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builds the annotated CFG of every routing, from the base CFG, and
// optimizes each one for routing.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_CONVERSION_OPERANDROUTING2_ANNOTATEDCFG_H
#define EXPERIMENTAL_CONVERSION_OPERANDROUTING2_ANNOTATEDCFG_H

#include "dynamatic/Conversion/OperandRouting2/BaseCfg.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include <utility>

namespace dynamatic {
namespace experimental {
namespace operand_routing2 {

using mlir::MapVector;
using mlir::TypedValue;

/// Returns every block use of the function exactly once,
/// in a deterministic order.
SetVector<BlockUse> collectBlockUses(Region &body);

/// Checks whether the block produces the value
bool isProducedIn(Value value, Block *block);

/// Builds the CFG for routing these block uses:
/// a copy of the base CFG,
/// annotated with the producers and live-outs of the block uses.
/// Also stores a copy of the blockUses set vector-
/// the routing algorithm processes many block arguments,
/// but does something different if a block argument is a block use
/// of the CFG it is routing
Cfg buildCfg(const Cfg &base, const SetVector<BlockUse> &blockUses);

enum class McOrLsq : unsigned { MC, LSQ };

/// A value known to be a memref
using MemrefValue = TypedValue<MemRefType>;

/// A value known to be a dataless control channel
using ControlValue = TypedValue<handshake::ControlType>;

/// The CFG for routing the control inputs of one memory interface,
/// sequentialized by a single memory token.
struct MemoryInterfaceCfg {
  Cfg cfg;
  // the basic blocks the memory interface takes control inputs from
  SetVector<Block *> interfacingBlocks;
  // whether the interface takes the region's end control
  bool takesEnd;
};

/// Builds the CFGs for routing the control inputs
/// of each memory interface of the function:
/// per memory region (each memref with a memStart),
/// one for its memory controller if it has one,
/// and one for its LSQ if it has one.
MapVector<std::pair<MemrefValue, McOrLsq>, MemoryInterfaceCfg>
collectMemoryInterfaceCfgs(
    const Cfg &base, const MapVector<MemrefValue, ControlValue> &memStarts);

/// Optimizes the CFG for routing:
/// removes the control flow
/// that does not concern its needed blocks,
/// one rewrite at a time, until none applies.
void optimizeCfg(Cfg &cfg);

} // namespace operand_routing2
} // namespace experimental
} // namespace dynamatic

namespace llvm {
template <>
struct DenseMapInfo<dynamatic::experimental::operand_routing2::McOrLsq> {
  using McOrLsq = dynamatic::experimental::operand_routing2::McOrLsq;
  static McOrLsq getEmptyKey() { return static_cast<McOrLsq>(~0U); }
  static McOrLsq getTombstoneKey() { return static_cast<McOrLsq>(~0U - 1); }
  static unsigned getHashValue(McOrLsq mcOrLsq) {
    return static_cast<unsigned>(mcOrLsq);
  }
  static bool isEqual(McOrLsq lhs, McOrLsq rhs) { return lhs == rhs; }
};
} // namespace llvm

#endif // EXPERIMENTAL_CONVERSION_OPERANDROUTING2_ANNOTATEDCFG_H
