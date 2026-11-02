//===- AnnotatedCfg.h - Annotated CFGs --------------------------*- C++ -*-===//
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

#ifndef EXPERIMENTAL_CONVERSION_CFTOHANDSHAKE_ANNOTATEDCFG_H
#define EXPERIMENTAL_CONVERSION_CFTOHANDSHAKE_ANNOTATEDCFG_H

#include "dynamatic/Conversion/CfToHandshake/BaseCfg.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include <utility>

namespace dynamatic {
namespace experimental {
namespace cf_to_handshake {

using mlir::MapVector;
using mlir::TypedValue;

/// Returns every block use of the function exactly once,
/// in a deterministic order.
SetVector<BlockUse> collectBlockUses(Region &body);

/// Checks whether the basic block produces the value
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

enum class TriggerKind : unsigned { Memory, Speculation };

/// The CFG for routing a single control token,
/// which triggers something in each of its trigger blocks:
/// the control inputs of a memory interface,
/// or the speculator of a speculated block.
struct TriggerCfg {
  Cfg cfg;
  TriggerKind kind;
  SetVector<Block *> triggerBlocks;
  // whether the cfg has an additional "end" trigger in the last exit block
  bool triggersEnd;
};

/// Builds the CFGs for routing the control inputs
/// of each memory interface of the function:
/// per memory region (each memref with a memStart),
/// one for its memory controller if it has one,
/// and one for its LSQ if it has one.
MapVector<std::pair<MemrefValue, McOrLsq>, TriggerCfg>
collectMemoryInterfaceCfgs(
    const Cfg &base, const MapVector<MemrefValue, ControlValue> &memStarts);

/// Builds the CFGs for routing the speculator triggers of the function:
/// one per basic block holding an op with the `dynamatic.speculate`
/// attribute, with that block as its only trigger block.
SmallVector<TriggerCfg, 0> collectSpeculationCfgs(const Cfg &base);

/// Optimizes the CFG for routing:
/// removes the control flow
/// that does not concern its needed blocks,
/// one rewrite at a time, until none applies.
void optimizeCfg(Cfg &cfg);

/// Optimizes a memory interface CFG for routing
void optimizeMemoryCfg(TriggerCfg &memoryCfg);

} // namespace cf_to_handshake
} // namespace experimental
} // namespace dynamatic

namespace llvm {
/// Lets the McOrLsq enum be used as a DenseMap key
template <>
struct DenseMapInfo<dynamatic::experimental::cf_to_handshake::McOrLsq> {
  using McOrLsq = dynamatic::experimental::cf_to_handshake::McOrLsq;
  static McOrLsq getEmptyKey() { return static_cast<McOrLsq>(~0U); }
  static McOrLsq getTombstoneKey() { return static_cast<McOrLsq>(~0U - 1); }
  static unsigned getHashValue(McOrLsq mcOrLsq) {
    return static_cast<unsigned>(mcOrLsq);
  }
  static bool isEqual(McOrLsq lhs, McOrLsq rhs) { return lhs == rhs; }
};
} // namespace llvm

#endif // EXPERIMENTAL_CONVERSION_CFTOHANDSHAKE_ANNOTATEDCFG_H
