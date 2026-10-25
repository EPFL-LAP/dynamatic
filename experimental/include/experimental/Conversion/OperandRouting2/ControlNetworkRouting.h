//===- ControlNetworkRouting.h - Control-network routing --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The control-network routing algorithm, run on an annotated CFG.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_CONVERSION_OPERANDROUTING2_CONTROLNETWORKROUTING_H
#define EXPERIMENTAL_CONVERSION_OPERANDROUTING2_CONTROLNETWORKROUTING_H

#include "dynamatic/Support/LLVM.h"
#include "experimental/Conversion/OperandRouting2/AnnotatedCfg.h"
#include "experimental/Conversion/OperandRouting2/BaseCfg.h"
#include "mlir/IR/Builders.h"

namespace dynamatic {
namespace experimental {
namespace operand_routing2 {

/// What the memory routing delivers to one memory region's interfaces,
/// in the form the memory interface builder takes
struct MemoryInterfaceInputs {
  // the MC's control input for each basic block, keyed by its number
  DenseMap<unsigned, Value> mcControls;
  // the LSQ's group allocation for each basic block, keyed by its number
  DenseMap<unsigned, Value> lsqControls;
  // the end control, for whichever interface takes it
  Value ctrlEnd;
};

/// Routes the block uses of each optimized CFG,
/// and the single memory token of each memory interface on its optimized
/// CFG, starting from its region's memStart.
/// Returns, per memory region with accesses, what its memory routing
/// delivers to its interfaces.
MapVector<MemrefValue, MemoryInterfaceInputs> routeCfgs(
    Region &body, const MapVector<MemrefValue, ControlValue> &memStarts,
    ArrayRef<Cfg> useCfgs,
    MapVector<std::pair<MemrefValue, McOrLsq>, MemoryInterfaceCfg> &memoryCfgs,
    OpBuilder &builder);

} // namespace operand_routing2
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_CONVERSION_OPERANDROUTING2_CONTROLNETWORKROUTING_H
