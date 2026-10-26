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

#include "dynamatic/Conversion/OperandRouting2/AnnotatedCfg.h"
#include "dynamatic/Conversion/OperandRouting2/BaseCfg.h"
#include "dynamatic/Conversion/OperandRouting2/TempOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"

namespace dynamatic {
namespace experimental {
namespace operand_routing2 {

namespace cf = mlir::cf;

/// Dictionary mapping from each basic block to its branch condition
///
/// Must be built before conversion
class BlockToCond {
public:
  /// Place a cond reader on the condition of each CondBranchOp,
  /// stored under the basic block of the CondBranchOp.
  ///
  /// Conversion makes each branch condition a handshake channel,
  /// but a CondBranchOp condition must be an i1,
  /// so each CondBranchOp takes a temporary i1 producer as its condition
  /// instead
  BlockToCond(Region &body, OpBuilder &builder) {
    // rebuilding a branch erases the op we are on, so use early inc
    for (auto condBrOp :
         llvm::make_early_inc_range(body.getOps<cf::CondBranchOp>())) {
      Block *block = condBrOp->getBlock();
      builder.setInsertionPoint(condBrOp);

      // add a cond reader and store it under the basic block:
      // it follows the condition through conversion,
      // and is the block use of the condition
      blockToCondReader[block] = TempCons(builder, condBrOp.getCondition());

      // a temporary i1 producer, for the branch to take as its condition
      blockToTempProd[block] = TempProd(builder, builder.getI1Type());

      // rebuild the branch with the temporary producer as its condition
      cf::CondBranchOp::create(
          builder, condBrOp.getLoc(), blockToTempProd[block].getValue(),
          condBrOp.getTrueDest(), condBrOp.getTrueOperands(),
          condBrOp.getFalseDest(), condBrOp.getFalseOperands());
      condBrOp.erase();
    }
  }

  /// index by basic block to get branch condition
  Value operator[](Block *block) const {
    assert(blockToCondReader.contains(block) &&
           "basic block has no branch condition");
    return blockToCondReader.lookup(block).getValue();
  }

  /// erase the cond readers, and the temporary producers
  /// of the CondBranchOp conditions,
  /// once the branches are erased
  void eraseTemps() {
    for (auto &[block, condReader] : blockToCondReader)
      condReader.erase();
    for (auto &[block, tempProd] : blockToTempProd)
      tempProd.erase();
  }

private:
  // dense maps internally
  DenseMap<Block *, TempCons> blockToCondReader;
  DenseMap<Block *, TempProd> blockToTempProd;
};

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
    Region &body, const BlockToCond &blockToCond,
    const MapVector<MemrefValue, ControlValue> &memStarts,
    ArrayRef<Cfg> useCfgs,
    MapVector<std::pair<MemrefValue, McOrLsq>, MemoryInterfaceCfg> &memoryCfgs,
    OpBuilder &builder);

} // namespace operand_routing2
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_CONVERSION_OPERANDROUTING2_CONTROLNETWORKROUTING_H
