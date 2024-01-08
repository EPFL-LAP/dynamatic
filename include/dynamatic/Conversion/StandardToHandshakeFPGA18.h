//===- StandardToHandshakeFPGA18.h - FPGA18's elastic pass ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --lower-std-to-handshake-fpga18 conversion pass along
// with a helper class for performing the lowering.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_STANDARDTOHANDSHAKEFPGA18_H
#define DYNAMATIC_CONVERSION_STANDARDTOHANDSHAKEFPGA18_H

#include "circt/Conversion/StandardToHandshake.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/Handshake.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::handshake;

namespace dynamatic {

// This class is used to inherit from CIRCT's standard-to-handshake lowering
// infrastructure and implementation while providing us a way to
// change/add/remove/reorder specific conversion steps to match FPGA18's elastic
// pass.
class HandshakeLoweringFPGA18 : public HandshakeLowering {
public:
  /// Groups memory operations by interface and group for a given memory region.
  struct MemAccesses {
    /// Memory operations for a simple memory controller, grouped by
    /// originating basic block.
    llvm::MapVector<Block *, SmallVector<Operation *>> mcPorts;
    /// Memory operations for an LSQ, grouped by belonging LSQ group.
    llvm::MapVector<unsigned, SmallVector<Operation *>> lsqPorts;
  };

  /// Stores a mapping between memory regions (identified by the function
  /// argument they correspond to) and the set of memory operations referencing
  /// them.
  using MemInterfacesInfo = llvm::MapVector<Value, MemAccesses>;

  /// Constructor simply forwards the region to its parent class and stores a
  /// reference to the top-level name analysis.
  explicit HandshakeLoweringFPGA18(Region &r, NameAnalysis &nameAnalysis)
      : HandshakeLowering(r), nameAnalysis(nameAnalysis) {}

  /// Creates the control-only network by adding a control-only argument to the
  /// region's entry block and forwarding it through all basic blocks.
  LogicalResult createControlOnlyNetwork(ConversionPatternRewriter &rewriter);

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

  /// Derive input information for all LSQs in the function (stores it in
  /// `memInputs`), while . Fails if the
  /// definition of an LSQ group is invalid; succeeds otherwise.

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
  /// region's end operation.
  LogicalResult createReturnNetwork(ConversionPatternRewriter &rewriter);

private:
  /// Name analysis to name new memory operations as they are created and keep
  /// reference accesses in memory dependencies consistent.
  NameAnalysis &nameAnalysis;
};

std::unique_ptr<dynamatic::DynamaticPass> createStandardToHandshakeFPGA18Pass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_STANDARDTOHANDSHAKEFPGA18_H
