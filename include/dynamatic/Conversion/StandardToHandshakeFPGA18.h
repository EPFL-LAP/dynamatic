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
#include "dynamatic/Support/DynamaticPass.h"
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
  /// Stores a list of opeartions grouped by the basic block they belong to.
  using MemBlockOps = llvm::MapVector<Block *, SmallVector<Operation *>>;

  /// Store a mapping between memory interfaces (identified by the function
  /// argument they correspond to) and the set of memory operations referencing
  /// them grouped by their owning block.
  using MemInterfacesInfo = llvm::MapVector<Value, MemBlockOps>;

  /// Constructor simply forwards its arguments to the parent class.
  explicit HandshakeLoweringFPGA18(Region &r) : HandshakeLowering(r) {}

  /// Creates the control-only network by adding a control-only argument to the
  /// region's entry block and forwarding it through all basic blocks.
  LogicalResult createControlOnlyNetwork(ConversionPatternRewriter &rewriter);

  /// Identifies all memory interfaces and their associated operations in the
  /// function, replaces all load/store-like operations by their handshake
  /// counterparts, and fills `memInfo` with information about which operations
  /// use which interface.
  LogicalResult replaceMemoryOps(ConversionPatternRewriter &rewriter,
                                 MemInterfacesInfo &memInfo);

  /// Creates the list(s) of inputs for the memory interface(s) associated with
  /// a single memory region. Fills the two vectors with, respectively, the list
  /// of inputs for a memory controller and the list of inputs for an LSQ. An
  /// empty list of inputs should be interpreted by the caller as "there is no
  /// need for such a memory interface for this memory region". If both lists of
  /// inputs are returned non-empty, then both interfaces must be placed, and
  /// channels between them must be added manually. Returns, in a pair, the
  /// number of loads that should connect directly to the memory controller and
  /// the number of loads that should connect directly to the LSQ.
  std::pair<unsigned, unsigned> deriveMemInterfaceInputs(
      MemBlockOps &allMemOps, ConversionPatternRewriter &rewriter,
      SmallVector<Value> &mcInputs, SmallVector<Value> &lsqInputs);

  /// Instantiates all memory interfaces and connects them to their respective
  /// load/store operations. To choose what type of interface to instantiate,
  /// looks for the `handshake::NoLSQAttr` attribute on memory operations, which
  /// indicates that a simple memory controller is enough for the access. For
  /// any memory region:
  /// - A single `handshake::MemoryControllerOp` will be instantiated if all of
  /// its accesses have the `handshake::NoLSQAttr`.
  /// - A single `handshake::LSQOp` will be instantiated if none of
  /// its accesses have the `handshake::NoLSQAttr`.
  /// - Both a `handhsake::MemoryControllerOp` and `handhsake::LSQOp` will be
  /// instantiated if some but not all of its accesses have the
  /// `handshake::NoLSQAttr`.
  LogicalResult connectToMemInterfaces(ConversionPatternRewriter &rewriter,
                                       MemInterfacesInfo &memInfo);

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
};

std::unique_ptr<dynamatic::DynamaticPass<false>>
createStandardToHandshakeFPGA18Pass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_STANDARDTOHANDSHAKEFPGA18_H
