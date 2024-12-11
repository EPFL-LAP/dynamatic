//===- FtdImplementation.h --- FTD conversion support -----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares some utility functions which are useful for both the fast token
// delivery algorithm and for the GSA anlaysis pass. All the functions are about
// anlayzing relationships between blocks and handshake operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_FTD_SUPPORT_H
#define DYNAMATIC_SUPPORT_FTD_SUPPORT_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "experimental/Analysis/GSAAnalysis.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

/// Add some regen multiplexers between an opearation and one of its operands
void addRegenOperandConsumer(ConversionPatternRewriter &rewriter,
                             dynamatic::handshake::FuncOp &funcOp,
                             Operation *consumerOp, Value operand);

/// Add suppression mechanism to all the inputs and outputs of a producer
void addSuppOperandConsumer(ConversionPatternRewriter &rewriter,
                            handshake::FuncOp &funcOp, Operation *consumerOp,
                            Value operand);

/// When the consumer is in a loop while the producer is not, the value must
/// be regenerated as many times as needed. This function is in charge of
/// adding some merges to the network, to that this can be done. The new
/// merge is moved inside of the loop, and it works like a reassignment
/// (cfr. FPGA'22, Section V.C).
void addRegen(handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter);

/// Given each pairs of producers and consumers within the circuit, the
/// producer might create a token which is never used by the corresponding
/// consumer, because of the control decisions. In this scenario, the token
/// must be suprressed. This function inserts a `SUPPRESS` block whenever it
/// is necessary, according to FPGA'22 (IV.C and V)
void addSupp(handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter);

/// Starting from the information collected by the gsa analysis pass,
/// instantiate some merge operations at the beginning of each block which
/// work as explicit phi functions.
LogicalResult addGsaGates(Region &region, ConversionPatternRewriter &rewriter,
                          const gsa::GSAAnalysis &gsa, Backedge startValue,
                          bool removeTerminators = true);

/// Use the GSA analysis to replace each non-init merge in the IR with a
/// multiplexer.
LogicalResult replaceMergeToGSA(handshake::FuncOp &funcOp,
                                ConversionPatternRewriter &rewriter);

/// Connect the values in `vals` by inserting some appropriate new SSA-nodes
/// (merges) across the control flow graph of the function. The new
/// `phi-network` is in charge of connecting them in accordance to their
/// position and their dominance. Given the set of operands `toSubstitue`, each
/// of them is modified with the correct input from the network.
LogicalResult createPhiNetwork(Region &funcRegion,
                               ConversionPatternRewriter &rewriter,
                               SmallVector<Value> &vals,
                               SmallVector<OpOperand *> &toSubstitue);

/// `deps` contains a map between an operand of an operation and a set of values
/// that operand is `dependent` on, meaning it is produced once that each of the
/// other operands are ready as well. This function generates a set of SSA-nodes
/// and appropriate joins to combine them together so that such a dependency is
/// fullfilled.
LogicalResult createPhiNetworkDeps(
    Region &funcRegion, ConversionPatternRewriter &rewriter,
    const DenseMap<OpOperand *, SmallVector<Value>> &dependenciesMap);

}; // namespace ftd
}; // namespace experimental
}; // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_FTD_SUPPORT_H
