//===- FtdImplementation.h --- Main FTD Algorithm ---------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the core functions to run the Fast Token Delivery algorithm,
// according to the original FPGA'22 paper by Elakhras et al.
// (https://ieeexplore.ieee.org/document/10035134).
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_FTD_IMPLEMENTATION_H
#define DYNAMATIC_SUPPORT_FTD_IMPLEMENTATION_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "experimental/Analysis/GSAAnalysis.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

/// This function implements the regeneration mechanism over a pair made of a
/// producer and a consumer (see `addRegen` description).
void addRegenOperandConsumer(PatternRewriter &rewriter,
                             dynamatic::handshake::FuncOp &funcOp,
                             Operation *consumerOp, Value operand);

/// This function implements the suppression mechanism over a pair made of a
/// producer and a consumer (see `addSupp` description).
void addSuppOperandConsumer(PatternRewriter &rewriter,
                            handshake::FuncOp &funcOp, Operation *consumerOp,
                            Value operand);

/// When the consumer is in a loop while the producer is not, the value must
/// be regenerated as many times as needed. This function is in charge of
/// adding some merges to the network, to that this can be done. The new
/// merge is moved inside of the loop, and it works like a reassignment
/// (cfr. FPGA'22, Section V.C).
void addRegen(handshake::FuncOp &funcOp, PatternRewriter &rewriter);

/// Given each pairs of producers and consumers within the circuit, the
/// producer might create a token which is never used by the corresponding
/// consumer, because of the control decisions. In this scenario, the token
/// must be suppressed. This function inserts a `SUPPRESS` block whenever it
/// is necessary, according to FPGA'22 (IV.C and V)
void addSupp(handshake::FuncOp &funcOp, PatternRewriter &rewriter);

/// Starting from the information collected by the gsa analysis pass,
/// instantiate some mux operations at the beginning of each block which
/// work as explicit phi functions. If `removeTerminators` is true, the `cf`
/// terminators in the function are modified to stop feeding the successive
/// blocks.
LogicalResult addGsaGates(Region &region, PatternRewriter &rewriter,
                          const gsa::GSAAnalysis &gsa, Backedge startValue,
                          bool removeTerminators = true);

/// For each non-init merge in the IR, run the GSA analysis to obtain its GSA
/// equivalent, then use `addGsaGates` to instantiate such operations in the IR.
LogicalResult replaceMergeToGSA(handshake::FuncOp &funcOp,
                                PatternRewriter &rewriter);

/// Connect the values in `vals` by inserting some appropriate new SSA-nodes
/// (merges) across the control flow graph of the function. The new
/// `phi-network` is in charge of connecting them in accordance to their
/// position and their dominance. Given the set of operands `toSubstitue`, each
/// of them is modified with the correct input from the network.
LogicalResult createPhiNetwork(Region &funcRegion, PatternRewriter &rewriter,
                               SmallVector<Value> &vals,
                               SmallVector<OpOperand *> &toSubstitue);

/// `deps` contains a map between an operand of an operation and a set of values
/// that operand is `dependent` on, meaning it is produced once that each of the
/// other operands are ready as well. This function generates a set of SSA-nodes
/// and appropriate joins to combine them together so that such a dependency is
/// fulfilled.
LogicalResult createPhiNetworkDeps(
    Region &funcRegion, PatternRewriter &rewriter,
    const DenseMap<OpOperand *, SmallVector<Value>> &dependenciesMap);

}; // namespace ftd
}; // namespace experimental
}; // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_FTD_IMPLEMENTATION_H
