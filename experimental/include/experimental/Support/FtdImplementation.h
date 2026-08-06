//===- FtdImplementation.h --- Main FTD Algorithm ---------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the top-level steps of the Fast Token Delivery (FTD) algorithm:
// converting phi functions to GSA gates, regeneration, suppression dispatch,
// and condition placeholders. The suppression circuit construction itself
// lives in FtdSuppression.h.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_FTD_IMPLEMENTATION_H
#define DYNAMATIC_SUPPORT_FTD_IMPLEMENTATION_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "experimental/Analysis/GSAAnalysis.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

// ShadowCFG is declared in FtdSupport.h

/// Create SourceOp condition placeholders for every conditional block in the
/// region. Must be called before addGsaGates.
void createAllCondPlaceholders(Region &region, OpBuilder &builder);

/// After addBranchOps (multi-block, handshake ConditionalBranchOps exist),
/// replace each condition placeholder with NotIOp(realCond) that captures
/// the actual handshake condition value.
void resolveCondPlaceholders(handshake::FuncOp funcOp, OpBuilder &builder,
                             ShadowCFG &shadow);

/// After addRegen/addSupp, short-circuit all NotIOp condition placeholders
/// and erase them along with their source+constant operands.
void finalizeCondPlaceholders(handshake::FuncOp funcOp);

/// This function implements the regeneration mechanism over a pair made of a
/// producer and a consumer (see `addRegen` description).
std::vector<Operation *> addRegenOperandConsumer(mlir::OpBuilder &builder,
                                                 handshake::FuncOp &funcOp,
                                                 mlir::Operation *consumerOp,
                                                 mlir::Value operand,
                                                 ShadowCFG &shadow);

/// This function implements the suppression mechanism over a pair made of a
/// producer and a consumer (see `addSupp` description).
std::vector<Operation *>
addSuppOperandConsumer(mlir::OpBuilder &builder, handshake::FuncOp &funcOp,

                       Operation *consumerOp, Value operand, ShadowCFG &shadow);

/// When the consumer is in a loop while the producer is not, the value must
/// be regenerated as many times as needed. This function is in charge of
/// adding some merges to the network, so that this can be done. The new
/// merge is moved inside of the loop, and it works like a reassignment.
void addRegen(handshake::FuncOp &funcOp, mlir::OpBuilder &builder,
              ShadowCFG &shadow);

/// Given each pairs of producers and consumers within the circuit, the
/// producer might create a token which is never used by the corresponding
/// consumer, because of the control decisions. In this scenario, the token
/// must be suppressed. This function inserts a `SUPPRESS` block whenever it
/// is necessary.
void addSupp(handshake::FuncOp &funcOp, mlir::OpBuilder &builder,
             ShadowCFG &shadow);

/// Starting from the information collected by the gsa analysis pass,
/// instantiate some mux operations at the beginning of each block which
/// work as explicit phi functions. If `removeTerminators` is true, the `cf`
/// terminators in the function are modified to stop feeding the successive
/// blocks.
LogicalResult addGsaGates(
    Region &region, PatternRewriter &rewriter, const gsa::GSAAnalysis &gsa,
    std::vector<Operation *> &newUnits,
    DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands = nullptr,
    bool removeTerminators = true);

/// For each non-init merge in the IR, run the GSA analysis to obtain its GSA
/// equivalent, then use `addGsaGates` to instantiate such operations in the IR.
LogicalResult replaceMergeToGSA(handshake::FuncOp &funcOp,
                                PatternRewriter &rewriter,
                                std::vector<Operation *> &newUnits);

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
