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
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

/// A temporary shadow of the original CFG, built after CfToHandshake
/// conversion flattens everything.  Encapsulates the shadow Region
/// (with real CF terminators) plus a condition-value map that bridges
/// shadow analysis to real handshake Values.
///
/// All analysis infrastructure (BlockIndexing, CFGLoopInfo, path
/// enumeration, dominance) operates on getRegion() as if the original
/// CFG were still alive.  The only thing the shadow cannot provide
/// natively is the real handshake condition Value for each cond_br
/// block — getCondition() provides that.
struct ShadowCFG {
  mlir::func::FuncOp shadowFunc;
  llvm::DenseMap<unsigned, mlir::Value> conditionMap;

  mlir::Region &getRegion() { return shadowFunc.getBody(); }

  mlir::Block *getBlock(unsigned bbIdx) {
    for (auto [i, blk] : llvm::enumerate(getRegion()))
      if (i == bbIdx)
        return &blk;
    llvm_unreachable("BB index out of range in shadow CFG");
  }

  unsigned getBlockIndex(mlir::Block *block) {
    for (auto [i, blk] : llvm::enumerate(getRegion()))
      if (&blk == block)
        return i;
    llvm_unreachable("Block not found in shadow CFG");
  }

  /// Get the real handshake condition Value for the cond_br in block bbIdx.
  /// Returns nullptr if the block had an unconditional branch.
  mlir::Value getCondition(unsigned bbIdx) {
    auto it = conditionMap.find(bbIdx);
    return (it != conditionMap.end()) ? it->second : nullptr;
  }

  mlir::Value getCondition(mlir::Block *block) {
    return getCondition(getBlockIndex(block));
  }

  void destroy() {
    if (shadowFunc)
      shadowFunc.erase();
  }
};

/// Create SourceOp condition placeholders for every conditional block in the
/// region. Must be called before addGsaGates.
void createAllCondPlaceholders(Region &region, OpBuilder &builder);

/// After addBranchOps (multi-block, handshake ConditionalBranchOps exist),
/// replace each SourceOp placeholder with XorOp(realCond, 0) that captures
/// the actual handshake condition value.
void resolveCondPlaceholders(handshake::FuncOp funcOp, OpBuilder &builder,
                             ShadowCFG &shadow);

/// After addFtdRegen/addSupp, short-circuit all XorOp condition placeholders
/// and erase them along with their source+constant operands.
void finalizeCondPlaceholders(handshake::FuncOp funcOp);

/// This function implements the regeneration mechanism over a pair made of a
/// producer and a consumer (see `addRegen` description).
void addRegenOperandConsumer(mlir::OpBuilder &builder,
                             handshake::FuncOp &funcOp,
                             mlir::Operation *consumerOp, mlir::Value operand,
                             ShadowCFG &shadow);

/// This function implements the suppression mechanism over a pair made of a
/// producer and a consumer (see `addSupp` description).
void addSuppOperandConsumer(mlir::OpBuilder &builder, handshake::FuncOp &funcOp,
                            Operation *consumerOp, Value operand,
                            ShadowCFG &shadow);

/// When the consumer is in a loop while the producer is not, the value must
/// be regenerated as many times as needed. This function is in charge of
/// adding some merges to the network, to that this can be done. The new
/// merge is moved inside of the loop, and it works like a reassignment
/// (cfr. FPGA'22, Section V.C).
void addRegen(handshake::FuncOp &funcOp, mlir::OpBuilder &builder,
                 ShadowCFG &shadow);

/// Given each pairs of producers and consumers within the circuit, the
/// producer might create a token which is never used by the corresponding
/// consumer, because of the control decisions. In this scenario, the token
/// must be suppressed. This function inserts a `SUPPRESS` block whenever it
/// is necessary, according to FPGA'22 (IV.C and V)
void addSupp(handshake::FuncOp &funcOp, mlir::OpBuilder &builder,
             ShadowCFG &shadow);

/// Starting from the information collected by the gsa analysis pass,
/// instantiate some mux operations at the beginning of each block which
/// work as explicit phi functions. If `removeTerminators` is true, the `cf`
/// terminators in the function are modified to stop feeding the successive
/// blocks.
LogicalResult addGsaGates(
    Region &region, PatternRewriter &rewriter, const gsa::GSAAnalysis &gsa,
    Backedge startValue,
    DenseMap<Value, SmallVector<Backedge, 2>> *pendingMuxOperands = nullptr,
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

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_FTD_IMPLEMENTATION_H
