//===- FtdCfToHandhsake.h - Convert func/cf to handhsake dialect -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --ftd-lower-cf-to-handshake conversion pass
// along with a helper class for performing the lowering.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_FTD_CF_TO_HANDSHAKE_H
#define DYNAMATIC_CONVERSION_FTD_CF_TO_HANDSHAKE_H

#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Analysis/GsaAnalysis.h"
#include "experimental/Conversion/FtdMemoryInterface.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

/// Structure to store a set of data structures useful for the whole FTD
/// algorithm.
struct FtdStoredOperations {

  /// contains all operations created by fast token delivery algorithm
  DenseSet<Operation *> allocationNetwork;

  /// contains all `handshake::MergeOp` created by `addRegen`
  DenseSet<Operation *> phiMerges;

  /// contains all `handshake::BranchOp` created by `manageMoreProdThanCons`
  /// or `manageDifferentRegeneration`
  DenseSet<Operation *> suppBranches;

  /// contains all `handshake::BranchOp` created by `manageSelfRegeneration`
  DenseSet<Operation *> selfGenBranches;

  /// contains all `handshake::MergeOp` added in the straight LSQ
  DenseSet<Operation *> memDepLoopMerges;

  /// contains all `handshake::MergeOp` added by expliciting phi functions
  DenseSet<Operation *> explicitPhiMerges;

  /// contains all `handshake::MuxOp` created by Shannon
  DenseSet<Operation *> shannonMUXes;

  /// contains all constants created by `addInit` or for Shannon’s
  DenseSet<Operation *> networkConstants;

  /// contains all the branches whose condition must be suppressed due to a
  /// `while` loop CFG structure
  DenseSet<Operation *> backwardBranches;

  /// For each condition of the block, represented in abstract as `cN` where `N`
  /// is the index of the basic block, associate its corresponding control
  /// value, Associates the condition of the block in string format to its
  /// corresponding control value. This is given by the condition of the
  /// terminator of the block, in case it's a conditional branch
  std::map<std::string, Value> conditionToValue;

  DenseMap<Block *, Value> blockControls;

  // Contains the init merges related to the MU functions
  DenseSet<Operation *> initMerges;

  // Contains the init merges related to the MU functions
  DenseSet<Operation *> initMergesConstants;
};

/// Convert a func-level function into an handshake-level function. A custom
/// behavior is defined so that the functionalities of the `fast delivery token`
/// methodology can be implemented.
class FtdLowerFuncToHandshake : public LowerFuncToHandshake {
public:
  // Use the same constructors from the base class
  FtdLowerFuncToHandshake(gsa::GsaAnalysis<mlir::func::FuncOp> &gsa,
                          NameAnalysis &namer, MLIRContext *ctx,
                          mlir::PatternBenefit benefit = 1)
      : LowerFuncToHandshake(namer, ctx, benefit), gsaAnalysis(gsa){};

  FtdLowerFuncToHandshake(gsa::GsaAnalysis<mlir::func::FuncOp> &gsa,
                          NameAnalysis &namer,
                          const TypeConverter &typeConverter, MLIRContext *ctx,
                          mlir::PatternBenefit benefit = 1)
      : LowerFuncToHandshake(namer, typeConverter, ctx, benefit),
        gsaAnalysis(gsa){};

  LogicalResult
  matchAndRewrite(mlir::func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

protected:
  /// Maintains the gsa analysis performed of the function before the conversion
  /// to the handshake dialect
  gsa::GsaAnalysis<mlir::func::FuncOp> &gsaAnalysis;

  LogicalResult ftdVerifyAndCreateMemInterfaces(
      handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter,
      MemInterfacesInfo &memInfo, FtdStoredOperations &ftdOps) const;

  /// Given a list of operations, return the list of memory dependencies for
  /// each block. This allows to build the group graph, which allows to
  /// determine the dependencies between memory access inside basic blocks.
  // Two types of hazards between the predecessors of one LSQ node:
  // (1) WAW between 2 Store operations,
  // (2) RAW and WAR between Load and Store operations
  void identifyMemoryDependencies(const SmallVector<Operation *> &operations,
                                  SmallVector<ProdConsMemDep> &allMemDeps,
                                  const mlir::CFGLoopInfo &li) const;

  /// For each pair of producer and consumer which are not in loop (thus
  /// considering, for each producer, only its forward dependenices) possibly
  /// add a merge between the pair, so that the
  LogicalResult addMergeNonLoop(handshake::FuncOp &funcOp, OpBuilder &builder,
                                SmallVector<ProdConsMemDep> &allMemDeps,
                                DenseSet<Group *> &groups,
                                DenseMap<Block *, Operation *> &forksGraph,
                                FtdStoredOperations &ftdOps,
                                Value startCtrl) const;

  /// For each pair of producer and consumer which are in loop possibly
  /// add a merge between the pair, so that the
  LogicalResult addMergeLoop(handshake::FuncOp &funcOp, OpBuilder &builder,
                             SmallVector<ProdConsMemDep> &allMemDeps,
                             DenseSet<Group *> &groups,
                             DenseMap<Block *, Operation *> &forksGraph,
                             FtdStoredOperations &ftdOps,
                             Value startCtrl) const;

  /// Convers arith-level constants to handshake-level constants. Constants are
  /// triggered by the start value of the corresponding function. The FTD
  /// algorithm is then in charge of connecting the constants to the rest of the
  /// network, in order for them to be re-generated
  LogicalResult convertConstants(ConversionPatternRewriter &rewriter,
                                 handshake::FuncOp &funcOp) const;

  /// Converts undefined operations (LLVM::UndefOp) with a default "0"
  /// constant triggered by the start signal of the corresponding function.
  LogicalResult convertUndefinedValues(ConversionPatternRewriter &rewriter,
                                       handshake::FuncOp &funcOp) const;

  /// When the consumer is in a loop while the producer is not, the value must
  /// be regenerated as many times as needed. This function is in charge of
  /// adding some merges to the network, to that this can be done. The new
  /// merge is moved inside of the loop, and it works like a reassignment
  /// (cfr. FPGA'22, Section V.C).
  LogicalResult addRegen(ConversionPatternRewriter &rewriter,
                         handshake::FuncOp &funcOp,
                         FtdStoredOperations &ftdOps) const;

  /// Given each pairs of producers and consumers within the circuit, the
  /// producer might create a token which is never used by the corresponding
  /// consumer, because of the control decisions. In this scenario, the token
  /// must be suprressed. This function inserts a `SUPPRESS` block whenever it
  /// is necessary, according to FPGA'22 (IV.C and V)
  LogicalResult addSupp(ConversionPatternRewriter &rewriter,
                        handshake::FuncOp &funcOp,
                        FtdStoredOperations &ftdOps) const;

  /// The relationship between a producer and a consumer might pass through many
  /// basic blocks with many entering conditions. All of them should be taken
  /// into account when handling the suppression mechanism. This function is
  /// called iteratively many times until all the relationships have been taken
  /// into account. In concrete, this means that the branches introduced in
  /// `addSupp` are now new producers which have to undergo the requirements of
  /// the FTD algorithm.
  LogicalResult addSuppBranches(ConversionPatternRewriter &rewriter,
                                handshake::FuncOp &funcOp,
                                FtdStoredOperations &ftdOps) const;

  /// The suppression mechanism must be used for the start token as well.
  /// However, in the handshake IR, the signal is considered as a value, so it
  /// cannot be handled by the prevvious `addSupp` functions. This funciton is
  /// in charge of handling this scenario, by adding appropriate suppressions
  /// for the start token.
  LogicalResult addSuppStart(ConversionPatternRewriter &rewriter,
                             handshake::FuncOp &funcOp,
                             FtdStoredOperations &ftdOps) const;

  LogicalResult addSuppBackwardLoop(ConversionPatternRewriter &rewriter,
                                    handshake::FuncOp &funcOp,
                                    FtdStoredOperations &ftdOps) const;

  /// Starting from the information collected by the gsa analysis pass,
  /// instantiate some merge operations at the beginning of each block which
  /// work as explicit phi functions.
  void addExplicitPhi(mlir::func::FuncOp funcOp,
                      ConversionPatternRewriter &rewriter,
                      FtdStoredOperations &ftdOps) const;
};
#define GEN_PASS_DECL_FTDCFTOHANDSHAKE
#define GEN_PASS_DEF_FTDCFTOHANDSHAKE
#include "experimental/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createFtdCfToHandshake();

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_FTD_CF_TO_HANDSHAKE_H
