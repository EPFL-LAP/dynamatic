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

#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/FuncMaximizeSSA.h"
#include "experimental/Conversion/FtdMemoryInterface.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

struct FtdStoredOperations {

  /// contains all operations created by fast token delivery algorithm
  SmallVector<Operation *> allocationNetwork;

  /// contains all `handshake::MergeOp` created by `addPhi`
  SmallVector<Operation *> phiMerges;

  /// contains all `handshake::BranchOp` created by `manageMoreProdThanCons`
  /// or `manageDifferentRegeneration`
  std::vector<Operation *> suppBranches;

  /// contains all `handshake::BranchOp` created by `manageSelfRegeneration`
  SmallVector<Operation *> selfGenBranches;

  /// contains all `handshake::MergeOp` added in the straight LSQ
  SmallVector<Operation *> memDepLoopMerges;

  /// contains all `handshake::MuxOp` created by Shannon
  SmallVector<Operation *> shannonMUXes;

  /// contains all constants created by `addInit` or for Shannon’s
  SmallVector<Operation *> networkConstants;
};

/// Convert a func-level function into an handshake-level function. A custom
/// behavior is defined so that the functionalities of the `fast delivery token`
/// methodology can be implemented.
class FtdLowerFuncToHandshake : public LowerFuncToHandshake {
public:
  // Use the same constructors from the base class
  FtdLowerFuncToHandshake(NameAnalysis &namer, MLIRContext *ctx,
                          mlir::PatternBenefit benefit = 1)
      : LowerFuncToHandshake(namer, ctx, benefit){};

  FtdLowerFuncToHandshake(NameAnalysis &namer,
                          const TypeConverter &typeConverter, MLIRContext *ctx,
                          mlir::PatternBenefit benefit = 1)
      : LowerFuncToHandshake(namer, typeConverter, ctx, benefit){};

  LogicalResult
  matchAndRewrite(mlir::func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

protected:
  LogicalResult ftdVerifyAndCreateMemInterfaces(
      handshake::FuncOp funcOp, ConversionPatternRewriter &rewriter,
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
};

#define GEN_PASS_DECL_FTDCFTOHANDSHAKE
#define GEN_PASS_DEF_FTDCFTOHANDSHAKE
#include "experimental/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createFtdCfToHandshake();

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_FTD_CF_TO_HANDSHAKE_H
