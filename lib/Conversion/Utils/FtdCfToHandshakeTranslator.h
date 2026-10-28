//===- FtdCfToHandshakeTranslator.h - FTD translation to handshake -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Translation of func-level functions into Handshake functions with the fast
// token delivery (FTD) methodology, without the dialect conversion framework.
// Used by the --ftd-lower-cf-to-handshake pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_LIB_CONVERSION_UTILS_FTDCFTOHANDSHAKETRANSLATOR_H
#define DYNAMATIC_LIB_CONVERSION_UTILS_FTDCFTOHANDSHAKETRANSLATOR_H

#include "CfToHandshakeTranslator.h"
#include "dynamatic/Analysis/GSAAnalysis.h"
#include "dynamatic/Support/FtdSupport.h"

namespace dynamatic {

/// Translates a func-level function into a Handshake function with the FTD
/// methodology. Unlike the base translation:
/// - block arguments become the GSA gates (muxes) computed by the GSA
///   analysis, instead of merge-like operations fed by the predecessors;
/// - terminators only branch the block's control: data tokens are delivered
///   by the regeneration and suppression circuits, which are added once the
///   function is translated;
/// - constants and undefined values are triggered by the function's start.
///
/// The func-level function is the CFG that the FTD analyses run on: it is left
/// untouched until the end of the translation.
class FtdCfToHandshakeTranslator : public CfToHandshakeTranslator {
public:
  FtdCfToHandshakeTranslator(mlir::func::FuncOp cfFunc,
                             mlir::OpBuilder &builder, NameAnalysis &namer,
                             const gsa::GSAAnalysis &gsaAnalysis)
      : CfToHandshakeTranslator(cfFunc, builder, namer),
        gsaAnalysis(gsaAnalysis) {}

protected:
  const gsa::GSAAnalysis &gsaAnalysis;
  /// Indexing of the func-level blocks used by the FTD analyses.
  std::optional<ftd::BlockIndexing> blockIndexing;
  /// The CFG that the FTD functions (e.g., ftd::addSupp) take as input. Their
  /// interface (ftd::ShadowCFG) bundles the CFG with the Handshake condition of
  /// each conditional block, which the circuits they create read. The CFG is
  /// the func-level function itself, which stays alive until the end of the
  /// translation, so no shadow copy of it is built; only the conditions are
  /// tracked here. Until the blocks are translated, the conditions are
  /// placeholders.
  ftd::ShadowCFG cfgView;
  /// Placeholder for the condition of each conditional block, replaced by the
  /// block's condition proxy once the blocks are translated.
  SmallVector<std::pair<Block *, Backedge>> condPlaceholders;
  /// Placeholders for the func-level values that the GSA gates take as
  /// operands, which may not be translated yet when the gates are created.
  SmallVector<std::pair<Value, Backedge>> gateOperandPlaceholders;

  LogicalResult initTranslation() override;
  LogicalResult createMerges(Block *block) override;
  SmallVector<Value> getForwardedValues(Operation *termOp,
                                        unsigned succIdx) override;
  Value getBranchCondition(mlir::cf::CondBranchOp condBrOp) override;
  bool isCstSourcable(mlir::arith::ConstantOp cstOp) override;
  Value getBlockControlChannel(Block *block) override;
  void finalizeBlocks() override;
  LogicalResult finalizeTranslation() override;

  /// Creates the muxes implementing the GSA gates of all non-entry blocks, and
  /// maps the block arguments to them.
  LogicalResult createGsaGates();
};

/// Lowers all func-level functions of the module to Handshake functions with
/// a `FtdCfToHandshakeTranslator`.
LogicalResult ftdLowerCfToHandshake(mlir::ModuleOp modOp, NameAnalysis &namer,
                                    const gsa::GSAAnalysis &gsa);

} // namespace dynamatic

#endif // DYNAMATIC_LIB_CONVERSION_UTILS_FTDCFTOHANDSHAKETRANSLATOR_H
