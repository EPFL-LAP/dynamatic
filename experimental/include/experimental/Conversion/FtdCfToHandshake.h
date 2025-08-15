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
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Analysis/GSAAnalysis.h"

namespace dynamatic {
namespace experimental {
namespace ftd {

/// Convert a func-level function into an handshake-level function. A custom
/// behavior is defined so that the functionalities of the `fast delivery token`
/// methodology can be implemented.
class FtdLowerFuncToHandshake : public LowerFuncToHandshake {
public:
  // Use the same constructors from the base class
  FtdLowerFuncToHandshake(ControlDependenceAnalysis &cda, gsa::GSAAnalysis &gsa,
                          NameAnalysis &namer, MLIRContext *ctx,
                          mlir::PatternBenefit benefit = 1)
      : LowerFuncToHandshake(namer, ctx, benefit), cdAnalysis(cda),
        gsaAnalysis(gsa){};

  FtdLowerFuncToHandshake(ControlDependenceAnalysis &cda, gsa::GSAAnalysis &gsa,
                          NameAnalysis &namer,
                          const TypeConverter &typeConverter, MLIRContext *ctx,
                          mlir::PatternBenefit benefit = 1)
      : LowerFuncToHandshake(namer, typeConverter, ctx, benefit),
        cdAnalysis(cda), gsaAnalysis(gsa){};

  LogicalResult
  matchAndRewrite(mlir::func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

protected:
  /// Store the control dependency analysis over the input function
  ControlDependenceAnalysis cdAnalysis;

  /// Store the GSA analysis over the input function
  gsa::GSAAnalysis gsaAnalysis;
};

template <typename SrcOp, typename DstOp>
struct FtdOneToOneConversion : public OneToOneConversion<SrcOp, DstOp> {
public:
  using OpAdaptor = typename SrcOp::Adaptor;

  FtdOneToOneConversion(NameAnalysis &namer, const TypeConverter &typeConverter,
                        MLIRContext *ctx)
      : dynamatic::OneToOneConversion<SrcOp, DstOp>(namer, typeConverter, ctx) {
  }

  LogicalResult
  matchAndRewrite(SrcOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <typename CastOp, typename ExtOp>
struct FtdConvertIndexCast : public ConvertIndexCast<CastOp, ExtOp> {
public:
  using OpAdaptor = typename CastOp::Adaptor;

  FtdConvertIndexCast(NameAnalysis &namer, const TypeConverter &typeConverter,
                      MLIRContext *ctx)
      : dynamatic::ConvertIndexCast<CastOp, ExtOp>(namer, typeConverter, ctx) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

#define GEN_PASS_DECL_FTDCFTOHANDSHAKE
#define GEN_PASS_DEF_FTDCFTOHANDSHAKE
#include "experimental/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createFtdCfToHandshake();

} // namespace ftd
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_FTD_CF_TO_HANDSHAKE_H
