//===- DynamaticPass.h - Support for Dynamatic passes -----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for pass writing in Dynamatic. Defines conversion patterns which
// reference a live naming analysis.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_DYNAMATICPASS_H
#define DYNAMATIC_SUPPORT_DYNAMATICPASS_H

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

/// Regular OpConversionPattern which additionally references a live naming
/// analysis.
template <typename SourceOp>
struct DynOpConversionPattern : public OpConversionPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  DynOpConversionPattern(NameAnalysis &namer, MLIRContext *ctx,
                         mlir::PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(ctx, benefit), namer(namer) {}

  DynOpConversionPattern(NameAnalysis &namer,
                         const TypeConverter &typeConverter, MLIRContext *ctx,
                         mlir::PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, ctx), namer(namer) {}

protected:
  /// Reference to the running pass's naming analysis.
  NameAnalysis &namer;
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_DYNAMATICPASS_H
