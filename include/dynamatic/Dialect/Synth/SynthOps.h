//===----------------------------------------------------------------------===//
// All the files related to the description of the Synth dialect
// have been modified from the original version present in the
// circt project at the following link:
// https://github.com/llvm/circt/tree/main/
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations of the Synth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_SYNTH_SYNTHOPS_H
#define DYNAMATIC_DIALECT_SYNTH_SYNTHOPS_H

#include "dynamatic/Dialect/Synth/SynthDialect.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"

#define GET_OP_CLASSES
#include "dynamatic/Dialect/Synth/Synth.h.inc"

namespace dynamatic {
namespace synth {
struct AndInverterVariadicOpConversion
    : mlir::OpRewritePattern<aig::AndInverterOp> {
  using OpRewritePattern<aig::AndInverterOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(aig::AndInverterOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace synth
ParseResult parseVariadicInvertibleOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands, Type &resultType,
    mlir::DenseBoolArrayAttr &inverted, NamedAttrList &attrDict);
void printVariadicInvertibleOperands(OpAsmPrinter &printer, Operation *op,
                                     OperandRange operands, Type resultType,
                                     mlir::DenseBoolArrayAttr inverted,
                                     DictionaryAttr attrDict);

} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_SYNTH_SYNTHOPS_H
