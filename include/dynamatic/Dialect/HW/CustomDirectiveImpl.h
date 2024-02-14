//===- CustomDirectiveImpl.h - Table-gen custom directive impl --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This file provides common custom directives for table-gen assembly formats.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HW_CUSTOM_DIRECTIVE_IMPL_H
#define DYNAMATIC_DIALECT_HW_CUSTOM_DIRECTIVE_IMPL_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

namespace dynamatic {

//===----------------------------------------------------------------------===//
// InputPortList Custom Directive
//===----------------------------------------------------------------------===//

/// Parse a list of instance input ports.
/// input-list ::= `(` ( input-element (`,` input-element )* )? `)`
/// input-element ::= identifier `:` value `:` type
ParseResult
parseInputPortList(OpAsmParser &parser,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inputs,
                   SmallVectorImpl<Type> &inputTypes, ArrayAttr &inputNames);

/// Print a list of instance input ports.
void printInputPortList(OpAsmPrinter &p, Operation *op, OperandRange inputs,
                        TypeRange inputTypes, ArrayAttr inputNames);

//===----------------------------------------------------------------------===//
// OutputPortList Custom Directive
//===----------------------------------------------------------------------===//

/// Parse a list of instance output ports.
/// output-list ::= `(` ( output-element (`,` output-element )* )? `)`
/// output-element ::= identifier `:` type
ParseResult parseOutputPortList(OpAsmParser &parser,
                                SmallVectorImpl<Type> &resultTypes,
                                ArrayAttr &resultNames);

/// Print a list of instance output ports.
void printOutputPortList(OpAsmPrinter &p, Operation *op, TypeRange resultTypes,
                         ArrayAttr resultNames);

//===----------------------------------------------------------------------===//
// OptionalParameterList Custom Directive
//===----------------------------------------------------------------------===//

/// Parse an parameter list if present.
/// module-parameter-list ::= `<` parameter-decl (`,` parameter-decl)* `>`
/// parameter-decl ::= identifier `:` type
/// parameter-decl ::= identifier `:` type `=` attribute
ParseResult parseOptionalParameterList(OpAsmParser &parser,
                                       ArrayAttr &parameters);

/// Print a parameter list for a module or instance.
void printOptionalParameterList(OpAsmPrinter &p, Operation *op,
                                ArrayAttr parameters);

} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HW_CUSTOM_DIRECTIVE_IMPL_H
