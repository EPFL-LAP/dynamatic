//===- ModuleImplementation.h - Module-like Op utilities --------*- C++ -*-===//
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
// This file provides utility functions for implementing module-like
// operations, in particular, parsing, and printing common to module-like
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HW_MODULE_IMPLEMENTATION_H
#define DYNAMATIC_DIALECT_HW_MODULE_IMPLEMENTATION_H

#include "dynamatic/Dialect/HW/HWTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"

namespace dynamatic {
namespace hw {

namespace module_like_impl {

struct PortParse : OpAsmParser::Argument {
  ModulePort::Direction direction;
  std::string rawName;
};

/// This is a variant of mlir::parseFunctionSignature that allows names on
/// result arguments.
ParseResult parseModuleFunctionSignature(
    OpAsmParser &parser, bool &isVariadic,
    SmallVectorImpl<OpAsmParser::Argument> &args,
    SmallVectorImpl<Attribute> &argNames, SmallVectorImpl<Attribute> &argLocs,
    SmallVectorImpl<Attribute> &resultNames,
    SmallVectorImpl<DictionaryAttr> &resultAttrs,
    SmallVectorImpl<Attribute> &resultLocs, TypeAttr &type);

/// Print a module signature with named results.
void printModuleSignature(OpAsmPrinter &p, Operation *op,
                          ArrayRef<Type> argTypes, bool isVariadic,
                          ArrayRef<Type> resultTypes, bool &needArgNamesAttr);

/// New Style parsing
ParseResult parseModuleSignature(OpAsmParser &parser,
                                 SmallVectorImpl<PortParse> &args,
                                 TypeAttr &modType);
void printModuleSignatureNew(OpAsmPrinter &p, Operation *op);

} // namespace module_like_impl
} // namespace hw
} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HW_MODULE_IMPLEMENTATION_H
