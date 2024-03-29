//===- HWDialect.td - HW dialect definition ----------------*- tablegen -*-===//
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
// This contains the HWDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HW_HW_DIALECT_TD
#define DYNAMATIC_DIALECT_HW_HW_DIALECT_TD

include "mlir/IR/OpBase.td"

def HWDialect : Dialect {
  let name = "hw";
  let cppNamespace = "::dynamatic::hw";

  let summary = "Types and operations for the hardware dialect";
  let description = [{
    This dialect defines the `hw` dialect, which is intended to be a generic
    representation of HW outside of a particular use-case.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;

  // Opt-out of properties for now, must migrate by LLVM 19.  #5273.
  let usePropertiesForAttributes = 0;

  let extraClassDeclaration = [{
    /// Register all HW types.
    void registerTypes();
    /// Register all HW attributes.
    void registerAttributes();

    Attribute parseAttribute(DialectAsmParser &p, Type type) const override;
    void printAttribute(Attribute attr, DialectAsmPrinter &p) const override;
  }];
}

// Base class for the operation in this dialect.
class HWOp<string mnemonic, list<Trait> traits = []> :
    Op<HWDialect, mnemonic, traits>;

#endif // DYNAMATIC_DIALECT_HW_HW_DIALECT_TD
