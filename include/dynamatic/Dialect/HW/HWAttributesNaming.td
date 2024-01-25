//===- HWAttributesNaming.td - Attributes for HW dialect ---*- tablegen -*-===//
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
// This file defines HW dialect attributes used in other dialects.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HW_HW_ATTRIBUTES_NAMING_TD
#define DYNAMATIC_DIALECT_HW_HW_ATTRIBUTES_NAMING_TD

include "dynamatic/Dialect/HW/HWDialect.td"
include "mlir/IR/AttrTypeBase.td"

def InnerRefAttr : AttrDef<HWDialect, "InnerRef"> {
  let summary = "Refer to a name inside a module";
  let description = [{
    This works like a symbol reference, but to a name inside a module.
  }];
  let mnemonic = "innerNameRef";
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$moduleRef,
                        "::mlir::StringAttr":$name);

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$module,
                                        "::mlir::StringAttr":$name), [{
      return $_get(
          module.getContext(), mlir::FlatSymbolRefAttr::get(module), name);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the InnerRefAttr for an operation and add the sym on it.
    static InnerRefAttr getFromOperation(mlir::Operation *op,
                                         mlir::StringAttr symName,
                                         mlir::StringAttr moduleName);

    /// Return the name of the referenced module.
    mlir::StringAttr getModule() const { return getModuleRef().getAttr(); }
  }];
}

#endif // DYNAMATIC_DIALECT_HW_HW_ATTRIBUTES_NAMING_TD
