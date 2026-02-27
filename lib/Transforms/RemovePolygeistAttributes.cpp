//===- RemovePolygeistAttributes.h - Remove useless attrs -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Removes a couple top-level module and function attributes set by Polygeist
// and which we do not care about.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"

using namespace mlir;

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_REMOVEPOLYGEISTATTRIBUTES
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

namespace {

/// Simple driver for the pass that removes attributes set by Polygeist.
struct RemovePolygeistAttributesPass
    : public dynamatic::impl::RemovePolygeistAttributesBase<
          RemovePolygeistAttributesPass> {

  void runDynamaticPass() override {
    // Remove all attributes from the top-level module
    mlir::ModuleOp modOp = getOperation();
    for (NamedAttribute attr :
         llvm::make_early_inc_range(modOp->getAttrDictionary()))
      modOp->removeAttr(attr.getName());

    for (func::FuncOp funcOp : modOp.getOps<func::FuncOp>()) {
      // Remove the llvm.linkage attribute from functions
      funcOp->removeAttr("llvm.linkage");
    }
  };
};

} // namespace
