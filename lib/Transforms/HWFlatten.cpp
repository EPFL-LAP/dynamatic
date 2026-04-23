//===- FlattenModules.cpp -------------------------------------------------===//
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

#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"

#define DEBUG_TYPE "hw-flatten-modules"

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
// Boilerplate: Include this for the pass option defintitions
namespace dynamatic {
// import auto-generated base class definition
// and put it under the dynamatic namespace.
#define GEN_PASS_DEF_FLATTENMODULES
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]
using namespace dynamatic;
using namespace hw;

namespace {

/// Inliner that marks all inlining as legal and wires hw.output operands back
/// to the instance results upon termination.
struct HWInliner : public mlir::InlinerInterface {
  using InlinerInterface::InlinerInterface;

  bool isLegalToInline(Region *, Region *, bool,
                       mlir::IRMapping &) const override {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool,
                       mlir::IRMapping &) const override {
    return true;
  }

  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToReplace) const override {
    assert(isa<hw::OutputOp>(op));
    for (auto [toReplace, replacement] :
         llvm::zip(valuesToReplace, op->getOperands()))
      toReplace.replaceAllUsesWith(replacement);
  }
};

struct FlattenModulesPass
    : public dynamatic::impl::FlattenModulesBase<FlattenModulesPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void FlattenModulesPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  Operation *top = getOperation();

  // Pre-pass: annotate the op that drives each output port of every hw module
  // with "hw.module_name" and "hw.port_name" attributes before inlining.
  // Because inlineRegion clones the body (shouldClone=true), these attributes
  // are preserved in the flattened parent.
  top->walk([&](hw::HWModuleOp module) {
    auto *outputOp = module.getBodyBlock()->getTerminator();
    unsigned outIdx = 0;
    for (auto &port : module.getPortList()) {
      if (port.isInput())
        continue;
      Value outputVal = outputOp->getOperand(outIdx);
      StringAttr modName = StringAttr::get(ctx, module.getName());
      StringAttr portName = port.name;
      if (Operation *defOp = outputVal.getDefiningOp()) {
        defOp->setAttr("hw.module_name", modName);
        defOp->setAttr("hw.port_name", portName);
      }
      ++outIdx;
    }
  });

  // Record which modules are instantiated before we start inlining, so we
  // know which ones to erase afterwards.
  DenseSet<StringAttr> instantiated;

  // Inline every hw.instance, restarting the walk after each mutation until
  // no instances remain.
  HWInliner inliner(&getContext());
  bool changed = true;
  while (changed) {
    changed = false;
    top->walk([&](hw::InstanceOp inst) -> WalkResult {
      auto target = dyn_cast_or_null<hw::HWModuleOp>(
          SymbolTable::lookupNearestSymbolFrom(inst, inst.getModuleNameAttr()));
      if (!target)
        return WalkResult::advance();

      // Mark this module as instantiated so we know to erase it later.
      instantiated.insert(target.getNameAttr());

      if (failed(mlir::inlineRegion(inliner, &target.getBody(), inst,
                                    inst.getOperands(), inst.getResults(),
                                    std::nullopt, /*shouldClone=*/true))) {
        inst.emitError("failed to inline '") << target.getName() << "'";
        signalPassFailure();
        return WalkResult::interrupt();
      }

      inst.erase();
      changed = true;
      return WalkResult::interrupt();
    });
  }

  // Erase all modules that have been inlined away.
  for (auto module :
       llvm::make_early_inc_range(top->getRegion(0).getOps<hw::HWModuleOp>())) {
    if (instantiated.count(module.getNameAttr())) {
      module.erase();
    }
  }
}