//===- ConsumeEdgeAttrMarker.cpp - Move marker attrs to producer ----------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace dynamatic;

namespace dynamatic {
#define GEN_PASS_DEF_CONSUMEEDGEATTRMARKER
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic

// The unregistered op name emitted by `translate-llvm-to-std` to pin
// edge-level attributes onto a specific SSA value.
static constexpr llvm::StringLiteral kEdgeAttrMarkerName =
    "dynamatic.edge_attr_marker";

// Only attributes in this namespace are migrated from the marker to the
// producer. This filters out attributes added to the marker by upstream
// machinery (e.g. `handshake.name` from name analysis) so they do not
// pollute the producer's attribute dictionary.
static constexpr llvm::StringLiteral kCopiedAttrPrefix = "dynamatic.";

namespace {
struct ConsumeEdgeAttrMarkerPass
    : public dynamatic::impl::ConsumeEdgeAttrMarkerBase<
          ConsumeEdgeAttrMarkerPass> {
  using ConsumeEdgeAttrMarkerBase::ConsumeEdgeAttrMarkerBase;
  void runOnOperation() override;
};
} // namespace

void ConsumeEdgeAttrMarkerPass::runOnOperation() {
  ModuleOp modOp = getOperation();

  SmallVector<Operation *> markers;
  modOp.walk([&](Operation *op) {
    if (op->getName().getStringRef() == kEdgeAttrMarkerName)
      markers.push_back(op);
  });

  for (Operation *markerOp : markers) {
    if (markerOp->getNumOperands() != 1) {
      markerOp->emitError()
          << kEdgeAttrMarkerName << " must have exactly one operand";
      return signalPassFailure();
    }
    if (markerOp->getNumResults() != 1) {
      markerOp->emitError()
          << kEdgeAttrMarkerName << " must have exactly one result";
      return signalPassFailure();
    }
    Value markedValue = markerOp->getOperand(0);
    Operation *producer = markedValue.getDefiningOp();
    if (!producer) {
      markerOp->emitError()
          << "marked value has no defining op (block arg / function arg) - "
             "cannot pin attributes";
      return signalPassFailure();
    }

    // Migrate only the dynamatic-namespaced attributes onto the producer.
    for (NamedAttribute attr : markerOp->getDiscardableAttrDictionary()) {
      if (attr.getName().strref().starts_with(kCopiedAttrPrefix))
        producer->setAttr(attr.getName(), attr.getValue());
    }

    // The marker is a passthrough: rewire its consumers back to the
    // original producer's value before erasing it.
    markerOp->getResult(0).replaceAllUsesWith(markedValue);

    markerOp->erase();
  }
}
