//===- ConsumeProducerOutputAttrMarker.cpp - Move marker attrs to producer ===//
//
// We store producer-output attributes on unregistered ops in the cf dialect
// rather than on a function call, since it creates a cleaner IR
// but still prevents the value from being folded away.
//
// Once we are ready to run CF to Handshake,
// we can place the attribute on the producer
// since we are no longer in danger of the value being folded away
// and this means we do not need to edit CF to Handshake
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
#define GEN_PASS_DEF_CONSUMEPRODUCEROUTPUTATTRMARKER
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic

// The unregistered op name emitted by `translate-llvm-to-std`
static constexpr llvm::StringLiteral producerOutputAttrMarkerName =
    "dynamatic.producer_output_attr_marker";

// Only attributes in the dynamatic namespace are moved
// from the marker to the producer
static constexpr llvm::StringLiteral dynamaticNamespace = "dynamatic.";

namespace {
struct ConsumeProducerOutputAttrMarkerPass
    : public dynamatic::impl::ConsumeProducerOutputAttrMarkerBase<
          ConsumeProducerOutputAttrMarkerPass> {
  using ConsumeProducerOutputAttrMarkerBase::
      ConsumeProducerOutputAttrMarkerBase;
  void runOnOperation() override {
    ModuleOp modOp = getOperation();

    // collect all of the producer-output attribute marker ops
    SmallVector<Operation *> markers;
    modOp.walk([&](Operation *op) {
      if (op->getName().getStringRef() == producerOutputAttrMarkerName)
        markers.push_back(op);
    });

    // for each marker
    for (Operation *markerOp : markers) {
      // here we verify that the unregistered op
      // has a sensible shape
      // e.g. one input one output
      if (markerOp->getNumOperands() != 1) {
        markerOp->emitError()
            << producerOutputAttrMarkerName << " must have exactly one operand";
        return signalPassFailure();
      }
      if (markerOp->getNumResults() != 1) {
        markerOp->emitError()
            << producerOutputAttrMarkerName << " must have exactly one result";
        return signalPassFailure();
      }

      // get the operand of the marker op
      Value producerOutput = markerOp->getOperand(0);

      // get the producer
      Operation *producer = producerOutput.getDefiningOp();

      if (!producer) {
        markerOp->emitError() << "producer-output attribute marker has no "
                                 "producer (block arg / function "
                                 "arg) - cannot pin attribute";
        return signalPassFailure();
      }

      // Record which output of the producer this marker applies to
      unsigned resultIdx = cast<OpResult>(producerOutput).getResultNumber();
      // and convert it to an IntegerAttr so we can store it on the IR
      auto idxAttr = IntegerAttr::get(
          IntegerType::get(markerOp->getContext(), 32), resultIdx);

      // Move only the attributes in the dynamatic namespace, and for each
      // one also set a sibling `<name>.result_idx` attribute on the producer.
      for (NamedAttribute attr : markerOp->getDiscardableAttrDictionary()) {
        if (!attr.getName().strref().starts_with(dynamaticNamespace))
          continue;
        // store the attribute on the producer
        producer->setAttr(attr.getName(), attr.getValue());
        // and store which result the attribute applies to
        producer->setAttr(attr.getName().strref().str() + ".result_idx",
                          idxAttr);
      }

      // The attribute marker op is a passthrough:
      // rewire its consumers back to the
      // original producer's value before erasing it.
      markerOp->getResult(0).replaceAllUsesWith(producerOutput);

      markerOp->erase();
    }
  }
};
} // namespace
