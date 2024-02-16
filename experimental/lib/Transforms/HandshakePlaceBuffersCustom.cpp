//===- HandshakePlaceBuffersCustom.cpp - Place buffers in DFG ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Buffer placement pass in Handshake functions, it takes the location (i.e.,
// the predecessor, and which output channel of it), type (i.e., opaque or
// transparent), and slots of the buffer that should be placed.
//
// This pass facilitates externally prototyping a custom buffer placement
// analysis, e.g., in Python. This also makes the results of some research
// artifacts (e.g., Mapbuf) developed in Python easily reproducible in the
// current Dynamatic framework.
//
// Currently, this pass only supports adding units for a IR that is already
// materized (i.e., each value is produced by exactly one producer, and consumed
// by exactly one consumer). For future work on unmaterialized IRs, we need to
// supply the producer--consumer pair as arguments
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "experimental/Transforms/HandshakePlaceBuffersCustom.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::buffer;

namespace {

struct HandshakePlaceBuffersCustomPass
    : public dynamatic::experimental::buffer::impl::
          HandshakePlaceBuffersCustomBase<HandshakePlaceBuffersCustomPass> {

  /// Trivial field-by-field constructor.
  HandshakePlaceBuffersCustomPass(const std::string &pred, const int &outid,
                                  const int &slots, const std::string &type) {
    this->pred = pred;
    this->outid = outid;
    this->slots = slots;
    this->type = type;
  }

  /// Called on the MLIR module provided as input.
  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Check if the IR is after being materialized, if not, reject the input
    // IR.
    if (failed(verifyIRMaterialized(modOp))) {
      modOp->emitError() << ERR_NON_MATERIALIZED_MOD;
      return signalPassFailure();
    }

    OpBuilder builder(ctx);
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    Operation *op = namer.getOp(pred);
    if (!op) {
      llvm::errs() << "No operation named \"" << pred << "\" exists\n";
      return signalPassFailure();
    }
    assert(outid <= op->getNumResults() &&
           "The output id exceeds the number of output ports!");
    Value channel = op->getResult(outid);
    // Set the insertion point to be before the original successor of the
    // channel.
    Operation *succ = *channel.getUsers().begin();
    builder.setInsertionPoint(succ);
    StringAttr slotName = StringAttr::get(ctx, "slots");
    StringRef bufOpType;
    if (type == "oehb") {
      // if the specified type is "oehb", then we add a Opaque buffer with
      // number of slots = slots.
      bufOpType = handshake::OEHBOp::getOperationName();
    } else if (type == "tehb") {
      // If the specified type is "tehb", then we add a Transparent buffer
      // with number of slots = slots.
      bufOpType = handshake::TEHBOp::getOperationName();
    } else {
      llvm::errs() << "Unknown buffer type: \"" << type << "\"!\n";
      return signalPassFailure();
    }
    NamedAttribute namedSlots(slotName, builder.getI32IntegerAttr(slots));
    auto *bufOp =
        builder.create(channel.getLoc(), StringAttr::get(ctx, bufOpType),
                       channel, {channel.getType()}, {namedSlots});
    inheritBB(succ, bufOp);
    Value bufferRes = bufOp->getResult(0);
    succ->replaceUsesOfWith(channel, bufferRes);
  }
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::buffer::createHandshakePlaceBuffersCustom(
    const std::string &pred, const unsigned &outid, const unsigned &slots,
    const std::string &type) {
  return std::make_unique<HandshakePlaceBuffersCustomPass>(pred, outid, slots,
                                                           type);
}
