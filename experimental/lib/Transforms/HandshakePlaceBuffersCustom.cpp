//===- HandshakePlaceBuffersCustom.cpp - Place buffers in DFG ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Buffer placement pass in Handshake functions, it takes the location (i.e.,
// the predecessor, and which output channel of it), type, and slots of the
// buffer that should be placed.
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
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;
using namespace dynamatic;

// [START Boiler-plate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_HANDSHAKEPLACEBUFFERSCUSTOM
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boiler-plate code for the MLIR pass]

namespace {

struct HandshakePlaceBuffersCustomPass
    : public dynamatic::experimental::impl::HandshakePlaceBuffersCustomBase<
          HandshakePlaceBuffersCustomPass> {

  using HandshakePlaceBuffersCustomBase::HandshakePlaceBuffersCustomBase;

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
    assert(outid < op->getNumResults() &&
           "The output id exceeds the number of output ports!");
    Value channel = op->getResult(outid);
    // Set the insertion point to be before the original successor of the
    // channel.
    Operation *succ = *channel.getUsers().begin();
    builder.setInsertionPoint(succ);

    transform(type.begin(), type.end(), type.begin(), ::toupper);

    // returns optional wrapper around buffer type enum
    auto bufferTypeOpt = handshake::symbolizeBufferType(type);

    if (!bufferTypeOpt.has_value()) {
      llvm::errs() << "Unknown buffer type: \"" << type << "\"!\n";
      return signalPassFailure();
    }

    // pull the enum itself from the optional
    auto bufferType = bufferTypeOpt.value();

    auto bufOp = builder.create<handshake::BufferOp>(channel.getLoc(), channel,
                                                     slots, bufferType);
    inheritBB(succ, bufOp);
    Value bufferRes = bufOp->getResult(0);
    succ->replaceUsesOfWith(channel, bufferRes);
  }
};
} // namespace
