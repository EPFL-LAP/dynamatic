//===- HandshakePlaceBuffersCustom.cpp - Place buffers in DFG ---------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Buffer placement in Handshake functions, taking a list of
// predecessor:successor pairs, for example "mux1:add1 mux2:add2"
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/HandshakePlaceBuffersCustom.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/Logging.h"

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
    if (failed(verifyIRMaterialized(modOp))) {
      modOp->emitError() << ERR_NON_MATERIALIZED_MOD;
      return;
    }

    OpBuilder builder(ctx);
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>())
      for (Operation &op : funcOp.getOps()) {
        if (getUniqueName(&op).str() == pred) {
          assert(outid <= op.getNumResults() &&
                 "The output id exceeds the number of output ports!");
          Value channel = op.getResult(outid);
          // Set the insertion point to be before the original successor of the
          // channel.
          Operation *succ = channel.getUses().begin()->getOwner();
          builder.setInsertionPoint(succ);
	  // if the specified type is "oehb", then we add a Opaque buffer with number of slots = slots.
          if (type == "oehb") {
            auto bufOp = builder.create<handshake::OEHBOp>(channel.getLoc(),
                                                           channel, slots);
            inheritBB(succ, bufOp);
            Value bufferRes = bufOp.getResult();
            succ->replaceUsesOfWith(channel, bufferRes);
	  // if the specified type is "tehb", then we add a Transparent buffer with number of slots = slots.
          } else if (type == "tehb") {
            auto bufOp = builder.create<handshake::TEHBOp>(channel.getLoc(),
                                                           channel, slots);
            inheritBB(succ, bufOp);
            Value bufferRes = bufOp.getResult();
            succ->replaceUsesOfWith(channel, bufferRes);
          } else {
    		llvm::errs() << "Unknown buffer type: \"" << type << "\"!\n";
    	return signalPassFailure();

	  }
          return;
        }
      }

    llvm::errs() << "The unit " << pred
                 << "does not exist in the dataflow graph!"
                 << "\n";
    return signalPassFailure();
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
