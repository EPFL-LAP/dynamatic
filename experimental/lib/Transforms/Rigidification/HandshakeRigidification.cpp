//===- HandshakeRigidificationcpp - Rigidification --------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --handshake-rigidification pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/CFDFC.h"
#include "experimental/Support/FormalProperty.h"
#include "experimental/Support/PropertyExpr.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <fstream>
#include <ostream>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_HANDSHAKERIGIDIFICATION
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

namespace {

struct HandshakeRigidificationPass
    : public dynamatic::experimental::impl::HandshakeRigidificationBase<
          HandshakeRigidificationPass> {
  using HandshakeRigidificationBase::HandshakeRigidificationBase;

  void runOnOperation() override;

private:
  // The channels are given by the valid signals of the result ports that
  // produce them
  LogicalResult insertReadyRemover(const Signal &valid);
  LogicalResult insertValidMerger(const Signal &ownerValid,
                                  const Signal &targetValid);
};
} // namespace

void HandshakeRigidificationPass::runOnOperation() {
  // Make sure all operation names are unique and haven't changed from what is
  // cached (if anything is cached)
  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  if (!namer.isAnalysisValid())
    return signalPassFailure();

  FormalPropertyTable table;
  if (failed(table.addPropertiesFromJSON(jsonPath)))
    llvm::errs() << "[WARNING] Formal property retrieval failed\n";

  for (const auto &property : table.getProperties()) {
    if (property.tag == FormalProperty::TAG::OPT &&
        property.check.value_or(false)) {

      std::vector<Signal> valids = property.getResultValidSignals();
      if (property.kind == FormalProperty::ABSENCE_OF_BACKPRESSURE) {
        assert(valids.size() == 1 && "expected the valid signal of a channel");
        if (failed(insertReadyRemover(valids[0])))
          return signalPassFailure();

      } else if (property.kind == FormalProperty::VALID_EQUIVALENCE) {
        assert(valids.size() == 2 &&
               "expected the valid signals of two channels");
        if (failed(insertValidMerger(valids[0], valids[1])))
          return signalPassFailure();
      }
    }
  }

  // Make sure all operation names are still unique and name operations that
  // do not currently have a name
  if (failed(namer.walk(NameAnalysis::UnnamedBehavior::NAME)))
    return signalPassFailure();
  markAnalysesPreserved<NameAnalysis>();
}

LogicalResult
HandshakeRigidificationPass::insertReadyRemover(const Signal &valid) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  Operation *ownerOp = getAnalysis<NameAnalysis>().getOp(valid.getOpName());
  auto channel = ownerOp->getResult(valid.getPortIndex());

  builder.setInsertionPointAfter(ownerOp);
  auto loc = channel.getLoc();

  auto newOp = builder.create<handshake::ReadyRemoverOp>(loc, channel);
  channel.replaceAllUsesExcept(newOp.getResult(), newOp);

  return success();
}

LogicalResult
HandshakeRigidificationPass::insertValidMerger(const Signal &ownerValid,
                                               const Signal &targetValid) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  Operation *ownerOp =
      getAnalysis<NameAnalysis>().getOp(ownerValid.getOpName());
  Operation *targetOp =
      getAnalysis<NameAnalysis>().getOp(targetValid.getOpName());
  auto ownerChannel = ownerOp->getResult(ownerValid.getPortIndex());
  auto targetChannel = targetOp->getResult(targetValid.getPortIndex());

  builder.setInsertionPointAfter(targetOp);
  Location loc =
      FusedLoc::get(ctx, {ownerChannel.getLoc(), targetChannel.getLoc()});

  auto newOp = builder.create<handshake::ValidMergerOp>(loc, ownerChannel,
                                                        targetChannel);

  ownerChannel.replaceAllUsesExcept(newOp.getLhsOut(), newOp);
  targetChannel.replaceAllUsesExcept(newOp.getRhsOut(), newOp);
  return success();
}
