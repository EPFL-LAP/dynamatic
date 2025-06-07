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

#include "experimental/Transforms/Rigidification/HandshakeRigidification.h"
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
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/FormalProperty.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <ostream>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::rigidification;

namespace {

struct HandshakeRigidificationPass
    : public dynamatic::experimental::rigidification::impl::
          HandshakeRigidificationBase<HandshakeRigidificationPass> {

  HandshakeRigidificationPass(const std::string &jsonPath = "") {
    this->jsonPath = jsonPath;
  }

  void runDynamaticPass() override;

private:
  LogicalResult insertReadyRemover(AbsenceOfBackpressure prop);
  LogicalResult insertValidMerger(ValidEquivalence prop);
};
} // namespace

void HandshakeRigidificationPass::runDynamaticPass() {
  FormalPropertyTable table;
  if (failed(table.addPropertiesFromJSON(jsonPath)))
    llvm::errs() << "[WARNING] Formal property retrieval failed\n";

  for (const auto &property : table.getProperties()) {
    if (property->getTag() == FormalProperty::TAG::OPT &&
        property->getCheck() != std::nullopt && *property->getCheck()) {

      if (auto *p = dyn_cast<AbsenceOfBackpressure>(property.get())) {
        if (failed(insertReadyRemover(*p)))
          return signalPassFailure();

      } else if (auto *p = dyn_cast<ValidEquivalence>(property.get())) {
        if (failed(insertValidMerger(*p)))
          return signalPassFailure();
      }
    }
  }
}

LogicalResult
HandshakeRigidificationPass::insertReadyRemover(AbsenceOfBackpressure prop) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  Operation *ownerOp = getAnalysis<NameAnalysis>().getOp(prop.getOwner());
  auto channel = ownerOp->getResult(prop.getOwnerIndex());

  builder.setInsertionPointAfter(ownerOp);
  auto loc = channel.getLoc();

  auto newOp = builder.create<handshake::ReadyRemoverOp>(loc, channel);
  channel.replaceAllUsesExcept(newOp.getResult(), newOp);

  return success();
}

LogicalResult
HandshakeRigidificationPass::insertValidMerger(ValidEquivalence prop) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  Operation *ownerOp = getAnalysis<NameAnalysis>().getOp(prop.getOwner());
  Operation *targetOp = getAnalysis<NameAnalysis>().getOp(prop.getTarget());
  auto ownerChannel = ownerOp->getResult(prop.getOwnerIndex());
  auto targetChannel = targetOp->getResult(prop.getTargetIndex());

  builder.setInsertionPointAfter(targetOp);
  Location loc =
      FusedLoc::get(ctx, {ownerChannel.getLoc(), targetChannel.getLoc()});

  auto newOp = builder.create<handshake::ValidMergerOp>(loc, ownerChannel,
                                                        targetChannel);

  ownerChannel.replaceAllUsesExcept(newOp.getLhs(), newOp);
  targetChannel.replaceAllUsesExcept(newOp.getRhs(), newOp);
  return success();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::rigidification::createRigidification(
    const std::string &jsonPath) {
  return std::make_unique<HandshakeRigidificationPass>(jsonPath);
}
