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
  LogicalResult insertRigidifier(AbsenceOfBackpressure prop, MLIRContext *ctx);
  LogicalResult insertValidMerger(ValidEquivalence prop, MLIRContext *ctx);
};
} // namespace

void HandshakeRigidificationPass::runDynamaticPass() {
  MLIRContext *ctx = &getContext();
  FormalPropertyTable table;
  if (failed(table.addPropertiesFromJSON(jsonPath)))
    llvm::errs() << "[WARNING] Formal property retrieval failed\n";

  for (auto &property : table.getProperties()) {
    if (property->getTag() == FormalProperty::TAG::OPT &&
        property->getCheck() == "true") {
      if (isa<AbsenceOfBackpressure>(property)) {
        auto *p = llvm::cast<AbsenceOfBackpressure>(property.get());
        if (failed(insertRigidifier(*p, ctx)))
          return signalPassFailure();
      } else if (isa<ValidEquivalence>(property)) {
        auto *p = llvm::cast<ValidEquivalence>(property.get());
        if (failed(insertValidMerger(*p, ctx)))
          return signalPassFailure();
      }
    }
  }
}

LogicalResult
HandshakeRigidificationPass::insertRigidifier(AbsenceOfBackpressure prop,
                                              MLIRContext *ctx) {
  OpBuilder builder(ctx);

  Operation *ownerOp = getAnalysis<NameAnalysis>().getOp(prop.getOwner());
  auto channel = ownerOp->getResult(prop.getOwnerIndex());

  builder.setInsertionPointAfter(ownerOp);
  auto loc = channel.getLoc();

  auto newOp = builder.create<handshake::RigidifierOp>(loc, channel);
  Value rigidificationRes = newOp.getResult();

  for (auto &use : llvm::make_early_inc_range(channel.getUses())) {
    if (use.getOwner() != newOp) {
      use.set(rigidificationRes);
    }
  }
  return success();
}

LogicalResult
HandshakeRigidificationPass::insertValidMerger(ValidEquivalence prop,
                                               MLIRContext *ctx) {
  OpBuilder builder(ctx);

  Operation *ownerOp = getAnalysis<NameAnalysis>().getOp(prop.getOwner());
  Operation *targetOp = getAnalysis<NameAnalysis>().getOp(prop.getTarget());
  auto ownerChannel = ownerOp->getResult(prop.getOwnerIndex());
  auto targetChannel = targetOp->getResult(prop.getTargetIndex());

  builder.setInsertionPointAfter(targetOp);
  auto loc = ownerChannel.getLoc();

  auto newOp = builder.create<handshake::ValidMergerOp>(loc, ownerChannel,
                                                        targetChannel);
  auto mergerRes0 = newOp.getResult(0);
  auto mergerRes1 = newOp.getResult(1);

  for (auto &use : llvm::make_early_inc_range(ownerChannel.getUses())) {
    if (use.getOwner() != newOp) {
      use.set(mergerRes0);
    }
  }
  for (auto &use : llvm::make_early_inc_range(targetChannel.getUses())) {
    if (use.getOwner() != newOp) {
      use.set(mergerRes1);
    }
  }
  return success();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::rigidification::createRigidification(
    const std::string &jsonPath) {
  return std::make_unique<HandshakeRigidificationPass>(jsonPath);
}
