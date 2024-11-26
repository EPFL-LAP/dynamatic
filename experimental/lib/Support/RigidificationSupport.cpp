//===- RigidificationSupport.cpp - rigidification support -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the rigidification functionality that eliminates some
// control signals to reduce the handshake overhead
//
//===----------------------------------------------------------------------===//
#include "experimental/Support/RigidificationSupport.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/ValueRange.h"

using namespace dynamatic;

LogicalResult rigidifyChannel(Value *channel, MLIRContext *ctx) {
  if (!(llvm::dyn_cast<handshake::ChannelType>(channel->getType())))
    return LogicalResult::failure();

  OpBuilder builder(ctx);
  builder.setInsertionPointAfter(channel->getDefiningOp());
  auto loc = channel->getLoc();

  auto newUnbundleOp = builder.create<handshake::UnbundleOp>(loc, *channel);
  Value data = newUnbundleOp.getResult(0);
  Value control = newUnbundleOp.getResult(1);
  handshake::ChannelType channelType = llvm::dyn_cast<handshake::ChannelType>(channel->getType());

  auto newBundleOp = builder.create<handshake::BundleOp>(
      loc, data, control, ValueRange(channelType.getExtraSignals()), channelType);

  Value bundleRes = newBundleOp.getResult(0);

  for (auto &use : llvm::make_early_inc_range(channel->getUses())) {
      if (use.getOwner() != newUnbundleOp) {
          use.set(bundleRes);
      }
  }
  return LogicalResult::success();
}