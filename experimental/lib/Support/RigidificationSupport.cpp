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

using namespace dynamatic;

LogicalResult rigidifyChannel(Value *channel) {
  // if (llvm::dyn_cast<handshake::ChannelType>(channel->getType()))
  return LogicalResult::success();
}