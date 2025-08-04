//===- BufferingSupport.cpp - Support for buffer placement ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Infrastructure for working around the buffer placement pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::buffer;

bool LazyChannelBufProps::updateIR() {
  bool updated = updateIRIfNecessary();
  if (updated)
    unchangedProps = props;
  return updated;
}

ChannelBufProps &LazyChannelBufProps::operator*() {
  if (!props.has_value())
    readAttribute();
  return *props;
  ;
}

ChannelBufProps *LazyChannelBufProps::operator->() {
  if (!props.has_value())
    readAttribute();
  return &*props;
}

LazyChannelBufProps::~LazyChannelBufProps() {
  if (updateOnDestruction)
    updateIRIfNecessary();
}

void LazyChannelBufProps::readAttribute() {
  ChannelBufPropsAttr optProps =
      getOperandAttr<ChannelBufPropsAttr>(*val.getUses().begin());
  props = optProps ? optProps.getProps() : ChannelBufProps();
  unchangedProps = props;
}

bool LazyChannelBufProps::updateIRIfNecessary() {
  if (!props.has_value() || *props == *unchangedProps)
    return false;
  setOperandAttr(*val.getUses().begin(),
                 ChannelBufPropsAttr::get(val.getContext(), *props));
  return true;
}

Channel::Channel(Value value, bool updateProps)
    : value(value), consumer(*value.getUsers().begin()),
      props(value, updateProps) {
  if (OpResult res = dyn_cast<OpResult>(value)) {
    producer = value.getDefiningOp();
    return;
  }
  // Channel must be a block argument: make the parent operation the "producer"
  BlockArgument arg = cast<BlockArgument>(value);
  producer = arg.getParentBlock()->getParentOp();
};

OpOperand &Channel::getOperand() const {
  for (OpOperand &oprd : consumer->getOpOperands()) {
    if (oprd.get() == value)
      return oprd;
  }
  llvm_unreachable("channel consumer does not have value as operand");
}

Operation *dynamatic::buffer::getChannelProducer(Value channel, size_t *idx) {
  if (OpResult res = dyn_cast<OpResult>(channel)) {
    if (idx)
      *idx = res.getResultNumber();
    return channel.getDefiningOp();
  }
  // Channel must be a block argument. In this case we only support buffering
  // properties the channel maps to a Handshake function argument
  BlockArgument arg = cast<BlockArgument>(channel);
  Operation *op = arg.getParentBlock()->getParentOp();
  if (isa<handshake::FuncOp>(op)) {
    if (idx)
      *idx = arg.getArgNumber();
    return op;
  }
  return nullptr;
}

LogicalResult dynamatic::buffer::mapChannelsToProperties(
    handshake::FuncOp funcOp, const TimingDatabase &timingDB,
    llvm::MapVector<Value, ChannelBufProps> &channelProps) {

  // Combines any channel-specific buffering properties coming from IR
  // annotations to internal buffer specifications and stores the combined
  // properties into the channel map. Fails and marks the MILP unsatisfiable if
  // any of those combined buffering properties become unsatisfiable.
  auto deriveBufferingProperties = [&](Channel &channel) -> LogicalResult {
    ChannelBufProps ogProps = *channel.props;
    if (!ogProps.isSatisfiable()) {
      std::stringstream ss;
      std::string channelName;
      ss << "Channel buffering properties of channel '"
         << getUniqueName(*channel.value.getUses().begin())
         << "' are unsatisfiable " << ogProps
         << "Cannot proceed with buffer placement.";
      return channel.consumer->emitError() << ss.str();
    }

    // Check for satisfiability
    if (!channel.props->isSatisfiable()) {
      std::stringstream ss;
      std::string channelName;
      ss << "Including internal component buffers into buffering "
            "properties of channel '"
         << getUniqueName(*channel.value.getUses().begin())
         << "' made them unsatisfiable.\nProperties were " << ogProps
         << "before inclusion and were changed to " << *channel.props
         << "Cannot proceed with buffer placement.";
      return channel.consumer->emitError() << ss.str();
    }
    channelProps[channel.value] = *channel.props;
    return success();
  };

  // Add channels originating from function arguments to the channel map
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    Channel channel(arg, funcOp, *arg.getUsers().begin());
    if (failed(deriveBufferingProperties(channel)))
      return failure();
  }

  // Add channels originating from operations' results to the channel map
  for (Operation &op : funcOp.getOps()) {
    for (auto [idx, res] : llvm::enumerate(op.getResults())) {
      Channel channel(res, &op, *res.getUsers().begin());
      if (failed(deriveBufferingProperties(channel)))
        return failure();
    }
  }

  return success();
}
