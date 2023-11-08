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
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace dynamatic;
using namespace dynamatic::buffer;

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
