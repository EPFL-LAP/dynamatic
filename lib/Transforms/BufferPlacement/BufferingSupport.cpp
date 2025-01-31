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

void Channel::addInternalBuffers(const TimingDatabase &timingDB) {
  // Add slots present at the source unit's output ports
  if (const TimingModel *model = timingDB.getModel(producer)) {
    props->minTrans += model->outputModel.transparentSlots;
    props->minOpaque += model->outputModel.opaqueSlots;
  }

  // Add slots present at the destination unit's input ports
  if (const TimingModel *model = timingDB.getModel(consumer)) {
    props->minTrans += model->inputModel.transparentSlots;
    props->minOpaque += model->inputModel.opaqueSlots;
  }
}

void PlacementResult::deductInternalBuffers(const Channel &channel,
                                            const TimingDatabase &timingDB) {
  unsigned numTransToDeduct = 0, numOpaqueToDeduct = 0;

  // Remove slots present at the source unit's output ports. If the channel is a
  // function argument, the model will be nullptr (since Handshake functions do
  // not have a timing model) and nothing will happen for the producer
  if (const TimingModel *model = timingDB.getModel(channel.producer)) {
    numTransToDeduct += model->outputModel.transparentSlots;
    numOpaqueToDeduct += model->outputModel.opaqueSlots;
  }

  // Remove slots present at the destination unit's input ports
  if (const TimingModel *model = timingDB.getModel(channel.consumer)) {
    numTransToDeduct += model->inputModel.transparentSlots;
    numOpaqueToDeduct += model->inputModel.opaqueSlots;
  }

  // Adjust placement results
  assert(numTrans >= numTransToDeduct && "not enough transparent slots");
  assert(numOpaque >= numOpaqueToDeduct && "not enough opaque slots");
  numTrans -= numTransToDeduct;
  numOpaque -= numOpaqueToDeduct;
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

    llvm::errs() << "------------------------1 \n";

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

    llvm::errs() << "***** " << "\n";
    // Increase the minimum number of slots if internal buffers are present, and
    // check for satisfiability
    channel.addInternalBuffers(timingDB);
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
    llvm::errs() << "************* " << "\n";
    return success();
  };


llvm::errs() << "------------------------404 \n";

  // Add channels originating from function arguments to the channel map
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    llvm::errs() << "[] " << idx << arg << "\n";
    // if (idx == 7){
    // for (auto a : arg.getUsers()){
    //   llvm::errs() << "[] " << a << "\n";
    //   llvm::errs() << "[] " << *a << "\n";
    //   for (auto b : (*a).getUsers()){
    //     llvm::errs() << "[] " << b << "\n";
    //     llvm::errs() << "[] " << *b << "\n";
    //     }
    // }
    // }

      
    Channel channel(arg, funcOp, *arg.getUsers().begin());
    if (failed(deriveBufferingProperties(channel))){
      llvm::errs() << "------------------------4 \n";
      exit(0);
    }

  }

  

  // Add channels originating from operations' results to the channel map
  for (Operation &op : funcOp.getOps()) {
    for (auto [idx, res] : llvm::enumerate(op.getResults())) {
      llvm::errs() << "[*] " << idx << res << "\n";
      Channel channel(res, &op, *res.getUsers().begin());
      if (failed(deriveBufferingProperties(channel))){

        llvm::errs() << "------------------------3 \n";
        exit(0);
      }

    }
  }

  llvm::errs() << " success ?! ------- \n";
  return success();
}
