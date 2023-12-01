//===- BufferPlacementMILP.cpp - MILP-based buffer placement ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the common MILP-based buffer placement infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"

using namespace llvm::sys;
using namespace circt;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

BufferPlacementMILP::BufferPlacementMILP(FuncInfo &funcInfo,
                                         const TimingDatabase &timingDB,
                                         GRBEnv &env, Logger *logger)
    : MILP<BufferPlacement>(env, logger), timingDB(timingDB),
      funcInfo(funcInfo) {

  // Combines any channel-specific buffering properties coming from IR
  // annotations to internal buffer specifications and stores the combined
  // properties into the channel map. Fails and marks the MILP unsatisfiable if
  // any of those combined buffering properties become unsatisfiable.
  auto deriveBufferingProperties = [&](Channel &channel) -> LogicalResult {
    // Increase the minimum number of slots if internal buffers are present, and
    // check for satisfiability
    if (failed(addInternalBuffers(channel))) {
      unsatisfiable = true;
      std::stringstream ss;
      std::string channelName;
      ss << "Including internal component buffers into buffering "
            "properties of channel '"
         << getUniqueName(*channel.value.getUses().begin())
         << "' made them unsatisfiable. Properties are " << *channel.props;
      if (logger)
        **logger << ss.str();
      return channel.producer->emitError() << ss.str();
    }
    channels[channel.value] = *channel.props;
    return success();
  };

  // Add channels originating from function arguments to the channel map
  for (auto [idx, arg] : llvm::enumerate(funcInfo.funcOp.getArguments())) {
    Channel channel(arg, funcInfo.funcOp, *arg.getUsers().begin());
    if (failed(deriveBufferingProperties(channel)))
      return;
  }

  // Add channels originating from operations' results to the channel map
  for (Operation &op : funcInfo.funcOp.getOps()) {
    for (auto [idx, res] : llvm::enumerate(op.getResults())) {
      Channel channel(res, &op, *res.getUsers().begin());
      if (failed(deriveBufferingProperties(channel)))
        return;
    }
  }

  markReadyToOptimize();
}

LogicalResult BufferPlacementMILP::addInternalBuffers(Channel &channel) {
  // Add slots present at the source unit's output ports
  std::string srcName = channel.producer->getName().getStringRef().str();
  if (const TimingModel *model = timingDB.getModel(channel.producer)) {
    channel.props->minTrans += model->outputModel.transparentSlots;
    channel.props->minOpaque += model->outputModel.opaqueSlots;
  }

  // Add slots present at the destination unit's input ports
  std::string dstName = channel.consumer->getName().getStringRef().str();
  if (const TimingModel *model = timingDB.getModel(channel.consumer)) {
    channel.props->minTrans += model->inputModel.transparentSlots;
    channel.props->minOpaque += model->inputModel.opaqueSlots;
  }

  return success(channel.props->isSatisfiable());
}

void BufferPlacementMILP::deductInternalBuffers(Channel &channel,
                                                PlacementResult &result) {
  std::string srcName = channel.producer->getName().getStringRef().str();
  std::string dstName = channel.consumer->getName().getStringRef().str();
  unsigned numTransToDeduct = 0, numOpaqueToDeduct = 0;

  // Remove slots present at the source unit's output ports
  if (const TimingModel *model = timingDB.getModel(channel.producer)) {
    numTransToDeduct += model->outputModel.transparentSlots;
    numOpaqueToDeduct += model->outputModel.opaqueSlots;
  }
  // Remove slots present at the destination unit's input ports
  if (const TimingModel *model = timingDB.getModel(channel.consumer)) {
    numTransToDeduct += model->inputModel.transparentSlots;
    numOpaqueToDeduct += model->inputModel.opaqueSlots;
  }

  assert(result.numTrans >= numTransToDeduct &&
         "not enough transparent slots were placed, the MILP was likely "
         "incorrectly configured");
  assert(result.numOpaque >= numOpaqueToDeduct &&
         "not enough opaque slots were placed, the MILP was likely "
         "incorrectly configured");
  result.numTrans -= numTransToDeduct;
  result.numOpaque -= numOpaqueToDeduct;
}

void BufferPlacementMILP::forEachIOPair(
    Operation *op, const std::function<void(Value, Value)> &callback) {
  for (Value opr : op->getOperands())
    if (!isa<MemRefType>(opr.getType()))
      for (OpResult res : op->getResults())
        if (!isa<MemRefType>(res.getType()))
          callback(opr, res);
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
