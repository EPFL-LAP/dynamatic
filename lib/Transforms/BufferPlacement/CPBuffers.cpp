//===- CPBuffers.cpp - Critical-path-only buffer placement ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/CPBuffers.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"

#define DEBUG_TYPE "cp-buffers"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::cpbuf;

CPBuffers::CPBuffers(CPSolver::SolverKind solverKind, int timeout,
                     FuncInfo &funcInfo, const TimingDatabase &timingDB,
                     double targetPeriod, StringRef writeTo)
    : BufferPlacementMILP(solverKind, timeout, funcInfo, timingDB, targetPeriod,
                          Algorithm::CPBUF, writeTo) {
  if (!unsatisfiable)
    setup();
}

void CPBuffers::extractResult(BufferPlacement &placement) {
  for (auto &[channel, chVars] : vars.channelVars) {
    // The old CP implementation deliberately leaves non-control unbundle
    // outputs untouched.
    if (auto *op = channel.getDefiningOp();
        op && isa<handshake::UnbundleOp>(op) &&
        !isa<handshake::ControlType>(channel.getType()))
      continue;

    unsigned numSlots =
        static_cast<unsigned>(model->getValue(chVars.bufNumSlots) + 0.5);
    bool breaksData =
        model->getValue(chVars.signalVars[SignalType::DATA].bufPresent) > 0;

    PlacementResult result;
    if (numSlots >= 1) {
      if (breaksData) {
        result.numOneSlotDV = 1;
        result.numFifoNone = numSlots - 1;
      } else {
        result.numFifoNone = numSlots;
      }
    }

    Operation *srcOp = channel.getDefiningOp();
    bool needsReadyPathBuffer =
        isa_and_nonnull<handshake::MuxOp, handshake::MergeOp>(srcOp) &&
        srcOp->getNumOperands() > 1;
    if (needsReadyPathBuffer && isChannelOnCycle(channel))
      result.numOneSlotR = 1;

    placement[channel] = result;
  }

  LLVM_DEBUG(logResults(placement););
}

void CPBuffers::addCustomChannelConstraints(Value channel) {
  ChannelVars &chVars = vars.channelVars[channel];
  handshake::ChannelBufProps &props = channelProps[channel];
  CPVar &dataBuf = chVars.signalVars[SignalType::DATA].bufPresent;

  if (props.minOpaque > 0) {
    model->addConstr(dataBuf == 1, "custom_forceOpaque");
    if (props.minTrans > 0) {
      model->addConstr(chVars.bufNumSlots >= props.minOpaque + props.minTrans,
                       "custom_minOpaqueAndTrans");
    } else {
      model->addConstr(chVars.bufNumSlots >= props.minOpaque,
                       "custom_minOpaque");
    }
  } else if (props.minTrans > 0) {
    model->addConstr(chVars.bufNumSlots >= props.minTrans + dataBuf,
                     "custom_minTrans");
  } else if (props.minSlots > 0) {
    model->addConstr(chVars.bufNumSlots >= props.minSlots, "custom_minSlots");
  }
  if (props.minOpaque + props.minTrans + props.minSlots > 0)
    model->addConstr(chVars.bufPresent == 1, "custom_forceBuffers");

  if (props.maxOpaque.has_value()) {
    if (*props.maxOpaque == 0)
      model->addConstr(dataBuf == 0, "custom_forceTransparent");
    if (props.maxTrans.has_value()) {
      unsigned maxSlots = *props.maxTrans + *props.maxOpaque;
      if (maxSlots == 0) {
        model->addConstr(chVars.bufPresent == 0, "custom_noBuffers");
        model->addConstr(chVars.bufNumSlots == 0, "custom_noSlots");
      } else {
        model->addConstr(chVars.bufNumSlots <= maxSlots, "custom_maxSlots");
      }
    }
  }
}

void CPBuffers::setup() {
  SmallVector<SignalType, 1> signalTypes{SignalType::DATA};
  const TimingModel *bufModel = nullptr;
  SmallVector<BufferingGroup> bufGroups;
  bufGroups.emplace_back(ArrayRef<SignalType>{SignalType::DATA}, bufModel);

  std::vector<Value> allChannels;
  for (auto &[channel, _] : channelProps) {
    allChannels.push_back(channel);
    addChannelVars(channel, signalTypes);
    addCustomChannelConstraints(channel);

    if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
        !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {
      addChannelTimingConstraints(channel, SignalType::DATA, bufModel);
      addBufferPresenceConstraints(channel);
      addBufferingGroupConstraints(channel, bufGroups);
    }
  }

  for (Operation &op : funcInfo.funcOp.getOps())
    addUnitTimingConstraints(&op, SignalType::DATA);

  // Match the original replace-lsq CP objective exactly: satisfy the timing
  // constraints with the fewest buffers, then prefer fewer total slots.
  LinExpr objective;
  for (Value channel : allChannels) {
    ChannelVars &chVars = vars.channelVars[channel];
    objective -= chVars.bufPresent;
    objective -= 0.1 * chVars.bufNumSlots;
  }
  model->setMaximizeObjective(objective);
  markReadyToOptimize();
}
