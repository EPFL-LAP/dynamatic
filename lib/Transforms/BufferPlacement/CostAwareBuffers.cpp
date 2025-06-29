//===- CostAwareBuffers.cpp - Cost-aware buffer placement -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements cost-aware smart buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/CostAwareBuffers.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "mlir/IR/Value.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::costaware;

CostAwareBuffers::CostAwareBuffers(GRBEnv &env, FuncInfo &funcInfo,
                                   const TimingDatabase &timingDB,
                                   double targetPeriod)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod) {
  if (!unsatisfiable)
    setup();
}

CostAwareBuffers::CostAwareBuffers(GRBEnv &env, FuncInfo &funcInfo,
                                   const TimingDatabase &timingDB,
                                   double targetPeriod, Logger &logger,
                                   StringRef milpName)
    : BufferPlacementMILP(env, funcInfo, timingDB, targetPeriod, logger,
                          milpName) {
  if (!unsatisfiable)
    setup();
}

void CostAwareBuffers::extractResult(BufferPlacement &placement) {
  // Iterate over all channels in the circuit
  for (auto &[channel, channelVars] : vars.channelVars) {
    // Extract number and type of slots from the MILP solution, as well as
    // channel-specific buffering properties
    unsigned numSlotsToPlace = static_cast<unsigned>(
        channelVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    if (numSlotsToPlace == 0)
      continue;

    unsigned dataLatency = static_cast<unsigned>(
        channelVars.dataLatency.get(GRB_DoubleAttr_X) + 0.5);
    unsigned readyLatency = static_cast<unsigned>(
        channelVars.signalVars[SignalType::READY].bufPresent.get(
            GRB_DoubleAttr_X) +
        0.5);
    bool useShiftReg = channelVars.shiftReg.get(GRB_DoubleAttr_X) > 0.5;

    PlacementResult result;
    // See 'docs/Specs/Buffering/Buffering.md'.
    // This algorithm does not use `FIFO_BREAK_DV`; instead, it represents it as
    // `ONE_SLOT_BREAK_DV` followed by `FIFO_BREAK_NONE`, because the
    // interpretation is cleaner: `BREAK_DV` modules represent slots with data
    // latency, while `BREAK_NONE` modules represent zero-latency slots.
    result.numOneSlotDV = useShiftReg ? 0 : dataLatency;
    result.numShiftRegDV = useShiftReg ? dataLatency : 0;
    result.numOneSlotR = readyLatency;
    result.numFifoNone = numSlotsToPlace - dataLatency - readyLatency;

    placement[channel] = result;
  }

  if (logger)
    logResults(placement);

  llvm::MapVector<size_t, double> cfdfcTPResult;
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfdfcVars)) {
    auto [cf, cfVars] = cfdfcWithVars;
    double tmpThroughput = cfVars.throughput.get(GRB_DoubleAttr_X);

    cfdfcTPResult[idx] = tmpThroughput;
  }

  // Create and add the handshake.tp attribute
  auto cfdfcTPMap = handshake::CFDFCThroughputAttr::get(
      funcInfo.funcOp.getContext(), cfdfcTPResult);
  setDialectAttr(funcInfo.funcOp, cfdfcTPMap);
}

void CostAwareBuffers::addCustomChannelConstraints(Value channel) {
  ChannelVars &chVars = vars.channelVars[channel];
  handshake::ChannelBufProps &props = channelProps[channel];
  GRBVar &dataLatency = chVars.dataLatency;
  for (mlir::Operation *user : channel.getUsers()) {
    if (isa<handshake::LoadOp>(user)) {
      props.minTrans = 0;
      break;
    }
  }

  if (props.minOpaque > 0) {
    // Force the MILP to place a minimum number of opaque slots
    model.addConstr(dataLatency >= props.minOpaque, "custom_forceOpaque");
  }
  if (props.minTrans > 0) {
    // Force the MILP to place a minimum number of transparent slots
    model.addConstr(chVars.bufNumSlots >= props.minTrans + dataLatency,
                    "custom_minTrans");
  }
  if (props.minSlots > 0) {
    // Force the MILP to place a minimum number of slots
    model.addConstr(chVars.bufNumSlots >= props.minSlots, "custom_minSlots");
  }
  if (props.minOpaque + props.minTrans + props.minSlots > 0)
    model.addConstr(chVars.bufPresent == 1, "custom_forceBuffers");

  // Set a maximum number of slots to be placed
  if (props.maxOpaque.has_value()) {
    if (*props.maxOpaque == 0) {
      // Force the MILP to use transparent slots
      model.addConstr(dataLatency == 0, "custom_forceTransparent");
    }
    if (props.maxTrans.has_value()) {
      // Force the MILP to use a maximum number of slots
      unsigned maxSlots = *props.maxTrans + *props.maxOpaque;
      if (maxSlots == 0) {
        model.addConstr(chVars.bufPresent == 0, "custom_noBuffers");
        model.addConstr(chVars.bufNumSlots == 0, "custom_noSlots");
      } else {
        model.addConstr(chVars.bufNumSlots <= maxSlots, "custom_maxSlots");
      }
    }
  }
}

void CostAwareBuffers::setup() {
  // Signals for which we have variables
  SmallVector<SignalType, 4> signals;
  signals.push_back(SignalType::DATA);
  signals.push_back(SignalType::VALID);
  signals.push_back(SignalType::READY);

  const TimingModel *bufModel = nullptr;

  // Create channel variables and constraints
  std::vector<Value> allChannels;
  for (auto &[channel, _] : channelProps) {
    allChannels.push_back(channel);
    addChannelVars(channel, signals);
    addCustomChannelConstraints(channel);

    // Add timing constraints for all channels in the function
    // that are not adjacent to a memory interface
    if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
        !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {
      addChannelTimingConstraints(channel, SignalType::DATA, bufModel);
      addChannelTimingConstraints(channel, SignalType::READY, bufModel);
      addBufferPresenceConstraints(channel);
      addBufferLatencyConstraints(channel);
    }
  }

  // Add timing constraints for all units in the function
  for (Operation &op : funcInfo.funcOp.getOps()) {
    addUnitTimingConstraints(&op, SignalType::DATA);
    addUnitTimingConstraints(&op, SignalType::READY);
  }

  // Create CFDFC variables and add throughput constraints for each CFDFC that
  // was marked to be optimized
  SmallVector<CFDFC *> cfdfcs;
  for (auto [cfdfc, optimize] : funcInfo.cfdfcs) {
    if (!optimize)
      continue;
    cfdfcs.push_back(cfdfc);
    addCFDFCVars(*cfdfc);
    // Add throughput constraints on each CFDFC
    addSteadyStateReachabilityConstraints(*cfdfc);
    addChannelThroughputConstraintsForIntegerLatencyChannel(*cfdfc);
    addUnitThroughputConstraints(*cfdfc);
  }

  // Add the MILP objective and mark the MILP ready to be optimized
  addBufferAreaAwareObjective(allChannels, cfdfcs);
  markReadyToOptimize();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
