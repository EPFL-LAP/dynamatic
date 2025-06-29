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

void CostAwareBuffers::addThroughputConstraintsWithBestThroughput(
    CFDFC &cfdfc, double bestThroughput) {

  CFDFCVars &cfVars = vars.cfdfcVars[&cfdfc];
  for (Value channel : cfdfc.channels) {
    // Get the ports the channels connect and their retiming MILP variables
    Operation *dstOp = *channel.getUsers().begin();

    // No throughput constraints on channels going to stores
    /// TODO: this is from legacy implementation, we should understand why we
    /// really do this and figure out if it makes sense (@lucas-rami: I don't
    /// think it does)
    if (isa<handshake::StoreOp>(dstOp))
      continue;

    /// TODO: The legacy implementation does not add any constraints here for
    /// the input channel to select operations that is less frequently
    /// executed. Temporarily, emulate the same behavior obtained from passing
    /// our DOTs to the old buffer pass by assuming the "true" input is always
    /// the least executed one
    if (auto selOp = dyn_cast<handshake::SelectOp>(dstOp))
      if (channel == selOp.getTrueValue())
        continue;

    // The channel must have variables for the data signal
    ChannelVars &chVars = vars.channelVars[channel];
    auto dataVars = chVars.signalVars.find(SignalType::DATA);
    bool dataFound = dataVars != chVars.signalVars.end();
    assert(dataFound && "missing data signal variables on channel variables");

    // Retrieve the MILP variables we need
    GRBVar &bufNumSlots = chVars.bufNumSlots;
    GRBVar &chThroughput = cfVars.channelThroughputs[channel];
    GRBVar &dataLatency = chVars.dataLatency;
    GRBVar &readyBuf = chVars.signalVars[SignalType::READY].bufPresent;
    GRBVar &shiftReg = chVars.shiftReg;

    std::string channelName = getUniqueName(*channel.getUses().begin());
    std::string shiftRegUbName = "shiftReg_ub_" + channelName;
    GRBVar shiftRegUb =
        model.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER, shiftRegUbName);
    model.addConstr(shiftRegUb <= dataLatency * bestThroughput + 0.99,
                    shiftRegUbName);
    model.addConstr(dataLatency * bestThroughput <= chThroughput,
                    "throughput_tokens_lb");
    GRBVar shiftRegLatency =
        model.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER, "shiftRegLatency");
    model.addConstr(shiftRegLatency <= dataLatency - shiftRegUb, "temp1");
    model.addConstr(shiftRegLatency <= 100 * shiftReg, "temp2");
    model.addConstr(shiftRegLatency >=
                        dataLatency - shiftRegUb - 100 * (1 - shiftReg),
                    "temp3");
    model.addConstr(chThroughput + readyBuf * bestThroughput +
                            shiftRegLatency <=
                        bufNumSlots,
                    "throughput_tokens_ub");
  }

  for (Operation *unit : cfdfc.units) {
    double latency;
    if (failed(timingDB.getLatency(unit, SignalType::DATA, latency)) ||
        latency == 0.0)
      continue;

    // Retrieve the MILP variables corresponding to the unit's fluid retiming
    UnitVars &unitVars = cfVars.unitVars[unit];
    GRBVar &retIn = unitVars.retIn;
    GRBVar &retOut = unitVars.retOut;

    // The fluid retiming of tokens across the non-combinational unit must
    // be the same as its latency multiplied by the CFDFC's throughput
    model.addConstr(bestThroughput * latency == retOut - retIn,
                    "through_unitRetiming");
  }

  model.addConstr(cfVars.throughput == bestThroughput, "throughput_constraint");
}

void CostAwareBuffers::setup() {

  if (linearize == false) {
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
        // addChannelTimingConstraints(channel, SignalType::VALID, bufModel);
        // addChannelTimingConstraints(channel, SignalType::READY, bufModel);
        addBufferPresenceConstraints(channel);
        addBufferLatencyConstraints(channel);
      }
    }

    // Add timing constraints for all units in the function
    for (Operation &op : funcInfo.funcOp.getOps()) {
      addUnitTimingConstraints(&op, SignalType::DATA);
      // addUnitTimingConstraints(&op, SignalType::VALID);
      // addUnitTimingConstraints(&op, SignalType::READY);
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

  } else {
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
        // addChannelTimingConstraints(channel, SignalType::VALID, bufModel);
        // addChannelTimingConstraints(channel, SignalType::READY, bufModel);
        addBufferPresenceConstraints(channel);
        // addBufferLatencyConstraints(channel);
      }
    }

    // Add timing constraints for all units in the function
    for (Operation &op : funcInfo.funcOp.getOps()) {
      addUnitTimingConstraints(&op, SignalType::DATA);
      // addUnitTimingConstraints(&op, SignalType::VALID);
      // addUnitTimingConstraints(&op, SignalType::READY);
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
      addChannelThroughputConstraintsForBinaryLatencyChannel(*cfdfc);
      addUnitThroughputConstraints(*cfdfc);
    }

    // Add the MILP objective and mark the MILP ready to be optimized
    addMaxThroughputObjective(allChannels, cfdfcs);
    markReadyToOptimize();

    int phase1Status;
    if (failed(optimize(&phase1Status))) {
      llvm::errs() << "Phase 1 optimization failed\n";
      return;
    }

    std::map<size_t, double> cfdfcThroughputs;
    for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfdfcVars)) {
      auto [cf, cfVars] = cfdfcWithVars;
      double throughput = cfVars.throughput.get(GRB_DoubleAttr_X);
      cfdfcThroughputs[idx] = throughput;
    }

    model.reset();
    resetMILPState();

    for (auto &[channel, _] : channelProps) {
      addChannelVars(channel, signals);
      addCustomChannelConstraints(channel);

      if (!channel.getDefiningOp<handshake::MemoryOpInterface>() &&
          !isa<handshake::MemoryOpInterface>(*channel.getUsers().begin())) {
        addChannelTimingConstraints(channel, SignalType::DATA, bufModel);
        // addChannelTimingConstraints(channel, SignalType::VALID, bufModel);
        addChannelTimingConstraints(channel, SignalType::READY, bufModel);
        addBufferPresenceConstraints(channel);
        addBufferLatencyConstraints(channel);
      }
    }

    for (Operation &op : funcInfo.funcOp.getOps()) {
      addUnitTimingConstraints(&op, SignalType::DATA);
      // addUnitTimingConstraints(&op, SignalType::VALID);
      addUnitTimingConstraints(&op, SignalType::READY);
    }

    cfdfcs.clear();
    size_t idx = 0;
    for (auto [cfdfc, optimize] : funcInfo.cfdfcs) {
      if (!optimize) {
        idx++;
        continue;
      }
      cfdfcs.push_back(cfdfc);
      addCFDFCVars(*cfdfc);
      addSteadyStateReachabilityConstraints(*cfdfc);
      addThroughputConstraintsWithBestThroughput(*cfdfc, cfdfcThroughputs[idx]);
      addUnitThroughputConstraints(*cfdfc);
      idx++;
    }

    addBufferAreaAwareObjective(allChannels, cfdfcs);
    markReadyToOptimize();
  }
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
