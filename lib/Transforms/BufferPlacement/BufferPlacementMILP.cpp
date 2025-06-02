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

#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Path.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;

/// Returns a textual name for a signal type.
static StringRef getSignalName(SignalType signalType) {
  switch (signalType) {
  case SignalType::DATA:
    return "data";
  case SignalType::VALID:
    return "valid";
  case SignalType::READY:
    return "ready";
  }
}

/// Returns the input and output port delays of the model for a specific signal
/// type. If the type is `SignalType::DATA`, the channel's bitwidth is used as a
/// parameter to determine the delays. If the model is nullptr, delays are
/// assumed to be 0.
static std::pair<double, double>
getPortDelays(Value channel, SignalType signalType, const TimingModel *model) {
  if (!model)
    return {0.0, 0.0};

  double inBufDelay = 0.0, outBufDelay = 0.0;
  unsigned bitwidth;
  switch (signalType) {
  case SignalType::DATA:
    bitwidth = getHandshakeTypeBitWidth(channel.getType());
    /// TODO: It's bad to discard these results, needs a safer way of querying
    /// for these delays
    (void)model->inputModel.dataDelay.getCeilMetric(bitwidth, inBufDelay);
    (void)model->outputModel.dataDelay.getCeilMetric(bitwidth, outBufDelay);
    return {inBufDelay, outBufDelay};
  case SignalType::VALID:
    return {model->inputModel.validDelay, model->outputModel.validDelay};
  case SignalType::READY:
    return {model->inputModel.readyDelay, model->outputModel.readyDelay};
  }
}

double BufferPlacementMILP::BufferingGroup::getCombinationalDelay(
    Value channel, SignalType signalType) const {
  if (!bufModel)
    return 0.0;

  unsigned bitwidth;
  double delay = 0.0;
  switch (signalType) {
  case SignalType::DATA:
    bitwidth = getHandshakeTypeBitWidth(channel.getType());
    /// TODO: It's bad to discard this result, needs a safer way of querying for
    /// this delay
    (void)bufModel->getTotalDataDelay(bitwidth, delay);
    return delay;
  case SignalType::VALID:
    return bufModel->getTotalValidDelay();
  case SignalType::READY:
    return bufModel->getTotalReadyDelay();
  }
}

BufferPlacementMILP::BufferPlacementMILP(GRBEnv &env, FuncInfo &funcInfo,
                                         const TimingDatabase &timingDB,
                                         double targetPeriod)
    : MILP<BufferPlacement>(env), timingDB(timingDB),
      targetPeriod(targetPeriod), funcInfo(funcInfo), logger(nullptr) {
  initialize();
}

BufferPlacementMILP::BufferPlacementMILP(GRBEnv &env, FuncInfo &funcInfo,
                                         const TimingDatabase &timingDB,
                                         double targetPeriod, Logger &logger,
                                         StringRef milpName)
    : MILP<BufferPlacement>(env, logger.getLogDir() +
                                     llvm::sys::path::get_separator() +
                                     milpName),
      timingDB(timingDB), targetPeriod(targetPeriod), funcInfo(funcInfo),
      logger(&logger) {
  initialize();
}

void BufferPlacementMILP::addChannelVars(Value channel,
                                         ArrayRef<SignalType> signals) {

  // Default-initialize channel variables and retrieve a reference
  ChannelVars &chVars = vars.channelVars[channel];
  std::string suffix = "_" + getUniqueName(*channel.getUses().begin());

  // Create a Gurobi variable of the given name and type for the channel
  auto createVar = [&](const llvm::Twine &name, char type) {
    return model.addVar(0, GRB_INFINITY, 0.0, type, (name + suffix).str());
  };

  // Signal-specific variables
  for (SignalType sig : signals) {
    ChannelSignalVars &signalVars = chVars.signalVars[sig];
    StringRef name = getSignalName(sig);
    signalVars.path.tIn = createVar(name + "PathIn", GRB_CONTINUOUS);
    signalVars.path.tOut = createVar(name + "PathOut", GRB_CONTINUOUS);
    signalVars.bufPresent = createVar(name + "BufPresent", GRB_BINARY);
  }

  // Variables for placement information
  chVars.bufPresent = createVar("bufPresent", GRB_BINARY);
  chVars.bufNumSlots = createVar("bufNumSlots", GRB_INTEGER);
  chVars.dataLatency = createVar("dataLatency", GRB_INTEGER);
  chVars.shiftReg = createVar("shiftReg", GRB_BINARY);

  // Update the model before returning so that these variables can be referenced
  // safely during the rest of model creation
  model.update();
}

void BufferPlacementMILP::addCFDFCVars(CFDFC &cfdfc) {
  // Create a set of variables for each CFDFC
  std::string prefix = "cfdfc" + std::to_string(vars.cfdfcVars.size()) + "_";
  CFDFCVars &cfVars = vars.cfdfcVars[&cfdfc];

  // Create a Gurobi variable of the given name (prefixed by the CFDFC index)
  auto createVar = [&](const llvm::Twine &name) {
    return model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        (prefix + name).str());
  };

  // Create a set of variables for each unit in the CFDFC
  for (Operation *unit : cfdfc.units) {
    std::string suffix = "_" + getUniqueName(unit).str();

    // Default-initialize unit variables and retrieve a reference
    UnitVars &unitVars = cfVars.unitVars[unit];
    unitVars.retIn = createVar("retIn" + suffix);

    // If the component is combinational (i.e., 0 latency) its output fluid
    // retiming equals its input fluid retiming, otherwise it is different
    double latency;
    if (failed(
            timingDB.getLatency(unit, SignalType::DATA, latency, targetPeriod)))
      latency = 0.0;
    if (latency == 0.0)
      unitVars.retOut = unitVars.retIn;
    else
      unitVars.retOut = createVar("retOut" + suffix);
  }

  // Create a variable to represent the throughput of each CFDFC channel
  for (Value channel : cfdfc.channels) {
    cfVars.channelThroughputs[channel] =
        createVar("throughput_" + getUniqueName(*channel.getUses().begin()));
  }

  // Create a variable for the CFDFC's throughput
  cfVars.throughput = createVar("throughput");

  // Update the model before returning so that these variables can be referenced
  // safely during the rest of model creation
  model.update();
}

void BufferPlacementMILP::addChannelTimingConstraints(
    Value channel, SignalType signalType, const TimingModel *bufModel,
    ArrayRef<BufferingGroup> before, ArrayRef<BufferingGroup> after) {

  ChannelVars &chVars = vars.channelVars[channel];
  double bigCst = targetPeriod * 10;

  // Sum up conditional delays of buffers before the one that cuts the path
  GRBLinExpr bufsBeforeDelay;
  for (const BufferingGroup &group : before)
    bufsBeforeDelay += chVars.signalVars[group.getRefSignal()].bufPresent *
                       group.getCombinationalDelay(channel, signalType);

  // Sum up conditional delays of buffers after the one that cuts the path
  GRBLinExpr bufsAfterDelay;
  for (const BufferingGroup &group : after)
    bufsAfterDelay += chVars.signalVars[group.getRefSignal()].bufPresent *
                      group.getCombinationalDelay(channel, signalType);

  ChannelBufProps &props = channelProps[channel];
  ChannelSignalVars &signalVars = chVars.signalVars[signalType];
  GRBVar &t1 = signalVars.path.tIn;
  GRBVar &t2 = signalVars.path.tOut;
  GRBVar &bufPresent = signalVars.bufPresent;
  auto [inBufDelay, outBufDelay] = getPortDelays(channel, signalType, bufModel);

  // Arrival time at channel's output must be lower than target clock period
  model.addConstr(t2 <= targetPeriod, "path_period");

  // If a buffer is present on the signal's path, then the arrival time at the
  // buffer's register must be lower than the clock period. The signal must
  // propagate on the channel through all potential buffers cutting other
  // signals before its own, and inside its own buffer's input pin logic
  double preBufCstDelay = props.inDelay + inBufDelay;
  model.addConstr(t1 + bufsBeforeDelay + bufPresent * preBufCstDelay <=
                      targetPeriod,
                  "path_bufferedChannelIn");

  // If a buffer is present on the signal's path, then the arrival time at the
  // channel's output must be greater than the propagation time through its own
  // buffer's output pin logic and all potential buffers cutting other signals
  // after its own
  double postBufCstDelay = outBufDelay + props.outDelay;
  model.addConstr(bufPresent * postBufCstDelay + bufsAfterDelay <= t2,
                  "path_bufferedChannelOut");

  // If there are no buffers cutting the signal's path, arrival time at
  // channel's output must still propagate through entire channel and all
  // potential buffers cutting through other signals
  GRBLinExpr unbufChannelDelay = bufsBeforeDelay + props.delay + bufsAfterDelay;
  model.addConstr(t1 + unbufChannelDelay - bigCst * bufPresent <= t2,
                  "path_unbufferedChannel");
}

void BufferPlacementMILP::addUnitTimingConstraints(Operation *unit,
                                                   SignalType signalType,
                                                   ChannelFilter filter) {
  // Add path constraints for units
  double latency;
  if (failed(timingDB.getLatency(unit, signalType, latency, targetPeriod)))
    latency = 0.0;

  if (latency == 0.0) {
    double delay;
    if (failed(timingDB.getTotalDelay(unit, signalType, delay)))
      delay = 0.0;

    // The delay of the unit must be positive.
    delay = std::max(delay, 0.001);

    // The unit is not pipelined, add a path constraint for each input/output
    // port pair in the unit
    forEachIOPair(unit, [&](Value in, Value out) {
      // The input/output channels must both be inside the CFDFC union
      if (!filter(in) || !filter(out))
        return;

      // Flip channels on ready path which goes upstream
      if (signalType == SignalType::READY)
        std::swap(in, out);

      GRBVar &tInPort = vars.channelVars[in].signalVars[signalType].path.tOut;
      GRBVar &tOutPort = vars.channelVars[out].signalVars[signalType].path.tIn;
      // Arrival time at unit's output port must be greater than arrival
      // time at unit's input port + the unit's combinational data delay
      model.addConstr(tOutPort >= tInPort + delay, "path_combDelay");
    });

    return;
  }

  // The unit is pipelined, add a constraint for every of the unit's inputs
  // and every of the unit's output ports

  // Input port constraints
  for (Value in : unit->getOperands()) {
    if (!filter(in))
      continue;

    double inPortDelay;
    if (failed(
            timingDB.getPortDelay(unit, signalType, PortType::IN, inPortDelay)))
      inPortDelay = 0.0;

    TimeVars &path = vars.channelVars[in].signalVars[signalType].path;
    GRBVar &tInPort = path.tOut;
    // Arrival time at unit's input port + input port delay must be less
    // than the target clock period
    model.addConstr(tInPort + inPortDelay <= targetPeriod, "path_inDelay");
  }

  // Output port constraints
  for (OpResult out : unit->getResults()) {
    if (!filter(out))
      continue;

    double outPortDelay;
    if (failed(timingDB.getPortDelay(unit, signalType, PortType::OUT,
                                     outPortDelay)))
      outPortDelay = 0.0;

    TimeVars &path = vars.channelVars[out].signalVars[signalType].path;
    GRBVar &tOutPort = path.tIn;
    // Arrival time at unit's output port is equal to the output port delay
    model.addConstr(tOutPort == outPortDelay, "path_outDelay");
  }
}

void BufferPlacementMILP::addBufferPresenceConstraints(Value channel) {

  ChannelVars &chVars = vars.channelVars[channel];
  GRBVar &bufPresent = chVars.bufPresent;
  GRBVar &bufNumSlots = chVars.bufNumSlots;

  // If there is at least one slot, there must be a buffer
  model.addConstr(bufNumSlots <= 100 * bufPresent, "buffer_presence");

  for (auto &[sig, signalVars] : chVars.signalVars) {
    // If there is a buffer present on a signal, then there is a buffer present
    // on the channel
    model.addConstr(signalVars.bufPresent <= bufPresent,
                    getSignalName(sig).str() + "_Presence");
  }
}

void BufferPlacementMILP::addBufferLatencyConstraints(Value channel) {

  ChannelVars &chVars = vars.channelVars[channel];
  GRBVar &bufNumSlots = chVars.bufNumSlots;
  GRBVar &dataBuf = chVars.signalVars[SignalType::DATA].bufPresent;
  GRBVar &validBuf = chVars.signalVars[SignalType::VALID].bufPresent;
  GRBVar &readyBuf = chVars.signalVars[SignalType::READY].bufPresent;
  GRBVar &dataLatency = chVars.dataLatency;

  // There is a buffer breaking data & valid iff dataLatency > 0
  model.addConstr(dataLatency <= 100 * dataBuf, "dataBuf_if_dataLatency");
  model.addConstr(dataLatency <= 100 * validBuf, "validBuf_if_dataLatency");
  model.addConstr(dataLatency >= dataBuf, "dataLatency_if_dataBuf");
  model.addConstr(dataLatency >= validBuf, "dataLatency_if_validBuf");

  // The dataBuf and validBuf must be equal
  // This constraint is not necessary, but may assist presolve.
  model.addConstr(dataBuf == validBuf, "dataBuf_validBuf_equal");
  // There must be enough slots for data and ready buffers.
  model.addConstr(dataLatency + readyBuf <= bufNumSlots, "slot_sufficiency");
}

void BufferPlacementMILP::addBufferingGroupConstraints(
    Value channel, ArrayRef<BufferingGroup> bufGroups) {

  ChannelVars &chVars = vars.channelVars[channel];
  GRBVar &bufNumSlots = chVars.bufNumSlots;

  // Compute the sum of the binary buffer presence over all signals that have
  // different buffers
  GRBLinExpr disjointBufPresentSum;
  for (const BufferingGroup &group : bufGroups) {
    GRBVar &groupBufPresent =
        chVars.signalVars[group.getRefSignal()].bufPresent;
    disjointBufPresentSum += groupBufPresent;

    // For each group, the binary buffer presence variable of different signals
    // must be equal
    StringRef refName = getSignalName(group.getRefSignal());
    for (SignalType sig : group.getOtherSignals()) {
      StringRef otherName = getSignalName(sig);
      model.addConstr(groupBufPresent == chVars.signalVars[sig].bufPresent,
                      "elastic_" + refName.str() + "_same_" + otherName.str());
    }
  }

  // There must be enough slots for all disjoint buffers
  model.addConstr(disjointBufPresentSum <= bufNumSlots, "elastic_slots");
}

void BufferPlacementMILP::addSteadyStateReachabilityConstraints(CFDFC &cfdfc) {

  CFDFCVars &cfVars = vars.cfdfcVars[&cfdfc];
  for (Value channel : cfdfc.channels) {
    // Get the ports the channels connect and their retiming MILP variables
    Operation *srcOp = channel.getDefiningOp();
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

    // Retrieve the MILP variables we need
    GRBVar &chTokenOccupancy = cfVars.channelThroughputs[channel];
    GRBVar &retSrc = cfVars.unitVars[srcOp].retOut;
    GRBVar &retDst = cfVars.unitVars[dstOp].retIn;
    unsigned backedge = cfdfc.backedges.contains(channel) ? 1 : 0;

    // If the channel isn't a backedge, its throughput equals the difference
    // between the fluid retiming of tokens at its endpoints. Otherwise, it is
    // one less than this difference
    model.addConstr(chTokenOccupancy - backedge == retDst - retSrc,
                    "throughput_channelRetiming");
  }
}

void BufferPlacementMILP::
    addChannelThroughputConstraintsForBinaryLatencyChannel(CFDFC &cfdfc) {

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
    GRBVar &dataBuf = dataVars->second.bufPresent;
    GRBVar &bufNumSlots = chVars.bufNumSlots;
    GRBVar &chTokenOccupancy = cfVars.channelThroughputs[channel];

    // The channel's throughput cannot exceed the number of buffer slots.
    model.addConstr(chTokenOccupancy <= bufNumSlots, "throughput_channel");

    // In the FPGA'20 paper:
    // - Buffers are assumed to break all signals simultaneously.
    // - Therefore: dataBuf == readyBuf
    //              R_c == dataBuf && readyBuf
    // - If R_c holds, then:
    //     - token occupancy >= CFDFC's throughput
    //     - bubble occupancy >= CFDFC's throughput
    //
    // (#427) In this implementation, R_c is decomposed into dataBuf and
    // readyBuf.
    // 1. If dataBuf holds, then token occupancy >= CFDFC's throughput;
    //    otherwise, token occupancy >= 0.
    // 2. If readyBuf holds, then bubble occupancy >= CFDFC's throughput;
    //    otherwise, bubble occupancy >= 0.

    // The following constraint encodes:
    // 1. If dataBuf holds, then token occupancy >= CFDFC's throughput;
    //    otherwise, token occupancy >= 0 (enforced by the variable’s lower
    //    bound).
    model.addConstr(cfVars.throughput - chTokenOccupancy + dataBuf <= 1,
                    "throughput_data");
    // In terms of the constraint on readyBuf:
    // 2. If readyBuf holds, then bubble occupancy >= CFDFC's throughput;
    //    otherwise, bubble occupancy >= 0.
    // This constraint can be combined with the constraint on the number of
    // buffer slots:
    // -  token occupancy + bubble occupancy <= numSlots
    // Assuming that we minimize the number of buffer slots, bubble occupancy
    // always takes the minimum feasible value. Therefore, the combined
    // constraints are equivalent to:
    // If dataBuf holds, then token occupancy + CFDFC's throughput <= numSlots;
    // otherwise, token occupancy <= numSlots. (Already enforced by the earlier
    // constraint named "throughput_channel")
    // The following constraint encodes the case where readyBuf holds, and is
    // trivially satisfied when readyBuf does not hold (since the earlier
    // constraint already enforces it):
    if (chVars.signalVars.count(SignalType::READY)) {
      auto readyBuf = chVars.signalVars[SignalType::READY].bufPresent;
      model.addConstr(
          chTokenOccupancy + cfVars.throughput + readyBuf - bufNumSlots <= 1,
          "throughput_ready");
    }
    // Note: Additional buffers may be needed to prevent combinational cycles
    // if the model does not select all three signals (or only selects DATA).
    // See extractResult() in FPGA20Buffers.cpp for an example.
  }
}

void BufferPlacementMILP::
    addChannelThroughputConstraintsForIntegerLatencyChannel(CFDFC &cfdfc) {

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

    // The channel must have variables for the data and ready signals
    ChannelVars &chVars = vars.channelVars[channel];
    auto dataVars = chVars.signalVars.find(SignalType::DATA);
    auto readyVars = chVars.signalVars.find(SignalType::READY);
    bool dataFound = dataVars != chVars.signalVars.end();
    bool readyFound = readyVars != chVars.signalVars.end();
    assert(dataFound && "missing data signal variables on channel variables");
    assert(readyFound && "missing ready signal variables on channel variables");

    // Retrieve the MILP variables we need
    GRBVar &bufNumSlots = chVars.bufNumSlots;
    GRBVar &chTokenOccupancy = cfVars.channelThroughputs[channel];
    GRBVar &throughput = cfVars.throughput;
    GRBVar &dataLatency = chVars.dataLatency;
    GRBVar &readyBuf = chVars.signalVars[SignalType::READY].bufPresent;
    GRBVar &shiftReg = chVars.shiftReg;

    // Token occupancy >= data latency * CFDFC's throughput.
    model.addQConstr(dataLatency * throughput <= chTokenOccupancy,
                     "throughput_tokens_lb");
    std::string channelName = getUniqueName(*channel.getUses().begin());
    std::string shiftRegExtraBubblesName = "shiftReg_ub_" + channelName;
    // Shift registers have more bubbles if II is higher than 1 (i.e.,
    // throughput < 1). In a shift register, every slot forwards data
    // simultaneously, but new tokens only arrive every II cycles. This means
    // that among every II consecutive slots, only one contains a token while
    // the rest are bubbles. Therefore, token occupancy is lower compared to
    // other buffer types with the same slot number.
    //
    // Create an intermediate variable to represent the extra bubbles
    // of the SHIFT_REG_BREAK_DV buffer.
    GRBVar shiftRegExtraBubbles = model.addVar(
        0, GRB_INFINITY, 0.0, GRB_INTEGER, shiftRegExtraBubblesName);

    // The extra bubbles of SHIFT_REG_BREAK_DV buffer is at least its slot
    // number (dataLatency) minus the ceiling of the product of data latency and
    // CFDFC throughput.
    // We approximate the ceiling function numerically to keep the model linear.
    model.addQConstr(shiftRegExtraBubbles >=
                         dataLatency - dataLatency * throughput - 0.99,
                     shiftRegExtraBubblesName);

    // Combine the following into a unified constraint:
    // 1. If readyBuf is used, bubble occupancy limits the CFDFC's throughput,
    //    i.e., bubble occupancy >= CFDFC's throughput;
    //    otherwise, bubble occupancy >= 0.
    // 2. Token occupancy + bubble occupancy <= slot number.
    // 3. Extra bubbles if SHIFT_REG_BREAK_DV is used.
    // Since there is no others reason to have more bubbles, the optimal
    // (minimum value) of the bubble due to readyBuf is the same as CFDFC's
    // throughput. Therefore, constraint 1 becomes:
    // 1. If readyBuf is used, bubble occupancy = CFDFC's throughput;
    //    otherwise, bubble occupancy = 0.
    // As a result, we model bubble occupancy as 'readyBuf * throughput'. This
    // term can be linearized, but it is not necessary because this is a
    // quadratic constaint.
    model.addQConstr(chTokenOccupancy + readyBuf * throughput +
                             shiftReg * shiftRegExtraBubbles <=
                         bufNumSlots,
                     "throughput_tokens_ub");
  }
}

void BufferPlacementMILP::addUnitThroughputConstraints(CFDFC &cfdfc) {
  CFDFCVars &cfVars = vars.cfdfcVars[&cfdfc];
  for (Operation *unit : cfdfc.units) {
    double latency;
    if (failed(timingDB.getLatency(unit, SignalType::DATA, latency,
                                   targetPeriod)) ||
        latency == 0.0)
      continue;

    // Retrieve the MILP variables corresponding to the unit's fluid retiming
    UnitVars &unitVars = cfVars.unitVars[unit];
    GRBVar &retIn = unitVars.retIn;
    GRBVar &retOut = unitVars.retOut;

    // The fluid retiming of tokens across the non-combinational unit must
    // be the same as its latency multiplied by the CFDFC's throughput
    model.addConstr(cfVars.throughput * latency == retOut - retIn,
                    "through_unitRetiming");
  }
}

unsigned BufferPlacementMILP::getChannelNumExecs(Value channel) {
  Operation *srcOp = channel.getDefiningOp();
  if (!srcOp)
    // A channel which originates from a function argument executes only once
    return 1;

  // Iterate over all CFDFCs which contain the channel to determine its total
  // number of executions. Backedges are executed one less time than "forward
  // edges" since they are only taken between executions of the cycle the CFDFC
  // represents
  unsigned numExec = isBackedge(channel) ? 0 : 1;
  for (auto &[cfdfc, _] : funcInfo.cfdfcs)
    if (cfdfc->channels.contains(channel))
      numExec += cfdfc->numExecs;
  return numExec;
}

void BufferPlacementMILP::addMaxThroughputObjective(ValueRange channels,
                                                    ArrayRef<CFDFC *> cfdfcs) {
  // Compute the total number of executions over channels that are part of any
  // CFDFC
  unsigned totalExecs = 0;
  for (Value channel : channels) {
    totalExecs += getChannelNumExecs(channel);
  }

  // Create the expression for the MILP objective
  GRBLinExpr objective;

  // For each CFDFC, add a throughput contribution to the objective, weighted
  // by the "importance" of the CFDFC
  double maxCoefCFDFC = 0.0;
  double fTotalExecs = static_cast<double>(totalExecs);
  if (totalExecs != 0) {
    for (CFDFC *cfdfc : cfdfcs) {
      double coef = (cfdfc->channels.size() * cfdfc->numExecs) / fTotalExecs;
      objective += coef * vars.cfdfcVars[cfdfc].throughput;
      maxCoefCFDFC = std::max(coef, maxCoefCFDFC);
    }
  }

  // In case we ran the MILP without providing any CFDFC, set the maximum CFDFC
  // coefficient to any positive value
  if (maxCoefCFDFC == 0.0)
    maxCoefCFDFC = 1.0;

  // For each channel, add a "penalty" in case a buffer is added to the channel,
  // and another penalty that depends on the number of slots
  double bufPenaltyMul = 1e-4;
  double slotPenaltyMul = 1e-5;
  for (Value channel : channels) {
    ChannelVars &chVars = vars.channelVars[channel];
    objective -= maxCoefCFDFC * bufPenaltyMul * chVars.bufPresent;
    objective -= maxCoefCFDFC * slotPenaltyMul * chVars.bufNumSlots;
  }

  // Finally, set the MILP objective
  model.setObjective(objective, GRB_MAXIMIZE);
}

void BufferPlacementMILP::addBufferAreaAwareObjective(
    ValueRange channels, ArrayRef<CFDFC *> cfdfcs) {
  // Compute the total number of executions over channels that are part of any
  // CFDFC
  unsigned totalExecs = 0;
  for (Value channel : channels) {
    totalExecs += getChannelNumExecs(channel);
  }

  // Create the expression for the MILP objective
  GRBLinExpr objective = 0;

  // For each CFDFC, add a throughput contribution to the objective, weighted
  // by the "importance" of the CFDFC
  double maxCoefCFDFC = 0.0;
  double fTotalExecs = static_cast<double>(totalExecs);
  if (totalExecs != 0) {
    for (CFDFC *cfdfc : cfdfcs) {
      double coef = (cfdfc->channels.size() * cfdfc->numExecs) / fTotalExecs;
      objective += coef * vars.cfdfcVars[cfdfc].throughput;
      maxCoefCFDFC = std::max(coef, maxCoefCFDFC);
    }
  }

  // In case we ran the MILP without providing any CFDFC, set the maximum CFDFC
  // coefficient to any positive value
  if (maxCoefCFDFC == 0.0)
    maxCoefCFDFC = 1.0;

  // The following parameters control penalties in the MILP objective. The
  // penalty for buffer presence is an empirical value, consistent with
  // 'addMaxThroughputObjective'. The slot penalties for each buffer type are
  // rough estimates, based on the number of LUTs as logic observed when each
  // buffer type was synthesized individually. To adjust these parameters during
  // tuning, simply modify the values here.

  // For each channel, add a "penalty" in case a buffer is added to the channel,
  // and another penalty that depends on the number of slots
  double bufPenaltyMul = 1e-4;
  // In general, buffers that break data paths have a lower area cost per slot,
  // while other types incur a higher cost
  double largeSlotPenaltyMul = 1e-4;
  double smallSlotPenaltyMul = 1e-5;
  // For SHIFT_REG_BREAK_DV, a small area cost is incurred when the buffer
  // exists Increasing the slot number only requires additional registers, not
  // LUTs We assign a minimal cost only to constrain its slot number
  double shiftRegPenaltyMul = 1e-5;
  double shiftRegSlotPenaltyMul = 1e-7;
  for (Value channel : channels) {
    ChannelVars &chVars = vars.channelVars[channel];
    GRBVar &bufPresent = chVars.bufPresent;
    GRBVar &bufNumSlots = chVars.bufNumSlots;
    GRBVar &dataLatency = chVars.dataLatency;
    GRBVar &shiftReg = chVars.shiftReg;
    objective -= maxCoefCFDFC * bufPenaltyMul * bufPresent;
    objective -=
        maxCoefCFDFC * largeSlotPenaltyMul * (bufNumSlots - dataLatency);
    objective -= maxCoefCFDFC * shiftRegPenaltyMul * shiftReg;

    // Linearization of dataLatency * shiftReg
    GRBVar latencyMulShiftReg =
        model.addVar(0, 100, 0.0, GRB_INTEGER, "latencyMulShiftReg");
    model.addConstr(latencyMulShiftReg <= dataLatency);
    model.addConstr(latencyMulShiftReg <= 100 * shiftReg);
    model.addConstr(latencyMulShiftReg >= dataLatency - (1 - shiftReg) * 100);
    objective -=
        maxCoefCFDFC * smallSlotPenaltyMul * (dataLatency - latencyMulShiftReg);
    objective -= maxCoefCFDFC * shiftRegSlotPenaltyMul * latencyMulShiftReg;
  }

  // Finally, set the MILP objective
  model.setObjective(objective, GRB_MAXIMIZE);
}

void BufferPlacementMILP::forEachIOPair(
    Operation *op, const std::function<void(Value, Value)> &callback) {
  for (Value opr : op->getOperands()) {
    if (!isa<MemRefType>(opr.getType())) {
      for (OpResult res : op->getResults()) {
        if (!isa<MemRefType>(res.getType()))
          callback(opr, res);
      }
    }
  }
}

void BufferPlacementMILP::logResults(BufferPlacement &placement) {
  assert(logger && "no logger was provided");
  mlir::raw_indented_ostream &os = **logger;

  os << "# ========================== #\n";
  os << "# Buffer Placement Decisions #\n";
  os << "# ========================== #\n\n";

  for (auto &[value, chVars] : vars.channelVars) {
    if (chVars.bufPresent.get(GRB_DoubleAttr_X) == 0)
      continue;

    // Extract number and type of slots
    unsigned numSlotsToPlace =
        static_cast<unsigned>(chVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);

    PlacementResult result = placement[value];
    ChannelBufProps &props = channelProps[value];

    // Log placement decision
    os << getUniqueName(*value.getUses().begin()) << ":\n";
    os.indent();
    std::stringstream propsStr;
    propsStr << props;
    os << "- Buffering constraints: " << propsStr.str() << "\n";
    os << "- MILP decision: " << numSlotsToPlace << " slot(s)\n";
    os << "- Placement decision: \n";
    os << result.numOneSlotDV << " OneSlotDV slot(s)\n";
    os << result.numOneSlotR << " OneSlotR slot(s)\n";
    os << result.numFifoDV << " FifoDV slot(s)\n";
    os << result.numFifoNone << " FifoNone slot(s)\n";
    os << result.numOneSlotDVR << " OneSlotDVR slot(s)\n";
    os << result.numShiftRegDV << " ShiftRegDV slot(s)\n";
    os.unindent();
    os << "\n";
  }

  os << "# ================= #\n";
  os << "# CFDFC Throughputs #\n";
  os << "# ================= #\n\n";

  // Log global CFDFC throuhgputs
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfdfcVars)) {
    auto [cf, cfVars] = cfdfcWithVars;
    double throughput = cfVars.throughput.get(GRB_DoubleAttr_X);
    os << "Throughput of CFDFC #" << idx << ": " << throughput << "\n";
  }

  os << "\n# =================== #\n";
  os << "# Channel Throughputs #\n";
  os << "# =================== #\n\n";

  // Log throughput of all channels in all CFDFCs
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfdfcVars)) {
    auto [cf, cfVars] = cfdfcWithVars;
    os << "Per-channel throughputs of CFDFC #" << idx << ":\n";
    os.indent();
    for (auto [val, channelTh] : cfVars.channelThroughputs) {
      os << getUniqueName(*val.getUses().begin()) << ": "
         << channelTh.get(GRB_DoubleAttr_X) << "\n";
    }
    os.unindent();
    os << "\n";
  }
}

void BufferPlacementMILP::initialize() {
  unsatisfiable =
      failed(mapChannelsToProperties(funcInfo.funcOp, timingDB, channelProps));

  // Initialize the large constant (for elasticity constraints)
  auto ops = funcInfo.funcOp.getOps();
  largeCst = std::distance(ops.begin(), ops.end()) + 2;
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
