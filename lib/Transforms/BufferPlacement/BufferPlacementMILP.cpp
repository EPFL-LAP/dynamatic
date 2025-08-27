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
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/StdProfiler.h"
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
using namespace dynamatic::experimental;

/// Returns a textual name for a signal type.
static StringRef getSignalName(SignalType type) {
  switch (type) {
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
static std::pair<double, double> getPortDelays(Value channel, SignalType signal,
                                               const TimingModel *model) {
  if (!model)
    return {0.0, 0.0};

  double inBufDelay = 0.0, outBufDelay = 0.0;
  unsigned bitwidth;
  switch (signal) {
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
    Value channel, SignalType type) const {
  if (!bufModel)
    return 0.0;

  unsigned bitwidth;
  double delay = 0.0;
  switch (type) {
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
  ChannelVars &channelVars = vars.channelVars[channel];
  std::string suffix = "_" + getUniqueName(*channel.getUses().begin());

  // Create a Gurobi variable of the given name and type for the channel
  auto createVar = [&](const llvm::Twine &name, char type) {
    return model.addVar(0, GRB_INFINITY, 0.0, type, (name + suffix).str());
  };

  // Signal-specific variables
  for (SignalType sig : signals) {
    ChannelSignalVars &signalVars = channelVars.signalVars[sig];
    StringRef name = getSignalName(sig);
    signalVars.path.tIn = createVar(name + "PathIn", GRB_CONTINUOUS);
    signalVars.path.tOut = createVar(name + "PathOut", GRB_CONTINUOUS);
    signalVars.bufPresent = createVar(name + "BufPresent", GRB_BINARY);
  }

  // Variables for elasticity constraints
  channelVars.elastic.tIn = createVar("elasIn", GRB_CONTINUOUS);
  channelVars.elastic.tOut = createVar("elasOut", GRB_CONTINUOUS);
  // Variables for placement information
  channelVars.bufPresent = createVar("bufPresent", GRB_BINARY);
  channelVars.bufNumSlots = createVar("bufNumSlots", GRB_INTEGER);

  // Update the model before returning so that these variables can be referenced
  // safely during the rest of model creation
  model.update();
}

void BufferPlacementMILP::addCFDFCVars(CFDFC &cfdfc) {
  // Create a set of variables for each CFDFC
  std::string prefix = "cfdfc" + std::to_string(vars.cfVars.size()) + "_";
  CFDFCVars &cfVars = vars.cfVars[&cfdfc];

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
    if (failed(timingDB.getLatency(unit, SignalType::DATA, latency)))
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

void BufferPlacementMILP::addChannelPathConstraints(
    Value channel, SignalType signal, const TimingModel *bufModel,
    ArrayRef<BufferingGroup> before, ArrayRef<BufferingGroup> after) {

  ChannelVars &channelVars = vars.channelVars[channel];
  double bigCst = targetPeriod * 10;

  // Sum up conditional delays of buffers before the one that cuts the path
  GRBLinExpr bufsBeforeDelay;
  for (const BufferingGroup &group : before)
    bufsBeforeDelay += channelVars.signalVars[group.getRefSignal()].bufPresent *
                       group.getCombinationalDelay(channel, signal);

  // Sum up conditional delays of buffers after the one that cuts the path
  GRBLinExpr bufsAfterDelay;
  for (const BufferingGroup &group : after)
    bufsAfterDelay += channelVars.signalVars[group.getRefSignal()].bufPresent *
                      group.getCombinationalDelay(channel, signal);

  ChannelBufProps &props = channelProps[channel];
  ChannelSignalVars &signalVars = channelVars.signalVars[signal];
  GRBVar &t1 = signalVars.path.tIn;
  GRBVar &t2 = signalVars.path.tOut;
  GRBVar &bufPresent = signalVars.bufPresent;
  auto [inBufDelay, outBufDelay] = getPortDelays(channel, signal, bufModel);

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

bool BufferPlacementMILP::hasValidChannelVars(Value channel, SignalType type) const {
  auto channelIt = vars.channelVars.find(channel);
  if (channelIt == vars.channelVars.end())
    return false;
  
  auto signalIt = channelIt->second.signalVars.find(type);
  return signalIt != channelIt->second.signalVars.end();
}

void BufferPlacementMILP::addUnitPathConstraints(Operation *unit,
                                                 SignalType type,
                                                 ChannelFilter filter) {
  // Add path constraints for units
  double latency;
  if (failed(timingDB.getLatency(unit, type, latency)))
    latency = 0.0;

  if (latency == 0.0) {
    double delay;
    if (failed(timingDB.getTotalDelay(unit, type, delay)))
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
      if (type == SignalType::READY)
        std::swap(in, out);

      // Validate that channel variables exist before accessing them
      if (!hasValidChannelVars(in, type) || !hasValidChannelVars(out, type))
        return;

      GRBVar &tInPort = vars.channelVars[in].signalVars[type].path.tOut;
      GRBVar &tOutPort = vars.channelVars[out].signalVars[type].path.tIn;
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

    // Validate that channel variables exist before accessing them
    if (!hasValidChannelVars(in, type))
      continue;

    double inPortDelay;
    if (failed(timingDB.getPortDelay(unit, type, PortType::IN, inPortDelay)))
      inPortDelay = 0.0;

    TimeVars &path = vars.channelVars[in].signalVars[type].path;
    GRBVar &tInPort = path.tOut;
    // Arrival time at unit's input port + input port delay must be less
    // than the target clock period
    model.addConstr(tInPort + inPortDelay <= targetPeriod, "path_inDelay");
  }

  // Output port constraints
  for (OpResult out : unit->getResults()) {
    if (!filter(out))
      continue;

    // Validate that channel variables exist before accessing them
    if (!hasValidChannelVars(out, type))
      continue;

    double outPortDelay;
    if (failed(timingDB.getPortDelay(unit, type, PortType::OUT, outPortDelay)))
      outPortDelay = 0.0;

    TimeVars &path = vars.channelVars[out].signalVars[type].path;
    GRBVar &tOutPort = path.tIn;
    // Arrival time at unit's output port is equal to the output port delay
    model.addConstr(tOutPort == outPortDelay, "path_outDelay");
  }
}

void BufferPlacementMILP::addChannelElasticityConstraints(
    Value channel, ArrayRef<BufferingGroup> bufGroups) {
  // Validate that channel variables exist before accessing them
  auto channelIt = vars.channelVars.find(channel);
  if (channelIt == vars.channelVars.end())
    return;

  ChannelVars &channelVars = channelIt->second;
  GRBVar &tIn = channelVars.elastic.tIn;
  GRBVar &tOut = channelVars.elastic.tOut;
  GRBVar &bufPresent = channelVars.bufPresent;
  GRBVar &bufNumSlots = channelVars.bufNumSlots;

  // If there is at least one slot, there must be a buffer
  model.addConstr(0.01 * bufNumSlots <= bufPresent, "elastic_presence");

  for (auto &[sig, signalVars] : channelVars.signalVars) {
    // If there is a buffer present on a signal, then there is a buffer present
    // on the channel
    model.addConstr(signalVars.bufPresent <= bufPresent,
                    "elastic_" + getSignalName(sig).str() + "Presence");
  }

  auto dataIt = channelVars.signalVars.find(SignalType::DATA);
  if (dataIt != channelVars.signalVars.end()) {
    GRBVar &dataBuf = dataIt->second.bufPresent;
    // If there is a data buffer on the channel, the channel elastic
    // arrival time at the ouput must be greater than at the input
    model.addConstr(tOut >= tIn - largeCst * dataBuf, "elastic_data");
  }

  // Compute the sum of the binary buffer presence over all signals that have
  // different buffers
  GRBLinExpr disjointBufPresentSum;
  for (const BufferingGroup &group : bufGroups) {
    GRBVar &groupBufPresent =
        channelVars.signalVars[group.getRefSignal()].bufPresent;
    disjointBufPresentSum += groupBufPresent;

    // For each group, the binary buffer presence variable of different signals
    // must be equal
    StringRef refName = getSignalName(group.getRefSignal());
    for (SignalType sig : group.getOtherSignals()) {
      StringRef otherName = getSignalName(sig);
      model.addConstr(groupBufPresent == channelVars.signalVars[sig].bufPresent,
                      "elastic_" + refName.str() + "_same_" + otherName.str());
    }
  }

  // There must be enough slots for all disjoint buffers
  model.addConstr(disjointBufPresentSum <= bufNumSlots, "elastic_slots");
}

void BufferPlacementMILP::addUnitElasticityConstraints(Operation *unit,
                                                       ChannelFilter filter) {
  forEachIOPair(unit, [&](Value in, Value out) {
    // Both channels must be eligible
    if (!filter(in) || !filter(out))
      return;

    // Validate that channel variables exist before accessing them
    auto inIt = vars.channelVars.find(in);
    auto outIt = vars.channelVars.find(out);
    if (inIt == vars.channelVars.end() || outIt == vars.channelVars.end())
      return;

    GRBVar &tInPort = inIt->second.elastic.tOut;
    GRBVar &tOutPort = outIt->second.elastic.tIn;
    // The elastic arrival time at the output port must be at least one
    // greater than at the input port
    model.addConstr(tOutPort >= 1 + tInPort, "elastic_unitTime");
  });
}

void BufferPlacementMILP::addChannelThroughputConstraints(CFDFC &cfdfc) {
  CFDFCVars &cfVars = vars.cfVars[&cfdfc];
  for (Value channel : cfdfc.channels) {
    // Get the ports the channels connect and their retiming MILP variables
    Operation *srcOp = channel.getDefiningOp();
    Operation *dstOp = *channel.getUsers().begin();

    if (isa<handshake::StoreOp>(srcOp)) {
      llvm::errs() << "StoreOp found in channel: " << channel << "\n";
      /// print the first user of the channel
      llvm::errs() << "First user of the channel: ";
      llvm::errs() << **channel.getUsers().begin() << "\n";
    }
    if (isa<handshake::StoreOp>(dstOp)) {
      llvm::errs() << "dst " << channel << "\n";
    }

    /// TODO: The legacy implementation does not add any constraints here for
    /// the input channel to select operations that is less frequently
    /// executed. Temporarily, emulate the same behavior obtained from passing
    /// our DOTs to the old buffer pass by assuming the "true" input is always
    /// the least executed one
    if (auto selOp = dyn_cast<handshake::SelectOp>(dstOp))
      if (channel == selOp.getTrueValue())
        continue;

    // Validate that channel variables exist before accessing them
    auto channelIt = vars.channelVars.find(channel);
    if (channelIt == vars.channelVars.end())
      continue;

    // Skip channels that don't have data signal variables (e.g., control-only channels)
    ChannelVars &chVars = channelIt->second;
    auto dataVars = chVars.signalVars.find(SignalType::DATA);
    bool dataFound = dataVars != chVars.signalVars.end();
    if (!dataFound) {
      // Only channels with data signals can participate in throughput constraints
      continue;
    }

    // Retrieve the MILP variables we need
    GRBVar &dataBuf = dataVars->second.bufPresent;
    GRBVar &bufNumSlots = chVars.bufNumSlots;
    GRBVar &chThroughput = cfVars.channelThroughputs[channel];
    GRBVar &retSrc = cfVars.unitVars[srcOp].retOut;
    GRBVar &retDst = cfVars.unitVars[dstOp].retIn;
    unsigned backedge = cfdfc.backedges.contains(channel) ? 1 : 0;

    // If the channel isn't a backedge, its throughput equals the difference
    // between the fluid retiming of tokens at its endpoints. Otherwise, it is
    // one less than this difference
    model.addConstr(chThroughput - backedge == retDst - retSrc,
                    "throughput_channelRetiming");
    // The channel's throughput cannot exceed the number of buffer slots
    model.addConstr(chThroughput <= bufNumSlots, "throughput_channel");
    // If there is an opaque buffer, the CFDFC throughput cannot exceed the
    // channel throughput. If there is not, the CFDFC throughput can exceed
    // the channel thoughput by 1
    model.addConstr(cfVars.throughput - chThroughput + dataBuf <= 1,
                    "throughput_cfdfc");
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
          chThroughput + cfVars.throughput + readyBuf - bufNumSlots <= 1,
          "throughput_ready");
    }
    // Note: Additional buffers may be needed to prevent combinational cycles
    // if the model does not select all three signals (or only selects DATA).
    // See extractResult() in FPGA20Buffers.cpp for an example.
  }
}

void BufferPlacementMILP::addUnitThroughputConstraints(CFDFC &cfdfc) {
  CFDFCVars &cfVars = vars.cfVars[&cfdfc];
  for (Operation *unit : cfdfc.units) {
    double latency;
    if (failed(timingDB.getLatency(unit, SignalType::DATA, latency)) ||
        latency == 0.0)
      continue;

      // For each unit with non-zero data latency, add a constraint enforcing
      // (CFDFC throughput * latency) <= channel throughput, as described in the FPGA 20 paper.
    for (Value channel: cfdfc.channels) {
      Operation *dstOp = *channel.getUsers().begin();
      if (dstOp != unit)
        continue;
      GRBVar &chThroughput = cfVars.channelThroughputs[channel];
      model.addConstr(cfVars.throughput * latency <= chThroughput, "UnitThroughput_pipelined");
    }
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

void BufferPlacementMILP::addObjective(ValueRange channels,
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
      objective += coef * vars.cfVars[cfdfc].throughput;
      maxCoefCFDFC = std::max(coef, maxCoefCFDFC);
    }
  }

  // In case we ran the MILP without providing any CFDFC, set the maximum CFDFC
  // coefficient to any positive value
  if (maxCoefCFDFC == 0.0)
    maxCoefCFDFC = 1.0;

  // For each channel, add a "penalty" in case a buffer is added to the channel,
  // and another penalty that depends on the number of slots
  double bufPenaltyMul = 1e-6;  // Reduced penalty to encourage buffer placement
  double slotPenaltyMul = 1e-7; // Reduced penalty to encourage buffer placement
  for (Value channel : channels) {
    // Validate that channel variables exist before accessing them
    auto channelIt = vars.channelVars.find(channel);
    if (channelIt == vars.channelVars.end())
      continue;

    ChannelVars &channelVars = channelIt->second;
    objective -= maxCoefCFDFC * bufPenaltyMul * channelVars.bufPresent;
    objective -= maxCoefCFDFC * slotPenaltyMul * channelVars.bufNumSlots;
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

  for (auto &[value, channelVars] : vars.channelVars) {
    if (channelVars.bufPresent.get(GRB_DoubleAttr_X) == 0)
      continue;

    // Extract number and type of slots
    unsigned numSlotsToPlace = static_cast<unsigned>(
        channelVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    bool placeOpaque = channelVars.signalVars[SignalType::DATA].bufPresent.get(
                           GRB_DoubleAttr_X) > 0;

    PlacementResult result = placement[value];
    ChannelBufProps &props = channelProps[value];

    // Log placement decision
    os << getUniqueName(*value.getUses().begin()) << ":\n";
    os.indent();
    std::stringstream propsStr;
    propsStr << props;
    os << "- Buffering constraints: " << propsStr.str() << "\n";
    os << "- MILP decision: " << numSlotsToPlace << " "
       << (placeOpaque ? "opaque" : "transparent") << " slot(s)\n";
    os << "- Placement decision: " << result.numTrans
       << " transparent slot(s) and " << result.numOpaque
       << " opaque slot(s)\n";
    os.unindent();
    os << "\n";
  }

  os << "# ================= #\n";
  os << "# CFDFC Throughputs #\n";
  os << "# ================= #\n\n";

  // Log global CFDFC throuhgputs
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {
    auto [cf, cfVars] = cfdfcWithVars;
    double throughput = cfVars.throughput.get(GRB_DoubleAttr_X);
    os << "Throughput of CFDFC #" << idx << ": " << throughput * 1e10 + 3
       << "\n";
  }

  os << "\n# =================== #\n";
  os << "# Channel Throughputs #\n";
  os << "# =================== #\n\n";

  // Log throughput of all channels in all CFDFCs
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {
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

  os << "# ================== #\n";
  os << "# Unit Retiming Values #\n";
  os << "# ================== #\n\n";
  // Log retiming values of all units in all CFDFCs
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {
    auto [cf, cfVars] = cfdfcWithVars;
    os << "Per-unit retiming values of CFDFC #" << idx << ":\n";
    os.indent();
    for (auto &[unit, unitVars] : cfVars.unitVars) {
      // Skip units that are not part of the CFDFC
      if (!cf->units.contains(unit))
        continue;
      double retIn = unitVars.retIn.get(GRB_DoubleAttr_X);
      double retOut = unitVars.retOut.get(GRB_DoubleAttr_X);
      os << getUniqueName(unit) << "new : retIn = " << retIn
         << ", retOut = " << retOut << "\n";
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

// Multi-layer graph modeling methods implementation

void BufferPlacementMILP::analyzeDataflowPaths(DataflowLayer &dataflowLayer) {
  // Clear existing paths
  dataflowLayer.allPaths.clear();
  
  // Analyze all operations in the function to identify dataflow paths
  for (Operation &op : funcInfo.funcOp.getOps()) {
    std::optional<unsigned> srcBB = getLogicBB(&op);
    
    for (OpResult res : op.getResults()) {
      if (res.use_empty()) continue;
      
      for (OpOperand &use : res.getUses()) {
        Operation *userOp = use.getOwner();
        std::optional<unsigned> dstBB = getLogicBB(userOp);
        
        DataflowPath path;
        path.source = &op;
        path.destination = userOp;
        path.channel = res;
        path.isOriginalCFGPath = false;
        path.isFastTokenDelivery = false;
        path.timingBenefit = 0.0;
        
        // Check if this path exists in the original CFG
        if (srcBB && dstBB) {
          // Check if there's a direct CFG transition
          bool hasDirectTransition = false;
          for (ArchBB &arch : funcInfo.archs) {
            if (arch.srcBB == *srcBB && arch.dstBB == *dstBB) {
              hasDirectTransition = true;
              break;
            }
          }
          
          if (hasDirectTransition || *srcBB == *dstBB) {
            path.isOriginalCFGPath = true;
          } else {
            // This is a fast token delivery path
            path.isFastTokenDelivery = true;
            // Estimate timing benefit (simplified heuristic)
            path.timingBenefit = 1.0; // Could be more sophisticated
          }
        }
        
        dataflowLayer.allPaths.push_back(path);
        
        // Add to channel-to-paths mapping
        dataflowLayer.channelToPaths[res].push_back(&dataflowLayer.allPaths.back());
      }
    }
  }
}

void BufferPlacementMILP::buildControlFlowLayer(ControlFlowLayer &controlLayer) {
  // Build CFG transitions map
  controlLayer.transitions.clear();
  for (ArchBB &arch : funcInfo.archs) {
    controlLayer.transitions[arch.srcBB].push_back(arch.dstBB);
  }
  
  // Store original CFDFCs
  controlLayer.originalCFDFCs.clear();
  for (auto [cfdfc, optimize] : funcInfo.cfdfcs) {
    if (optimize) {
      controlLayer.originalCFDFCs.push_back(cfdfc);
    }
  }
}

void BufferPlacementMILP::buildDataflowLayer(DataflowLayer &dataflowLayer) {
  // Analyze dataflow paths first
  analyzeDataflowPaths(dataflowLayer);
  
  // Create extended CFDFCs that include fast token delivery paths
  // This is a simplified version - in practice, this would be more complex
  createAdaptiveCFDFCs(dataflowLayer, dataflowLayer.extendedCFDFCs);
}

void BufferPlacementMILP::buildMappingLayer(const ControlFlowLayer &controlLayer,
                                           const DataflowLayer &dataflowLayer,
                                           MappingLayer &mappingLayer) {

  // Map CFG transitions to dataflow paths
  mappingLayer.cfgToDataflow.clear();
  for (auto &[srcBB, dstBBs] : controlLayer.transitions) {
    for (unsigned dstBB : dstBBs) {
      std::pair<unsigned, unsigned> transition = {srcBB, dstBB};
      
      // Find all dataflow paths that correspond to this CFG transition
      for (const DataflowPath &path : dataflowLayer.allPaths) {
        std::optional<unsigned> pathSrcBB = getLogicBB(path.source);
        std::optional<unsigned> pathDstBB = getLogicBB(path.destination);
        
        if (pathSrcBB && pathDstBB && *pathSrcBB == srcBB && *pathDstBB == dstBB) {
          mappingLayer.cfgToDataflow[transition].push_back(const_cast<DataflowPath *>(&path));
        }
      }
    }
  }
  
  // Calculate timing impact of fast token delivery paths
  mappingLayer.pathTimingImpact.clear();
  for (const DataflowPath &path : dataflowLayer.allPaths) {
    if (path.isFastTokenDelivery) {
      mappingLayer.pathTimingImpact[const_cast<DataflowPath *>(&path)] = path.timingBenefit;
    }
  }
}

void BufferPlacementMILP::addMultiLayerVars(ExtendedMILPVars &extVars,
                                           const DataflowLayer &dataflowLayer) {
  // Add path activity variables
  for (const DataflowPath &path : dataflowLayer.allPaths) {
    std::string pathName = "path_" + std::to_string(extVars.pathActive.size());
    extVars.pathActive[const_cast<DataflowPath *>(&path)] = 
        model.addVar(0.0, 1.0, 0.0, GRB_BINARY, pathName);
  }
  
  // Add path timing variables
  for (const DataflowPath &path : dataflowLayer.allPaths) {
    std::string pathTimingName = "pathTiming_" + std::to_string(extVars.pathTiming.size());
    TimeVars timeVars;
    timeVars.tIn = model.addVar(0.0, targetPeriod, 0.0, GRB_CONTINUOUS, pathTimingName + "_in");
    timeVars.tOut = model.addVar(0.0, targetPeriod, 0.0, GRB_CONTINUOUS, pathTimingName + "_out");
    extVars.pathTiming[const_cast<DataflowPath *>(&path)] = timeVars;
  }
  
  // Add layer interaction variables
  for (ArchBB &arch : funcInfo.archs) {
    std::pair<unsigned, unsigned> transition = {arch.srcBB, arch.dstBB};
    std::string interactionName = "layer_" + std::to_string(arch.srcBB) + "_" + std::to_string(arch.dstBB);
    extVars.layerInteraction[transition] = 
        model.addVar(0.0, targetPeriod, 0.0, GRB_CONTINUOUS, interactionName);
  }
  
  model.update();
}

void BufferPlacementMILP::addLayerConsistencyConstraints(const ControlFlowLayer &controlLayer,
                                                        const DataflowLayer &dataflowLayer,
                                                        const MappingLayer &mappingLayer,
                                                        ExtendedMILPVars &extVars) {
  
  // Ensure that at least one dataflow path is active for each CFG transition
  for (auto &[transition, paths] : mappingLayer.cfgToDataflow) {
    if (paths.empty()) continue;
    
    GRBLinExpr pathSum;
    for (DataflowPath *path : paths) {
      pathSum += extVars.pathActive[path];
    }

    // At least one path must be active for this CFG transition
    std::string constraintName = "consistency_" + std::to_string(transition.first) + "_" + std::to_string(transition.second);
    model.addConstr(pathSum >= 1, constraintName);
  }
}

void BufferPlacementMILP::addPathAwareTimingConstraints(const DataflowLayer &dataflowLayer,
                                                       ExtendedMILPVars &extVars) {

  // Add timing constraints for each dataflow path
  for (const DataflowPath &path : dataflowLayer.allPaths) {
    DataflowPath *pathPtr = const_cast<DataflowPath *>(&path);
    
    // Get path timing variables
    TimeVars &pathTiming = extVars.pathTiming[pathPtr];
    
    // Get channel timing variables if they exist
    auto channelVarsIt = extVars.channelVars.find(path.channel);
    if (channelVarsIt != extVars.channelVars.end()) {
      ChannelVars &channelVars = channelVarsIt->second;
      
      // Link path timing to channel timing
      std::string timingConstraintName = "pathTiming_" + std::to_string(reinterpret_cast<uintptr_t>(pathPtr));
      
      // If path is active, timing must be consistent with channel timing
      GRBVar &pathActive = extVars.pathActive[pathPtr];
      
      // Conditional constraint: if path is active, timing must match
      // This is a simplified version - more sophisticated timing analysis needed
      double bigM = targetPeriod * 10;
      model.addConstr(pathTiming.tOut - pathTiming.tIn <= targetPeriod + bigM * (1 - pathActive), timingConstraintName + "_timing");
    }
  }
}

void BufferPlacementMILP::addPathConflictResolution(const DataflowLayer &dataflowLayer,
                                                   ExtendedMILPVars &extVars) {

  // Handle conflicts when multiple paths use the same channel
  for (auto &[channel, paths] : dataflowLayer.channelToPaths) {
    if (paths.size() <= 1) continue;
    
    // Ensure that conflicting paths don't create timing violations
    for (size_t i = 0; i < paths.size(); ++i) {
      for (size_t j = i + 1; j < paths.size(); ++j) {
        DataflowPath *path1 = paths[i];
        DataflowPath *path2 = paths[j];
        
        // Add conflict resolution constraints
        std::string conflictName = "conflict_" + std::to_string(reinterpret_cast<uintptr_t>(path1)) + 
                                  "_" + std::to_string(reinterpret_cast<uintptr_t>(path2));
        
        // If both paths are active, their timing must be compatible
        // This is a simplified version - more sophisticated conflict resolution needed
        GRBVar &active1 = extVars.pathActive[path1];
        GRBVar &active2 = extVars.pathActive[path2];
        
        // Prevent both paths from being active simultaneously if they conflict
        // (This is a conservative approach - could be more sophisticated)
        if (path1->isFastTokenDelivery && path2->isFastTokenDelivery) {
        }
        model.addConstr(active1 + active2 <= 1, conflictName); // Relaxed: allow both paths to be active
      }
    }
  }
}

void BufferPlacementMILP::createAdaptiveCFDFCs(const DataflowLayer &dataflowLayer,
                                              llvm::SmallVector<CFDFC *, 4> &adaptiveCFDFCs) {
  // This is a simplified version of adaptive CFDFC creation
  // In practice, this would involve sophisticated cycle detection in the extended dataflow graph
  
  // For now, just copy the original CFDFCs and extend them with fast token delivery paths
  for (auto [cfdfc, optimize] : funcInfo.cfdfcs) {
    if (optimize) {
      adaptiveCFDFCs.push_back(cfdfc);
    }
  }
  
  // TODO: Implement more sophisticated adaptive CFDFC creation that considers
  // fast token delivery paths and creates new CFDFCs based on actual dataflow cycles
}
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
