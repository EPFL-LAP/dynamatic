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
#include <cmath>
#include <limits>
#include <queue>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;

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
                                         double targetPeriod, bool bufPenalty)
    : MILP<BufferPlacement>(env), timingDB(timingDB),
      targetPeriod(targetPeriod), funcInfo(funcInfo), logger(nullptr),
      bufPenalty(bufPenalty) {
  initialize();
}

BufferPlacementMILP::BufferPlacementMILP(GRBEnv &env, FuncInfo &funcInfo,
                                         const TimingDatabase &timingDB,
                                         double targetPeriod, bool bufPenalty,
                                         Logger &logger, StringRef milpName)
    : MILP<BufferPlacement>(env, logger.getLogDir() +
                                     llvm::sys::path::get_separator() +
                                     milpName),
      timingDB(timingDB), targetPeriod(targetPeriod), funcInfo(funcInfo),
      logger(&logger), bufPenalty(bufPenalty) {
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

  // Record the channel-internal propagation edge for critical-path
  // reconstruction. If a buffer cuts this signal the edge will simply not be
  // tight in the solution and the back-trace will stop at the buffer.
  if (logger)
    timingEdges.push_back({PathPin{channel, signal, /*isOut=*/false},
                           PathPin{channel, signal, /*isOut=*/true},
                           props.delay, "channel"});
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

    // The unit is not pipelined, add a path constraint for each input/output
    // port pair in the unit
    forEachIOPair(unit, [&](Value in, Value out) {
      // The input/output channels must both be inside the CFDFC union
      if (!filter(in) || !filter(out))
        return;

      // Flip channels on ready path which goes upstream
      if (type == SignalType::READY)
        std::swap(in, out);

      GRBVar &tInPort = vars.channelVars[in].signalVars[type].path.tOut;
      GRBVar &tOutPort = vars.channelVars[out].signalVars[type].path.tIn;
      // Arrival time at unit's output port must be greater than arrival
      // time at unit's input port + the unit's combinational data delay
      model.addConstr(tOutPort >= tInPort + delay, "path_combDelay");

      // Record the through-unit propagation edge for critical-path
      // reconstruction. For the ready signal `in`/`out` have been swapped
      // above, so this correctly captures the upstream direction.
      if (logger)
        timingEdges.push_back({PathPin{in, type, /*isOut=*/true},
                               PathPin{out, type, /*isOut=*/false}, delay,
                               "unit"});
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
  ChannelVars &channelVars = vars.channelVars[channel];
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

    GRBVar &tInPort = vars.channelVars[in].elastic.tOut;
    GRBVar &tOutPort = vars.channelVars[out].elastic.tIn;
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

    // No throughput constraints on channels going to LSQ stores
    if (isa<handshake::LSQStoreOp>(dstOp))
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
    // If there is an opaque buffer, the summed channel and CFDFC throughputs
    // cannot exceed the number of buffer slots. If there is not, the combined
    // throughput can exceed the number of slots by 1
    model.addConstr(chThroughput + cfVars.throughput + dataBuf - bufNumSlots <=
                        1,
                    "throughput_combined");
  }
}

void BufferPlacementMILP::addUnitThroughputConstraints(CFDFC &cfdfc) {
  CFDFCVars &cfVars = vars.cfVars[&cfdfc];
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

void BufferPlacementMILP::addObjective(ValueRange channels,
                                       ArrayRef<CFDFC *> cfdfcs,
                                       bool minimizeSlack) {
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
  if (bufPenalty) {
    double bufPenaltyMul = 1e-4;
    double slotPenaltyMul = 1e-5;
    for (Value channel : channels) {
      ChannelVars &channelVars = vars.channelVars[channel];
      objective -= maxCoefCFDFC * bufPenaltyMul * channelVars.bufPresent;
      objective -= maxCoefCFDFC * slotPenaltyMul * channelVars.bufNumSlots;
    }
  }

  // Finally, set the MILP objective
  if (minimizeSlack)
    addSlackMinimizingObjective(objective);
  else
    model.setObjective(objective, GRB_MAXIMIZE);
}

void BufferPlacementMILP::addSlackMinimizingObjective(
    const GRBLinExpr &primaryObjective) {
  // All hierarchical objectives share the model's optimization sense, so we
  // maximize both the primary objective and the negated arrival-time sum (i.e.,
  // we minimize arrival times).
  model.set(GRB_IntAttr_ModelSense, GRB_MAXIMIZE);

  // Primary objective: highest priority, so its optimal value is preserved.
  model.setObjectiveN(primaryObjective, /*index=*/0, /*priority=*/1);

  // Secondary objective: strictly lower priority. Minimizing the sum of all
  // arrival-time variables pins them to their true static-timing-analysis
  // values in the solution, which makes the reported slack exact.
  GRBLinExpr arrivalSum;
  for (auto &[channel, channelVars] : vars.channelVars)
    for (auto &[signal, signalVars] : channelVars.signalVars)
      arrivalSum += signalVars.path.tIn + signalVars.path.tOut;
  model.setObjectiveN(-arrivalSum, /*index=*/1, /*priority=*/0);
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
    os << "Throughput of CFDFC #" << idx << ": " << throughput << "\n";
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
}

void BufferPlacementMILP::logCriticalPath(unsigned numPaths) {
  assert(logger && "no logger was provided");
  mlir::raw_indented_ostream &os = **logger;

  os << "\n# ================= #\n";
  os << "# Critical Paths    #\n";
  os << "# ================= #\n\n";

  // Returns the arrival time of a pin in the solution.
  auto arrivalOf = [&](const PathPin &pin) -> double {
    TimeVars &path = vars.channelVars[pin.channel].signalVars[pin.signal].path;
    return (pin.isOut ? path.tOut : path.tIn).get(GRB_DoubleAttr_X);
  };

  // Returns a readable name for a pin, e.g. "myChannel [data].out".
  auto nameOf = [&](const PathPin &pin) -> std::string {
    std::string name;
    if (!pin.channel.getUses().empty())
      name = getUniqueName(*pin.channel.getUses().begin());
    else
      name = "<unnamed>";
    return name + " [" + getSignalName(pin.signal).str() + "]" +
           (pin.isOut ? ".out" : ".in");
  };

  // Returns whether a buffer register cuts `pin`'s channel-internal edge, which
  // makes the pin a segment start (the register launches a fresh path).
  auto isBufferCut = [&](const PathPin &pin) -> bool {
    return vars.channelVars[pin.channel]
               .signalVars[pin.signal]
               .bufPresent.get(GRB_DoubleAttr_X) > 0.5;
  };

  // A path grown backward from an endpoint. `pins` runs endpoint -> current
  // frontier and `hops[i]` is the edge feeding `pins[i]` from `pins[i + 1]`.
  // `acc` is the delay accumulated from the endpoint to the frontier pin.
  //
  // `bound = arrivalOf(frontier) + acc` is an upper bound on the endpoint
  // arrival of any completion of this prefix: extending backward through an
  // edge src->dst adds its delay to `acc` but, since the model guarantees
  // arrivalOf(src) + delay <= arrivalOf(dst), the bound can only stay equal (a
  // tight, exactly-critical edge) or decrease. Popping the frontier by
  // decreasing `bound` therefore yields complete paths in exact order of
  // decreasing endpoint arrival = increasing slack, so the first `numPaths`
  // completed are the globally most critical ones. Unlike a pure critical-path
  // trace this also follows slacky edges, so sub-critical branches surface once
  // the tighter paths ahead of them are exhausted; reported paths may overlap.
  struct PartialPath {
    SmallVector<PathPin> pins;
    SmallVector<const TimingEdge *> hops;
    double acc;
    double bound;
  };

  auto byBound = [](const PartialPath &a, const PartialPath &b) {
    return a.bound < b.bound; // std::priority_queue pops the largest first
  };
  
  std::priority_queue<PartialPath, std::vector<PartialPath>, decltype(byBound)>
      frontier(byBound);

  // Seed with every output pin as a candidate endpoint (its arrival is `tOut`,
  // where the "arrival <= targetPeriod" constraint binds).
  for (auto &[channel, channelVars] : vars.channelVars)
    for (auto &[signal, _] : channelVars.signalVars) {
      PathPin endpoint{channel, signal, /*isOut=*/true};
      frontier.push({{endpoint}, {}, 0.0, arrivalOf(endpoint)});
    }

  if (frontier.empty()) {
    os << "No path variables in the model.\n";
    return;
  }

  os << "Target period : " << targetPeriod << " ns\n\n";
  os << "Reporting up to the " << numPaths
     << " most critical paths (overlaps allowed).\n\n";

  auto printPath = [&](const PartialPath &path, unsigned idx) {
    const PathPin &endpoint = path.pins.front();
    // The endpoint arrival contributed by *this* path is exactly `bound`
    // (= arrivalOf(start) + total delay along the path).
    double pathArrival = path.bound;
    os << "Critical path #" << idx << "  (slack " << (targetPeriod - pathArrival)
       << " ns, endpoint " << nameOf(endpoint) << ", arrival " << pathArrival
       << " ns)\n";
    os.indent();
    double cum = pathArrival; // signal arrival at pins[i] along this path
    for (unsigned i = 0; i < path.pins.size(); ++i) {
      const PathPin &pin = path.pins[i];
      os << nameOf(pin) << "  (arrival " << cum << " ns)\n";
      if (i < path.hops.size()) {
        const TimingEdge *edge = path.hops[i];
        os << "^ via " << edge->kind << " (delay " << edge->delay << " ns)\n";
        cum -= edge->delay;
      } else {
        // Last pin: explain why the segment starts here.
        if (isBufferCut(pin))
          os << "^ start: buffer register\n";
        else if (!pin.channel.getDefiningOp())
          os << "^ start: circuit input\n";
        else
          os << "^ start: register / port\n";
      }
    }
    os.unindent();
    os << "\n";
  };

  // Best-first enumeration. Cap total expansions as a safety net against
  // pathological fan-out in large circuits.
  constexpr unsigned maxPops = 200000;
  unsigned reported = 0, pops = 0;
  while (!frontier.empty() && reported < numPaths && pops < maxPops) {
    PartialPath cur = frontier.top();
    frontier.pop();
    ++pops;
    const PathPin &frontierPin = cur.pins.back();

    // Collect the followable incoming edges of the frontier pin: skip
    // buffer-cut channel edges (the register ends the segment) and edges that
    // would revisit a pin already on the path (cycle guard).
    SmallVector<const TimingEdge *> incoming;
    for (const TimingEdge &edge : timingEdges) {
      if (edge.dst != frontierPin)
        continue;
      if (edge.kind == "channel" && isBufferCut(frontierPin))
        continue;
      if (llvm::is_contained(cur.pins, edge.src))
        continue;
      incoming.push_back(&edge);
    }

    if (incoming.empty()) {
      // The frontier pin is a segment start, so `cur` is a complete path. Skip
      // degenerate single-pin "paths" (an endpoint that is itself a start).
      if (cur.pins.size() >= 2)
        printPath(cur, ++reported);
      continue;
    }

    // Branch on every followable incoming edge.
    for (const TimingEdge *edge : incoming) {
      PartialPath next = cur;
      next.pins.push_back(edge->src);
      next.hops.push_back(edge);
      next.acc += edge->delay;
      next.bound = arrivalOf(edge->src) + next.acc;
      frontier.push(std::move(next));
    }
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
