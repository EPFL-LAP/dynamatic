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
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LogicBB.h"
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
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace circt;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

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

/// Returns the bitwidth of a channel.
static unsigned getChannelBitwidth(Value channel) {
  Type channelType = channel.getType();
  if (isa<NoneType>(channelType))
    return 0;
  if (isa<IntegerType, FloatType>(channelType))
    return channelType.getIntOrFloatBitWidth();
  if (isa<IndexType>(channelType))
    return IndexType::kInternalStorageBitWidth;
  llvm_unreachable("unsupported channel type");
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
    bitwidth = getChannelBitwidth(channel);
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
    bitwidth = getChannelBitwidth(channel);
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
  unsatisfiable = failed(mapChannelsToProperties());
}

BufferPlacementMILP::BufferPlacementMILP(GRBEnv &env, FuncInfo &funcInfo,
                                         const TimingDatabase &timingDB,
                                         double targetPeriod, Logger &logger,
                                         StringRef milpName)
    : MILP<BufferPlacement>(env, logger.getLogDir() + path::get_separator() +
                                     milpName),
      timingDB(timingDB), targetPeriod(targetPeriod), funcInfo(funcInfo),
      logger(&logger) {
  unsatisfiable = failed(mapChannelsToProperties());
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

  for (auto [sig, signalVars] : channelVars.signalVars) {
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
    if (arith::SelectOp selOp = dyn_cast<arith::SelectOp>(dstOp))
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
  double bufPenaltyMul = 1e-4;
  double slotPenaltyMul = 1e-5;
  for (Value channel : channels) {
    ChannelVars &channelVars = vars.channelVars[channel];
    objective -= maxCoefCFDFC * bufPenaltyMul * channelVars.bufPresent;
    objective -= maxCoefCFDFC * slotPenaltyMul * channelVars.bufNumSlots;
  }

  // Finally, set the MILP objective
  model.setObjective(objective, GRB_MAXIMIZE);
}

void BufferPlacementMILP::addInternalBuffers(Channel &channel) {
  // Add slots present at the source unit's output ports
  if (const TimingModel *model = timingDB.getModel(channel.producer)) {
    channel.props->minTrans += model->outputModel.transparentSlots;
    channel.props->minOpaque += model->outputModel.opaqueSlots;
  }

  // Add slots present at the destination unit's input ports
  if (const TimingModel *model = timingDB.getModel(channel.consumer)) {
    channel.props->minTrans += model->inputModel.transparentSlots;
    channel.props->minOpaque += model->inputModel.opaqueSlots;
  }
}

void BufferPlacementMILP::deductInternalBuffers(Value channel,
                                                PlacementResult &result) {
  unsigned numTransToDeduct = 0, numOpaqueToDeduct = 0;

  if (isa<OpResult>(channel)) {
    // Remove slots present at the source unit's output ports
    if (const TimingModel *model = timingDB.getModel(channel.getDefiningOp())) {
      numTransToDeduct += model->outputModel.transparentSlots;
      numOpaqueToDeduct += model->outputModel.opaqueSlots;
    }
  }

  // Remove slots present at the destination unit's input ports
  Operation *consumer = *channel.getUsers().begin();
  if (const TimingModel *model = timingDB.getModel(consumer)) {
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
  for (Value opr : op->getOperands()) {
    if (!isa<MemRefType>(opr.getType())) {
      for (OpResult res : op->getResults())
        if (!isa<MemRefType>(res.getType()))
          callback(opr, res);
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

LogicalResult BufferPlacementMILP::mapChannelsToProperties() {
  // Initialize the large constant (for elasticity constraints)
  auto ops = funcInfo.funcOp.getOps();
  largeCst = std::distance(ops.begin(), ops.end()) + 2;

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

    // Increase the minimum number of slots if internal buffers are present, and
    // check for satisfiability
    addInternalBuffers(channel);
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
  for (auto [idx, arg] : llvm::enumerate(funcInfo.funcOp.getArguments())) {
    Channel channel(arg, funcInfo.funcOp, *arg.getUsers().begin());
    if (failed(deriveBufferingProperties(channel)))
      return failure();
  }

  // Add channels originating from operations' results to the channel map
  for (Operation &op : funcInfo.funcOp.getOps()) {
    for (auto [idx, res] : llvm::enumerate(op.getResults())) {
      Channel channel(res, &op, *res.getUsers().begin());
      if (failed(deriveBufferingProperties(channel)))
        return failure();
    }
  }

  return success();
}
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
