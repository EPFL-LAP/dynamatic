//===- OptimizeMILP.cpp - optimize MILP model over CFDFC  -------*- C++ -*-===//
//
// This file implements the MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "dynamatic/Transforms/BufferPlacement/OptimizeMILP.h"
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
#include <fstream>
#include <functional>
#include <iostream>

using namespace circt;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

/// Returns the short name of an operation to be used as part of the name for a
/// Gurobi variable.
static inline std::string getVarName(Operation *op) {
  return op ? op->getName().stripDialect().str() : "arg";
}

static unsigned getTotalNumExecs(Value channel,
                                 llvm::MapVector<CFDFC *, bool> &cfdfcs) {
  Operation *srcOp = channel.getDefiningOp();
  if (!srcOp)
    // A channel which originates from a function argument executes only once
    return 1;

  Operation *dstOp = *channel.getUsers().begin();
  if (isa<handshake::SinkOp, handshake::MemoryControllerOp>(dstOp) ||
      isa<handshake::MemoryControllerOp>(srcOp))
    // Channels to/from a memory controller or to a sink are "never executed"
    return 0;

  // Iterate over all CFDFCs which contain the channel to determine its total
  // number of executions. Backedges are executed one less time than "forward
  // edges" since they are only taken between executions of the cycle the CFDFC
  // represents
  unsigned numExec = isBackedge(srcOp, dstOp) ? 0 : 1;
  for (auto &[cfdfc, _] : cfdfcs)
    if (cfdfc->channels.contains(channel))
      numExec += cfdfc->numExec;
  return numExec;
}

BufferPlacementMILP::BufferPlacementMILP(handshake::FuncOp funcOp,
                                         llvm::MapVector<CFDFC *, bool> &cfdfcs,
                                         TimingDatabase &timingDB,
                                         double targetPeriod, double maxPeriod,
                                         GRBEnv &env, double timeLimit)
    : targetPeriod(targetPeriod), maxPeriod(maxPeriod), funcOp(funcOp),
      cfdfcs(cfdfcs), timingDB(timingDB),
      numUnits(std::distance(funcOp.getOps().begin(), funcOp.getOps().end())),
      model(GRBModel(env)) {
  // Set a time limit for the MILP
  model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit);

  // Combines any channel-specific buffering properties coming from IR
  // annotations to internal buffer specifications and stores the combined
  // properties into the channel map. Fails and marks the MILP unsatisfiable if
  // any of those combined buffering properties become unsatisfiable.
  auto deriveBufferingProperties = [&](Channel &channel) -> LogicalResult {
    // Increase the minimum number of slots if internal buffers are present, and
    // check for satisfiability
    unsatisfiable = failed(addInternalBuffers(channel));
    if (!unsatisfiable) {
      std::stringstream stream;
      stream << "Including internal component buffers into buffering "
                "properties of outgoing channel made them unsatisfiable. "
                "Properties are "
             << *channel.props;
      return channel.producer.emitError() << stream.str();
    }
    channels[channel.value] = *channel.props;
    return failure(unsatisfiable);
  };

  // Add channels originating from function arguments to the channel map
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    Channel channel(arg, *funcOp, **arg.getUsers().begin());
    if (failed(deriveBufferingProperties(channel)))
      return;
  }

  // Add channels originating from operations' results to the channel map
  for (Operation &op : funcOp.getOps()) {
    for (auto [idx, res] : llvm::enumerate(op.getResults())) {
      Channel channel(res, op, **res.getUsers().begin());
      if (failed(deriveBufferingProperties(channel)))
        return;
    }
  }
}

bool BufferPlacementMILP::arePlacementConstraintsSatisfiable() {
  return !unsatisfiable;
}

LogicalResult BufferPlacementMILP::setup() {
  if (failed(createVars()))
    return failure();

  // All constraints apply over all channels and units
  std::vector<Value> allChannels(channels.size());
  for (auto &[channel, _] : channels)
    allChannels.push_back(channel);
  std::vector<Operation *> allUnits;
  for (Operation &op : funcOp.getOps())
    allUnits.push_back(&op);

  if (failed(addCustomChannelConstraints(allChannels)) ||
      failed(addPathConstraints(allChannels, allUnits)) ||
      failed(addElasticityConstraints(allChannels, allUnits)))
    return failure();

  // Add throughput constraints over each CFDFC that was marked to be optimized
  for (auto &[cfdfc, _] : vars.cfdfcs)
    if (cfdfcs[cfdfc])
      if (failed(addThroughputConstraints(*cfdfc)))
        return failure();

  // Finally, add the MILP objective
  return addObjective();
}

LogicalResult
BufferPlacementMILP::optimize(DenseMap<Value, PlacementResult> &placement) {
  if (unsatisfiable)
    return funcOp->emitError() << "The MILP is unsatisfiable: customized "
                                  "channel constraints are incompatible "
                                  "with buffers included inside units.";

  // Optimize the model, then check whether we found an optimal solution or
  // whether we reached the time limit
  model.optimize();
  if (!(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) ||
      (model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT &&
       model.get(GRB_DoubleAttr_ObjVal) > 0))
    return funcOp.emitError()
           << "Failed to optimize the buffer placement MILP.";

  // Fill in placement information
  for (auto &[value, channelVars] : vars.channels) {
    if (channelVars.bufPresent.get(GRB_DoubleAttr_X) > 0) {
      unsigned numSlotsToPlace = static_cast<unsigned>(
          channelVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
      bool placeOpaque = channelVars.bufIsOpaque.get(GRB_DoubleAttr_X) > 0;

      ChannelBufProps &props = channels[value];
      PlacementResult result;

      if (placeOpaque && numSlotsToPlace > 0) {
        // We want as many slots as possible to be transparent and at least one
        // opaque slot, while satisfying all buffering constraints
        unsigned actualMinOpaque = std::max(1U, props.minOpaque);
        if (props.maxTrans.has_value() &&
            (props.maxTrans.value() < numSlotsToPlace - actualMinOpaque)) {
          result.numTrans = props.maxTrans.value();
          result.numOpaque = numSlotsToPlace - result.numTrans;
        } else {
          result.numOpaque = actualMinOpaque;
          result.numTrans = numSlotsToPlace - result.numOpaque;
        }
      } else
        // All slots should be transparent
        result.numTrans = numSlotsToPlace;

      Channel channel(value);
      deductInternalBuffers(channel, result);
      placement[value] = result;
    }
  }
  return success();
}

LogicalResult BufferPlacementMILP::createVars() {
  for (auto [idx, cfdfcAndOpt] : llvm::enumerate(cfdfcs))
    if (failed(createCFDFCVars(*cfdfcAndOpt.first, idx)))
      return failure();
  if (failed(createChannelVars()))
    return failure();

  // Update the model before returning so that these variables can be referenced
  // safely during the rest of model creation
  model.update();
  return success();
}

LogicalResult BufferPlacementMILP::createCFDFCVars(CFDFC &cfdfc, unsigned uid) {
  std::string cfdfcID = std::to_string(uid);
  CFDFCVars cfdfcVars;

  // Create a set of variables for each CFDFC unit
  for (auto [idx, unit] : llvm::enumerate(cfdfc.units)) {
    std::string unitID = std::to_string(idx);

    // Create the two unit variables
    UnitVars unitVar;
    std::string unitName = getVarName(unit);
    unitVar.retIn =
        model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                     "mg" + cfdfcID + "_inRetimeTok_" + unitName + unitID);

    // If the component is combinational (i.e., 0 latency) its output fluid
    // retiming equals its input fluid retiming, otherwise it is different
    double latency;
    if (failed(timingDB.getLatency(unit, latency)))
      return failure();
    if (latency == 0.0)
      unitVar.retOut = unitVar.retIn;
    else
      unitVar.retOut =
          model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                       "mg" + cfdfcID + "_outRetimeTok_" + unitName + unitID);

    cfdfcVars.units[unit] = unitVar;
  }

  // Create a variable for each CFDFC channel
  for (auto [idx, channel] : llvm::enumerate(cfdfc.channels)) {
    // Construct a name for the variable
    Operation *srcOp = channel.getDefiningOp();
    Operation *dstOp = *channel.getUsers().begin();
    std::string varName = "mg" + cfdfcID + "_" + getVarName(srcOp) + "_" +
                          getVarName(dstOp) + "_" + std::to_string(idx);
    // Add a variable for the channel's throughput
    cfdfcVars.channelThroughputs[channel] = model.addVar(
        0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "throughput_" + varName);
  }

  // Create a variable for the CFDFC's throughput
  cfdfcVars.thoughput = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                     "cfdfcThroughput_" + cfdfcID);

  // Add the CFDFC variables to the global set of variables
  vars.cfdfcs[&cfdfc] = cfdfcVars;
  return success();
}

LogicalResult BufferPlacementMILP::createChannelVars() {
  // Create a set of variables for each channel in the circuit
  for (auto [idx, channelAndProps] : llvm::enumerate(channels)) {
    auto &channel = channelAndProps.first;

    // Construct a suffix for all variable names
    Operation *srcOp = channel.getDefiningOp();
    Operation *dstOp = *channel.getUsers().begin();
    std::string suffix = "_" + getVarName(srcOp) + "_" + getVarName(dstOp) +
                         "_" + std::to_string(idx);

    // Create the set of variables for the channel
    ChannelVars channelVar;
    channelVar.tPathIn = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                      "timePathIn" + suffix);
    channelVar.tPathOut = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                       "timePathOut" + suffix);
    channelVar.tElasIn = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                      "timeElasticIn" + suffix);
    channelVar.tElasOut = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                       "timeElasticOut" + suffix);
    channelVar.bufPresent =
        model.addVar(0, GRB_INFINITY, 0.0, GRB_BINARY, "bufPresent" + suffix);
    channelVar.bufIsOpaque =
        model.addVar(0, GRB_INFINITY, 0.0, GRB_BINARY, "bufIsOpaque" + suffix);
    channelVar.bufNumSlots =
        model.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER, "bufNumSlots" + suffix);

    vars.channels[channel] = channelVar;
  }
  return success();
}

LogicalResult
BufferPlacementMILP::addCustomChannelConstraints(ValueRange customChannels) {
  for (Value channel : customChannels) {
    ChannelVars &chVars = vars.channels[channel];
    ChannelBufProps &props = channels[channel];

    // If the properties ask for both opaque and transaprent slots, let opaque
    // slots take over. Transparents slots will be placed "manually" from the
    // total number of slots indicated by the MILP's result
    if (props.minTrans > 0 && props.minOpaque > 0) {
      Operation *producer = getChannelProducer(channel);
      assert(producer && "channel producer must exist");
      producer->emitWarning()
          << "Outgoing channel requests placement of at least one transparent "
             "and at least one opaque slot on the channel, which the MILP does "
             "not formally support. To honor the properties, the MILP will be "
             "configured to only place opaque slots, some of which will be "
             "converted to transparent slots when parsing the MILP's solution.";
      unsigned minTotalSlots = props.minOpaque + props.minTrans;
      model.addConstr(chVars.bufNumSlots >= minTotalSlots);
      model.addConstr(chVars.bufIsOpaque == 1);

    } else if (props.minOpaque > 0) {
      // Force the MILP to place a minimum number of opaque slots
      model.addConstr(chVars.bufNumSlots >= props.minOpaque);
      model.addConstr(chVars.bufIsOpaque == 1);
    } else if (props.minTrans > 0)
      // Force the MILP to place a minimum number of transparent slots
      model.addConstr(chVars.bufNumSlots >= props.minTrans);

    // Set a maximum number of slots to be placed
    if (props.maxOpaque.has_value()) {
      if (*props.maxOpaque == 0)
        // Force the MILP to use transparent slots
        model.addConstr(chVars.bufIsOpaque == 1);
      if (props.maxTrans.has_value())
        // Force the MILP to use a maximum number of slots
        model.addConstr(chVars.bufNumSlots <=
                        *props.maxTrans + *props.maxOpaque);
    }
  }
  return success();
}

LogicalResult
BufferPlacementMILP::addPathConstraints(ValueRange pathChannels,
                                        ArrayRef<Operation *> pathUnits) {
  // Add path constraints for channels
  for (Value channel : pathChannels) {
    ChannelVars &chVars = vars.channels[channel];
    GRBVar &t1 = chVars.tPathIn;
    GRBVar &t2 = chVars.tPathOut;
    // Arrival time at channel's input must be lower than target clock period
    model.addConstr(t1 <= targetPeriod);
    // Arrival time at channel's output must be lower than target clock period
    model.addConstr(t2 <= targetPeriod);
    // If there isn't an opaque buffer on the channel, arrival time at channel's
    // output must be greater than at channel's input
    model.addConstr(t2 >= t1 - maxPeriod * chVars.bufIsOpaque);
  }

  // Add path constraints for units
  for (Operation *op : pathUnits) {
    double latency, dataDelay;
    if (failed(timingDB.getTotalDelay(op, SignalType::DATA, dataDelay)) ||
        failed(timingDB.getLatency(op, latency)))
      return failure();

    if (latency == 0.0) {
      // The unit is not pipelined, add a path constraint for each input/output
      // port pair in the unit
      forEachIOPair(op, [&](Value in, Value out) {
        GRBVar &tInPort = vars.channels[in].tPathOut;
        GRBVar &tOutPort = vars.channels[out].tPathIn;
        // Arrival time at unit's output port must be greater than arrival time
        // at unit's input port + the unit's combinational delay
        model.addConstr(tOutPort >= tInPort + dataDelay);
      });
    } else {
      // The unit is pipelined, add a constraint for every of the unit's inputs
      // and every of the unit's output ports

      // Input port constraints
      for (Value inChannel : op->getOperands()) {
        if (!vars.channels.contains(inChannel))
          continue;

        double inPortDelay;
        if (failed(timingDB.getPortDelay(op, SignalType::DATA, PortType::IN,
                                         inPortDelay)))
          return failure();

        GRBVar &tInPort = vars.channels[inChannel].tPathOut;
        // Arrival time at unit's input port + input port delay must be less
        // than the target clock period
        model.addConstr(tInPort + inPortDelay <= targetPeriod);
      }

      // Output port constraints
      for (OpResult outChannel : op->getResults()) {
        if (!vars.channels.contains(outChannel))
          continue;

        double outPortDelay;
        if (failed(timingDB.getPortDelay(op, SignalType::DATA, PortType::OUT,
                                         outPortDelay)))
          return failure();

        GRBVar &tOutPort = vars.channels[outChannel].tPathIn;
        // Arrival time at unit's output port is equal to the output port delay
        model.addConstr(tOutPort == outPortDelay);
      }
    }
  }
  return success();
}

LogicalResult BufferPlacementMILP::addElasticityConstraints(
    ValueRange elasticChannels, ArrayRef<Operation *> elasticUnits) {
  // Add elasticity constraints for channels
  for (Value channel : elasticChannels) {
    ChannelVars &chVars = vars.channels[channel];
    GRBVar &tIn = chVars.tElasIn;
    GRBVar &tOut = chVars.tElasOut;
    // Upper bound for the longest rigid path
    unsigned cstCoef = numUnits + 2;

    // If there is an opaque buffer on the channel, the channel elastic arrival
    // time at the ouput must be greater than at the input (breaks cycles!)
    model.addConstr(tOut >= tIn - cstCoef * chVars.bufIsOpaque);
    // If there is an opaque buffer, there must be at least one slot
    model.addConstr(chVars.bufNumSlots >= chVars.bufIsOpaque);
    // If there is at least one slot, there must be a buffer
    model.addConstr(chVars.bufPresent >= 0.01 * chVars.bufNumSlots);

    /// NOTE: it looks like the legacy implementation only adds these three
    /// constraints when the channel is part of a CFDFC, otherwise it adds a
    /// single different constraint.
  }

  // Add an elasticity constraint for every input/output port pair in the
  // elastic units
  for (Operation *op : elasticUnits) {
    forEachIOPair(op, [&](Value in, Value out) {
      GRBVar &tInPort = vars.channels[in].tElasOut;
      GRBVar &tOutPort = vars.channels[out].tElasIn;
      // The elastic arrival time at the output port must be at least one
      // greater than at the input port
      model.addConstr(tOutPort >= 1 + tInPort);
    });
  }
  return success();
}

LogicalResult BufferPlacementMILP::addThroughputConstraints(CFDFC &cfdfc) {
  CFDFCVars &cfdfcVars = vars.cfdfcs[&cfdfc];

  // Add a set of constraints for each CFDFC channel
  for (auto &[channel, chThroughput] : cfdfcVars.channelThroughputs) {
    Operation *srcOp = channel.getDefiningOp();
    Operation *dstOp = *(channel.getUsers().begin());

    ChannelVars &chVars = vars.channels[channel];
    GRBVar &retSrc = cfdfcVars.units[srcOp].retOut;
    GRBVar &retDst = cfdfcVars.units[dstOp].retIn;
    GRBVar &bufIsOpaque = chVars.bufIsOpaque;
    GRBVar &bufNumSlots = chVars.bufNumSlots;
    GRBVar &throughput = cfdfcVars.thoughput;
    unsigned backedge = isBackedge(srcOp, dstOp) ? 1 : 0;

    /// NOTE: the legacy implementation does not add any constraints here for
    /// the input channel to select operations that is less frequently
    /// executed (determined from CFDFC analysis I think). To be clean, this
    /// behavior should be achieved using channel-specific buffer properties
    /// (like it is done partially now in the hardcoded properties).

    // If the channel isn't a backedge, its throuhgput equals the difference
    // between the fluid retiming of tokens at its endpoints. Otherwise, it is
    // one less than this difference
    model.addConstr(chThroughput - backedge == retDst - retSrc);
    // If there is an opaque buffer, the CFDFC throughput cannot exceed the
    // channel throughput. If there is not, the CFDFC throughput can exceed the
    // channel thoughput by 1
    model.addConstr(throughput - chThroughput + bufIsOpaque <= 1);
    // If there is an opaque buffer, the summed channel and CFDFC throuhgputs
    // cannot exceed the number of buffer slots. If there is not, the combined
    // throuhgput can exceed the number of slots by 1
    model.addConstr(chThroughput + throughput + bufIsOpaque - bufNumSlots <= 1);
    // The channel's throughput cannot exceed the number of buffer slots
    model.addConstr(chThroughput <= bufNumSlots);
  }

  // Add a constraint for each pipelined CFDFC unit
  for (auto &[op, unitVars] : cfdfcVars.units) {
    double latency;
    if (failed(timingDB.getLatency(op, latency)))
      return failure();
    if (latency != 0.0) {
      GRBVar &retIn = unitVars.retIn;
      GRBVar &retOut = unitVars.retOut;
      GRBVar &throughput = cfdfcVars.thoughput;
      // The fluid retiming of tokens across the non-combinational unit must be
      // the same as its latency multiplied by the CFDFC's throughput
      model.addConstr(throughput * latency == retOut - retIn);
    }
  }
  return success();
}

LogicalResult BufferPlacementMILP::addObjective() {
  // Compute the total number of executions over all channels
  unsigned totalExecs = 0;
  for (auto &[channel, _] : vars.channels)
    totalExecs += getTotalNumExecs(channel, cfdfcs);

  // Create the expression for the MILP objective
  GRBLinExpr objective;

  // For each CFDFC, add a throuhgput contribution to the objective, weighted by
  // the "importance" of the CFDFC
  double maxCoefCFDFC = 0.0;
  if (totalExecs != 0) {
    for (auto &[cfdfc, cfdfcVars] : vars.cfdfcs) {
      double coef = cfdfc->channels.size() * cfdfc->numExec /
                    static_cast<double>(totalExecs);
      objective += coef * cfdfcVars.thoughput;
      maxCoefCFDFC = std::max(coef, maxCoefCFDFC);
    }
  }

  // In case we ran the MILP without providing any CFDFC, set the maximum CFDFC
  // coefficient to any positive value
  if (maxCoefCFDFC == 0.0)
    maxCoefCFDFC = 1.0;

  // For each channel, add a "penalty" in case a buffer is added to the channel
  double bufPenaltyMul = 1e-4;
  double slotPenaltyMul = 1e-5;
  for (auto &[_, chVar] : vars.channels)
    objective -= maxCoefCFDFC * (bufPenaltyMul * chVar.bufPresent +
                                 slotPenaltyMul * chVar.bufNumSlots);

  // Finally, set the MILP objective
  model.setObjective(objective, GRB_MAXIMIZE);
  return success();
}

LogicalResult BufferPlacementMILP::addInternalBuffers(Channel &channel) {
  // Add slots present at the source unit's output ports
  std::string srcName = channel.producer.getName().getStringRef().str();
  if (const TimingModel *model = timingDB.getModel(&channel.producer)) {
    channel.props->minTrans += model->outputModel.transparentSlots;
    channel.props->minOpaque += model->outputModel.opaqueSlots;
  }

  // Add slots present at the destination unit's input ports
  std::string dstName = channel.consumer.getName().getStringRef().str();
  if (const TimingModel *model = timingDB.getModel(&channel.consumer)) {
    channel.props->minTrans += model->inputModel.transparentSlots;
    channel.props->minOpaque += model->inputModel.opaqueSlots;
  }

  return success(channel.props->isSatisfiable());
}

void BufferPlacementMILP::deductInternalBuffers(Channel &channel,
                                                PlacementResult &result) {
  std::string srcName = channel.producer.getName().getStringRef().str();
  std::string dstName = channel.consumer.getName().getStringRef().str();
  unsigned numTransToDeduct = 0, numOpaqueToDeduct = 0;

  // Remove slots present at the source unit's output ports
  if (const TimingModel *model = timingDB.getModel(&channel.producer)) {
    numTransToDeduct += model->outputModel.transparentSlots;
    numOpaqueToDeduct += model->outputModel.opaqueSlots;
  }
  // Remove slots present at the destination unit's input ports
  if (const TimingModel *model = timingDB.getModel(&channel.consumer)) {
    numTransToDeduct += model->inputModel.transparentSlots;
    numOpaqueToDeduct += model->inputModel.opaqueSlots;
  }

  assert(numTransToDeduct > result.numTrans &&
         "not enough transparent slots were placed, the MILP was likely "
         "incorrectly configured");
  assert(numOpaqueToDeduct > result.numOpaque &&
         "not enough opaque slots were placed, the MILP was likely "
         "incorrectly configured");
  result.numTrans -= numTransToDeduct;
  result.numOpaque -= numOpaqueToDeduct;
}

void BufferPlacementMILP::forEachIOPair(
    Operation *op, const std::function<void(Value, Value)> &callback) {
  for (auto inChannel : op->getOperands()) {
    if (!vars.channels.contains(inChannel))
      continue;
    for (OpResult outChannel : op->getResults()) {
      if (!vars.channels.contains(outChannel))
        continue;
      callback(inChannel, outChannel);
    }
  }
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
