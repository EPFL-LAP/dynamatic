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
#include "dynamatic/Support/LogicBB.h"
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
#include "llvm/Support/Debug.h"
#include <fstream>
#include <functional>
#include <iostream>

#define DEBUG_TYPE "BUFFER_PLACEMENT"

using namespace circt;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

BufferPlacementMILP::BufferPlacementMILP(FuncInfo &funcInfo,
                                         const TimingDatabase &timingDB,
                                         double targetPeriod, double maxPeriod,
                                         GRBEnv &env)
    : timingDB(timingDB), targetPeriod(targetPeriod), maxPeriod(maxPeriod),
      funcInfo(funcInfo), model(GRBModel(env)) {
  // Set a 3-minutes time limit for the MILP
  model.getEnv().set(GRB_DoubleParam_TimeLimit, 180);

  // Give a unique name to each operation
  std::map<std::string, unsigned> instanceNameCntr;
  for (Operation &op : funcInfo.funcOp.getOps()) {
    std::string shortName = op.getName().stripDialect().str();
    nameUniquer[&op] =
        shortName + std::to_string(instanceNameCntr[shortName]++);
  }

  // Combines any channel-specific buffering properties coming from IR
  // annotations to internal buffer specifications and stores the combined
  // properties into the channel map. Fails and marks the MILP unsatisfiable if
  // any of those combined buffering properties become unsatisfiable.
  auto deriveBufferingProperties = [&](Channel &channel) -> LogicalResult {
    // Increase the minimum number of slots if internal buffers are present, and
    // check for satisfiability
    if (failed(addInternalBuffers(channel))) {
      unsatisfiable = true;
      std::stringstream stream;
      stream << "Including internal component buffers into buffering "
                "properties of outgoing channel made them unsatisfiable. "
                "Properties are "
             << *channel.props;
      return channel.producer.emitError() << stream.str();
    }
    channels[channel.value] = *channel.props;
    return success();
  };

  // Add channels originating from function arguments to the channel map
  for (auto [idx, arg] : llvm::enumerate(funcInfo.funcOp.getArguments())) {
    Channel channel(arg, *funcInfo.funcOp, **arg.getUsers().begin());
    if (failed(deriveBufferingProperties(channel)))
      return;
  }

  // Add channels originating from operations' results to the channel map
  for (Operation &op : funcInfo.funcOp.getOps()) {
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

  std::vector<Value> allChannels, allBufferizableChannels;
  for (auto &[channel, _] : channels) {
    allChannels.push_back(channel);
    if (channels[channel].isBufferizable())
      allBufferizableChannels.push_back(channel);
  }
  std::vector<Operation *> allUnits;
  for (Operation &op : funcInfo.funcOp.getOps())
    allUnits.push_back(&op);

  if (failed(addCustomChannelConstraints(allChannels)) ||
      failed(addPathConstraints(allBufferizableChannels, allUnits)) ||
      failed(addElasticityConstraints(allBufferizableChannels, allUnits)))
    return failure();

  // Add throughput constraints over each CFDFC that was marked to be optimized
  for (auto &[cfdfc, _] : vars.cfdfcs)
    if (funcInfo.cfdfcs[cfdfc])
      if (failed(addThroughputConstraints(*cfdfc)))
        return failure();

  // Finally, add the MILP objective
  return addObjective();
}

LogicalResult
BufferPlacementMILP::optimize(DenseMap<Value, PlacementResult> &placement) {
  if (unsatisfiable)
    return funcInfo.funcOp->emitError()
           << "The MILP is unsatisfiable: customized "
              "channel constraints are incompatible "
              "with buffers included inside units.";

  // Optimize the model, then check whether we found an optimal solution or
  // whether we reached the time limit
  model.write("model.lp");
  model.optimize();
  model.write("solutions.json");
  int status = model.get(GRB_IntAttr_Status);
  if (status != GRB_OPTIMAL && status != GRB_TIME_LIMIT)
    return funcInfo.funcOp->emitError()
           << "Gurobi failed with status code " << status;

  // Fill in placement information
  for (auto &[value, channelVars] : vars.channels) {
    if (channelVars.bufPresent.get(GRB_DoubleAttr_X) == 0)
      continue;

    unsigned numSlotsToPlace = static_cast<unsigned>(
        channelVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    bool placeOpaque = channelVars.bufIsOpaque.get(GRB_DoubleAttr_X) > 0;

    PlacementResult result;
    ChannelBufProps &props = channels[value];

    if (placeOpaque && numSlotsToPlace > 0) {
      /// NOTE: This matches the behavior of the legacy buffer placement pass
      /// However, a better placement may be achieved using the commented out
      /// logic below.
      result.numTrans = props.minTrans;
      result.numOpaque = numSlotsToPlace - props.minTrans;

      // We want as many slots as possible to be transparent and at least one
      // opaque slot, while satisfying all buffering constraints
      // unsigned actualMinOpaque = std::max(1U, props.minOpaque);
      // if (props.maxTrans.has_value() &&
      //     (props.maxTrans.value() < numSlotsToPlace - actualMinOpaque)) {
      //   result.numTrans = props.maxTrans.value();
      //   result.numOpaque = numSlotsToPlace - result.numTrans;
      // } else {
      //   result.numOpaque = actualMinOpaque;
      //   result.numTrans = numSlotsToPlace - result.numOpaque;
      // }
    } else
      // All slots should be transparent
      result.numTrans = numSlotsToPlace;

    // Print the placement decision
    LLVM_DEBUG(std::stringstream propsStr; propsStr << props;
               llvm::errs()
               << "=== PLACEMENT DECISION ===\nThe MILP determined that "
               << numSlotsToPlace << " "
               << (placeOpaque ? "opaque" : "transparent")
               << " slots should be placed on channel\n"
               << getChannelName(value) << ", which has buffering properties: "
               << propsStr.str() << ".\n"
               << "Returned placement specifies " << result.numTrans
               << " transparent slots and " << result.numOpaque
               << " opaque slots.\n";);

    Channel channel(value);
    deductInternalBuffers(channel, result);
    placement[value] = result;
  }

  return success();
}

LogicalResult BufferPlacementMILP::createVars() {
  for (auto [idx, cfdfcAndOpt] : llvm::enumerate(funcInfo.cfdfcs))
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
  std::string prefix = "cfdfc" + std::to_string(uid) + "_";
  CFDFCVars cfdfcVars;

  // Create a continuous Gurobi variable of the given name
  auto createVar = [&](const std::string &name) {
    return model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, name);
  };

  // Create a set of variables for each CFDFC unit
  for (auto [idx, unit] : llvm::enumerate(cfdfc.units)) {
    // Create the two unit variables
    UnitVars unitVar;
    std::string unitName = nameUniquer[unit] + std::to_string(idx);
    std::string varName = prefix + "inRetimeTok_" + unitName;
    unitVar.retIn = createVar(varName);

    // If the component is combinational (i.e., 0 latency) its output fluid
    // retiming equals its input fluid retiming, otherwise it is different
    double latency;
    if (failed(timingDB.getLatency(unit, latency)))
      latency = 0.0;
    if (latency == 0.0)
      unitVar.retOut = unitVar.retIn;
    else
      unitVar.retOut = createVar(prefix + "outRetimeTok_" + unitName);

    cfdfcVars.units[unit] = unitVar;
  }

  // Create a variable to represent the throughput of each CFDFC channel
  for (auto [idx, channel] : llvm::enumerate(cfdfc.channels))
    cfdfcVars.channelThroughputs[channel] =
        createVar(prefix + "throughput_" + getChannelName(channel));

  // Create a variable for the CFDFC's throughput
  cfdfcVars.throughput = createVar(prefix + "throughput");

  // Add the CFDFC variables to the global set of variables
  vars.cfdfcs[&cfdfc] = cfdfcVars;
  return success();
}

LogicalResult BufferPlacementMILP::createChannelVars() {
  // Create a set of variables for each channel in the circuit
  for (auto [idx, channelAndProps] : llvm::enumerate(channels)) {
    auto &channel = channelAndProps.first;

    // Construct a suffix for all variable names
    std::string suffix = "_" + getChannelName(channel);

    // Create a Gurobi variable of the given type and name
    auto createVar = [&](char type, const std::string &name) {
      return model.addVar(0, GRB_INFINITY, 0.0, type, name + suffix);
    };

    // Create the set of variables for the channel
    ChannelVars channelVars;
    channelVars.tPathIn = createVar(GRB_CONTINUOUS, "tPathIn");
    channelVars.tPathOut = createVar(GRB_CONTINUOUS, "tPathOut");
    channelVars.tElasIn = createVar(GRB_CONTINUOUS, "tElasIn");
    channelVars.tElasOut = createVar(GRB_CONTINUOUS, "tElasOut");
    channelVars.bufPresent = createVar(GRB_BINARY, "bufPresent");
    channelVars.bufIsOpaque = createVar(GRB_BINARY, "bufIsOpaque");
    channelVars.bufNumSlots = createVar(GRB_INTEGER, "bufNumSlots");

    vars.channels[channel] = channelVars;
  }
  return success();
}

LogicalResult
BufferPlacementMILP::addCustomChannelConstraints(ValueRange customChannels) {
  for (Value channel : customChannels) {
    ChannelVars &chVars = vars.channels[channel];
    ChannelBufProps &props = channels[channel];

    if (props.minOpaque > 0) {
      // Force the MILP to use opaque slots
      model.addConstr(chVars.bufIsOpaque == 1, "custom_forceOpaque");
      if (props.minTrans > 0) {
        // If the properties ask for both opaque and transaprent slots, let
        // opaque slots take over. Transparents slots will be placed "manually"
        // from the total number of slots indicated by the MILP's result
        size_t idx;
        Operation *producer = getChannelProducer(channel, &idx);
        assert(producer && "channel producer must exist");
        producer->emitWarning()
            << "Outgoing channel " << idx
            << " requests placement of at least one transparent and at least "
               "one opaque slot on the channel, which the MILP does not "
               "formally support. To honor the properties, the MILP will be "
               "configured to only place opaque slots, some of which will be "
               "converted to transparent slots when parsing the MILP's "
               "solution.";
        unsigned minTotalSlots = props.minOpaque + props.minTrans;
        model.addConstr(chVars.bufNumSlots >= minTotalSlots,
                        "custom_minOpaqueAndTrans");
      } else
        // Force the MILP to place a minimum number of opaque slots
        model.addConstr(chVars.bufNumSlots >= props.minOpaque,
                        "custom_minOpaque");
    } else if (props.minTrans > 0)
      // Force the MILP to place a minimum number of transparent slots
      model.addConstr(chVars.bufNumSlots >= props.minTrans + chVars.bufIsOpaque,
                      "custom_minTrans");
    if (props.minOpaque + props.minTrans > 0)
      model.addConstr(chVars.bufPresent == 1, "custom_forceBuffers");

    // Set a maximum number of slots to be placed
    if (props.maxOpaque.has_value()) {
      if (*props.maxOpaque == 0)
        // Force the MILP to use transparent slots
        model.addConstr(chVars.bufIsOpaque == 0, "custom_forceTransparent");
      if (props.maxTrans.has_value()) {
        // Force the MILP to use a maximum number of slots
        unsigned maxSlots = *props.maxTrans + *props.maxOpaque;
        if (maxSlots == 0) {
          model.addConstr(chVars.bufPresent == 0, "custom_noBuffers");
          model.addConstr(chVars.bufNumSlots == 0, "custom_noSlots");
        } else
          model.addConstr(chVars.bufNumSlots <= maxSlots, "custom_maxSlots");
      }
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
    model.addConstr(t1 <= targetPeriod, "path_channelInPeriod");
    // Arrival time at channel's output must be lower than target clock period
    model.addConstr(t2 <= targetPeriod, "path_channelOutPeriod");
    // If there isn't an opaque buffer on the channel, arrival time at channel's
    // output must be greater than at channel's input
    model.addConstr(t2 >= t1 - maxPeriod * chVars.bufIsOpaque,
                    "path_opaqueChannel");
  }

  // Add path constraints for units
  for (Operation *op : pathUnits) {
    double latency;
    if (failed(timingDB.getLatency(op, latency)))
      latency = 0.0;

    if (latency == 0.0) {
      double dataDelay;
      if (failed(timingDB.getTotalDelay(op, SignalType::DATA, dataDelay)))
        dataDelay = 0.0;

      // The unit is not pipelined, add a path constraint for each input/output
      // port pair in the unit
      forEachIOPair(op, [&](Value in, Value out) {
        GRBVar &tInPort = vars.channels[in].tPathOut;
        GRBVar &tOutPort = vars.channels[out].tPathIn;
        // Arrival time at unit's output port must be greater than arrival
        // time at unit's input port + the unit's combinational data delay
        model.addConstr(tOutPort >= tInPort + dataDelay, "path_combDelay");
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
          inPortDelay = 0.0;

        GRBVar &tInPort = vars.channels[inChannel].tPathOut;
        // Arrival time at unit's input port + input port delay must be less
        // than the target clock period
        model.addConstr(tInPort + inPortDelay <= targetPeriod, "path_inDelay");
      }

      // Output port constraints
      for (OpResult outChannel : op->getResults()) {
        if (!vars.channels.contains(outChannel))
          continue;

        double outPortDelay;
        if (failed(timingDB.getPortDelay(op, SignalType::DATA, PortType::OUT,
                                         outPortDelay)))
          outPortDelay = 0.0;

        GRBVar &tOutPort = vars.channels[outChannel].tPathIn;
        // Arrival time at unit's output port is equal to the output port delay
        model.addConstr(tOutPort == outPortDelay, "path_outDelay");
      }
    }
  }
  return success();
}

LogicalResult BufferPlacementMILP::addElasticityConstraints(
    ValueRange elasticChannels, ArrayRef<Operation *> elasticUnits) {
  // Upper bound for the longest rigid path
  unsigned cstCoef = std::distance(funcInfo.funcOp.getOps().begin(),
                                   funcInfo.funcOp.getOps().end()) +
                     2;

  // Add elasticity constraints for channels
  for (Value channel : elasticChannels) {
    ChannelVars &chVars = vars.channels[channel];
    GRBVar &tIn = chVars.tElasIn;
    GRBVar &tOut = chVars.tElasOut;
    GRBVar &present = chVars.bufPresent;
    GRBVar &opaque = chVars.bufIsOpaque;
    GRBVar &numSlots = chVars.bufNumSlots;

    // If there is an opaque buffer on the channel, the channel elastic
    // arrival time at the ouput must be greater than at the input (breaks
    // cycles!)
    model.addConstr(tOut >= tIn - cstCoef * opaque, "elastic_cycle");
    // If there is an opaque buffer, there must be at least one slot
    model.addConstr(numSlots >= opaque, "elastic_slots");
    // If there is at least one slot, there must be a buffer
    model.addConstr(present >= 0.01 * numSlots, "elastic_present");
  }

  // Add an elasticity constraint for every input/output port pair in the
  // elastic units
  for (Operation *op : elasticUnits) {
    forEachIOPair(op, [&](Value in, Value out) {
      GRBVar &tInPort = vars.channels[in].tElasOut;
      GRBVar &tOutPort = vars.channels[out].tElasIn;
      // The elastic arrival time at the output port must be at least one
      // greater than at the input port
      model.addConstr(tOutPort >= 1 + tInPort, "elastic_unitTime");
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

    assert(vars.channels.contains(channel) && "unknown channel");
    ChannelVars &chVars = vars.channels[channel];
    GRBVar &retSrc = cfdfcVars.units[srcOp].retOut;
    GRBVar &retDst = cfdfcVars.units[dstOp].retIn;
    GRBVar &bufIsOpaque = chVars.bufIsOpaque;
    GRBVar &bufNumSlots = chVars.bufNumSlots;
    GRBVar &throughput = cfdfcVars.throughput;
    unsigned backedge = cfdfc.backedges.contains(channel) ? 1 : 0;

    /// TODO: The legacy implementation does not add any constraints here for
    /// the input channel to select operations that is less frequently
    /// executed. Temporarily, emulate the same behavior obtained from passing
    /// our DOTs to the old buffer pass by assuming the "true" input is always
    /// the least executed one
    if (arith::SelectOp selOp = dyn_cast<arith::SelectOp>(dstOp))
      if (channel == selOp.getTrueValue())
        continue;

    // If the channel isn't a backedge, its throughput equals the difference
    // between the fluid retiming of tokens at its endpoints. Otherwise, it is
    // one less than this difference
    model.addConstr(chThroughput - backedge == retDst - retSrc,
                    "throughput_channelRetiming");
    // If there is an opaque buffer, the CFDFC throughput cannot exceed the
    // channel throughput. If there is not, the CFDFC throughput can exceed
    // the channel thoughput by 1
    model.addConstr(throughput - chThroughput + bufIsOpaque <= 1,
                    "throughput_cfdfc");
    // If there is an opaque buffer, the summed channel and CFDFC throughputs
    // cannot exceed the number of buffer slots. If there is not, the combined
    // throughput can exceed the number of slots by 1
    model.addConstr(chThroughput + throughput + bufIsOpaque - bufNumSlots <= 1,
                    "throughput_combined");
    // The channel's throughput cannot exceed the number of buffer slots
    model.addConstr(chThroughput <= bufNumSlots, "throughput_channel");
  }

  // Add a constraint for each pipelined CFDFC unit
  for (auto &[op, unitVars] : cfdfcVars.units) {
    double latency;
    if (failed(timingDB.getLatency(op, latency)) || latency == 0.0)
      continue;

    GRBVar &retIn = unitVars.retIn;
    GRBVar &retOut = unitVars.retOut;
    GRBVar &throughput = cfdfcVars.throughput;
    // The fluid retiming of tokens across the non-combinational unit must
    // be the same as its latency multiplied by the CFDFC's throughput
    model.addConstr(throughput * latency == retOut - retIn,
                    "through_unitRetiming");
  }
  return success();
}

LogicalResult BufferPlacementMILP::addObjective() {
  // Compute the total number of executions over all channels
  unsigned totalExecs = 0;
  for (auto &[channel, _] : vars.channels)
    totalExecs += getChannelNumExecs(channel);
  LLVM_DEBUG(llvm::errs() << "Total number of channel executions is "
                          << totalExecs << "\n";);

  // Create the expression for the MILP objective
  GRBLinExpr objective;

  // For each CFDFC, add a throughput contribution to the objective, weighted
  // by the "importance" of the CFDFC
  double maxCoefCFDFC = 0.0;
  if (totalExecs != 0) {
    for (auto &[cfdfc, cfdfcVars] : vars.cfdfcs) {
      if (!funcInfo.cfdfcs[cfdfc])
        continue;
      double coef = cfdfc->channels.size() * cfdfc->numExecs /
                    static_cast<double>(totalExecs);
      objective += coef * cfdfcVars.throughput;
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
  for (auto &[channel, chVar] : vars.channels) {
    objective -= maxCoefCFDFC * bufPenaltyMul * chVar.bufPresent;
    objective -= maxCoefCFDFC * slotPenaltyMul * chVar.bufNumSlots;
  }

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

  assert(result.numTrans >= numTransToDeduct &&
         "not enough transparent slots were placed, the MILP was likely "
         "incorrectly configured");
  assert(result.numOpaque >= numOpaqueToDeduct &&
         "not enough opaque slots were placed, the MILP was likely "
         "incorrectly configured");
  result.numTrans -= numTransToDeduct;
  result.numOpaque -= numOpaqueToDeduct;
}

std::string BufferPlacementMILP::getChannelName(Value channel) {
  Operation *consumer = *channel.getUsers().begin();
  if (BlockArgument arg = dyn_cast<BlockArgument>(channel)) {
    return "arg" + std::to_string(arg.getArgNumber()) + "_" +
           nameUniquer[consumer];
  }
  OpResult res = dyn_cast<OpResult>(channel);
  return nameUniquer[res.getDefiningOp()] + "_" +
         std::to_string(res.getResultNumber()) + "_" + nameUniquer[consumer];
}

void BufferPlacementMILP::forEachIOPair(
    Operation *op, const std::function<void(Value, Value)> &callback) {
  for (Value opr : op->getOperands())
    if (!isa<MemRefType>(opr.getType()))
      for (OpResult res : op->getResults())
        if (!isa<MemRefType>(res.getType()))
          callback(opr, res);
}

unsigned BufferPlacementMILP::getChannelNumExecs(Value channel) {
  Operation *srcOp = channel.getDefiningOp();
  Operation *dstOp = *channel.getUsers().begin();
  if ((srcOp && !getLogicBB(srcOp).has_value()) ||
      !getLogicBB(dstOp).has_value())
    // Channels to/from an operation outside basic blocks are "never executed"
    return 0;

  // A block is the exit if it has no successors
  auto isExit = [&](unsigned bb) -> bool {
    return llvm::all_of(funcInfo.archs, [&](experimental::ArchBB &arch) {
      return arch.srcBB != bb;
    });
  };

  // Iterate over all CFDFCs which contain the channel to determine its total
  // number of executions. Backedges are executed one less time than "forward
  // edges" since they are only taken between executions of the cycle the CFDFC
  // represents
  unsigned srcBB = srcOp ? *getLogicBB(srcOp) : ENTRY_BB;
  unsigned dstBB = *getLogicBB(dstOp);
  if (llvm::isa_and_nonnull<handshake::BranchOp,
                            handshake::ConditionalBranchOp>(srcOp) &&
      isa<handshake::MergeLikeOpInterface>(dstOp)) {
    for (experimental::ArchBB &arch : funcInfo.archs)
      if (arch.srcBB == srcBB && arch.dstBB == dstBB)
        return arch.numTrans;
    llvm_unreachable("no arch corresponds to transition");
  }

  unsigned numExec = 0;
  if (isExit(srcBB)) {
    // Accumulate number of transitions from predecessors
    for (experimental::ArchBB &arch : funcInfo.archs)
      if (arch.dstBB == srcBB)
        numExec += arch.numTrans;
    return numExec;
  }
  // Accumulate number of transitions to successors
  for (experimental::ArchBB &arch : funcInfo.archs)
    if (arch.srcBB == srcBB)
      numExec += arch.numTrans;
  return numExec;
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
