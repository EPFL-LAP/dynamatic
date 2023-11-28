//===- FPGA20Buffers.cpp - FPGA'20 buffer placement -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements FPGA'20 smart buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Analysis/NameAnalysis.h"
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

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace circt;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpga20;

FPGA20Buffers::FPGA20Buffers(FuncInfo &funcInfo, const TimingDatabase &timingDB,
                             GRBEnv &env, Logger *logger, double targetPeriod,
                             double maxPeriod, bool legacyPlacement)
    : BufferPlacementMILP(funcInfo, timingDB, env, logger),
      targetPeriod(targetPeriod), maxPeriod(maxPeriod),
      legacyPlacement(legacyPlacement) {
  if (status == MILPStatus::UNSAT_PROPERTIES)
    return;
  if (succeeded(setup()))
    status = BufferPlacementMILP::MILPStatus::READY;
}

LogicalResult
FPGA20Buffers::getPlacement(DenseMap<Value, PlacementResult> &placement) {
  if (status != MILPStatus::OPTIMIZED) {
    std::stringstream ss;
    ss << status;
    return funcInfo.funcOp->emitError()
           << "Buffer placements cannot be extracted from MILP (reason: "
           << ss.str() << ").";
  }

  // Iterate over all channels in the circuit
  for (auto &[value, channelVars] : vars.channels) {
    if (channelVars.bufPresent.get(GRB_DoubleAttr_X) == 0)
      continue;

    // Extract number and type of slots from the MILP solution, as well as
    // channel-specific buffering properties
    unsigned numSlotsToPlace = static_cast<unsigned>(
        channelVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    bool placeOpaque = channelVars.bufIsOpaque.get(GRB_DoubleAttr_X) > 0;
    ChannelBufProps &props = channels[value];

    PlacementResult result;
    if (placeOpaque && numSlotsToPlace > 0) {
      if (legacyPlacement) {
        // Satisfy the transparent slots requirement, all other slots are opaque
        result.numTrans = props.minTrans;
        result.numOpaque = numSlotsToPlace - props.minTrans;
      } else {
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
      }
    } else {
      // All slots should be transparent
      result.numTrans = numSlotsToPlace;
    }

    Channel channel(value);
    deductInternalBuffers(channel, result);
    placement[value] = result;
  }

  if (logger)
    logResults(placement);
  return success();
}

LogicalResult FPGA20Buffers::setup() {
  if (failed(createVars()))
    return failure();

  // Aggregate all units in a vector
  std::vector<Operation *> allUnits;
  for (Operation &op : funcInfo.funcOp.getOps())
    allUnits.push_back(&op);

  // Aggregate all channels in a vector
  std::vector<Value> allChannels;
  for (auto &[channel, _] : channels)
    allChannels.push_back(channel);

  // Exclude channels that connect to memory interfaces for path and elasticity
  // constraints
  std::vector<Value> nonMemChannels;
  llvm::copy_if(
      allChannels, std::back_inserter(nonMemChannels), [&](Value val) {
        return !val.getDefiningOp<handshake::MemoryOpInterface>() &&
               !isa<handshake::MemoryOpInterface>(*val.getUsers().begin());
      });

  // Create custom, path, and elasticity constraints
  if (failed(addCustomChannelConstraints(allChannels)) ||
      failed(addPathConstraints(nonMemChannels, allUnits)) ||
      failed(addElasticityConstraints(nonMemChannels, allUnits)))
    return failure();

  // Add throughput constraints over each CFDFC that was marked to be optimized
  for (auto &[cfdfc, _] : vars.cfdfcs)
    if (funcInfo.cfdfcs[cfdfc])
      if (failed(addThroughputConstraints(*cfdfc)))
        return failure();

  // Finally, add the MILP objective
  return addObjective();
}

LogicalResult FPGA20Buffers::createVars() {
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

LogicalResult FPGA20Buffers::createCFDFCVars(CFDFC &cfdfc, unsigned uid) {
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
    std::string unitName = getUniqueName(unit) + std::to_string(idx);
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
    cfdfcVars.channelThroughputs[channel] = createVar(
        prefix + "throughput_" + getUniqueName(*channel.getUses().begin()));

  // Create a variable for the CFDFC's throughput
  cfdfcVars.throughput = createVar(prefix + "throughput");

  // Add the CFDFC variables to the global set of variables
  vars.cfdfcs[&cfdfc] = cfdfcVars;
  return success();
}

LogicalResult FPGA20Buffers::createChannelVars() {
  // Create a set of variables for each channel in the circuit
  for (auto [idx, channelAndProps] : llvm::enumerate(channels)) {
    auto &channel = channelAndProps.first;

    // Construct a suffix for all variable names
    std::string suffix = "_" + getUniqueName(*channel.getUses().begin());

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
FPGA20Buffers::addCustomChannelConstraints(ValueRange customChannels) {
  for (Value channel : customChannels) {
    ChannelVars &chVars = vars.channels[channel];
    ChannelBufProps &props = channels[channel];

    if (props.minOpaque > 0) {
      // Force the MILP to use opaque slots
      model.addConstr(chVars.bufIsOpaque == 1, "custom_forceOpaque");
      if (props.minTrans > 0) {
        // If the properties ask for both opaque and transparent slots, let
        // opaque slots take over. Transparents slots will be placed "manually"
        // from the total number of slots indicated by the MILP's result
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
FPGA20Buffers::addPathConstraints(ValueRange pathChannels,
                                  ArrayRef<Operation *> pathUnits) {
  // Manually get the timing model for buffers
  const TimingModel *bufModel = timingDB.getModel(OperationName(
      handshake::BufferOp::getOperationName(), funcInfo.funcOp->getContext()));
  double bigCst = targetPeriod * 10;

  // Add path constraints for channels
  for (Value channel : pathChannels) {

    // Get delays for a buffer that would be placed on this channel
    double inBufDelay = 0.0, outBufDelay = 0.0, dataBufDelay = 0.0;
    if (bufModel) {
      Type channelType = channel.getType();
      unsigned bitwidth = 0;
      if (isa<IntegerType, FloatType>(channelType))
        bitwidth = channelType.getIntOrFloatBitWidth();
      if (failed(bufModel->inputModel.dataDelay.getCeilMetric(bitwidth,
                                                              inBufDelay)) ||
          failed(bufModel->outputModel.dataDelay.getCeilMetric(bitwidth,
                                                               outBufDelay)) ||
          failed(bufModel->dataDelay.getCeilMetric(bitwidth, dataBufDelay)))
        return failure();
      // Add the input and output port delays to the total buffer delay
      dataBufDelay += inBufDelay + outBufDelay;
    }

    ChannelVars &chVars = vars.channels[channel];
    ChannelBufProps &props = channels[channel];
    GRBVar &t1 = chVars.tPathIn;
    GRBVar &t2 = chVars.tPathOut;
    GRBVar &present = chVars.bufPresent;
    GRBVar &opaque = chVars.bufIsOpaque;

    // Arrival time at channel's input must be lower than target clock period
    model.addConstr(t1 + present * (props.inDelay + inBufDelay) <= targetPeriod,
                    "path_channelInPeriod");
    // Arrival time at channel's output must be lower than target clock period
    model.addConstr(t2 <= targetPeriod, "path_channelOutPeriod");

    // If there is an opaque buffer, arrival time at channel's output must be
    // greater than the delay between the buffer's internal register and the
    // post-buffer channel delay
    double bufToOutDelay = outBufDelay + props.outDelay;
    if (bufToOutDelay > 0)
      model.addConstr(opaque * bufToOutDelay <= t2, "path_opaqueChannel");

    // If there is a transparent buffer, arrival time at channel's output must
    // be greater than at channel's input (+ whole channel and buffer delay)
    double inToOutDelay = props.inDelay + dataBufDelay + props.outDelay;
    model.addConstr(t1 + inToOutDelay - bigCst * (opaque - present + 1) <= t2,
                    "path_transparentChannel");

    // If there are no buffers, arrival time at channel's output must be greater
    // than at channel's input (+ channel delay)
    model.addConstr(t1 + props.delay - bigCst * present <= t2,
                    "path_unbufferedChannel");
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

LogicalResult
FPGA20Buffers::addElasticityConstraints(ValueRange elasticChannels,
                                        ArrayRef<Operation *> elasticUnits) {
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

LogicalResult FPGA20Buffers::addThroughputConstraints(CFDFC &cfdfc) {
  CFDFCVars &cfdfcVars = vars.cfdfcs[&cfdfc];

  // Add a set of constraints for each CFDFC channel
  for (auto &[channel, chThroughput] : cfdfcVars.channelThroughputs) {

    // No throughput constraints on channels going to LSQ stores
    if (isa<handshake::LSQStoreOp>(*channel.getUsers().begin()))
      continue;

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

LogicalResult FPGA20Buffers::addObjective() {
  // Compute the total number of executions over all channels
  unsigned totalExecs = 0;
  for (auto &[channel, _] : vars.channels)
    totalExecs += getChannelNumExecs(channel);

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

unsigned FPGA20Buffers::getChannelNumExecs(Value channel) {
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

void FPGA20Buffers::logResults(DenseMap<Value, PlacementResult> &placement) {
  assert(logger && "no logger was provided");
  mlir::raw_indented_ostream &os = **logger;

  os << "# ========================== #\n";
  os << "# Buffer Placement Decisions #\n";
  os << "# ========================== #\n\n";

  for (auto &[value, channelVars] : vars.channels) {
    if (channelVars.bufPresent.get(GRB_DoubleAttr_X) == 0)
      continue;

    // Extract number and type of slots
    unsigned numSlotsToPlace = static_cast<unsigned>(
        channelVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    bool placeOpaque = channelVars.bufIsOpaque.get(GRB_DoubleAttr_X) > 0;

    PlacementResult result = placement[value];
    ChannelBufProps &props = channels[value];

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
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfdfcs)) {
    auto [cf, cfVars] = cfdfcWithVars;
    double throughput = cfVars.throughput.get(GRB_DoubleAttr_X);
    os << "Throughput of CFDFC #" << idx << ": " << throughput << "\n";
  }

  os << "\n# =================== #\n";
  os << "# Channel Throughputs #\n";
  os << "# =================== #\n\n";

  // Log throughput of all channels in all CFDFCs
  for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfdfcs)) {
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
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
