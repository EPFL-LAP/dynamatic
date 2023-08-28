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

/// Determines whether a channel is connected to a memory interface.
static bool connectToMemoryInterface(Value channel) {
  Operation *srcOp = channel.getDefiningOp();
  Operation *dstOp = *(channel.getUsers().begin());
  return (isa_and_nonnull<handshake::MemoryControllerOp>(srcOp)) ||
         isa<handshake::MemoryControllerOp>(dstOp);
}

BufferPlacementMILP::BufferPlacementMILP(
    handshake::FuncOp funcOp, llvm::MapVector<CFDFC *, bool> &cfdfcs,
    std::map<std::string, UnitInfo> &unitInfo, double targetPeriod,
    double maxPeriod, GRBEnv &env, double timeLimit)
    : targetPeriod(targetPeriod), maxPeriod(maxPeriod), funcOp(funcOp),
      cfdfcs(cfdfcs), unitInfo(unitInfo),
      numUnits(std::distance(funcOp.getOps().begin(), funcOp.getOps().end())),
      model(GRBModel(env)) {
  // Set a time limit for the MILP
  model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit);

  auto addChannelWithProps = [&](Value channel) -> LogicalResult {
    ChannelBufProps props;
    if (failed(getFPGA20BufProps(channel, props))) {
      unsatisfiable = true;
      return failure();
    }
    channels[channel] = props;
    return success();
  };

  /// NOTE: what do we do with the other function arguments? Should we add their
  /// respective channels to? It would make some of the code down the line
  /// simpler.

  Value start = funcOp.front().getArguments().back();
  assert(start && "last function argument must be start");
  for (Operation *op : start.getUsers())
    for (Value opr : op->getOperands())
      if (failed(addChannelWithProps(opr)))
        return;
  for (Operation &op : funcOp.getOps())
    for (OpResult res : op.getResults())
      if (failed(addChannelWithProps(res)))
        return;
}

bool BufferPlacementMILP::arePlacementConstraintsSatisfiable() {
  return !unsatisfiable;
}

LogicalResult BufferPlacementMILP::setup() {
  createVars();
  addCustomChannelConstraints();

  /// NOTE: I don't think that filtering out channels connecting to memory
  /// interfaces should be necessary here, as the fact that these channels are
  /// unbufferizable should already be encoded in the specific buffering
  /// properties of these channels

  // Add path constraints over the entire circuit (all channels and units)
  std::vector<Value> allChannels(channels.size());
  for (auto &[channel, _] : channels)
    if (!connectToMemoryInterface(channel))
      allChannels.push_back(channel);
  std::vector<Operation *> allUnits;
  for (Operation &op : funcOp.getOps())
    allUnits.push_back(&op);
  addPathConstraints(allChannels, allUnits);

  // Add elasticity constraints over the enture circuit (all channels and units)
  addElasticityConstraints(allChannels, allUnits);

  // Add throughput constraints over each CFDFC that was marked to be optimized
  for (auto &[cfdfc, _] : vars.cfdfcs)
    if (cfdfcs[cfdfc])
      addThroughputConstraints(*cfdfc);

  // Finally, add the MILP objective
  addObjective();
  return success();
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
  for (auto &[ch, chVarMap] : vars.channels) {
    if (chVarMap.bufPresent.get(GRB_DoubleAttr_X) > 0) {
      PlacementResult result;
      result.numSlots = static_cast<unsigned>(
          chVarMap.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
      result.opaque = chVarMap.bufIsOpaque.get(GRB_DoubleAttr_X) > 0;
      placement[ch] = result;
    }
  }
  return success();
}

void BufferPlacementMILP::createVars() {
  for (auto [idx, cfdfcAndOpt] : llvm::enumerate(cfdfcs))
    createCFDFCVars(*cfdfcAndOpt.first, idx);
  createChannelVars();

  // Update the model before returning so that these variables can be referenced
  // safely during the rest of model creation
  model.update();
}

void BufferPlacementMILP::createCFDFCVars(CFDFC &cfdfc, unsigned uid) {
  std::string cfdfcID = std::to_string(uid);
  CFDFCVars cfdfcVars;

  // Create a set of variables for each CFDFC unit
  for (auto [idx, unit] : llvm::enumerate(cfdfc.units)) {
    std::string unitID = std::to_string(idx);

    // Create the two unit variables
    UnitVars unitVar;
    std::string unitName = getOperationShortStrName(unit);
    unitVar.retIn =
        model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                     "mg" + cfdfcID + "_inRetimeTok_" + unitName + unitID);
    if (getUnitLatency(unit, unitInfo) == 0.0)
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
    std::string srcName = srcOp ? getOperationShortStrName(srcOp) : "arg";
    std::string dstName = getOperationShortStrName(dstOp);
    std::string varName = "mg" + cfdfcID + "_" + srcName + "_" + dstName + "_" +
                          std::to_string(idx);
    // Add a variable for the channel's throughput
    cfdfcVars.channelThroughputs[channel] = model.addVar(
        0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "throughput_" + varName);
  }

  // Create a variable for the CFDFC's throughput
  cfdfcVars.thoughput = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                     "cfdfcThroughput_" + cfdfcID);

  // Add the CFDFC variables to the global set of variables
  vars.cfdfcs[&cfdfc] = cfdfcVars;
}

void BufferPlacementMILP::createChannelVars() {
  // Create a set of variables for each channel in the circuit
  for (auto [idx, channelAndProps] : llvm::enumerate(channels)) {
    auto &channel = channelAndProps.first;

    // Construct a suffix for all variable names
    Operation *srcOp = channel.getDefiningOp();
    Operation *dstOp = *channel.getUsers().begin();
    std::string srcOpName = srcOp ? getOperationShortStrName(srcOp) : "arg";
    std::string dstOpName = getOperationShortStrName(dstOp);
    std::string suffix =
        "_" + srcOpName + "_" + dstOpName + "_" + std::to_string(idx);

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
}

void BufferPlacementMILP::addCustomChannelConstraints() {
  for (auto &[channel, chVars] : vars.channels) {
    if (channels[channel].minOpaque > 0) {
      // Set a minimum number of slots to be placed
      model.addConstr(chVars.bufNumSlots >= channels[channel].minOpaque);
      model.addConstr(chVars.bufIsOpaque >= 0);
    } else if (channels[channel].minTrans > 0) {
      model.addConstr(chVars.bufNumSlots >= channels[channel].minTrans);
      // model.addConstr(chVars.bufIsOp <= 0);
    }

    // Set a maximum number of slots to be placed
    if (channels[channel].maxOpaque.has_value())
      model.addConstr(chVars.bufNumSlots <=
                      channels[channel].maxOpaque.value());
    if (channels[channel].maxTrans.has_value())
      model.addConstr(chVars.bufNumSlots <= channels[channel].maxTrans.value());
  }
}

void BufferPlacementMILP::addPathConstraints(ValueRange pathChannels,
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
    double unitDelay = getCombinationalDelay(op, unitInfo, "data");
    if (getUnitLatency(op, unitInfo) == 0.0) {
      // The unit is not pipelined, add a path constraint for each input/output
      // port pair in the unit
      forEachIOPair(op, [&](Value in, Value out) {
        GRBVar &tInPort = vars.channels[in].tPathOut;
        GRBVar &tOutPort = vars.channels[out].tPathIn;
        // Arrival time at unit's output port must be greater than arrival time
        // at unit's input port + the unit's combinational delay
        model.addConstr(tOutPort >= tInPort + unitDelay);
      });
    } else {
      // The unit is pipelined, add a constraint for every of the unit's inputs
      // and every of the unit's output ports

      // Input port constraints
      for (Value inChannel : op->getOperands()) {
        if (!vars.channels.contains(inChannel))
          continue;

        std::string out = "out";
        double inPortDelay = getPortDelay(inChannel, unitInfo, out);
        GRBVar &tInPort = vars.channels[inChannel].tPathOut;
        // Arrival time at unit's input port + input port delay must be less
        // than the target clock period
        model.addConstr(tInPort + inPortDelay <= targetPeriod);
      }

      // Output port constraints
      for (OpResult outChannel : op->getResults()) {
        if (!vars.channels.contains(outChannel))
          continue;

        std::string in = "in";
        double outPortDelay = getPortDelay(outChannel, unitInfo, in);
        GRBVar &tOutPort = vars.channels[outChannel].tPathIn;
        // Arrival time at unit's output port is equal to the output port delay
        model.addConstr(tOutPort == outPortDelay);
      }
    }
  }
}

void BufferPlacementMILP::addElasticityConstraints(
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
}

void BufferPlacementMILP::addThroughputConstraints(CFDFC &cfdfc) {
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
    if (double latency = getUnitLatency(op, unitInfo); latency != 0.0) {
      GRBVar &retIn = unitVars.retIn;
      GRBVar &retOut = unitVars.retOut;
      GRBVar &throughput = cfdfcVars.thoughput;
      // The fluid retiming of tokens across the non-combinational unit must be
      // the same as its latency multiplied by the CFDFC's throughput
      model.addConstr(throughput * latency == retOut - retIn);
    }
  }
}

void BufferPlacementMILP::addObjective() {
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
}

LogicalResult BufferPlacementMILP::getFPGA20BufProps(Value channel,
                                                     ChannelBufProps &props) {
  Operation *srcOp = channel.getDefiningOp();
  // Skip channels from function arguments
  if (!srcOp)
    return success();

  Operation *dstOp = *(channel.getUsers().begin());

  // Merges with more than one input should have at least a transparent slot at
  // their output
  if (isa<handshake::MergeOp>(srcOp) && srcOp->getNumOperands() > 1)
    props.minTrans = 1;

  /// TODO: make the least frequently executed input data channel of select
  /// operations unbufferizable, now it's just ethe false input by default.
  if (auto selectOp = dyn_cast<arith::SelectOp>(dstOp)) {
    if (selectOp.getFalseValue() == channel) {
      props.maxTrans = 0;
      props.minOpaque = 0;
    }
  }

  // Channels connected to memory interfaces are not bufferizable
  if (isa<handshake::MemoryControllerOp>(srcOp) ||
      isa<handshake::MemoryControllerOp>(dstOp)) {
    props.maxOpaque = 0;
    props.maxTrans = 0;
  }

  // Combine channel-specific rules to description of internal unit buffers
  std::string srcName = srcOp->getName().getStringRef().str();
  std::string dstName = dstOp->getName().getStringRef().str();
  if (unitInfo.count(srcName) > 0) {
    props.minTrans += unitInfo[srcName].outPortTransBuf;
    props.minOpaque += unitInfo[srcName].outPortOpBuf;
  }
  if (unitInfo.count(dstName) > 0) {
    props.minTrans += unitInfo[dstName].inPortTransBuf;
    props.minOpaque += unitInfo[dstName].inPortOpBuf;
  }

  /// TODO: this verification is at the very least incomplete. We need to check
  /// whether the maximum number of allowed slots of each type isn't strictly
  /// lower than the number of slots inside the unit's IO port that the channel
  /// connects to. The logic also seems weird, but may be due to the fact that
  /// the MILP cannot represent a channel where X transparent slots and Y opaque
  /// slots must be placed
  if (props.minTrans > 0 && props.minOpaque > 0)
    return failure(); // cannot satisfy the constraint
  return success();
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
