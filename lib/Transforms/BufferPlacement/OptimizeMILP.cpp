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
#include "llvm/ADT/SmallVector.h"
#include <fstream>
#include <iostream>

using namespace circt;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

static unsigned getChannelFreq(Value channel,
                               llvm::MapVector<CFDFC *, bool> &cfdfcs) {
  Operation *srcOp = channel.getDefiningOp();
  Operation *dstOp = *channel.getUsers().begin();

  // if is a start node or end node, return 1
  if (!srcOp || !dstOp)
    return 1;

  if (isa<SinkOp, MemoryControllerOp>(dstOp) || isa<MemoryControllerOp>(srcOp))
    return 0;

  unsigned freq = 1;
  if (isBackEdge(srcOp, dstOp))
    freq = 0;
  //  execution times equals to the sum over all CFDFCs
  for (auto &[cfdfc, _] : cfdfcs)
    if (cfdfc->channels.contains(channel))
      freq += cfdfc->numExec;

  return freq;
}

/// Whether the path is considered to be covered in path and elasticity
/// constraints. Current version only consider mem_controller, future
/// version should take account of lsq and more operations.
static bool coverPath(Value channel) {
  Operation *srcOp = channel.getDefiningOp();
  Operation *dstOp = *(channel.getUsers().begin());
  // If both src operation and dst operation exist, and neither of them is
  // memory controller unit, the channel is covered.
  if (srcOp && dstOp)
    if (isa<MemoryControllerOp>(srcOp) || isa<MemoryControllerOp>(dstOp))
      return false;
  return true;
}

/// Create time path constraints over channels.
/// t1 is the input time of the channel, t2 is the output time of the channel.
static void createPathConstrs(GRBModel &model, GRBVar &t1, GRBVar &t2,
                              GRBVar &bufOp, double period,
                              double bufDelay = 0.0) {
  model.addConstr(t1 <= period);
  model.addConstr(t2 <= period);
  model.addConstr(t2 >= t1 - 2 * period * bufOp);
}

// create elasticity constraints w.r.t channels
static void createElasticityConstrs(GRBModel &model, GRBVar &t1, GRBVar &t2,
                                    GRBVar &bufOp, GRBVar &bufNSlots,
                                    GRBVar &hasBuf, unsigned cstCoef,
                                    double period) {
  model.addConstr(t2 >= t1 - cstCoef * bufOp);
  model.addConstr(bufNSlots >= bufOp);
  model.addConstr(hasBuf >= 0.01 * bufNSlots);
}

/// Throughput constraints over a channel
static void createThroughputConstrs(GRBModel &model, GRBVar &retSrc,
                                    GRBVar &retDst, GRBVar &thrptTok,
                                    GRBVar &thrpt, GRBVar &isOp,
                                    GRBVar &bufNSlots, const int tok) {
  model.addConstr(retSrc - retDst + thrptTok == tok);
  model.addConstr(thrpt + isOp - thrptTok <= 1);
  model.addConstr(thrptTok + thrpt + isOp - bufNSlots <= 1);
  model.addConstr(thrptTok - bufNSlots <= 0);
}

BufferPlacementMILP::BufferPlacementMILP(
    handshake::FuncOp funcOp, llvm::MapVector<CFDFC *, bool> &cfdfcs,
    std::map<std::string, UnitInfo> &unitInfo, double targetPeriod, GRBEnv &env,
    double timeLimit)
    : funcOp(funcOp), cfdfcs(cfdfcs), unitInfo(unitInfo),
      targetPeriod(targetPeriod),
      numUnits(std::distance(funcOp.getOps().begin(), funcOp.getOps().end())),
      model(GRBModel(env)) {
  // Set a time limit for the MILP
  model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit);

  auto addChannelProps = [&](Value channel) -> LogicalResult {
    ChannelBufProps props;
    if (failed(hardcodeBufProps(channel, props))) {
      unsatisfiable = true;
      return failure();
    }
    channels[channel] = props;
    return success();
  };

  /// TODO: (RamirezLucas) What to do about the other function argument?
  Value start = funcOp.front().getArguments().back();
  assert(start && "last function argument must be start");
  for (Operation *op : start.getUsers())
    for (Value opr : op->getOperands())
      if (failed(addChannelProps(opr)))
        return;
  for (Operation &op : funcOp.getOps())
    for (OpResult res : op.getResults())
      if (failed(addChannelProps(res)))
        return;

  // Compeltely create the MILP (variables, constraints, objective)
  setupMILP();
}

void BufferPlacementMILP::setupMILP() {
  initializeVars();
  addCustomChannelConstraints();
  addPathConstraints();
  addElasticityConstraints();
  addThroughputConstraints();
  addObjective();
}

/// Initialize the variables in the MILP model
void BufferPlacementMILP::initializeVars() {
  // create variables
  for (auto [cfdfcIdx, cfdfcAndOpt] : llvm::enumerate(cfdfcs)) {
    auto &[cfdfc, opt] = cfdfcAndOpt;
    vars.units[cfdfc] = DenseMap<Operation *, UnitVar>{};
    vars.channelThroughput[cfdfc] = DenseMap<Value, GRBVar>{};
    for (auto [unitIdx, unit] : llvm::enumerate(cfdfc->units)) {
      UnitVar unitVar;

      // init unit variables
      std::string unitName = getOperationShortStrName(unit);
      unitVar.retIn =
          model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                       "mg" + std::to_string(cfdfcIdx) + "_inRetimeTok_" +
                           unitName + std::to_string(unitIdx));
      if (getUnitLatency(unit, unitInfo) < 1e-10)
        unitVar.retOut = unitVar.retIn;
      else
        unitVar.retOut =
            model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                         "mg" + std::to_string(cfdfcIdx) + "_outRetimeTok_" +
                             unitName + std::to_string(unitIdx));
      vars.units[cfdfc][unit] = unitVar;
    }

    // If the CFDFC should not be optimized, stop here
    if (!opt)
      continue;

    // init channel variables w.r.t the optimized CFDFC
    for (auto [chInd, channel] : llvm::enumerate(cfdfc->channels)) {
      std::string srcName = "arg_start";
      std::string dstName = "arg_end";
      Operation *srcOp = channel.getDefiningOp();
      Operation *dstOp = *channel.getUsers().begin();
      // Define the channel variable names w.r.t to its connected units
      if (srcOp)
        srcName = getOperationShortStrName(srcOp);
      if (dstOp)
        dstName = getOperationShortStrName(dstOp);

      std::string chName = "mg" + std::to_string(cfdfcIdx) + "_" + srcName +
                           "_" + dstName + "_" + std::to_string(chInd);
      vars.channelThroughput[cfdfc][channel] =
          model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "thrpt_" + chName);
    }
  }
  model.update();

  // create channel vars
  for (auto [ind, channelAndProps] : llvm::enumerate(channels)) {
    auto &[channel, _] = channelAndProps;
    Operation *srcOp = channel.getDefiningOp();
    Operation *dstOp = *channel.getUsers().begin();

    if (!srcOp && !dstOp)
      continue;

    std::string srcOpName = "arg_input";
    std::string dstOpName = "arg_end";

    // Define the channel variable names w.r.t to its connected units
    if (srcOp)
      srcOpName = getOperationShortStrName(srcOp);
    if (dstOp)
      dstOpName = getOperationShortStrName(dstOp);

    // create channel variable
    ChannelVar channelVar;

    std::string chName =
        srcOpName + "_" + dstOpName + "_" + std::to_string(ind);

    channelVar.tDataIn = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                      "timePathIn_" + chName);
    channelVar.tDataOut = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                       "timePathOut_" + chName);

    channelVar.tElasIn = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                      "timeElasticIn_" + chName);
    channelVar.tElasOut = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                       "timeElasticOut_" + chName);

    channelVar.bufNSlots =
        model.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER, chName + "_bufNSlots");

    channelVar.hasBuf =
        model.addVar(0, GRB_INFINITY, 0.0, GRB_BINARY, chName + "_hasBuf");
    channelVar.bufIsOp =
        model.addVar(0, GRB_INFINITY, 0.0, GRB_BINARY, chName + "_bufIsOp");
    vars.channels[channel] = channelVar;
    model.update();
  }

  // Create variables representing the circuit's throughput
  for (auto [idx, cfdfcAndOpt] : llvm::enumerate(cfdfcs))
    vars.circuitThroughput[cfdfcAndOpt.first] = model.addVar(
        0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "thrpt" + std::to_string(idx));
}

/// Create constraints that is prerequisite for buffer placement
void BufferPlacementMILP::addCustomChannelConstraints() {
  for (auto &[ch, chVars] : vars.channels) {
    // set min value of the buffer
    if (channels[ch].minOpaque > 0) {
      model.addConstr(chVars.bufNSlots >= channels[ch].minOpaque);
      model.addConstr(chVars.bufIsOp >= 0);
    } else if (channels[ch].minTrans > 0) {
      model.addConstr(chVars.bufNSlots >= channels[ch].minTrans);
      // model.addConstr(chVars.bufIsOp <= 0);
    }

    // set max value of the buffer
    if (channels[ch].maxOpaque.has_value())
      model.addConstr(chVars.bufNSlots <= channels[ch].maxOpaque.value());

    if (channels[ch].maxTrans.has_value())
      model.addConstr(chVars.bufNSlots <= channels[ch].maxTrans.value());
  }
}

LogicalResult BufferPlacementMILP::hardcodeBufProps(Value channel,
                                                    ChannelBufProps &props) {
  Operation *srcOp = channel.getDefiningOp();
  Operation *dstOp = *(channel.getUsers().begin());

  // skip the channel that is the block argument
  if (!srcOp || !dstOp)
    return success();

  std::string srcName = srcOp->getName().getStringRef().str();
  std::string dstName = dstOp->getName().getStringRef().str();
  // set merge with multiple input to have at least one transparent buffer
  if (isa<handshake::MergeOp>(srcOp) && srcOp->getNumOperands() > 1)
    props.minTrans = 1;

  // TODO: set selectOp always select the frequent input
  if (isa<arith::SelectOp>(dstOp))
    if (dstOp->getOperand(2) == channel) {
      props.maxTrans = 0;
      props.minOpaque = 0;
    }

  if (isa<handshake::MemoryControllerOp>(srcOp) ||
      isa<handshake::MemoryControllerOp>(dstOp)) {
    props.maxOpaque = 0;
    props.maxTrans = 0;
  }

  // set channel buffer properties w.r.t to input file
  if (unitInfo.count(srcName) > 0) {
    props.minTrans += unitInfo[srcName].outPortTransBuf;
    props.minOpaque += unitInfo[srcName].outPortOpBuf;
  }

  if (unitInfo.count(dstName) > 0) {
    props.minTrans += unitInfo[dstName].inPortTransBuf;
    props.minOpaque += unitInfo[dstName].inPortOpBuf;
  }

  if (props.minTrans > 0 && props.minOpaque > 0)
    return failure(); // cannot satisfy the constraint
  return success();
}

/// Create constraints that describe the circuits behavior
void BufferPlacementMILP::addPathConstraints() {
  // Channel constraints
  for (auto &[channel, _] : channels) {
    // update the model to get the lower bound and upper bound of the vars
    model.update();
    if (!vars.channels.contains(channel) || !coverPath(channel))
      continue;

    auto chVars = vars.channels[channel];

    GRBVar &t1 = chVars.tDataIn;
    GRBVar &t2 = chVars.tDataOut;
    GRBVar &bufOp = chVars.bufIsOp;

    createPathConstrs(model, t1, t2, bufOp, targetPeriod);
  }

  // Units constraints
  for (auto &op : funcOp.getOps()) {
    double delayData = getCombinationalDelay(&op, unitInfo, "data");

    double latency = getUnitLatency(&op, unitInfo);
    if (latency == 0)
      // iterate all input port to all output port for a unit
      for (auto inChVal : op.getOperands()) {
        // Define variables w.r.t to input port
        if (!vars.channels.contains(inChVal))
          continue;

        GRBVar &tIn = vars.channels[inChVal].tDataOut;
        // GRBVar &tElasIn = channelVars[inChVal].tElasOut;
        for (auto outChVal : op.getResults()) {
          if (!vars.channels.contains(outChVal))
            continue;

          // Define variables w.r.t to output port
          GRBVar &tOut = vars.channels[outChVal].tDataIn;
          model.addConstr(tOut >= delayData + tIn);
        }
      }
    // if the unit is pipelined
    else {
      // Define constraints w.r.t to input port
      for (auto inChVal : op.getOperands()) {
        std::string out = "out";
        double inPortDelay = getPortDelay(inChVal, unitInfo, out);
        if (!vars.channels.contains(inChVal))
          continue;

        GRBVar &tIn = vars.channels[inChVal].tDataOut;
        model.addConstr(tIn <= targetPeriod - inPortDelay);
      }

      // Define constraints w.r.t to output port
      for (auto outChVal : op.getResults()) {
        std::string in = "in";
        double outPortDelay = getPortDelay(outChVal, unitInfo, in);

        if (!vars.channels.contains(outChVal))
          continue;

        GRBVar &tOut = vars.channels[outChVal].tDataIn;
        model.addConstr(tOut == outPortDelay);
      }
    }
  }
}

/// Create constraints that describe the circuits behavior
void BufferPlacementMILP::addElasticityConstraints() {
  // Channel constraints
  for (auto &[channel, _] : channels) {
    // update the model to get the lower bound and upper bound of the vars
    model.update();
    if (!vars.channels.contains(channel) || !coverPath(channel))
      continue;

    auto chVars = vars.channels[channel];

    GRBVar &tElas1 = chVars.tElasIn;
    GRBVar &tElas2 = chVars.tElasOut;

    GRBVar &bufOp = chVars.bufIsOp;
    GRBVar &bufNSlots = chVars.bufNSlots;
    GRBVar &hasBuf = chVars.hasBuf;

    createElasticityConstrs(model, tElas1, tElas2, bufOp, bufNSlots, hasBuf,
                            numUnits + 2, targetPeriod);
  }

  // Units constraints
  for (auto &op : funcOp.getOps()) {
    // iterate all input port to all output port for a unit
    for (auto inChVal : op.getOperands()) {
      if (!vars.channels.contains(inChVal))
        continue;

      // Define variables w.r.t to input port
      GRBVar &tElasIn = vars.channels[inChVal].tElasOut;
      for (auto outChVal : op.getResults()) {
        if (!vars.channels.contains(outChVal))
          continue;

        // Define variables w.r.t to output port
        GRBVar &tElasOut = vars.channels[outChVal].tElasIn;
        model.addConstr(tElasOut >= 1 + tElasIn);
      }
    }
  }
}

void BufferPlacementMILP::addThroughputConstraints() {
  for (auto &[cfdfc, throughputVars] : vars.channelThroughput) {
    for (Value ch : cfdfc->channels) {

      if (!throughputVars.contains(ch))
        continue;

      Operation *srcOp = ch.getDefiningOp();
      Operation *dstOp = *(ch.getUsers().begin());

      GRBVar &thrptTok = throughputVars[ch];
      int tok = isBackEdge(srcOp, dstOp) ? 1 : 0;
      GRBVar &retSrc = vars.units[cfdfc][srcOp].retOut;
      GRBVar &retDst = vars.units[cfdfc][dstOp].retIn;

      GRBVar &bufOp = vars.channels[ch].bufIsOp;
      GRBVar &bufNSlots = vars.channels[ch].bufNSlots;
      createThroughputConstrs(model, retSrc, retDst, thrptTok,
                              vars.circuitThroughput[cfdfc], bufOp, bufNSlots,
                              tok);
    }
  }
  for (auto &[cfdfc, unitVars] : vars.units)
    for (auto &[op, var] : unitVars) {
      GRBVar &retIn = var.retIn;
      GRBVar &retOut = var.retOut;
      double latency = getUnitLatency(op, unitInfo);
      if (latency > 0)
        model.addConstr(retOut - retIn ==
                        latency * vars.circuitThroughput[cfdfc]);
    }
}

void BufferPlacementMILP::addObjective() {
  GRBLinExpr objExpr;
  double lumbdaCoef1 = 1e-4;
  double lumbdaCoef2 = 1e-5;

  double totalFreq = 0.0;
  double highestCoef = 0.0;
  for (auto [channel, _] : vars.channels)
    totalFreq += static_cast<double>(getChannelFreq(channel, cfdfcs));

  for (auto &[cfdfc, throughputVar] : vars.circuitThroughput) {
    double coef = cfdfc->channels.size() * cfdfc->numExec / totalFreq;
    objExpr += coef * throughputVar;
    highestCoef = std::max(coef, highestCoef);
  }

  for (auto &[_, chVar] : vars.channels)
    objExpr -= highestCoef *
               (lumbdaCoef1 * chVar.hasBuf + lumbdaCoef2 * chVar.bufNSlots);

  model.setObjective(objExpr, GRB_MAXIMIZE);
}

LogicalResult
BufferPlacementMILP::optimize(DenseMap<Value, PlacementResult> &placement) {
  // Optimize the model, then check whether we found an optimal solution or
  // whether we reached the time limit
  model.optimize();
  if (!(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) ||
      (model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT &&
       model.get(GRB_DoubleAttr_ObjVal) > 0))
    return failure();

  // Fill in placement information
  for (auto &[ch, chVarMap] : vars.channels) {
    if (chVarMap.hasBuf.get(GRB_DoubleAttr_X) > 0) {
      PlacementResult result;
      result.numSlots =
          static_cast<unsigned>(chVarMap.bufNSlots.get(GRB_DoubleAttr_X) + 0.5);
      result.opaque = chVarMap.bufIsOp.get(GRB_DoubleAttr_X) > 0;
      placement[ch] = result;
    }
  }
  return success();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
