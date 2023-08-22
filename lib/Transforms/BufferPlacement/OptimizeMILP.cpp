//===- OptimizeMILP.cpp - optimize MILP model over CFDFC  -------*- C++ -*-===//
//
// This file implements the MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/OptimizeMILP.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include <fstream>
#include <iostream>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

static unsigned getChannelFreq(Value channel, std::vector<CFDFC> &cfdfcList) {
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
  for (auto cfdfc : cfdfcList)
    if (std::find(cfdfc.channels.begin(), cfdfc.channels.end(), channel) !=
        cfdfc.channels.end())
      freq += cfdfc.numExec;

  return freq;
}

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED

namespace {
/// Data structure to store the variables w.r.t to a unit(operation), including
/// whether it belongs to a CFDFC, and its retime variables.
struct UnitVar {
  bool select;
  GRBVar retIn, retOut;
};

/// Data structure to store the variables w.r.t to a channel(value), including
/// whether it belongs to a CFDFC, and its time, throughput, and buffer
/// placement decision.
struct ChannelVar {
  bool select;
  GRBVar tDataIn, tDataOut, tElasIn, tElasOut;
  GRBVar bufIsOp, bufNSlots, hasBuf;
};

} // namespace

/// Whether the path is considered to be covered in path and elasticity
/// constraints. Current version only consider mem_controller, future version
/// should take account of lsq and more operations.
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

/// Initialize the variables in the MILP model
static void
initVarsInMILP(handshake::FuncOp funcOp, GRBModel &modelBuf,
               std::vector<CFDFC> cfdfcList, std::vector<unsigned> cfdfcInds,
               std::vector<Value> &allChannels,
               std::vector<DenseMap<Operation *, UnitVar>> &unitVars,
               std::vector<DenseMap<Value, GRBVar>> &chThrptToks,
               DenseMap<Value, ChannelVar> &channelVars,
               std::map<std::string, UnitInfo> unitInfo) {
  // create variables
  for (auto [ind, cfdfc] : llvm::enumerate(cfdfcList)) {
    unitVars.emplace_back();
    chThrptToks.emplace_back();
    for (auto [unitInd, unit] : llvm::enumerate(cfdfc.units)) {
      UnitVar unitVar;

      // init unit variables
      std::string unitName = getOperationShortStrName(unit);
      unitVar.retIn =
          modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                          "mg" + std::to_string(ind) + "_inRetimeTok_" +
                              unitName + std::to_string(unitInd));
      if (getUnitLatency(unit, unitInfo) < 1e-10)
        unitVar.retOut = unitVar.retIn;
      else
        unitVar.retOut =
            modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                            "mg" + std::to_string(ind) + "_outRetimeTok_" +
                                unitName + std::to_string(unitInd));
      unitVars[ind][unit] = unitVar;
    }

    // init channel variables w.r.t the optimized CFDFC
    if (std::find(cfdfcInds.begin(), cfdfcInds.end(), ind) == cfdfcInds.end())
      continue;
    for (auto [chInd, channel] : llvm::enumerate(cfdfc.channels)) {
      std::string srcName = "arg_start";
      std::string dstName = "arg_end";
      Operation *srcOp = channel.getDefiningOp();
      Operation *dstOp = *channel.getUsers().begin();
      // Define the channel variable names w.r.t to its connected units
      if (srcOp)
        srcName = getOperationShortStrName(srcOp);
      if (dstOp)
        dstName = getOperationShortStrName(dstOp);

      std::string chName = "mg" + std::to_string(ind) + "_" + srcName + "_" +
                           dstName + "_" + std::to_string(chInd);
      chThrptToks[ind][channel] = modelBuf.addVar(
          0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "thrpt_" + chName);
    }
  }
  modelBuf.update();

  // create channel vars
  for (auto [ind, val] : llvm::enumerate(allChannels)) {
    Operation *srcOp = val.getDefiningOp();
    Operation *dstOp = *val.getUsers().begin();

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

    channelVar.tDataIn = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                         "timePathIn_" + chName);
    channelVar.tDataOut = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                          "timePathOut_" + chName);

    channelVar.tElasIn = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                         "timeElasticIn_" + chName);
    channelVar.tElasOut = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                          "timeElasticOut_" + chName);

    channelVar.bufNSlots = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER,
                                           chName + "_bufNSlots");

    channelVar.hasBuf =
        modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_BINARY, chName + "_hasBuf");
    channelVar.bufIsOp =
        modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_BINARY, chName + "_bufIsOp");
    channelVars[val] = channelVar;
    modelBuf.update();
  }
}

/// Create time path constraints over channels.
/// t1 is the input time of the channel, t2 is the output time of the channel.
static void createPathConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                              GRBVar &bufOp, double period,
                              double bufDelay = 0.0) {
  modelBuf.addConstr(t1 <= period);
  modelBuf.addConstr(t2 <= period);
  modelBuf.addConstr(t2 >= t1 - 2 * period * bufOp);
}

// create elasticity constraints w.r.t channels
static void createElasticityConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                                    GRBVar &bufOp, GRBVar &bufNSlots,
                                    GRBVar &hasBuf, unsigned cstCoef,
                                    double period) {
  modelBuf.addConstr(t2 >= t1 - cstCoef * bufOp);
  modelBuf.addConstr(bufNSlots >= bufOp);
  modelBuf.addConstr(hasBuf >= 0.01 * bufNSlots);
}

/// Throughput constraints over a channel
static void createThroughputConstrs(GRBModel &modelBuf, GRBVar &retSrc,
                                    GRBVar &retDst, GRBVar &thrptTok,
                                    GRBVar &thrpt, GRBVar &isOp,
                                    GRBVar &bufNSlots, const int tok) {
  modelBuf.addConstr(retSrc - retDst + thrptTok == tok);
  modelBuf.addConstr(thrpt + isOp - thrptTok <= 1);
  modelBuf.addConstr(thrptTok + thrpt + isOp - bufNSlots <= 1);
  modelBuf.addConstr(thrptTok - bufNSlots <= 0);
}

/// Create constraints that describe the circuits behavior
static void createModelPathConstraints(
    GRBModel &modelBuf, double targetCP, handshake::FuncOp &funcOp,
    std::vector<Value> &allChannels, DenseMap<Value, ChannelVar> &channelVars,
    std::map<std::string, UnitInfo> unitInfo) {
  // Channel constraints
  for (Value ch : allChannels) {
    // update the model to get the lower bound and upper bound of the vars
    modelBuf.update();
    if (!channelVars.contains(ch) || !coverPath(ch))
      continue;

    auto chVars = channelVars[ch];

    GRBVar &t1 = chVars.tDataIn;
    GRBVar &t2 = chVars.tDataOut;
    GRBVar &bufOp = chVars.bufIsOp;

    createPathConstrs(modelBuf, t1, t2, bufOp, targetCP);
  }

  // Units constraints
  for (auto &op : funcOp.getOps()) {
    double delayData = getCombinationalDelay(&op, unitInfo, "data");

    double latency = getUnitLatency(&op, unitInfo);
    if (latency == 0)
      // iterate all input port to all output port for a unit
      for (auto inChVal : op.getOperands()) {
        // Define variables w.r.t to input port
        if (!channelVars.contains(inChVal))
          continue;

        GRBVar &tIn = channelVars[inChVal].tDataOut;
        // GRBVar &tElasIn = channelVars[inChVal].tElasOut;
        for (auto outChVal : op.getResults()) {
          if (!channelVars.contains(outChVal))
            continue;

          // Define variables w.r.t to output port
          GRBVar &tOut = channelVars[outChVal].tDataIn;
          modelBuf.addConstr(tOut >= delayData + tIn);
        }
      }
    // if the unit is pipelined
    else {
      // Define constraints w.r.t to input port
      for (auto inChVal : op.getOperands()) {
        std::string out = "out";
        double inPortDelay = getPortDelay(inChVal, unitInfo, out);
        if (!channelVars.contains(inChVal))
          continue;

        GRBVar &tIn = channelVars[inChVal].tDataOut;
        modelBuf.addConstr(tIn <= targetCP - inPortDelay);
      }

      // Define constraints w.r.t to output port
      for (auto outChVal : op.getResults()) {
        std::string in = "in";
        double outPortDelay = getPortDelay(outChVal, unitInfo, in);

        if (!channelVars.contains(outChVal))
          continue;

        GRBVar &tOut = channelVars[outChVal].tDataIn;
        modelBuf.addConstr(tOut == outPortDelay);
      }
    }
  }
}

/// Create constraints that describe the circuits behavior
static void createModelElasticityConstraints(
    GRBModel &modelBuf, double targetCP, FuncOp &funcOp,
    std::vector<Value> &allChannels, unsigned unitNum,
    DenseMap<Value, ChannelVar> &channelVars,
    std::map<std::string, UnitInfo> &unitInfo) {
  // Channel constraints
  for (Value ch : allChannels) {
    // update the model to get the lower bound and upper bound of the vars
    modelBuf.update();
    if (!channelVars.contains(ch) || !coverPath(ch))
      continue;

    auto chVars = channelVars[ch];

    GRBVar &tElas1 = chVars.tElasIn;
    GRBVar &tElas2 = chVars.tElasOut;

    GRBVar &bufOp = chVars.bufIsOp;
    GRBVar &bufNSlots = chVars.bufNSlots;
    GRBVar &hasBuf = chVars.hasBuf;

    createElasticityConstrs(modelBuf, tElas1, tElas2, bufOp, bufNSlots, hasBuf,
                            unitNum + 2, targetCP);
  }

  // Units constraints
  for (auto &op : funcOp.getOps()) {
    // iterate all input port to all output port for a unit
    for (auto inChVal : op.getOperands()) {
      if (!channelVars.contains(inChVal))
        continue;

      // Define variables w.r.t to input port
      GRBVar &tElasIn = channelVars[inChVal].tElasOut;
      for (auto outChVal : op.getResults()) {
        if (!channelVars.contains(outChVal))
          continue;

        // Define variables w.r.t to output port
        GRBVar &tElasOut = channelVars[outChVal].tElasIn;
        modelBuf.addConstr(tElasOut >= 1 + tElasIn);
      }
    }
  }
}

static void createModelThrptConstraints(
    GRBModel &modelBuf, std::vector<GRBVar> &circtThrpt,
    std::vector<DenseMap<Value, GRBVar>> &chThrptToks,
    std::vector<CFDFC> &cfdfcList, DenseMap<Value, ChannelVar> &channelVars,
    std::vector<DenseMap<Operation *, UnitVar>> &unitVars,
    std::map<std::string, UnitInfo> &unitInfo) {
  for (auto [ind, subMG] : llvm::enumerate(chThrptToks)) {
    for (auto ch : cfdfcList[ind].channels) {

      if (!subMG.contains(ch))
        continue;

      Operation *srcOp = ch.getDefiningOp();
      Operation *dstOp = *(ch.getUsers().begin());

      GRBVar &thrptTok = subMG[ch];
      int tok = isBackEdge(srcOp, dstOp) ? 1 : 0;
      GRBVar &retSrc = unitVars[ind][srcOp].retOut;
      GRBVar &retDst = unitVars[ind][dstOp].retIn;

      GRBVar &bufOp = channelVars[ch].bufIsOp;
      GRBVar &bufNSlots = channelVars[ch].bufNSlots;
      createThroughputConstrs(modelBuf, retSrc, retDst, thrptTok,
                              circtThrpt[ind], bufOp, bufNSlots, tok);
    }
  }
  for (auto [ind, subMG] : llvm::enumerate(unitVars))
    for (auto &[op, unitVar] : subMG) {
      GRBVar &retIn = unitVar.retIn;
      GRBVar &retOut = unitVar.retOut;
      double latency = getUnitLatency(op, unitInfo);
      if (latency > 0)
        modelBuf.addConstr(retOut - retIn == latency * circtThrpt[ind]);
    }
}

/// Create constraints that is prerequisite for buffer placement
static void
setCustomizedConstraints(GRBModel &modelBuf,
                         DenseMap<Value, ChannelVar> &channelVars,
                         DenseMap<Value, ChannelBufProps> &channelBufProps,
                         DenseMap<Value, Result> &res) {
  for (auto &[ch, chVars] : channelVars) {
    // set min value of the buffer
    if (channelBufProps[ch].minOpaque > 0) {
      modelBuf.addConstr(chVars.bufNSlots >= channelBufProps[ch].minOpaque);
      modelBuf.addConstr(chVars.bufIsOp >= 0);
    } else if (channelBufProps[ch].minTrans > 0) {
      modelBuf.addConstr(chVars.bufNSlots >= channelBufProps[ch].minTrans);
      // modelBuf.addConstr(chVars.bufIsOp <= 0);
    }

    // set max value of the buffer
    if (channelBufProps[ch].maxOpaque.has_value())
      modelBuf.addConstr(chVars.bufNSlots <=
                         channelBufProps[ch].maxOpaque.value());

    if (channelBufProps[ch].maxTrans.has_value())
      modelBuf.addConstr(chVars.bufNSlots <=
                         channelBufProps[ch].maxTrans.value());
  }
  for (auto &[ch, result] : res) {
    modelBuf.addConstr(channelVars[ch].bufNSlots >= res[ch].numSlots);
    modelBuf.addConstr(channelVars[ch].bufIsOp >= res[ch].opaque);
  }
}

// Create MILP cost function
static void createModelObjective(GRBModel &modelBuf,
                                 std::vector<GRBVar> &circtThrpts,
                                 std::vector<CFDFC> &cfdfcList,
                                 DenseMap<Value, ChannelVar> &channelVars) {
  GRBLinExpr objExpr;
  double lumbdaCoef1 = 1e-4;
  double lumbdaCoef2 = 1e-5;

  double totalFreq = 0.0;
  double highestCoef = 0.0;
  for (auto [channel, _] : channelVars)
    totalFreq += static_cast<double>(getChannelFreq(channel, cfdfcList));

  for (auto [ind, thrpt] : llvm::enumerate(circtThrpts)) {
    double coef =
        cfdfcList[ind].channels.size() * cfdfcList[ind].numExec / totalFreq;
    highestCoef = std::max(coef, highestCoef);
    objExpr += coef * thrpt;
  }

  for (auto &[_, chVar] : channelVars) {
    objExpr -= highestCoef *
               (lumbdaCoef1 * chVar.hasBuf + lumbdaCoef2 * chVar.bufNSlots);
  }

  modelBuf.setObjective(objExpr, GRB_MAXIMIZE);
}
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

LogicalResult buffer::placeBufferInCFDFCircuit(
    DenseMap<Value, Result> &res, handshake::FuncOp &funcOp,
    std::vector<Value> &allChannels, std::vector<CFDFC> &cfdfcList,
    std::vector<unsigned> &cfdfcInds, double targetCP, int timeLimit,
    bool setCustom, std::map<std::string, UnitInfo> &unitInfo,
    DenseMap<Value, ChannelBufProps> &channelBufProps) {

#ifdef DYNAMATIC_GUROBI_NOT_INSTALLED
  llvm::errs() << "Project was built without Gurobi installed, can't run "
                  "buffer placement\n";
  return failure();
#else
  // create a Gurobi environment

  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();
  GRBModel modelBuf = GRBModel(env);

  // create variables
  std::vector<DenseMap<Operation *, UnitVar>> unitVars;
  std::vector<DenseMap<Value, GRBVar>> chThrptToks;
  DenseMap<Value, ChannelVar> channelVars;
  std::vector<GRBVar> circtThrpts;

  // create the variable to noate the overall circuit throughput
  for (auto [ind, _] : llvm::enumerate(cfdfcList)) {
    GRBVar circtThrpt = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                        "thrpt" + std::to_string(ind));
    circtThrpts.push_back(circtThrpt);
  }

  // Compute the total numbers of units in the circuit as a coef for elasticity
  // constraints
  unsigned unitNum =
      std::distance(funcOp.getOps().begin(), funcOp.getOps().end());

  // initialize variables
  initVarsInMILP(funcOp, modelBuf, cfdfcList, cfdfcInds, allChannels, unitVars,
                 chThrptToks, channelVars, unitInfo);

  // define customized constraints
  if (setCustom)
    setCustomizedConstraints(modelBuf, channelVars, channelBufProps, res);

  // create circuits constraints
  createModelPathConstraints(modelBuf, targetCP, funcOp, allChannels,
                             channelVars, unitInfo);

  createModelElasticityConstraints(modelBuf, targetCP, funcOp, allChannels,
                                   unitNum, channelVars, unitInfo);

  createModelThrptConstraints(modelBuf, circtThrpts, chThrptToks, cfdfcList,
                              channelVars, unitVars, unitInfo);

  // create cost function
  createModelObjective(modelBuf, circtThrpts, cfdfcList, channelVars);

  modelBuf.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit);
  modelBuf.optimize();

  //  The result is optimal if the model is solved to optimality or the time
  //  limit and obtain position throughput
  bool isOptimal = (modelBuf.get(GRB_IntAttr_Status) == GRB_OPTIMAL) ||
                   (modelBuf.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT &&
                    modelBuf.get(GRB_DoubleAttr_ObjVal) > 0);

  if (!isOptimal) {
    llvm::errs() << "no optimal sol\n";
    return failure();
  }

  // load answer to the result
  for (auto &[ch, chVarMap] : channelVars) {
    if (chVarMap.hasBuf.get(GRB_DoubleAttr_X) > 0) {
      Result result;
      result.numSlots =
          static_cast<int>(chVarMap.bufNSlots.get(GRB_DoubleAttr_X) + 0.5);
      result.opaque = chVarMap.bufIsOp.get(GRB_DoubleAttr_X) > 0;
      res[ch] = result;
    }
  }
  return success();
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
}