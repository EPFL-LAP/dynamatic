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

unsigned buffer::getPortInd(Operation *op, Value val) {
  for (auto [indVal, port] : llvm::enumerate(op->getResults())) {
    if (port == val) {
      return indVal;
    }
  }
  return UINT_MAX;
}

/// Get the pointer to the channel that defines the channel variables
static std::optional<Value *>
inChannelMap(const std::map<Value *, ChannelVar> &channelVars, Value ch) {
  for (auto &[chVal, chVar] : channelVars) {
    if (*chVal == ch)
      return chVal;
  }
  return nullptr;
}

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
/// Initialize the variables in the MILP model
static void initVarsInMILP(handshake::FuncOp funcOp, GRBModel &modelBuf,
                           std::vector<Operation *> &units,
                           std::vector<Value> &channels,
                           std::vector<Value> &allChannels,
                           DenseMap<Operation *, UnitVar> &unitVars,
                           DenseMap<Value, ChannelVar> &channelVars,
                           std::map<std::string, UnitInfo> unitInfo) {
  // create variables
  unsigned unitInd = 0;
  for (auto &op : funcOp.getOps()) {
    UnitVar unitVar;
    unitVar.select = false;
    // If in the CFDFC, set select to true
    if (std::find(units.begin(), units.end(), &op) != units.end())
      unitVar.select = true;

    // init unit variables
    std::string unitName = getOperationShortStrName(&op);
    unitVar.retIn =
        modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "inRetimeTok_" + unitName + std::to_string(unitInd));
    if (getUnitLatency(&op, unitInfo) < 1e-10)
      unitVar.retOut = unitVar.retIn;
    else
      unitVar.retOut =
          modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                          "outRetimeTok_" + unitName + std::to_string(unitInd));
    unitVars[&op] = unitVar;
    unitInd++;
  }

  modelBuf.update();

  // create channel vars
  for (auto [ind, val] : llvm::enumerate(allChannels)) {
    Operation *srcOp = val.getDefiningOp();
    Operation *dstOp = getUserOp(val);

    if (!srcOp && !dstOp)
      continue;

    std::string srcOpVarName = "arg_start";
    std::string dstOpVarName = "arg_end";

    if (srcOp)
      // Define the channel variable names w.r.t to its connected units
      srcOpVarName = unitVars[srcOp].retIn.get(GRB_StringAttr_VarName);
    if (dstOp)
      dstOpVarName = unitVars[dstOp].retIn.get(GRB_StringAttr_VarName);

    std::string srcOpName = srcOpVarName.substr(srcOpVarName.find('_', 0) + 1);
    std::string dstOpName = dstOpVarName.substr(dstOpVarName.find('_', 0) + 1);

    // create channel variable
    ChannelVar channelVar;
    std::string chName =
        srcOpName + "_" + dstOpName + "_" + std::to_string(ind);

    channelVar.select = false;
    if (std::find(channels.begin(), channels.end(), val) != channels.end())
      channelVar.select = true;

    channelVar.tDataIn =
        modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "timePathIn_" + chName);
    channelVar.tDataOut =
        modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "timePathOut_" + chName);

    channelVar.tElasIn = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                         "timeElasticIn_" + chName);
    channelVar.tElasOut = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                          "timeElasticOut_" + chName);

    channelVar.bufNSlots = modelBuf.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER,
                                           chName + "_bufNSlots");
    channelVar.thrptTok = modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0,
                                          GRB_CONTINUOUS, "thrpt_" + chName);
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

/// Create time path constraints over units.
/// t1 is the input time of the unit, t2 is the output time of the unit.
static void createPathConstrs(GRBModel &modelBuf, GRBVar &tIn, GRBVar &tOut,
                              double delay, double latency, double period,
                              double inPortDelay = 0.0,
                              double outPortDelay = 0.0) {
  // if the unit is combinational
  if (latency == 0)
    modelBuf.addConstr(tOut >= delay + tIn);
  else {
    // if the unit is pipelined
    modelBuf.addConstr(tIn <= period - inPortDelay);
    modelBuf.addConstr(tOut == outPortDelay);
  }
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

// create elasticity constraints w.r.t units
static void createElasticityConstrs(GRBModel &modelBuf, GRBVar &tIn,
                                    GRBVar &tOut, double delay, double latency,
                                    double period) {
  modelBuf.addConstr(tOut >= 1 + tIn);
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

/// Throughput constraints over a unit, relugated by the input and output
/// retiming values
static void createThroughputConstrs(GRBModel &modelBuf, GRBVar &retIn,
                                    GRBVar &retout, GRBVar &thrpt,
                                    double latency) {
  if (latency > 0)
    modelBuf.addConstr(retout - retIn == latency * thrpt);
}

/// Create constraints that describe the circuits behavior
static LogicalResult
createModelConstraints(GRBModel &modelBuf, GRBVar &thrpt, double targetCP,
                       DenseMap<Operation *, UnitVar> &unitVars,
                       DenseMap<Value, ChannelVar> &channelVars,
                       std::map<std::string, UnitInfo> unitInfo) {
  // Channel constraints
  for (auto [ch, chVars] : channelVars) {
    // update the model to get the lower bound and upper bound of the vars
    modelBuf.update();

    // place buffers if maxinum buffer slots is larger then 0 and the channel
    // is selected
    if (chVars.bufNSlots.get(GRB_DoubleAttr_UB) <= 0)
      continue;

    Operation *srcOp = ch.getDefiningOp();
    Operation *dstOp = *(ch.getUsers().begin());
    GRBVar &t1 = chVars.tDataIn;
    GRBVar &t2 = chVars.tDataOut;

    GRBVar &tElas1 = chVars.tElasIn;
    GRBVar &tElas2 = chVars.tElasOut;

    GRBVar &bufOp = chVars.bufIsOp;
    GRBVar &bufNSlots = chVars.bufNSlots;
    GRBVar &hasBuf = chVars.hasBuf;

    createPathConstrs(modelBuf, t1, t2, bufOp, targetCP);
    createElasticityConstrs(modelBuf, tElas1, tElas2, bufOp, bufNSlots, hasBuf,
                            unitVars.size() + 1, targetCP);

    if (chVars.select && chVars.bufNSlots.get(GRB_DoubleAttr_UB) > 0)
      if (unitVars.count(dstOp) > 0) {
        GRBVar &thrptTok = chVars.thrptTok;
        int tok = isBackEdge(srcOp, dstOp) ? 1 : 0;
        GRBVar &retSrc = unitVars[srcOp].retOut;
        GRBVar &retDst = unitVars[dstOp].retIn;
        createThroughputConstrs(modelBuf, retSrc, retDst, thrptTok, thrpt,
                                bufOp, bufNSlots, tok);
      }
  }

  // Units constraints
  for (auto [op, unitVar] : unitVars) {
    double delayData = getCombinationalDelay(op, unitInfo, "data");
    double delayValid = getCombinationalDelay(op, unitInfo, "valid");
    double delayReady = getCombinationalDelay(op, unitInfo, "ready");
    double latency = getUnitLatency(op, unitInfo);

    GRBVar &retIn = unitVar.retIn;
    GRBVar &retOut = unitVar.retOut;

    if (unitVar.select)
      createThroughputConstrs(modelBuf, retIn, retOut, thrpt, latency);

    // iterate all input port to all output port for a unit
    for (auto inChVal : op->getOperands()) {
      // Define variables w.r.t to input port
      double inPortDelay = getPortDelay(inChVal, unitInfo, "out");
      if (channelVars.contains(inChVal) == false)
        continue;

      GRBVar &tIn = channelVars[inChVal].tDataOut;
      GRBVar &tElasIn = channelVars[inChVal].tElasOut;

      for (auto outChVal : op->getResults()) {
        // Define variables w.r.t to output port
        double outPortDelay = getPortDelay(outChVal, unitInfo, "in");

        if (channelVars.contains(outChVal) == false)
          continue;

        GRBVar &tOut = channelVars[outChVal].tDataIn;
        GRBVar &tElasOut = channelVars[outChVal].tElasIn;

        createPathConstrs(modelBuf, tIn, tOut, delayData, latency, targetCP);
        createElasticityConstrs(modelBuf, tElasIn, tElasOut, delayData, latency,
                                targetCP);
      }
    }
  }
  return success();
}

/// Create constraints that is prerequisite for buffer placement
static LogicalResult
setCustomizedConstraints(GRBModel &modelBuf,
                         DenseMap<Value, ChannelVar> channelVars,
                         DenseMap<Value, ChannelBufProps> &channelBufProps,
                         DenseMap<Value, Result> &res) {
  for (auto chVarMap : channelVars) {
    auto &[ch, chVars] = chVarMap;
    // set min value of the buffer
    if (channelBufProps[ch].minNonTrans > 0) {
      modelBuf.addConstr(chVars.bufNSlots >= channelBufProps[ch].minNonTrans);
      modelBuf.addConstr(chVars.bufIsOp >= 0);
    } else if (channelBufProps[ch].minTrans > 0) {
      modelBuf.addConstr(chVars.bufNSlots >= channelBufProps[ch].minTrans);
      modelBuf.addConstr(chVars.bufIsOp <= 0);
    }

    // set max value of the buffer
    if (channelBufProps[ch].maxNonTrans.has_value())
      channelVars[ch].bufNSlots.set(GRB_DoubleAttr_UB,
                                    channelBufProps[ch].maxNonTrans.value());

    if (channelBufProps[ch].maxTrans.has_value())
      channelVars[ch].bufNSlots.set(GRB_DoubleAttr_UB,
                                    channelBufProps[ch].maxTrans.value());
  }
  for (auto &[ch, result] : res) {
    modelBuf.addConstr(channelVars[ch].bufNSlots >= res[ch].numSlots);
    modelBuf.addConstr(channelVars[ch].bufIsOp >= res[ch].opaque);
  }

  return success();
}

// Create MILP cost function
static void createModelObjective(GRBModel &modelBuf, GRBVar &thrpt,
                                 DenseMap<Value, ChannelVar> channelVars) {
  GRBLinExpr objExpr = thrpt;

  double lumbdaCoef1 = 9.98546 * 1e-5;
  double lumbdaCoef2 = 9.98546 * 1e-6;
  for (auto &[_, chVar] : channelVars) {
    objExpr -= lumbdaCoef1 * chVar.hasBuf + lumbdaCoef2 * chVar.bufNSlots;
  }

  modelBuf.setObjective(objExpr, GRB_MAXIMIZE);
}
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

LogicalResult buffer::placeBufferInCFDFCircuit(
    handshake::FuncOp funcOp, std::vector<Value> &allChannels,
    CFDFC &cfdfcCircuit, DenseMap<Value, Result> &res, double targetCP,
    std::map<std::string, UnitInfo> unitInfo,
    DenseMap<Value, ChannelBufProps> channelBufProps) {

#ifdef DYNAMATIC_GUROBI_NOT_INSTALLED
  llvm::errs() << "Project was built without Gurobi installed, can't run "
                  "CFDFC extraction\n";
  return failure();
#else
  // create a Gurobi environment

  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();
  GRBModel modelBuf = GRBModel(env);

  // create variables
  DenseMap<Operation *, UnitVar> unitVars;
  DenseMap<Value, ChannelVar> channelVars;

  // create the variable to noate the overall circuit throughput
  GRBVar circtThrpt = modelBuf.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "thrpt");

  // initialize variables
  initVarsInMILP(funcOp, modelBuf, cfdfcCircuit.units, cfdfcCircuit.channels,
                 allChannels, unitVars, channelVars, unitInfo);

  // define customized constraints
  setCustomizedConstraints(modelBuf, channelVars, channelBufProps, res);

  // create circuits constraints
  createModelConstraints(modelBuf, circtThrpt, targetCP, unitVars, channelVars,
                         unitInfo);

  // create cost function
  createModelObjective(modelBuf, circtThrpt, channelVars);

  modelBuf.optimize();
  modelBuf.write("/home/yuxuan/Downloads/model.lp");

  if (modelBuf.get(GRB_IntAttr_Status) != GRB_OPTIMAL ||
      circtThrpt.get(GRB_DoubleAttr_X) <= 0) {
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
      if (res.count(ch) > 0)
        res[ch] = res[ch] + result;
      else
        res[ch] = result;
    }
  }
  return success();
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
}