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
static void
initVarsInMILP(handshake::FuncOp funcOp, GRBModel &modelBuf,
               std::vector<CFDFC> cfdfcList, unsigned cfdfcInd,
               std::vector<Value> &allChannels,
               std::vector<DenseMap<Operation *, UnitVar>> &unitVars,
               std::vector<DenseMap<Value, GRBVar>> &chThrptToks,
               DenseMap<Value, ChannelVar> &channelVars,
               std::map<std::string, UnitInfo> unitInfo) {
  // create variables
  unsigned unitInd = 0;
  std::vector<Operation *> units = cfdfcList[cfdfcInd].units;
  std::vector<Value> channels = cfdfcList[cfdfcInd].channels;

  for (auto [ind, cfdfc] : llvm::enumerate(cfdfcList)) {
    unitVars.push_back(DenseMap<Operation *, UnitVar>());
    chThrptToks.push_back(DenseMap<Value, GRBVar>());
    for (auto [unitInd, unit] : llvm::enumerate(cfdfc.units)) {
      UnitVar unitVar;
      unitVar.select = false;

      // If in the CFDFC, set select to true
      if (std::find(units.begin(), units.end(), unit) != units.end())
        unitVar.select = true;

      // init unit variables
      std::string unitName = getOperationShortStrName(unit);
      unitVar.retIn =
          modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                          "mg" + std::to_string(ind) + "_inRetimeTok_" +
                              unitName + std::to_string(unitInd));
      if (getUnitLatency(unit, unitInfo) < 1e-10)
        unitVar.retOut = unitVar.retIn;
      else
        unitVar.retOut =
            modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                            "mg" + std::to_string(ind) + "_outRetimeTok_" +
                                unitName + std::to_string(unitInd));
      unitVars[ind][unit] = unitVar;
    }

    for (auto [chInd, channel] : llvm::enumerate(cfdfc.channels)) {
      std::string srcName = "arg_start";
      std::string dstName = "arg_end";
      Operation *srcOp = channel.getDefiningOp();
      Operation *dstOp = getUserOp(channel);
      // Define the channel variable names w.r.t to its connected units
      if (srcOp)
        srcName = getOperationShortStrName(srcOp);
      if (dstOp)
        dstName = getOperationShortStrName(dstOp);

      std::string chName = "mg" + std::to_string(ind) + "_" + srcName + "_" +
                           dstName + "_" + std::to_string(chInd);
      chThrptToks[ind][channel] = modelBuf.addVar(
          -GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "thrpt_" + chName);
    }
  }
  modelBuf.update();

  // create channel vars
  for (auto [ind, val] : llvm::enumerate(allChannels)) {
    Operation *srcOp = val.getDefiningOp();
    Operation *dstOp = getUserOp(val);

    if (!srcOp || !dstOp)
      continue;

    std::string srcOpName = "arg_start";
    std::string dstOpName = "arg_end";

    // if (srcOp && dstOp) {
    // Define the channel variable names w.r.t to its connected units
    srcOpName = getOperationShortStrName(srcOp);
    dstOpName = getOperationShortStrName(dstOp);
    // }

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

/// Create constraints that describe the circuits behavior
static LogicalResult
createModelPathConstraints(GRBModel &modelBuf, double targetCP, FuncOp &funcOp,
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

    GRBVar &t1 = chVars.tDataIn;
    GRBVar &t2 = chVars.tDataOut;

    // GRBVar &tElas1 = chVars.tElasIn;
    // GRBVar &tElas2 = chVars.tElasOut;

    GRBVar &bufOp = chVars.bufIsOp;
    GRBVar &bufNSlots = chVars.bufNSlots;
    GRBVar &hasBuf = chVars.hasBuf;

    createPathConstrs(modelBuf, t1, t2, bufOp, targetCP);
  }

  // Units constraints
  for (auto &op : funcOp.getOps()) {
    double delayData = getCombinationalDelay(&op, unitInfo, "data");
    double delayValid = getCombinationalDelay(&op, unitInfo, "valid");
    double delayReady = getCombinationalDelay(&op, unitInfo, "ready");
    double latency = getUnitLatency(&op, unitInfo);

    // iterate all input port to all output port for a unit
    if (latency == 0)
      for (auto inChVal : op.getOperands()) {
        // Define variables w.r.t to input port
        double inPortDelay = getPortDelay(inChVal, unitInfo, "out");
        if (channelVars.contains(inChVal) == false)
          continue;

        GRBVar &tIn = channelVars[inChVal].tDataOut;
        // GRBVar &tElasIn = channelVars[inChVal].tElasOut;
        for (auto outChVal : op.getResults()) {
          // Define variables w.r.t to output port
          double outPortDelay = getPortDelay(outChVal, unitInfo, "in");

          if (channelVars.contains(outChVal) == false)
            continue;

          GRBVar &tOut = channelVars[outChVal].tDataIn;
          // GRBVar &tElasOut = channelVars[outChVal].tElasIn;
          modelBuf.addConstr(tOut >= delayData + tIn);
        }
      }
    else {
      for (auto inChVal : op.getOperands()) {
        double inPortDelay = getPortDelay(inChVal, unitInfo, "out");
        if (channelVars.contains(inChVal) == false)
          continue;

        GRBVar &tIn = channelVars[inChVal].tDataOut;
        GRBVar &tElasIn = channelVars[inChVal].tElasOut;
        modelBuf.addConstr(tIn <= targetCP - inPortDelay);
      }
      for (auto outChVal : op.getResults()) {
        // Define variables w.r.t to output port
        double outPortDelay = getPortDelay(outChVal, unitInfo, "in");

        if (channelVars.contains(outChVal) == false)
          continue;

        GRBVar &tOut = channelVars[outChVal].tDataIn;
        modelBuf.addConstr(tOut == outPortDelay);
      }
    }
  }
  return success();
}

/// Create constraints that describe the circuits behavior
static LogicalResult
createModelElasticityConstraints(GRBModel &modelBuf, double targetCP,
                                 FuncOp &funcOp, unsigned unitNum,
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

    GRBVar &tElas1 = chVars.tElasIn;
    GRBVar &tElas2 = chVars.tElasOut;

    GRBVar &bufOp = chVars.bufIsOp;
    GRBVar &bufNSlots = chVars.bufNSlots;
    GRBVar &hasBuf = chVars.hasBuf;

    createElasticityConstrs(modelBuf, tElas1, tElas2, bufOp, bufNSlots, hasBuf,
                            unitNum + 1, targetCP);
  }

  // Units constraints
  for (auto &op : funcOp.getOps()) {
    double delayData = getCombinationalDelay(&op, unitInfo, "data");
    double delayValid = getCombinationalDelay(&op, unitInfo, "valid");
    double delayReady = getCombinationalDelay(&op, unitInfo, "ready");
    double latency = getUnitLatency(&op, unitInfo);

    // iterate all input port to all output port for a unit

    for (auto inChVal : op.getOperands()) {
      // Define variables w.r.t to input port
      double inPortDelay = getPortDelay(inChVal, unitInfo, "out");
      if (channelVars.contains(inChVal) == false)
        continue;

      GRBVar &tElasIn = channelVars[inChVal].tElasOut;
      for (auto outChVal : op.getResults()) {
        // Define variables w.r.t to output port
        double outPortDelay = getPortDelay(outChVal, unitInfo, "in");

        if (channelVars.contains(outChVal) == false)
          continue;

        GRBVar &tElasOut = channelVars[outChVal].tElasIn;
        createElasticityConstrs(modelBuf, tElasIn, tElasOut, delayData, latency,
                                targetCP);
      }
    }
  }
  return success();
}

static void createModelThrptConstraints(
    GRBModel &modelBuf, std::vector<GRBVar> &circtThrpt,
    std::vector<DenseMap<Value, GRBVar>> &chThrptToks,
    DenseMap<Value, ChannelVar> &channelVars,
    std::vector<DenseMap<Operation *, UnitVar>> &unitVars,
    std::map<std::string, UnitInfo> &unitInfo) {
  for (auto [ind, subMG] : llvm::enumerate(chThrptToks)) {
    for (auto &[ch, thrptTok] : subMG) {
      Operation *srcOp = ch.getDefiningOp();
      Operation *dstOp = *(ch.getUsers().begin());
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
static LogicalResult
setCustomizedConstraints(GRBModel &modelBuf,
                         DenseMap<Value, ChannelVar> channelVars,
                         DenseMap<Value, ChannelBufProps> &channelBufProps,
                         DenseMap<Value, Result> &res) {
  for (auto chVarMap : channelVars) {
    auto &[ch, chVars] = chVarMap;
    // set min value of the buffer
    if (channelBufProps[ch].minOpaque > 0) {
      modelBuf.addConstr(chVars.bufNSlots >= channelBufProps[ch].minOpaque);
      modelBuf.addConstr(chVars.bufIsOp >= 0);
    } else if (channelBufProps[ch].minTrans > 0) {
      modelBuf.addConstr(chVars.bufNSlots >= channelBufProps[ch].minTrans);
      modelBuf.addConstr(chVars.bufIsOp <= 0);
    }

    // set max value of the buffer
    if (channelBufProps[ch].maxOpaque.has_value())
      channelVars[ch].bufNSlots.set(GRB_DoubleAttr_UB,
                                    channelBufProps[ch].maxOpaque.value());

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
static void createModelObjective(GRBModel &modelBuf,
                                 std::vector<GRBVar> &circtThrpts,
                                 std::vector<CFDFC> &cfdfcList,
                                 DenseMap<Value, ChannelVar> channelVars) {
  GRBLinExpr objExpr;
  double lumbdaCoef1 = 1e-5;
  double lumbdaCoef2 = 1e-6;

  double totalFreq = 0.0;
  double highest_coef = 0.0;
  for (auto &cfdfc : cfdfcList)
    totalFreq += static_cast<double>(cfdfc.execN) * cfdfc.channels.size();

  for (auto [ind, thrpt] : llvm::enumerate(circtThrpts)) {
    double coef =
        cfdfcList[ind].channels.size() * cfdfcList[ind].execN / totalFreq;
    if (coef > highest_coef)
      highest_coef = coef;
    objExpr += coef * thrpt;
  }

  for (auto &[_, chVar] : channelVars) {
    objExpr -= highest_coef *
               (lumbdaCoef1 * chVar.hasBuf + lumbdaCoef2 * chVar.bufNSlots);
  }

  modelBuf.setObjective(objExpr, GRB_MAXIMIZE);
}
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

LogicalResult buffer::placeBufferInCFDFCircuit(
    DenseMap<Value, Result> &res, handshake::FuncOp funcOp,
    std::vector<Value> &allChannels, std::vector<CFDFC> cfdfcList,
    unsigned cfdfcInd, double targetCP,
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
  std::vector<DenseMap<Operation *, UnitVar>> unitVars;
  std::vector<DenseMap<Value, GRBVar>> chThrptToks;
  DenseMap<Value, ChannelVar> channelVars;
  std::vector<GRBVar> circtThrpts;

  // create the variable to noate the overall circuit throughput
  for (auto [ind, _] : llvm::enumerate(cfdfcList)) {
    GRBVar circtThrpt =
        modelBuf.addVar(-GRB_CONTINUOUS, GRB_CONTINUOUS, 0.0, GRB_CONTINUOUS,
                        "thrpt" + std::to_string(ind));
    circtThrpts.push_back(circtThrpt);
  }

  // initialize variables
  initVarsInMILP(funcOp, modelBuf, cfdfcList, cfdfcInd, allChannels, unitVars,
                 chThrptToks, channelVars, unitInfo);

  // define customized constraints
  setCustomizedConstraints(modelBuf, channelVars, channelBufProps, res);

  // create circuits constraints
  unsigned unitNum = 0;
  for (auto &op : funcOp.getOps())
    unitNum++;
  createModelPathConstraints(modelBuf, targetCP, funcOp, channelVars, unitInfo);

  createModelElasticityConstraints(modelBuf, targetCP, funcOp, unitNum,
                                   channelVars, unitInfo);

  createModelThrptConstraints(modelBuf, circtThrpts, chThrptToks, channelVars,
                              unitVars, unitInfo);

  // create cost function
  createModelObjective(modelBuf, circtThrpts, cfdfcList, channelVars);

  modelBuf.write("/home/yuxuan/Downloads/model.lp");
  modelBuf.optimize();

  if (modelBuf.get(GRB_IntAttr_Status) != GRB_OPTIMAL ||
      circtThrpts[cfdfcInd].get(GRB_DoubleAttr_X) <= 0) {
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