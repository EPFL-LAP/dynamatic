//===- OptimizeMILP.cpp - optimize MILP model over CFDFC  -------*- C++ -*-===//
//
// This file implements the MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/OptimizeMILP.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Transforms/BufferPlacement/ParseCircuitJson.h"
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
  unsigned indVal = 0;
  for (auto port : op->getResults()) {
    if (port == val) {
      return indVal;
    }
    indVal++;
  }
  return UINT_MAX;
}

// unsigned buffer::getOutPortInd(Operation *op, Value val) {
//   unsigned indVal = 0;
//   for (auto port : op->getResults()) {
//     if (port == val) {
//       return indVal;
//     }
//     indVal++;
//   }
//   return UINT_MAX;
// }

static std::optional<Value *>
inChannelMap(const std::map<Value *, ChannelVar> &channelVars, Value ch) {
  for (auto &[chVal, chVar] : channelVars) {
    if (*chVal == ch)
      return chVal;
  }
  return nullptr;
}

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
static void initVarsInMILP(handshake::FuncOp funcOp, GRBModel &modelBuf,
                           std::vector<Operation *> &units,
                           std::vector<Value> &channels,
                           std::vector<Value> &allChannels,
                           std::map<Operation *, UnitVar> &unitVars,
                           std::map<Value *, ChannelVar> &channelVars,
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

  llvm::errs() << "     init unit vars\n";
  modelBuf.update();

  // create channel vars
  for (auto [ind, val] : llvm::enumerate(allChannels)) {
    Operation *srcOp = val.getDefiningOp();
    Operation *dstOp = getUserOp(val);

    if (srcOp == nullptr || dstOp == nullptr)
      continue;

    std::string srcOpVarName =
        unitVars[srcOp].retIn.get(GRB_StringAttr_VarName);
    std::string srcOpName = srcOpVarName.substr(srcOpVarName.find('_', 0) + 1);
    std::string dstOpVarName =
        unitVars[dstOp].retIn.get(GRB_StringAttr_VarName);
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

    channelVar.tElasIn =
        modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "timeElasticIn_" + chName);
    channelVar.tElasOut =
        modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "timeElasticOut_" + chName);

    channelVar.bufNSlots = modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0,
                                           GRB_INTEGER, chName + "_bufNSlots");
    channelVar.thrptTok = modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0,
                                          GRB_CONTINUOUS, "thrpt_" + chName);
    channelVar.hasBuf = modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0,
                                        GRB_BINARY, chName + "_hasBuf");
    channelVar.bufIsOp = modelBuf.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0,
                                         GRB_BINARY, chName + "_bufIsOp");

    // channelVar.tValidIn = modelBuf.addVar(0.0, GRB_INFINITY, 0.0,
    //                                       GRB_CONTINUOUS, "tValidIn_" +
    //                                       chName);

    // channelVar.tValidOut = modelBuf.addVar(
    //     0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "tValidOut_" + chName);
    // channelVar.tReadyIn = modelBuf.addVar(0.0, GRB_INFINITY, 0.0,
    //                                       GRB_CONTINUOUS, "tReadyIn_" +
    //                                       chName);
    // channelVar.tReadyOut = modelBuf.addVar(
    //     0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "tReadyOut_" + chName);

    // channelVar.valbufIsOp =
    //     modelBuf.addVar(0.0, 1.0, 0.0, GRB_BINARY, chName + "valbufIsOp");
    // channelVar.readybufIsOp =
    //     modelBuf.addVar(0.0, 1.0, 0.0, GRB_BINARY, chName + "readybufIsOp");

    // Value ch = dyn_cast<Value>(val);
    channelVars[&val] = channelVar;
    modelBuf.update();
    // portInd++;
  }
}

static void initVarsInMILP(GRBModel &modelBuf, std::vector<Operation *> &units,
                           std::vector<Value> &channels,
                           std::map<Operation *, UnitVar> &unitVars,
                           std::map<Value *, ChannelVar> &channelVars) {
  // create variables

  for (auto [ind, unit] : llvm::enumerate(units)) {
    // create variables for each unit

    UnitVar unitVar;
    std::string unitName = getOperationShortStrName(unit);
    llvm::errs() << "~~~~~~~creating vars :" << ind << " : " << *unit << "\n";
    unitVar.retIn =
        modelBuf.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS,
                        "inRetimeTok_" + unitName + std::to_string(ind));
    unitVar.retOut =
        modelBuf.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS,
                        "outRetimeTok_" + unitName + std::to_string(ind));
    unitVars[unit] = unitVar;
  }

  for (auto [ind, val] : llvm::enumerate(channels)) {
    ChannelVar channelVar;
    std::string srcOpName = getOperationShortStrName(val.getDefiningOp());
    std::string dstOpName = getOperationShortStrName(getUserOp(val));

    llvm::errs() << "~~~~~~~creating vars :" << ind << " : " << val << "\n";
    llvm::errs() << srcOpName << " -> " << dstOpName << "\n";
    channelVar.tDataIn =
        modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "timePathOut_" + srcOpName + std::to_string(ind));
    channelVar.tDataOut =
        modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "timePathIn_" + dstOpName + std::to_string(ind));

    channelVar.tElasIn =
        modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "timeElasticOut_" + srcOpName + std::to_string(ind));
    channelVar.tElasOut =
        modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "timeElasticIn_" + dstOpName + std::to_string(ind));
    channelVar.tValidIn =
        modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "tValidIn" + std::to_string(ind));
    channelVar.tValidOut =
        modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "tValidOut" + std::to_string(ind));
    channelVar.tReadyIn =
        modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "tReadyIn" + std::to_string(ind));
    channelVar.tReadyOut =
        modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                        "tReadyOut" + std::to_string(ind));
    channelVar.thrptTok = modelBuf.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS,
                                          "thrpt" + std::to_string(ind));
    channelVar.bufIsOp = modelBuf.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                                         srcOpName + "_" + dstOpName +
                                             "_bufIsOp" + std::to_string(ind));

    channelVar.valbufIsOp = modelBuf.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                                            "valbufIsOp" + std::to_string(ind));
    channelVar.readybufIsOp = modelBuf.addVar(
        0.0, 1.0, 0.0, GRB_BINARY, "readybufIsOp" + std::to_string(ind));

    channelVar.bufNSlots = modelBuf.addVar(
        0.0, GRB_INFINITY, 0.0, GRB_INTEGER,
        srcOpName + "_" + dstOpName + "_bufNSlots" + std::to_string(ind));

    channelVar.hasBuf = modelBuf.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                                        srcOpName + "_" + dstOpName +
                                            "_hasBuf" + std::to_string(ind));

    channelVars[&val] = channelVar;
  }
}

static void createPathConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                              GRBVar &bufOp, double period,
                              unsigned constrInd) {
  modelBuf.addConstr(t1 <= period);
  modelBuf.addConstr(t2 <= period);
  modelBuf.addConstr(t2 >= t1 - 2 * period * bufOp);
}

static void createPathConstrs(GRBModel &modelBuf, GRBVar &tIn, GRBVar &tOut,
                              double delay, double latency, double period,
                              std::string constrInd, double inPortDelay = 0.0,
                              double outPortDelay = 0.0) {
  if (latency < 1e-10)
    modelBuf.addConstr(tOut >= delay + tIn);
  else {
    // llvm::errs() << tIn.get(GRB_StringAttr_VarName) << " "
    //              << tOut.get(GRB_StringAttr_VarName) << "is pipelined
    //              unit\n";
    modelBuf.addConstr(tIn <= period - inPortDelay);
    modelBuf.addConstr(tOut == outPortDelay);
  }
}

// Data path constratins for a channel
// t1, t2 are the time stamps in the source and destination of the channel
// bufOp: whether the inserted buffer in a channel is opaque or not;
// bufRdyOp: whether the inserted buffer in a channel for the handshake ready
// signal is opaque or not;
// period: circuit period
// constrInd: index of the constraint
// bufDelay: delay of the inserted buffer in the channel
// ctrlBufDelay: delay of the inserted buffer for the handshake ready signal
static void createDataPathConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                                  GRBVar &bufOp, GRBVar &bufRdyOp,
                                  double period, unsigned constrInd,
                                  double bufDelay = 0.0,
                                  double ctrlBufDelay = 0.1) {
  modelBuf.addConstr(t1 <= period, "dataChStart" + std::to_string(constrInd));
  modelBuf.addConstr(t2 <= period, "dataChEnd" + std::to_string(constrInd));
  modelBuf.addConstr(t2 >= t1 - 2 * period * bufOp + ctrlBufDelay * bufRdyOp,
                     "dataChPath" + std::to_string(constrInd));
  modelBuf.addConstr(t2 >= bufDelay,
                     "dataPath1Buf" + std::to_string(constrInd));
  modelBuf.addConstr(t2 >= ctrlBufDelay * bufRdyOp,
                     "dataPath2Buf" + std::to_string(constrInd));
}

// Data path constratins for a unit
static void createDataPathConstrs(GRBModel &modelBuf, GRBVar &tIn, GRBVar &tOut,
                                  double delay, double latency, double period,
                                  std::string constrInd,
                                  double inPortDelay = 0.0,
                                  double outPortDelay = 0.0) {
  // llvm::errs() << "combination delay for constraints: " << delay << "\n";
  // llvm::errs() << "latency for constraints: " << latency << "\n";
  if (latency < 1e-10)
    modelBuf.addConstr(tOut >= delay + tIn, "unitDataPath" + constrInd);
  else {
    modelBuf.addConstr(tIn <= period - inPortDelay,
                       "unitData1Path" + constrInd);
    modelBuf.addConstr(tOut <= outPortDelay, "unitData2Path" + constrInd);
  }
}

static void createReadyPathConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                                   GRBVar &bufOp, GRBVar &bufRdyOp,
                                   GRBVar &bufNSlots, double period,
                                   unsigned constrInd, double bufDelay = 0.0,
                                   double ctrlBufDelay = 0.1) {
  modelBuf.addConstr(t1 <= period, "readyChStart" + std::to_string(constrInd));
  modelBuf.addConstr(t2 <= period, "readyChEnd" + std::to_string(constrInd));
  modelBuf.addConstr(t2 >= t1 - 2 * period * bufRdyOp + ctrlBufDelay * bufOp,
                     "readyChPath" + std::to_string(constrInd));
  modelBuf.addConstr(t2 >= bufDelay,
                     "readyPath1Buf" + std::to_string(constrInd));
  modelBuf.addConstr(t2 >= ctrlBufDelay * bufOp,
                     "readyPath2Buf" + std::to_string(constrInd));
  modelBuf.addConstr(bufNSlots >= bufOp + bufRdyOp,
                     "readySlotsSum" + std::to_string(constrInd));
}

static void createValidPathConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                                   GRBVar &bufValOp, GRBVar &bufRdyOp,
                                   GRBVar &bufOp, GRBVar &bufNSlots,
                                   double period, unsigned constrInd,
                                   double bufDelay = 0.0,
                                   double ctrlBufDelay = 0.1) {
  modelBuf.addConstr(t1 <= period, "validChStart" + std::to_string(constrInd));
  modelBuf.addConstr(t2 <= period, "validChEnd" + std::to_string(constrInd));
  modelBuf.addConstr(t2 >= t1 - 2 * period * bufValOp + ctrlBufDelay * bufRdyOp,
                     "validChPath" + std::to_string(constrInd));
  modelBuf.addConstr(t2 >= bufDelay,
                     "validPat1hBuf" + std::to_string(constrInd));
  modelBuf.addConstr(t2 >= ctrlBufDelay * bufRdyOp,
                     "validPath2Buf" + std::to_string(constrInd));
  //  buffer consistency constraints
  modelBuf.addConstr(bufValOp == bufOp,
                     "validBufConsist" + std::to_string(constrInd));
}

// create control path constraints through a unit
static void createCtrlPathConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                                  double delay, std::string constrInd,
                                  std::string type = "valid") {
  modelBuf.addConstr(t2 >= t1 + delay, type + "Path" + constrInd);
}

// create elasticity constraints w.r.t channels
static void createElasticityConstrs(GRBModel &modelBuf, GRBVar &t1, GRBVar &t2,
                                    GRBVar &bufOp, GRBVar &bufNSlots,
                                    GRBVar &hasBuf, unsigned cstCoef,
                                    double period, unsigned constrInd) {
  modelBuf.addConstr(t2 >= t1 - cstCoef * bufOp);
  modelBuf.addConstr(bufNSlots >= bufOp);
  modelBuf.addConstr(hasBuf >= 0.01 * bufNSlots);
}

// create elasticity constraints w.r.t units
static void createElasticityConstrs(GRBModel &modelBuf, GRBVar &tIn,
                                    GRBVar &tOut, double delay, double latency,
                                    double period, std::string constrInd) {
  modelBuf.addConstr(tOut >= 1 + tIn);
}

/// Throughput constraints over a channel
static void createThroughputConstrs(GRBModel &modelBuf, GRBVar &retSrc,
                                    GRBVar &retDst, GRBVar &thrptTok,
                                    GRBVar &thrpt, GRBVar &isOp,
                                    GRBVar &bufNSlots, const int tok,
                                    unsigned constrInd) {
  modelBuf.addConstr(retSrc - retDst + thrptTok == tok);
  modelBuf.addConstr(thrpt + isOp - thrptTok <= 1);
  modelBuf.addConstr(thrptTok + thrpt + isOp - bufNSlots <= 1);
  modelBuf.addConstr(thrptTok - bufNSlots <= 0);
}

/// Throughput constraints over a unit, relugated by the input and output
/// retiming values
static void createThroughputConstrs(GRBModel &modelBuf, GRBVar &retIn,
                                    GRBVar &retout, GRBVar &thrpt,
                                    double latency, unsigned constrInd) {
  if (latency > 0.1)
    modelBuf.addConstr(retout - retIn == latency * thrpt);
}

static LogicalResult
createModelConstraints(GRBModel &modelBuf, GRBVar &thrpt, double targetCP,
                       std::map<Operation *, UnitVar> &unitVars,
                       std::map<Value *, ChannelVar> &channelVars,
                       std::map<std::string, UnitInfo> unitInfo) {
  for (auto [ind, chVarMap] : llvm::enumerate(channelVars)) {
    auto &[ch, chVars] = chVarMap;
    // llvm::errs() << *ch << "\n";

    // update the model to get the lower bound and upper
    // bound of the vars
    modelBuf.update();

    // // place buffers if maxinum buffer slots is larger then 0 and the channel
    // is
    // // selected
    // if (chVars.bufNSlots.get(GRB_DoubleAttr_UB) <= 0)
    //   continue;

    Operation *srcOp = ch->getDefiningOp();
    Operation *dstOp = *(ch->getUsers().begin());
    GRBVar &t1 = chVars.tDataIn;
    GRBVar &t2 = chVars.tDataOut;

    GRBVar &tElas1 = chVars.tElasIn;
    GRBVar &tElas2 = chVars.tElasOut;

    GRBVar &bufOp = chVars.bufIsOp;
    GRBVar &bufNSlots = chVars.bufNSlots;
    GRBVar &hasBuf = chVars.hasBuf;
    GRBVar &thrptTok = chVars.thrptTok;

    // GRBVar &tValid1 = chVars.tValidIn;
    // GRBVar &tValid2 = chVars.tValidOut;
    // GRBVar &tReady1 = chVars.tReadyIn;
    // GRBVar &tReady2 = chVars.tReadyOut;

    // GRBVar &bufValOp = chVars.valbufIsOp;
    // GRBVar &bufRdyOp = chVars.readybufIsOp;

    GRBVar &retSrc = unitVars[srcOp].retOut;

    int tok = isBackEdge(srcOp, dstOp) ? 1 : 0;
    createPathConstrs(modelBuf, t1, t2, bufOp, targetCP, ind);
    // createDataPathConstrs(modelBuf, t1, t2, bufOp, bufRdyOp, targetCP, ind);
    // // ready signal is ooposite to the channel direction
    // createReadyPathConstrs(modelBuf, tReady2, tReady1, bufOp, bufRdyOp,
    //                        bufNSlots, targetCP, ind);
    // createValidPathConstrs(modelBuf, tValid1, tValid2, bufValOp, bufRdyOp,
    //                        bufOp, bufNSlots, targetCP, ind);
    createElasticityConstrs(modelBuf, tElas1, tElas2, bufOp, bufNSlots, hasBuf,
                            unitVars.size() + 1, targetCP, ind);

    if (chVars.select && chVars.bufNSlots.get(GRB_DoubleAttr_UB) > 0)
      if (unitVars.count(dstOp) > 0) {
        GRBVar &retDst = unitVars[dstOp].retIn;
        createThroughputConstrs(modelBuf, retSrc, retDst, thrptTok, thrpt,
                                bufOp, bufNSlots, tok, ind);
      }
  }

  for (auto [ind, uVarMap] : llvm::enumerate(unitVars)) {
    auto &[op, unitVar] = uVarMap;

    double delayData = getCombinationalDelay(op, unitInfo, "data");
    double delayValid = getCombinationalDelay(op, unitInfo, "valid");
    double delayReady = getCombinationalDelay(op, unitInfo, "ready");
    double latency = getUnitLatency(op, unitInfo);

    // llvm::errs() << "unit: " << *op << " delaydata " << delayData
    //              << " delayValid " << delayValid << " delayReady " <<
    //              delayReady
    //              << " latency " << latency << "\n";

    GRBVar &retIn = unitVars[op].retIn;
    GRBVar &retOut = unitVar.retOut;

    if (unitVar.select)
      createThroughputConstrs(modelBuf, retIn, retOut, thrpt, latency, ind);

    // iterate over all paths throughout the units
    // temporally iterate all input port to all output port
    // for a unit

    for (auto inChVal : op->getOperands()) {
      // only check the selected input channels
      unsigned channelInd = 0;
      if (auto inCh = inChannelMap(channelVars, inChVal).value();
          inCh != nullptr) {
        if (channelVars[inCh].bufNSlots.get(GRB_DoubleAttr_UB) <= 0)
          continue;
        double inPortDelay = getPortDelay(inChVal, unitInfo, "out");

        GRBVar &tIn = channelVars[inCh].tDataOut;
        // GRBVar &tValidIn = channelVars[inCh].tValidOut;
        // GRBVar &tReadyIn = channelVars[inCh].tReadyOut;
        GRBVar &tElasIn = channelVars[inCh].tElasOut;

        for (auto outChVal : op->getResults())
          // check all the output channels
          if (auto outCh = inChannelMap(channelVars, outChVal).value();
              outCh != nullptr) {
            if (channelVars[outCh].bufNSlots.get(GRB_DoubleAttr_UB) <= 0)
              continue;
            double outPortDelay = getPortDelay(outChVal, unitInfo, "in");

            GRBVar &tOut = channelVars[outCh].tDataIn;
            // GRBVar &tValidOut = channelVars[outCh].tValidIn;
            // GRBVar &tReadyOut = channelVars[outCh].tReadyIn;
            GRBVar &tElasOut = channelVars[outCh].tElasIn;

            std::string constrName =
                std::to_string(ind) + "_" + std::to_string(channelInd);
            channelInd++;
            createPathConstrs(modelBuf, tIn, tOut, delayData, latency, targetCP,
                              constrName);
            // createDataPathConstrs(modelBuf, tIn, tOut, delayData, latency,
            //                       targetCP, constrName, inPortDelay,
            //                       outPortDelay);
            // createCtrlPathConstrs(modelBuf, tValidIn, tValidOut, delayValid,
            //                       constrName, "valid");
            // createCtrlPathConstrs(modelBuf, tReadyOut, tReadyIn, delayReady,
            //                       constrName, "ready");
            createElasticityConstrs(modelBuf, tElasIn, tElasOut, delayData,
                                    latency, targetCP, constrName);
          }
      }
    }
  }
  return success();
}

static LogicalResult
setCustomizedConstraints(GRBModel &modelBuf,
                         std::map<Value *, ChannelVar> channelVars,
                         std::map<Value *, ChannelBufProps> &channelBufProps,
                         std::map<Value *, Result> &res) {
  for (auto chVarMap : channelVars) {
    auto &[ch, chVars] = chVarMap;
    channelVars[ch].bufNSlots.set(GRB_DoubleAttr_LB, 0);
    channelVars[ch].bufNSlots.set(GRB_DoubleAttr_UB, 1e10);
    // set min value of the buffer
    if (channelBufProps[ch].minNonTrans > 0) {
      channelVars[ch].bufIsOp.set(GRB_DoubleAttr_LB, 1);
      channelVars[ch].bufNSlots.set(GRB_DoubleAttr_LB,
                                    channelBufProps[ch].minNonTrans);
    } else if (channelBufProps[ch].minTrans > 0) {

      channelVars[ch].bufIsOp.set(GRB_DoubleAttr_UB, 0);
      channelVars[ch].bufNSlots.set(GRB_DoubleAttr_LB,
                                    channelBufProps[ch].minTrans);
    }

    // set max value of the buffer
    if (channelBufProps[ch].maxNonTrans.has_value())
      channelVars[ch].bufNSlots.set(GRB_DoubleAttr_UB,
                                    channelBufProps[ch].maxNonTrans.value());

    if (channelBufProps[ch].maxTrans.has_value())
      channelVars[ch].bufNSlots.set(GRB_DoubleAttr_UB,
                                    channelBufProps[ch].maxNonTrans.value());
  }
  for (auto &[ch, result] : res) {
    channelVars[ch].bufNSlots.set(GRB_DoubleAttr_LB, res[ch].numSlots);
    channelVars[ch].bufIsOp.set(GRB_DoubleAttr_LB, res[ch].opaque);
    channelVars[ch].bufIsOp.set(GRB_DoubleAttr_UB, res[ch].opaque);
  }

  return success();
}

static void createModelObjective(GRBModel &modelBuf, GRBVar &thrpt,
                                 std::map<Value *, ChannelVar> channelVars) {
  GRBLinExpr objExpr = thrpt;

  double lumbdaCoef1 = 1e-5;
  double lumbdaCoef2 = 1e-6;
  for (auto &[_, chVar] : channelVars) {
    objExpr -= lumbdaCoef1 * chVar.hasBuf + lumbdaCoef2 * chVar.bufNSlots;
  }

  modelBuf.setObjective(objExpr, GRB_MAXIMIZE);
}
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

static LogicalResult
setChannelBufProps(std::vector<Value> &channels,
                   std::map<Value *, ChannelBufProps> &ChannelBufProps,
                   std::map<std::string, UnitInfo> &unitInfo) {
  for (auto &ch : channels) {
    Operation *srcOp = ch.getDefiningOp();
    Operation *dstOp = *(ch.getUsers().begin());
    llvm::errs() << *srcOp << "\n";
    llvm::errs() << *dstOp << "\n";

    std::string srcName = getOperationShortStrName(srcOp);
    std::string dstName = getOperationShortStrName(dstOp);
    // set merge with multiple input to have at least one
    // transparent buffer
    if (isa<handshake::MergeOp>(srcOp) && srcOp->getNumOperands() > 1) {
      if (getPortInd(srcOp, ch) == 1)
        ChannelBufProps[&ch].minTrans = 1;
    }

    if (isa<handshake::MuxOp>(srcOp))
      ChannelBufProps[&ch].minTrans = 1;

    // TODO: set selectOp always select the frequent input
    if (isa<arith::SelectOp>(srcOp))
      if (srcOp->getOperand(0) == ch) {
        ChannelBufProps[&ch].maxTrans = 0;
        ChannelBufProps[&ch].maxNonTrans = 1;
      }

    if (isa<handshake::MemoryControllerOp>(srcOp) ||
        isa<handshake::MemoryControllerOp>(dstOp)) {
      ChannelBufProps[&ch].maxNonTrans = 0;
      ChannelBufProps[&ch].maxTrans = 0;
    }

    // set channel buffer properties w.r.t to input file
    if (unitInfo.count(srcName) > 0) {
      ChannelBufProps[&ch].minTrans += ChannelBufProps[&ch].minTrans,
          unitInfo[srcName].outPortTransBuf;
      ChannelBufProps[&ch].minNonTrans += unitInfo[srcName].outPortOpBuf;
    }

    if (unitInfo.count(dstName) > 0) {
      ChannelBufProps[&ch].minTrans += unitInfo[dstName].inPortTransBuf;
      ChannelBufProps[&ch].minNonTrans += unitInfo[dstName].inPortOpBuf;
    }

    if (ChannelBufProps[&ch].minTrans > 0 &&
        ChannelBufProps[&ch].minNonTrans > 0)
      return failure(); // cannot satisfy the constraint
  }
  return success();
}

LogicalResult buffer::placeBufferInCFDFCircuit(handshake::FuncOp funcOp,
                                               std::vector<Value> allChannels,
                                               CFDFC &CFDFCircuit,
                                               std::map<Value *, Result> &res,
                                               double targetCP) {
  std::vector<double> delayData, delayValid, delayReady;
  std::string timefile = "/home/yuxuan/Downloads/default.json";

  std::map<std::string, UnitInfo> unitInfo;
  std::map<Value *, ChannelBufProps> ChannelBufProps;

  parseJson(timefile, unitInfo);
  // for (auto &[key, time] : unitInfo) {
  //   llvm::errs() << key << "================\n";
  //   time.print();
  // }

  // load the buffer information of the units to channel
  if (failed(setChannelBufProps(allChannels, ChannelBufProps, unitInfo)))
    return failure();

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
  std::map<Operation *, UnitVar> unitVars;
  std::map<Value *, ChannelVar> channelVars;

  // create the variable to noate the overall circuit throughput
  GRBVar circtThrpt = modelBuf.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "thrpt");

  // initialize variables
  initVarsInMILP(funcOp, modelBuf, CFDFCircuit.units, CFDFCircuit.channels,
                 allChannels, unitVars, channelVars, unitInfo);
  llvm::errs() << unitVars.size() << "\n";
  llvm::errs() << channelVars.size() << "\n";

  llvm::errs() << "init variables successfully!\n";
  //  define customized constraints
  setCustomizedConstraints(modelBuf, channelVars, ChannelBufProps, res);
  llvm::errs() << "set customized constraints successfully!\n";

  //  create constraints
  createModelConstraints(modelBuf, circtThrpt, targetCP, unitVars, channelVars,
                         unitInfo);
  llvm::errs() << "create constraints successfully!\n";

  createModelObjective(modelBuf, circtThrpt, channelVars);
  llvm::errs() << "set objects successfully!\n";
  modelBuf.write("/home/yuxuan/Downloads/model.lp");
  modelBuf.optimize();

  if (modelBuf.get(GRB_IntAttr_Status) != GRB_OPTIMAL ||
      circtThrpt.get(GRB_DoubleAttr_X) <= 0) {
    llvm::errs() << "No optimal sulotion found\n";
    return failure();
  }

  llvm::errs() << "Circuit throughput: " << circtThrpt.get(GRB_DoubleAttr_X)
               << "\n";

  // load answer to the result
  for (auto &[ch, chVarMap] : channelVars) {
    auto srcOp = ch->getDefiningOp();
    auto dstOp = getUserOp(*ch);

    llvm::errs() << chVarMap.tDataIn.get(GRB_StringAttr_VarName) << " " << *ch
                 << " dataIn: " << chVarMap.tDataIn.get(GRB_DoubleAttr_X)
                 << " dataOut: " << chVarMap.tDataOut.get(GRB_DoubleAttr_X)
                 << "\n";
    llvm::errs() << "opaque: " << chVarMap.bufIsOp.get(GRB_DoubleAttr_X) << "; "
                 << "; total: " << chVarMap.bufNSlots.get(GRB_DoubleAttr_X)
                 << " throughput: " << chVarMap.thrptTok.get(GRB_DoubleAttr_X)
                 << " retime in: "
                 << unitVars[srcOp].retOut.get(GRB_DoubleAttr_X)
                 << " retime out: "
                 << unitVars[srcOp].retIn.get(GRB_DoubleAttr_X) << "\n ";

    if (chVarMap.hasBuf.get(GRB_DoubleAttr_X) > 0) {
      Result result;
      result.numSlots =
          static_cast<int>(chVarMap.bufNSlots.get(GRB_DoubleAttr_X) + 0.5);
      result.opaque = chVarMap.bufIsOp.get(GRB_DoubleAttr_X) > 0;
      // result.transparent = chVarMap.readybufIsOp.get(GRB_DoubleAttr_X) > 0;
      if (res.count(ch) > 0)
        res[ch] = res[ch] + result;
      else
        res[ch] = result;
    }
  }
  return success();
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
}