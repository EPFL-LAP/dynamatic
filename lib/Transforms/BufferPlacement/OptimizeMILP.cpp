//===- OptimizeMILP.cpp - optimize MILP model over CFDFC  -------*- C++ -*-===//
//
// This file implements the MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/OptimizeMILP.h"
#include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"
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
#include "gurobi_c++.h"
#include <fstream>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

void buffer::DataflowCircuit::printCircuits() {
  llvm::errs() << "===========================\n";
  for (auto unit : units) {
    llvm::errs() << "operation: " << *(unit) << "\n";
  }
}

std::vector<std::vector<float>>
buffer::DataflowCircuit::readInfoFromFile(const std::string &filename) {
  std::vector<std::vector<float>> info;

  std::ifstream file(filename);
  assert(file.is_open() && "Error opening delay info file");

  std::string line;
  while (std::getline(file, line)) {
    std::vector<float> row;
    std::istringstream iss(line);
    std::string value;

    while (std::getline(iss, value, ',')) {
      float num = std::stof(value);
      row.push_back(num);
    }

    assert(!row.empty() && "Error reading delay info file");
    info.push_back(row);
  }

  file.close();

  return info;
}

double buffer::DataflowCircuit::getTimeInfo(Operation *op, std::string infoName) {
  std::string opName = op->getName().getStringRef().str();
  size_t dotPos = opName.find('.');
  std::string opType = opName.substr(dotPos+1);
  // llvm::errs() << "opType: " << opType << 
  //   "delay : " << this->delayInfo[compNameToIndex[opType]][0] << "\n";
  if (infoName == "delay")
    return this->delayInfo[compNameToIndex[opType]][0];
  return this->latencyInfo[compNameToIndex[opType]][1];
}

static void initVarsInMILP(GRBModel &modelBuf,
                           std::vector<Operation *> &units,
                           std::vector<Value> &channels,
                           std::map<Operation *, UnitVar> &unitVars, 
                           std::map<Value *, ChannelVar> &channelVars) {
  // create variables

  for (auto [ind, unit] : llvm::enumerate(units)) {
    // create variables for each unit
    llvm::errs() << "unit: " << *unit << "\n";

    UnitVar unitVar;
    unitVar.retIn = modelBuf.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "retIn"+std::to_string(ind));
    unitVar.retOut = modelBuf.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "retOut"+std::to_string(ind));
    unitVar.tElasIn = modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "tElasIn"+std::to_string(ind));
    unitVar.tElasOut = modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "tElasOut"+std::to_string(ind));
    unitVars[unit] = unitVar;
  }

  for (auto [ind, val] : llvm::enumerate(channels)) {
    llvm::errs() << "channel: " << val << "\n";
    ChannelVar channelVar;
    channelVar.tIn = modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "tIn"+std::to_string(ind));
    channelVar.tOut = modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "tOut"+std::to_string(ind));
    channelVar.thrptTok = modelBuf.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "thrpt"+std::to_string(ind));
    channelVar.bufFlop = modelBuf.addVar(0.0, 1.0, 0.0, GRB_BINARY, "bufFlop"+std::to_string(ind));
    channelVar.bufNSlots = modelBuf.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER, "bufNSlots"+std::to_string(ind));
    channelVar.hasBuf = modelBuf.addVar(0.0, 1.0, 0.0, GRB_BINARY, "hasBuf"+std::to_string(ind));
    
    channelVars[&val] = channelVar;
  }
}

static void createPathConstrs(GRBModel &modelBuf,
                              GRBVar &t1, GRBVar &t2, GRBVar &bufFlop,
                              double period,
                              unsigned constrInd) {
  modelBuf.addConstr(t1 <= period, "chStart"+std::to_string(constrInd));
  modelBuf.addConstr(t2 <= period, "chEnd"+std::to_string(constrInd));
  modelBuf.addConstr(t2>= t1 - 2*period*bufFlop, "path"+std::to_string(constrInd));
}

static void createPathConstrs(GRBModel &modelBuf,
                             GRBVar &tIn, GRBVar &tOut,
                              double delay, double latency, double period,
                              unsigned constrInd) {
  if (latency == 0.0)
    modelBuf.addConstr(tOut >= delay + tIn, "unitPath"+std::to_string(constrInd));
  else
    modelBuf.addConstr(tIn <= period, "unitPath"+std::to_string(constrInd));

  
}

static void createElasticityConstrs(GRBModel &modelBuf,
                              GRBVar &tIn, GRBVar &tOut, 
                              double delay, double latency, double period,
                              unsigned constrInd) {
  modelBuf.addConstr(tOut >= 1 + tIn, "unitElas"+std::to_string(constrInd));
}

static void createElasticityConstrs(GRBModel &modelBuf,
                              GRBVar &t1, GRBVar &t2, GRBVar &bufFlop,
                              GRBVar &bufNSlots, GRBVar &hasBuf,
                              unsigned cstCoef, double period, unsigned constrInd) {
  modelBuf.addConstr(t2 >= t1 - cstCoef*bufFlop, "chElas"+std::to_string(constrInd));
  modelBuf.addConstr(bufNSlots >= bufFlop, "bufNSlots"+std::to_string(constrInd));
  modelBuf.addConstr(hasBuf >= 0.01*bufNSlots, "hasBuf"+std::to_string(constrInd));
}

static void createThroughputConstrs(GRBModel &modelBuf,
    GRBVar &retSrc, GRBVar &retDst, GRBVar &thrptTok, GRBVar &thrpt,
    GRBVar &hasFlop, GRBVar &bufNSlots,
    const int tok, unsigned constrInd) {
    modelBuf.addConstr(retSrc - retDst + thrptTok == tok,
        "ret"+std::to_string(constrInd)+"_1");
    modelBuf.addConstr(thrpt + hasFlop - thrptTok <= 1, 
        "ret"+std::to_string(constrInd)+"_2");
    modelBuf.addConstr(thrptTok + thrpt + hasFlop - bufNSlots <= 1, 
        "ret"+std::to_string(constrInd)+"_3");
    modelBuf.addConstr(thrptTok - bufNSlots <= 0, 
        "ret"+std::to_string(constrInd)+"_4");
}

static void createThroughputConstrs(GRBModel &modelBuf,
    GRBVar &retSrc, GRBVar &retDst, GRBVar &thrpt,
    double latency,  unsigned constrInd) {
  if (latency != 0.0)
    modelBuf.addConstr(retDst - retSrc == thrpt, 
        "ret"+std::to_string(constrInd)+"_5");
}

static void createStrategyConstrs(GRBModel &modelBuf,
          ChannelConstraints &outConstr, 
          ChannelVar &channelVar,
          unsigned constrInd) {
  if (!outConstr.bufferizable) {
    modelBuf.addConstr(channelVar.hasBuf == 0,
       "customConstr" + std::to_string(constrInd) + "_0");
    return;
  }
  if (outConstr.minSlots.has_value()) {
    // llvm::errs() << "force insert buf\n";
    modelBuf.addConstr(channelVar.hasBuf >= 1,
        "customConstr" + std::to_string(constrInd) + "_1");
    modelBuf.addConstr(channelVar.bufNSlots >= outConstr.minSlots.value(),
        "customConstr" + std::to_string(constrInd) + "_2");
  }
  if (outConstr.maxSlots.has_value()) {
    modelBuf.addConstr(channelVar.bufNSlots <= outConstr.maxSlots.value(),
        "customConstr" + std::to_string(constrInd) + "_3");
  }
  if (!outConstr.nonTransparentAllowed) {
    modelBuf.addConstr(channelVar.bufFlop==1, 
      "customConstr"+ std::to_string(constrInd) + "_4");
  }
}

LogicalResult buffer::DataflowCircuit::
createModelConstraints(BufferPlacementStrategy &strategy,
  GRBModel &modelBuf, GRBVar &thrpt,
  std::map<Operation *, UnitVar> &unitVars,
  std::map<Value *, ChannelVar> &channelVars) {
  // create constraints
  for (auto [ind, chPair] : llvm::enumerate(channelVars)) {
      Operation *srcOp = chPair.first->getDefiningOp();
      Operation *dstOp = getUserOp(chPair.first);

      GRBVar &t1 = chPair.second.tOut;
      GRBVar &t2 = chPair.second.tIn;
      GRBVar &bufFlop = chPair.second.bufFlop;
      GRBVar &bufNSlots = chPair.second.bufNSlots;
      GRBVar &hasBuf = chPair.second.hasBuf;
      GRBVar &thrptTok = chPair.second.thrptTok;

      GRBVar &retSrc = unitVars[srcOp].retOut;

      int tok = isBackEdge(chPair.first) ? 1 : 0;
      createPathConstrs(modelBuf, t1, t2, bufFlop, this->targetCP, ind);
      createElasticityConstrs(modelBuf, t1, t2, bufFlop, bufNSlots, hasBuf, 
          this->units.size()+1, this->targetCP, ind);
      if (unitVars.count(dstOp) > 0) {
        GRBVar &retDst = unitVars[dstOp].retIn;
        createThroughputConstrs(modelBuf, retSrc, retDst, thrptTok, thrpt,
            bufFlop, bufNSlots, tok, ind);

      }
      ChannelConstraints constr = strategy.getChannelConstraints(chPair.first);
      createStrategyConstrs(modelBuf, constr, chPair.second, ind);

  }

  for (auto [ind, uPair] : llvm::enumerate(unitVars)) {
    double delay = getTimeInfo(uPair.first, "delay");
    double latency = getTimeInfo(uPair.first, "latency");

    GRBVar &ret1 = uPair.second.retOut;
    for (auto Oprand : uPair.first->getOperands()) {

      // check all the input channels
      if (channelVars.count(&Oprand) > 0) {
        Operation *sucOp = getUserOp(Oprand);
        GRBVar &ret2 = unitVars[sucOp].retIn;

        GRBVar &tIn = channelVars[&Oprand].tOut;
        GRBVar &tElasIn = uPair.second.tElasIn;
        for (auto result : uPair.first->getResults()) 
          // check all the output channels
          if (channelVars.count(&result) > 0) {
            llvm::errs() << "unit " << *uPair.first <<"\n";
            GRBVar &tOut = channelVars[&result].tIn;
            GRBVar &tElasOut = uPair.second.tElasOut;
            createPathConstrs(modelBuf, tIn, tOut, 
                            delay, latency, this->targetCP, ind);
            createElasticityConstrs(modelBuf, tElasIn, tElasOut, 
                            delay, latency, this->targetCP, ind);
            createThroughputConstrs(modelBuf, ret1, ret2, thrpt, latency, ind);
          }
      }
    }
  }
  return success();
}

static void createModelObjective(GRBModel &modelBuf, GRBVar &thrpt,
    std::map<Value *, ChannelVar> channelVars){
  GRBLinExpr objExpr = thrpt; 

  double lumbdaCoef = 1e-7;
  for (auto pair : channelVars) {
    objExpr -= lumbdaCoef * pair.second.bufNSlots;
  }

  modelBuf.setObjective(objExpr, GRB_MAXIMIZE);
  }

LogicalResult 
buffer::DataflowCircuit::createMILPModel(BufferPlacementStrategy &strategy,
  std::map<Value *, Result> &res) {
  // create a Gurobi environment
  GRBEnv env = GRBEnv();
  env.set(GRB_IntParam_LogToConsole, false);
  env.set(GRB_IntParam_OutputFlag, 0);
  // env.set("LogFile", "mip1.log");
  env.start();
  GRBModel modelBuf = GRBModel(env);

  this->delayInfo = 
    this->readInfoFromFile(this->delayFile);

  this->latencyInfo = 
    this->readInfoFromFile(this->latencyFile);

  // create variables
  std::map<Operation *, UnitVar> unitVars;
  std::map<Value *, ChannelVar> channelVars;

  GRBVar circtThrpt = modelBuf.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "thrpt");

  // initialize variables
  initVarsInMILP(modelBuf, this->units, this->channels, 
                 unitVars, channelVars);
  createModelConstraints(strategy, modelBuf, circtThrpt, unitVars, channelVars);
  createModelObjective(modelBuf, circtThrpt, channelVars);

  modelBuf.optimize();
  if (modelBuf.get(GRB_IntAttr_Status) != GRB_OPTIMAL ||
      circtThrpt.get(GRB_DoubleAttr_X) <= 0) 
      llvm::errs() << "No optimal sulotion found" << "\n";   

  // load answer to the result
  for (auto chPair : channelVars) {
    llvm::errs() << *chPair.first << " " <<
      chPair.second.hasBuf.get(GRB_DoubleAttr_X) << "\n";
    if (chPair.second.hasBuf.get(GRB_DoubleAttr_X) > 0) {
      Result result;
      result.numSlots = static_cast<int>
            (chPair.second.bufNSlots.get(GRB_DoubleAttr_X));
      if (chPair.second.bufFlop.get(GRB_DoubleAttr_X) > 0) {
        // llvm::errs() << *chPair.first << " insert " <<
        // "transparent " << chPair.second.bufNSlots.get(GRB_DoubleAttr_X);
        result.transparent = true;
        
        
      }
      else if (chPair.second.bufNSlots.get(GRB_DoubleAttr_X) > 0) {
        // llvm::errs() << *chPair.first << " insert " <<
        // "nontransparent " << chPair.second.bufNSlots.get(GRB_DoubleAttr_X);
        result.transparent = false;
        // res[chPair.first] = result;
      }
      res[chPair.first] = result;
    }
  }
  return success();
}