//===- UtilsForExtractMG.cpp - Extract MG for optimization* C++ ---------*-===//
//
// This file implements function supports for CFDFCircuit extraction.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"
#include "dynamatic/Conversion/StandardToHandshakeFPGA18.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"
#include "gurobi_c++.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

static bool isNumber(const std::string &str) {
  for (char c : str)
    if (!isdigit(c) && c != ' ')
      return false;
  return true;
}

int buffer::getBBIndex(Operation *op) {
  for (auto attr : op->getAttrs()) {
    if (attr.getName() == BB_ATTR)
      return dyn_cast<IntegerAttr>(attr.getValue()).getValue().getZExtValue();
  }
  return -1;
}

bool buffer::isEntryOp(Operation *op) {
  for (auto operand : op->getOperands())
    if (!operand.getDefiningOp())
      return true;
  return false;
}

bool buffer::isBackEdge(Operation *opSrc, Operation *opDst) {
  if (opDst->isProperAncestor(opSrc))
    return true;
  return false;
}

LogicalResult buffer::readSimulateFile(const std::string &fileName,
                              std::map<ArchBB *, unsigned> &archs,
                              std::map<int, bool> &bbs) {
  std::ifstream inFile(fileName);

  if (!inFile)
    return failure();

  std::string line;

  // Skip the header line
  std::getline(inFile, line);

  while (std::getline(inFile, line)) {
    std::istringstream iss(line);
    ArchBB *arch = new ArchBB();
    ArchBB *pArch = arch;

    std::string token;
    std::getline(iss, token, ',');

    if(!isNumber(token))
      return failure();
    arch->srcBB = std::stoi(token);
    
    std::getline(iss, token, ',');
    if(!isNumber(token))
      return failure();
    arch->dstBB  = std::stoi(token);
    
    std::getline(iss, token, ',');
    if(!isNumber(token))
      return failure();
    arch->execFreq  = std::stoi(token);

    std::getline(iss, token, ',');
    if(!isNumber(token))
      return failure();
    arch->isBackEdge = std::stoi(token) == 1 ? true : false;

    archs[pArch] = false;
    if (bbs.count(arch->srcBB) == 0)
      bbs[arch->srcBB] = false;
    if (bbs.count(arch->dstBB) == 0)
      bbs[arch->dstBB] = false;
  }
  return success();
}

static int initVarInMILP(GRBModel &modelMILP, std::map<int, GRBVar> &sBB,
                         std::map<std::string, GRBVar> &sArc,
                         std::vector<ArchBB *> &archNames,
                         std::vector<unsigned> &bbNames) {
  int cstMaxN = 0;

  for (unsigned bb : bbNames) {
    // define variables for basic blocks selection
    sBB[bb] =
        modelMILP.addVar(0.0, 1, 0.0, GRB_BINARY, "sBB_" + std::to_string(bb));
    for (auto arch : archNames)
      if (arch->srcBB == bb) {
        // define variables for edges selection
        std::string arcName = "sArc_" + std::to_string(arch->srcBB) + "_" +
                              std::to_string(arch->dstBB);
        sArc[arcName] = modelMILP.addVar(0.0, 1, 0.0, GRB_BINARY, arcName);
        cstMaxN = std::max(cstMaxN, arch->execFreq);
      }
  }
  // define maximum execution cycles variables
  sArc["valExecN"] = modelMILP.addVar(0, cstMaxN, 0.0, GRB_INTEGER, "valExecN");

  return cstMaxN;
}

static void setObjective(GRBModel &modelMILP,
                         std::map<std::string, GRBVar> &sArc) {
  GRBQuadExpr objExpr;
  // cost function: max: \sum_{e \in E} s_e * execFreq_e
  for (auto pair : sArc) {
    if (pair.first == "valExecN")
      continue;
    auto sE = pair.second;
    objExpr += sArc["valExecN"] * sE;
  }
  modelMILP.setObjective(objExpr, GRB_MAXIMIZE);
}

static ArchBB *findArchWithVarName(const std::string &varName,
                                   std::vector<ArchBB *> &archs) {
  for (auto arch : archs) {
    std::string arcName = "sArc_" + std::to_string(arch->srcBB) + "_" +
                          std::to_string(arch->dstBB);
    if (arcName == varName)
      return arch;
  }
  return nullptr;
}

static std::vector<std::string>
getBBsInArcVars(int bb, std::map<std::string, GRBVar> &sArc,
                std::vector<ArchBB *> &archs) {
  std::vector<std::string> varNames;

  for (auto pair : sArc) {
    if (pair.first == "valExecN")
      continue;
    auto arch = findArchWithVarName(pair.first, archs);
    if (arch && arch->srcBB == bb) 
      varNames.push_back(pair.first);
    
  }
  return varNames;
}

static std::vector<std::string>
getBBsOutArcVars(int bb, std::map<std::string, GRBVar> &sArc,
                 std::vector<ArchBB *> &archs) {
  std::vector<std::string> varNames;
  for (auto pair : sArc) {
    if (pair.first == "valExecN")
      continue;
    auto arch = findArchWithVarName(pair.first, archs);
    if (arch && arch->dstBB == bb) 
      varNames.push_back(pair.first);
    
  }
  return varNames;
}

static LogicalResult setEdgeConstrs(GRBModel &modelMILP, int cstMaxN,
                           std::map<std::string, GRBVar> &sArc,
                           std::vector<ArchBB *> &archs) {

  auto valExecN = sArc["valExecN"];
  GRBLinExpr backEdgeConstr;

  for (auto [constrInd, pair] : llvm::enumerate(sArc)) {
    if (pair.first == "valExecN")
      continue;
    auto arcEntity = findArchWithVarName(pair.first, archs);
    if (!arcEntity)
      return failure();
    unsigned N_e = arcEntity->execFreq;
    auto S_e = pair.second;
    // for each edge e: N <= S_e x N_e + (1-S_e) x cstMaxN
    modelMILP.addConstr(valExecN <= S_e * N_e + (1 - S_e) * cstMaxN,
                        "cN" + std::to_string(constrInd));
    // Only select one back archs: for each bb \in Back(CFG): sum(S_e) = 1
    if (arcEntity->isBackEdge) 
      backEdgeConstr += S_e;
    
  }
  modelMILP.addConstr(backEdgeConstr == 1, "cBack");
  return success();
}

static void setBBConstrs(GRBModel &modelMILP, std::map<int, GRBVar> &sBB,
                         std::map<std::string, GRBVar> &sArc,
                         std::vector<ArchBB *> &archs) {

  for (auto [constrInd, pair] : llvm::enumerate(sBB)) {
    // only 1 input arch if bb is selected;
    // no input arch if bb is not selected
    GRBLinExpr constraintInExpr;
    auto inArcs = getBBsInArcVars(pair.first, sArc, archs);
    for (auto arch : inArcs)
      if (sArc.count(arch) > 0)
        constraintInExpr += sArc[arch];
    modelMILP.addConstr(constraintInExpr == pair.second,
                        "cIn" + std::to_string(constrInd));

    // only 1 output arch if bb is selected;
    // no output arch if bb is not selected
    GRBLinExpr constraintOutExpr;
    auto outArcs = getBBsOutArcVars(pair.first, sArc, archs);
    for (auto arch : outArcs)
      if (sArc.count(arch) > 0)
        constraintOutExpr += sArc[arch];
    modelMILP.addConstr(constraintOutExpr == pair.second,
                        "cOut" + std::to_string(constrInd));
  }
};

bool buffer::isSelect(std::map<int, bool> &bbs, Value *val) {
  Operation *srcOp = val->getDefiningOp();
  Operation * dstOp;
  for (auto user : val->getUsers())
    dstOp = user;

  unsigned srcBB = getBBIndex(srcOp);
  unsigned dstBB = getBBIndex(dstOp);

  // if srcOp and dstOp are in the same BB, and the edge is not backedge
  // then the edge is selected depends on the BB
  if (srcBB == dstBB && bbs.count(srcBB) > 0)
    if (!isBackEdge(srcOp, dstOp))
      return bbs[srcBB];
  return false;
}

bool buffer::isSelect(std::map<ArchBB *, unsigned> &archs, Value *val) {
  Operation *srcOp = val->getDefiningOp();
  Operation *dstOp = val->getDefiningOp();

  unsigned srcBB = getBBIndex(srcOp);
  unsigned dstBB = getBBIndex(dstOp);

  for (auto pair : archs) 
    if (pair.first->srcBB == srcBB && pair.first->dstBB == dstBB && pair.second > 0)
      return true;
  return false;
}

int buffer::extractCFDFCircuit(std::map<ArchBB *, unsigned> &archs,
                               std::map<int, bool> &bbs) {
  // store variable names
  std::vector<ArchBB *> archNames;
  std::vector<unsigned> bbNames;

  for (auto pair : archs) {
    // pair.first->print();
    archNames.push_back(pair.first);
  }
  for (auto pair : bbs) {
    bbNames.push_back(pair.first);
  }

  // Create MILP model for CFDFCircuit extraction
  // Init a gurobi model
  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "mip1.log");
  env.start();
  GRBModel modelMILP = GRBModel(env);

  // Define variables
  std::map<std::string, GRBVar> sArc;
  std::map<int, GRBVar> sBB;

  int cstMaxN = initVarInMILP(modelMILP, sBB, sArc, archNames, bbNames);
  setObjective(modelMILP, sArc);
  if (failed(setEdgeConstrs(modelMILP, cstMaxN, sArc, archNames)))
    return -1;
  
  setBBConstrs(modelMILP, sBB, sArc, archNames);

  modelMILP.optimize();

  if (modelMILP.get(GRB_IntAttr_Status) != GRB_OPTIMAL ||
      sArc["valExecN"].get(GRB_DoubleAttr_X) <= 0)
    return -1;

  // load answer to the bb map
  for (auto pair : sBB)
    if (bbs.count(pair.first) > 0) 
      bbs[pair.first] = pair.second.get(GRB_DoubleAttr_X) > 0 ? true : false;

  int execN = static_cast<int>(sArc["valExecN"].get(GRB_DoubleAttr_X));

  // load answer to the arch map
  for (auto pair : sArc) {
    if (pair.first == "valExecN")
      continue;
    auto arch = findArchWithVarName(pair.first, archNames);
    archs[arch] = pair.second.get(GRB_DoubleAttr_X) > 0 ? 1 : 0;
    // update the connection information after CFDFC extraction
    if (archs[arch] > 0)
      arch->execFreq -= execN;
  }
  return execN;
}

