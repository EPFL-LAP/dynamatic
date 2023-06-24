//===- UtilsForExtractMG.cpp - Extract MG for optimization* C++ ---------*-===//
//
// This file implements function supports for CFDFCircuit extraction.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LogicBB.h"
#include "gurobi_c++.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

/// Determine whether a string is a valid number from the simulation file
static bool isNumber(const std::string &str) {
  for (char c : str)
    if (!isdigit(c) && c != ' ')
      return false;
  return true;
}

/// Parse the token from the simulation file and write the value if it is valid
static bool parseAndValidateToken(std::istringstream &iss, std::string &token,
                                  unsigned &value) {
  std::getline(iss, token, ',');
  if (!isNumber(token))
    return false;
  value = std::stoi(token);
  return true;
}

LogicalResult buffer::readSimulateFile(const std::string &fileName,
                                       std::map<ArchBB *, bool> &archs,
                                       std::map<unsigned, bool> &bbs) {
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
    if (!parseAndValidateToken(iss, token, arch->srcBB))
      return failure();

    if (!parseAndValidateToken(iss, token, arch->dstBB))
      return failure();

    if (!parseAndValidateToken(iss, token, arch->execFreq))
      return failure();

    unsigned backEdge;
    if (!parseAndValidateToken(iss, token, backEdge))
      return failure();
    arch->isBackEdge = (backEdge == 1);

    archs[pArch] = false;
    bbs[arch->srcBB] = false;
    bbs[arch->dstBB] = false;
  }
  return success();
}

int buffer::getBBIndex(Operation *op) {
  for (auto attr : op->getAttrs()) {
    if (attr.getName() == BB_ATTR)
      return dyn_cast<IntegerAttr>(attr.getValue()).getValue().getZExtValue();
  }
  return -1;
}

bool buffer::isBackEdge(Operation *opSrc, Operation *opDst) {
  if (opDst->isProperAncestor(opSrc))
    return true;
  return false;
}

/// Initialize the variables in the extract CFDFC MILP model
static int initVarInMILP(GRBModel &modelMILP, std::map<ArchBB *, GRBVar> &sArc,
                         std::map<unsigned, GRBVar> &sBB,
                         std::map<ArchBB *, bool> &archs,
                         std::map<unsigned, bool> &bbs) {
  unsigned cstMaxN = 0;

  for (auto pair : bbs) {
    // define variables for basic blocks selection
    unsigned bbInd = pair.first;
    sBB[bbInd] = modelMILP.addVar(0.0, 1, 0.0, GRB_BINARY,
                                  "sBB_" + std::to_string(bbInd));
  }

  // define variables for edges selection
  for (auto archPair : archs) {
    ArchBB *arch = archPair.first;
    std::string arcName = "sArc_" + std::to_string(arch->srcBB) + "_" +
                          std::to_string(arch->dstBB);
    sArc[arch] = modelMILP.addVar(0.0, 1, 0.0, GRB_BINARY, arcName);
    cstMaxN = std::max(cstMaxN, arch->execFreq);
  }

  return cstMaxN;
}

static void setObjective(GRBModel &modelMILP, std::map<ArchBB *, GRBVar> &sArc,
                         GRBVar &valExecN) {
  GRBQuadExpr objExpr;
  // cost function: max: \sum_{e \in E} s_e * execFreq_e
  for (auto pair : sArc) {
    auto sE = pair.second;
    objExpr += valExecN * sE;
  }
  modelMILP.setObjective(objExpr, GRB_MAXIMIZE);
}

/// get all the input archs to a basic block
static std::vector<ArchBB *> getBBsInArcVars(unsigned bb,
                                             std::map<ArchBB *, GRBVar> &sArc) {
  std::vector<ArchBB *> varNames;
  for (auto pair : sArc)
    if (pair.first->dstBB == bb)
      varNames.push_back(pair.first);
  return varNames;
}

/// get all the output archs from a basic block
static std::vector<ArchBB *>
getBBsOutArcVars(unsigned bb, std::map<ArchBB *, GRBVar> &sArc) {
  std::vector<ArchBB *> varNames;
  for (auto pair : sArc)
    if (pair.first->srcBB == bb)
      varNames.push_back(pair.first);
  return varNames;
}

static void setEdgeConstrs(GRBModel &modelMILP, int cstMaxN,
                           std::map<ArchBB *, GRBVar> &sArc, GRBVar &valExecN) {

  GRBLinExpr backEdgeConstr;

  for (auto [constrInd, pair] : llvm::enumerate(sArc)) {
    auto arcEntity = pair.first;
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
}

static void setBBConstrs(GRBModel &modelMILP, std::map<unsigned, GRBVar> &sBB,
                         std::map<ArchBB *, GRBVar> &sArc) {

  for (auto [constrInd, pair] : llvm::enumerate(sBB)) {
    // only 1 input arch if bb is selected;
    // no input arch if bb is not selected
    GRBLinExpr constraintInExpr;
    auto inArcs = getBBsInArcVars(pair.first, sArc);
    for (auto arch : inArcs)
      constraintInExpr += sArc[arch];
    modelMILP.addConstr(constraintInExpr == pair.second,
                        "cIn" + std::to_string(constrInd));
    // only 1 output arch if bb is selected;
    // no output arch if bb is not selected
    GRBLinExpr constraintOutExpr;
    auto outArcs = getBBsOutArcVars(pair.first, sArc);
    for (auto arch : outArcs)
      constraintOutExpr += sArc[arch];
    modelMILP.addConstr(constraintOutExpr == pair.second,
                        "cOut" + std::to_string(constrInd));
  }
};

bool buffer::isSelect(std::map<unsigned, bool> &bbs, Value *val) {
  Operation *srcOp = val->getDefiningOp();
  Operation *dstOp;
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

bool buffer::isSelect(std::map<ArchBB *, bool> &archs, Value *val) {
  Operation *srcOp = val->getDefiningOp();
  Operation *dstOp = val->getDefiningOp();

  unsigned srcBB = getBBIndex(srcOp);
  unsigned dstBB = getBBIndex(dstOp);

  for (auto pair : archs)
    if (pair.first->srcBB == srcBB && pair.first->dstBB == dstBB)
      return pair.second;
  return false;
}

unsigned buffer::extractCFDFCircuit(std::map<ArchBB *, bool> &archs,
                                    std::map<unsigned, bool> &bbs) {

  // Create MILP model for CFDFCircuit extraction
  // Init a gurobi model
  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "mip1.log");
  // cancel the printout output
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();
  GRBModel modelMILP = GRBModel(env);

  // Define variables
  std::map<ArchBB *, GRBVar> sArc;
  std::map<unsigned, GRBVar> sBB;

  int cstMaxN = initVarInMILP(modelMILP, sArc, sBB, archs, bbs);
  // define maximum execution cycles variables
  GRBVar valExecN = modelMILP.addVar(0, cstMaxN, 0.0, GRB_INTEGER, "valExecN");

  setObjective(modelMILP, sArc, valExecN);
  setEdgeConstrs(modelMILP, cstMaxN, sArc, valExecN);
  setBBConstrs(modelMILP, sBB, sArc);
  modelMILP.optimize();

  if (modelMILP.get(GRB_IntAttr_Status) != GRB_OPTIMAL ||
      valExecN.get(GRB_DoubleAttr_X) <= 0)
    return 0;

  // load answer to the bb map
  for (auto pair : sBB)
    if (bbs.count(pair.first) > 0)
      bbs[pair.first] = pair.second.get(GRB_DoubleAttr_X) > 0 ? true : false;

  int execN = static_cast<int>(valExecN.get(GRB_DoubleAttr_X));

  // load answer to the arch map
  for (auto pair : sArc) {
    auto arch = pair.first;
    archs[arch] = pair.second.get(GRB_DoubleAttr_X) > 0 ? true : false;
    // update the connection information after CFDFC extraction
    if (archs[arch] > 0)
      arch->execFreq -= execN;
  }
  return execN;
}
