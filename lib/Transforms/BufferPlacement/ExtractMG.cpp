//===- UtilsForExtractMG.cpp - Extract MG for optimization* C++ ---------*-===//
//
// This file implements function supports for CFDFCircuit extraction.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"
#include <fstream>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

/// Determine whether a string is a valid number from the simulation file
static bool isNumber(const std::string &str) {
  return std::all_of(str.begin(), str.end(),
                     [](char c) { return isdigit(c) || c == ' '; });
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

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED

/// Initialize the variables in the extract CFDFC MILP model
static int initVarInMILP(GRBModel &modelMILP, std::map<ArchBB *, GRBVar> &sArc,
                         std::map<unsigned, GRBVar> &sBB,
                         std::map<ArchBB *, bool> &archs,
                         std::map<unsigned, bool> &bbs) {
  unsigned cstMaxN = 0;

  for (auto &[bbInd, _] : bbs)
    // define variables for basic blocks selection
    sBB[bbInd] = modelMILP.addVar(0.0, 1, 0.0, GRB_BINARY,
                                  "sBB_" + std::to_string(bbInd));

  // define variables for edges selection
  for (auto &[arch, _] : archs) {
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
  for (auto &[_, var] : sArc)
    objExpr += valExecN * var;

  modelMILP.setObjective(objExpr, GRB_MAXIMIZE);
}

/// get all the input archs to a basic block
static std::vector<ArchBB *> getBBsInArcVars(unsigned bb,
                                             std::map<ArchBB *, GRBVar> &sArc) {
  std::vector<ArchBB *> varNames;
  for (auto &[arch, _] : sArc)
    if (arch->dstBB == bb)
      varNames.push_back(arch);
  return varNames;
}

/// get all the output archs from a basic block
static std::vector<ArchBB *>
getBBsOutArcVars(unsigned bb, std::map<ArchBB *, GRBVar> &sArc) {
  std::vector<ArchBB *> varNames;
  for (auto &[arch, _] : sArc)
    if (arch->srcBB == bb)
      varNames.push_back(arch);
  return varNames;
}

static void setEdgeConstrs(GRBModel &modelMILP, int cstMaxN,
                           std::map<ArchBB *, GRBVar> &sArc, GRBVar &valExecN) {

  GRBLinExpr backEdgeConstr;

  for (auto [constrInd, pair] : llvm::enumerate(sArc)) {
    auto &[arcEntity, varSE] = pair; // pair = [ArchBB*, GRBVar]
    unsigned cstNE = arcEntity->execFreq;
    // for each edge e: N <= S_e x N_e + (1-S_e) x cstMaxN
    modelMILP.addConstr(valExecN <= varSE * cstNE + (1 - varSE) * cstMaxN,
                        "cN" + std::to_string(constrInd));
    // Only select one back archs: for each bb \in Back(CFG): sum(S_e) = 1
    if (arcEntity->isBackEdge)
      backEdgeConstr += varSE;
  }
  modelMILP.addConstr(backEdgeConstr == 1, "cBack");
}

static void setBBConstrs(GRBModel &modelMILP, std::map<unsigned, GRBVar> &sBB,
                         std::map<ArchBB *, GRBVar> &sArc) {

  for (auto &[bbInd, varBB] : sBB) {
    // only 1 input arch if bb is selected;
    // no input arch if bb is not selected
    GRBLinExpr constraintInExpr;
    auto inArcs = getBBsInArcVars(bbInd, sArc);
    for (ArchBB *arch : inArcs)
      constraintInExpr += sArc[arch];
    modelMILP.addConstr(constraintInExpr == varBB,
                        "cIn" + std::to_string(bbInd));
    // only 1 output arch if bb is selected;
    // no output arch if bb is not selected
    GRBLinExpr constraintOutExpr;
    auto outArcs = getBBsOutArcVars(bbInd, sArc);
    for (ArchBB *arch : outArcs)
      constraintOutExpr += sArc[arch];
    modelMILP.addConstr(constraintOutExpr == varBB,
                        "cOut" + std::to_string(bbInd));
  }
};

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

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

    archs[arch] = false;
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

unsigned buffer::getChannelFreq(Value channel, std::vector<CFDFC> &cfdfcList) {
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
      freq += cfdfc.execN;

  return freq;
}

bool buffer::isBackEdge(Operation *opSrc, Operation *opDst) {
  if (opDst->isProperAncestor(opSrc))
    return true;
  if (isa<BranchOp, ConditionalBranchOp>(opSrc) &&
      isa<MuxOp, MergeOp, ControlMergeOp>(opDst))
    return getBBIndex(opSrc) >= getBBIndex(opDst);

  return false;
}

bool buffer::isSelect(std::map<unsigned, bool> &bbs, Value val) {
  Operation *srcOp = val.getDefiningOp();
  auto firstUser = val.getUsers().begin();
  assert(firstUser != val.getUsers().end() &&
         "value has no uses, run fork/sink materialization before extracting "
         "CFDFCs");
  Operation *dstOp = *firstUser;

  unsigned srcBB = getBBIndex(srcOp);
  unsigned dstBB = getBBIndex(dstOp);

  // if srcOp and dstOp are in the same BB, and the edge is not backedge
  // then the edge is selected depends on the BB
  if (bbs.count(dstBB) > 0 && srcBB == dstBB)
    if (!isBackEdge(srcOp, dstOp))
      return bbs[srcBB];
  return false;
}

bool buffer::isSelect(std::map<ArchBB *, bool> &archs, Value val) {
  Operation *srcOp = val.getDefiningOp();
  auto firstUser = val.getUsers().begin();
  assert(firstUser != val.getUsers().end() &&
         "value has no uses, run fork/sink materialization before extracting "
         "CFDFCs");
  Operation *dstOp = *firstUser;

  unsigned srcBB = getBBIndex(srcOp);
  unsigned dstBB = getBBIndex(dstOp);

  for (auto &[arch, varSelArch] : archs)
    if (arch->srcBB == srcBB && arch->dstBB == dstBB)
      return varSelArch;

  return false;
}

LogicalResult buffer::extractCFDFCircuit(std::map<ArchBB *, bool> &archs,
                                         std::map<unsigned, bool> &bbs,
                                         unsigned &freq) {

#ifdef DYNAMATIC_GUROBI_NOT_INSTALLED
  llvm::errs() << "Project was built without Gurobi installed, can't run "
                  "CFDFC extraction\n";
  return failure();
#else

  // Create MILP model for CFDFCircuit extraction
  // Init a gurobi model
  GRBEnv env = GRBEnv(true);
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
      valExecN.get(GRB_DoubleAttr_X) <= 0) {
    freq = 0;
    return success();
  }

  // load answer to the bb map
  for (auto &[bbInd, varBB] : sBB)
    if (bbs.count(bbInd) > 0 && varBB.get(GRB_DoubleAttr_X) > 0)
      bbs[bbInd] = true;
    else
      bbs[bbInd] = false;

  freq = static_cast<unsigned>(valExecN.get(GRB_DoubleAttr_X));

  // load answer to the arch map
  for (auto &[arch, varArc] : sArc) {
    if (archs.count(arch) > 0 && varArc.get(GRB_DoubleAttr_X) > 0) {
      archs[arch] = true;
      // update the connection information after CFDFC extraction
      arch->execFreq -= freq;
    } else
      archs[arch] = false;
  }

  return success();

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
}
