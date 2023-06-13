//===- UtilsForExtractMG.cpp - Extract MG for optimization* C++ ---------*-===//
//
// This file implements function supports for CFDFCircuit extraction.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/UtilsForExtractMG.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
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

void buffer::readSimulateFile(const std::string & fileName, 
                              std::map<archBB*, int> &archs, 
                              std::map<int, int> &bbs) {
  std::ifstream inFile(fileName);

  if (!inFile)
    assert(false && "Cannot open the file\n");

  std::string line;

  // Skip the header line
  std::getline(inFile, line);

  while (std::getline(inFile, line)) {
    std::istringstream iss(line);
    archBB *arch = new archBB();

    std::string token;
    std::getline(iss, token, ',');
    arch->srcBB = std::stoi(token);

    std::getline(iss, token, ',');
    arch->dstBB = std::stoi(token);

    std::getline(iss, token, ',');
    arch->execFreq = std::stoi(token);

    if (!std::getline(iss, token, ',')) 
      arch->isBackEdge = arch->srcBB >= arch->dstBB ? true : false;
    else 
      arch->isBackEdge = std::stoi(token) == 1 ? true : false;
    
    archs[arch] = false;
    if (bbs.count(arch->srcBB) == 0)
      bbs[arch->srcBB] = false;
    if (bbs.count(arch->dstBB) == 0)
      bbs[arch->dstBB] = false;
  }
}

static void printCFDFCircuit(std::map<archBB*, int> &archs, 
                              std::map<int, int> &bbs) {
  llvm::errs() << "CFDFCircuit:\n";
  for (auto bb : bbs) 
    if (bb.second > 0) {
      llvm::errs() << "BB: " << bb.first << "\n"; 
      for (auto arch : archs) 
        if (arch.first->srcBB == bb.first && arch.second > 0) 
          llvm::errs() << "  " << arch.first->srcBB << " -> " << arch.first->dstBB 
                        << " : " << "(freq " << arch.first->execFreq << " ; back edge  " 
                        << arch.first->isBackEdge << ")\n";
    }
}

static void printCircuit(std::map<archBB*, int> &archs, 
                              std::map<int, int> &bbs) {
  llvm::errs() << "==========Circuit:============\n";
  for (auto bb : bbs) {
      llvm::errs() << "BB: " << bb.first << "\n"; 
      for (auto arch : archs) 
        if (arch.first->srcBB == bb.first) 
          llvm::errs() << "  " << arch.first->srcBB << " -> " << arch.first->dstBB 
                        << " : " << "(freq " << arch.first->execFreq << " ; back edge  " 
                        << arch.first->isBackEdge << ")\n";
    }
}

static int initVarInMILP(GRBModel &modelMILP, 
                          std::map<int, GRBVar> &sBB, 
                          std::map<std::string, GRBVar> &sArc, 
                          std::vector<archBB*> &archNames, 
                          std::vector<int> &bbNames) {
  int cstMaxN = 0;
  
  for (auto bb : bbNames) {
    sBB[bb] = modelMILP.addVar(0.0, 1, 0.0, GRB_BINARY, "sBB_" + std::to_string(bb));
    for (auto arch : archNames)
      if (arch->srcBB == bb) {
        std::string arcName = "sArc_" + std::to_string(arch->srcBB) + 
                              "_" + std::to_string(arch->dstBB);
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
  for (auto pair : sArc) {
    if (pair.first == "valExecN")
      continue;
    auto S_e = pair.second;
    objExpr += sArc["valExecN"] * S_e;
  }
  modelMILP.setObjective(objExpr, GRB_MAXIMIZE);
}

static archBB* findArchWithVarName(const std::string &varName, 
                                  std::vector<archBB*> &archs) {
  for (auto arch : archs) {
    std::string arcName = "sArc_" + std::to_string(arch->srcBB) + 
                          "_" + std::to_string(arch->dstBB);
    if (arcName == varName)
      return arch;
  }
  assert(false && "Cannot find arch with var name");
}

static std::vector<std::string> getBBsInArcVars(int bb, 
                                                std::map<std::string, GRBVar> &sArc,
                                                std::vector<archBB*> &archs) {
  std::vector<std::string> varNames;
  for (auto pair : sArc) {
    if (pair.first == "valExecN")
      continue;
    auto arch = findArchWithVarName(pair.first, archs);
    if (arch->srcBB == bb) {
      varNames.push_back(pair.first);
    }
  }
  return varNames;
}

static std::vector<std::string> getBBsOutArcVars(int bb, 
                                                std::map<std::string, GRBVar> &sArc,
                                                std::vector<archBB*> &archs) {
  std::vector<std::string> varNames;
  for (auto pair : sArc) {
    if (pair.first == "valExecN")
      continue;
    auto arch = findArchWithVarName(pair.first, archs);
    if (arch->dstBB == bb) {
      varNames.push_back(pair.first);
    }
  }
  return varNames;
}

static void setEdgeConstrs(GRBModel &modelMILP, 
                          int cstMaxN,
                          std::map<std::string, GRBVar> &sArc,
                          std::vector<archBB*> &archs) {

  auto valExecN = sArc["valExecN"];
  GRBLinExpr backEdgeConstr;

  int constrInd = 0;
  for (auto pair : sArc) {
    if (pair.first == "valExecN")
      continue;
    auto arcEntity = findArchWithVarName(pair.first, archs);
    unsigned N_e = arcEntity->execFreq;
    auto S_e = pair.second;
    // for each edge e: N <= S_e x N_e + (1-S_e) x cstMaxN
    modelMILP.addConstr(valExecN <= S_e * N_e + (1 - S_e) * cstMaxN,
                        "cN" + std::to_string(constrInd));
    ++constrInd;
    // Only select one back archs: for each bb \in Back(CFG): sum(S_e) = 1
    if (arcEntity->isBackEdge) {
      backEdgeConstr += S_e;
    }
  }
  modelMILP.addConstr(backEdgeConstr == 1, "cBack");
}

static void setBBConstrs(GRBModel &modelMILP, 
                         std::map<int, GRBVar> &sBB,
                         std::map<std::string, GRBVar> &sArc,
                         std::vector<archBB*> &archs) {

  int constrInd = 0;
  for (auto pair : sBB) {
    // input archs
    GRBLinExpr constraintInExpr;
    auto inArcs = getBBsInArcVars(pair.first, sArc, archs);
    for (auto arch : inArcs)
      if (sArc.count(arch) > 0)
        constraintInExpr += sArc[arch];
    modelMILP.addConstr(constraintInExpr == pair.second,
                        "cIn" + std::to_string(constrInd));

    // output archs
    GRBLinExpr constraintOutExpr;
    auto outArcs = getBBsOutArcVars(pair.first, sArc, archs);
    for (auto arch : outArcs)
      if (sArc.count(arch) > 0)
        constraintOutExpr += sArc[arch];
    modelMILP.addConstr(constraintOutExpr == pair.second,
                        "cOut" + std::to_string(constrInd));
    ++constrInd;
  }

};

static bool isSelect(std::map<archBB*, int> &archs, channel *ch) {
  int srcBB = getBBIndex(ch->opSrc);
  int dstBB = getBBIndex(ch->opDst);
  for (auto pair : archs) {
    if (pair.first->srcBB == srcBB && pair.first->dstBB == dstBB)
      return true;
  }
  return false;
}

int buffer::extractCFDFCircuit(std::map<archBB*, int> &archs,
                               std::map<int, int> &bbs) {
  // store variable names
  std::vector<archBB*> archNames;
  std::vector<int> bbNames;

  for (auto pair : archs) {
    archNames.push_back(pair.first);
  }
  for (auto pair : bbs) {
    bbNames.push_back(pair.first);
  }
  // printCircuit(archs, bbs);
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
  setEdgeConstrs(modelMILP, cstMaxN, sArc, archNames);
  setBBConstrs(modelMILP, sBB, sArc, archNames);
  modelMILP.optimize();

  if (modelMILP.get(GRB_IntAttr_Status) != GRB_OPTIMAL ||
      sArc["valExecN"].get(GRB_DoubleAttr_X) <= 0)
    return -1;

  // load answer to the bb map
  for (auto pair : sBB) 
    if (bbs.count(pair.first) > 0)
      bbs[pair.first] = pair.second.get(GRB_DoubleAttr_X) > 0 ? 1 : 0;
  
  int execN = static_cast<int>(sArc["valExecN"].get(GRB_DoubleAttr_X));
  // load answer to the arch map
  for (auto pair : sArc) {
    if (pair.first == "valExecN") 
      continue;
    
    auto arch = findArchWithVarName(pair.first, archNames);
    archs[arch] = pair.second.get(GRB_DoubleAttr_X) > 0 ? 1 : 0;
  }

  // llvm::errs() << "exec times: " << sArc["valExecN"].get(GRB_DoubleAttr_X) << "\n";
  // printCFDFCircuit(archs, bbs);
  // update the connection information after CFDFC extraction    
  for (auto pair: sArc) {
    if (pair.first == "valExecN")
      continue;
    auto arch = findArchWithVarName(pair.first, archNames);
    if (archs[arch] > 0) 
      arch->execFreq -= execN;
  }

  return execN;
}

dataFlowCircuit *buffer::createCFDFCircuit(std::vector<unit *> &unitList,
                       std::map<archBB*, int> &archs,
                       std::map<int, int> &bbs) {

  dataFlowCircuit *circuit = new dataFlowCircuit();
  for (auto unit : unitList) {
    int bbIndex = getBBIndex(unit->op);
    
    // insert units in the selected basic blocks
    if (bbs.count(bbIndex)>0 && bbs[bbIndex] > 0) {
      circuit->units.push_back(unit);
      // insert channels if it is selected
      for (auto ports : unit->outPorts) 
        for (auto ch : ports->cntChannels) 
          if (isSelect(archs, ch)) 
            circuit->channels.push_back(ch);    
    }
  }
  // circuit->printCircuits();
  return circuit;
}
