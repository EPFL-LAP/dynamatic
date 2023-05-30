//===- UtilsForMILPSolver.cpp - Gurobi solver for MILP  ---------*- C++ -*-===//
//
// This file implements the MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/UtilsForMILPSolver.h"
#include <algorithm>
#include <gurobi_c++.h>
#include <map>

using namespace dynamatic;
using namespace dynamatic::buffer;

// GRBVar MILP::newVar(const std::string &name, VarType type,
//            double lower_bound, double upper_bound) {
//   // legacy-dynamatic:
//   //
//   https://github.com/lana555/dynamatic/blob/master/Buffers/src/MILP_Model.h#LL563C1-L563C1
//   // Vars.push_back(var {name, type, lower_bound, upper_bound, 0});
//   GRBVar var;
//   if (type == REAL) {
//     var = modelMILP.addVar(lower_bound, upper_bound, 0,  GRB_CONTINUOUS,
//     name);
//   } else if (type == INTEGER) {
//     var = modelMILP.addVar(lower_bound, upper_bound, 0, GRB_INTEGER, name);
//   } else if (type == BOOLEAN) {
//     var = modelMILP.addVar(lower_bound, upper_bound, 0, GRB_BINARY, name);
//   } else {
//     llvm_unreachable("Unknown variable type");
//   }
//   return var;
// }

channel *buffer::findChannelWithVarName(std::string varName,
                                        std::vector<basicBlock *> &bbList) {
  unsigned srcBB = std::stoi(varName.substr(2, varName.find("_e") - 2));
  unsigned channelInd = std::stoi(varName.substr(
      varName.find("_e") + 2, varName.find("_bb") - varName.find("_e") - 2));

  return bbList[srcBB]->outChannels[channelInd];
}

static bool buffer::toSameDstOp(std::string var1, std::string var2,
                                std::vector<basicBlock *> &bbList) {
  auto *channel1 = findChannelWithVarName(var1, bbList);
  auto *channel2 = findChannelWithVarName(var2, bbList);

  return (channel1->opDst).value() == (channel2->opDst).value();
}

static bool buffer::fromSameSrcOp(std::string var1, std::string var2,
                                  std::vector<basicBlock *> &bbList) {
  auto *channel1 = findChannelWithVarName(var1, bbList);
  auto *channel2 = findChannelWithVarName(var2, bbList);

  return (channel1->opSrc).value() == (channel2->opSrc).value();
}

std::vector<std::string>
buffer::findSameDstOpStrings(const std::string &inputString,
                             const std::vector<std::string> &stringList,
                             std::vector<basicBlock *> &bbList) {
  std::vector<std::string> resultStrings;

  for (const std::string &str : stringList) {
    if (toSameDstOp(inputString, str, bbList)) {
      resultStrings.push_back(str);
    }
  }
  return resultStrings;
}

std::vector<std::string>
buffer::findSameSrcOpStrings(const std::string &inputString,
                             const std::vector<std::string> &stringList,
                             std::vector<basicBlock *> &bbList) {
  std::vector<std::string> resultStrings;

  for (const std::string &str : stringList) {
    if (fromSameSrcOp(inputString, str, bbList)) {
      resultStrings.push_back(str);
    }
  }
  return resultStrings;
}

unsigned getSrcBBIndFromVarName(std::string varName) {
  return std::stoi(varName.substr(2, varName.find("_e") - 2));
}

unsigned getDstBBIndFromVarName(std::string varName) {
  size_t lastUnderscorePos = varName.rfind('bb');
  return std::stoi(varName.substr(lastUnderscorePos + 1));
}

std::vector<std::vector<std::string>>
getArchsFromSameBB(const std::map<std::string, GRBVar> &sArc) {
  std::vector<std::vector<std::string>> result;

  // Create a map to group strings by bb
  std::map<std::string, std::vector<std::string>> bbMap;

  // Iterate over the map sArc
  for (const auto &pair : sArc) {
    const std::string &str = pair.first;
    const std::string bb = str.substr(0, str.find('_')); // Extract the bb name

    // Check if the bb exists in the map
    if (bbMap.find(bb) == bbMap.end()) {
      // If the bb doesn't exist, create a new vector and insert it into the map
      std::vector<std::string> strings;
      strings.push_back(str);
      bbMap.insert({bb, strings});
    } else {
      // If the bb already exists, add the string to the existing vector
      bbMap[bb].push_back(str);
    }
  }

  // Convert the map values to a vector of vectors
  for (const auto &pair : bbMap) {
    result.push_back(pair.second);
  }

  return result;
}

std::vector<std::vector<std::string>>
getArchsToSameBB(const std::map<std::string, GRBVar> &sArc) {
  std::vector<std::vector<std::string>> result;

  // Create a map to group strings by bb
  std::map<std::string, std::vector<std::string>> bbMap;

  // Iterate over the map sArc
  for (const auto &pair : sArc) {
    const std::string &str = pair.first;
    const std::string bb = str.substr(str.find('_') + 1); // Extract the bb name

    // Check if the bb exists in the map
    if (bbMap.find(bb) == bbMap.end()) {
      // If the bb doesn't exist, create a new vector and insert it into the map
      std::vector<std::string> strings;
      strings.push_back(str);
      bbMap.insert({bb, strings});
    } else {
      // If the bb already exists, add the string to the existing vector
      bbMap[bb].push_back(str);
    }
  }

  // Convert the map values to a vector of vectors
  for (const auto &pair : bbMap) {
    result.push_back(pair.second);
  }

  return result;
}

arch *findArchWithVarName(std::string varName,
                          std::vector<basicBlock *> &bbList) {
  unsigned srcBB = getSrcBBIndFromVarName(varName);
  unsigned dstBB = getDstBBIndFromVarName(varName);

  basicBlock *bbSrc = findExistsBB(srcBB, bbList);
  basicBlock *bbDst = findExistsBB(dstBB, bbList);

  return findExistsArch(bbSrc, bbDst, bbSrc->outArchs);
}

std::vector<std::string> getBackArchs(const std::map<std::string, GRBVar> &sArc,
                                      std::vector<basicBlock *> &bbList) {
  std::vector<std::string> result;

  // Create a map to group strings by bb
  std::map<std::string, std::vector<std::string>> bbMap;

  // Iterate over the map sArc
  for (const auto &pair : sArc) {
    const std::string &varName = pair.first;
    unsigned srcBB = getSrcBBIndFromVarName(varName);
    unsigned dstBB = getDstBBIndFromVarName(varName);

    basicBlock *bbSrc = findExistsBB(srcBB, bbList);

    for (auto arch : bbSrc->outArchs) {
      // llvm::errs() << "From: " << bbSrc->index << " To: "
      // <<arch->bbDst->index<< "\n"; llvm::errs() << "srcBB: " << srcBB <<
      // "---dstBB: " <<dstBB<< "\n\n";
      if (arch->bbDst->index == dstBB && arch->isBackEdge) {
        // llvm::errs() << "back arch: " << varName << "\n";
        result.push_back(pair.first);
        break;
      }
    }
  }

  return result;
}

std::vector<std::string>
getAllInArcNames(std::string varBBName,
                 const std::map<std::string, GRBVar> &sArc) {
  std::vector<std::string> result;

  unsigned bbDstInd = stoi(varBBName.substr(2)); // named after bbi
  // Iterate over the map sArc
  for (const auto &pair : sArc) {
    const std::string &str = pair.first;
    unsigned bbInd = getDstBBIndFromVarName(str);

    if (bbInd == bbDstInd)
      result.push_back(pair.first);
  }
  return result;
}

std::vector<std::string>
getAllOutArcNames(std::string varBBName,
                  const std::map<std::string, GRBVar> &sArc) {
  std::vector<std::string> result;

  unsigned bbDstInd = stoi(varBBName.substr(2)); // named after bbi
  // Iterate over the map sArc
  for (const auto &pair : sArc) {
    const std::string &str = pair.first;
    unsigned bbInd = getSrcBBIndFromVarName(str);

    if (bbInd == bbDstInd)
      result.push_back(pair.first);
  }
  return result;
}

buffer::dataFlowCircuit *
buffer::extractMarkedGraphBB(handshake::FuncOp funcOp, MLIRContext *ctx,
                             std::vector<basicBlock *> &bbList) {

  // Create a gurobi model
  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "mip1.log");
  env.start();
  GRBModel modelMILP = GRBModel(env);
  // Define variables
  std::map<std::string, GRBVar> sBB;
  std::map<std::string, GRBVar> sArc;

  unsigned cstMaxN = 0;

  for (auto bb : bbList) {
    // define basic block selection variables
    std::string valBB = "bb" + std::to_string(bb->index);
    sBB[valBB] = modelMILP.addVar(0, 1, 0.0, GRB_BINARY, valBB);

    // define edge selection variables
    for (int i = 0; i < bb->outArchs.size(); ++i) {
      auto arc = bb->outArchs[i];
      std::string valArcSel = "bb" + std::to_string(arc->bbSrc->index) + "_e" +
                              std::to_string(i) + "_bb" +
                              std::to_string(arc->bbDst->index);

      sArc[valArcSel] = modelMILP.addVar(0, 1, 0.0, GRB_BINARY, valArcSel);

      cstMaxN = std::max(cstMaxN, arc->freq);
    }
  }
  // define maximum execution cycles variables
  GRBVar valExecN = modelMILP.addVar(0, cstMaxN, 0.0, GRB_INTEGER, "valExecN");

  // Set objective
  GRBQuadExpr objExpr;
  for (auto pair : sArc) {
    auto S_e = pair.second;
    objExpr += valExecN * S_e;
  }
  modelMILP.setObjective(objExpr, GRB_MAXIMIZE);

  // Define constraints
  int constrInd = 0;

  // for each edge e: N <= S_e x N_e + (1-S_e) x cstMaxN
  for (auto pair : sArc) {
    auto arcEntity = findArchWithVarName(pair.first, bbList);
    unsigned N_e = arcEntity->freq;
    auto S_e = pair.second;
    modelMILP.addConstr(valExecN <= S_e * N_e + (1 - S_e) * cstMaxN,
                        "cN" + std::to_string(constrInd));
    ++constrInd;
  }

  // Only select one back archs:
  // for each bb \in Back(CFG): sum(S_e) = 1
  auto groupBackArchs = getBackArchs(sArc, bbList);
  GRBLinExpr constraintExpr;
  for (auto arch : groupBackArchs)
    if (sArc.count(arch) > 0)
      constraintExpr += sArc[arch];

  modelMILP.addConstr(constraintExpr == 1, "cBack");

  // If a bb is selected,
  // exactly one of its input and output archs is selected
  constrInd = 0;
  for (auto pair : sBB) {
    // input archs
    GRBLinExpr constraintInExpr;
    auto inArcs = getAllInArcNames(pair.first, sArc);
    for (auto arch : inArcs)
      if (sArc.count(arch) > 0)
        constraintInExpr += sArc[arch];

    modelMILP.addConstr(constraintInExpr == pair.second,
                        "cIn" + std::to_string(constrInd));

    // output archs
    GRBLinExpr constraintOutExpr;
    auto outArcs = getAllOutArcNames(pair.first, sArc);
    for (auto arch : outArcs)
      if (sArc.count(arch) > 0)
        constraintOutExpr += sArc[arch];

    modelMILP.addConstr(constraintOutExpr == pair.second,
                        "cOut" + std::to_string(constrInd));
    constrInd++;
  }

  modelMILP.optimize();

  if (valExecN.get(GRB_DoubleAttr_X) <= 0)
    return nullptr;
  else
    modelMILP.write("/home/yuxuan/Projects/dynamatic-utils/compile/debug.lp");


  // create a data flow circuit from the solution
  dataFlowCircuit *dfCircuit = new dataFlowCircuit();
  dfCircuit->execN = static_cast<int>(valExecN.get(GRB_DoubleAttr_X));

  for (auto pair : sBB) {
    auto x = sBB[pair.first];
    if (x.get(GRB_DoubleAttr_X) == 1) {
      basicBlock *bb = findExistsBB(std::stoi(pair.first.substr(2)), bbList);
      dfCircuit->insertSelBB(funcOp, bb);
    }
  }

  // create channels between BBs, and subtract the frequency of the channel
  for (auto pair : sArc) {
    auto x = sArc[pair.first];
    if (x.get(GRB_DoubleAttr_X) == 1) {
      arch *arc = findArchWithVarName(pair.first, bbList);
      dfCircuit->insertSelArc(arc);
      if (arc->isBackEdge)
        arc->freq -= valExecN.get(GRB_DoubleAttr_X);
      else
        arc->freq = 0;
    // llvm::errs() << pair.first << " " << arc->freq << "\n";
    }
    // llvm::errs() << x.get(GRB_StringAttr_VarName) << " "
    //      << x.get(GRB_DoubleAttr_X) << "\n";
  }

  // dfCircuit->printCircuits();

  return dfCircuit;
}