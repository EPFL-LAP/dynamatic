//===- UtilsForMILPSolver.cpp - Gurobi solver for MILP  ---------*- C++ -*-===//
//
// This file implements the MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/UtilsForMILPSolver.h"
#include <gurobi_c++.h>
#include <algorithm>
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

arch *buffer::findArcWithVarName(std::string varName,
                         std::vector<basicBlock *> &bbList) {
  unsigned srcBB = std::stoi(varName.substr(2, varName.find("_e") - 2));
  unsigned arcInd = std::stoi(varName.substr(
      varName.find("_e") + 2, varName.find("_bb") - varName.find("_e") - 2));

  return bbList[srcBB]->outArcs[arcInd];
}


static bool buffer::toSameDstOp(std::string var1, std::string var2,
                        std::vector<basicBlock *> &bbList) {
  arch *arc1 = findArcWithVarName(var1, bbList);
  arch *arc2 = findArcWithVarName(var2, bbList);

  return (arc1->opDst).value() == (arc2->opDst).value();
}

static bool buffer::fromSameSrcOp(std::string var1, std::string var2,
                          std::vector<basicBlock *> &bbList) {
  arch *arc1 = findArcWithVarName(var1, bbList);
  arch *arc2 = findArcWithVarName(var2, bbList);

  return (arc1->opSrc).value() == (arc2->opSrc).value();
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

void buffer::extractMarkedGraphBB(std::vector<basicBlock *> &bbList) {

  // try{
    // GRBEnv env = GRBEnv(true);
    // env.set("LogFile", "mip1.log");
    // env.start();
    // GRBModel modelMILP = GRBModel(env);
    // Define variables
    std::map<std::string, GRBVar> sBB;
    std::map<std::string, GRBVar> sArc;
    std::map<std::string, GRBVar> nArc;
    // std::vector<std::string> ArcVarNames;

    unsigned cstMaxN = 0;

    // for (auto bb : bbList) {
    //   std::string valBB = "bb" + std::to_string(bb->index);
    //   sBB[valBB] = modelMILP.addVar(0, 1, 0.0, GRB_BINARY, valBB);

    //   // define out archs
    //   for (int i = 0; i < bb->outArcs.size(); ++i) {
    //     auto arc = bb->outArcs[i];
    //     std::string valArcSel = "bb" + std::to_string(arc->bbSrc->index) + "_e" +
    //                             std::to_string(i) + "_bb" +
    //                             std::to_string(arc->bbDst->index);
    //     ArcVarNames.push_back(valArcSel);
    //     sArc[valArcSel] = modelMILP.addVar(0, 1, 0.0, GRB_BINARY, valArcSel);

    //     std::string valArcN = "bb" + std::to_string(arc->bbSrc->index) + "_e" +
    //                           std::to_string(i) + "_bb" +
    //                           std::to_string(arc->bbDst->index);
    //     nArc[valArcN] = modelMILP.addVar(0, arc->freq, 0.0, GRB_CONTINUOUS, valArcN);
    //     // ArcVarNames.push_back(valArcN);

    //     cstMaxN = std::max(cstMaxN, arc->freq);
    //   }
    // }
    // GRBVar valExecN = modelMILP.addVar(0, cstMaxN, 0.0, GRB_INTEGER, "valExecN");

    // // Define constraints
    // int constrInd = 0;
    // // All in archs to the same dst op equals to 1
    // for (auto valInArc : ArcVarNames) {
    //   std::vector<std::string> sameDstArcs =
    //       findSameDstOpStrings(valInArc, ArcVarNames, bbList);
      
    //   GRBLinExpr constraintExpr;
    //   for (const std::string& result : sameDstArcs) {
    //       if (sArc.count(result) > 0) {
    //           constraintExpr += sArc[result];
    //       }
    //   }
    //   modelMILP.addConstr(constraintExpr == 1, "cin"+std::to_string(constrInd));
    //   ++constrInd;
    // }
  // } catch(GRBException e) {
  //   std::cout << "Error code = " << e.getErrorCode() << "\n";
  //   std::cout << e.getMessage() << "\n";
  // } catch(...) {
  //   std::cout << "Exception during optimization" <<"\n";
  // }
  


  // constrInd = 0;
  // // All out archs from the same src op equals to 1
  // for (auto valInArc : ArcVarNames) {
  //   std::vector<std::string> sameSrcArcs =
  //       findSameSrcOpStrings(valInArc, ArcVarNames, bbList);
    
  //   GRBLinExpr constraintExpr;
  //   for (const std::string& result : sameSrcArcs) {
  //       if (sArc.count(result) > 0) {
  //           constraintExpr += sArc[result];
  //       }
  //   }
  //   modelMILP.addConstr(constraintExpr == 1, "cout"+std::to_string(constrInd));
  //   ++constrInd;
  // }

  // modelMILP.write("/home/yuxuan/Projects/dynamatic-utils/compile/debug.lp");
  
  // int numConstraints = modelMILP.get(GRB_IntAttr_NumConstrs);
  // GRBConstr* constraints = modelMILP.getConstrs();

  // for (int i = 0; i < numConstraints; i++) {
  //     GRBConstr constr = constraints[i];
  //     std::string constrName = constr.get(GRB_StringAttr_ConstrName);

  //     // GRBConstr a = modelMILP.getConstrByName(constrName);
  //     // constr.getExpr();

  //     std::cout << "Constraint " << i << ": " << constrName << std::endl;
  //     // std::cout << "   Expression: " << lhsExpr << " " << sense << " " << rhsValue << std::endl;
  // }

}