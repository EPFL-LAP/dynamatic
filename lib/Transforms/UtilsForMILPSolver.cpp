//===- UtilsForMILPSolver.cpp - Gurobi solver for MILP  ---------*- C++ -*-===//
//
// This file implements the MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/UtilsForMILPSolver.h"
#include "dynamatic/Transforms/HandshakePlaceBuffers.h"
#include <algorithm>
#include <map>

using namespace dynamatic;

int newVar(const std::string &name, VarType type,
           double lower_bound, double upper_bound, std::vector<Var> &Vars) {
  // if (Name2Var.count(n) > 0) {
  //     setError("Variable " + n + " multiply defined.");
  //     return -1;
  // }
  // Name2Var[n] = Vars.size();
  Vars.push_back(Var {name, type, lower_bound, upper_bound, 0});
  // if (type == REAL) numRealVars++;
  // else if (type == INTEGER) numIntegerVars++;
  // else numBooleanVars++;
  // return Vars.size() - 1;
}

void extractMarkedGraphBB(std::vector<basicBlock *> &bbList) {
  // create variables
  std::map<basicBlock *, int> sBB;
  std::map<arch *, int> sArc;
  std::map<arch *, int> nArc;
  std::vector<Var> Vars;

  unsigned cstMaxN = 0;
  
  for (auto bb : bbList) {
    sBB[bb] =
        newVar("sBB_" + std::to_string(bb->index), BOOLEAN, 0, 1);
    for (int i = 0; i < bb->outArcs.size(); ++i) {
      auto arc = bb->outArcs[i];
      sArc[arc] = newVar("sArc_" + std::to_string(arc->bbSrc->index) + "_e" +
                          std::to_string(i) + "_" +
                          std::to_string(arc->bbDst->index),
                         BOOLEAN, 0, 1, Vars);
      nArc[arc] = newVar("nArc_" + std::to_string(arc->bbSrc->index) + "_e" +
                          std::to_string(i) + "_" +
                          std::to_string(arc->bbDst->index),
                          REAL, 0, arc->freq, Vars);
      cstMaxN = std::max(cstMaxN, arc->freq);
    }
  }

  int valExecN = newVar("valExecN", INTEGER, 0, cstMaxN, Vars);
}