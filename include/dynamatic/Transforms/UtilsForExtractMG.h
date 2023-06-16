//===- UtilsForExtractMG.h - utils for extracting marked graph *- C++ ---*-===//
//
// This file declaresfunction supports for CFDFCircuit extraction.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_UTILSFOREXTRACTMG_H
#define DYNAMATIC_TRANSFORMS_UTILSFOREXTRACTMG_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/UtilsForPlaceBuffers.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include <fstream>
#include <gurobi_c++.h>

namespace dynamatic {
namespace buffer {

struct archBB {
  int srcBB, dstBB;
  int execFreq;
  bool isBackEdge;
};

/// Define the MILP CFDFC extraction models, and write the optimization results
/// to the map.
int extractCFDFCircuit(std::map<archBB *, int> &archs, std::map<int, int> &bbs);

/// Create the CFDFCircuit from the unitList(the DFS operations graph),
/// and archs, and bbs that store the CFDFC extraction results indicating
/// selected (1) or not (0).
dataFlowCircuit *createCFDFCircuit(std::vector<unit *> &unitList,
                                   std::map<archBB *, int> &archs,
                                   std::map<int, int> &bbs);

/// Read the simulation file of standard level execution and store the results
/// in the map.
void readSimulateFile(const std::string &fileName,
                      std::map<archBB *, int> &archs, std::map<int, int> &bbs);
} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_UTILSFOREXTRACTMG_H