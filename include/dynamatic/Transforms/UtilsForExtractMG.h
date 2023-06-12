//===- UtilsForExtractMG.h - utils for extracting marked graph *- C++ ---*-===//
//
// This file declaresfunction supports for CFDFCircuit extraction.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_UTILSFOREXTRACTMG_H
#define DYNAMATIC_TRANSFORMS_UTILSFOREXTRACTMG_H

#include "dynamatic/Transforms/UtilsForPlaceBuffers.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
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

int extractCFDFCircuit(std::map<archBB*, int> &archs,
                        std::map<int, int> &bbs);

dataFlowCircuit *createCFDFCircuit(std::vector<unit *> &unitList,
                       std::map<archBB*, int> &archs,
                       std::map<int, int> &bbs);

void printBBWithArchs(int bb, std::map<archBB*, int> &archs);

void printCFDFCircuit(std::map<archBB*, int> &archs, 
                      std::map<int, int> &bbs);

void readSimulateFile(const std::string & fileName, 
                      std::map<archBB* , int> &archs, 
                      std::map<int, int> &bbs);
} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_UTILSFOREXTRACTMG_H
