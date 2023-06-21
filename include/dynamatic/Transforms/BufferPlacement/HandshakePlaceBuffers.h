//===- HandshakePlaceBuffers.h - Place buffers in DFG -----------*- C++ -*-===//
//
// This file declares the --place-buffers pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"

namespace dynamatic {
  
namespace buffer {

/// An arch stores the basic information (execution frequency, isBackEdge)
/// of an arch  between basic blocks.
struct arch {
  int srcBB, dstBB;
  unsigned freq;
  bool isBackEdge = false;
};


/// Deep first search the handshake file to get the units connection graph.
void dfsHandshakeGraph(Operation *opNode, 
                       std::vector<Operation *> &visited);

struct DataflowCircuit {
  double targetCP, maxCP;
  std::vector<Operation *> units;
  std::vector<Value *> channels;
  std::vector<int> selBBs;

  int execN = 0;

  void printCircuits();
};

/// Create the CFDFCircuit from the unitList(the DFS operations graph),
/// and archs, and bbs that store the CFDFC extraction results indicating
/// selected (1) or not (0).
DataflowCircuit createCFDFCircuit(std::vector<Operation *> &unitList,
                                   std::map<ArchBB *, bool> &archs,
                                   std::map<unsigned, bool> &bbs);
} // namespace buffer

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakePlaceBuffersPass(bool firstMG = false,
                                std::string stdLevelInfo = "");

} // namespace dynamatic
#endif // DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
