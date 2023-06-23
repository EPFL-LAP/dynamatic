//===- HandshakePlaceBuffers.h - Place buffers in DFG -----------*- C++ -*-===//
//
// This file declares the --place-buffers pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_PLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_PLACEBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"
#include "dynamatic/Transforms/BufferPlacement/OptimizeMILP.h"

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

} // namespace buffer

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakePlaceBuffersPass(bool firstMG = false,
                                std::string stdLevelInfo = "");

} // namespace dynamatic
#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_PLACEBUFFERS_H
