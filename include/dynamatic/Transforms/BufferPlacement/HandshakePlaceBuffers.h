//===- HandshakePlaceBuffers.h - Place buffers in DFG -----------*- C++ -*-===//
//
// This file declares the --place-buffers pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace buffer {

using namespace circt;
using namespace circt::handshake;

/// An arch stores the basic information (execution frequency, isBackEdge)
/// of an arch  between basic blocks.
struct arch {
  int srcBB, dstBB;
  unsigned freq;
  bool isBackEdge = false;
};

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
DataflowCircuit createCFDFCircuit(handshake::FuncOp funcOp,
                                  std::map<ArchBB *, bool> &archs,
                                  std::map<unsigned, bool> &bbs);
} // namespace buffer

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakePlaceBuffersPass(bool firstMG = false,
                                std::string stdLevelInfo = "");

} // namespace dynamatic
#endif // DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
