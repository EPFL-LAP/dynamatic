//===- HandshakePlaceBuffers.h - Place buffers in DFG -----------*- C++ -*-===//
//
// This file declares the --place-buffers pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakePlaceBuffersPass();

struct arch;

struct basicBlock {
  unsigned index = UINT_MAX;
  unsigned freq = UINT_MAX;
  bool selBB = false;
  bool isEntryBB = false;
  bool isExitBB = false;
  std::vector<arch *> inArcs;
  std::vector<arch *> outArcs;
};

struct arch {
  unsigned freq;
  basicBlock *bbSrc, *bbDst;
  std::optional<Operation *> opSrc, opDst;
  bool selArc = false;
  bool isBackEdge = false;

  // If opDst and opSrc are not in the same basic blocks, and
  // if opDst's users are in the same basic blocks as opDst, it is an in-edge.
  bool isInEdge = false;

  // If opDst and opSrc are not in the same basic blocks, and
  // if opSrc's users are in the same basic blocks as opSrc, it is an out-edge.
  bool isOutEdge = false;

  arch() {}
  arch(const unsigned freq, basicBlock *bbSrc, basicBlock *bbDst)
      : freq(freq), bbSrc(bbSrc), bbDst(bbDst) {}
};

struct dataFlowGraphBB {
  std::vector<arch> archList;
  std::vector<basicBlock> bbList;
  unsigned cstMaxN;
  unsigned valExecN;
};

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
