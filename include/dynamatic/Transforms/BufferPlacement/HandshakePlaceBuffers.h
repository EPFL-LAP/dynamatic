//===- HandshakePlaceBuffers.h - Place buffers in DFG -----------*- C++ -*-===//
//
// This file declares the --handshake-place-buffers pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace dynamatic {
namespace buffer {
/// Data structure for control-free dataflow circuits which
/// stores the units and channels inside it.
/// The CFDFC has properties for the buffer placement optimization, including
/// the target period, and maximum period.
struct CFDFC {
  double targetCP, maxCP;
  std::vector<Operation *> units;
  std::vector<Value> channels;

  unsigned execN = 0; // execution times of CFDFC
};

} // namespace buffer

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakePlaceBuffersPass(bool firstMG = false,
                                std::string stdLevelInfo = "");

} // namespace dynamatic
#endif // DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
