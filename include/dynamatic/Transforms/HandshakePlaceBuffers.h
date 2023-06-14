//===- HandshakePlaceBuffers.h - Place buffers in DFG -----------*- C++ -*-===//
//
// This file declares the --place-buffers pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/UtilsForPlaceBuffers.h"
#include "dynamatic/Transforms/UtilsForExtractMG.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakePlaceBuffersPass(bool firstMG=false, std::string stdLevelInfo="");

namespace buffer {

class BufferPlacementStrategy {
public:
  std::optional<int> minSlots = {}; // min number of slots (none means no minimum value)
  std::optional<int> maxSlots = {}; // max number of slots (none means no maximum value)
  bool transparentAllowed = true; // allowed to place transparent buffers?
  bool nonTransparentAllowed = true; // allowed to place non-transparent buffers?
  bool bufferizable = true; // allowed to place a buffer at all?

  virtual void getChannelConstraints(Value *valPort);

  virtual ~BufferPlacementStrategy() = default;
};

} // namespace buffer
} // namespace dynamatic


#endif // DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
