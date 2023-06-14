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
/// Strategy class to control the contraints over CFDFC optimization.
/// The class exposes overridable filter functions to allow dynamatically selct
/// addtional constraints over the  CFDFC optimization.
class constrStrategy {
public:
  int maxBufSize = INT_MAX;
  int minBufSize = -1;
  /// Determine whether a specific port type should be constrained (i.e.,
  /// For port definition op is mux or merge that have multiple input ports)
  /// port constraint should be added. 
  virtual bool placeOneBufferOnPort (Value *valPort) {return false;};

  /// Whether set constraints on maximum buffer size. 
  virtual bool constrainMaxBufSize () {return false;};

  /// Whether set constraints on minimum buffer size.
  virtual bool constrainMinBufSize () {return false;};

  /// Whether adopt constraints published on FPL22.
  virtual bool constrainOnFPL22 () {return false;};

  virtual ~constrStrategy() = default;
};

} // namespace buffer
} // namespace dynamatic


#endif // DYNAMATIC_TRANSFORMS_PLACEBUFFERS_H
