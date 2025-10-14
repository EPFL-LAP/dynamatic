#pragma once
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace llvm;
using namespace dynamatic::buffer;

namespace dynamatic {

using ListOfCFDFCs = std::vector<CFDFC>;

/// This Analysis is preserved after buffer placement
struct CFDFCAnalysis {
  std::map<handshake::FuncOp, ListOfCFDFCs> mapFuncOpToCFDFCs;
  CFDFCAnalysis(mlir::Operation *modOp) {}
};

}; // namespace dynamatic
