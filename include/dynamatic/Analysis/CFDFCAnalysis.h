#pragma once
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace llvm;
using namespace dynamatic::buffer;

namespace dynamatic {

struct BufferPlacementResult {
  std::vector<CFDFC> cfdfcs;
  std::vector<double> cfdfcThroughput;
};

/// This Analysis is preserved after buffer placement
struct PerformanceAnalysis {
  DenseMap<handshake::FuncOp, BufferPlacementResult> results;
  PerformanceAnalysis(mlir::ModuleOp *modOp) {}
};

}; // namespace dynamatic
