//===- HandshakePlaceBuffers.cpp - Place buffers in DFG ---------*- C++ -*-===//
//
// This file implements the --place-buffers pass for throughput optimization by
// inserting buffers in the data flow graphs.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakePlaceBuffers.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include <optional>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

static LogicalResult insertBuffers(handshake::FuncOp funcOp, MLIRContext *ctx) {

  std::vector<Operation *> visitedOpList;
  std::vector<unsigned> bbIndexList;
  // std::vector<basicBlock *> bbList;
  std::vector<unit *> unitList;

  unsigned maxBBInd = 0;

  // entry operations for DFS 
  SmallVector<Operation *> entryOps;
  // build the dataflowCircuit from the handshake level
  for (auto &op : funcOp.getOps()) {
    unsigned bbInd = getBBIndex(&op); 
    if (isEntryOp(&op, visitedOpList))
      dfsHandshakeGraph(&op, visitedOpList, unitList);
    if (bbInd != UINT_MAX && bbInd > maxBBInd)
      maxBBInd = bbInd;
  }

  // speficy by a flag, read the bb file 
  return success();
}


namespace {

/// Simple driver for prepare for legacy pass.
struct PlaceBuffersPass : public PlaceBuffersBase<PlaceBuffersPass> {

  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if (failed(insertBuffers(funcOp, &getContext())))
        return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakePlaceBuffersPass() {
  return std::make_unique<PlaceBuffersPass>();
}
