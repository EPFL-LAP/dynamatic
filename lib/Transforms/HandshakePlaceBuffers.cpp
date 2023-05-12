//===- HandshakePlaceBuffers.cpp - Place buffers in DFG ---------*- C++ -*-===//
//
// This file implements the --place-buffers pass for throughput optimization by
// inserting buffers in the data flow graphs.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakePlaceBuffers.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

static LogicalResult insertBuffers(func::FuncOp funcOp, MLIRContext *ctx) {
  // Operation *entryOp = foundEntryOp(funcOp);
  return success();
}
namespace {

/// Simple driver for prepare for legacy pass.
struct PlaceBuffersPass
    : public PlaceBuffersBase<PlaceBuffersPass> {

  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto funcOp : m.getOps<func::FuncOp>())
      if (failed(insertBuffers(funcOp, &getContext())))
        return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakePlaceBuffersPass() {
  return std::make_unique<PlaceBuffersPass>();
}
