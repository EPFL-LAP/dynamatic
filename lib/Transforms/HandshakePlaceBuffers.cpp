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
#include <optional>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
static LogicalResult insertBuffers(handshake::FuncOp funcOp, MLIRContext *ctx) {
  std::vector<Operation *> visitedOpList;
  std::vector<unsigned> bbIndexList;
  std::vector<basicBlock *> bbList;

  Operation *entryOp = foundEntryOp(funcOp, visitedOpList);

  if (entryOp == nullptr) {
    return failure();
  }

  unsigned maxBBInd = 0;
  for (auto &op : funcOp.getOps())
    if (unsigned bbInd = getBBIndex(&op); bbInd != UINT_MAX && bbInd > maxBBInd)
      maxBBInd = bbInd;

  // build the entry basic block
  basicBlock *entryBB = new basicBlock();
  entryBB->isEntryBB = true;
  entryBB->index = 0;
  // bbIndexList.push_back(entryBB->index);
  bbList.push_back(entryBB);
  dfsBBGraphs(entryOp, visitedOpList, entryBB, bbList);

  basicBlock *prevBB = entryBB;

  llvm::errs() << "---------------entry bb finished-------------\n";

  // DFS from the entryBB, the BB graph can be fully traversed
  // as every basic blocks are connected.
  dfsBB(entryBB, bbList, bbIndexList, visitedOpList);
  assert(bbIndexList.size() == maxBBInd + 1 &&
         "Disconnected basic blocks exist!");

  printBBConnectivity(bbList);

  extractMarkedGraphBB(bbList);

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
