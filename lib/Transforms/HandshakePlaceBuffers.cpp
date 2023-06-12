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

static LogicalResult insertBuffers(handshake::FuncOp funcOp, 
                                   MLIRContext *ctx,
                                   bool stdLevel) {

  std::vector<Operation *> visitedOpList;
  // std::vector<unsigned> bbIndexList;
  std::vector<unit *> unitList;

  unsigned maxBBInd = 0;
  // unsigned numOps;

  // entry operations for DFS 
  SmallVector<Operation *> entryOps;
  // build the dataflowCircuit from the handshake level
  for (auto &op : funcOp.getOps()) {
    unsigned bbInd = getBBIndex(&op); 
    if (isEntryOp(&op, visitedOpList))
      dfsHandshakeGraph(&op, unitList, visitedOpList);
    if (bbInd != UINT_MAX && bbInd > maxBBInd)
      maxBBInd = bbInd;
  }

  std::vector<dataFlowCircuit *> dataFlowCircuitList;
  // speficy by a flag, read the bb file 
  if (stdLevel){
    std::string folder = 
      "/home/yuxuan/Projects/dynamatic-utils/benchmarks/FPL22/";
    std::string fileName = folder + funcOp.getName().str() + 
                           "/dynamatic/comp/std_bb.dat";
    std::map<archBB*, int> archs;
    std::map<int, int> bbs;

    readSimulateFile(fileName, archs, bbs);

    int execNum = buffer::extractCFDFCircuit(archs, bbs);
    while (execNum > 0) {
      printCFDFCircuit(archs, bbs);
      auto circuit = createCFDFCircuit(unitList, archs, bbs);
      dataFlowCircuitList.push_back(circuit);
      execNum = buffer::extractCFDFCircuit(archs, bbs);
    }
    
  }

  return success();
}


namespace {

/// Simple driver for prepare for legacy pass.
struct PlaceBuffersPass : public PlaceBuffersBase<PlaceBuffersPass> {

  PlaceBuffersPass(bool stdLevel) {
    this->stdLevel = stdLevel;
  }
  
  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if (failed(insertBuffers(funcOp, &getContext(), stdLevel)))
        return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakePlaceBuffersPass(bool stdLevel) {
  return std::make_unique<PlaceBuffersPass>(stdLevel);
}
