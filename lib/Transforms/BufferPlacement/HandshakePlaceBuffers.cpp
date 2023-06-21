//===- HandshakePlaceBuffers.cpp - Place buffers in DFG ---------*- C++ -*-===//
//
// This file implements the --place-buffers pass for throughput optimization by
// inserting buffers in the data flow graphs.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
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




void buffer::dfsHandshakeGraph(Operation *opNode, 
                               std::vector<Operation *> &visited) {

  if (std::find(visited.begin(), visited.end(), opNode) != visited.end()) {
    return;
  }
  // marked as visited
  visited.push_back(opNode);
  // dfs the successor operation
  for (auto sucOp : opNode->getUsers()) 
    if (!sucOp->getUsers().empty())
      dfsHandshakeGraph(sucOp, visited);
}

/// ================== dataFlowCircuit Function ================== ///
void buffer::DataflowCircuit::printCircuits() {
  llvm::errs() << "===========================\n";
  for (auto unit : units) {
    llvm::errs() << "operation: " << *(unit) << "\n";
  }
}

DataflowCircuit buffer::createCFDFCircuit(std::vector<Operation *> &unitList,
                                           std::map<ArchBB *, bool> &archs,
                                           std::map<unsigned, bool> &bbs) {
  DataflowCircuit circuit = DataflowCircuit();
  for (auto unit : unitList) {
    int bbIndex = getBBIndex(unit);

    // insert units in the selected basic blocks
    if (bbs.count(bbIndex) > 0 && bbs[bbIndex]) {
      circuit.units.push_back(unit);
      // insert channels if it is selected
      for (auto port : unit->getResults())
        if (isSelect(archs, &port) || isSelect(bbs, &port))
          circuit.channels.push_back(&port);
    }
  }
  circuit.printCircuits();
  return circuit;
}

static LogicalResult insertBuffers(handshake::FuncOp funcOp, MLIRContext *ctx,
                                   bool firstMG, std::string stdLevelInfo) {

  if (failed(verifyAllValuesHasOneUse(funcOp))) {
    funcOp.emitOpError() << "not all values are used exactly once";
    return failure(); // or do something that makes sense in the context
  }

  std::vector<Operation *> visitedOpList;

  // DFS build the DataflowCircuit from the handshake level
  for (auto &op : funcOp.getOps())
    if (isEntryOp(&op))
      dfsHandshakeGraph(&op, visitedOpList);

  // create CFDFC circuits
  std::vector<DataflowCircuit *> DataflowCircuitList;

  // read the bb file from std level
  std::map<ArchBB *, bool> archs;
  std::map<unsigned, bool> bbs;
  if (failed(readSimulateFile(stdLevelInfo, archs, bbs)))
    return failure();

  int execNum = extractCFDFCircuit(archs, bbs);
  while (execNum > 0) {
    // write the execution frequency to the DataflowCircuit
    llvm::errs() << "execNum: " << execNum << "\n";
    auto circuit = createCFDFCircuit(visitedOpList, archs, bbs);
    circuit.execN = execNum;
    DataflowCircuitList.push_back(&circuit);
    if (firstMG)
      break;
    execNum = extractCFDFCircuit(archs, bbs);
  }
  
  return success();
}

namespace {
struct PlaceBuffersPass : public PlaceBuffersBase<PlaceBuffersPass> {

  PlaceBuffersPass(bool firstMG, std::string stdLevelInfo) {
    this->firstMG = firstMG;
    this->stdLevelInfo = stdLevelInfo;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if (failed(insertBuffers(funcOp, &getContext(), firstMG, stdLevelInfo)))
        return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakePlaceBuffersPass(bool firstMG,
                                           std::string stdLevelInfo) {
  return std::make_unique<PlaceBuffersPass>(firstMG, stdLevelInfo);
}