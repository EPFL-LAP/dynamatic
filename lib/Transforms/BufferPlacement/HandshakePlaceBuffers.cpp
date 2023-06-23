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

/// Create the CFDFCircuit from the unitList(the DFS operations graph),
/// and archs, and bbs that store the CFDFC extraction results indicating
/// selected (1) or not (0).
static DataflowCircuit createCFDFCircuit(std::vector<Operation *> &unitList,
                                           std::map<ArchBB *, bool> &archs,
                                           std::map<unsigned, bool> &bbs) {
  DataflowCircuit circuit = DataflowCircuit();
  for (auto unit : unitList) {
    int bbIndex = getBBIndex(unit);

    // insert units in the selected basic blocks
    if (bbs.count(bbIndex) > 0 && bbs[bbIndex]) {
      // llvm::errs() << "insert unit: " << *unit << "\n";
      circuit.units.push_back(unit);
      // insert channels if it is selected
      for (auto port : unit->getResults())
        if (isSelect(archs, &port) || isSelect(bbs, &port)){
          circuit.channels.push_back(port);
          // llvm::errs() << "insert channel: " << port << "\n";
        }
    }
  }
  circuit.printCircuits();
  return circuit;
}

static LogicalResult instantiateBuffers(std::map<Value *, Result> &res,
    MLIRContext *ctx) {
  OpBuilder builder(ctx);
  for (auto pair : res) {
    llvm::errs() << "insert buffer for: " << pair.second.numSlots << "\n";
    if (pair.second.numSlots > 0) {
      Operation *opSrc = pair.first->getDefiningOp();
      Operation *opDst = getUserOp(pair.first);
      builder.setInsertionPointAfter(opSrc);
      auto bufferOp = builder.create<handshake::BufferOp>
                        (opSrc->getLoc(),
                        opSrc->getResult(0).getType(),
                        opSrc->getResult(0));

      if (pair.second.transparent)
        bufferOp.setBufferType(BufferTypeEnum::fifo);
      else
        bufferOp.setBufferType(BufferTypeEnum::seq);
      bufferOp.setSlots(pair.second.numSlots);
      opSrc->getResult(0).replaceUsesWithIf(bufferOp.getResult(),
                                            [&](OpOperand &operand) {
                                              // return true;
                                              return operand.getOwner() == opDst;
                                            });
    }
  }
  return success();
}

static LogicalResult insertBuffers(handshake::FuncOp funcOp, MLIRContext *ctx,
                                   BufferPlacementStrategy &strategy,
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
  std::vector<DataflowCircuit > dataflowCircuitList;

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
    circuit.targetCP = 3.0;
    dataflowCircuitList.push_back(circuit);
    if (firstMG)
      break;
    execNum = extractCFDFCircuit(archs, bbs);
  }

  for (auto dataflowCirct : dataflowCircuitList) {
    std::map<Value *, Result> res;
    dataflowCirct.createMILPModel(strategy, res);
    instantiateBuffers(res, ctx);
  }
  
  return success();
}

namespace {
class customBufferPlaceStrategy : public BufferPlacementStrategy {
  public:
    ChannelConstraints getChannelConstraints(Value *val) override {
      ChannelConstraints constraints;
      // set the channel constraints according to the global constraints
      constraints = this->globalConstraints;

      Operation *dstOp;
      for (auto user : val->getUsers()) {
        dstOp = user;
        break;
      }
      
      if (isa<handshake::MergeOp, handshake::MuxOp>(val->getDefiningOp())) {
        constraints.minSlots = 1;
        constraints.maxSlots = 1;
        constraints.bufferizable = true;
        constraints.transparentAllowed = true;
        constraints.nonTransparentAllowed = false;
      }

      if (isa<handshake::ControlMergeOp>(val->getDefiningOp()) ||
          isa<handshake::ControlMergeOp>(dstOp) ) {
        constraints.bufferizable = false;
      }
      return constraints;
    }
};

struct PlaceBuffersPass : public PlaceBuffersBase<PlaceBuffersPass> {

  PlaceBuffersPass(bool firstMG, std::string stdLevelInfo) {
    this->firstMG = firstMG;
    this->stdLevelInfo = stdLevelInfo;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    customBufferPlaceStrategy strategy;
    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if (failed(insertBuffers(funcOp, &getContext(), strategy, firstMG, stdLevelInfo)))
        return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakePlaceBuffersPass(bool firstMG,
                                           std::string stdLevelInfo) {
  return std::make_unique<PlaceBuffersPass>(firstMG, stdLevelInfo);
}