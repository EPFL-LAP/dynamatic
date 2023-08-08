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
#include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"
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

/// Create the CFDFCircuit based on the extraction results
static CFDFC createCFDFCircuit(handshake::FuncOp funcOp,
                               std::map<ArchBB *, bool> &archs,
                               std::map<unsigned, bool> &bbs) {
  CFDFC circuit = CFDFC();
  for (auto &op : funcOp.getOps()) {
    int bbIndex = getBBIndex(&op);

    // insert units in the selected basic blocks
    if (bbs.count(bbIndex) > 0 && bbs[bbIndex]) {
      circuit.units.push_back(&op);
      // insert channels if it is selected
      for (auto port : op.getResults())
        if (isSelect(archs, port) || isSelect(bbs, port))
          circuit.channels.push_back(port);
    }
  }
  return circuit;
}

static void deleleArchMap(std::map<ArchBB *, bool> &archs) {
  for (auto it = archs.begin(); it != archs.end(); ++it)
    delete it->first;
  // Clear the map
  archs.clear();
}

static LogicalResult insertBuffers(handshake::FuncOp funcOp, MLIRContext *ctx,
                                   bool firstMG, std::string stdLevelInfo) {

  if (failed(verifyAllValuesHasOneUse(funcOp))) {
    funcOp.emitOpError() << "not all values are used exactly once";
    return failure();
  }

  // vectors to store CFDFC circuits
  std::vector<CFDFC> cfdfcList;

  // read the simulation file from std level, create map to indicate whether
  // the bb is selected, and whether the arch between bbs is selected in each
  // round of extraction
  std::map<ArchBB *, bool> archs;
  std::map<unsigned, bool> bbs;
  if (failed(readSimulateFile(stdLevelInfo, archs, bbs)))
    return failure();

  unsigned freq;
  if (failed(extractCFDFCircuit(archs, bbs, freq))) {
    deleleArchMap(archs);
    return failure();
  }
  while (freq > 0) {
    // write the execution frequency to the CFDFC
    auto circuit = createCFDFCircuit(funcOp, archs, bbs);
    circuit.execN = freq;
    cfdfcList.push_back(circuit);
    if (firstMG)
      break;
    if (failed(extractCFDFCircuit(archs, bbs, freq))) {
      deleleArchMap(archs);
      return failure();
    }
  }

  deleleArchMap(archs);

  return success();
}

namespace {
struct HandshakePlaceBuffersPass
    : public HandshakePlaceBuffersBase<HandshakePlaceBuffersPass> {

  HandshakePlaceBuffersPass(bool firstMG, std::string stdLevelInfo) {
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
  return std::make_unique<HandshakePlaceBuffersPass>(firstMG, stdLevelInfo);
}
