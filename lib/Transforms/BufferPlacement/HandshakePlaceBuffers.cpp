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
#include "dynamatic/Transforms/BufferPlacement/OptimizeMILP.h"
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

static LogicalResult instantiateBuffers(std::map<Value *, Result> &res,
                                        MLIRContext *ctx) {
  OpBuilder builder(ctx);
  for (auto &[channel, result] : res) {

    if (result.numSlots > 0) {
      Operation *opSrc = channel->getDefiningOp();
      Operation *opDst = getUserOp(*channel);

      unsigned numOp = 0;
      unsigned numTrans = 0;

      if (result.opaque) {
        numOp = 1;
        numTrans = 1;
      }
      if (result.transparent)
        numTrans = 1;

      if (int numBuf = result.numSlots - numOp - numTrans; numBuf > 0)
        numTrans += numBuf;

      builder.setInsertionPointAfter(opSrc);
      unsigned indVal = getPortInd(opSrc, *channel);

      if (indVal == UINT_MAX)
        continue;

      llvm::errs() << "insert buffer for: " << *channel << "opque: " << numOp
                   << "; transparent: " << numTrans << "\n";

      if (numOp > 0) {
        // insert opque buffer
        auto bufferOp = builder.create<handshake::BufferOp>(
            opDst->getLoc(), opSrc->getResult(indVal).getType(),
            opSrc->getResult(indVal));
        bufferOp.setBufferType(BufferTypeEnum::seq);
        bufferOp.setSlots(numOp);

        auto bufferTrans = builder.create<handshake::BufferOp>(
            bufferOp->getLoc(), bufferOp->getResult(0).getType(),
            bufferOp->getResult(0));
        bufferTrans.setSlots(numTrans);
        bufferTrans.setBufferType(BufferTypeEnum::fifo);
        opSrc->getResult(indVal).replaceUsesWithIf(
            bufferTrans.getResult(), [&](OpOperand &operand) {
              // return true;
              return operand.getOwner() == opDst;
            });
      } else if (numTrans > 0) {
        auto bufferTrans = builder.create<handshake::BufferOp>(
            opDst->getLoc(), opSrc->getResult(indVal).getType(),
            opSrc->getResult(indVal));
        bufferTrans.setSlots(numTrans);
        bufferTrans.setBufferType(BufferTypeEnum::fifo);
        opSrc->getResult(indVal).replaceUsesWithIf(
            bufferTrans.getResult(), [&](OpOperand &operand) {
              // return true;
              return operand.getOwner() == opDst;
            });
      }
    }
  }

  return success();
}

/// Delete the created archs map for MG extraction to avoid memory leak
static void deleleArchMap(std::map<ArchBB *, bool> &archs) {
  for (auto it = archs.begin(); it != archs.end(); ++it)
    delete it->first;
  // Clear the map
  archs.clear();
}

static LogicalResult insertBuffers(handshake::FuncOp funcOp, MLIRContext *ctx,
                                   ChannelBufProps &strategy, bool firstMG,
                                   std::string stdLevelInfo, double targetCP) {

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

  // Create the MILP model of buffer placement, and write the results of the
  // model to insertBufResult.
  // Instantiate the buffers according to the results.
  for (auto dataflowCirct : cfdfcList) {
    std::map<Value *, Result> insertBufResult;
    placeBufferInCFDFCircuit(dataflowCirct, insertBufResult, targetCP);
    instantiateBuffers(insertBufResult, ctx);
  }

  return success();
}

namespace {
struct HandshakePlaceBuffersPass
    : public HandshakePlaceBuffersBase<HandshakePlaceBuffersPass> {

  HandshakePlaceBuffersPass(bool firstMG, std::string stdLevelInfo,
                            double targetCP) {
    this->firstMG = firstMG;
    this->stdLevelInfo = stdLevelInfo;
    this->targetCP = targetCP;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    ChannelBufProps strategy;
    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if (failed(insertBuffers(funcOp, &getContext(), strategy, firstMG,
                               stdLevelInfo, targetCP)))
        return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakePlaceBuffersPass(bool firstMG,
                                           std::string stdLevelInfo,
                                           double targetCP) {
  return std::make_unique<HandshakePlaceBuffersPass>(firstMG, stdLevelInfo,
                                                     targetCP);
}
