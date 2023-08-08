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
    if (bbIndex < 0)
      continue;

    // insert units in the selected basic blocks
    if (bbs[bbIndex]) {
      circuit.units.push_back(&op);
      // insert channels if it is selected
      for (auto port : op.getResults())
        if (isSelect(archs, port) || isSelect(bbs, port))
          circuit.channels.push_back(port);
    }
  }
  return circuit;
}

/// Instantiate the buffers based on the results of the MILP
static LogicalResult instantiateBuffers(DenseMap<Value, Result> &res,
                                        MLIRContext *ctx) {
  OpBuilder builder(ctx);
  for (auto &[channel, result] : res) {
    if (result.numSlots == 0)
      continue;
    Operation *opSrc = channel.getDefiningOp();
    Operation *opDst = getUserOp(channel);

    unsigned numOpque = 0;
    unsigned numTrans = 0;

    if (result.opaque)
      numOpque = result.numSlots;
    else
      numTrans = result.numSlots;

    builder.setInsertionPointAfter(opSrc);
    unsigned indVal = getPortInd(opSrc, channel);
    assert(indVal != UINT_MAX && "Insert buffers in non exsiting channels");

    if (numOpque > 0) {
      // insert opque buffer
      Value bufferOperand = opSrc->getResult(indVal);
      Value bufferRes =
          builder
              .create<handshake::BufferOp>(opSrc->getLoc(), bufferOperand,
                                           numOpque, BufferTypeEnum::seq)
              .getResult();
      if (numTrans > 0)
        bufferRes =
            builder
                .create<handshake::BufferOp>(opSrc->getLoc(), bufferRes,
                                             numTrans, BufferTypeEnum::fifo)
                .getResult();

      opDst->replaceUsesOfWith(bufferOperand, bufferRes);
    }

    // insert all transparent buffers
    if (numTrans > 0 && numOpque == 0) {
      Value bufferOperand = opSrc->getResult(indVal);
      Value bufferRes =
          builder
              .create<handshake::BufferOp>(opSrc->getLoc(), bufferOperand,
                                           numTrans, BufferTypeEnum::fifo)
              .getResult();
      opDst->replaceUsesOfWith(bufferOperand, bufferRes);
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

namespace {

struct HandshakePlaceBuffersPass
    : public HandshakePlaceBuffersBase<HandshakePlaceBuffersPass> {

  HandshakePlaceBuffersPass(bool firstMG, std::string stdLevelInfo,
                            std::string timefile, double targetCP) {
    this->firstMG = firstMG;
    this->stdLevelInfo = stdLevelInfo;
    this->timefile = timefile;
    this->targetCP = targetCP;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    ChannelBufProps strategy;
    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if (failed(insertBuffers(funcOp, &getContext())))
        return signalPassFailure();
  };

private:
  LogicalResult insertBuffers(FuncOp &funcOp, MLIRContext *ctx);
};

LogicalResult HandshakePlaceBuffersPass::insertBuffers(FuncOp &funcOp,
                                                       MLIRContext *ctx) {

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

  // Create the CFDFC index list that will be optimized
  std::vector<unsigned> cfdfcInds;
  unsigned cfdfcInd = 0;
  while (freq > 0) {
    // write the execution frequency to the CFDFC
    auto circuit = createCFDFCircuit(funcOp, archs, bbs);
    cfdfcInds.push_back(cfdfcInd++);
    circuit.execN = freq;
    cfdfcList.push_back(circuit);
    if (firstMG)
      break;
    if (failed(extractCFDFCircuit(archs, bbs, freq))) {
      deleleArchMap(archs);
      return failure();
    }
  }

  // Delete the CFDFC related results
  deleleArchMap(archs);

  // Instantiate all the channels of MILP model in different CFDFC
  std::vector<Value> allChannels;
  auto startNode = *(funcOp.front().getArguments().end() - 1);
  for (auto op : startNode.getUsers())
    for (auto opr : op->getOperands())
      allChannels.push_back(opr);

  for (auto &op : funcOp.getOps())
    for (auto resOp : op.getResults())
      allChannels.push_back(resOp);

  // Create the MILP model of buffer placement, and write the results of the
  // model to insertBufResult.

  std::map<std::string, UnitInfo> unitInfo;
  DenseMap<Value, ChannelBufProps> channelBufProps;

  parseJson(timefile, unitInfo);

  // load the buffer information of the units to channel
  if (failed(setChannelBufProps(allChannels, channelBufProps, unitInfo)))
    return failure();

  DenseMap<Value, Result> insertBufResult;

  if (failed(placeBufferInCFDFCircuit(insertBufResult, funcOp, allChannels,
                                      cfdfcList, cfdfcInds, targetCP, unitInfo,
                                      channelBufProps)))
    return failure();

  instantiateBuffers(insertBufResult, ctx);

  return success();
}
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakePlaceBuffersPass(bool firstMG,
                                           std::string stdLevelInfo,
                                           std::string timefile,
                                           double targetCP) {
  return std::make_unique<HandshakePlaceBuffersPass>(firstMG, stdLevelInfo,
                                                     timefile, targetCP);
}
