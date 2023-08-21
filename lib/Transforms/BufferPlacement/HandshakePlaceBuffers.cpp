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
#include "experimental/Support/StdProfiler.h"
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
using namespace dynamatic::experimental;

/// Create the CFDFCircuit based on the extraction results
static CFDFC createCFDFCircuit(handshake::FuncOp funcOp, SelectedArchs &archs,
                               SelectedBBs &bbs) {
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
    Operation *opDst = *channel.getUsers().begin();

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

namespace {

struct HandshakePlaceBuffersPass
    : public HandshakePlaceBuffersBase<HandshakePlaceBuffersPass> {

  HandshakePlaceBuffersPass(bool firstCFDFC, std::string &stdLevelInfo,
                            std::string &timefile, double targetCP,
                            int timeLimit, bool setCustom) {
    this->firstCFDFC = firstCFDFC;
    this->stdLevelInfo = stdLevelInfo;
    this->timefile = timefile;
    this->targetCP = targetCP;
    this->timeLimit = timeLimit;
    this->setCustom = setCustom;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

#ifdef DYNAMATIC_GUROBI_NOT_INSTALLED
    mod.emitError() << "Project was built without Gurobi installed, can't "
                       "run smart buffer placement pass\n";
    return signalPassFailure();
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

    // Buffer placement requires that all values are used exactly once in the IR
    for (auto funcOp : mod.getOps<handshake::FuncOp>())
      if (failed(verifyAllValuesHasOneUse(funcOp))) {
        funcOp.emitOpError() << "not all values are used exactly once";
        return signalPassFailure();
      }

    // Place buffers in each function
    for (auto funcOp : mod.getOps<handshake::FuncOp>())
      if (failed(insertBuffers(funcOp)))
        return signalPassFailure();
  };

private:
  LogicalResult insertBuffers(FuncOp funcOp);
};
} // namespace

LogicalResult HandshakePlaceBuffersPass::insertBuffers(FuncOp funcOp) {

  // vectors to store CFDFC circuits
  std::vector<CFDFC> cfdfcList;

  // read the simulation file from std level, create map to indicate whether
  // the bb is selected, and whether the arch between bbs is selected in each
  // round of extraction

  // Read the CSV containing arch information (number of transitions between
  // pair of basic blocks)
  SmallVector<ArchBB> archList;
  if (failed(StdProfiler::readCSV(stdLevelInfo, archList)))
    return failure();

  // Map each arch to a boolean that indicates whether it is selected in each
  // CFDFC extraction round. We use a pointer to each arch as the key type to
  // allow us to modify their frequencies during CFDFC extractions without
  // messing up key hashes
  SelectedArchs archs;
  // Similarly, map each basic block ID to a boolean to indicate whether it is
  // selected in each CFDFC extraction round.
  SelectedBBs bbs;
  for (ArchBB &arch : archList) {
    archs.insert(std::make_pair(&arch, false));
    bbs.insert(std::make_pair(arch.srcBB, false));
    bbs.insert(std::make_pair(arch.dstBB, false));
  }

  unsigned freq;
  if (failed(extractCFDFC(archs, bbs, freq))) {
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
    if (firstCFDFC)
      break;
    if (failed(extractCFDFC(archs, bbs, freq))) {
      return failure();
    }
  }

  // Instantiate all the channels of MILP model in different CFDFC
  std::vector<Value> allChannels;
  auto startNode = *(funcOp.front().getArguments().end() - 1);
  for (Operation *op : startNode.getUsers())
    for (auto opr : op->getOperands())
      allChannels.push_back(opr);

  for (Operation &op : funcOp.getOps())
    for (auto resOp : op.getResults())
      allChannels.push_back(resOp);

  // Create the MILP model of buffer placement, and write the results of the
  // model to insertBufResult.

  std::map<std::string, UnitInfo> unitInfo;
  DenseMap<Value, ChannelBufProps> channelBufProps;

  if (failed(parseJson(timefile, unitInfo)))
    return failure();

  // load the buffer information of the units to channel
  if (failed(setChannelBufProps(allChannels, channelBufProps, unitInfo)))
    return failure();

  DenseMap<Value, Result> insertBufResult;

  if (failed(placeBufferInCFDFCircuit(insertBufResult, funcOp, allChannels,
                                      cfdfcList, cfdfcInds, targetCP, timeLimit,
                                      setCustom, unitInfo, channelBufProps)))
    return failure();

  if (failed(instantiateBuffers(insertBufResult, &getContext())))
    return failure();

  return success();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakePlaceBuffersPass(bool firstCFDFC,
                                           std::string stdLevelInfo,
                                           std::string timefile,
                                           double targetCP, int timeLimit,
                                           bool setCustom) {
  return std::make_unique<HandshakePlaceBuffersPass>(
      firstCFDFC, stdLevelInfo, timefile, targetCP, timeLimit, setCustom);
}
