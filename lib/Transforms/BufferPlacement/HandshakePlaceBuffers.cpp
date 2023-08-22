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

static unsigned getPortInd(Operation *op, Value val) {
  for (auto [indVal, port] : llvm::enumerate(op->getResults())) {
    if (port == val) {
      return indVal;
    }
  }
  return UINT_MAX;
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
#else
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
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
  };

private:
  LogicalResult insertBuffers(FuncOp funcOp);
};
} // namespace

LogicalResult HandshakePlaceBuffersPass::insertBuffers(FuncOp funcOp) {
  // Read the CSV containing arch information (number of transitions between
  // pair of basic blocks)
  SmallVector<ArchBB> archList;
  if (failed(StdProfiler::readCSV(stdLevelInfo, archList)))
    return failure();

  // Store all archs in a set. We use a pointer to each arch as the key type to
  // allow us to modify their frequencies during CFDFC extractions without
  // messing up key hashes
  ArchSet archs;
  // Similarly, store all block IDs in a set.
  BBSet bbs;
  for (ArchBB &arch : archList) {
    archs.insert(&arch);
    bbs.insert(arch.srcBB);
    bbs.insert(arch.dstBB);
  }

  // List of CFDFCs extracted from the function
  std::vector<CFDFC> cfdfcs;

  // Create the CFDFC index list that will be optimized
  std::vector<unsigned> cfdfcInds;
  unsigned cfdfcInd = 0;

  ArchSet selectedArchs;
  BBSet selectedBBs;
  do {
    // Clear the sets of selected archs and BBs
    selectedArchs.clear();
    selectedBBs.clear();

    // Try to extract the next CFDFC
    unsigned numTrans = extractCFDFC(archs, bbs, selectedArchs, selectedBBs);
    if (numTrans == 0)
      break;

    // Create the CFDFC from the set of selected archs and BBs
    cfdfcs.emplace_back(funcOp, selectedArchs, selectedBBs, numTrans);
    cfdfcInds.push_back(cfdfcInd++);
  } while (firstCFDFC);

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
                                      cfdfcs, cfdfcInds, targetCP, timeLimit,
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
