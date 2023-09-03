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
#include "dynamatic/Transforms/BufferPlacement/ExtractCFDFC.h"
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

#ifdef DYNAMATIC_GUROBI_NOT_INSTALLED
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.emitError() << "Project was built without Gurobi installed, can't "
                       "run smart buffer placement pass\n";
    return signalPassFailure();
  }
#else
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Buffer placement requires that all values are used exactly once in the IR
    for (handshake::FuncOp funcOp : mod.getOps<handshake::FuncOp>()) {
      if (failed(verifyAllValuesHasOneUse(funcOp))) {
        funcOp.emitOpError() << "not all values are used exactly once";
        return signalPassFailure();
      }
    }

    // Read the operations' timing models from disk
    TimingDatabase timingDB(&getContext());
    if (failed(TimingDatabase::readFromJSON(timefile, timingDB)))
      return signalPassFailure();

    // Place buffers in each function
    for (handshake::FuncOp funcOp : mod.getOps<handshake::FuncOp>()) {
      // Get CFDFCs from the function
      SmallVector<CFDFC> cfdfcs;
      if (failed(getCFDFCs(funcOp, cfdfcs)))
        return signalPassFailure();

      // Solve the MILP to obtain a buffer placement
      DenseMap<Value, PlacementResult> placement;
      if (failed(getBufferPlacement(funcOp, cfdfcs, timingDB, placement)))
        return signalPassFailure();

      if (failed(instantiateBuffers(placement)))
        return signalPassFailure();
    }
  };

private:
  LogicalResult getCFDFCs(FuncOp funcOp, SmallVector<CFDFC> &cfdfcs);

  LogicalResult getBufferPlacement(FuncOp funcOp, SmallVector<CFDFC> &cfdfcs,
                                   TimingDatabase &timingDB,
                                   DenseMap<Value, PlacementResult> &placement);

  LogicalResult instantiateBuffers(DenseMap<Value, PlacementResult> &res);
#endif
};
} // namespace

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
LogicalResult HandshakePlaceBuffersPass::getCFDFCs(FuncOp funcOp,
                                                   SmallVector<CFDFC> &cfdfcs) {

  // Read the CSV containing arch information (number of transitions between
  // pairs of basic blocks)
  SmallVector<ArchBB> archList;
  if (failed(StdProfiler::readCSV(stdLevelInfo, archList)))
    return funcOp->emitError()
           << "Failed to reaf profiling information from CSV";

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

  // Set of selected archs
  ArchSet selectedArchs;
  // Set of selected basic blocks
  BBSet selectedBBs;
  // Number of executions
  unsigned numExec;
  do {
    // Clear the sets of selected archs and BBs
    selectedArchs.clear();
    selectedBBs.clear();

    // Try to extract the next CFDFC
    if (failed(extractCFDFC(funcOp, archs, bbs, selectedArchs, selectedBBs,
                            numExec)))
      return failure();
    if (numExec == 0)
      break;

    // Create the CFDFC from the set of selected archs and BBs
    cfdfcs.emplace_back(funcOp, selectedArchs, selectedBBs, numExec);
  } while (firstCFDFC);

  return success();
}

LogicalResult HandshakePlaceBuffersPass::getBufferPlacement(
    FuncOp funcOp, SmallVector<CFDFC> &cfdfcs, TimingDatabase &timingDB,
    DenseMap<Value, PlacementResult> &placement) {
  // All CFDFCs must be optimized
  llvm::MapVector<CFDFC *, bool> cfdfcsOpt;
  for (CFDFC &cd : cfdfcs)
    cfdfcsOpt[&cd] = true;

  // Create Gurobi environment
  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();

  // Create and solve the MILP
  BufferPlacementMILP milp(funcOp, cfdfcsOpt, timingDB, targetCP,
                           targetCP * 2.0, env, timeLimit);
  return success(milp.arePlacementConstraintsSatisfiable() &&
                 !failed(milp.setup()) && !failed(milp.optimize(placement)));
}

LogicalResult HandshakePlaceBuffersPass::instantiateBuffers(
    DenseMap<Value, PlacementResult> &res) {
  OpBuilder builder(&getContext());
  for (auto &[channel, placement] : res) {
    Operation *opSrc = channel.getDefiningOp();
    Operation *opDst = *channel.getUsers().begin();
    builder.setInsertionPointAfter(opSrc);

    Value bufferIn = channel;
    if (placement.numOpaque > 0) {
      // Insert an opaque buffer
      Value bufferRes = builder
                            .create<handshake::BufferOp>(
                                channel.getLoc(), bufferIn, placement.numOpaque,
                                BufferTypeEnum::seq)
                            .getResult();
      opDst->replaceUsesOfWith(bufferIn, bufferRes);
      bufferIn = bufferRes;
    }
    if (placement.numTrans > 0) {
      // Insert a transparent buffer, potentially after an opaque buffer
      Value bufferRes = builder
                            .create<handshake::BufferOp>(
                                channel.getLoc(), bufferIn, placement.numTrans,
                                BufferTypeEnum::fifo)
                            .getResult();
      opDst->replaceUsesOfWith(bufferIn, bufferRes);
    }
  }
  return success();
}
#endif

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::buffer::createHandshakePlaceBuffersPass(
    bool firstCFDFC, std::string stdLevelInfo, std::string timefile,
    double targetCP, int timeLimit, bool setCustom) {
  return std::make_unique<HandshakePlaceBuffersPass>(
      firstCFDFC, stdLevelInfo, timefile, targetCP, timeLimit, setCustom);
}
