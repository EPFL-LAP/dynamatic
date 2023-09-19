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
#include "dynamatic/Support/LogicBB.h"
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
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "BUFFER_PLACEMENT"

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
                            std::string &timefile, double targetCP) {
    this->firstCFDFC = firstCFDFC;
    this->stdLevelInfo = stdLevelInfo;
    this->timefile = timefile;
    this->targetCP = targetCP;
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
      FuncInfo funcInfo(funcOp);

      // Read the CSV containing arch information (number of transitions between
      // pairs of basic blocks) from disk
      SmallVector<ArchBB> archs;
      if (failed(StdProfiler::readCSV(stdLevelInfo, funcInfo.archs))) {
        funcOp->emitError() << "Failed to read profiling information from CSV";
        return signalPassFailure();
      }

      // Get CFDFCs from the function
      SmallVector<CFDFC> cfdfcs;
      if (failed(getCFDFCs(funcInfo, cfdfcs)))
        return signalPassFailure();

      // All extracted CFDFCs must be optimized
      for (CFDFC &cf : cfdfcs)
        funcInfo.cfdfcs[&cf] = true;

      // Solve the MILP to obtain a buffer placement
      DenseMap<Value, PlacementResult> placement;
      if (failed(getBufferPlacement(funcInfo, timingDB, placement)))
        return signalPassFailure();

      if (failed(instantiateBuffers(placement)))
        return signalPassFailure();
    }
  };

private:
  LogicalResult getCFDFCs(FuncInfo &funcInfo, SmallVector<CFDFC> &cfdfcs);

  LogicalResult getBufferPlacement(FuncInfo &funcInfo, TimingDatabase &timingDB,
                                   DenseMap<Value, PlacementResult> &placement);

  LogicalResult instantiateBuffers(DenseMap<Value, PlacementResult> &res);
#endif
};
} // namespace

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
LogicalResult HandshakePlaceBuffersPass::getCFDFCs(FuncInfo &funcInfo,
                                                   SmallVector<CFDFC> &cfdfcs) {
  SmallVector<ArchBB> archsCopy(funcInfo.archs);

  // Store all archs in a set. We use a pointer to each arch as the key type to
  // allow us to modify their frequencies during CFDFC extractions without
  // messing up key hashes
  ArchSet archs;
  // Similarly, store all block IDs in a set.
  BBSet bbs;
  for (ArchBB &arch : archsCopy) {
    archs.insert(&arch);
    bbs.insert(arch.srcBB);
    bbs.insert(arch.dstBB);
  }

  // Set of selected archs
  ArchSet selectedArchs;
  // Number of executions
  unsigned numExecs;
  do {
    // Clear the sets of selected archs and BBs
    selectedArchs.clear();

    // Try to extract the next CFDFC
    if (failed(
            extractCFDFC(funcInfo.funcOp, archs, bbs, selectedArchs, numExecs)))
      return failure();
    if (numExecs == 0)
      break;

    // Create the CFDFC from the set of selected archs and BBs
    cfdfcs.emplace_back(funcInfo.funcOp, selectedArchs, numExecs);
    // Print the CFDFC that was identified
    LLVM_DEBUG({
      CFDFC &cfdfc = cfdfcs.back();
      llvm::errs() << "Identified CFDFC with " << cfdfc.numExecs
                   << " executions, " << cfdfc.units.size() << " units, and "
                   << cfdfc.channels.size()
                   << " channels: " << selectedArchs.front()->srcBB;
      for (ArchBB *arch : selectedArchs) {
        llvm::errs() << " -> " << arch->dstBB;
      }
      llvm::errs() << "\n";
    });
  } while (!firstCFDFC);

  return success();
}

LogicalResult HandshakePlaceBuffersPass::getBufferPlacement(
    FuncInfo &funcInfo, TimingDatabase &timingDB,
    DenseMap<Value, PlacementResult> &placement) {

  // Create Gurobi environment
  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();

  // Create and solve the MILP
  BufferPlacementMILP milp(funcInfo, timingDB, targetCP, targetCP * 2.0, env);
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
      auto bufOp = builder.create<handshake::BufferOp>(
          channel.getLoc(), bufferIn, placement.numOpaque, BufferTypeEnum::seq);
      inheritBB(opSrc, bufOp);
      Value bufferRes = bufOp.getResult();

      opDst->replaceUsesOfWith(bufferIn, bufferRes);
      bufferIn = bufferRes;
    }
    if (placement.numTrans > 0) {
      // Insert a transparent buffer, potentially after an opaque buffer
      auto bufOp = builder.create<handshake::BufferOp>(
          channel.getLoc(), bufferIn, placement.numTrans, BufferTypeEnum::fifo);
      inheritBB(opSrc, bufOp);
      Value bufferRes = bufOp.getResult();
      opDst->replaceUsesOfWith(bufferIn, bufferRes);
    }
  }
  return success();
}
#endif

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::buffer::createHandshakePlaceBuffersPass(bool firstCFDFC,
                                                   std::string stdLevelInfo,
                                                   std::string timefile,
                                                   double targetCP) {
  return std::make_unique<HandshakePlaceBuffersPass>(firstCFDFC, stdLevelInfo,
                                                     timefile, targetCP);
}
