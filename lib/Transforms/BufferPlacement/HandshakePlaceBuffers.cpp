//===- HandshakePlaceBuffers.cpp - Place buffers in DFG ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --place-buffers pass for throughput optimization by
// inserting buffers in the data flow graphs.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "experimental/Support/StdProfiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include <string>

using namespace llvm::sys;
using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;

namespace {

/// Thin wrapper around a `Logger` that allows to conditionally create the
/// resources to write to the file based on a flag provided during objet
/// creation. This enables us to use RAII to deallocate the logger only if it
/// was created.
class BufferLogger {
public:
  /// The underlying logger object, which may remain nullptr.
  Logger *log = nullptr;

  /// Optionally allocates a logger based on whether the `dumpLogs` flag is set.
  /// If it is, the log file's location is determined based om the provided
  /// function's name. On error, `ec` will contain a non-zero error code
  /// and the logger should not be used.
  BufferLogger(handshake::FuncOp funcOp, bool dumpLogs, std::error_code &ec);

  /// Returns the underlying logger, which may be nullptr.
  Logger *operator*() { return log; }

  /// Returns the underlying indented wrtier stream to the log file. Requires
  /// the object to have been created with the `dumpLogs` flag set to true.
  mlir::raw_indented_ostream &getStream() {
    assert(log && "logger was not allocated");
    return **log;
  }

  BufferLogger(const BufferLogger *) = delete;
  BufferLogger operator=(const BufferLogger *) = delete;

  /// Deletes the underlying logger object if it was allocated.
  ~BufferLogger() {
    if (log)
      delete log;
  }
};
} // namespace

BufferLogger::BufferLogger(handshake::FuncOp funcOp, bool dumpLogs,
                           std::error_code &ec) {
  if (!dumpLogs)
    return;

  std::string sep = path::get_separator().str();
  std::string fp = "buffer-placement" + sep + funcOp.getName().str() + sep;
  log = new Logger(fp + "placement.log", ec);
}

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED

/// Logs arch and CFDFC information (sequence of basic blocks, number of
/// executions, channels, units) to the logger.
static void logFuncInfo(FuncInfo &info, Logger &log) {
  mlir::raw_indented_ostream &os = *log;
  os << "# ===== #\n";
  os << "# Archs #\n";
  os << "# ===== #\n\n";

  for (ArchBB &arch : info.archs) {
    os << arch.srcBB << " -> " << arch.dstBB << " with " << arch.numTrans
       << " executions";
    if (arch.isBackEdge)
      os << " (backedge)";
    os << "\n";
  }

  os << "\n# ================ #\n";
  os << "# Extracted CFDFCs #\n";
  os << "# ================ #\n\n";

  for (auto [idx, cfAndOpt] : llvm::enumerate(info.cfdfcs)) {
    auto &[cf, _] = cfAndOpt;
    os << "CFDFC #" << idx << ": ";
    for (size_t i = 0, e = cf->cycle.size() - 1; i < e; ++i)
      os << cf->cycle[i] << " -> ";
    os << cf->cycle.back() << "\n";
    os.indent();
    os << "- Number of executions: " << cf->numExecs << "\n";
    os << "- Number of units: " << cf->units.size() << "\n";
    os << "- Number of channels: " << cf->channels.size() << "\n";
    os << "- Number of backedges: " << cf->backedges.size() << "\n\n";
    os.unindent();
  }
}

/// Performs a number of verifications to make sure that the Handshake function
/// whose information is passed as argument is valid for buffer placement.
static LogicalResult verifyFuncValidForPlacement(FuncInfo &info) {
  // Store all archs in a map for fast query time
  DenseMap<unsigned, llvm::SmallDenseSet<unsigned, 2>> transitions;
  for (ArchBB &arch : info.archs)
    transitions[arch.srcBB].insert(arch.dstBB);

  // Store the BB to which each block belongs for quick access later
  DenseMap<Operation *, std::optional<unsigned>> opBlocks;
  for (Operation &op : info.funcOp.getOps())
    opBlocks[&op] = getLogicBB(&op);

  for (Operation &op : info.funcOp.getOps()) {
    // Most operations should belong to a basic block for buffer placement to
    // work correctly. Don't outright fail in case one operation is outside of
    // all blocks but warn the user
    if (!isa<handshake::SinkOp, handshake::MemoryControllerOp>(&op))
      if (!getLogicBB(&op).has_value())
        op.emitWarning() << "Operation does not belong to any block, MILP "
                            "behavior may be suboptimal or incorrect.";

    std::optional<unsigned> srcBB = opBlocks[&op];
    for (OpResult res : op.getResults()) {
      Operation *user = *res.getUsers().begin();
      std::optional<unsigned> dstBB = opBlocks[user];

      // All transitions between blocks must exist in the original CFG
      if (srcBB.has_value() && dstBB.has_value() && *srcBB != *dstBB &&
          !transitions[*srcBB].contains(*dstBB))
        return op.emitError()
               << "Result " << res.getResultNumber() << " defined in block "
               << *srcBB << " is used in block " << *dstBB
               << ". This connection does not exist according to the CFG "
                  "graph. Solving the buffer placement MILP would yield an "
                  "incorrect placement.";
    }
  }
  return success();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

namespace {
struct HandshakePlaceBuffersPass
    : public HandshakePlaceBuffersBase<HandshakePlaceBuffersPass> {

  HandshakePlaceBuffersPass(const std::string &frequencies,
                            const std::string &timingModels, bool firstCFDFC,
                            double targetCP, unsigned timeout, bool dumpLogs) {
    this->frequencies = frequencies;
    this->timingModels = timingModels;
    this->firstCFDFC = firstCFDFC;
    this->targetCP = targetCP;
    this->timeout = timeout;
    this->dumpLogs = dumpLogs;
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
    DenseMap<handshake::FuncOp, FuncInfo> funcToInfo;

    // Verify that the IR is in a valid state for buffer placement
    for (handshake::FuncOp funcOp : mod.getOps<handshake::FuncOp>()) {
      // Buffer placement requires that all values are used exactly once
      if (failed(verifyAllValuesHasOneUse(funcOp))) {
        funcOp.emitOpError() << "Not all values are used exactly once";
        return signalPassFailure();
      }

      // Insert the function information into the map and retrieve a reference
      // to it
      funcToInfo.insert(std::make_pair(funcOp, FuncInfo(funcOp)));
      FuncInfo &info = funcToInfo[funcOp];

      // Read the CSV containing arch information (number of transitions between
      // pairs of basic blocks) from disk
      SmallVector<ArchBB> archs;
      if (failed(StdProfiler::readCSV(frequencies, info.archs))) {
        funcOp->emitError() << "Failed to read profiling information from CSV";
        return signalPassFailure();
      }

      // Now that we have the arch information, we can check the circuit's
      // structure
      if (failed(verifyFuncValidForPlacement(info)))
        return signalPassFailure();
    }

    // Read the operations' timing models from disk
    TimingDatabase timingDB(&getContext());
    if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
      return signalPassFailure();

    // Place buffers in each function
    for (handshake::FuncOp funcOp : mod.getOps<handshake::FuncOp>()) {
      FuncInfo &info = funcToInfo[funcOp];

      // Use a wrapper around a logger to benefit from RAII
      std::error_code ec;
      BufferLogger logger(funcOp, dumpLogs, ec);
      if (ec.value() != 0) {
        funcOp->emitError() << "Failed to create logger for function "
                            << funcOp.getName() << "\n"
                            << ec.message();
        return signalPassFailure();
      }

      // Get CFDFCs from the function
      SmallVector<CFDFC> cfdfcs;
      if (failed(getCFDFCs(info, logger, cfdfcs)))
        return signalPassFailure();

      // All extracted CFDFCs must be optimized
      for (CFDFC &cf : cfdfcs)
        info.cfdfcs[&cf] = true;

      if (dumpLogs)
        logFuncInfo(info, **logger);

      // Solve the MILP to obtain a buffer placement
      DenseMap<Value, PlacementResult> placement;
      if (failed(getBufferPlacement(info, timingDB, logger, placement)))
        return signalPassFailure();

      if (failed(instantiateBuffers(placement)))
        return signalPassFailure();
    }
  };

private:
  /// Identifes (using and MILP) and extract CFDFCs from the function using
  /// estimated transition frequencies between blocks.
  LogicalResult getCFDFCs(FuncInfo &info, BufferLogger &logger,
                          SmallVector<CFDFC> &cfdfcs);

  /// Computes an optimal buffer placement by solving a large MILP over the
  /// entire dataflow circuit represented by the function.
  LogicalResult getBufferPlacement(FuncInfo &info, TimingDatabase &timingDB,
                                   BufferLogger &logger,
                                   DenseMap<Value, PlacementResult> &placement);

  /// Instantiates buffers identified by the buffer placement MILP inside the
  /// IR.
  LogicalResult instantiateBuffers(DenseMap<Value, PlacementResult> &res);
#endif
};
} // namespace

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
LogicalResult HandshakePlaceBuffersPass::getCFDFCs(FuncInfo &info,
                                                   BufferLogger &logger,
                                                   SmallVector<CFDFC> &cfdfcs) {
  SmallVector<ArchBB> archsCopy(info.archs);

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

    // Path where to dump the MILP model and solutions, if necessary
    std::string logPath = "";
    if (dumpLogs)
      logPath = logger.log->getLogDir() + path::get_separator().str() +
                "cfdfc" + std::to_string(cfdfcs.size());

    // Try to extract the next CFDFC
    if (failed(extractCFDFC(info.funcOp, archs, bbs, selectedArchs, numExecs,
                            logPath)))
      return failure();
    if (numExecs == 0)
      break;

    // Create the CFDFC from the set of selected archs and BBs
    cfdfcs.emplace_back(info.funcOp, selectedArchs, numExecs);
  } while (!firstCFDFC);

  return success();
}

LogicalResult HandshakePlaceBuffersPass::getBufferPlacement(
    FuncInfo &info, TimingDatabase &timingDB, BufferLogger &log,
    DenseMap<Value, PlacementResult> &placement) {

  // Create Gurobi environment
  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  if (timeout > 0)
    env.set(GRB_DoubleParam_TimeLimit, timeout);
  env.start();

  Logger *milpLog = dumpLogs ? *log : nullptr;

  // Create and solve the MILP
  fpga20::FPGA20Buffers milp(info, timingDB, env, milpLog, targetCP,
                             targetCP * 2.0);
  return success(!failed(milp.optimize()) &&
                 !failed(milp.getPlacement(placement)));
}

LogicalResult HandshakePlaceBuffersPass::instantiateBuffers(
    DenseMap<Value, PlacementResult> &res) {
  OpBuilder builder(&getContext());
  for (auto &[channel, placement] : res) {
    Operation *opSrc = channel.getDefiningOp();
    Operation *opDst = *channel.getUsers().begin();
    builder.setInsertionPointAfter(opSrc);

    Value bufferIn = channel;
    auto placeBuffer = [&](BufferTypeEnum bufType, unsigned numSlots) {
      if (numSlots == 0)
        return;

      // Insert an opaque buffer
      auto bufOp = builder.create<handshake::BufferOp>(
          bufferIn.getLoc(), bufferIn, numSlots, bufType);
      inheritBB(opSrc, bufOp);
      Value bufferRes = bufOp.getResult();

      opDst->replaceUsesOfWith(bufferIn, bufferRes);
      bufferIn = bufferRes;
    };

    if (placement.opaqueBeforeTrans) {
      placeBuffer(BufferTypeEnum::seq, placement.numOpaque);
      placeBuffer(BufferTypeEnum::fifo, placement.numTrans);
    } else {
      placeBuffer(BufferTypeEnum::fifo, placement.numTrans);
      placeBuffer(BufferTypeEnum::seq, placement.numOpaque);
    }
  }
  return success();
}
#endif

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::buffer::createHandshakePlaceBuffersPass(
    const std::string &frequencies, const std::string &timingModels,
    bool firstCFDFC, double targetCP, unsigned timeout, bool dumpLogs) {
  return std::make_unique<HandshakePlaceBuffersPass>(
      frequencies, timingModels, firstCFDFC, targetCP, timeout, dumpLogs);
}
