//===- HandshakePlaceBuffers.cpp - Place buffers in DFG ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Buffer placement in Handshake functions, using a set of available algorithms
// (all of which currently require Gurobi to solve MILPs). Buffers are placed to
// ensure circuit correctness and increase performance.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "dynamatic/Transforms/BufferPlacement/FPL22Buffers.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "experimental/Support/StdProfiler.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include <string>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;

/// Algorithms that do not require solving an MILP.
static constexpr llvm::StringLiteral ON_MERGES("on-merges");
#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
/// Algorithms that do require solving an MILP.
static constexpr llvm::StringLiteral FPGA20("fpga20"), FPL22("fpl22");
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

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

  /// Returns the underlying indented writer stream to the log file. Requires
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

  std::string sep = llvm::sys::path::get_separator().str();
  std::string fp = "buffer-placement" + sep + funcOp.getName().str() + sep;
  log = new Logger(fp + "placement.log", ec);
}

HandshakePlaceBuffersPass::HandshakePlaceBuffersPass(
    StringRef algorithm, StringRef frequencies, StringRef timingModels,
    bool firstCFDFC, double targetCP, unsigned timeout, bool dumpLogs) {
  this->algorithm = algorithm.str();
  this->frequencies = frequencies.str();
  this->timingModels = timingModels.str();
  this->firstCFDFC = firstCFDFC;
  this->targetCP = targetCP;
  this->timeout = timeout;
  this->dumpLogs = dumpLogs;
}

void HandshakePlaceBuffersPass::runDynamaticPass() {
  // Buffer placement requires that all values are used exactly once
  mlir::ModuleOp modOp = getOperation();
  if (failed(verifyIRMaterialized(modOp))) {
    modOp->emitError() << ERR_NON_MATERIALIZED_MOD;
    return;
  }

  // Map algorithms to the function to call to execute them
  llvm::MapVector<StringRef, LogicalResult (HandshakePlaceBuffersPass::*)()>
      allAlgorithms;
  allAlgorithms[ON_MERGES] = &HandshakePlaceBuffersPass::placeWithoutUsingMILP;
#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
  allAlgorithms[FPGA20] = &HandshakePlaceBuffersPass::placeUsingMILP;
  allAlgorithms[FPL22] = &HandshakePlaceBuffersPass::placeUsingMILP;
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

  // Check that the algorithm exists
  if (!allAlgorithms.contains(algorithm)) {
    llvm::errs() << "Unknown algorithm '" << algorithm
                 << "', possible choices are:\n";
    for (auto &algo : allAlgorithms)
      llvm::errs() << "\t- " << algo.first << "\n";
#ifdef DYNAMATIC_GUROBI_NOT_INSTALLED
    llvm::errs()
        << "\tYou cannot use any of the MILP-based placement algorithms "
           "because CMake did not detect a Gurobi installation on your "
           "machine. Install Gurobi and rebuild to make these options "
           "available.\n";
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
    return signalPassFailure();
  }

  // Make sure all operations are named (used to generate unique MILP variable
  // names).

  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  namer.nameAllUnnamedOps();

  // Call the right function
  auto func = allAlgorithms[algorithm];
  if (failed(((*this).*(func))()))
    return signalPassFailure();

  // run the delay selection logic again, writing it to the IR for processing in
  // the backend
  // In order tp avoid interleaving this IR writing with the value extraction,
  // we keep it seperate. this does mean redudant logic, but the Database
  // parsing is not a performance bottleneck, so this should be acceptable.
  // TODO : this should go into a bespoke function

  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    llvm::errs() << "=== TimindDB read failed ===\n";
  else
    llvm::errs() << "=== TimindDB read succeeded ===\n";
  modOp.walk([&](mlir::Operation *op) {
    if (llvm::isa<dynamatic::handshake::ArithOpInterface>(op)) {
      double delay;
      if (!failed(timingDB.getInternalCombinationalDelay(op, SignalType::DATA,
                                                         delay, targetCP))) {
        std::string delayStr = std::to_string(delay);
        std::replace(delayStr.begin(), delayStr.end(), '.', '_');
        op->setAttr("selected_delay",
                    mlir::StringAttr::get(op->getContext(), delayStr));
      }
    }
  });
}

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
LogicalResult HandshakePlaceBuffersPass::placeUsingMILP() {
  // Make sure that all operations in the IR are named (used to generate
  // variable names in the MILP)
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();
  if (!nameAnalysis.isAnalysisValid())
    return failure();
  if (!nameAnalysis.areAllOpsNamed()) {
    if (failed(nameAnalysis.walk(NameAnalysis::UnnamedBehavior::NAME)))
      return failure();
  }
  markAnalysesPreserved<NameAnalysis>();

  mlir::ModuleOp modOp = getOperation();

  // Check IR invariants and parse basic block archs from disk
  DenseMap<handshake::FuncOp, FuncInfo> funcToInfo;
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    funcToInfo.insert(std::make_pair(funcOp, FuncInfo(funcOp)));
    FuncInfo &info = funcToInfo[funcOp];

    // Read the CSV containing arch information (number of transitions between
    // pairs of basic blocks) from disk. While the rest of this pass works if
    // the module contains multiple functions, this only makes sense if the
    // module has a single function
    if (failed(StdProfiler::readCSV(frequencies, info.archs))) {
      return funcOp->emitError()
             << "Failed to read profiling information from CSV";
    }

    if (failed(checkFuncInvariants(info)))
      return failure();
  }

  // Read the operations' timing models from disk
  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    return failure();

  // Place buffers in each function
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    if (failed(placeBuffers(funcToInfo[funcOp], timingDB)))
      return failure();
  }
  return success();
}

LogicalResult HandshakePlaceBuffersPass::checkFuncInvariants(FuncInfo &info) {
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
    if (!isa<handshake::SinkOp, handshake::MemoryOpInterface>(&op)) {
      if (!getLogicBB(&op).has_value()) {
        op.emitWarning() << "Operation does not belong to any block, MILP "
                            "behavior may be suboptimal or incorrect.";
      }
    }

    std::optional<unsigned> srcBB = opBlocks[&op];
    for (OpResult res : op.getResults()) {
      Operation *user = *res.getUsers().begin();
      std::optional<unsigned> dstBB = opBlocks[user];

      // All transitions between blocks must exist in the original CFG
      if (srcBB && dstBB && *srcBB != *dstBB &&
          !transitions[*srcBB].contains(*dstBB)) {
        auto endBB = *opBlocks.at(info.funcOp.getBodyBlock()->getTerminator());
        if (isa<ControlType>(res.getType()) && srcBB == ENTRY_BB &&
            dstBB == endBB) {
          /// NOTE: (lucas-rami) This is probably the start->end control
          /// channel which goes from the entry block to the exit block. This
          /// is fine in general so we let this pass without triggering a
          /// warning or error
          continue;
        }

        return op.emitError()
               << "Result " << res.getResultNumber() << " defined in block "
               << *srcBB << " is used in block " << *dstBB
               << ". This connection does not exist according to the CFG "
                  "graph. Solving the buffer placement MILP would yield an "
                  "incorrect placement.";
      }
    }
  }
  return success();
}

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

  os.flush();
}

LogicalResult
HandshakePlaceBuffersPass::placeBuffers(FuncInfo &info,
                                        TimingDatabase &timingDB) {
  // Use a wrapper around a logger to benefit from RAII
  std::error_code ec;
  BufferLogger bufLogger(info.funcOp, dumpLogs, ec);
  if (ec.value() != 0) {
    info.funcOp->emitError() << "Failed to create logger for function "
                             << info.funcOp.getName() << "\n"
                             << ec.message();
    return failure();
  }
  Logger *logger = dumpLogs ? *bufLogger : nullptr;

  // Get CFDFCs from the function unless the functions has no archs (i.e.,
  // it has a single block) in which case there are no CFDFCs
  SmallVector<CFDFC> cfdfcs;
  if (!info.archs.empty() && failed(getCFDFCs(info, logger, cfdfcs)))
    return failure();

  // All extracted CFDFCs must be optimized
  for (CFDFC &cf : cfdfcs)
    info.cfdfcs[&cf] = true;

  // Create a new map for the cfdfc extraction
  llvm::MapVector<size_t, std::vector<unsigned>> cfdfcResult;

  // Iterate through all extracted cfdfc
  for (auto [idx, cfAndOpt] : llvm::enumerate(info.cfdfcs)) {
    auto &[cf, _] = cfAndOpt;
    for (size_t i = 0, e = cf->cycle.size() - 1; i < e; ++i) {
      // Add the bb to the corresponding in the cfdfc result map
      cfdfcResult[idx].push_back(cf->cycle[i]);
    }
    cfdfcResult[idx].push_back(cf->cycle.back());
  }

  // Create and add the handshake.cfdfc attribute
  auto cfdfcMap =
      handshake::CFDFCToBBListAttr::get(info.funcOp.getContext(), cfdfcResult);
  setDialectAttr(info.funcOp, cfdfcMap);

  if (dumpLogs)
    logFuncInfo(info, *logger);

  // Solve the MILP to obtain a buffer placement
  BufferPlacement placement;
  if (failed(getBufferPlacement(info, timingDB, logger, placement)))
    return failure();

  instantiateBuffers(placement);
  return success();
}

LogicalResult HandshakePlaceBuffersPass::getCFDFCs(FuncInfo &info,
                                                   Logger *logger,
                                                   SmallVector<CFDFC> &cfdfcs) {
  SmallVector<ArchBB> archsCopy(info.archs);

  // Store all archs in a set. We use a pointer to each arch as the key type
  // to allow us to modify their frequencies during CFDFC extractions without
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
    if (logger)
      logPath = logger->getLogDir() + llvm::sys::path::get_separator().str() +
                "cfdfc" + std::to_string(cfdfcs.size());

    // Try to extract the next CFDFC
    int milpStat;
    if (failed(extractCFDFC(info.funcOp, archs, bbs, selectedArchs, numExecs,
                            logPath, &milpStat)))
      return info.funcOp->emitError()
             << "CFDFC extraction MILP failed with status " << milpStat << ". "
             << getGurobiOptStatusDesc(milpStat);
    if (numExecs == 0)
      break;

    // Create the CFDFC from the set of selected archs and BBs
    cfdfcs.emplace_back(info.funcOp, selectedArchs, numExecs);
  } while (!firstCFDFC);

  return success();
}

/// TODO
static void logCFDFCUnions(FuncInfo &info, Logger &log,
                           std::vector<CFDFCUnion> &disjointUnions) {
  mlir::raw_indented_ostream &os = *log;

  // Map each individual CFDFC to its iteration index
  std::map<CFDFC *, size_t> cfIndices;
  for (auto [idx, cfAndOpt] : llvm::enumerate(info.cfdfcs))
    cfIndices[cfAndOpt.first] = idx;

  os << "# ====================== #\n";
  os << "# Disjoint CFDFCs Unions #\n";
  os << "# ====================== #\n\n";

  // For each CFDFC union, display the blocks it encompasses as well as the
  // individual CFDFCs that fell into it
  for (auto [idx, cfUnion] : llvm::enumerate(disjointUnions)) {

    // Display the blocks making up the union
    auto blockIt = cfUnion.blocks.begin(), blockEnd = cfUnion.blocks.end();
    os << "CFDFC Union #" << idx << ": " << *blockIt;
    while (++blockIt != blockEnd)
      os << ", " << *blockIt;
    os << "\n";

    // Display the block cycle of each CFDFC in the union and some meta
    // information about the union
    os.indent();
    for (CFDFC *cf : cfUnion.cfdfcs) {
      auto cycleIt = cf->cycle.begin(), cycleEnd = cf->cycle.end();
      os << "- CFDFC #" << cfIndices[cf] << ": " << *cycleIt;
      while (++cycleIt != cycleEnd)
        os << " -> " << *cycleIt;
      os << "\n";
    }
    os << "- Number of block: " << cfUnion.blocks.size() << "\n";
    os << "- Number of units: " << cfUnion.units.size() << "\n";
    os << "- Number of channels: " << cfUnion.channels.size() << "\n";
    os << "- Number of backedges: " << cfUnion.backedges.size() << "\n";
    os.unindent();
    os << "\n";
  }
}

/// Wraps a call to solveMILP and conditionally passes the logger and MILP name
/// to the MILP's constructor as last arguments if the logger is not null.
template <typename MILP, typename... Args>
static inline LogicalResult
checkLoggerAndSolve(Logger *logger, StringRef milpName,
                    BufferPlacement &placement, Args &&...args) {
  if (logger) {
    return solveMILP<MILP>(placement, std::forward<Args>(args)..., *logger,
                           milpName);
  }
  return solveMILP<MILP>(placement, std::forward<Args>(args)...);
}

LogicalResult HandshakePlaceBuffersPass::getBufferPlacement(
    FuncInfo &info, TimingDatabase &timingDB, Logger *logger,
    BufferPlacement &placement) {
  // Create Gurobi environment
  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  if (timeout > 0)
    env.set(GRB_DoubleParam_TimeLimit, timeout);
  env.start();

  if (algorithm == FPGA20) {
    // Create and solve the MILP
    return checkLoggerAndSolve<fpga20::FPGA20Buffers>(
        logger, "placement", placement, env, info, timingDB, targetCP);
  }
  if (algorithm == FPL22) {
    // Create disjoint block unions of all CFDFCs
    SmallVector<CFDFC *, 8> cfdfcs;
    std::vector<CFDFCUnion> disjointUnions;
    llvm::transform(info.cfdfcs, std::back_inserter(cfdfcs),
                    [](auto cfAndOpt) { return cfAndOpt.first; });
    getDisjointBlockUnions(cfdfcs, disjointUnions);
    if (logger)
      logCFDFCUnions(info, *logger, disjointUnions);

    // Create and solve an MILP for each CFDFC union. Placement decisions get
    // accumulated over all MILPs. It's not possible to override a previous
    // placement decision because each CFDFC union is disjoint from the others
    for (auto [idx, cfUnion] : llvm::enumerate(disjointUnions)) {
      std::string milpName = "cfdfc_placement_" + std::to_string(idx);
      if (failed(checkLoggerAndSolve<fpl22::CFDFCUnionBuffers>(
              logger, milpName, placement, env, info, timingDB, targetCP,
              cfUnion)))
        return failure();
    }

    // Solve last MILP on channels/units that are not part of any CFDFC
    return checkLoggerAndSolve<fpl22::OutOfCycleBuffers>(
        logger, "out_of_cycle", placement, env, info, timingDB, targetCP);
  }

  llvm_unreachable("unknown algorithm");
}
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

LogicalResult HandshakePlaceBuffersPass::placeWithoutUsingMILP() {
  // The only strategy at this point is to place buffers on the output
  // channels of all merge-like operations. We still want to respect
  // channel-specific buffering constraints

  // Read the operations' timing models from disk
  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    return failure();

  for (handshake::FuncOp funcOp : getOperation().getOps<handshake::FuncOp>()) {
    // Map all channels in the function to their specific buffering
    // properties, adjusting for internal buffers present inside the units
    llvm::MapVector<Value, ChannelBufProps> channelProps;
    if (failed(mapChannelsToProperties(funcOp, timingDB, channelProps)))
      return failure();

    // Make sure that the data output channels of all merge-like operations
    // have at least one opaque and one transparent slot, unless a constraint
    // explicitly prevents us from putting a buffer there
    for (auto mergeLikeOp : funcOp.getOps<MergeLikeOpInterface>()) {
      ChannelBufProps &resProps = channelProps[mergeLikeOp->getResult(0)];
      if (resProps.maxTrans.value_or(1) >= 1) {
        resProps.minTrans = std::max(resProps.minTrans, 1U);
      } else {
        mergeLikeOp->emitWarning()
            << "Cannot place transparent buffer on merge-like operation's "
               "output due to channel-specific buffering constraints. This "
               "may "
               "yield an invalid buffering.";
      }
      if (resProps.maxOpaque.value_or(1) >= 1) {
        resProps.minOpaque = std::max(resProps.minOpaque, 1U);
      } else {
        mergeLikeOp->emitWarning()
            << "Cannot place opaque buffer on merge-like operation's "
               "output due to channel-specific buffering constraints. This "
               "may "
               "yield an invalid buffering.";
      }
    }

    // Place the minimal number of buffers (as specified by the buffering
    // constraints on each channel) for each channel, deducting internal unit
    // buffers at the same time
    BufferPlacement placement;
    for (auto &[channel, props] : channelProps) {
      PlacementResult result;
      result.numOneSlotDV = props.minOpaque;
      result.numOneSlotR = props.minTrans;
      placement[channel] = result;
    }
    instantiateBuffers(placement);
  }

  return success();
}

void HandshakePlaceBuffersPass::instantiateBuffers(BufferPlacement &placement) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  for (auto &[channel, placeRes] : placement) {
    Operation *opDst = *channel.getUsers().begin();
    builder.setInsertionPoint(opDst);

    Value bufferIn = channel;
    auto placeBuffer = [&](const TimingInfo &timing,
                           const StringRef &bufferType, unsigned numSlots) {
      if (numSlots == 0)
        return;

      auto bufOp = builder.create<handshake::BufferOp>(
          bufferIn.getLoc(), bufferIn, timing, numSlots, bufferType);
      inheritBB(opDst, bufOp);
      nameAnalysis.setName(bufOp);

      Value bufferRes = bufOp->getResult(0);
      opDst->replaceUsesOfWith(bufferIn, bufferRes);
      bufferIn = bufferRes;
    };

    /// Prefered order of each buffer type on a channel:
    /// {SHIFT_REG_BREAK_DV, ONE_SLOT_BREAK_DVR, ONE_SLOT_BREAK_DV,
    /// FIFO_BREAK_DV, FIFO_BREAK_NONE, ONE_SLOT_BREAK_R}
    placeBuffer(TimingInfo::break_dv(), BufferOp::SHIFT_REG_BREAK_DV,
                placeRes.numShiftRegDV);
    for (unsigned int i = 0; i < placeRes.numOneSlotDVR; i++) {
      placeBuffer(TimingInfo::break_dvr(), BufferOp::ONE_SLOT_BREAK_DVR, 1);
    }
    for (unsigned int i = 0; i < placeRes.numOneSlotDV; i++) {
      placeBuffer(TimingInfo::break_dv(), BufferOp::ONE_SLOT_BREAK_DV, 1);
    }
    placeBuffer(TimingInfo::break_dv(), BufferOp::FIFO_BREAK_DV,
                placeRes.numFifoDV);
    placeBuffer(TimingInfo::break_none(), BufferOp::FIFO_BREAK_NONE,
                placeRes.numFifoNone);
    for (unsigned int i = 0; i < placeRes.numOneSlotR; i++) {
      placeBuffer(TimingInfo::break_r(), BufferOp::ONE_SLOT_BREAK_R, 1);
    }
  }
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::buffer::createHandshakePlaceBuffers(
    StringRef algorithm, StringRef frequencies, StringRef timingModels,
    bool firstCFDFC, double targetCP, unsigned timeout, bool dumpLogs) {
  return std::make_unique<HandshakePlaceBuffersPass>(
      algorithm, frequencies, timingModels, firstCFDFC, targetCP, timeout,
      dumpLogs);
}