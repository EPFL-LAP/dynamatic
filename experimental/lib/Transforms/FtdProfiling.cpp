//===- FtdProfiling.h - Runs a profiling algorithm  -----*----- C++ -*-----===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/FtdProfiling.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/TimingModels.h"
#include "experimental/Support/CFGAnnotation.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <fstream>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

namespace {

using MuxesInLoop = std::map<std::string, DenseSet<Operation *>>;
using path_t = std::vector<Operation *>;
using v_path_t = std::vector<path_t>;
using s_block = DenseSet<Block *>;

constexpr double ITERATIONS_PER_LOOP = 10;
constexpr double CLOCK_PERIOD = 4;

static void exportGsaGatesInfo(handshake::FuncOp funcOp, NameAnalysis &namer) {

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  std::ofstream ofs;
  MuxesInLoop mil;

  ofs.open("ftdscripting/gsaGatesInfo.txt", std::ofstream::out);
  std::string loopDescription = "";
  llvm::raw_string_ostream loopDescriptionStream(loopDescription);

  for (auto phi : funcOp.getBody().getOps<handshake::MuxOp>()) {
    ofs << namer.getName(phi).str();
    if (llvm::isa<handshake::MergeOp>(phi->getOperand(0).getDefiningOp()))
      ofs << " (MU)\n";
    else
      ofs << " (GAMMA)\n";

    if (!loopInfo.getLoopFor(phi->getBlock())) {
      ofs << "Not inside any loop\n";
    } else {
      loopInfo.getLoopFor(phi->getBlock())
          ->print(loopDescriptionStream, false, false, 0);
      auto loopStr = loopDescriptionStream.str();
      if (!mil.count(loopStr))
        mil[loopStr] = DenseSet<Operation *>();
      mil[loopStr].insert(phi);
    }
    ofs << loopDescription << "\n";
    loopDescription = "";
  }

  ofs.close();
}

static void dfsAllPaths(Operation *start, DenseSet<Operation *> &end,
                        path_t &path, DenseSet<Operation *> &visited,
                        v_path_t &allPaths, bool firstStep,
                        const s_block *blocksInLoop) {

  path.push_back(start);
  visited.insert(start);

  // If we are at the end of the path, then add it to the list of paths
  if (end.contains(start) && !firstStep) {
    allPaths.push_back(path);
  } else {
    // Else, for each successor which was not visited, run DFS again
    for (auto result : start->getResults()) {
      for (auto *user : result.getUsers()) {
        if (!blocksInLoop->contains(user->getBlock())) {
          continue;
        }
        if (end.contains(user) || !visited.contains(user)) {
          dfsAllPaths(user, end, path, visited, allPaths, false, blocksInLoop);
        }
      }
    }
  }

  path.pop_back();
  visited.erase(start);
}

static void dfsAllPaths(OpOperand &start, Operation *last, path_t &path,
                        DenseSet<Operation *> &visited, v_path_t &allPaths,
                        const ftd::BlockIndexing &bi) {

  Operation *startOp = start.getOwner();

  if (llvm::isa_and_nonnull<handshake::ControlMergeOp>(startOp) ||
      llvm::isa_and_nonnull<handshake::MemoryControllerOp>(startOp) ||
      llvm::isa_and_nonnull<handshake::LSQOp>(startOp)) {
    return;
  }

  if (startOp) {
    path.push_back(startOp);
    visited.insert(startOp);
  }

  // If we are at the end of the path, then add it to the list of paths
  if (startOp && last == startOp) {
    allPaths.push_back(path);
  } else {
    // Else, for each successor which was not visited, run DFS again
    for (auto result : startOp->getResults()) {
      for (auto &use : result.getUses()) {
        Operation *user = use.getOwner();
        if (bi.isGreater(start.getOwner()->getBlock(), user->getBlock()))
          continue;
        if (!visited.contains(user)) {
          dfsAllPaths(use, last, path, visited, allPaths, bi);
        }
      }
    }
  }

  if (startOp) {
    path.pop_back();
    visited.erase(startOp);
  }
}

static void findAllPaths(Operation *start, DenseSet<Operation *> &end,
                         v_path_t &allPaths, const s_block *blocksInLoop) {
  path_t path;
  DenseSet<Operation *> visited;
  dfsAllPaths(start, end, path, visited, allPaths, true, blocksInLoop);
}

static void findAllPaths(Value start, Operation *last, v_path_t &allPaths,
                         const ftd::BlockIndexing &bi) {
  path_t path;
  DenseSet<Operation *> visited;
  for (auto &use : start.getUses())
    dfsAllPaths(use, last, path, visited, allPaths, bi);
}

static DenseSet<Block *> getBlocksInLoop(CFGLoop *loop) {
  auto blocksInLoop = loop->getBlocksVector();
  s_block set;
  for (auto *x : blocksInLoop)
    set.insert(x);
  return set;
}

static double getOperationDelay(Operation *op, const TimingDatabase &timingDB) {

  double delay = 0, latency = 0;

  if (failed(timingDB.getLatency(op, SignalType::DATA, latency)) ||
      failed(timingDB.getTotalDelay(op, SignalType::DATA, delay)))
    return 0;

  return latency == 0 ? delay : latency * CLOCK_PERIOD;
}

using info_per_loop_t = DenseMap<CFGLoop *, std::pair<double, path_t>>;

static double getPathDelay(const path_t &path,
                           const info_per_loop_t &worstPathLoop,
                           const CFGLoop *loop,
                           const TimingDatabase &timingDB) {

  // Current timing
  double timing = 0;
  // Nested loop that we are currently traversing
  CFGLoop *inLoop = nullptr;
  // Blocks of the loop we are traversing
  s_block blocksInLoop;

  llvm::dbgs() << "\n----------------------\n";

  // For each operation in the loop
  for (Operation *op : path) {

    // If we were into a loop, check if we are now out of it
    if (inLoop) {
      if (!blocksInLoop.contains(op->getBlock()) &&
          !llvm::isa_and_nonnull<handshake::MemoryControllerOp>(op)) {
        llvm::dbgs() << "OUT LOOP: ";
        inLoop->print(llvm::dbgs(), false, false, 0);
        llvm::dbgs() << "\n";
        timing += worstPathLoop.at(inLoop).first * ITERATIONS_PER_LOOP;
        inLoop = nullptr;
      }
    }

    // If we are not in a loop and we are traversing another loop header, then
    // mark the current loop
    if (!inLoop) {
      if (!llvm::isa_and_nonnull<handshake::ConditionalBranchOp>(op)) {
        for (auto &[otherLoop, info] : worstPathLoop) {
          if (otherLoop != loop && otherLoop->getHeader() == op->getBlock()) {
            llvm::dbgs() << "IN  LOOP: ";
            otherLoop->print(llvm::dbgs(), false, false, 0);
            llvm::dbgs() << "\n";

            inLoop = otherLoop;
            blocksInLoop = getBlocksInLoop(inLoop);
          }
        }
      }
    }

    // Print current operation
    op->print(llvm::dbgs());
    llvm::dbgs() << "\n";

    // If not in loop, get the delay of the current operation.
    if (!inLoop) {
      double delay = getOperationDelay(op, timingDB);
      timing += delay;
      llvm::dbgs() << " --- > " << delay << "\n";
    }
  }

  // Delay of the path
  llvm::dbgs() << "Total delay: " << timing << "\n";
  llvm::dbgs() << "\n----------------------\n";

  return timing;
}

static void analyzeWCET(handshake::FuncOp funcOp, NameAnalysis &namer,
                        TimingDatabase &timingDB) {
  llvm::dbgs() << "Starting WCET analysis...\n";

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  ftd::BlockIndexing bi(region);

  auto loops = loopInfo.getLoopsInPreorder();
  auto loopsR = llvm::reverse(loops);

  info_per_loop_t worstPathLoop;

  for (auto *loop : loopsR) {

    worstPathLoop.insert({loop, {0, {}}});

    // Get all the muxes in the loop header and build a set out of it
    auto muxesInHeader = loop->getHeader()->getOps<handshake::MuxOp>();
    DenseSet<Operation *> muxesInHeaderSet;
    for (auto x : muxesInHeader)
      muxesInHeaderSet.insert(x);

    // Get all the blocks in the loop and build a set out of it
    auto blocksInLoop = getBlocksInLoop(loop);

    // For each mux in the loop header, find all the paths which end in a mux
    // of the same loop and have operations that are all in the blocks of the
    // loop.
    v_path_t allpaths;
    for (Operation *mux : muxesInHeader) {
      v_path_t result;
      findAllPaths(mux, muxesInHeaderSet, result, &blocksInLoop);
      for (auto &x : result)
        allpaths.push_back(x);
    }

    // Consider each path and find the worst in terms of timing
    for (auto &path : allpaths) {
      double timing = getPathDelay(path, worstPathLoop, loop, timingDB);
      // Update the timing
      if (timing > worstPathLoop.at(loop).first) {
        worstPathLoop[loop] = {timing, path};
      }
    }
  }

  double worstTiming = 0;
  path_t worstPath;
  Operation *endOp = *funcOp.getOps<handshake::EndOp>().begin();
  for (auto &argument : funcOp.getArguments()) {
    v_path_t allpaths;
    findAllPaths(argument, endOp, allpaths, bi);

    for (auto &path : allpaths) {
      double timing = getPathDelay(path, worstPathLoop, nullptr, timingDB);
      worstPath = (timing > worstTiming) ? path : worstPath;
      worstTiming = (timing > worstTiming) ? timing : worstTiming;
    }
  }

  llvm::dbgs() << "\n----------------------\n";

  for (auto &[loop, info] : worstPathLoop) {
    loop->print(llvm::dbgs());
    llvm::dbgs() << "\n";
    llvm::dbgs() << info.first << "\n";
    for (auto *op : info.second) {
      op->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "Total\n";
  llvm::dbgs() << worstTiming << "\n";
  for (auto *op : worstPath) {
    op->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }
  llvm::dbgs() << "\n";

  llvm::dbgs() << "Done with WCET analysis...\n";
}

struct FtdProfilingPass
    : public dynamatic::experimental::ftd::impl::FtdProfilingBase<
          FtdProfilingPass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    mlir::ModuleOp module = getOperation();
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    ConversionPatternRewriter rewriter(ctx);

    std::string timingModels = "./data/components.json";
    TimingDatabase timingDB(&getContext());
    if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
      signalPassFailure();

    for (handshake::FuncOp funcOp : module.getOps<handshake::FuncOp>()) {
      if (failed(cfg::restoreCfStructure(funcOp, rewriter)))
        signalPassFailure();

      exportGsaGatesInfo(funcOp, namer);
      analyzeWCET(funcOp, namer, timingDB);

      if (failed(cfg::flattenFunction(funcOp)))
        signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::ftd::createFtdProfiling() {
  return std::make_unique<FtdProfilingPass>();
}
