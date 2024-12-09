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
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <fstream>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

namespace {

using path_t = std::vector<Operation *>;
using v_path_t = std::vector<path_t>;
using pathPerMux_t =
    std::map<Operation *, std::tuple<v_path_t, v_path_t, v_path_t>>;

/// Recursive function which allows to obtain all the paths from block
/// `start` to block `end` using a DFS.
static void dfsAllPaths(Operation *start, DenseSet<Operation *> &end,
                        path_t &path, DenseSet<Operation *> &visited,
                        pathPerMux_t &allPaths, bool firstStep = false,
                        unsigned operand = 0) {

  path.push_back(start);
  visited.insert(start);

  // If we are at the end of the path, then add it to the list of paths
  if (end.contains(start) && !firstStep) {
    if (operand == 0)
      std::get<0>(allPaths[start]).push_back(path);
    if (operand == 1)
      std::get<1>(allPaths[start]).push_back(path);
    if (operand == 2)
      std::get<2>(allPaths[start]).push_back(path);
  } else {
    // Else, for each successor which was not visited, run DFS again
    for (auto result : start->getResults()) {
      for (auto &use : result.getUses()) {
        Operation *user = use.getOwner();
        if (end.contains(user) || !visited.contains(user)) {
          dfsAllPaths(user, end, path, visited, allPaths, false,
                      use.getOperandNumber());
        }
      }
    }
  }

  path.pop_back();
  visited.erase(start);
}

static void findAllPaths(Operation *start, DenseSet<Operation *> &end,
                         pathPerMux_t &allPaths) {
  path_t path;
  DenseSet<Operation *> visited;
  dfsAllPaths(start, end, path, visited, allPaths, true);
}

using MuxesInLoop = std::map<std::string, DenseSet<Operation *>>;

/// Export GSA information
static void exportGsaGatesInfo(handshake::FuncOp funcOp, NameAnalysis &namer,
                               TimingDatabase &timingDB) {

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  std::ofstream ofs;
  MuxesInLoop mil;

  // We print to this file information about the GSA gates and the innermost
  // loops, if any, containing each.
  ofs.open("ftdscripting/gsaGatesInfo.txt", std::ofstream::out);
  std::string loopDescription;
  llvm::raw_string_ostream loopDescriptionStream(loopDescription);

  auto muxes = funcOp.getBody().getOps<handshake::MuxOp>();
  for (auto phi : muxes) {
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
  double delay;

  /// For each multiplexer, store all the paths which start start from any mux
  /// at the same loop level and end in one of their operands. The paths are
  /// collected according to the related operand.
  pathPerMux_t pathsPerMux;
  for (auto &[nameLoop, listMuxes] : mil) {
    for (auto *mux : listMuxes)
      findAllPaths(mux, listMuxes, pathsPerMux);
  }

  for (auto &[source, pathsOperands] : pathsPerMux) {

    llvm::dbgs() << "===========================\n";

    double maxDelay0 = 0, maxDelay1 = 0, maxDelay2 = 0, minDelay0 = 99999,
           minDelay1 = 99999, minDelay2 = 99999;

    for (auto &path : std::get<0>(pathsOperands)) {
      double delaySum = 0;
      for (auto *op : path) {
        if (failed(timingDB.getTotalDelay(op, SignalType::DATA, delay)))
          delay = 0;
        delaySum += delay;
      }
      maxDelay0 = maxDelay0 < delaySum ? delaySum : maxDelay0,
      minDelay0 = minDelay0 > delaySum ? delaySum : minDelay0;
    }

    for (auto &path : std::get<1>(pathsOperands)) {
      double delaySum = 0;
      for (auto *op : path) {
        if (failed(timingDB.getTotalDelay(op, SignalType::DATA, delay)))
          delay = 0;
        delaySum += delay;
      }
      maxDelay1 = maxDelay1 < delaySum ? delaySum : maxDelay1,
      minDelay1 = minDelay1 > delaySum ? delaySum : minDelay1;
    }

    for (auto &path : std::get<2>(pathsOperands)) {
      double delaySum = 0;
      for (auto *op : path) {
        if (failed(timingDB.getTotalDelay(op, SignalType::DATA, delay)))
          delay = 0;
        delaySum += delay;
      }
      maxDelay2 = maxDelay2 < delaySum ? delaySum : maxDelay2,
      minDelay2 = minDelay2 > delaySum ? delaySum : minDelay2;
    }

    llvm::dbgs() << "Max delay input 0: " << maxDelay0 << "\n";
    llvm::dbgs() << "Max delay input 1: " << maxDelay1 << "\n";
    llvm::dbgs() << "Max delay input 2: " << maxDelay2 << "\n";
    llvm::dbgs() << "Min delay input 0: " << minDelay0 << "\n";
    llvm::dbgs() << "Min delay input 1: " << minDelay1 << "\n";
    llvm::dbgs() << "Min delay input 2: " << minDelay2 << "\n";
  }
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

      exportGsaGatesInfo(funcOp, namer, timingDB);

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
