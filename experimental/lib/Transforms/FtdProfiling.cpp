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
#include "experimental/Support/CFGAnnotation.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <fstream>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

namespace {

/// Recursive function which allows to obtain all the paths from block `start`
/// to block `end` using a DFS.
static void dfsAllPaths(Operation *start, DenseSet<Operation *> &end,
                        std::vector<Operation *> &path,
                        DenseSet<Operation *> &visited,
                        std::vector<std::vector<Operation *>> &allPaths,
                        bool firstStep = false) {

  path.push_back(start);
  visited.insert(start);

  // If we are at the end of the path, then add it to the list of paths
  if (end.contains(start) && !firstStep) {
    allPaths.push_back(path);
  } else {
    // Else, for each successor which was not visited, run DFS again
    for (auto result : start->getResults()) {
      for (auto &use : result.getUses()) {
        Operation *user = use.getOwner();
        if (end.contains(user) || !visited.contains(user)) {
          dfsAllPaths(user, end, path, visited, allPaths);
        }
      }
    }
  }

  path.pop_back();
  visited.erase(start);
}

static std::vector<std::vector<Operation *>>
findAllPaths(Operation *start, DenseSet<Operation *> &end) {
  std::vector<std::vector<Operation *>> allPaths;
  std::vector<Operation *> path;
  DenseSet<Operation *> visited;
  dfsAllPaths(start, end, path, visited, allPaths, true);
  return allPaths;
}

using MuxesInLoop = std::map<std::string, DenseSet<Operation *>>;

/// Export GSA information
static MuxesInLoop exportGsaGatesInfo(handshake::FuncOp funcOp,
                                      NameAnalysis &namer) {

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

  for (auto &[nameLoop, listMuxes] : mil) {
    for (auto *mux : listMuxes) {
      llvm::dbgs() << "\n\nStarting point: ";
      mux->print(llvm::dbgs());
      llvm::dbgs() << "\n";
      auto result = findAllPaths(mux, listMuxes);
      for (auto &path : result) {
        llvm::dbgs() << "=================\n";
        for (auto *op : path) {
          op->print(llvm::dbgs());
          llvm::dbgs() << "\n";
        }
        llvm::dbgs() << "=================\n";
      }
    }
  }

  return mil;
}

struct FtdProfilingPass
    : public dynamatic::experimental::ftd::impl::FtdProfilingBase<
          FtdProfilingPass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    mlir::ModuleOp module = getOperation();
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    ConversionPatternRewriter rewriter(ctx);

    for (handshake::FuncOp funcOp : module.getOps<handshake::FuncOp>()) {
      if (failed(cfg::restoreCfStructure(funcOp, rewriter)))
        signalPassFailure();

      exportGsaGatesInfo(funcOp, namer);

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
