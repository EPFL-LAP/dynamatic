//===- Crush.cpp - Credit-Based Resource Sharing ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/ResourceSharing/Crush.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/MILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "dynamatic/Transforms/BufferPlacement/FPL22Buffers.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "experimental/Transforms/ResourceSharing/SharingSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using std::all_of;
using std::none_of;

using namespace llvm;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::sharing;

using namespace dynamatic::buffer;

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
/// Algorithms that do require solving an MILP.
static constexpr llvm::StringLiteral FPGA20("fpga20"),
    FPGA20_LEGACY("fpga20-legacy"), FPL22("fpl22");
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

// extracted data from buffer placement
struct FuncPerfInfo {

  // For each CFC, the achieved throughput
  std::map<size_t, double> cfThroughput;

  // A list of performance critical CFCs.
  std::set<size_t> critCfcs;

  // The set of units of each CFC.
  std::map<size_t, std::set<Operation *>> cfUnits;

  // The set of channels of each CFC.
  std::map<size_t, std::set<Channel *>> cfChannels;

  // The set of strongly connected components of each CFC.
  // For instance, (muli1, 1) and (muli2, 1) means two units, muli1 and muli2
  // are in the same SCC (with id 1).
  std::map<size_t, std::map<Operation *, size_t>> cfSccs;
};

// SharingInfo: for each funcOp, its extracted FuncPerfInfo.
using SharingInfo = std::map<handshake::FuncOp *, FuncPerfInfo>;

// SharingGroups: a list of operations that share the same unit.
using SharingGroups = std::map<size_t, std::set<Operation *>>;

namespace dynamatic {
namespace buffer {
namespace fpga20 {

// An wrapper class for extracting CFDFC performance from FPGA20 buffers.
class FPGA20BuffersWrapper : public FPGA20Buffers {
public:
  // constructor
  FPGA20BuffersWrapper(SharingInfo &info, GRBEnv &env, FuncInfo &funcInfo,
                       const TimingDatabase &timingDB, double targetPeriod,
                       bool legacyPlacement, Logger &logger, StringRef milpName)
      : FPGA20Buffers(env, funcInfo, timingDB, targetPeriod, legacyPlacement,
                      logger, milpName),
        info(info){};
  FPGA20BuffersWrapper(SharingInfo &info, GRBEnv &env, FuncInfo &funcInfo,
                       const TimingDatabase &timingDB, double targetPeriod,
                       bool legacyPlacement)
      : FPGA20Buffers(env, funcInfo, timingDB, targetPeriod, legacyPlacement),
        info(info){};
  SharingInfo &info;
  void extractResult(BufferPlacement &placement) override {
    // Run the FPGA20Buffers's extractResult as it is
    FPGA20Buffers::extractResult(placement);

    // Map each individual CFDFC to its iteration index
    std::map<CFDFC *, size_t> cfIndices;
    for (auto [idx, cfAndOpt] : llvm::enumerate(funcInfo.cfdfcs))
      cfIndices[cfAndOpt.first] = idx;

    // Extract result: save global CFDFC throuhgputs into info
    for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {
      auto [cf, cfVars] = cfdfcWithVars;
      double throughput = cfVars.throughput.get(GRB_DoubleAttr_X);

      info[&funcInfo.funcOp].cfThroughput[cfIndices[cf]] = throughput;
      // info.cfcThroughput[cf] = throughput;
      info[&funcInfo.funcOp].cfUnits[cfIndices[cf]] = {};

      // Track the units of the CFC
      for (auto *op : cf->units)
        info[&funcInfo.funcOp].cfUnits[cfIndices[cf]].insert(op);

      // Track the channels of the CFC
      for (auto val : cf->channels) {
        Channel *ch = new Channel(val);

        info[&funcInfo.funcOp].cfChannels[cfIndices[cf]].insert(ch);
      }
    }

    SmallVector<CFDFC *, 8> cfdfcs;
    std::vector<CFDFCUnion> disjointUnions;
    llvm::transform(funcInfo.cfdfcs, std::back_inserter(cfdfcs),
                    [](auto cfAndOpt) { return cfAndOpt.first; });
    getDisjointBlockUnions(cfdfcs, disjointUnions);

    // Instrumentation: for each CFDFC Union, mark the most-frequently-executed
    // CFC as performance critical

    for (auto &cfUnion : disjointUnions) {

      auto *critCf =
          std::max_element(cfUnion.cfdfcs.begin(), cfUnion.cfdfcs.end(),
                           [](CFDFC const *l, CFDFC const *r) {
                             return l->numExecs < r->numExecs;
                           });
      if (!critCf) {
        funcInfo.funcOp->emitError()
            << "Failed running determining performance critical CFC";
        return;
      }

      info[&funcInfo.funcOp].critCfcs.emplace(cfIndices[*critCf]);
      llvm::errs() << "Frequency of crit cfc: " << (*critCf)->numExecs << "\n";
    }
  }
};

} // namespace fpga20
} // namespace buffer
} // namespace dynamatic

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

namespace {
using fpga20::FPGA20BuffersWrapper;

// An wrapper class that applies buffer p
// extracts the report.
struct HandshakePlaceBuffersPassWrapper : public HandshakePlaceBuffersPass {
  HandshakePlaceBuffersPassWrapper(SharingInfo &info, StringRef algorithm,
                                   StringRef frequencies,
                                   StringRef timingModels, bool firstCFDFC,
                                   double targetCP, unsigned timeout,
                                   bool dumpLogs)
      : HandshakePlaceBuffersPass(algorithm, frequencies, timingModels,
                                  firstCFDFC, targetCP, timeout, dumpLogs),
        info(info){};
  SharingInfo &info;

  LogicalResult getBufferPlacement(FuncInfo &funcInfo, TimingDatabase &timingDB,
                                   Logger *logger,
                                   BufferPlacement &placement) override {

    // Create Gurobi environment
    GRBEnv env = GRBEnv(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    if (timeout > 0)
      env.set(GRB_DoubleParam_TimeLimit, timeout);
    env.start();

    if (algorithm == FPGA20 || algorithm == FPGA20_LEGACY) {
      // Create and solve the MILP
      return checkLoggerAndSolve<FPGA20BuffersWrapper>(
          logger, "placement", placement, info, env, funcInfo, timingDB,
          targetCP, algorithm != FPGA20);
    }
    if (algorithm == FPL22) {
      // Create disjoint block unions of all CFDFCs
      SmallVector<CFDFC *, 8> cfdfcs;
      std::vector<CFDFCUnion> disjointUnions;
      llvm::transform(funcInfo.cfdfcs, std::back_inserter(cfdfcs),
                      [](auto cfAndOpt) { return cfAndOpt.first; });
      getDisjointBlockUnions(cfdfcs, disjointUnions);

      // Create and solve an MILP for each CFDFC union. Placement decisions get
      // accumulated over all MILPs. It's not possible to override a previous
      // placement decision because each CFDFC union is disjoint from the others
      for (auto [idx, cfUnion] : llvm::enumerate(disjointUnions)) {
        std::string milpName = "cfdfc_placement_" + std::to_string(idx);
        if (failed(checkLoggerAndSolve<fpl22::CFDFCUnionBuffers>(
                logger, milpName, placement, env, funcInfo, timingDB, targetCP,
                cfUnion)))
          return failure();
      }

      // Solve last MILP on channels/units that are not part of any CFDFC
      return checkLoggerAndSolve<fpl22::OutOfCycleBuffers>(
          logger, "out_of_cycle", placement, env, funcInfo, timingDB, targetCP);
    }

    llvm_unreachable("unknown algorithm");
  }
};

struct CreditBasedSharingPass
    : public dynamatic::experimental::sharing::impl::CreditBasedSharingBase<
          CreditBasedSharingPass> {

  CreditBasedSharingPass(StringRef algorithm, StringRef frequencies,
                         StringRef timingModels, bool firstCFDFC,
                         double targetCP, unsigned timeout, bool dumpLogs) {
    this->algorithm = algorithm.str();
    this->frequencies = frequencies.str();
    this->timingModels = timingModels.str();
    this->firstCFDFC = firstCFDFC;
    this->targetCP = targetCP;
    this->timeout = timeout;
    this->dumpLogs = dumpLogs;
  }

  void runDynamaticPass() override;

  SmallVector<mlir::Operation *> getSharingTargets(handshake::FuncOp funcOp) {
    SmallVector<Operation *> sharingTargets;

    for (auto &op : funcOp.getOps()) {
      if (isa<SHARING_TARGETS>(op)) {
        sharingTargets.emplace_back(&op);
      }
    }
    return sharingTargets;
  }

  // Call the wrapper class HandshakePlaceBuffersPassWrapper, which again wraps
  // FPGA20BuffersWrapper
  LogicalResult runBufferPlacementPass(ModuleOp &modOp, SharingInfo &data) {
    TimingDatabase timingDB(&getContext());
    if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
      return failure();

    // running buffer placement on current module
    mlir::PassManager pm(&getContext());
    pm.addPass(std::make_unique<HandshakePlaceBuffersPassWrapper>(
        data, algorithm, frequencies, timingModels, firstCFDFC, targetCP,
        timeout, dumpLogs));
    if (failed(pm.run(modOp))) {
      return failure();
    }
    return success();
  }
};
} // namespace

// for two sharing groups, check if the following criteria hold.
bool checkGroupMergable(const std::set<Operation *> &g1,
                        const std::set<Operation *> &g2,
                        FuncPerfInfo funcPerfInfo) {

  std::set<Operation *> gMerged;
  gMerged.insert(g1.begin(), g1.end());
  gMerged.insert(g2.begin(), g2.end());

  OperationName opName = (*(gMerged.begin()))->getName();

  // 1. The merged group must have operations of the same type.
  for (auto *op : gMerged) {
    if (op->getName() != opName)
      return false;
  }

  // 2. For each CFC, the sum of occupancy must be smaller than the capacity
  // (i.e., units in CFC must no greater than the II).
  // This is equivalent to checking that throughput * n_ops <= 1;

  // 3. For each CFC, there must be no two operations have the same SCC ID (this
  // is simplified).
  for (auto cf : funcPerfInfo.critCfcs) {
    // for each cf, numOps contains the number of operations
    // that are in the merged group and also in cf
    unsigned numOps = 0;

    // listOfSccIds: a list of SCC IDs that the group has
    // it is used to check if there are any duplicates (i.e.,
    // two operation that are in the same SCC cannot be in the
    // same sharing group).
    std::vector<size_t> listOfSccIds;
    for (auto *op : (funcPerfInfo.cfUnits)[cf]) {
      // In the op is in (SCC union MergedGroup):
      if (gMerged.find(op) != gMerged.end()) {
        // increase number of Ops
        numOps++;
        // Push back the SCC ID of each op inside the group and
        // also SCC;
        listOfSccIds.push_back((funcPerfInfo.cfSccs)[cf][op]);
      }
    }
    // Check if there are any duplicates:
    std::set<size_t> setOfSccIds(listOfSccIds.begin(), listOfSccIds.end());
    // Check if numOps * cfcThroughput <= 1 and no duplicate SCC
    // IDs.
    if (!(numOps * (funcPerfInfo.cfThroughput)[cf] <= 1) ||
        (listOfSccIds.size() == setOfSccIds.size())) {
      return false;
    }
  }
  // return (all_of(funcPerfInfo.critCfcs.begin(), funcPerfInfo.critCfcs.end(),
  //                [funcPerfInfo, gMerged](auto cf) {
  //                  // for each cf, numOps contains the number of operations
  //                  // that are in the merged group and also in cf
  //                  unsigned numOps = 0;

  //                  // listOfSccIds: a list of SCC IDs that the group has
  //                  // it is used to check if there are any duplicates (i.e.,
  //                  // two operation that are in the same SCC cannot be in the
  //                  // same sharing group).
  //                  std::vector<size_t> listOfSccIds;
  //                  for (auto op : (funcPerfInfo.cfUnits)[cf]) {
  //                    // In the op is in (SCC union MergedGroup):
  //                    if (gMerged.find(op) != gMerged.end()) {
  //                      // increase number of Ops
  //                      numOps++;
  //                      // Push back the SCC ID of each op inside the group
  //                      and
  //                      // also SCC;
  //                      listOfSccIds.push_back((funcPerfInfo.cfSccs)[cf][op]);
  //                    }
  //                  }
  //                  // Check if there are any duplicates:
  //                  std::set<size_t> setOfSccIds(listOfSccIds.begin(),
  //                                               listOfSccIds.end());
  //                  // Check if numOps * cfcThroughput <= 1 and no duplicate
  //                  SCC
  //                  // IDs.
  //                  return (numOps * (funcPerfInfo.cfThroughput)[cf] <= 1) ||
  //                         (listOfSccIds.size() == setOfSccIds.size());
  //                }));
  return true;
}

void CreditBasedSharingPass::runDynamaticPass() {
  llvm::errs() << "***** Resource Sharing *****\n";

  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  // Buffer placement requires that all values are used exactly once
  ModuleOp modOp = getOperation();
  if (failed(verifyIRMaterialized(modOp))) {
    modOp->emitError() << ERR_NON_MATERIALIZED_MOD;
    return;
  }

  SharingInfo data;

  // Run buffer placement pass and fill data with performance analysis
  // information
  if (failed(runBufferPlacementPass(modOp, data))) {
    modOp->emitError() << "Failed running buffer placement";
    return;
  }

  // for (auto [funcOp, info] : data) {
  //   for (auto [idx, cf] : info.cfUnits) {
  //     for (auto *op : cf)
  //       llvm::errs() << namer.getName(op) << " ";
  //   }
  //   llvm::errs() << "\n";
  // }

  // for (auto [funcOp, info] : data) {
  //   for (auto [idx, cf] : info.cfChannels) {
  //     for (auto *ch : cf)
  //       llvm::errs() << namer.getName(ch->producer) << "->"
  //                    << namer.getName(ch->consumer) << " ";
  //   }
  //   llvm::errs() << "\n";
  // }

  for (auto [funcOp, info] : data) {
    for (auto [idx, throughput] : info.cfThroughput) {
      llvm::errs() << "Throughput of CFDFC #" << idx << " is " << throughput
                   << "\n";
    }
  }

  for (auto &[funcOp, info] : data) {
    // Check the sharing targets
    SmallVector<Operation *> sharingTargets = getSharingTargets(*funcOp);

    llvm::errs() << "Sharing Targets:";
    for (Operation *op : sharingTargets) {
      llvm::errs() << namer.getName(op) << " ";
    }
    llvm::errs() << "\n";

    // Initialize the sharing groups:
    SharingGroups sharingGroups;
    for (auto [id, op] : llvm::enumerate(sharingTargets)) {
      std::set<Operation *> g = {op};
      sharingGroups.emplace(id, g);
    }

    // Determine SCCs
    for (auto critCfc : info.critCfcs) {
      std::map<Operation *, size_t> sccMap =
          getSccsInCfc(info.cfUnits[critCfc], info.cfChannels[critCfc]);
      info.cfSccs.emplace(critCfc, sccMap);
      llvm::errs() << "SCC ID of CFC #" << critCfc << "\n";
      for (auto [op, sccId] : info.cfSccs[critCfc]) {
        llvm::errs() << namer.getName(op) << " " << sccId << "\n";
      }
    }

    // Merge groups
    bool modified = false;
    while (modified) {
      for (auto &[id, group] : sharingGroups) {
        // check if two groups are mergable.
      }
    }
  }
}

namespace dynamatic {
namespace experimental {
namespace sharing {

/// Returns a unique pointer to an operation pass that matches MLIR modules.
std::unique_ptr<dynamatic::DynamaticPass>
createCreditBasedSharing(StringRef algorithm, StringRef frequencies,
                         StringRef timingModels, bool firstCFDFC,
                         double targetCP, unsigned timeout, bool dumpLogs) {
  return std::make_unique<CreditBasedSharingPass>(algorithm, frequencies,
                                                  timingModels, firstCFDFC,
                                                  targetCP, timeout, dumpLogs);
}

} // namespace sharing
} // namespace experimental
} // namespace dynamatic
