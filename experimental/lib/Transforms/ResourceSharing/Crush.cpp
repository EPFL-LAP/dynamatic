//===- FCCM22Sharing.cpp - Resource Sharing ---------*- C++ -*-===//
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
struct PerfOptInfo {

  // // The list of units of each CFC
  // llvm::MapVector<CFDFC *, llvm::SetVector<Operation *>> cfUnits;

  // // The list of units of each CFC
  // llvm::MapVector<CFDFC *, llvm::SetVector<Value *>> cfEdges;

  // For each CFC, the achieved throughput
  std::map<size_t, double> cfThroughput;

  // A list of performance critical CFCs.
  std::set<size_t> critCfcs;

  // The list of units of each CFC
  std::map<size_t, llvm::SetVector<Operation *>> cfUnits;
};

namespace dynamatic {
namespace buffer {
namespace fpga20 {

// An wrapper class for extracting CFDFC performance from FPGA20 buffers.
class FPGA20BuffersWrapper : public FPGA20Buffers {
public:
  // constructor
  FPGA20BuffersWrapper(PerfOptInfo &info, GRBEnv &env, FuncInfo &funcInfo,
                       const TimingDatabase &timingDB, double targetPeriod,
                       bool legacyPlacement, Logger &logger, StringRef milpName)
      : FPGA20Buffers(env, funcInfo, timingDB, targetPeriod, legacyPlacement,
                      logger, milpName),
        info(info){};
  FPGA20BuffersWrapper(PerfOptInfo &info, GRBEnv &env, FuncInfo &funcInfo,
                       const TimingDatabase &timingDB, double targetPeriod,
                       bool legacyPlacement)
      : FPGA20Buffers(env, funcInfo, timingDB, targetPeriod, legacyPlacement),
        info(info){};
  PerfOptInfo &info;
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

      info.cfThroughput[cfIndices[cf]] = throughput;
      // info.cfcThroughput[cf] = throughput;
      info.cfUnits[cfIndices[cf]] = {};

      // track the operations of the CFC
      for (auto *op : cf->units) {
        info.cfUnits[cfIndices[cf]].insert(op);
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

      info.critCfcs.emplace(cfIndices[*critCf]);
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
  HandshakePlaceBuffersPassWrapper(PerfOptInfo &info, StringRef algorithm,
                                   StringRef frequencies,
                                   StringRef timingModels, bool firstCFDFC,
                                   double targetCP, unsigned timeout,
                                   bool dumpLogs)
      : HandshakePlaceBuffersPass(algorithm, frequencies, timingModels,
                                  firstCFDFC, targetCP, timeout, dumpLogs),
        info(info){};
  PerfOptInfo &info;

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
  LogicalResult runBufferPlacementPass(ModuleOp &modOp, PerfOptInfo &data) {
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

  PerfOptInfo data;

  // Run buffer placement pass and fill data with performance analysis
  // information
  if (failed(runBufferPlacementPass(modOp, data))) {
    modOp->emitError() << "Failed running buffer placement";
    return;
  }

  for (auto cf : data.critCfcs) {

    for (auto *op : data.cfUnits[cf])
      llvm::errs() << namer.getName(op) << " ";

    llvm::errs() << "\n";
  }

  // TODO: check the extracted data
  for (auto [idx, throughput] : data.cfThroughput) {
    llvm::errs() << "Throughput of CFDFC #" << idx << " is " << throughput
                 << "\n";
  }

  for (handshake::FuncOp funcOp : getOperation().getOps<handshake::FuncOp>()) {
    SmallVector<Operation *> sharingTargets = getSharingTargets(funcOp);

    llvm::errs() << "Sharing Targets:";
    for (Operation *op : sharingTargets) {
      llvm::errs() << namer.getName(op) << " ";
    }
    llvm::errs() << "\n";

    // check the sharing targets
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
