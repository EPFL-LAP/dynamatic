//===- FCCM22Sharing.cpp - Resource Sharing ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/ResourceSharing/Crush.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/MILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "dynamatic/Transforms/BufferPlacement/FPL22Buffers.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <map>
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

  // For each CFC, the achieved throughput
  llvm::MapVector<CFDFC *, double> cfcThroughput;
};

namespace dynamatic {
namespace buffer {
namespace fpga20 {

// An wrapper class for instrumenting FPGA20 buffers.
class MyFPGA20Buffers : public FPGA20Buffers {
public:
  // constructor
  MyFPGA20Buffers(PerfOptInfo &info, GRBEnv &env, FuncInfo &funcInfo,
                  const TimingDatabase &timingDB, double targetPeriod,
                  bool legacyPlacement, Logger &logger, StringRef milpName)
      : FPGA20Buffers(env, funcInfo, timingDB, targetPeriod, legacyPlacement,
                      logger, milpName),
        info(info){};
  MyFPGA20Buffers(PerfOptInfo &info, GRBEnv &env, FuncInfo &funcInfo,
                  const TimingDatabase &timingDB, double targetPeriod,
                  bool legacyPlacement)
      : FPGA20Buffers(env, funcInfo, timingDB, targetPeriod, legacyPlacement),
        info(info){};
  PerfOptInfo &info;
  void extractResult(BufferPlacement &placement) override {
    FPGA20Buffers::extractResult(placement);
    // do my things
    llvm::errs() << "Extracting data!\n";
    // Log global CFDFC throuhgputs
    for (auto [idx, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {
      auto [cf, cfVars] = cfdfcWithVars;
      double throughput = cfVars.throughput.get(GRB_DoubleAttr_X);
      // llvm::errs() << "Throughput of CFDFC #" << idx << ": " << throughput
      //              << "\n";
      info.cfcThroughput[cf] = throughput;
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
using fpga20::MyFPGA20Buffers;

// An wrapper class that applies buffer p
// extracts the report.
struct BufferPlacementWrapperPass : public HandshakePlaceBuffersPass {
  BufferPlacementWrapperPass(PerfOptInfo &info, StringRef algorithm,
                             StringRef frequencies, StringRef timingModels,
                             bool firstCFDFC, double targetCP, unsigned timeout,
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
      return checkLoggerAndSolve<MyFPGA20Buffers>(
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
};
} // namespace

void CreditBasedSharingPass::runDynamaticPass() {
  llvm::errs() << "***** Resource Sharing *****\n";

  ModuleOp modOp = getOperation();

  // TODO: how to handle different functions?
  PerfOptInfo data;

  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    return signalPassFailure();

  // running buffer placement on current module
  mlir::PassManager pm(&getContext());
  pm.addPass(std::make_unique<BufferPlacementWrapperPass>(
      data, algorithm, frequencies, timingModels, firstCFDFC, targetCP, timeout,
      dumpLogs));
  if (failed(pm.run(modOp))) {
    return signalPassFailure();
  }

  // check the extracted data
  for (auto [idx, cfdfc] : llvm::enumerate(data.cfcThroughput)) {
    llvm::errs() << "Throughput of CFDFC #" << idx << " is " << cfdfc.second
                 << "\n";
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
