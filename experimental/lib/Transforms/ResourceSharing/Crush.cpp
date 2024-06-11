//===- Crush.cpp - Credit-Based Resource Sharing ----------------*- C++ -*-===//
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
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/MILP.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "dynamatic/Transforms/BufferPlacement/FPL22Buffers.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "experimental/Transforms/ResourceSharing/SharingSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <list>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::sharing;

using namespace dynamatic::buffer;

/// Algorithms that do not require solving an MILP.
static constexpr llvm::StringLiteral ON_MERGES("on-merges");
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

// A sharing group Group holds a list of operations that share one unit.
using Group = std::vector<Operation *>;

// SharingGroups: a list of operations that share the same unit.
using SharingGroups = std::list<Group>;

// Wrapper function for saving the data retrived from buffer placement milp
// algorithm into a SharingInfo structure.
void loadFuncPerfInfo(SharingInfo &sharingInfo, MILPVars &vars,
                      FuncInfo &funcInfo) {

  // Map each individual CFDFC to its iteration index
  std::map<CFDFC *, size_t> cfIndices;

  SmallVector<CFDFC *, 8> cfdfcs;
  std::vector<CFDFCUnion> disjointUnions;
  llvm::transform(funcInfo.cfdfcs, std::back_inserter(cfdfcs),
                  [](auto cfAndOpt) { return cfAndOpt.first; });

  getDisjointBlockUnions(cfdfcs, disjointUnions);

  // Map each CFDFC to a numeric ID.
  for (auto [id, cfAndOpt] : llvm::enumerate(funcInfo.cfdfcs))
    cfIndices[cfAndOpt.first] = id;

  // Extract result: save global CFDFC throuhgputs into sharingInfo
  for (auto [id, cfdfcWithVars] : llvm::enumerate(vars.cfVars)) {

    auto [cf, cfVars] = cfdfcWithVars;
    double throughput = cfVars.throughput.get(GRB_DoubleAttr_X);

    sharingInfo[&funcInfo.funcOp].cfThroughput[cfIndices[cf]] = throughput;

    sharingInfo[&funcInfo.funcOp].cfUnits[cfIndices[cf]] =
        std::set(cf->units.begin(), cf->units.end());

    // // Track the units of the CFC
    // for (auto *op : cf->units)
    //   sharingInfo[&funcInfo.funcOp].cfUnits[cfIndices[cf]].insert(op);

    // Track the channels of the CFC
    for (Value val : cf->channels) {
      Channel *ch = new Channel(val);

      sharingInfo[&funcInfo.funcOp].cfChannels[cfIndices[cf]].insert(ch);
    }
  }

  // or each CFDFC Union, mark the most-frequently-executed
  // CFC as performance critical.
  for (CFDFCUnion &cfUnion : disjointUnions) {

    CFDFC **critCf =
        std::max_element(cfUnion.cfdfcs.begin(), cfUnion.cfdfcs.end(),
                         [](CFDFC const *l, CFDFC const *r) {
                           return l->numExecs < r->numExecs;
                         });
    if (!critCf) {
      funcInfo.funcOp->emitError()
          << "Failed running determining performance critical CFC";
      return;
    }

    sharingInfo[&funcInfo.funcOp].critCfcs.emplace(cfIndices[*critCf]);
  }
}

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
namespace dynamatic {
namespace buffer {
namespace fpga20 {

// An wrapper class for extracting CFDFC performance from FPGA20 buffers.
class FPGA20BuffersWrapper : public FPGA20Buffers {
public:
  // constructor
  FPGA20BuffersWrapper(SharingInfo &sharingInfo, GRBEnv &env,
                       FuncInfo &funcInfo, const TimingDatabase &timingDB,
                       double targetPeriod, bool legacyPlacement,
                       Logger &logger, StringRef milpName)
      : FPGA20Buffers(env, funcInfo, timingDB, targetPeriod, legacyPlacement,
                      logger, milpName),
        sharingInfo(sharingInfo){};
  FPGA20BuffersWrapper(SharingInfo &sharingInfo, GRBEnv &env,
                       FuncInfo &funcInfo, const TimingDatabase &timingDB,
                       double targetPeriod, bool legacyPlacement)
      : FPGA20Buffers(env, funcInfo, timingDB, targetPeriod, legacyPlacement),
        sharingInfo(sharingInfo){};

private:
  SharingInfo &sharingInfo;
  void extractResult(BufferPlacement &placement) override {
    // Run the FPGA20Buffers's extractResult as it is
    FPGA20Buffers::extractResult(placement);

    loadFuncPerfInfo(sharingInfo, vars, funcInfo);
  }
};

} // namespace fpga20

namespace fpl22 {

class FPL22BuffersWraper : public CFDFCUnionBuffers {
public:
  FPL22BuffersWraper(SharingInfo &sharingInfo, GRBEnv &env, FuncInfo &funcInfo,
                     const TimingDatabase &timingDB, double targetPeriod,
                     CFDFCUnion &cfUnion, Logger &logger, StringRef milpName)
      : CFDFCUnionBuffers(env, funcInfo, timingDB, targetPeriod, cfUnion,
                          logger, milpName),
        sharingInfo(sharingInfo){};
  FPL22BuffersWraper(SharingInfo &sharingInfo, GRBEnv &env, FuncInfo &funcInfo,
                     const TimingDatabase &timingDB, double targetPeriod,
                     CFDFCUnion &cfUnion)
      : CFDFCUnionBuffers(env, funcInfo, timingDB, targetPeriod, cfUnion),
        sharingInfo(sharingInfo){};

private:
  SharingInfo &sharingInfo;

  void extractResult(BufferPlacement &placement) override {
    // Run the FPL22BuffersBase's extractResult as it is
    FPL22BuffersBase::extractResult(placement);

    loadFuncPerfInfo(sharingInfo, vars, funcInfo);
  }
};

} // namespace fpl22

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

/// Wraps a call to solveMILP and conditionally passes the logger and MILP name
/// to the MILP's constructor as last arguments if the logger is not null.
template <typename MILP, typename... Args>
static inline LogicalResult
checkLoggerAndSolve(Logger *logger, StringRef milpName,
                    BufferPlacement &placement, Args &&...args) {
  if (logger)
    return solveMILP<MILP>(placement, std::forward<Args>(args)..., *logger,
                           milpName);
  return solveMILP<MILP>(placement, std::forward<Args>(args)...);
}

namespace {
using fpga20::FPGA20BuffersWrapper;
using fpl22::FPL22BuffersWraper;

// An wrapper class that applies buffer p
// extracts the report.
struct HandshakePlaceBuffersPassWrapper : public HandshakePlaceBuffersPass {
  HandshakePlaceBuffersPassWrapper(SharingInfo &sharingInfo,
                                   StringRef algorithm, StringRef frequencies,
                                   StringRef timingModels, bool firstCFDFC,
                                   double targetCP, unsigned timeout,
                                   bool dumpLogs)
      : HandshakePlaceBuffersPass(algorithm, frequencies, timingModels,
                                  firstCFDFC, targetCP, timeout, dumpLogs),
        sharingInfo(sharingInfo){};
  SharingInfo &sharingInfo;

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
  LogicalResult getBufferPlacement(FuncInfo &funcInfo, TimingDatabase &timingDB,
                                   Logger *logger,
                                   BufferPlacement &placement) override {

    // Create Gurobi environment
    GRBEnv env = GRBEnv(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    if (timeout > 0)
      env.set(GRB_DoubleParam_TimeLimit, timeout);
    env.start();

    if (algorithm == FPGA20 || algorithm == FPGA20_LEGACY)
      // Create and solve the MILP
      return checkLoggerAndSolve<FPGA20BuffersWrapper>(
          logger, "placement", placement, sharingInfo, env, funcInfo, timingDB,
          targetCP, algorithm != FPGA20);
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
      for (auto [id, cfUnion] : llvm::enumerate(disjointUnions)) {
        std::string milpName = "cfdfc_placement_" + std::to_string(id);
        if (failed(checkLoggerAndSolve<FPL22BuffersWraper>(
                logger, milpName, placement, sharingInfo, env, funcInfo,
                timingDB, targetCP, cfUnion)))
          return failure();
      }

      // Solve last MILP on channels/units that are not part of any CFDFC
      return checkLoggerAndSolve<fpl22::OutOfCycleBuffers>(
          logger, "out_of_cycle", placement, env, funcInfo, timingDB, targetCP);
    }

    llvm_unreachable("unknown algorithm");
  }
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
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

    for (Operation &op : funcOp.getOps()) {
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
    if (failed(pm.run(modOp)))
      return failure();

    return success();
  }
};
} // namespace

// For two sharing groups, check if the following criteria hold (see
// descriptions below).
bool checkGroupMergable(const Group &g1, const Group &g2,
                        FuncPerfInfo funcPerfInfo) {
  if (g1.empty() || g2.empty())
    return false;

  std::set<Operation *> gMerged;
  gMerged.insert(g1.begin(), g1.end());
  gMerged.insert(g2.begin(), g2.end());

  OperationName opName = (*(gMerged.begin()))->getName();

  // 1. The merged group must have operations of the same type.
  for (Operation *op : gMerged)
    if (op->getName() != opName)
      return false;

  // 2. For each CFC, the sum of occupancy must be smaller than the capacity
  // (i.e., units in CFC must no greater than the II).
  // This is equivalent to checking that throughput * n_ops <= 1;

  // 3. For each CFC, there must be no two operations have the same SCC ID (this
  // is simplified).
  for (unsigned long cf : funcPerfInfo.critCfcs) {
    // for each cf, numOps contains the number of operations
    // that are in the merged group and also in cf
    unsigned numOps = 0;

    // listOfSccIds: a list of SCC IDs that the group has
    // it is used to check if there are any duplicates (i.e.,
    // two operation that are in the same SCC cannot be in the
    // same sharing group).
    std::vector<size_t> listOfSccIds;
    for (Operation *op : funcPerfInfo.cfUnits[cf]) {
      // In the op is in (SCC union MergedGroup):
      if (gMerged.find(op) != gMerged.end()) {
        // increase number of Ops
        numOps++;
        // Push back the SCC ID of each op inside the group and
        // also SCC;
        listOfSccIds.push_back(funcPerfInfo.cfSccs[cf][op]);
      }
    }
    // Check if there are any duplicates:
    std::set<size_t> setOfSccIds(listOfSccIds.begin(), listOfSccIds.end());
    // Check if numOps * cfcThroughput <= 1 and no duplicate SCC
    // IDs.
    if (numOps * (funcPerfInfo.cfThroughput)[cf] > 1)
      return false;
    if ((listOfSccIds.size() != setOfSccIds.size()))
      return false;
  }

  // If none of the checks has failed, then return true
  return true;
}

// A greedy algorithm that test checkGroupMergable on combination of
// 2 groups, if success then the 2 given groups are merged, and immediately
// returns true if successfully merged groups, otherwise it returns false.
bool tryMergeGroups(SharingGroups &sharingGroups, const FuncPerfInfo &info) {
  for (auto g1 = sharingGroups.begin(); g1 != sharingGroups.end(); g1++)
    for (auto g2 = std::next(g1); g2 != sharingGroups.end(); g2++)
      if (checkGroupMergable(*g1, *g2, info)) {
        // If all three criteria met, then merge the second group into the
        // first group.
        Group unionGroup = *g1;
        // unionGroup.insert(g1->begin(), g1->end());
        unionGroup.insert(unionGroup.end(), g2->begin(), g2->end());
        sharingGroups.push_back(unionGroup);
        sharingGroups.erase(g1);
        sharingGroups.erase(g2);
        return true;
      }
  return false;
}

void logGroups(const SharingGroups &sharingGroups, NameAnalysis &namer) {
  for (const Group &group : sharingGroups) {
    llvm::errs() << "group : {";
    for (auto *op : group) {
      llvm::errs() << namer.getName(op) << " ";
    }
    llvm::errs() << "}\n";
  }
}

void sortGroups(SharingGroups &sharingGroups, FuncPerfInfo &info) {
  for (Group &g : sharingGroups) {
    // use bubble sort to sort each group:
    if (g.size() <= 1)
      continue;
    bool modified = false;

    do {
      modified = false;
      for (size_t i = 1; i < g.size(); i++)
        for (auto cf : info.critCfcs) {

          auto op1 = info.cfSccs[cf].find(g[i - 1]);
          auto op2 = info.cfSccs[cf].find(g[i]);
          if (op1 != info.cfSccs[cf].end() && op2 != info.cfSccs[cf].end() &&
              op1->second > op2->second) {
            iter_swap(g.begin() + i, g.begin() + i - 1);
            modified = true;
          }
        }
    } while (modified);
  }
}

// Set opOccupancy[op] to the occupancy required to achieve maximum performance
// of all performance critical CFCs.
void getOpOccupancy(const SmallVector<Operation *> &sharingTargets,
                    llvm::MapVector<Operation *, double> &opOccupancy,
                    TimingDatabase &timingDB, FuncPerfInfo &funcPerfInfo) {

  double latency;
  for (Operation *target : sharingTargets) {
    // By default, the op is assigned with no occupancy. If a performance
    // critical CFC contains that op, then we set the occupancy to the occupancy
    // of op in that CFC.
    opOccupancy[target] = 0.0;
    for (auto cf : funcPerfInfo.critCfcs) {
      if (funcPerfInfo.cfUnits[cf].find(target) !=
          funcPerfInfo.cfUnits[cf].end()) {
        if (failed(timingDB.getLatency(target, SignalType::DATA, latency)))
          latency = 0.0;
        // Occupancy = Latency / II = Latency * Throughput.
        opOccupancy[target] = latency * funcPerfInfo.cfThroughput[cf];
      }
    }
  }
}

LogicalResult
sharingWrapperInsertion(handshake::FuncOp &funcOp, SharingGroups &sharingGroups,
                        MLIRContext *ctx,
                        MapVector<Operation *, double> &opOccupancy) {
  OpBuilder builder(ctx);
  for (Group group : sharingGroups) {

    // If the group only has one operation or has no operations, then go to next
    // group
    if (group.size() <= 1)
      continue;

    // Elect one operation as the shared operation.
    Operation *sharedOp = *group.begin();

    // Enumerate the operands of the original pre-sharing operations.
    llvm::SmallVector<Value> sharingWrapperInputs;
    for (Operation *op : group)
      for (Value val : op->getOperands())
        sharingWrapperInputs.push_back(val);

    // Check if the number of results is exactly 1.
    assert(sharedOp->getNumResults() == 1 &&
           "Sharing wrapper currently only supports operation with a single "
           "return value.");

    // The result of the shared operation is also one of the inputs of the
    // sharing wrapper.
    sharingWrapperInputs.push_back(sharedOp->getResult(0));

    assert(group.size() * sharedOp->getNumOperands() +
                   sharedOp->getNumResults() ==
               sharingWrapperInputs.size() &&
           "The sharing wrapper has an incorrect number of input ports.");

    // Determining the number of credits of each operation that share the unit
    // based on the maximum achievable occupancy in critical CFCs.
    llvm::SmallVector<int64_t> credits;
    for (Operation *op : group) {
      double occupancy = opOccupancy[op];
      // The number of credits must be an integer. It is incremented by 1 to
      // hide the latency of returning a credit, and accounts for token staying
      // in the output buffers due to the effect of sharing.
      credits.push_back(1 + std::ceil(occupancy));
    }

    // Retrieve the type reference the the number of credits per each operation.
    NamedAttribute namedCreditsAttr(
        StringAttr::get(ctx, "credits"),
        builder.getDenseI64ArrayAttr(llvm::ArrayRef<int64_t>(credits)));

    // Result types (this also tracks the number of results).
    llvm::SmallVector<Type> sharingWrapperOutputTypes;

    // The outputs of the original operations are also the outputs of the
    // sharing wrapper.
    for (Operation *op : group) {
      sharingWrapperOutputTypes.push_back(op->getResultTypes()[0]);
    }

    // The inputs of the shared operation is also the output of the sharing
    // wrapper.
    sharingWrapperOutputTypes.insert(sharingWrapperOutputTypes.end(),
                                     sharedOp->getOperandTypes().begin(),
                                     sharedOp->getOperandTypes().end());

    assert(sharingWrapperOutputTypes.size() ==
               sharedOp->getNumOperands() + group.size() &&
           "The sharing wrapper has an incorrect number of output ports.");

    builder.setInsertionPoint(*group.begin());
    handshake::SharingWrapperOp wrapperOp =
        builder.create<handshake::SharingWrapperOp>(
            sharedOp->getLoc(), sharingWrapperOutputTypes, sharingWrapperInputs,
            namedCreditsAttr);

    for (auto [id, op] : llvm::enumerate(group))
      op->getResult(0).replaceAllUsesWith(wrapperOp->getResult(id));

    for (auto [id, val] : llvm::enumerate(sharedOp->getOperands()))
      sharedOp->replaceUsesOfWith(val, wrapperOp->getResult(id + group.size()));

    // Remove all operations in the sharing group except for the shared one.
    for (Operation *op : group)
      if (op != *(group.begin()))
        op->erase();
  }

  return success();
}

void CreditBasedSharingPass::runDynamaticPass() {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    signalPassFailure();

  // Buffer placement requires that all values are used exactly once
  ModuleOp modOp = getOperation();
  if (failed(verifyIRMaterialized(modOp))) {
    modOp->emitError() << ERR_NON_MATERIALIZED_MOD;
    return;
  }

  SharingInfo sharingInfo;

  // Run buffer placement pass and fill sharingInfo with performance analysis
  // information
  if (failed(runBufferPlacementPass(modOp, sharingInfo))) {
    modOp->emitError() << "Failed running buffer placement";
    return;
  }

  // If buffers are placed naively, then no critical CFC is set for each funcOp.
  // We can also share operations naively.
  if (algorithm == ON_MERGES)
    for (handshake::FuncOp funcOp :
         getOperation().getOps<handshake::FuncOp>()) {
      FuncPerfInfo funcPerfInfo;
      sharingInfo[&funcOp] = funcPerfInfo;
    }

  for (auto &[funcOp, funcPerfInfo] : sharingInfo) {

    // Check the sharing targets
    SmallVector<Operation *> sharingTargets = getSharingTargets(*funcOp);

    // opOccupancy: maps each operation to the maximum occupancy it has to
    // achieve.
    llvm::MapVector<Operation *, double> opOccupancy;
    getOpOccupancy(sharingTargets, opOccupancy, timingDB, funcPerfInfo);

    // Initialize the sharing groups:
    SharingGroups sharingGroups;
    for (auto [id, op] : llvm::enumerate(sharingTargets))
      sharingGroups.emplace_back(Group{op});

    // Determine SCCs
    for (auto critCfc : funcPerfInfo.critCfcs) {
      std::map<Operation *, size_t> sccMap = getSccsInCfc(
          funcPerfInfo.cfUnits[critCfc], funcPerfInfo.cfChannels[critCfc]);
      funcPerfInfo.cfSccs.emplace(critCfc, sccMap);
    }

    llvm::errs() << "Initial groups\n";
    logGroups(sharingGroups, namer);

    // Merge groups
    for (bool continueMerging = true; continueMerging;)
      continueMerging = tryMergeGroups(sharingGroups, funcPerfInfo);

    // log
    llvm::errs() << "Finished merging\n";
    logGroups(sharingGroups, namer);

    // Sort each sharing group according to their SCC ID.
    sortGroups(sharingGroups, funcPerfInfo);
    // log
    llvm::errs() << "Sorted groups\n";
    logGroups(sharingGroups, namer);

    // For each sharing group, unite them with a sharing wrapper and shared
    // operation.
    if (failed(
            sharingWrapperInsertion(*funcOp, sharingGroups, ctx, opOccupancy)))
      signalPassFailure();
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
