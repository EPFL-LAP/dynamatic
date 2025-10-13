//===- Crush.cpp - Credit-Based Resource Sharing ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements the credit-based resource sharing pass
// It contains the following components:
// - An implementation of the sharing target decision heuristic; the heuristic
//   decides sharing groups---groups of operations that will share the same unit
//   in the circuit
// - An implementation of the access priority decision heuristic; for a sharing
//   group, the heuristic decides which operation to start first when multiple
//   of them can start at the same time.
// - An implementation of the MLIR transformation strategy to replace multiple
//   operations in the sharing group with a single operation, and add a sharing
//   wrapper around it to manage access to the share operation.
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/ResourceSharing/Crush.h"
#include "dynamatic/Analysis/CFDFCAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "experimental/Transforms/ResourceSharing/SharingSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <list>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::sharing;
using namespace dynamatic::buffer;

namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_CREDITBASEDSHARING
#include "experimental/Transforms/Passes.h.inc"
}; // namespace experimental
}; // namespace dynamatic

static constexpr unsigned MAX_GROUP_SIZE = 20;

// A FuncPerfInfo holds the extracted data from buffer placement, for a single
// handshake FuncOp.
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
using SharingInfo = std::map<handshake::FuncOp, FuncPerfInfo>;

// A sharing group Group holds a list of operations that share one unit.
using Group = std::vector<Operation *>;

// SharingGroups: a list of operations that share the same unit.
using SharingGroups = std::list<Group>;

namespace {

void loadFuncPerfInfoFromAnalysis(handshake::FuncOp funcOp,
                                  SharingInfo &sharingInfo,
                                  CFDFCAnalysis &analysis) {

  SmallVector<CFDFC *> cfdfcPtrs;

  for (auto &cfdfc : analysis.mapFuncOpToCFDFCs[funcOp]) {
    // Note: here cfdfc must be a reference to the objects in the vector,
    // otherwise the copy that "&cfdfc" points to will immediately goes out of
    // scope after the loop.
    cfdfcPtrs.push_back(&cfdfc);
  }

  std::vector<CFDFCUnion> disjointUnions;
  getDisjointBlockUnions(cfdfcPtrs, disjointUnions);

  // Map each individual CFDFC to its iteration index
  std::map<CFDFC *, size_t> cfIndices;

  for (auto [cfdfcIdx, cfdfc] :
       llvm::enumerate(analysis.mapFuncOpToCFDFCs[funcOp])) {
    cfIndices[&cfdfc] = cfdfcIdx;
  }

  for (auto &cfdfc : analysis.mapFuncOpToCFDFCs[funcOp]) {
    sharingInfo[funcOp].cfThroughput[cfIndices[&cfdfc]] = cfdfc.throughput;
    sharingInfo[funcOp].cfUnits[cfIndices[&cfdfc]] =
        std::set(cfdfc.units.begin(), cfdfc.units.end());
    // Track the channels of the CFC
    for (Value val : cfdfc.channels) {
      Channel *ch = new Channel(val);
      sharingInfo[funcOp].cfChannels[cfIndices[&cfdfc]].insert(ch);
    }
  }
  // For each CFDFC Union, mark the most-frequently-executed CFC as performance
  // critical.
  for (CFDFCUnion &cfUnion : disjointUnions) {
    CFDFC **critCf =
        std::max_element(cfUnion.cfdfcs.begin(), cfUnion.cfdfcs.end(),
                         [](CFDFC const *l, CFDFC const *r) {
                           return l->numExecs < r->numExecs;
                         });
    if (!critCf) {
      funcOp->emitError()
          << "Failed running determining performance critical CFC";
      return;
    }

    sharingInfo[funcOp].critCfcs.emplace(cfIndices[*critCf]);
  }
}

// For two sharing groups, check if the following criteria hold (see
// descriptions below).
bool checkGroupMergable(const Group &g1, const Group &g2,
                        FuncPerfInfo funcPerfInfo) {
  if (g1.empty() || g2.empty())
    return false;

  std::set<Operation *> gMerged;
  gMerged.insert(g1.begin(), g1.end());
  gMerged.insert(g2.begin(), g2.end());

  if (gMerged.size() > MAX_GROUP_SIZE)
    return false;

  // All operations in the group must have the same type as the first op in the
  // group
  OperationName opName = (*(gMerged.begin()))->getName();
  Type groupType = (*(gMerged.begin()))->getResultTypes().front();

  // 1. The merged group must have operations of the same type (both op type and
  // data type).
  for (Operation *op : gMerged) {
    if (op->getName() != opName)
      return false;
    if (op->getResultTypes().front() != groupType)
      return false;
  }

  // 2. For each CFC, the sum of occupancy must be smaller than the capacity
  // (i.e., units in CFC must no greater than the II).
  // This is equivalent to checking that throughput * n_ops <= 1;

  // 3. For each CFC, there must be no two operations have the same SCC ID (this
  // is simplified). TODO: we could try to check that if two operations in the
  // same SCC never start at the same time, then we can put them into the same
  // group without hurting the performance.
  for (unsigned long cf : funcPerfInfo.critCfcs) {
    // For each cf, numOps contains the number of operations
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
        unionGroup.insert(unionGroup.end(), g2->begin(), g2->end());
        sharingGroups.push_back(unionGroup);
        sharingGroups.erase(g1);
        sharingGroups.erase(g2);
        return true;
      }
  return false;
}

// For a given sharingGroup, we determine an access priority order that does not
// hurt the performance.
void sortGroups(SharingGroups &sharingGroups, FuncPerfInfo &info) {
  for (Group &g : sharingGroups) {
    // Use bubble sort to sort each group:
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
                    TimingDatabase &timingDB, FuncPerfInfo &funcPerfInfo,
                    double targetCP) {

  double latency;
  for (Operation *target : sharingTargets) {
    // By default, the op is assigned with no occupancy. If a performance
    // critical CFC contains that op, then we set the occupancy to the occupancy
    // of op in that CFC.
    opOccupancy[target] = 0.0;
    for (auto cf : funcPerfInfo.critCfcs) {
      if (funcPerfInfo.cfUnits[cf].find(target) !=
          funcPerfInfo.cfUnits[cf].end()) {
        if (failed(timingDB.getLatency(target, SignalType::DATA, latency,
                                       targetCP)))
          latency = 0.0;
        // Formula for operation occupancy:
        // Occupancy = Latency / II = Latency * Throughput.
        opOccupancy[target] = latency * funcPerfInfo.cfThroughput[cf];
      }
    }
  }
}

/// Replaces the first use of `oldVal` by `newVal` in the operation's operands.
/// Asserts if the operation's operands do not contain the old value.
void replaceFirstUse(Operation *op, Value oldVal, Value newVal) {
  for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i) {
    if (op->getOperand(i) == oldVal) {
      op->setOperand(i, newVal);
      return;
    }
  }
  llvm_unreachable("failed to find operation operand");
}

} // namespace

struct CreditBasedSharingPass
    : public dynamatic::experimental::impl::CreditBasedSharingBase<
          CreditBasedSharingPass> {

  using CreditBasedSharingBase::CreditBasedSharingBase;
  void runOnOperation() override;

  LogicalResult sharingInFuncOp(handshake::FuncOp funcOp,
                                FuncPerfInfo &funcPerfInfo, NameAnalysis &namer,
                                TimingDatabase &timingDB, double targetCP);

  LogicalResult sharingWrapperInsertion(
      handshake::FuncOp &funcOp, SharingGroups &sharingGroups,
      MapVector<Operation *, double> &opOccupancy, TimingDatabase &timingDB);

  // This class method finds all sharing targets for a given handshake function
  SmallVector<mlir::Operation *> getSharingTargets(handshake::FuncOp funcOp) {
    SmallVector<Operation *> targets;

    funcOp.walk([&](Operation *op) {
      if (isa<
              // clang-format off
              handshake::MulFOp,
              handshake::AddFOp,
              handshake::SubFOp,
              handshake::MulIOp,
              handshake::DivUIOp,
              handshake::DivSIOp,
              handshake::DivFOp
              // clang-format on
              >(op)) {
        assert(op->getNumOperands() > 1 && op->getNumResults() == 1 &&
               "Invalid sharing target is being added to the list of sharing "
               "targets! Currently operations with 1 input or more than 1 "
               "outputs are not supported!");
        targets.emplace_back(op);
      }
    });

    return targets;
  }
};

// This function
// 1. Replaces a group of operations with a single operation
// 2. Creates a sharing wrapper that manages selecting the correct inputs of the
//    operations and dispatches the result to the correct outputs.
LogicalResult CreditBasedSharingPass::sharingWrapperInsertion(
    handshake::FuncOp &funcOp, SharingGroups &sharingGroups,
    MapVector<Operation *, double> &opOccupancy, TimingDatabase &timingDB) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  for (Group group : sharingGroups) {

    // If the group only has one operation or has no operations, then go to next
    // group
    if (group.size() <= 1)
      continue;

    // Elect one operation as the shared operation.
    Operation *sharedOp = *group.begin();

    double latency;
    if (failed(
            timingDB.getLatency(sharedOp, SignalType::DATA, latency, targetCP)))
      latency = 0.0;

    // Maps each original successor and the input operand (Value)
    std::vector<std::tuple<Operation *, Value>> succValueMap;
    for (Operation *op : group)
      for (Operation *succ : op->getResult(0).getUsers())
        succValueMap.emplace_back(succ, op->getResult(0));

    // The output values from the predecessors of the operations in the group.
    llvm::SmallVector<Value, 24> dataOperands;
    for (Operation *op : group)
      llvm::copy(op->getOperands(), std::back_inserter(dataOperands));

    // Check if the number of results is exactly 1.
    assert(sharedOp->getNumResults() == 1 &&
           "Sharing wrapper currently only supports operation with a single "
           "return value.");

    // Result types (this also tracks the number of results).
    llvm::SmallVector<Type> sharingWrapperOutputTypes;

    // The outputs of the original operations are also the outputs of the
    // sharing wrapper.
    for (Operation *op : group)
      sharingWrapperOutputTypes.push_back(op->getResultTypes()[0]);

    // The inputs of the shared operation is also the output of the sharing
    // wrapper.
    sharingWrapperOutputTypes.insert(sharingWrapperOutputTypes.end(),
                                     sharedOp->getOperandTypes().begin(),
                                     sharedOp->getOperandTypes().end());

    assert(group.size() * sharedOp->getNumOperands() == dataOperands.size() &&
           "The sharing wrapper has an incorrect number of input ports.");

    // Determining the number of credits of each operation that share the
    // unit based on the maximum achievable occupancy in critical CFCs.
    llvm::SmallVector<int64_t> credits;
    for (Operation *op : group) {
      double occupancy = opOccupancy[op];
      // The number of credits must be an integer. It is incremented by 1 to
      // hide the latency of returning a credit, and accounts for token
      // staying in the output buffers due to the effect of sharing.
      credits.push_back(1 + std::ceil(occupancy));
    }

    assert(sharingWrapperOutputTypes.size() ==
               sharedOp->getNumOperands() + group.size() &&
           "The sharing wrapper has an incorrect number of output ports.");

    builder.setInsertionPoint(*group.begin());
    handshake::SharingWrapperOp wrapperOp =
        builder.create<handshake::SharingWrapperOp>(
            sharedOp->getLoc(), sharingWrapperOutputTypes, dataOperands,
            sharedOp->getResult(0), llvm::ArrayRef<int64_t>(credits),
            credits.size(), sharedOp->getNumOperands(),
            (unsigned)round(latency));

    // Replace original connection from op->successor to
    // sharingWrapper->successor
    for (auto [id, succValue] : llvm::enumerate(succValueMap)) {
      auto [succ, val] = succValue;
      replaceFirstUse(succ, val, wrapperOp->getResult(id));
    }

    // If operation1 in the group is feeding another operation2 in the group,
    // the above method will retain operation1->wrapperOp, instead of
    // wrapperOp->wrapperOp. The code below will correct this case.
    for (auto origInputValue : dataOperands)
      for (auto [outId, op] : llvm::enumerate(group))
        if (op == origInputValue.getDefiningOp())
          wrapperOp->replaceUsesOfWith(origInputValue,
                                       wrapperOp.getResult(outId));

    wrapperOp->setOperand(wrapperOp.getNumOperands() - 1,
                          sharedOp->getResult(0));

    // Connect the last outputs of the sharing wrapper to the input of the
    // shared operation.
    for (auto [id, val] : llvm::enumerate(sharedOp->getOperands()))
      sharedOp->replaceUsesOfWith(val, wrapperOp->getResult(id + group.size()));

    // Remove all the operations in the group except for the shared one.
    for (Operation *op : group)
      if (op != sharedOp)
        op->erase();

    // After sharing, BB ID becomes meaningless for the shared unit, so we
    // simply remove it
    sharedOp->removeAttr(dynamatic::BB_ATTR_NAME);
  }

  return success();
}

LogicalResult CreditBasedSharingPass::sharingInFuncOp(
    handshake::FuncOp funcOp, FuncPerfInfo &funcPerfInfo, NameAnalysis &namer,
    TimingDatabase &timingDB, double targetCP) {

  // Get all the sharing targets within the funcOp
  SmallVector<Operation *> sharingTargets = getSharingTargets(funcOp);

  // opOccupancy: maps each operation to the maximum occupancy it has to
  // achieve.
  llvm::MapVector<Operation *, double> opOccupancy;
  getOpOccupancy(sharingTargets, opOccupancy, timingDB, funcPerfInfo, targetCP);

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

  // Merge groups
  for (bool continueMerging = true; continueMerging;)
    continueMerging = tryMergeGroups(sharingGroups, funcPerfInfo);

  // Sort each sharing group according to their SCC ID.
  sortGroups(sharingGroups, funcPerfInfo);

  // For each sharing group, unite them with a sharing wrapper and shared
  // operation.
  return sharingWrapperInsertion(funcOp, sharingGroups, opOccupancy, timingDB);
}

void CreditBasedSharingPass::runOnOperation() {
  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  TimingDatabase timingDB;
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    signalPassFailure();

  // Buffer placement requires that all values are used exactly once
  auto modOp = dyn_cast<ModuleOp>(getOperation());
  if (failed(verifyIRMaterialized(modOp))) {
    modOp->emitError() << ERR_NON_MATERIALIZED_MOD;
    return;
  }

  SharingInfo sharingInfo;
  auto performanceAnalysis = getCachedAnalysis<CFDFCAnalysis>();

  if (!performanceAnalysis.has_value()) {
    llvm::errs() << "Performance analysis result NOT available, share "
                    "functional units as much as possible...\n";
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
      FuncPerfInfo funcPerfInfo;
      sharingInfo[funcOp] = funcPerfInfo;
    }
  } else {
    llvm::errs() << "Performance analysis available, share functional units as "
                    "much as possible while maintaining the performance...\n";
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
      loadFuncPerfInfoFromAnalysis(funcOp, sharingInfo,
                                   performanceAnalysis.value());
    }
  }

  // If buffers are placed naively, then no critical CFC is set for each funcOp.
  // We can also share operations naively.

  // Apply resource sharing for each function in the module op.
  for (auto &[funcOp, funcPerfInfo] : sharingInfo) {
    if (failed(
            sharingInFuncOp(funcOp, funcPerfInfo, namer, timingDB, targetCP))) {
      signalPassFailure();
    }
  }
}
