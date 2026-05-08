//===- HandshakeAnnotateProperties.cpp - Property annotation ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --handshake-annotate-properties pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/IndexChannelAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LinearAlgebra/Gaussian.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/FormalProperty.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <ostream>
#include <unordered_set>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Analysis/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_HANDSHAKEANNOTATEPROPERTIES
#include "experimental/Analysis/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

namespace {

struct HandshakeAnnotatePropertiesPass
    : public dynamatic::experimental::impl::HandshakeAnnotatePropertiesBase<
          HandshakeAnnotatePropertiesPass> {

  using HandshakeAnnotatePropertiesBase::HandshakeAnnotatePropertiesBase;

  void runDynamaticPass() override;

private:
  unsigned int uid;
  json::Array propertyTable;

  LogicalResult annotateProperty(ModuleOp modOp, FormalProperty::TYPE t);
  LogicalResult annotateQueriedProperties();
  LogicalResult annotateAbsenceOfBackpressure(ModuleOp modOp);
  LogicalResult annotateValidEquivalence(ModuleOp modOp);
  LogicalResult annotateValidEquivalenceBetweenOps(Operation &op1,
                                                   Operation &op2);
  LogicalResult annotateEagerForkNotAllOutputSent(ModuleOp modOp);
  LogicalResult
  annotateCopiedSlotsRec(std::unordered_set<std::string> &visitedSet,
                         handshake::EagerForkLikeOpInterface &originFork,
                         Operation &curOp);
  LogicalResult annotateCopiedSlots(Operation &op);
  LogicalResult annotateCopiedSlotsOfAllForks(ModuleOp modOp);
  LogicalResult annotateReconvergentPathFlow(ModuleOp modOp);
  LogicalResult annotateEntryTokenOrderPaths(ControlMergeOp cmerge,
                                             int32_t entryValue);
  LogicalResult annotateEntryTokenOrder(ModuleOp modOp);
  LogicalResult annotateSingleEntryToken(ModuleOp modOp);
  LogicalResult annotateExitTokenOrder(ModuleOp modOp);
};

bool isChannelToBeChecked(OpResult res) {
  // The channel connected to EndOp, MemoryControllerOp, and LSQOp don't appear
  // in the properties database for the following reasons:
  // - EndOp: the operation doesn't exist in the output model; the property
  //   creation is still possible but requires to get the names of the model's
  //   I/O signals (not implemented yet)
  // - MemeoryControllerOp and LSQOp: only load and stores can be connected to
  //   these Ops, therefore we cannot rigidify their channels with the
  //   ReadyRemoverOp and ValidMergerOp
  if (isa<handshake::EndOp, handshake::MemoryControllerOp, handshake::LSQOp>(
          res.getOwner()))
    return false;

  return std::all_of(
      res.getUsers().begin(), res.getUsers().end(), [](auto *user) {
        return !isa<handshake::EndOp, handshake::MemoryControllerOp,
                    handshake::LSQOp>(*user);
      });
}
} // namespace

LogicalResult
HandshakeAnnotatePropertiesPass::annotateValidEquivalenceBetweenOps(
    Operation &op1, Operation &op2) {
  for (auto [i, res1] : llvm::enumerate(op1.getResults()))
    for (auto [j, res2] : llvm::enumerate(op2.getResults())) {
      // equivalence is symmetrical so it needs to be checked only once for
      // each pair of signals when the Ops are the same
      if ((&op1 != &op2 || i < j) && isChannelToBeChecked(res1) &&
          isChannelToBeChecked(res2)) {
        ValidEquivalence p(uid, FormalProperty::TAG::OPT, res1, res2);

        propertyTable.push_back(p.toJSON());
        uid++;
      }
    }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateValidEquivalence(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (auto &op : funcOp.getOps()) {
      if (failed(annotateValidEquivalenceBetweenOps(op, op))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateAbsenceOfBackpressure(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      for (auto [resIndex, res] : llvm::enumerate(op.getResults()))
        if (isChannelToBeChecked(res)) {

          AbsenceOfBackpressure p(uid, FormalProperty::TAG::OPT, res);

          propertyTable.push_back(p.toJSON());
          uid++;
        }
    }
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateEagerForkNotAllOutputSent(
    ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (auto forkOp = dyn_cast<handshake::EagerForkLikeOpInterface>(op)) {
        EagerForkNotAllOutputSent p(uid, FormalProperty::TAG::INVAR, forkOp);

        propertyTable.push_back(p.toJSON());
        uid++;
      }
    }
  }
  return success();
}

LogicalResult HandshakeAnnotatePropertiesPass::annotateCopiedSlotsRec(
    std::unordered_set<std::string> &visitedSet,
    handshake::EagerForkLikeOpInterface &originFork, Operation &curOp) {

  // If this operation has been visited, there is nothing to do
  std::string id = getUniqueName(&curOp).str();
  if (auto iter = visitedSet.find(id); iter != visitedSet.end()) {
    return success();
  }
  visitedSet.insert(id);

  // If this operation contains a slot, the copied slot has been found and can
  // be annotated
  if (auto bufferOp = dyn_cast<handshake::BufferLikeOpInterface>(curOp)) {
    CopiedSlotsOfActiveForkAreFull p(uid, FormalProperty::TAG::INVAR, bufferOp,
                                     originFork);
    propertyTable.push_back(p.toJSON());
    uid++;
    return success();
  }

  if (auto latencyOp = dyn_cast<handshake::LatencyInterface>(curOp)) {
    CopiedSlotsOfActiveForkAreFull p(uid, FormalProperty::TAG::INVAR, latencyOp,
                                     originFork);
    propertyTable.push_back(p.toJSON());
    uid++;
    return success();
  }

  if (auto mergeOp = dyn_cast<handshake::MergeLikeOpInterface>(curOp)) {
    // TODO: Which of the previous paths should be followed?
    return success();
  }

  // Only JoinLikeOps or single-operand ops are remaining, but ideally a
  // dyn_cast would happen for either case
  for (auto value : curOp.getOperands()) {
    Operation *prevOpPtr = value.getDefiningOp();
    if (prevOpPtr == nullptr)
      // if there is no defining op, the value must be a constant, and does not
      // need to be annotated
      continue;
    Operation &prevOp = *prevOpPtr;
    if (failed(annotateCopiedSlotsRec(visitedSet, originFork, prevOp))) {
      return failure();
    }
  }

  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateCopiedSlots(Operation &op) {
  std::unordered_set<std::string> visitedSet = {};
  if (auto forkOp = dyn_cast<handshake::EagerForkLikeOpInterface>(op)) {
    return annotateCopiedSlotsRec(visitedSet, forkOp, op);
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateCopiedSlotsOfAllForks(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (failed(annotateCopiedSlots(op)))
        return failure();
    }
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateReconvergentPathFlow(ModuleOp modOp) {
  auto &indexChannelAnalysis = getAnalysis<dynamatic::IndexChannelAnalysis>();

  // Local equations extracted in constructor
  FlowEquationExtractor extractor(indexChannelAnalysis);
  // This fails when some operations in the module are not yet handled
  if (failed(extractor.extractAll(modOp))) {
    return failure();
  }

  // Create a matrix, and map all variables to an column index
  FlowSystem indices(extractor.equations);
  MatIntType &matrix = indices.matrix;

  // Verify that the registry data structure is correct
  assert(indices.registry.verify());

  // bring to row-echelon form
  gaussianElimination(matrix);

  size_t rows = matrix.size1();
  for (size_t row = 0; row < rows; ++row) {
    bool canAnnotate = true;
    for (size_t col = 0; col < indices.nLambdas; ++col) {
      if (matrix(row, col) != 0) {
        canAnnotate = false;
        break;
      }
    }

    if (!canAnnotate) {
      continue;
    }

    FlowExpression expr = indices.getRowAsExpression(row);
    if (expr.terms.size() == 0) {
      continue;
    }
    ReconvergentPathFlow p(uid, FormalProperty::TAG::INVAR);
    p.addEquation(expr);
    if (p.getEquations().size() > 0) {
      uid++;
      propertyTable.push_back(p.toJSON());
    }
  }
  return success();
}

namespace {
struct EntryCMergePath {
  std::vector<EffectiveSlotNamer> slots;
  ControlMergeOp cmerge;
  int32_t entryValue;
};
std::vector<EntryCMergePath> findEntryCMergePaths(BlockArgument startChannel) {
  struct PartialPath {
    std::vector<EffectiveSlotNamer> slots;
    mlir::Value cur;
  };
  std::vector<EntryCMergePath> ret;
  std::vector<PartialPath> stack;
  PartialPath start = {
      .slots = {},
      .cur = startChannel,
  };
  // TODO: Make this an entry slot (which is in a different PR)
  Operation *op = startChannel.getOwner()->getParentOp();
  auto nameAttr =
      op->getAttrOfType<ArrayAttr>("argNames")[startChannel.getArgNumber()];
  std::string name = dyn_cast<StringAttr>(nameAttr).str();
  llvm::errs() << name << "\n";
  start.slots.emplace_back(
      std::make_unique<BufferSlotFullNamer>(name, "valid", "", 0));
  stack.push_back(start);
  while (!stack.empty()) {
    PartialPath path = std::move(stack.back());
    stack.pop_back();

    Operation *next = path.cur.getUses().begin()->getOwner();
    if (auto cmerge = dyn_cast<ControlMergeOp>(next)) {
      int32_t entry;
      for (auto [i, input] : llvm::enumerate(cmerge.getDataOperands())) {
        if (input == path.cur) {
          entry = i;
        }
      }
      EntryCMergePath retPath = {
          .slots = path.slots,
          .cmerge = cmerge,
          .entryValue = entry,
      };
      ret.push_back(retPath);
    }
    if (auto buffer = dyn_cast<BufferOp>(next)) {
      for (auto &slot : buffer.getInternalSlotStateNamers()) {
        path.slots.emplace_back(std::make_unique<BufferSlotFullNamer>(slot));
      }
      path.cur = buffer.getResult();
      stack.push_back(std::move(path));
      continue;
    }
    if (auto fork = dyn_cast<ForkOp>(next)) {
      auto sents = fork.getInternalSentStateNamers();
      for (auto [i, channel] : llvm::enumerate(next->getResults())) {
        PartialPath nextPath = {
            .slots = path.slots,
            .cur = channel,
        };
        assert(!nextPath.slots.empty());
        EffectiveSlotNamer &back = nextPath.slots.back();
        back.copiedSents.push_back(sents[i]);
        stack.push_back(nextPath);
      }
      continue;
    }
  }
  return ret;
}

std::vector<std::vector<EffectiveSlotNamer>>
findCMergeMuxPaths(ControlMergeOp cmerge) {
  struct PartialPath {
    std::vector<EffectiveSlotNamer> slots;
    mlir::Value cur;
  };
  std::vector<std::vector<EffectiveSlotNamer>> ret{};
  std::vector<PartialPath> stack;
  EffectiveSlotNamer mergeSlot(std::make_unique<BufferSlotFullNamer>(
      cmerge.getInternalSlotStateNamers()[0]));
  PartialPath start = {
      .slots = {},
      .cur = cmerge.getIndex(),
  };
  start.slots.push_back(mergeSlot);
  stack.push_back(start);
  while (!stack.empty()) {
    PartialPath path = stack.back();
    stack.pop_back();

    Operation *next = path.cur.getUses().begin()->getOwner();
    if (auto mux = dyn_cast<MuxOp>(next)) {
      // Path is terminated by MuxOp, so this is the end of the path
      ret.push_back(std::move(path.slots));
      continue;
    }
    if (auto buffer = dyn_cast<BufferOp>(next)) {
      for (auto &slot : buffer.getInternalSlotStateNamers()) {
        path.slots.emplace_back(std::make_unique<BufferSlotFullNamer>(slot));
      }
      path.cur = buffer.getResult();
      stack.push_back(std::move(path));
      continue;
    }
    if (auto fork = dyn_cast<ForkOp>(next)) {
      auto sents = fork.getInternalSentStateNamers();
      for (auto [i, channel] : llvm::enumerate(next->getResults())) {
        PartialPath nextPath = {
            .slots = path.slots,
            .cur = channel,
        };
        EffectiveSlotNamer &back = nextPath.slots.back();
        back.copiedSents.push_back(sents[i]);
        stack.push_back(nextPath);
      }
      continue;
    }

    llvm::report_fatal_error("unexpected op detected");
  }
  return ret;
}
} // namespace

LogicalResult HandshakeAnnotatePropertiesPass::annotateEntryTokenOrderPaths(
    ControlMergeOp cmerge, int32_t entryValue) {
  for (const auto &path : findCMergeMuxPaths(cmerge)) {
    if (path.size() < 2) {
      // The regex of this invariant trivially holds for any path of length 1
      continue;
    }
    EntryTokenOrder p(uid++, FormalProperty::TAG::INVAR, path, entryValue);
    propertyTable.push_back(p.toJSON());
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateEntryTokenOrder(ModuleOp modOp) {
  for (auto funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (BlockArgument arg : funcOp.getRegion().getArguments()) {
      for (const auto &path : findEntryCMergePaths(arg)) {
        if (failed(
                annotateEntryTokenOrderPaths(path.cmerge, path.entryValue))) {
          return failure();
        }
      }
    }
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateSingleEntryToken(ModuleOp modOp) {
  for (auto funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (BlockArgument arg : funcOp.getRegion().getArguments()) {
      for (const auto &ec : findEntryCMergePaths(arg)) {
        for (const auto &cm : findCMergeMuxPaths(ec.cmerge)) {
          SingleEntryToken p(uid++, FormalProperty::TAG::INVAR, ec.slots, cm);
          propertyTable.push_back(p.toJSON());
        }
      }
    }
  }
  return success();
}
struct BranchOpDecision {
  // true -> this branch loops towards itself
  // false -> this branch exits
  bool trueLoop;
  bool falseLoop;

  inline bool isDecider() { return trueLoop != falseLoop; }
};

bool reachable(mlir::Value start, Operation *target) {
  llvm::DenseSet<Operation *> visited;
  std::vector<mlir::Value> stack;
  stack.push_back(start);
  while (!stack.empty()) {
    Operation *op = stack.back().getUses().begin()->getOwner();
    stack.pop_back();
    if (visited.contains(op)) {
      continue;
    }
    visited.insert(op);
    if (op == target)
      return true;
    if (isa<ArithOpInterface, ConditionalBranchOp, ForkOp, BufferOp, LoadOp,
            MuxOp, MergeOp, ControlMergeOp>(op)) {
      for (auto res : op->getResults()) {
        stack.push_back(res);
      }
    }
  }
  return false;
}

BranchOpDecision findBranchLoops(ConditionalBranchOp branch) {
  BranchOpDecision ret;
  ret.trueLoop = reachable(branch.getTrueResult(), branch);
  ret.falseLoop = reachable(branch.getFalseResult(), branch);
  return ret;
}

std::vector<EffectiveSlotNamer>
getConditionHolders(ConditionalBranchOp branch) {
  std::vector<mlir::Value> stack;
  stack.push_back(branch.getConditionOperand());
  std::vector<EagerForkSentNamer> tailingSents;
  std::vector<EffectiveSlotNamer> backPath;
  while (!stack.empty()) {
    mlir::Value cur = stack.back();
    stack.pop_back();
    Operation *op = cur.getDefiningOp();

    if (!op) {
      continue;
    }
    if (auto forkOp = dyn_cast<ForkOp>(op)) {
      auto sents = forkOp.getInternalSentStateNamers();
      for (auto [i, chan] : llvm::enumerate(forkOp.getResults())) {
        if (chan == cur) {
          tailingSents.push_back(sents[i]);
          break;
        }
      }
      stack.push_back(forkOp.getOperand());
    }
    if (auto buffer = dyn_cast<BufferOp>(op)) {
      std::vector<std::unique_ptr<InternalStateNamer>> slots =
          getAllSlotsOfOperation(buffer);
      EffectiveSlotNamer last =
          EffectiveSlotNamer(std::move(slots.back()), std::move(tailingSents));
      tailingSents = std::vector<EagerForkSentNamer>();
      backPath.push_back(last);
      for (int i = slots.size() - 2; i >= 0; --i) {
        backPath.emplace_back(std::move(slots[i]));
      }
      stack.push_back(buffer.getOperand());
    }

    if (auto loadOp = dyn_cast<LoadOp>(op)) {
      auto slots = getAllSlotsOfOperation(loadOp);
      backPath.emplace_back(std::move(slots[slots.size() - 1]),
                            std::move(tailingSents));
    }
  }
  // Reverse path so the order corresponds to the circuit order
  std::reverse(backPath.begin(), backPath.end());
  return backPath;
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateExitTokenOrder(ModuleOp modOp) {
  for (auto funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (auto branch = dyn_cast<ConditionalBranchOp>(op)) {
        auto dec = findBranchLoops(branch);
        if (!dec.isDecider()) {
          // Either both branches exit and this invariant is handled by 6
          // (assuming this branch is part of an IOG), or both branches loop and
          // this invariant cannot say anything
          continue;
        }
        // If trueLoop: false output exits => exitValue = 0
        int32_t exitValue = dec.trueLoop ? 0 : 1;
        auto slots = getConditionHolders(branch);
        ExitTokenOrder p(uid++, FormalProperty::TAG::INVAR, slots, exitValue);
      }
    }
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateProperty(ModuleOp modOp,
                                                  FormalProperty::TYPE t) {
  switch (t) {
  case FormalProperty::TYPE::AbsenceOfBackpressure:
    return annotateAbsenceOfBackpressure(modOp);
  case FormalProperty::TYPE::ValidEquivalence:
    return annotateValidEquivalence(modOp);
  case FormalProperty::TYPE::EagerForkNotAllOutputSent:
    return annotateEagerForkNotAllOutputSent(modOp);
  case FormalProperty::TYPE::CopiedSlotsOfActiveForksAreFull:
    return annotateCopiedSlotsOfAllForks(modOp);
  case FormalProperty::TYPE::ReconvergentPathFlow:
    return annotateReconvergentPathFlow(modOp);
  case FormalProperty::TYPE::EntryTokenOrder:
    return annotateEntryTokenOrder(modOp);
  case FormalProperty::TYPE::SingleEntryToken:
    return annotateSingleEntryToken(modOp);
  }
  return failure();
}
LogicalResult HandshakeAnnotatePropertiesPass::annotateQueriedProperties() {
  ModuleOp modOp = getOperation();
  LogicalResult res = success();
  if (annotateList != "") {
    for (auto &elem : llvm::split(annotateList, ',')) {
      std::string typeStr = elem.trim().str();
      if (auto t = FormalProperty::typeFromStr(typeStr)) {
        if (failed(annotateProperty(modOp, *t)))
          res = failure();
      } else {
        llvm::errs() << typeStr << " is not a property\n";
        res = failure();
      }
    }
    return res;
  }
  if (annotateProperties) {
    if (failed(annotateAbsenceOfBackpressure(modOp)))
      return failure();
    if (failed(annotateValidEquivalence(modOp)))
      return failure();
  }
  if (annotateInvariants) {
    if (failed(annotateEagerForkNotAllOutputSent(modOp)))
      return failure();
    if (failed(annotateCopiedSlotsOfAllForks(modOp)))
      return failure();
    if (failed(annotateReconvergentPathFlow(modOp)))
      return failure();
    if (failed(annotateEntryTokenOrder(modOp)))
      return failure();
    if (failed(annotateSingleEntryToken(modOp)))
      return failure();
  }
  return success();
}

void HandshakeAnnotatePropertiesPass::runDynamaticPass() {
  if (failed(annotateQueriedProperties())) {
    return signalPassFailure();
  }

  llvm::json::Value jsonVal(std::move(propertyTable));

  std::error_code EC;
  llvm::raw_fd_ostream jsonOut(jsonPath, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return;

  jsonOut << formatv("{0:2}", jsonVal);
}
