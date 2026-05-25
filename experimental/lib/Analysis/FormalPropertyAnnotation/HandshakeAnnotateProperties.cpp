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
#include "experimental/Support/IOG.h"
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

  LogicalResult annotateProperty(ModuleOp modOp, const std::vector<IOG> &iogs,
                                 FormalProperty::TYPE t);
  LogicalResult annotateQueriedProperties(const std::vector<IOG> &iogs);
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
  LogicalResult annotateEagerForkPath(ModuleOp modOp);
  LogicalResult annotateReconvergentPathFlow(ModuleOp modOp);
  LogicalResult annotateIOGSingleToken(const IOG &iog);
  LogicalResult annotateIOGConsecutiveTokens(const IOG &iog);
  LogicalResult annotateEntryTokenOrderPaths(ControlMergeOp cmerge,
                                             int32_t entryValue);
  LogicalResult annotateEntryTokenOrder(ModuleOp modOp);
  LogicalResult annotateSingleEntryToken(ModuleOp modOp);
  LogicalResult annotateExitTokenOrder(const std::vector<IOG> &iog);
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
HandshakeAnnotatePropertiesPass::annotateEagerForkPath(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (auto forkOp = dyn_cast<ForkOp>(op)) {
        EagerForkPathTokenCopiedMaximumOnce p(uid, FormalProperty::TAG::INVAR,
                                              forkOp);

        propertyTable.push_back(p.toJSON());
        uid++;
      }
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
// This function finds appropriate fork sent state namers for the consecutive
// tokens invariant: Given the IOG, a starting slot, and an ending slot, it
// determines all forks for which:
// 1. The start buffer is the copied slot of the fork
// 2. The fork is part of a path from start to end along the IOG
std::vector<EagerForkSentNamer>
findCopiedSents(const IOG &iog, Operation *startSlot, Operation *endSlot) {
  auto pathSet = iog.findAllPaths(startSlot, endSlot);
  std::vector<EagerForkSentNamer> sents;
  std::vector<Operation *> stack;
  stack.push_back(pathSet.start);
  bool first = true;
  while (!stack.empty()) {
    Operation *cur = stack.back();
    stack.pop_back();
    // Skip the check for slots for the first operation, as the first operation
    // contains the starting slot but should not terminate the search
    if (!first) {
      auto slots = getAllSlotsOfOperation(cur);
      if (!slots.empty()) {
        // Stop following this path if it contains a buffer, as following forks
        // will not have the start buffer as copied slot
        continue;
      }
    }
    first = false;
    for (OpResult forward : cur->getResults()) {
      if (!iog.contains(forward)) {
        continue;
      }
      Operation *next = *forward.getUsers().begin();
      assert(iog.contains(next));
      if (pathSet.units.find(next) == pathSet.units.end()) {
        continue;
      }

      if (auto forkOp = dyn_cast<EagerForkLikeOpInterface>(cur)) {
        sents.push_back(
            forkOp.getInternalSentStateNamers()[forward.getResultNumber()]);
      }

      stack.push_back(next);
    }
  }
  return sents;
}
} // namespace

LogicalResult
HandshakeAnnotatePropertiesPass::annotateIOGSingleToken(const IOG &iog) {
  std::vector<std::unique_ptr<InternalStateNamer>> slots(0);

  // We model the entry node as a buffer that initially has one token.
  slots.push_back(std::make_unique<EntrySlotNamer>(iog.entry));
  std::vector<EagerForkSentNamer> forks(0);
  // Collecting the slots and sents inside the IOG. The invariant relation of
  // num(slots) = 1 + num(eager fork sents)
  for (auto &op : iog.units) {
    for (auto &slot : getAllSlotsOfOperation(op)) {
      slots.push_back(std::move(slot));
    }
    if (auto forkOp = dyn_cast<EagerForkLikeOpInterface>(op)) {
      auto forkSlots = forkOp.getInternalSentStateNamers();
      int count = 0;
      for (auto [i, channel] : llvm::enumerate(forkOp->getResults())) {
        if (iog.contains(channel)) {
          forks.push_back(forkSlots[i]);
          count += 1;
        }
      }
      assert(count == 1);
    }
  }
  auto p = IOGSingleToken(uid, FormalProperty::TAG::INVAR, std::move(slots),
                          std::move(forks));
  uid++;
  propertyTable.push_back(p.toJSON());
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateIOGConsecutiveTokens(const IOG &iog) {
  std::vector<std::pair<Operation *, TokenCountNamer>> slotOps;
  for (auto &op : iog.units) {
    auto slotCountNamer = getTokenCountNamerOfOperation(op);
    if (slotCountNamer.has_value()) {
      slotOps.push_back({op, std::move(*slotCountNamer)});
    }
  }
  for (auto slot1 = slotOps.begin(); slot1 != slotOps.end(); ++slot1) {
    for (auto slot2 = slot1 + 1; slot2 != slotOps.end(); ++slot2) {
      if (slot1->first == slot2->first) {
        // TODO: Handle loops, i.e. if the slot contains >=2 tokens, there
        // should be a copied fork within a loop
        if (slot1->second.slots->size() >= 2) {
          slot1->first->emitWarning("Should annotate self-loop");
        }
        continue;
      }

      std::vector<EagerForkSentNamer> copiedSents =
          findCopiedSents(iog, slot1->first, slot2->first);

      std::vector<EagerForkSentNamer> extra =
          findCopiedSents(iog, slot2->first, slot1->first);

      copiedSents.insert(copiedSents.end(), extra.begin(), extra.end());

      // Note:
      // Even if the copiedSents is empty, this invariant is interesting! It
      // means that both slots cannot be occupied at the same time, as there is
      // only (at most) one token in the IOG

      auto p = IOGConsecutiveTokens(uid, FormalProperty::TAG::INVAR,
                                    slot1->second, slot2->second, copiedSents);
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

// This function is used to find entry control merges (i.e. control merges with
// one input coming from an entry node), and returns the effective slots along
// the path. It is used in the annotation of the entry token order invariant and
// single entry token invariant
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
  start.slots.emplace_back(std::make_unique<EntrySlotNamer>(startChannel));
  stack.push_back(start);
  while (!stack.empty()) {
    PartialPath path = std::move(stack.back());
    stack.pop_back();

    Operation *next = *path.cur.getUsers().begin();
    if (auto cmerge = dyn_cast<ControlMergeOp>(next)) {
      // Path is terminated by ControlMergeOp, so this is the end of the path

      // CMerge uses the following logic to generate index token:
      // - When a CMerge receives a token from the i-th data input channel, it
      // sends a token with value i to the index channel Here, if the path to
      // CMerge ends at the 0-th channel, then an entry token will cause the
      // CMerge to emit a token carrying a value = 0 to the index channel. This
      // function determines the actual entry value of the index channel.
      auto getEntryValue = [&](ControlMergeOp cmerge, Value entryChannel) {
        for (auto [i, input] : llvm::enumerate(cmerge.getDataOperands())) {
          if (input == entryChannel) {
            return (int32_t)i;
          }
        }
        llvm::report_fatal_error(
            "entryChannel is not a data operand of cmerge");
      };
      EntryCMergePath retPath = {
          .slots = path.slots,
          .cmerge = cmerge,
          .entryValue = getEntryValue(cmerge, path.cur),
      };
      ret.push_back(retPath);
    } else if (auto buffer = dyn_cast<BufferOp>(next)) {
      // Add the slots of this buffer to the list of effective slot (copied
      // sents will be added later)
      for (auto &slot : buffer.getInternalSlotStateNamers()) {
        path.slots.emplace_back(std::make_unique<BufferSlotFullNamer>(slot));
      }
      path.cur = buffer.getResult();
      stack.push_back(std::move(path));
    } else if (auto fork = dyn_cast<ForkOp>(next)) {
      // Branch into multiple paths, and add the sent state of the selected
      // channel as a copied sent for the last slot
      // Note: This last slot always exists, as the entry contains a
      // slot
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
    }
  }
  return ret;
}

// This function finds any path from a control merge to a mux operation. Note
// that there can be multiple paths due to forks that replicate the index token
// of the CMerge to multiple muxes.
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

    Operation *next = *path.cur.getUsers().begin();
    if (auto mux = dyn_cast<MuxOp>(next)) {
      // Path is terminated by MuxOp, so this is the end of the path
      ret.push_back(std::move(path.slots));
    } else if (auto buffer = dyn_cast<BufferOp>(next)) {
      // Add the slots of this buffer to the list of effective slot (copied
      // sents will be added later)
      for (auto &slot : buffer.getInternalSlotStateNamers()) {
        path.slots.emplace_back(std::make_unique<BufferSlotFullNamer>(slot));
      }
      path.cur = buffer.getResult();
      stack.push_back(std::move(path));
    } else if (auto fork = dyn_cast<ForkOp>(next)) {
      // Branch into multiple paths, and add the sent state of the selected
      // channel as a copied sent for the last slot
      // Note: This last slot always exists, as the initial control merge
      // contains a slot
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
    } else {
      llvm::report_fatal_error("unexpected op detected");
    }
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

namespace {
// This function returns the `decider` operation of a branch, meaning the
// operation that emits the final condition token used by the branch.
std::optional<Operation *> getDecider(ConditionalBranchOp branch) {
  mlir::Value next = branch.getConditionOperand();
  while (true) {
    Operation *op = next.getDefiningOp();

    if (!op)
      return std::nullopt;

    // Forks and buffers only propagate the condition token without modifying it
    if (auto forkOp = dyn_cast<ForkOp>(op)) {
      next = forkOp.getOperand();
    } else if (auto buffer = dyn_cast<BufferOp>(op)) {
      next = buffer.getOperand();
    } else {
      return op;
    }
  }
}

// Paths from decider to the branch might contain a fork before they contain a
// slot. If this is the case, these forks cannot be added as copied sents of any
// effective slot. However, in the path annotation from ancestor to branch,
// these copied sents are necessary for the path of effective slots. The
// UnstartedPath struct represents this partial path for this usage. It stores
// the initial sents along the path `startSents`, then the effective slots of
// the decider->branch path from the first slot onwards `started`, and finally
// the branch operation that terminates the path. See the comment before the
// declaration of ExitTokenNoAncestors for a visual explanation
struct UnstartedPath {
  std::vector<EffectiveSlotNamer> started;
  std::vector<EagerForkSentNamer> startSents;
  ConditionalBranchOp end;
};

std::vector<UnstartedPath> findDeciderBranchPaths(Operation *dec) {
  struct PartialPath {
    std::vector<EffectiveSlotNamer> slots;
    std::vector<EagerForkSentNamer> startSents;
    mlir::Value cur;
  };
  std::vector<UnstartedPath> ret{};
  std::vector<PartialPath> stack;
  mlir::Value startChannel;
  if (auto arith = dyn_cast<ArithOpInterface>(dec)) {
    startChannel = arith->getResult(0);
  } else if (auto load = dyn_cast<LoadOp>(dec)) {
    load->emitError("TODO: Handle load decider");
    llvm::report_fatal_error("");
  } else {
    load->emitError("Unexpected decider type");
    llvm::report_fatal_error("");
  }
  PartialPath start = {
      .slots = {},
      .startSents = {},
      .cur = startChannel,
  };
  stack.push_back(start);
  while (!stack.empty()) {
    PartialPath path = stack.back();
    stack.pop_back();

    Operation *next = *path.cur.getUsers().begin();
    if (auto fork = dyn_cast<ForkOp>(next)) {
      auto sents = fork.getInternalSentStateNamers();
      for (auto [i, channel] : llvm::enumerate(next->getResults())) {
        PartialPath nextPath = {
            .slots = path.slots,
            .startSents = path.startSents,
            .cur = channel,
        };
        if (!nextPath.slots.empty()) {
          EffectiveSlotNamer &back = nextPath.slots.back();
          back.copiedSents.push_back(sents[i]);
        } else {
          path.startSents.push_back(sents[i]);
        }
        stack.push_back(nextPath);
      }
    } else if (auto buffer = dyn_cast<BufferOp>(next)) {
      for (auto &slot : buffer.getInternalSlotStateNamers()) {
        path.slots.emplace_back(std::make_unique<BufferSlotFullNamer>(slot));
      }
      path.cur = buffer.getResult();
      stack.push_back(std::move(path));
    } else if (auto branch = dyn_cast<ConditionalBranchOp>(next)) {
      // Path is terminated by ConditionalBranchOp, so this is the end of the
      // path
      ret.push_back(
          {std::move(path.slots), std::move(path.startSents), branch});
    } else {
      llvm::report_fatal_error("unexpected op detected");
    }
  }
  return ret;
}

// This function finds the ancestor slots of the decider as shown in the diagram
// before the ExitTokenNoAncestors declaration. The reason the exit branch is
// required for this search is so that the search does not go back past this
// branch and overlap with the decider-branch path
std::vector<std::vector<EffectiveSlotNamer>>
findAncestorSlots(const IOG &iog, Operation *dec, ConditionalBranchOp exit) {
  struct PartialPath {
    mlir::Value channel;
    // `back` means the paths are annotated backwards, meaning the paths flow in
    // the direction of operands (rather than results)
    std::vector<EffectiveSlotNamer> backPath;
    // `backSents` stores eager fork sent states that should be added to the
    // next fork found along this path, in backwards order in comparison to the
    // direction tokens flow
    std::vector<EagerForkSentNamer> backSents;
    llvm::DenseSet<Operation *> visited;
  };
  std::vector<PartialPath> stack;
  std::vector<EffectiveSlotNamer> startSlots;
  PartialPath start{
      .channel = *dec->getOperands().begin(),
      .backPath = {},
      .backSents = {},
      .visited = {},
  };
  std::vector<std::vector<EffectiveSlotNamer>> ret;
  stack.push_back(std::move(start));
  while (!stack.empty()) {
    // In this function, the operations are not handled case-by-case, but
    // instead using their interfaces as any operation is allowed here. The
    // order cannot be changed without breaking the behaviour:
    // First, terminate in bad cases (otherwise potentially too many things are
    // annotated).
    // Second, if applicable, an eager fork output is added as a
    // copied sent to the next slot.
    // Third, slots are annotated.
    // Lastly, all inputs are followed backwards (regardless of the number of
    // inputs, can be 1 for buffer or fork, or 2 for arithmetic operations or
    // loads)
    auto cur = std::move(stack.back());
    stack.pop_back();

    if (!iog.contains(cur.channel)) {
      assert(false && "should not happen");
      continue;
    }

    if (cur.channel == iog.entry) {
      EffectiveSlotNamer es(std::make_unique<EntrySlotNamer>(iog.entry));
      while (!cur.backSents.empty()) {
        es.addCopiedSent(cur.backSents.back());
        cur.backSents.pop_back();
      }
      cur.backPath.push_back(std::move(es));
    }

    Operation *op = cur.channel.getDefiningOp();
    // The search for ancestor slots is terminated by any of the following:
    // 1. No defining op
    // 2. op is not part of the IOG (ancestors must be part of the IOG)
    // 3. The operation is the exit branch (Any operation before this is part of
    // the decider-branch path, and is therefore not an ancestor of those slots)
    // 4. op has already been visited (to avoid loops)
    if (!op || !iog.contains(op) || op == exit ||
        cur.visited.find(op) != cur.visited.end()) {
      std::vector<EffectiveSlotNamer> forwardPath{};
      while (!cur.backPath.empty()) {
        forwardPath.push_back(std::move(cur.backPath.back()));
        cur.backPath.pop_back();
      }
      ret.push_back(std::move(forwardPath));
      continue;
    }
    cur.visited.insert(op);

    if (auto fork = dyn_cast<EagerForkLikeOpInterface>(op)) {
      // Store the appropriate sent state as a copied sent for the next slot
      // found
      auto sents = fork.getInternalSentStateNamers();
      for (auto [i, chan] : llvm::enumerate(fork->getResults())) {
        if (chan == cur.channel) {
          cur.backSents.push_back(sents[i]);
        }
      }
    }
    auto curSlots = getAllSlotsOfOperation(op);
    // Annotate the slots backwards, using the stored copied sents to build
    // effective slots.
    for (int i = curSlots.size() - 1; i >= 0; --i) {
      auto &slot = curSlots[i];
      EffectiveSlotNamer es(std::move(slot));
      while (!cur.backSents.empty()) {
        es.addCopiedSent(cur.backSents.back());
        cur.backSents.pop_back();
      }
      cur.backPath.push_back(std::move(es));
    }

    // Follow the inputs of the operand
    for (auto back : op->getOperands()) {
      if (!iog.contains(back)) {
        continue;
      }
      cur.channel = back;
      stack.push_back(cur);
    }
  }
  return ret;
}

} // namespace

LogicalResult HandshakeAnnotatePropertiesPass::annotateExitTokenOrder(
    const std::vector<IOG> &iogs) {
  llvm::DenseMap<ConditionalBranchOp, int32_t> exitBranches;
  for (const auto &iog : iogs) {
    for (auto *op : iog.units) {
      if (auto branch = dyn_cast<ConditionalBranchOp>(op)) {
        if (auto optExitValue = iog.isExitBranch(branch)) {
          exitBranches.insert({branch, *optExitValue});
        }
      }
    }
  }
  llvm::DenseMap<Operation *, int32_t> deciders;
  for (auto [branch, exitValue] : exitBranches) {
    Operation *dec = *getDecider(branch);
    deciders.insert({dec, exitValue});
  }
  for (auto [dec, exitValue] : deciders) {
    auto slotPaths = findDeciderBranchPaths(dec);
    // [START Enumerating ExitTokenOrder invariants]
    for (const auto &slots : slotPaths) {
      if (slots.started.size() < 2) {
        // Invariant is trivially true
        continue;
      }
      ExitTokenOrder p(uid++, FormalProperty::TAG::INVAR, slots.started,
                       exitValue);
      propertyTable.push_back(p.toJSON());
    }
    // [END Enumerating ExitTokenOrder invariants]

    // [START Enumerating ExitTokenNoAncestors invariants]
    for (const auto &iog : iogs) {
      if (!iog.contains(dec)) {
        continue;
      }

      // Find the branch that is part of the IOG of the decider
      std::optional<ConditionalBranchOp> iogBranch;
      for (const auto &slots : slotPaths) {
        if (iog.contains(slots.end)) {
          assert(!iogBranch);
          iogBranch = slots.end;
        }
      }
      if (!iogBranch) {
        return failure();
      }

      auto ancestorPaths = findAncestorSlots(iog, dec, *iogBranch);
      for (auto ancestorPath : ancestorPaths) {
        if (ancestorPath.empty())
          continue;
        for (const auto &decPath : slotPaths) {
          if (decPath.started.empty())
            continue;
          // Extend the ancestor path using the startSents
          for (const auto &startSent : decPath.startSents) {
            ancestorPath.back().addCopiedSent(startSent);
          }
          std::vector<std::shared_ptr<InternalStateNamer>> exitSlots;
          exitSlots.reserve(decPath.started.size());
          for (const auto &es : decPath.started) {
            exitSlots.push_back(es.slot);
          }
          ExitTokenNoAncestors p(uid++, FormalProperty::TAG::INVAR, exitSlots,
                                 ancestorPath, exitValue);
          propertyTable.push_back(p.toJSON());
        }
      }
    }
    // [END Enumerating ExitTokenNoAncestors invariants]
  }
  return success();
}

LogicalResult HandshakeAnnotatePropertiesPass::annotateProperty(
    ModuleOp modOp, const std::vector<IOG> &iogs, FormalProperty::TYPE t) {
  switch (t) {
  case FormalProperty::TYPE::AbsenceOfBackpressure:
    return annotateAbsenceOfBackpressure(modOp);
  case FormalProperty::TYPE::ValidEquivalence:
    return annotateValidEquivalence(modOp);
  case FormalProperty::TYPE::EagerForkNotAllOutputSent:
    return annotateEagerForkNotAllOutputSent(modOp);
  case FormalProperty::TYPE::CopiedSlotsOfActiveForksAreFull:
    return annotateCopiedSlotsOfAllForks(modOp);
  case FormalProperty::TYPE::EagerForkPathTokenCopiedMaximumOnce:
    return annotateEagerForkPath(modOp);
  case FormalProperty::TYPE::ReconvergentPathFlow:
    return annotateReconvergentPathFlow(modOp);
  case FormalProperty::TYPE::IOGSingleToken:
    for (const auto &iog : iogs) {
      if (failed(annotateIOGSingleToken(iog)))
        return failure();
    }
    return success();
  case FormalProperty::TYPE::IOGConsecutiveTokens:
    for (const auto &iog : iogs) {
      if (failed(annotateIOGConsecutiveTokens(iog)))
        return failure();
    }
    return success();
  case FormalProperty::TYPE::EntryTokenOrder:
    return annotateEntryTokenOrder(modOp);
  case FormalProperty::TYPE::SingleEntryToken:
    return annotateSingleEntryToken(modOp);
  case FormalProperty::TYPE::ExitTokenOrder:
    return annotateExitTokenOrder(iogs);
  case FormalProperty::TYPE::ExitTokenNoAncestors:
    llvm::errs() << "ExitTokenNoAncestors is annotated with ExitTokenOrder\n";
    return failure();
  }
  return failure();
}

LogicalResult HandshakeAnnotatePropertiesPass::annotateQueriedProperties(
    const std::vector<IOG> &iogs) {
  ModuleOp modOp = getOperation();
  LogicalResult res = success();
  if (annotateList != "") {
    for (auto &elem : llvm::split(annotateList, ',')) {
      std::string typeStr = elem.trim().str();
      if (auto t = FormalProperty::typeFromStr(typeStr)) {
        if (failed(annotateProperty(modOp, iogs, *t)))
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
    if (failed(annotateEagerForkPath(modOp)))
      return failure();
    if (failed(annotateReconvergentPathFlow(modOp)))
      return failure();

    for (const auto &iog : iogs) {
      if (failed(annotateIOGSingleToken(iog)))
        return failure();
      if (failed(annotateIOGConsecutiveTokens(iog)))
        return failure();
    }
    if (failed(annotateEntryTokenOrder(modOp)))
      return failure();
    if (failed(annotateSingleEntryToken(modOp)))
      return failure();
    if (failed(annotateExitTokenOrder(iogs)))
      return failure();
  }
  return success();
}

void HandshakeAnnotatePropertiesPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();
  auto iogs = findAllIOGs(modOp);
  llvm::DenseSet<SinkOp> sinks;
  for (auto &iog : iogs) {
    for (Operation *unit : iog.units) {
      if (auto sinkOp = dyn_cast<SinkOp>(unit)) {
        sinks.insert(sinkOp);
      }
    }
  }

  for (SinkOp sink : sinks) {
    OpBuilder builder(sink);
    DeadBufferOp deadBuffer =
        builder.create<DeadBufferOp>(sink.getLoc(), sink.getOperand());
    sink.erase();
    for (auto &iog : iogs) {
      if (iog.contains(sink)) {
        iog.units.erase(sink);
        iog.units.insert(deadBuffer);
      }
    }
  }
  getAnalysis<NameAnalysis>().nameAllUnnamedOps();

  if (failed(annotateQueriedProperties(iogs))) {
    return signalPassFailure();
  }

  llvm::json::Value jsonVal(std::move(propertyTable));

  std::error_code EC;
  llvm::raw_fd_ostream jsonOut(jsonPath, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return;

  jsonOut << formatv("{0:2}", jsonVal);
}
