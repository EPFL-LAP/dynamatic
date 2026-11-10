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

#include "Utils/Elimination.h"
#include "Utils/FlowExpression.h"
#include "Utils/IOG.h"
#include "Utils/IndexChannelAnalysis.h"
#include "Utils/TraversalUtils.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOpInternalStateNamer.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/CFDFC.h"
#include "experimental/Support/FormalProperty.h"
#include "experimental/Support/PropertyExpr.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <ostream>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

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

  void runOnOperation() override;

private:
  /// Identifier of the next property
  uint64_t uid = 0;
  std::vector<FormalProperty> propertyTable;
  /// The IOGs of the module, computed once for all properties derived from
  /// them
  std::vector<IOG> iogs;

  /// Generates the properties of one or more kinds over the whole module.
  using Generator =
      LogicalResult (HandshakeAnnotatePropertiesPass::*)(ModuleOp);
  struct GeneratorEntry {
    /// The name with which the generator is selected in `annotate-list`
    StringLiteral name;
    Generator generator;
    /// Whether the generator produces invariants (`annotate-invariants`) or
    /// properties to check (`annotate-properties`)
    bool isInvariant;
  };
  static const GeneratorEntry GENERATORS[];

  /// Adds a property to the table under the next identifier.
  void addProperty(StringRef kind, FormalProperty::TAG tag, PropertyExpr expr);

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
  LogicalResult annotateEagerForkPath(ModuleOp modOp);
  LogicalResult annotateDuplicatedTokenLocation(ModuleOp modOp);
  LogicalResult annotateReconvergentPathFlow(ModuleOp modOp);
  LogicalResult annotateIOGSingleToken(ModuleOp modOp);
  LogicalResult annotateIOGConsecutiveTokens(ModuleOp modOp);
  LogicalResult annotateEntryTokenOrderPaths(ControlMergeOp cmerge,
                                             int32_t entryValue);
  LogicalResult annotateEntryTokenOrder(ModuleOp modOp);
  LogicalResult annotateSingleEntryToken(ModuleOp modOp);
  LogicalResult annotateExitTokenOrder(ModuleOp modOp);
};

// The generators in the order in which they run by default. The generator of
// ExitTokenOrder also generates the ExitTokenNoAncestors invariants, as both
// are derived from the same paths.
// clang-format off
const HandshakeAnnotatePropertiesPass::GeneratorEntry
    HandshakeAnnotatePropertiesPass::GENERATORS[] = {
    {FormalProperty::ABSENCE_OF_BACKPRESSURE,                   &HandshakeAnnotatePropertiesPass::annotateAbsenceOfBackpressure,     /*isInvariant=*/false},
    {FormalProperty::VALID_EQUIVALENCE,                         &HandshakeAnnotatePropertiesPass::annotateValidEquivalence,          /*isInvariant=*/false},
    {FormalProperty::EAGER_FORK_NOT_ALL_OUTPUT_SENT,            &HandshakeAnnotatePropertiesPass::annotateEagerForkNotAllOutputSent, /*isInvariant=*/true},
    {FormalProperty::COPIED_SLOTS_OF_ACTIVE_FORKS_ARE_FULL,     &HandshakeAnnotatePropertiesPass::annotateCopiedSlotsOfAllForks,     /*isInvariant=*/true},
    {FormalProperty::EAGER_FORK_PATH_TOKEN_COPIED_MAXIMUM_ONCE, &HandshakeAnnotatePropertiesPass::annotateEagerForkPath,             /*isInvariant=*/true},
    {FormalProperty::DUPLICATED_TOKEN_LOCATION,                 &HandshakeAnnotatePropertiesPass::annotateDuplicatedTokenLocation,   /*isInvariant=*/true},
    {FormalProperty::RECONVERGENT_PATH_FLOW,                    &HandshakeAnnotatePropertiesPass::annotateReconvergentPathFlow,      /*isInvariant=*/true},
    {FormalProperty::IOG_SINGLE_TOKEN,                          &HandshakeAnnotatePropertiesPass::annotateIOGSingleToken,            /*isInvariant=*/true},
    {FormalProperty::IOG_CONSECUTIVE_TOKENS,                    &HandshakeAnnotatePropertiesPass::annotateIOGConsecutiveTokens,      /*isInvariant=*/true},
    {FormalProperty::ENTRY_TOKEN_ORDER,                         &HandshakeAnnotatePropertiesPass::annotateEntryTokenOrder,           /*isInvariant=*/true},
    {FormalProperty::SINGLE_ENTRY_TOKEN,                        &HandshakeAnnotatePropertiesPass::annotateSingleEntryToken,          /*isInvariant=*/true},
    {FormalProperty::EXIT_TOKEN_ORDER,                          &HandshakeAnnotatePropertiesPass::annotateExitTokenOrder,            /*isInvariant=*/true},
};
// clang-format on

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

/// The `kind` signal of the result port of the unit producing `res` that `res`
/// leaves through.
static Signal getResultSignal(Signal::Kind kind, OpResult res) {
  Operation *op = res.getOwner();
  return Signal(kind, getUniqueName(op), Signal::Result, res.getResultNumber(),
                cast<handshake::NamedIOInterface>(op).getResultName(
                    res.getResultNumber()));
}

/// The `kind` signal of the operand port of the unit using `operand` that
/// `operand` enters through.
static Signal getOperandSignal(Signal::Kind kind, OpOperand &operand) {
  Operation *op = operand.getOwner();
  return Signal(kind, getUniqueName(op), Signal::Operand,
                operand.getOperandNumber(),
                cast<handshake::NamedIOInterface>(op).getOperandName(
                    operand.getOperandNumber()));
}

/// State leaves for each of `namers`.
template <typename Namer>
static std::vector<PropertyExpr> getStates(const std::vector<Namer> &namers) {
  std::vector<PropertyExpr> states;
  states.reserve(namers.size());
  for (const Namer &namer : namers)
    states.emplace_back(State(namer));
  return states;
}

void HandshakeAnnotatePropertiesPass::addProperty(StringRef kind,
                                                  FormalProperty::TAG tag,
                                                  PropertyExpr expr) {
  propertyTable.push_back({/*id=*/uid++, /*kind=*/kind.str(), /*tag=*/tag,
                           /*check=*/std::nullopt, /*expr=*/std::move(expr)});
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateValidEquivalenceBetweenOps(
    Operation &op1, Operation &op2) {
  for (auto [i, res1] : llvm::enumerate(op1.getResults()))
    for (auto [j, res2] : llvm::enumerate(op2.getResults())) {
      // equivalence is symmetrical so it needs to be checked only once for
      // each pair of signals when the Ops are the same
      if ((&op1 != &op2 || i < j) && isChannelToBeChecked(res1) &&
          isChannelToBeChecked(res2)) {
        // The two channels are valid in the same cycles, e.g.:
        //   (fork0.outs_0_valid <-> fork0.outs_2_valid)
        PropertyExpr expr =
            BinaryExpr(getResultSignal(Signal::Valid, res1), BinaryExpr::Iff,
                       getResultSignal(Signal::Valid, res2));
        addProperty(FormalProperty::VALID_EQUIVALENCE, FormalProperty::TAG::OPT,
                    expr);
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

          // Whenever the channel is valid, its user is ready, e.g.:
          //   (mux0.outs_valid -> fork2.ins_ready)
          OpOperand &use = *res.getUses().begin();
          PropertyExpr expr = implies(getResultSignal(Signal::Valid, res),
                                      getOperandSignal(Signal::Ready, use));
          addProperty(FormalProperty::ABSENCE_OF_BACKPRESSURE,
                      FormalProperty::TAG::OPT, expr);
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
        // Not all outputs are in the sent state at the same time, e.g.:
        //   (count(fork1.outs_0_sent, fork1.outs_1_sent) < 2)
        std::vector<EagerForkSentNamer> sents =
            forkOp.getInternalSentStateNamers();
        PropertyExpr expr =
            BinaryExpr(NaryExpr(NaryExpr::Count, getStates(sents)),
                       BinaryExpr::Lt, Constant::integer(sents.size()));
        addProperty(FormalProperty::EAGER_FORK_NOT_ALL_OUTPUT_SENT,
                    FormalProperty::TAG::INVAR, expr);
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

  // If this operation contains a slot, the copied slot (i.e., the last slot of
  // the operation) has been found: if any output of the fork is in the sent
  // state, the copied slot holds the token being copied, e.g.:
  //   ((fork4.outs_0_sent | fork4.outs_1_sent) -> buffer8.full)
  auto annotateCopiedSlot = [&](const PropertyExpr &copiedSlot) {
    PropertyExpr anySent =
        disjunction(getStates(originFork.getInternalSentStateNamers()));
    addProperty(FormalProperty::COPIED_SLOTS_OF_ACTIVE_FORKS_ARE_FULL,
                FormalProperty::TAG::INVAR, implies(anySent, copiedSlot));
  };
  if (auto bufferOp = dyn_cast<handshake::BufferLikeOpInterface>(curOp)) {
    annotateCopiedSlot(State(bufferOp.getInternalSlotStateNamers().back()));
    return success();
  }

  // A unit with latency: the copied slot is its last pipeline slot
  if (auto latencyOp = dyn_cast<handshake::LatencyInterface>(curOp)) {
    std::vector<PipelineSlotNamer> slots = latencyOp.getPipelineSlots();
    if (!slots.empty()) {
      annotateCopiedSlot(State(slots.back()));
      return success();
    }
    // Corner case: a unit with a latency of 0 has no pipeline slot, so the
    // walk continues to its operands below, as for the other units without
    // slots
    assert(succeeded(latencyOp.getLatency()) && *latencyOp.getLatency() == 0 &&
           "a unit without pipeline slots must have a latency of 0");
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
        // If any output of the fork is in the sent state, the input of the
        // fork is valid (see the description of the invariant in the
        // documentation), e.g.:
        //   ((fork1.outs_0_sent | fork1.outs_1_sent) -> fork1.ins_valid)
        PropertyExpr anySent =
            disjunction(getStates(forkOp.getInternalSentStateNamers()));
        // A fork has a single operand
        addProperty(FormalProperty::EAGER_FORK_PATH_TOKEN_COPIED_MAXIMUM_ONCE,
                    FormalProperty::TAG::INVAR,
                    implies(anySent, getOperandSignal(Signal::Valid,
                                                      op.getOpOperand(0))));
      }
    }
  }
  return success();
}

namespace {

/// A branch side that the duplicated token took: the branch and the value of
/// its condition (1 for the true side, 0 for the false side).
using BranchSide = std::pair<ConditionalBranchOp, int32_t>;

/// Whether the unit of `use`, whenever it consumes the token of the channel of
/// `use`, puts a token on a channel of `trap`. A token is lost in sinks and
/// stores, and kept forever in dead buffers and at the end of the function.
static bool satisfiesTrapCondition(OpOperand &use,
                                   const llvm::DenseSet<Value> &trap) {
  Operation *op = use.getOwner();
  if (isa<DeadBufferOp, EndOp>(op))
    return true;
  // The address of a load comes back as the loaded data, through the memory
  // controller
  if (auto loadOp = dyn_cast<LoadOp>(op))
    return trap.contains(loadOp.getDataResult());
  if (isa<SinkOp, StoreOp, MemoryControllerOp, LSQOp>(op))
    return false;
  // A fork puts a copy on every output, so one output in the trap suffices
  if (isa<EagerForkLikeOpInterface>(op))
    return llvm::any_of(op->getResults(),
                        [&](Value res) { return trap.contains(res); });
  // Any other unit (e.g., a branch, whose token takes either side) passes the
  // token to one of its outputs
  return op->getNumResults() > 0 &&
         llvm::all_of(op->getResults(),
                      [&](Value res) { return trap.contains(res); });
}

/// Returns the channels of the maximal trap of `funcOp`. This is the greatest
/// fixpoint of `satisfiesTrapCondition`: all channels are assumed to be in the
/// trap, and the ones that violate the condition are removed until nothing
/// changes, so that a cycle without any sink stays in the trap.
static llvm::DenseSet<Value>
findTokenTrappingChannels(handshake::FuncOp funcOp) {
  std::vector<Value> channels;
  for (BlockArgument arg : funcOp.getArguments())
    channels.push_back(arg);
  for (Operation &op : funcOp.getOps())
    for (Value res : op.getResults())
      channels.push_back(res);

  llvm::DenseSet<Value> trap;
  for (Value channel : channels)
    if (channel.hasOneUse())
      trap.insert(channel);

  bool changed = true;
  while (changed) {
    changed = false;
    for (Value channel : channels) {
      if (trap.contains(channel) &&
          !satisfiesTrapCondition(*channel.getUses().begin(), trap)) {
        trap.erase(channel);
        changed = true;
      }
    }
  }
  return trap;
}

/// Whether `op` holds at least one slot.
static bool hasSlots(Operation *op) {
  return !getAllSlotsOfOperation(op).empty();
}

/// The units whose last slot holds the original token while `fork` has sent a
/// copy: the units with slots right before `fork`. Merge-like units end the
/// walk, as the original could be in any of their inputs.
static llvm::SetVector<Operation *> findOriginalUnits(ForkOp fork) {
  llvm::SetVector<Operation *> before =
      reachBackward(SmallVector<Value>(fork->getOperands()), [](Operation *op) {
        return hasSlots(op) || isa<MergeLikeOpInterface>(op);
      });
  return llvm::SetVector<Operation *>(
      llvm::from_range, llvm::make_filter_range(before, hasSlots));
}

/// Returns the condition under which the copy that a fork sent on `copied`
/// went through `side` of a branch, if it can be read from the circuit: the
/// condition of the branch comes from an eager fork that holds a value
/// computed from another output of the fork, through units without slots,
/// e.g., for fork4 -> cmpi0 -> fork5 -> cond_br1:
///   (fork4.outs_1_sent | (fork5.outs_0_sent & (fork5.ins = FALSE)))
static std::optional<PropertyExpr> deriveBranchSideExpr(BranchSide side,
                                                        OpResult copied) {
  auto fork = cast<ForkOp>(copied.getOwner());
  // While the other output of the fork is not sent, the computation still holds
  // the same token, so the condition fork still holds the value with which the
  // branch routed the copy. The condition may pass through buffers after the
  // condition fork: the branch consumes the conditions in order, together with
  // their data, so the condition that routed the copy is the one that the fork
  // sent
  auto [branch, sideValue] = side;
  Value condition = branch.getConditionOperand();
  while (auto bufferOp = dyn_cast_or_null<BufferOp>(condition.getDefiningOp()))
    condition = bufferOp.getOperand();
  auto conditionFork = dyn_cast_or_null<ForkOp>(condition.getDefiningOp());
  if (!conditionFork || conditionFork == fork)
    return std::nullopt;

  // The units that compute the condition, back to the slots and the fork. A
  // slot holds its token until the computation completes
  llvm::SetVector<Operation *> computation =
      reachBackward(SmallVector<Value>(conditionFork->getOperands()),
                    [&](Operation *op) { return op == fork || hasSlots(op); });
  computation.insert(conditionFork);
  // The value of a merge-like unit may come from another input
  if (llvm::any_of(computation, [](Operation *op) {
        return isa<MergeLikeOpInterface>(op) && !hasSlots(op);
      }))
    return std::nullopt;

  // The outputs of the fork from which the condition is computed
  auto forkSents = fork.getInternalSentStateNamers();
  std::vector<PropertyExpr> terms;
  llvm::SmallSet<unsigned, 4> sourceOutputs;
  for (Operation *op : computation) {
    if (op == fork || hasSlots(op))
      continue;
    for (Value operand : op->getOperands()) {
      auto output = dyn_cast<OpResult>(operand);
      if (!output || output.getOwner() != fork || output == copied)
        continue;
      auto [outputIt, isNewOutput] =
          sourceOutputs.insert(output.getResultNumber());
      if (isNewOutput)
        terms.emplace_back(State(forkSents[output.getResultNumber()]));
    }
  }
  if (terms.empty())
    return std::nullopt;
  unsigned conditionOutput = cast<OpResult>(condition).getResultNumber();
  terms.emplace_back(State(
      conditionFork.getInternalSentStateNamers()[conditionOutput], sideValue));
  return disjunction(terms);
}

/// Returns the output of `next` (a fork other than the one that sent the copy)
/// that the copy is followed on, among the outputs in `trap`. A copy on one
/// output suffices, and following a single output keeps the region small:
/// while `next` has not sent on it, the copy is still before `next`. The
/// output that leads back to the fork that sent the copy, i.e., to one of the
/// `reachingFork` units, is preferred, as the copy is blocked there behind the
/// original.
static std::optional<OpResult>
getFollowedOutput(EagerForkLikeOpInterface next,
                  const llvm::DenseSet<Value> &trap,
                  const llvm::SetVector<Operation *> &reachingFork) {
  std::optional<OpResult> followed;
  for (OpResult out : next->getResults()) {
    if (!trap.contains(out))
      continue;
    if (reachingFork.contains(out.getUses().begin()->getOwner()))
      return out;
    if (!followed)
      followed = out;
  }
  return followed;
}

/// Intersects `sides` into the sides that `map` holds for `key`, or inserts
/// them if `key` is new. Returns whether the entry was inserted or shrank.
template <typename MapT, typename KeyT>
static bool intersectInto(MapT &map, KeyT key,
                          const llvm::SetVector<BranchSide> &sides) {
  auto [it, isNewKey] = map.try_emplace(key, sides);
  if (isNewKey)
    return true;
  auto &[knownKey, knownSides] = *it;
  // The intersection can only remove sides
  size_t numKnownSides = knownSides.size();
  llvm::set_intersect(knownSides, sides);
  return knownSides.size() != numKnownSides;
}

/// Returns the units that the copy that `fork` sent on `copied` can reach
/// along the channels accepted by `isFollowed`, without passing `fork`. Each
/// unit comes with the branch sides that the copy took on all paths to it,
/// among those whose condition `deriveBranchSideExpr` can read.
static llvm::MapVector<Operation *, llvm::SetVector<BranchSide>>
reachForwardAndCollectFollowedSides(
    ForkOp fork, OpResult copied, llvm::function_ref<bool(Value)> isFollowed) {
  // The sides of a channel only shrink when it is reached again, so the walk
  // terminates
  llvm::DenseMap<Value, llvm::SetVector<BranchSide>>
      reachedChannelsWithFollowedSides;
  llvm::MapVector<Operation *, llvm::SetVector<BranchSide>>
      reachedUnitsWithFollowedSides;
  // Each channel comes with the sides that the copy took to reach it. An entry
  // can be stale (pushed before the sides of its channel shrank): its sides
  // are then a superset, which the intersections below absorb.
  SmallVector<std::pair<Value, llvm::SetVector<BranchSide>>> channelStack;
  reachedChannelsWithFollowedSides[copied] = {};
  channelStack.push_back({copied, {}});
  while (!channelStack.empty()) {
    auto [reachedChannel, channelSides] = channelStack.pop_back_val();
    Operation *successor = *reachedChannel.getUsers().begin();
    assert(successor != fork &&
           "the walk never follows a channel into the fork that holds the "
           "original");

    // The sides that the copy took on every path to the unit. The
    // intersection drops a side that only some paths took, so the slots of
    // the unit count without its condition (sound, but weaker than the
    // disjunction of the conditions of the paths). The paths differ:
    // - at a merge after an if-else, or at a loop header (entry and back
    //   edge), which receives the copy from either side;
    // - at a join whose inputs come through different control flow;
    // - after either of these, as the unit inherits the sides of its input.
    // A stale entry of `channelStack` changes nothing, as its sides are a
    // superset of the known ones
    intersectInto(reachedUnitsWithFollowedSides, successor, channelSides);

    for (OpResult res : getFlowSuccessors(*reachedChannel.getUses().begin())) {
      // The copy cannot pass the fork that holds the original
      if (!isFollowed(res) || *res.getUsers().begin() == fork)
        continue;
      llvm::SetVector<BranchSide> resSides = channelSides;
      // The copy took this side of the branch. If it entered as the condition,
      // the side condition can only come from another output of `fork` that
      // the computation of the condition consumed, so it holds trivially
      if (auto branch = dyn_cast<ConditionalBranchOp>(successor)) {
        int32_t value = res == branch.getTrueResult() ? 1 : 0;
        if (deriveBranchSideExpr({branch, value}, copied))
          resSides.insert({branch, value});
      }
      if (intersectInto(reachedChannelsWithFollowedSides, res, resSides))
        channelStack.push_back(
            {res, reachedChannelsWithFollowedSides.lookup(res)});
    }
  }
  return reachedUnitsWithFollowedSides;
}

} // namespace

LogicalResult HandshakeAnnotatePropertiesPass::annotateDuplicatedTokenLocation(
    ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    // A trap is a set of channels such that every unit that consumes a token
    // from one of them puts a token on one of them, with a branch passing its
    // token to either side (as in a Petri net). A token on a channel of the
    // trap thus always has a descendant in the trap. The maximal trap is the
    // union of all traps
    llvm::DenseSet<Value> trap = findTokenTrappingChannels(funcOp);
    for (ForkOp fork : funcOp.getOps<ForkOp>()) {
      llvm::SetVector<Operation *> originalUnits = findOriginalUnits(fork);
      auto forkSents = fork.getInternalSentStateNamers();
      llvm::SetVector<Operation *> reachingFork =
          reachBackward(SmallVector<Value>(fork->getOperands()),
                        [](Operation *) { return false; });
      // The channels that the copy is followed on: those of the trap, and only
      // one output of each fork other than `fork`
      auto isFollowed = [&](Value channel) {
        auto next =
            dyn_cast_or_null<EagerForkLikeOpInterface>(channel.getDefiningOp());
        if (!next || next.getOperation() == fork)
          return trap.contains(channel);
        return getFollowedOutput(next, trap, reachingFork) == channel;
      };

      for (OpResult copied : fork->getResults()) {
        // A copy outside the trap can disappear, so it has no location
        if (!trap.contains(copied))
          continue;
        unsigned copiedOutput = copied.getResultNumber();

        llvm::MapVector<Operation *, llvm::SetVector<BranchSide>>
            reachedUnitsWithFollowedSides =
                reachForwardAndCollectFollowedSides(fork, copied, isFollowed);

        // The slots that the copy can be in, except the last slots of the
        // units that hold the original
        size_t numSlots = 0;
        for (auto &[op, sides] : reachedUnitsWithFollowedSides)
          numSlots +=
              getAllSlotsOfOperation(op).size() - originalUnits.contains(op);
        if (numSlots == 0)
          continue;

        // The buffers right after the copied output, which hold the copy with
        // the value of the original. Only 1-bit channels (e.g., decisions) are
        // constrained by value, which takes one term per value.
        llvm::SetVector<Operation *> sameValueBuffers;
        bool isOneBit = false;
        if (auto channelType = dyn_cast<ChannelType>(copied.getType()))
          isOneBit = channelType.getDataBitWidth() == 1;
        for (Value channel = copied; isOneBit && trap.contains(channel);) {
          auto bufferOp =
              dyn_cast<BufferOp>(channel.getUses().begin()->getOwner());
          if (!bufferOp || !sameValueBuffers.insert(bufferOp))
            break;
          channel = bufferOp.getResult();
        }

        // If the fork has sent the copy, a slot that the copy can reach holds
        // a token that is not an original whose copy has moved on, and that
        // has the value of the original if the slot is right after the fork,
        // e.g.:
        //   (fork4.outs_0_sent -> (buffer0.outs_valid_i
        //      | (buffer1.full & !fork2.outs_2_sent
        //         & (fork4.outs_1_sent
        //            | (fork5.outs_0_sent & (fork5.ins = TRUE))))
        //      | ...))
        //   (fork5.outs_0_sent -> (
        //      ((buffer8.full & (buffer8.reg = FALSE))
        //       & (fork5.outs_0_sent & (fork5.ins = FALSE)))
        //      | ((buffer8.full & (buffer8.reg = TRUE))
        //         & (fork5.outs_0_sent & (fork5.ins = TRUE)))
        //      | ...))
        std::vector<PropertyExpr> reachedEffectiveSlots;
        // The effective slots of the units that the copy reaches, with respect
        // to the dataflow subgraph of the followed channels: the last slot of
        // a unit only counts until the forks after it have copied its token on
        // the followed channels
        for (auto &[op, sides] : reachedUnitsWithFollowedSides) {
          // The forks right after the unit, through units without slots, copy
          // the token of its last slot. An eager fork keeps its input until it
          // has sent on all outputs, so once one of them has sent on its
          // followed output, the copy has moved on although the slot is still
          // full
          llvm::SetVector<Operation *> after = reachForward(
              SmallVector<Value>(op->getResults()),
              [&](Operation *next) { return next == fork || hasSlots(next); },
              isFollowed);
          std::vector<EagerForkSentNamer> copiedSents;
          // A unit with slots that is an eager fork (e.g., a control merge)
          // sends the token of its last slot itself
          if (auto unitFork = dyn_cast<EagerForkLikeOpInterface>(op)) {
            auto unitForkSents = unitFork.getInternalSentStateNamers();
            for (OpResult out : op->getResults())
              if (isFollowed(out))
                copiedSents.push_back(unitForkSents[out.getResultNumber()]);
          }
          for (Operation *next : after) {
            // The slots of a control merge come before its outputs, so its
            // sent states say nothing about the token of an earlier unit
            auto nextFork = dyn_cast<ForkOp>(next);
            if (!nextFork || nextFork == fork)
              continue;
            auto nextForkSents = nextFork.getInternalSentStateNamers();
            for (OpResult out : nextFork->getResults())
              if (isFollowed(out))
                copiedSents.push_back(nextForkSents[out.getResultNumber()]);
          }

          std::vector<PropertyExpr> sideConditions;
          for (const BranchSide &side : sides)
            sideConditions.push_back(*deriveBranchSideExpr(side, copied));
          auto slots = getAllSlotsOfOperation(op);
          for (auto [i, slot] : llvm::enumerate(slots)) {
            bool isLast = i + 1 == slots.size();
            if (isLast && originalUnits.contains(op))
              continue;
            // The last slot only counts until the forks after it have copied
            // its token
            std::shared_ptr<InternalStateNamer> slotNamer =
                isLast ? std::make_shared<EffectiveSlotNamer>(std::move(slot),
                                                              copiedSents)
                       : std::shared_ptr<InternalStateNamer>(std::move(slot));
            std::vector<PropertyExpr> terms{State(slotNamer)};
            if (sameValueBuffers.contains(op)) {
              std::vector<PropertyExpr> sameValue;
              for (int32_t value : {0, 1})
                sameValue.push_back(
                    conjunction({State(slotNamer, value),
                                 State(forkSents[copiedOutput], value)}));
              terms = {disjunction(sameValue)};
            }
            llvm::append_range(terms, sideConditions);
            reachedEffectiveSlots.push_back(conjunction(terms));
          }
        }
        addProperty(FormalProperty::DUPLICATED_TOKEN_LOCATION,
                    FormalProperty::TAG::INVAR,
                    implies(State(forkSents[copiedOutput]),
                            disjunction(reachedEffectiveSlots)));

        // The copy entered the buffers last, and nothing enters behind it
        // while the fork holds the original, so the newest token in the
        // buffers is the copy (older tokens can be ahead of it), e.g.:
        //   ((fork5.outs_0_sent & buffer8.full)
        //    -> (((buffer8.reg = FALSE) & (fork5.ins = FALSE))
        //        | ((buffer8.reg = TRUE) & (fork5.ins = TRUE))))
        std::vector<PropertyExpr> newerSlotsEmpty{
            State(forkSents[copiedOutput])};
        for (Operation *bufferOp : sameValueBuffers) {
          for (auto &slot : getAllSlotsOfOperation(bufferOp)) {
            std::shared_ptr<InternalStateNamer> slotNamer = std::move(slot);
            std::vector<PropertyExpr> sameValue;
            for (int32_t value : {0, 1})
              sameValue.push_back(
                  conjunction({State(slotNamer, value),
                               State(forkSents[copiedOutput], value)}));
            std::vector<PropertyExpr> isNewest = newerSlotsEmpty;
            isNewest.push_back(State(slotNamer));
            addProperty(FormalProperty::DUPLICATED_TOKEN_LOCATION,
                        FormalProperty::TAG::INVAR,
                        implies(conjunction(isNewest), disjunction(sameValue)));
            newerSlotsEmpty.push_back(!State(slotNamer));
          }
        }
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

  // Map all variables to a column index, the lambdas first
  FlowSystem indices(extractor.equations);

  // Verify that the registry data structure is correct
  assert(indices.registry.verify());

  // The equations over the internal states only, without the lambdas, which
  // cannot be annotated
  std::vector<SmallVector<int64_t>> implied = eliminateVariables(
      indices.equations, indices.registry.size(), indices.nLambdas);
  for (const SmallVector<int64_t> &row : implied) {
    // The row is an equation over internal states only: the sum of the
    // internal states weighted by the coefficients of the row is 0, e.g.:
    //   (((toint(fork4.outs_0_sent) * 1) + (toint(fork4.outs_1_sent) * -1)
    //     + (toint((fork5.outs_0_sent & (fork5.ins = TRUE))) * -1)) = 0)
    // The terms are in column order, which is the same in every run.
    std::vector<PropertyExpr> terms;
    for (auto [i, coef] : llvm::enumerate(row)) {
      if (coef == 0)
        continue;
      const FlowVariable &var = indices.registry.getVar(indices.nLambdas + i);
      if (var.isConstant()) {
        terms.emplace_back(Constant::integer(coef));
        continue;
      }
      auto *state = std::get_if<FlowInternalState>(&var.variable);
      assert(state && "annotatable flow variable without internal state");
      // A variable that tracks the tokens of one value only counts the state
      // when it holds a token of that value
      std::optional<int32_t> constraint;
      if (var.indexTokenTracker)
        constraint = var.indexTokenTracker->getSingleValue();
      terms.emplace_back(BinaryExpr(
          UnaryExpr(UnaryExpr::ToInt, State(state->namer, constraint)),
          BinaryExpr::Mul, Constant::integer(coef)));
    }
    if (terms.empty())
      continue;
    addProperty(FormalProperty::RECONVERGENT_PATH_FLOW,
                FormalProperty::TAG::INVAR,
                BinaryExpr(sum(terms), BinaryExpr::Eq, Constant::integer(0)));
  }
  return success();
}

namespace {
// A unit of an IOG that holds slots, or the entry of the IOG, which is modeled
// as a slot that initially holds the token of the IOG.
struct IOGSlotHolder {
  // The unit, or null for the entry
  Operation *unit;
  // The channels that the tokens enter through: the operands of the unit, or
  // none for the entry, which nothing in the circuit refills
  SmallVector<Value> inputs;
  // The channels that the tokens leave through: the results of the unit, or
  // the entry itself
  SmallVector<Value> outputs;
  // Whether a token is in the slots
  PropertyExpr occupied;
  // The number of slots
  size_t numSlots;
};

// This function finds appropriate fork outputs for the consecutive tokens
// invariant: Given the IOG, a starting slot holder, and an ending slot holder,
// it determines the outputs in the IOG of all forks for which:
// 1. The start holder holds the copied slot of the fork
// 2. The fork is part of a path from start to end along the IOG
// There is no path into the entry, as nothing produces it.
SmallVector<Value> findCopiedSents(const IOG &iog, const IOGSlotHolder &start,
                                   const IOGSlotHolder &end) {
  // The units that reach the end without going through the start. If the start
  // unit is null, the start is the entry channel: no unit is a boundary, and
  // the walk stops at the entry, which no unit produces
  llvm::SetVector<Operation *> reachingEnd = reachBackward(
      /*starts=*/end.inputs,
      /*isBoundary=*/[&](Operation *op) { return op == start.unit; },
      /*isFollowed=*/[&](Value channel) { return iog.contains(channel); });
  SmallVector<Value> sents;
  SmallVector<Value> stack(start.outputs);
  while (!stack.empty()) {
    Value channel = stack.pop_back_val();
    if (!iog.contains(channel))
      continue;
    Operation *next = *channel.getUsers().begin();
    assert(iog.contains(next));
    if (next != end.unit && !reachingEnd.contains(next))
      continue;
    if (isa_and_nonnull<EagerForkLikeOpInterface>(channel.getDefiningOp()))
      sents.push_back(channel);
    // Stop following this path if it contains a buffer, as following forks
    // will not have the start buffer as copied slot
    if (getAllSlotsOfOperation(next).empty())
      llvm::append_range(stack, next->getResults());
  }
  return sents;
}

// The sent state of a fork output
EagerForkSentNamer getSentNamer(Value forkOutput) {
  auto forkOp = cast<EagerForkLikeOpInterface>(forkOutput.getDefiningOp());
  return forkOp.getInternalSentStateNamers()[cast<OpResult>(forkOutput)
                                                 .getResultNumber()];
}

// Identifies the IOGConsecutiveTokens invariant of two slot holders: it does
// not depend on the IOG, nor on the order of the holders and of the fork
// outputs.
struct IOGPairKey {
  // The units of the two holders, ordered by address; null for the entry
  Operation *unit1;
  Operation *unit2;
  // The entry channel if one of the holders is the entry, null otherwise
  Value entry;
  // The fork outputs that can copy a token between the holders, ordered by
  // address and unique, to compare them as a set
  SmallVector<Value> forkOutputs;

  IOGPairKey(const IOGSlotHolder &slot1, const IOGSlotHolder &slot2,
             ArrayRef<Value> copiedSents)
      : unit1(slot1.unit), unit2(slot2.unit), forkOutputs(copiedSents) {
    if (std::less<Operation *>()(unit2, unit1))
      std::swap(unit1, unit2);
    // The entry holds no unit, and has its channel as only output
    if (!unit1)
      entry = (slot1.unit ? slot2 : slot1).outputs.front();
    auto byAddress = [](Value lhs, Value rhs) {
      return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
    };
    llvm::sort(forkOutputs, byAddress);
    // The search from each holder of the pair can find the same output
    forkOutputs.erase(llvm::unique(forkOutputs), forkOutputs.end());
  }

  bool operator==(const IOGPairKey &other) const {
    return unit1 == other.unit1 && unit2 == other.unit2 &&
           entry == other.entry && forkOutputs == other.forkOutputs;
  }
};
} // namespace

template <>
struct llvm::DenseMapInfo<IOGPairKey> {
  static unsigned getHashValue(const IOGPairKey &key) {
    return hash_combine(
        key.unit1, key.unit2, key.entry,
        hash_combine_range(key.forkOutputs.begin(), key.forkOutputs.end()));
  }
  static bool isEqual(const IOGPairKey &lhs, const IOGPairKey &rhs) {
    return lhs == rhs;
  }
};

LogicalResult
HandshakeAnnotatePropertiesPass::annotateIOGSingleToken(ModuleOp modOp) {
  for (const IOG &iog : iogs) {
    // We model the entry node as a buffer that initially has one token.
    std::vector<PropertyExpr> slots{
        State(std::make_shared<EntrySlotNamer>(iog.entry))};
    std::vector<PropertyExpr> forks;
    // Collecting the slots and sents inside the IOG. The invariant relation of
    // num(slots) = 1 + num(eager fork sents), e.g.:
    //   (count(start_valid, testbench.end_full)
    //    = (1 + count(fork0.outs_1_sent)))
    for (auto &op : iog.units) {
      for (auto &slot : getAllSlotsOfOperation(op)) {
        slots.emplace_back(State(std::move(slot)));
      }
      if (auto forkOp = dyn_cast<EagerForkLikeOpInterface>(op)) {
        auto forkSlots = forkOp.getInternalSentStateNamers();
#ifndef NDEBUG
        // Number of outputs of the fork in the IOG
        int count = 0;
#endif
        for (auto [i, channel] : llvm::enumerate(forkOp->getResults())) {
          if (iog.contains(channel)) {
            forks.emplace_back(State(forkSlots[i]));
#ifndef NDEBUG
            count += 1;
#endif
          }
        }
        // An IOG contains exactly one output of each of its forks
        assert(count == 1);
      }
    }
    PropertyExpr tokens = Constant::integer(1);
    if (!forks.empty())
      tokens = sum({tokens, NaryExpr(NaryExpr::Count, forks)});
    addProperty(
        FormalProperty::IOG_SINGLE_TOKEN, FormalProperty::TAG::INVAR,
        BinaryExpr(NaryExpr(NaryExpr::Count, slots), BinaryExpr::Eq, tokens));
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateIOGConsecutiveTokens(ModuleOp modOp) {
  // The pairs already annotated: IOGs that differ only by fork outputs
  // elsewhere have the same pairs
  llvm::DenseSet<IOGPairKey> annotatedPairs;
  for (const IOG &iog : iogs) {
    // The entry forms a pair with every other slot, e.g.:
    //   ((start_valid & (count(buffer3.full) > 0)) -> fork0.outs_0_sent)
    std::vector<IOGSlotHolder> holders;
    holders.push_back(
        {/*unit=*/nullptr, /*inputs=*/{}, /*outputs=*/{iog.entry},
         /*occupied=*/State(std::make_shared<EntrySlotNamer>(iog.entry)),
         /*numSlots=*/1});
    for (auto &op : iog.units) {
      auto slotCountNamer = getTokenCountNamerOfOperation(op);
      if (slotCountNamer.has_value()) {
        size_t numSlots = slotCountNamer->slots->size();
        holders.push_back({/*unit=*/op,
                           /*inputs=*/SmallVector<Value>(op->getOperands()),
                           /*outputs=*/SmallVector<Value>(op->getResults()),
                           /*occupied=*/
                           BinaryExpr(State(*slotCountNamer), BinaryExpr::Gt,
                                      Constant::integer(0)),
                           /*numSlots=*/numSlots});
      }
    }
    for (auto slot1 = holders.begin(); slot1 != holders.end(); ++slot1) {
      for (auto slot2 = slot1 + 1; slot2 != holders.end(); ++slot2) {
        if (slot1->unit == slot2->unit) {
          // TODO: Handle loops, i.e. if the slot contains >=2 tokens, there
          // should be a copied fork within a loop
          if (slot1->numSlots >= 2) {
            slot1->unit->emitWarning("Should annotate self-loop");
          }
          continue;
        }

        SmallVector<Value> copiedSents = findCopiedSents(iog, *slot1, *slot2);

        SmallVector<Value> extra = findCopiedSents(iog, *slot2, *slot1);

        copiedSents.append(extra.begin(), extra.end());

        // Another IOG with the same two slots and fork outputs already gave the
        // same invariant
        if (!annotatedPairs.insert(IOGPairKey(*slot1, *slot2, copiedSents))
                 .second)
          continue;

        // Note:
        // Even if the copiedSents is empty, this invariant is interesting! It
        // means that both slots cannot be occupied at the same time, as there
        // is only (at most) one token in the IOG

        // If both slots hold a token, one of the forks copied it, e.g.:
        //   (((count(buffer1.full) > 0) & (count(buffer0.outs_valid_i) > 0))
        //    -> (fork2.outs_2_sent | fork4.outs_0_sent))
        // or, if no fork can copy the token between the two slots:
        //   (((count(buffer14.full) > 0) & (count(testbench.end_full) > 0))
        //    -> FALSE)
        std::vector<EagerForkSentNamer> sentNamers;
        for (Value output : copiedSents)
          sentNamers.push_back(getSentNamer(output));
        PropertyExpr bothOccupied =
            conjunction({slot1->occupied, slot2->occupied});
        addProperty(FormalProperty::IOG_CONSECUTIVE_TOKENS,
                    FormalProperty::TAG::INVAR,
                    implies(bothOccupied, disjunction(getStates(sentNamers))));
      }
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
//
// TODO: Only the control merges that the entry argument reaches directly are
// found, so the inner loops of a loop nest and the loops after another loop get
// no entry invariants, although their entry inputs (those that are not back
// edges) carry at most one token while the loop is active when the merge is on
// the control network of a sequential CFG. Find the entry paths of every
// control merge instead, walking backward from each entry input to the unit
// where its token originates (an argument, an outer control merge, or a
// branch).
std::vector<EntryCMergePath> findEntryCMergePaths(BlockArgument startChannel) {
  // The entry is modeled as a slot that initially holds one token
  std::vector<EffectiveSlotNamer> entrySlot;
  entrySlot.emplace_back(std::make_unique<EntrySlotNamer>(startChannel));
  std::vector<SlotPath> paths =
      findSlotPaths(startChannel, std::move(entrySlot), [](OpOperand &operand) {
        Operation *op = operand.getOwner();
        if (isa<ControlMergeOp>(op))
          return PathStep::End;
        // A control merge is an eager fork too, but must not match here: the
        // check above ends the path at it
        if (isa<BufferOp, EagerForkLikeOpInterface>(op))
          return PathStep::Traverse;
        // The entry token reaches another unit before any control merge, so
        // the path does not lead to an entry control merge
        //
        // TODO: This drops the paths through a branch of the control network.
        // A path that arrives on the data operand of a branch (unconditional,
        // or conditional before a loop guarded by an if) could be followed to
        // the output that leads to the control merge, as the branch does not
        // copy the token; one that arrives on the condition carries another
        // token
        return PathStep::Drop;
      });

  std::vector<EntryCMergePath> entryPaths;
  for (SlotPath &path : paths) {
    // When a control merge receives a token from its i-th input, it sends a
    // token with value i to its index output: the entry value is the index of
    // the input that the path ends at. A control merge only has data operands,
    // so this is the operand number.
    entryPaths.push_back(
        {/*slots=*/std::move(path.slots),
         /*cmerge=*/cast<ControlMergeOp>(path.end->getOwner()),
         /*entryValue=*/(int32_t)path.end->getOperandNumber()});
  }
  return entryPaths;
}

// This function finds any path from a control merge to a mux operation. Note
// that there can be multiple paths due to forks that replicate the index token
// of the CMerge to multiple muxes. Each path is the list of its effective
// slots: a slot only counts if the sents after it have not copied its token.
// For example, with two muxes:
//
//   control_merge0 --index--> fork0 --outs_0----------------> mux0
//       [slot]                  |
//                               +-----outs_1--> buffer0 --> mux1
//                                                [slot]
//
// there are two paths, with these effective slots:
//   to mux0: (control_merge0 slot & !index_sent & !fork0.outs_0_sent)
//   to mux1: (control_merge0 slot & !index_sent & !fork0.outs_1_sent), buffer0
std::vector<std::vector<EffectiveSlotNamer>>
findCMergeMuxPaths(ControlMergeOp cmerge) {
  // The paths start at the slot of the control merge, which holds the index
  std::vector<EffectiveSlotNamer> mergeSlot;
  mergeSlot.emplace_back(std::make_unique<BufferSlotFullNamer>(
      cmerge.getInternalSlotStateNamers()[0]));
  // The index output that the paths start from copies the token of the slot
  mergeSlot.back().addCopiedSent(getSentNamer(cmerge.getIndex()));
  std::vector<SlotPath> paths = findSlotPaths(
      cmerge.getIndex(), std::move(mergeSlot), [](OpOperand &operand) {
        Operation *op = operand.getOwner();
        if (isa<MuxOp>(op))
          return PathStep::End;
        if (isa<BufferOp, EagerForkLikeOpInterface>(op))
          return PathStep::Traverse;
        return PathStep::Unexpected;
      });

  std::vector<std::vector<EffectiveSlotNamer>> slotPaths;
  for (SlotPath &path : paths)
    slotPaths.push_back(std::move(path.slots));
  return slotPaths;
}
} // namespace

LogicalResult HandshakeAnnotatePropertiesPass::annotateEntryTokenOrderPaths(
    ControlMergeOp cmerge, int32_t entryValue) {
  for (const auto &path : findCMergeMuxPaths(cmerge)) {
    if (path.size() < 2) {
      // The regex of this invariant trivially holds for any path of length 1
      continue;
    }
    // If a slot holds the entry token, all later slots on the path are empty,
    // e.g., for the path control_merge0 -> buffer3 -> buffer4 (2 slots) and the
    // entry token FALSE:
    //   (((control_merge0.slot_full & (control_merge0.data = FALSE) & ...)
    //     -> ((! buffer3.outs_valid_i) & (! buffer4.b0.full)
    //         & (! buffer4.b1.full)))
    //    & ((buffer3.outs_valid_i & (buffer3.data = FALSE))
    //       -> ((! buffer4.b0.full) & (! buffer4.b1.full)))
    //    & ((buffer4.b0.full & (buffer4.b0.reg = FALSE)) -> (!
    //    buffer4.b1.full)))
    std::vector<PropertyExpr> implications;
    for (auto start = path.begin(); start + 1 != path.end(); ++start) {
      std::vector<PropertyExpr> laterSlotsEmpty;
      for (auto later = start + 1; later != path.end(); ++later)
        laterSlotsEmpty.push_back(!State(*later));
      implications.push_back(
          implies(State(*start, entryValue), conjunction(laterSlotsEmpty)));
    }
    addProperty(FormalProperty::ENTRY_TOKEN_ORDER, FormalProperty::TAG::INVAR,
                conjunction(implications));
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
          // If the path from the control merge to the mux holds a token, the
          // path from the entry to the control merge is empty, e.g.:
          //   ((count(control_merge0.slot_full, buffer8.full) > 0)
          //    -> (count((start_valid & !fork0.outs_2_sent)) = 0))
          // The control merge keeps the token in its slot after sending it on
          // the path, so its slot counts whatever its copied sents are
          std::vector<PropertyExpr> cmStates = getStates(cm);
          cmStates.front() = State(cm.front().slot);
          PropertyExpr cmOccupied =
              BinaryExpr(NaryExpr(NaryExpr::Count, cmStates), BinaryExpr::Gt,
                         Constant::integer(0));
          PropertyExpr ecEmpty =
              BinaryExpr(NaryExpr(NaryExpr::Count, getStates(ec.slots)),
                         BinaryExpr::Eq, Constant::integer(0));
          addProperty(FormalProperty::SINGLE_ENTRY_TOKEN,
                      FormalProperty::TAG::INVAR, implies(cmOccupied, ecEmpty));
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

std::vector<UnstartedPath> findDeciderBranchPaths(Operation *decider) {
  struct PartialPath {
    std::vector<EffectiveSlotNamer> slots;
    std::vector<EagerForkSentNamer> startSents;
    mlir::Value cur;
  };
  std::vector<UnstartedPath> ret{};
  std::vector<PartialPath> stack;
  mlir::Value startChannel;
  if (auto arith = dyn_cast<ArithOpInterface>(decider)) {
    startChannel = arith->getResult(0);
  } else if (auto load = dyn_cast<LoadOp>(decider)) {
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
    if (auto fork = dyn_cast<EagerForkLikeOpInterface>(next)) {
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
          nextPath.startSents.push_back(sents[i]);
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

// The slots of the IOG from which a token can reach `decider` without
// going through the branch `exit`, which would overlap with the path from the
// decider to the branch. A slot holds an ancestor of the decider's token only
// if that token has not been copied towards the decider, so each slot is
// effective with the copied sents of all the fork outputs between it and the
// next slots on the way to the decider. `decSents` are the outputs of the
// decider's own forks on the path of the exit token, before its first slot;
// they are also copied sents of the slots right before the decider.
//
// For example, in a loop that counts down from n:
//
//                n
//                |
//                v
//              mux0 <------------------+
//                |                     |
//                v                     |
//     +-outs_1- fork0                  |
//     |          |                     |
//     |        outs_0                  |
//     |          v                     |
//     |       buffer1               buffer0
//     |          |                     ^
//     |          v                     |
//     |        cmpi0 (decider)         |
//     |          |                     |
//     |          v                     |
//     |        fork1 --> ...           |
//     |          |                     |
//     |        outs_0                  |
//     |          v                     |
//     |       buffer2                  |
//     |          |                     |
//     |          v                     |
//     +----> cond_br0 (`exit`) --------+
//                |
//                v
//               ...
//
// Apart from the more obvious ancestor slots like buffer1, the ancestors also
// include the back edge (buffer0), whose token reaches the decider in the next
// iteration. While the exit token is in buffer2, the effective slots are empty,
// i.e., each slot can only hold a token once the fork after it has copied that
// token towards the decider:
//   (n_valid & !fork0.outs_0_sent)
//   (buffer0 full & !fork0.outs_0_sent)
//   (buffer1 full & !fork1.outs_0_sent), from `decSents`: buffer1 keeps its
//     token until fork1 has sent the decision on every output
std::vector<EffectiveSlotNamer>
findAncestorSlots(const IOG &iog, Operation *decider, ConditionalBranchOp exit,
                  ArrayRef<EagerForkSentNamer> decSents) {
  llvm::SetVector<Operation *> ancestors = reachBackward(
      /*starts=*/SmallVector<Value>(decider->getOperands()),
      /*isBoundary=*/[&](Operation *op) { return op == exit; },
      /*isFollowed=*/[&](Value channel) { return iog.contains(channel); });
  ancestors.remove(exit);

  // The copied sents of the slots whose tokens leave through `outputs`
  auto getCopiedSents = [&](ArrayRef<Value> outputs) {
    std::vector<EagerForkSentNamer> sents;
    bool reachesDecider = false;
    llvm::DenseSet<Value> visited;
    SmallVector<Value> stack(outputs);
    while (!stack.empty()) {
      Value channel = stack.pop_back_val();
      if (!iog.contains(channel) || !visited.insert(channel).second)
        continue;
      Operation *next = *channel.getUsers().begin();
      if (next != decider && !ancestors.contains(next))
        continue;
      if (isa_and_nonnull<EagerForkLikeOpInterface>(channel.getDefiningOp()))
        sents.push_back(getSentNamer(channel));
      if (next == decider)
        reachesDecider = true;
      // Later forks do not copy the token of this slot, but of the next one
      else if (getAllSlotsOfOperation(next).empty())
        llvm::append_range(stack, next->getResults());
    }
    // The decider has no slot, so the decider's own fork outputs have to be
    // taken into account for the ancestor slot right before the decider: it
    // keeps its token until those forks have sent the decision on every
    // output, while the exit token may already be on the path to `exit`
    if (reachesDecider)
      llvm::append_range(sents, decSents);
    return sents;
  };

  std::vector<EffectiveSlotNamer> slots;
  // The entry is an ancestor if the token it holds reaches the decider
  Operation *entryUser = *iog.entry.getUsers().begin();
  if (entryUser == decider || ancestors.contains(entryUser)) {
    EffectiveSlotNamer entry(std::make_unique<EntrySlotNamer>(iog.entry));
    for (EagerForkSentNamer &sent : getCopiedSents({iog.entry}))
      entry.addCopiedSent(std::move(sent));
    slots.push_back(std::move(entry));
  }
  for (Operation *op : ancestors) {
    auto opSlots = getAllSlotsOfOperation(op);
    if (opSlots.empty())
      continue;
    // Tokens leave a unit from its last slot
    std::unique_ptr<InternalStateNamer> lastSlot = std::move(opSlots.back());
    opSlots.pop_back();
    for (auto &slot : opSlots)
      slots.emplace_back(std::move(slot));
    EffectiveSlotNamer effectiveSlot(std::move(lastSlot));
    for (EagerForkSentNamer &sent :
         getCopiedSents(SmallVector<Value>(op->getResults())))
      effectiveSlot.addCopiedSent(std::move(sent));
    slots.push_back(std::move(effectiveSlot));
  }
  return slots;
}

} // namespace

LogicalResult
HandshakeAnnotatePropertiesPass::annotateExitTokenOrder(ModuleOp modOp) {
  // Map vectors iterate in insertion order, so that the invariants are
  // annotated in the same order in every run
  llvm::MapVector<ConditionalBranchOp, int32_t> exitBranches;
  for (const auto &iog : iogs) {
    for (auto *op : iog.units) {
      if (auto branch = dyn_cast<ConditionalBranchOp>(op)) {
        if (auto optExitValue = iog.isExitBranch(branch)) {
          exitBranches.insert({branch, *optExitValue});
        }
      }
    }
  }
  llvm::MapVector<Operation *, int32_t> deciders;
  for (auto [branch, exitValue] : exitBranches) {
    Operation *decider = *getDecider(branch);
    deciders.insert({decider, exitValue});
  }
  for (auto [decider, exitValue] : deciders) {
    auto slotPaths = findDeciderBranchPaths(decider);
    // [START Enumerating ExitTokenOrder invariants]
    for (const auto &slots : slotPaths) {
      if (slots.started.size() < 2) {
        // Invariant is trivially true
        continue;
      }
      // If a slot holds the exit token, all earlier slots on the path are
      // empty, e.g., for the path buffer9 -> buffer10 (2 slots) and the exit
      // token FALSE:
      //   (((buffer10.b0.full & (buffer10.b0.reg = FALSE))
      //     -> (! buffer9.outs_valid_i))
      //    & ((buffer10.b1.full & (buffer10.b1.reg = FALSE))
      //       -> ((! buffer9.outs_valid_i) & (! buffer10.b0.full))))
      std::vector<PropertyExpr> implications;
      std::vector<PropertyExpr> earlierSlotsEmpty{
          !State(slots.started.front())};
      for (auto start = slots.started.begin() + 1; start != slots.started.end();
           ++start) {
        implications.push_back(
            implies(State(*start, exitValue), conjunction(earlierSlotsEmpty)));
        earlierSlotsEmpty.push_back(!State(*start));
      }
      addProperty(FormalProperty::EXIT_TOKEN_ORDER, FormalProperty::TAG::INVAR,
                  conjunction(implications));
    }
    // [END Enumerating ExitTokenOrder invariants]

    // [START Enumerating ExitTokenNoAncestors invariants]
    for (const auto &iog : iogs) {
      if (!iog.contains(decider)) {
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

      for (const auto &decPath : slotPaths) {
        if (decPath.started.empty())
          continue;
        std::vector<EffectiveSlotNamer> ancestors =
            findAncestorSlots(iog, decider, *iogBranch, decPath.startSents);
        if (ancestors.empty())
          continue;
        // If a slot on the path from the decider to the branch holds the
        // exit token, the ancestors of the decider are empty, e.g.:
        //   ((count((buffer9.outs_valid_i & (buffer9.data = FALSE)),
        //           (buffer10.b0.full & (buffer10.b0.reg = FALSE)), ...) > 0)
        //    -> (count(buffer7.outs_valid_i,
        //              (buffer8.full & !fork4.outs_1_sent), ...) = 0))
        std::vector<PropertyExpr> exitTokens;
        exitTokens.reserve(decPath.started.size());
        for (const auto &es : decPath.started)
          exitTokens.emplace_back(State(es.slot, exitValue));
        PropertyExpr anyExitToken =
            BinaryExpr(NaryExpr(NaryExpr::Count, exitTokens), BinaryExpr::Gt,
                       Constant::integer(0));
        PropertyExpr ancestorsEmpty =
            BinaryExpr(NaryExpr(NaryExpr::Count, getStates(ancestors)),
                       BinaryExpr::Eq, Constant::integer(0));
        addProperty(FormalProperty::EXIT_TOKEN_NO_ANCESTORS,
                    FormalProperty::TAG::INVAR,
                    implies(anyExitToken, ancestorsEmpty));
      }
    }
    // [END Enumerating ExitTokenNoAncestors invariants]
  }
  return success();
}

LogicalResult HandshakeAnnotatePropertiesPass::annotateQueriedProperties() {
  ModuleOp modOp = getOperation();
  if (!annotateList.empty()) {
    LogicalResult res = success();
    for (StringRef name : llvm::split(annotateList, ',')) {
      name = name.trim();
      const GeneratorEntry *entry = llvm::find_if(
          GENERATORS, [&](const GeneratorEntry &e) { return e.name == name; });
      if (entry == std::end(GENERATORS)) {
        llvm::errs() << name << " is not a property\n";
        res = failure();
      } else if (failed((this->*entry->generator)(modOp))) {
        res = failure();
      }
    }
    return res;
  }

  for (const GeneratorEntry &entry : GENERATORS) {
    bool selected = entry.isInvariant ? annotateInvariants : annotateProperties;
    if (selected && failed((this->*entry.generator)(modOp)))
      return failure();
  }
  return success();
}

void HandshakeAnnotatePropertiesPass::runOnOperation() {
  // Make sure all operation names are unique and haven't changed from what is
  // cached (if anything is cached)
  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  if (!namer.isAnalysisValid())
    return signalPassFailure();

  ModuleOp modOp = getOperation();
  iogs = findAllIOGs(modOp);
  // Ordered by insertion, so that the dead buffers replacing the sinks are
  // named in the same order in every run
  llvm::SetVector<SinkOp> sinks;
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
        DeadBufferOp::create(builder, sink.getLoc(), sink.getOperand());
    sink.erase();
    for (auto &iog : iogs) {
      if (iog.contains(sink)) {
        iog.units.remove(sink);
        iog.units.insert(deadBuffer);
      }
    }
  }
  namer.nameAllUnnamedOps();

  if (failed(annotateQueriedProperties())) {
    return signalPassFailure();
  }

  // Make sure all operation names are still unique and name operations that
  // do not currently have a name
  if (failed(namer.walk(NameAnalysis::UnnamedBehavior::NAME)))
    return signalPassFailure();
  markAnalysesPreserved<NameAnalysis>();

  llvm::json::Value jsonVal(std::move(propertyTable));

  std::error_code EC;
  llvm::raw_fd_ostream jsonOut(jsonPath, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return;

  jsonOut << formatv("{0:2}", jsonVal);
}
