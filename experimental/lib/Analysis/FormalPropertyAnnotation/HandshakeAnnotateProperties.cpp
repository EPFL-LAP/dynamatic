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

#include "Utils/FlowExpression.h"
#include "Utils/Gaussian.h"
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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
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
  LogicalResult annotateReconvergentPathFlow(ModuleOp modOp);
  LogicalResult annotateIOGSingleToken(ModuleOp modOp);
  LogicalResult annotateIOGConsecutiveTokens(ModuleOp modOp);
  void annotateIOGConsecutiveTokensOf(const IOG &iog);
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

  if (auto latencyOp = dyn_cast<handshake::LatencyInterface>(curOp)) {
    annotateCopiedSlot(State(latencyOp.getPipelineSlots().back()));
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

    // The row is an equation over internal states only: the sum of the
    // internal states weighted by the coefficients of the row is 0, e.g.:
    //   (((toint(fork4.outs_0_sent) * 1) + (toint(fork4.outs_1_sent) * -1)
    //     + (toint((fork5.outs_0_sent & (fork5.ins = TRUE))) * -1)) = 0)
    // The terms are in column order, which is the same in every run.
    std::vector<PropertyExpr> terms;
    for (size_t col = indices.nLambdas; col < indices.registry.size(); ++col) {
      int coef = matrix(row, col);
      if (coef == 0)
        continue;
      const FlowVariable &var = indices.registry.getVar(col);
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
  for (const IOG &iog : iogs)
    annotateIOGConsecutiveTokensOf(iog);
  return success();
}

void HandshakeAnnotatePropertiesPass::annotateIOGConsecutiveTokensOf(
    const IOG &iog) {
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

      // If both slots hold a token, one of the forks copied it, e.g.:
      //   (((count(buffer1.full) > 0) & (count(buffer0.outs_valid_i) > 0))
      //    -> (fork2.outs_2_sent | fork4.outs_0_sent))
      // or, if no fork can copy the token between the two slots:
      //   (((count(buffer14.full) > 0) & (count(testbench.end_full) > 0))
      //    -> FALSE)
      auto isOccupied = [](const TokenCountNamer &slot) {
        return BinaryExpr(State(slot), BinaryExpr::Gt, Constant::integer(0));
      };
      PropertyExpr bothOccupied =
          conjunction({isOccupied(slot1->second), isOccupied(slot2->second)});
      addProperty(FormalProperty::IOG_CONSECUTIVE_TOKENS,
                  FormalProperty::TAG::INVAR,
                  implies(bothOccupied, disjunction(getStates(copiedSents))));
    }
  }
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
  // The entry is modeled as a slot that initially holds one token
  std::vector<EffectiveSlotNamer> entrySlot;
  entrySlot.emplace_back(std::make_unique<EntrySlotNamer>(startChannel));
  std::vector<SlotPath> paths =
      findSlotPaths(startChannel, std::move(entrySlot), [](OpOperand &operand) {
        Operation *op = operand.getOwner();
        if (isa<ControlMergeOp>(op))
          return PathStep::End;
        if (isa<BufferOp, ForkOp>(op))
          return PathStep::Traverse;
        // The entry token reaches another unit before any control merge, so
        // the path does not lead to an entry control merge
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
// of the CMerge to multiple muxes.
std::vector<std::vector<EffectiveSlotNamer>>
findCMergeMuxPaths(ControlMergeOp cmerge) {
  // The paths start at the slot of the control merge, which holds the index
  std::vector<EffectiveSlotNamer> mergeSlot;
  mergeSlot.emplace_back(std::make_unique<BufferSlotFullNamer>(
      cmerge.getInternalSlotStateNamers()[0]));
  std::vector<SlotPath> paths = findSlotPaths(
      cmerge.getIndex(), std::move(mergeSlot), [](OpOperand &operand) {
        Operation *op = operand.getOwner();
        if (isa<MuxOp>(op))
          return PathStep::End;
        if (isa<BufferOp, ForkOp>(op))
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
          //   ((count((control_merge0.slot_full & !fork3.outs_0_sent)) > 0)
          //    -> (count((start_valid & !fork0.outs_2_sent)) = 0))
          PropertyExpr cmOccupied =
              BinaryExpr(NaryExpr(NaryExpr::Count, getStates(cm)),
                         BinaryExpr::Gt, Constant::integer(0));
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
              BinaryExpr(NaryExpr(NaryExpr::Count, getStates(ancestorPath)),
                         BinaryExpr::Eq, Constant::integer(0));
          addProperty(FormalProperty::EXIT_TOKEN_NO_ANCESTORS,
                      FormalProperty::TAG::INVAR,
                      implies(anyExitToken, ancestorsEmpty));
        }
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
        builder.create<DeadBufferOp>(sink.getLoc(), sink.getOperand());
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
