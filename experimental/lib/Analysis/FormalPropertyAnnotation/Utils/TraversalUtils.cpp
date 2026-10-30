//===- TraversalUtils.cpp - Paths of tokens through a circuit -------------===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements the flow edges of the Handshake units and the walks along them.
///
//===----------------------------------------------------------------------===//

#include "TraversalUtils.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace dynamatic;
using namespace dynamatic::handshake;

namespace {
/// How tokens flow from the operands to the results of a unit.
enum class FlowKind {
  /// A token on any operand can leave through any result.
  AllToAll,
  /// Tokens leave the circuit or are consumed, so no token leaves through a
  /// result.
  Terminal,
  /// A load passes the address to the memory interface, which returns the
  /// data. This round trip is collapsed into one edge, from the address
  /// operand to the data result.
  Load,
  /// The flow through the unit is not modeled.
  Unsupported
};
} // namespace

static FlowKind getFlowKind(Operation *op) {
  if (isa<SinkOp, DeadBufferOp, StoreOp, MemoryControllerOp, LSQOp, EndOp>(op))
    return FlowKind::Terminal;
  if (isa<LoadOp>(op))
    return FlowKind::Load;
  // For a mux, a control merge, and a branch, this includes the select, the
  // index, and the condition. For example, a token that decides a branch
  // reaches the unit that the branch sends its data to.
  if (isa<ArithOpInterface, NotIOp, ForkOp, LazyForkOp, BufferOp, MergeOp,
          MuxOp, ControlMergeOp, BranchOp, ConditionalBranchOp, JoinOp,
          SourceOp>(op))
    return FlowKind::AllToAll;
  return FlowKind::Unsupported;
}

[[noreturn]] static void reportUnsupportedFlow(Operation *op) {
  op->emitError("the flow of tokens through this unit is not modeled");
  llvm::report_fatal_error("unsupported unit in the flow of tokens");
}

SmallVector<OpResult> dynamatic::getFlowSuccessors(OpOperand &operand) {
  Operation *op = operand.getOwner();
  switch (getFlowKind(op)) {
  case FlowKind::AllToAll:
    return SmallVector<OpResult>(op->getResults());
  case FlowKind::Terminal:
    return {};
  case FlowKind::Load: {
    auto loadOp = cast<LoadOp>(op);
    // The data coming back from the memory interface is part of the edge from
    // the address, so it has no edge of its own.
    if (operand.get() != loadOp.getAddress())
      return {};
    return {cast<OpResult>(loadOp.getDataResult())};
  }
  case FlowKind::Unsupported:
    reportUnsupportedFlow(op);
  }
  llvm_unreachable("unknown flow kind");
}

SmallVector<OpOperand *> dynamatic::getFlowPredecessors(OpResult result) {
  Operation *op = result.getOwner();
  switch (getFlowKind(op)) {
  case FlowKind::AllToAll:
    return SmallVector<OpOperand *>(
        llvm::make_pointer_range(op->getOpOperands()));
  case FlowKind::Terminal:
    return {};
  case FlowKind::Load: {
    auto loadOp = cast<LoadOp>(op);
    // The address going to the memory interface leaves the modeled flow.
    if (result != loadOp.getDataResult())
      return {};
    return {&loadOp.getAddressMutable()};
  }
  case FlowKind::Unsupported:
    reportUnsupportedFlow(op);
  }
  llvm_unreachable("unknown flow kind");
}

llvm::SetVector<Operation *>
dynamatic::reachForward(ArrayRef<Value> starts,
                        function_ref<bool(Operation *)> isBoundary) {
  llvm::SetVector<Operation *> reached;
  // The channels are visited rather than the units, as the results that a
  // token can leave through depend on the operand it arrives on (see loads).
  llvm::DenseSet<Value> visited;
  SmallVector<Value> worklist(starts);
  while (!worklist.empty()) {
    Value channel = worklist.pop_back_val();
    if (!visited.insert(channel).second)
      continue;
    for (OpOperand &use : channel.getUses()) {
      Operation *user = use.getOwner();
      reached.insert(user);
      if (!isBoundary(user))
        llvm::append_range(worklist, getFlowSuccessors(use));
    }
  }
  return reached;
}

llvm::SetVector<Operation *>
dynamatic::reachBackward(ArrayRef<Value> starts,
                         function_ref<bool(Operation *)> isBoundary) {
  llvm::SetVector<Operation *> reached;
  llvm::DenseSet<Value> visited;
  SmallVector<Value> worklist(starts);
  while (!worklist.empty()) {
    Value channel = worklist.pop_back_val();
    if (!visited.insert(channel).second)
      continue;
    auto result = dyn_cast<OpResult>(channel);
    // A function argument has no unit producing it.
    if (!result)
      continue;
    Operation *producer = result.getOwner();
    reached.insert(producer);
    if (isBoundary(producer))
      continue;
    for (OpOperand *operand : getFlowPredecessors(result))
      worklist.push_back(operand->get());
  }
  return reached;
}

std::vector<SlotPath>
dynamatic::findSlotPaths(Value start,
                         std::vector<EffectiveSlotNamer> initialSlots,
                         function_ref<PathStep(OpOperand &)> step) {
  // A path from `start` up to `channel`, which still has to be extended.
  struct PartialPath {
    SlotPath path;
    Value channel;
    // The units on the path, so that the path does not go around a cycle.
    SmallVector<Operation *> units;
  };
  std::vector<SlotPath> paths;
  std::vector<PartialPath> stack;
  stack.push_back({/*path=*/{/*leadingSents=*/{},
                             /*slots=*/std::move(initialSlots),
                             /*end=*/nullptr},
                   /*channel=*/start,
                   /*units=*/{}});
  while (!stack.empty()) {
    PartialPath partial = std::move(stack.back());
    stack.pop_back();
    for (OpOperand &use : partial.channel.getUses()) {
      Operation *op = use.getOwner();
      switch (step(use)) {
      case PathStep::End: {
        SlotPath path = partial.path;
        path.end = &use;
        paths.push_back(std::move(path));
        continue;
      }
      case PathStep::Drop:
        continue;
      case PathStep::Unexpected:
        op->emitError("unexpected unit on a path of tokens");
        llvm::report_fatal_error("unexpected unit on a path of tokens");
      case PathStep::Traverse:
        break;
      }
      // A path that comes back to a unit it already went through only repeats
      // a part of itself, so it is discarded.
      if (llvm::is_contained(partial.units, op))
        continue;

      // The slots of the unit come before its eager fork outputs. For
      // example, the slot of a control merge holds the token before it is
      // sent to the outputs.
      SlotPath throughUnit = partial.path;
      for (std::unique_ptr<InternalStateNamer> &slot :
           getAllSlotsOfOperation(op))
        throughUnit.slots.emplace_back(std::move(slot));
      auto eagerFork = dyn_cast<EagerForkLikeOpInterface>(op);
      std::vector<EagerForkSentNamer> sents;
      if (eagerFork)
        sents = eagerFork.getInternalSentStateNamers();

      for (OpResult result : getFlowSuccessors(use)) {
        PartialPath next{/*path=*/throughUnit, /*channel=*/result,
                         /*units=*/partial.units};
        next.units.push_back(op);
        // The output taken by the path copies the token of the last slot, or
        // of the start if the path has no slot yet.
        if (eagerFork) {
          EagerForkSentNamer &sent = sents[result.getResultNumber()];
          if (next.path.slots.empty())
            next.path.leadingSents.push_back(sent);
          else
            next.path.slots.back().addCopiedSent(sent);
        }
        stack.push_back(std::move(next));
      }
    }
  }
  return paths;
}
