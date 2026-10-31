#include "IOG.h"
#include "TraversalUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace dynamatic::handshake;

namespace dynamatic {
IOGPathSet IOG::findUnitsOnPaths(Operation *startOp, Operation *endOp) const {
  IOGPathSet paths(startOp, endOp);
  auto inIOG = [&](Value channel) { return contains(channel); };
  // The units reachable from the start without going through the end, and the
  // units that reach the end without going through the start
  llvm::SetVector<Operation *> forward = reachForward(
      SmallVector<Value>(startOp->getResults()),
      [&](Operation *op) { return op == endOp; }, inIOG);
  forward.insert(startOp);
  llvm::SetVector<Operation *> backward = reachBackward(
      SmallVector<Value>(endOp->getOperands()),
      [&](Operation *op) { return op == startOp; }, inIOG);
  backward.insert(endOp);
  // The units on a path from the start to the end are both
  for (Operation *op : forward)
    if (backward.contains(op))
      paths.units.insert(op);
  return paths;
}

std::optional<int32_t>
IOG::isExitBranch(handshake::ConditionalBranchOp branch) const {
  assert(contains(branch));
  Operation *opTrue = *branch.getTrueResult().getUsers().begin();
  auto paths = findUnitsOnPaths(opTrue, branch);
  bool trueLoop = !(paths.units.find(branch) == paths.units.end());

  Operation *opFalse = *branch.getFalseResult().getUsers().begin();
  paths = findUnitsOnPaths(opFalse, branch);
  bool falseLoop = !(paths.units.find(branch) == paths.units.end());

  if (trueLoop == falseLoop) {
    return std::nullopt;
  }

  int32_t exitValue = trueLoop ? 0 : 1;
  return exitValue;
}

namespace {
/// The ways in which a unit can be part of an IOG. An IOG holds exactly one of
/// the input groups and one of the output groups of each of its units (none
/// for a unit without outputs, e.g., a sink), and no other channel of the
/// unit. For example:
/// - fork: inputs = {{in}}, outputs = {{out0}, {out1}, ...}
/// - mux: inputs = {{data0, data1, ...}, {select}}, outputs = {{result}}
/// - control merge: inputs = {{data0, data1, ...}}, outputs = {{outs},
///   {index}}
/// - conditional branch: inputs = {{data}, {condition}}, outputs = {{true,
///   false}}
/// - join (e.g., addi): inputs = {{lhs}, {rhs}}, outputs = {{result}}
/// - sink: inputs = {{in}}, outputs = {}
struct IOGPortGroups {
  /// The input groups, each a set of operands.
  SmallVector</*group=*/SmallVector<Value>> inputs;
  /// The output groups, each a set of results.
  SmallVector</*group=*/SmallVector<Value>> outputs;
};
} // namespace

/// One group per channel, i.e., exactly one of the channels.
static SmallVector<SmallVector<Value>> eachOf(ValueRange channels) {
  SmallVector<SmallVector<Value>> groups;
  for (Value channel : channels)
    groups.push_back({channel});
  return groups;
}

/// A single group of all the channels.
static SmallVector<SmallVector<Value>> allOf(ValueRange channels) {
  return {SmallVector<Value>(channels)};
}

/// The groups of the channels of `op` (see `IOGPortGroups`), or std::nullopt if
/// no IOG contains `op`.
///
/// The groups follow from what an IOG is for: the number of tokens in the IOG
/// only changes when a fork copies a token, so that it can be counted from the
/// sent states of the forks. Each unit therefore turns one token of the IOG
/// into one token of the IOG, or keeps it (e.g., a sink).
static std::optional<IOGPortGroups> getIOGPortGroups(Operation *op) {
  using Groups = std::optional<IOGPortGroups>;
  return llvm::TypeSwitch<Operation *, Groups>(op)
      .Case<EndOp>([](EndOp endOp) -> Groups {
        // The end joins one token from each result of the function, so the IOG
        // takes one of them, as the other tokens are not counted by the IOG
        return IOGPortGroups{eachOf(endOp->getOperands()), {}};
      })
      .Case<BufferOp>([](BufferOp bufOp) -> Groups {
        return IOGPortGroups{allOf(bufOp->getOperands()),
                             allOf(bufOp->getResults())};
      })
      .Case<ConditionalBranchOp>([](ConditionalBranchOp condBr) -> Groups {
        // A branch joins the data with the condition, so the IOG takes one of
        // them, and sends the token to one of its outputs, which can be any, so
        // the IOG takes both
        return IOGPortGroups{
            {{condBr.getDataOperand()}, {condBr.getConditionOperand()}},
            allOf(condBr->getResults())};
      })
      .Case<ForkOp>([](ForkOp forkOp) -> Groups {
        // A fork copies the token to all its outputs, so the IOG takes one of
        // them, as the other copies leave the IOG
        return IOGPortGroups{allOf(forkOp->getOperands()),
                             eachOf(forkOp->getResults())};
      })
      .Case<MuxOp>([](MuxOp muxOp) -> Groups {
        // A mux passes on a token from one of its data inputs, which can be
        // any, and joins it with the select, so the IOG takes either all the
        // data inputs or the select
        SmallVector<SmallVector<Value>> inputs = allOf(muxOp.getDataOperands());
        inputs.push_back({muxOp.getSelectOperand()});
        return IOGPortGroups{inputs, allOf(muxOp->getResults())};
      })
      .Case<ControlMergeOp>([](ControlMergeOp cmerge) -> Groups {
        // A control merge passes on a token from one of its inputs, which can
        // be any, so the IOG takes all of them, and copies it to its two
        // outputs, so the IOG takes one of them
        return IOGPortGroups{allOf(cmerge.getDataOperands()),
                             eachOf(cmerge->getResults())};
      })
      .Case<SinkOp, DeadBufferOp>([](Operation *sink) -> Groups {
        // A sink is the end of the IOG. The annotation replaces the sinks of
        // the IOGs by dead buffers, which keep the token
        return IOGPortGroups{allOf(sink->getOperands()), {}};
      })
      .Case<LoadOp>([](LoadOp loadOp) -> Groups {
        // The round trip through the memory interface is collapsed, as in the
        // flow edges of the load (see `getFlowSuccessors`)
        return IOGPortGroups{{{loadOp.getAddress()}},
                             {{loadOp.getDataResult()}}};
      })
      .Case<StoreOp, SourceOp, MemoryControllerOp>([](Operation *) -> Groups {
        // A token leaves the circuit through a store and enters it through a
        // source, so the number of tokens in the IOG would not be fixed. The
        // memory controllers are only reached by control signals, which give
        // trivial IOGs that can be skipped
        return std::nullopt;
      })
      .Default([](Operation *op) -> Groups {
        // An arithmetic unit joins one token from each operand, so the IOG
        // takes one of them, as the other tokens are not counted by the IOG
        if (isa<ArithOpInterface>(op))
          return IOGPortGroups{eachOf(op->getOperands()), {{op->getResult(0)}}};
        op->emitError("unhandled case in IOG finding");
        return std::nullopt;
      });
}

namespace {
struct IOGCandidate {
  IOG iog;
  // Stack of channels at the edge that are not yet inserted into the IOG
  std::vector<mlir::Value> dfsQueue;
  // Set of channels that have been declared illegal by local rules
  llvm::DenseSet<mlir::Value> illegalChannels;

  // Is the partial IOG contained within this struct invalid? I.e., is an
  // illegal edge part of the IOG?
  bool invalidIOG = false;

  IOGCandidate() = default;
  IOGCandidate(BlockArgument entry) {
    iog.entry = entry;
    markFollowed(entry);
  }

  bool isLegal(mlir::Value channel) {
    auto iter = illegalChannels.find(channel);
    return iter == illegalChannels.end();
  }

  void markFollowed(mlir::Value channel) {
    if (!isLegal(channel)) {
      // A non-legal channel has been followed, so mark this IOG as non-valid
      invalidIOG = true;
      return;
    }
    if (iog.contains(channel)) {
      return;
    }
    dfsQueue.push_back(channel);
    iog.channels.insert(channel);
  }

  void markIllegal(mlir::Value channel) {
    if (iog.contains(channel)) {
      invalidIOG = true;
    }
    // The legality of unhandled channels is checked in getNextOp
    illegalChannels.insert(channel);
  }

  // Get an operation that needs to be added to the IOG:
  // OP_BEFORE ---channel--> OP_AFTER
  // 1. Pop an edge from the unhandled edges stack
  // 2. Check if OP_BEFORE has been handled already. If not, return it for
  // handling
  // 3. Check if OP_AFTER has been handled already. If not, return it for
  // handling
  // 4. Both sides of this edge have been added, so mark this edge as done and
  // repeat for the next edge on the stack
  std::optional<Operation *> getNextOpToAdd() {
    // Stop making progress if an illegal edge has been taken
    if (invalidIOG) {
      return std::nullopt;
    }

    if (dfsQueue.empty()) {
      // All handling done!
      return std::nullopt;
    }

    // Look at channel at the top of the `unhandled` stack
    auto channel = dfsQueue.back();

    // If this channel has been marked as illegal, abort this branch
    if (!isLegal(channel)) {
      invalidIOG = true;
      return std::nullopt;
    }

    // To fully handle a channel, both the operation before and after have to be
    // handled
    Operation *before = channel.getDefiningOp();
    if (before && !iog.contains(before)) {
      iog.units.insert(before);
      return before;
    }

    Operation *after = channel.getUses().begin()->getOwner();
    if (after && !iog.contains(after)) {
      iog.units.insert(after);
      return after;
    }

    // Both sides of this channel have been handled, so this channel is done and
    // can be removed/inserted into the appropriate locations
    dfsQueue.pop_back();

    // Return next channel on unhandled stack
    return getNextOpToAdd();
  }
};

class IOGFinder {
public:
  IOGFinder(BlockArgument entry) {
    IOGCandidate cp(entry);
    candidates.push_back(cp);
  }
  ~IOGFinder() = default;

  // Take a step towards computing all IOGs. Does nothing if there are no
  // partial IOGs.
  // Example usage:
  // IOGFinder finder = ...
  // while (!finder.isDone()) {
  //   finder.step();
  // }
  // auto iogs = finder.getIOGs();
  //
  //
  // The algorithm works as follows:
  // 1. Pick a candidate IOG to work on
  // 2. Within this IOG, get the next operation that should be added to the IOG
  // 3. Handle the operation according to local rules, eliminating the candidate
  // or splitting into multiple possible candidates as necessary
  void step() {
    if (candidates.empty())
      return;
    auto candidate = std::move(candidates.back());
    candidates.pop_back();
    if (candidate.invalidIOG)
      return;

    std::optional<Operation *> optOp = candidate.getNextOpToAdd();
    if (optOp == std::nullopt) {
      if (!candidate.invalidIOG) {
        finals.push_back(candidate.iog);
      }
      return;
    }
    Operation *op = *optOp;

    handleUnit(candidate, op);
  }

  // Handles an operation for a specific IOG candidate: the candidate is split
  // into one candidate per way in which the unit can be part of the IOG (see
  // `getIOGPortGroups`), and the candidates that contradict the channels
  // already decided are discarded
  void handleUnit(IOGCandidate candidate, Operation *op) {
    candidate.iog.units.insert(op);
    std::optional<IOGPortGroups> groups = getIOGPortGroups(op);
    if (!groups)
      return;
    // A unit without outputs in the IOG (e.g., a sink) takes no output group
    SmallVector<SmallVector<Value>> outputs = groups->outputs;
    if (outputs.empty())
      outputs.emplace_back();
    for (const SmallVector<Value> &in : groups->inputs) {
      for (const SmallVector<Value> &out : outputs) {
        IOGCandidate next = candidate;
        for (Value channel : in)
          next.markFollowed(channel);
        for (Value channel : out)
          next.markFollowed(channel);
        // The other channels of the unit are not part of the IOG
        for (Value channel : op->getOperands())
          if (!llvm::is_contained(in, channel))
            next.markIllegal(channel);
        for (Value channel : op->getResults())
          if (!llvm::is_contained(out, channel))
            next.markIllegal(channel);
        if (!next.invalidIOG)
          candidates.push_back(std::move(next));
      }
    }
  }

  inline bool isDone() { return candidates.empty(); }

  inline std::vector<IOG> getIOGs() {
    assert(isDone());
    return std::move(finals);
  }

private:
  std::vector<IOGCandidate> candidates;
  std::vector<IOG> finals;
};

} // namespace

static std::vector<IOG> findAllIOGsWithEntry(BlockArgument entry) {
  IOGFinder finder(entry);
  while (!finder.isDone()) {
    finder.step();
  }
  return finder.getIOGs();
}

std::vector<IOG> findAllIOGs(ModuleOp modOp) {
  std::vector<IOG> ret{};
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    // Each argument corresponds to an entry node.
    for (BlockArgument arg : funcOp.getRegion().getArguments()) {
      for (auto &iog : findAllIOGsWithEntry(arg)) {
        ret.push_back(std::move(iog));
      }
    }
  }
  return ret;
}

} // namespace dynamatic
