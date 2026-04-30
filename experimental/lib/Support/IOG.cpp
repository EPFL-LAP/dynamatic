#include "experimental/Support/IOG.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace dynamatic {
IOGPathSet::IOGPathSet(const IOG &iog, Operation *startA, Operation *endA)
    : start(startA), end(endA) {
  std::vector<Operation *> stack;
  stack.push_back(start);
  std::unordered_set<Operation *> forward;
  while (!stack.empty()) {
    Operation *op = stack.back();
    stack.pop_back();

    if (forward.find(op) != forward.end()) {
      continue;
    }

    forward.insert(op);

    if (op == end) {
      continue;
    }

    for (auto channel : op->getResults()) {
      if (!iog.contains(channel)) {
        continue;
      }
      Operation *next = channel.getUses().begin()->getOwner();
      assert(iog.contains(next));
      stack.push_back(next);
    }
  }

  stack.push_back(end);
  std::unordered_set<Operation *> back;

  while (!stack.empty()) {
    Operation *op = stack.back();
    stack.pop_back();

    if (back.find(op) != back.end()) {
      continue;
    }

    back.insert(op);

    if (op == start) {
      continue;
    }

    for (auto channel : op->getOperands()) {
      if (!iog.contains(channel)) {
        continue;
      }
      Operation *next = channel.getDefiningOp();
      if (next == nullptr) {
        continue;
      }
      assert(iog.contains(next));
      stack.push_back(next);
    }
  }

  for (Operation *op : forward) {
    if (back.find(op) != back.end()) {
      units.insert(op);
    }
  }
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

using namespace dynamatic::handshake;
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
  // or splitting into multiple possible possibilities as necessary
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

  // Handles an operation for a specific IOG candidate by:
  //
  // 1. Inserting the unit into the IOG
  // 2. Marking necessary edges as followed (e.g. all merge inputs)
  // 3. Splitting into multiple potential candidates if necessary: For example,
  // only a single fork output can be followed, so there is a different possible
  // IOG for each output
  void handleUnit(IOGCandidate candidate, Operation *op) {
    candidate.iog.units.insert(op);
    if (auto endOp = dyn_cast<EndOp>(op)) {
      followSingle(candidate, endOp.getOperands());
    } else if (auto bufOp = dyn_cast<BufferOp>(op)) {
      candidate.markFollowed(bufOp.getOperand());
      candidate.markFollowed(bufOp.getResult());
      candidates.push_back(candidate);
    } else if (auto condBr = dyn_cast<ConditionalBranchOp>(op)) {
      candidate.markFollowed(condBr.getTrueResult());
      candidate.markFollowed(condBr.getFalseResult());
      followSingle(candidate, condBr.getOperands());
    } else if (auto forkOp = dyn_cast<ForkOp>(op)) {
      candidate.markFollowed(forkOp.getOperand());
      followSingle(candidate, forkOp.getResults());
    } else if (auto muxOp = dyn_cast<MuxOp>(op)) {
      // Either all data inputs, or the select input. Output always
      candidate.markFollowed(muxOp.getResult());
      auto dataFollower = candidate;
      auto selectFollower = candidate;
      for (auto data : muxOp.getDataOperands()) {
        dataFollower.markFollowed(data);
        selectFollower.markIllegal(data);
      }

      dataFollower.markIllegal(muxOp.getSelectOperand());
      selectFollower.markFollowed(muxOp.getSelectOperand());

      candidates.push_back(dataFollower);
      candidates.push_back(selectFollower);
    } else if (auto cmerge = dyn_cast<ControlMergeOp>(op)) {
      for (auto data : cmerge.getDataOperands()) {
        candidate.markFollowed(data);
      }
      followSingle(candidate, cmerge.getResults());
    } else if (auto arithOp = dyn_cast<ArithOpInterface>(op)) {
      candidate.markFollowed(arithOp->getResults()[0]);
      followSingle(candidate, arithOp->getOperands());
    } else if (isa<SinkOp, StoreOp>(op)) {
      // These act as terminating units, and no further edges need to be
      // followed
      candidates.push_back(candidate);
    } else if (auto loadOp = dyn_cast<LoadOp>(op)) {
      candidate.markFollowed(loadOp.getAddress());
      // checkpoint.follow(loadOp.getAddressResult());
      // checkpoint.follow(loadOp.getData());
      candidate.markFollowed(loadOp.getDataResult());
      candidates.push_back(candidate);
    } else if (isa<SourceOp, MemoryControllerOp>(op)) {
      // SourceOps can not be part of an IOG, as they act like entry nodes. This
      // means that the partial IOGs looking at these operations can be
      // discarded, so nothing needs to be done
      //
      // MemoryControllers are only accessed by control signals during this
      // annotation, as the edges between loadOps/storeOps and the MC are not
      // added to the IOG. These control signals usually grant trivial IOGs that
      // can be skipped.
    } else {
      op->emitError("unhandled case in IOG finding");
    }
  }

  inline bool isDone() { return candidates.empty(); }

  inline std::vector<IOG> getIOGs() {
    assert(isDone());
    return std::move(finals);
  }

private:
  // This function takes a checkpoint and a list of channels. For each channel
  // c, it pushes the IOG where *only* channel c is taken, and all the rest are
  // illegal. Used for forks (which only allow exactly one of its outputs) and
  // join operations (which only allow exactly one of its inputs)
  template <typename T>
  void followSingle(const IOGCandidate &checkpoint, T options) {
    size_t n = 0;
    mlir::Value singleTaken = nullptr;
    for (mlir::Value channel : options) {
      if (checkpoint.iog.contains(channel)) {
        n += 1;
        singleTaken = channel;
      }
    }
    if (n == 1) {
      IOGCandidate goHere = checkpoint;
      // singleTaken is already part of the IOG, so no need to follow it
      // goHere.follow(singleTaken);
      for (mlir::Value other : options) {
        if (other == singleTaken) {
          continue;
        }
        goHere.markIllegal(other);
      }
      candidates.push_back(goHere);
      return;
    }
    if (n > 1) {
      // Multiple paths taken, but only single is allowed! => No IOGs from this
      // branch
      return;
    }
    for (mlir::Value channel : options) {
      IOGCandidate goHere = checkpoint;
      goHere.markFollowed(channel);
      for (mlir::Value other : options) {
        if (channel == other)
          continue;
        goHere.markIllegal(other);
      }
      candidates.push_back(goHere);
    }
  }

  std::vector<IOGCandidate> candidates;
  std::vector<IOG> finals;
};

std::vector<IOG> findAllIOGsWithEntry(BlockArgument entry) {
  IOGFinder finder(entry);
  while (!finder.isDone()) {
    finder.step();
  }
  return finder.getIOGs();
}

} // namespace

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
