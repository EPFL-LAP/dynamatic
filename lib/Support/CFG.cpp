//===- CFG.cpp - CFG-related analysis and helpers ---------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the infrastructure for CFG-style analysis in Handshake functions.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/CFG.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

dynamatic::LogicBBs dynamatic::getLogicBBs(handshake::FuncOp funcOp) {
  dynamatic::LogicBBs logicBBs;
  for (auto &op : funcOp.getOps())
    if (auto bbAttr = op.getAttrOfType<mlir::IntegerAttr>(BB_ATTR_NAME); bbAttr)
      logicBBs.blocks[bbAttr.getValue().getZExtValue()].push_back(&op);
    else
      logicBBs.outOfBlocks.push_back(&op);
  return logicBBs;
}

void dynamatic::setBB(Operation *op, int bb) {
  auto ui32 = IntegerType::get(op->getContext(), 32,
                               IntegerType::SignednessSemantics::Unsigned);
  auto attr = IntegerAttr::get(ui32, bb);
  op->setAttr(BB_ATTR_NAME, attr);
}

bool dynamatic::inheritBB(Operation *srcOp, Operation *dstOp) {
  if (auto bb = srcOp->getAttrOfType<mlir::IntegerAttr>(BB_ATTR_NAME)) {
    dstOp->setAttr(BB_ATTR_NAME, bb);
    return true;
  }
  return false;
}

bool dynamatic::inheritBBFromValue(Value val, Operation *dstOp) {
  if (Operation *defOp = val.getDefiningOp())
    return inheritBB(defOp, dstOp);
  dstOp->setAttr(BB_ATTR_NAME,
                 OpBuilder(val.getContext()).getUI32IntegerAttr(ENTRY_BB));
  return true;
}

std::optional<unsigned> dynamatic::getLogicBB(Operation *op) {
  if (auto bb = op->getAttrOfType<mlir::IntegerAttr>(BB_ATTR_NAME))
    return bb.getUInt();
  return {};
}

/// Determines whether all operations are in the same basic block. If any
/// operation has no basic block attached to it, returns false.
static bool areOpsInSameBlock(SmallVector<Operation *> &ops) {
  std::optional<unsigned> uniqueBB;
  for (Operation *op : ops) {
    std::optional<unsigned> bb = getLogicBB(op);
    if (!bb.has_value())
      return false;
    if (!uniqueBB.has_value())
      uniqueBB = bb;
    else if (*uniqueBB != bb)
      return false;
  }
  return true;
}

/// Attempts to identify an operation's predecessor block, which is either the
/// block the operation belongs to, or (if the latter isn't defined), the unique
/// first block reached by recursively backtracking through the def-use chain of
/// the operation's operands. On successfull identification of the predecessor
/// block, returns true and sets the block; otherwise returns false.
static bool backtrackToBlock(Operation *op, unsigned &bb,
                             SmallPtrSet<Operation *, 4> &visited) {
  if (!op)
    return false;

  // Succeeds if the operation belongs to a basic block
  std::optional<unsigned> optBB = getLogicBB(op);
  if (optBB.has_value()) {
    bb = *optBB;
    return true;
  }

  // Avoid looping between operations outside blocks
  if (auto [_, newOp] = visited.insert(op); !newOp)
    return false;

  // Succeeds if by backtracking through all operands we reach a single block
  bool firstOperand = true;
  for (Value opr : op->getOperands()) {
    unsigned operandBB;
    if (!backtrackToBlock(opr.getDefiningOp(), operandBB, visited))
      return false;
    if (firstOperand) {
      firstOperand = false;
      bb = operandBB;
    } else if (bb != operandBB) {
      return false;
    }
  }
  return !firstOperand;
}

/// Attempts to identify an operation's successor block, which is either the
/// block the operation belongs to, or (if the latter isn't defined), the unique
/// first block reached by recursively following through the uses of the
/// operation's results. On successfull identification of the successor block,
/// returns true and sets the block; otherwise returns false.
static bool followToBlock(Operation *op, unsigned &bb,
                          SmallPtrSet<Operation *, 4> &visited) {
  // Succeeds if the operation belongs to a basic block
  std::optional<unsigned> optBB = getLogicBB(op);
  if (optBB.has_value()) {
    bb = *optBB;
    return true;
  }

  // Avoid looping between operations outside blocks
  if (auto [_, newOp] = visited.insert(op); !newOp)
    return false;

  // Succeeds if by following all results through their users we reach a single
  // block
  bool firstOperand = true;
  for (OpResult res : op->getResults()) {
    for (Operation *user : res.getUsers()) {
      unsigned resultBB;
      if (!followToBlock(user, resultBB, visited))
        return false;
      if (firstOperand) {
        firstOperand = false;
        bb = resultBB;
      } else if (bb != resultBB) {
        return false;
      }
    }
  }
  return !firstOperand;
}

/// Determines whether the operation is of a nature which can be traversed
/// outside blocks during backedge identification.
static inline bool canGoThroughOutsideBlocks(Operation *op) {
  return isa<handshake::ForkOp, handshake::ExtUIOp, handshake::ExtSIOp,
             handshake::TruncIOp>(op);
}

/// Attempts to backtrack through forks and bitwidth modification operations
/// till reaching a branch-like operation. On success, returns the branch-like
/// operation that was backtracked to (or the passed operation if it was itself
/// branch-like); otherwise, returns nullptr.
static Operation *backtrackToBranch(Operation *op) {
  do {
    if (!op)
      break;
    if (isa<handshake::BranchOp, handshake::ConditionalBranchOp>(op))
      return op;
    if (canGoThroughOutsideBlocks(op))
      op = op->getOperand(0).getDefiningOp();
    else
      break;
  } while (true);
  return nullptr;
}

/// Attempts to follow the def-use chains of all the operation's results through
/// forks and bitwidth modification operations till reaching merge-like
/// operations that all belong to the same basic block. On success, returns one
/// of the merge-like operations reached by a def-use chain (or the passed
/// operation if it was itself merge-like); otherwise, returns nullptr.
static Operation *followToMerge(Operation *op) {
  if (isa<handshake::MergeLikeOpInterface>(op))
    return op;
  if (canGoThroughOutsideBlocks(op)) {
    // All users of the operation's results must lead to merges within a unique
    // block
    SmallVector<Operation *> mergeOps;
    for (OpResult res : op->getResults()) {
      for (Operation *user : res.getUsers()) {
        Operation *op = followToMerge(user);
        if (!op)
          return nullptr;
        mergeOps.push_back(op);
      }
    }
    if (mergeOps.empty() || !areOpsInSameBlock(mergeOps))
      return nullptr;
    return mergeOps.front();
  }
  return nullptr;
}

bool dynamatic::getBBEndpoints(Value val, Operation *user,
                               BBEndpoints &endpoints) {
  assert(llvm::find(val.getUsers(), user) != val.getUsers().end() &&
         "expected user to have val as operand");

  SmallPtrSet<Operation *, 4> visited;
  Operation *defOp = val.getDefiningOp();
  unsigned srcBB, dstBB;
  if (!backtrackToBlock(defOp, srcBB, visited))
    return false;

  // Reuse the same (cleared) visited set to avoid recreating the object
  visited.clear();
  if (!followToBlock(user, dstBB, visited))
    return false;

  endpoints.srcBB = srcBB;
  endpoints.dstBB = dstBB;
  return true;
}

bool dynamatic::getBBEndpoints(Value val, BBEndpoints &endpoints) {
  auto users = val.getUsers();
  assert(std::distance(users.begin(), users.end()) == 1 &&
         "value must have a single user");
  return getBBEndpoints(val, *users.begin(), endpoints);
}

bool dynamatic::isBackedge(Value val, Operation *user, BBEndpoints *endpoints) {
  // Get the value's BB endpoints
  BBEndpoints bbs;
  if (!getBBEndpoints(val, user, bbs))
    return false;
  if (endpoints)
    *endpoints = bbs;

  // Block IDs are ordered in original program order, so an edge going from a
  // higher block ID to a lower block ID is a backedge
  if (bbs.srcBB > bbs.dstBB)
    return true;
  if (bbs.srcBB < bbs.dstBB)
    return false;

  // If both source and destination blocks are identical, the edge must be
  // located between a branch-like operation and a merge-like operation
  Operation *brOp = backtrackToBranch(val.getDefiningOp());
  if (!brOp)
    return false;
  Operation *mergeOp = followToMerge(user);
  if (!mergeOp)
    return false;

  // Check that the branch and merge are part of the same block indicated by the
  // edge's BB endpoints (should be the case in all non-degenerate cases)
  std::optional<unsigned> brBB = getLogicBB(brOp);
  std::optional<unsigned> mergeBB = getLogicBB(mergeOp);
  return brBB.has_value() && mergeBB.has_value() && *brBB == *mergeBB &&
         *brBB == bbs.srcBB;
}

bool dynamatic::isBackedge(Value val, BBEndpoints *endpoints) {
  auto users = val.getUsers();
  assert(std::distance(users.begin(), users.end()) == 1 &&
         "value must have a single user");
  return isBackedge(val, *users.begin(), endpoints);
}

namespace {
/// Define a Control-Flow Graph Edge as a OpOperand
using CFGEdge = OpOperand;

/// Define a comparator between BBEndpoints
struct EndpointComparator {
  bool operator()(const BBEndpoints &a, const BBEndpoints &b) const {
    if (a.srcBB != b.srcBB)
      return a.srcBB < b.srcBB;
    return a.dstBB < b.dstBB;
  }
};

} // namespace

/// Returns the endpoints of the given CFGEdge.
/// If a BB is not specified for each endpoint, it is set to nullopt.
static BBEndpointsOptional getCFGEdgeEndpoints(const CFGEdge &edge) {
  BBEndpointsOptional endpoints;

  Operation *definingOp = edge.get().getDefiningOp();
  if (!definingOp) {
    endpoints.srcBB = std::nullopt;
  } else {
    if (auto bb = getLogicBB(definingOp)) {
      endpoints.srcBB = *bb;
    } else {
      endpoints.srcBB = std::nullopt;
    }
  }

  // Owner always exists, unlike the defining op.
  if (auto bb = getLogicBB(edge.getOwner())) {
    endpoints.dstBB = *bb;
  } else {
    endpoints.dstBB = std::nullopt;
  }

  return endpoints;
}

BBtoArcsMap dynamatic::getBBPredecessorArcs(handshake::FuncOp funcOp) {
  // Join all predecessors of a BB
  BBtoArcsMap predecessorArcs;

  // Traverse all operations within funcOp to find edges between BBs, including
  // self-edges, and save them in a map from the Endpoints to the edges
  funcOp->walk([&](Operation *op) {
    for (CFGEdge &edge : op->getOpOperands()) {
      BBEndpointsOptional endpoints = getCFGEdgeEndpoints(edge);
      // The dstBB should be always defined (to be consistent with
      // "BBPredecessorArcs")
      if (!endpoints.dstBB.has_value())
        continue;

      // Store the edge if it is a Backedge or connects two different BBs
      if (isBackedge(edge.get(), op) || endpoints.srcBB != endpoints.dstBB) {
        bool arcExists = false;
        for (BBArc &arc : predecessorArcs[*endpoints.dstBB]) {
          if (arc.srcBB == endpoints.srcBB) {
            // If the arc already exists, add the edge to it
            arc.edges.insert(&edge);
            arcExists = true;
            break;
          }
        }
        if (!arcExists) {
          // Create a new arc.
          BBArc arc;
          arc.srcBB = endpoints.srcBB;
          arc.dstBB = endpoints.dstBB;
          arc.edges.insert(&edge);
          predecessorArcs[*endpoints.dstBB].push_back(arc);
        }
      }
    }
  });

  return predecessorArcs;
}

bool dynamatic::cannotBelongToCFG(Operation *op) {
  return isa<handshake::MemoryOpInterface, handshake::SinkOp>(op);
}

HandshakeCFG::HandshakeCFG(handshake::FuncOp funcOp) : funcOp(funcOp) {
  for (Operation &op : funcOp.getOps()) {
    if (cannotBelongToCFG(&op))
      continue;

    // Get the source basic block
    std::optional<unsigned> srcBB = getLogicBB(&op);
    assert(srcBB && "source operation must belong to block");

    for (OpResult res : op.getResults()) {
      for (Operation *user : res.getUsers()) {
        if (cannotBelongToCFG(user))
          continue;

        // Get the destination basic block and store the connection
        std::optional<unsigned> dstBB = getLogicBB(user);
        assert(dstBB && "destination operation must belong to block");
        if (*srcBB != *dstBB || isBackedge(res, user))
          successors[*srcBB].insert(*dstBB);
      }
    }
  }
}

void HandshakeCFG::getNonCyclicPaths(unsigned from, unsigned to,
                                     SmallVector<CFGPath> &paths) {
  // Both blocks must exist in the CFG
  assert(successors.contains(from) && "source block must exist in the CFG");
  assert(successors.contains(to) && "destination block must exist in the CFG");

  mlir::SetVector<unsigned> pathSoFar;
  pathSoFar.insert(from);
  findPathsTo(pathSoFar, to, paths);
}

LogicalResult
HandshakeCFG::getControlValues(DenseMap<unsigned, Value> &ctrlVals) {
  // Maintain a list of operations on the control path as we go through the
  // circuit
  DenseSet<Operation *> exploredOps;
  mlir::SetVector<Operation *> ctrlOps;

  // Adds the users of an operation's results to the list of control operations,
  // except if those have already been explored.
  auto addToCtrlOps = [&](auto users) {
    for (Operation *userOp : users) {
      if (!exploredOps.contains(userOp))
        ctrlOps.insert(userOp);
    }
  };

  // Updates the control value registered for a specific block in the map.
  // Fails if a control already existed and is different from the new value,
  // succeeds otherwise.
  auto updateCtrl = [&](unsigned bb, Value newCtrl) -> LogicalResult {
    if (auto [it, newBB] = ctrlVals.insert({bb, newCtrl}); !newBB) {
      if (it->second != newCtrl)
        return failure();
    }
    return success();
  };

  // Explore the control network one operation at a time till we've iterated
  // over all of it
  Value ctrl = funcOp.getArguments().back();
  if (failed(updateCtrl(ENTRY_BB, ctrl))) {
    assert(false && "cannot set control for entry BB");
  }

  addToCtrlOps(ctrl.getUsers());
  while (!ctrlOps.empty()) {
    Operation *ctrlOp = ctrlOps.pop_back_val();
    exploredOps.insert(ctrlOp);

    // Do not care about operations that are outside the CFG
    if (cannotBelongToCFG(ctrlOp))
      continue;

    // Guaranteed to succeed because of asserts in class constructor
    unsigned bb = *getLogicBB(ctrlOp);

    // This kills of all paths going out of the control network by ignoring all
    // operation types that do not forward the control signal to at least one of
    // their outputs
    LogicalResult res =
        llvm::TypeSwitch<Operation *, LogicalResult>(ctrlOp)
            .Case<handshake::ForkOp, handshake::LazyForkOp, handshake::BufferOp,
                  handshake::BranchOp, handshake::ConditionalBranchOp>(
                [&](auto) {
                  addToCtrlOps(ctrlOp->getUsers());
                  return success();
                })
            .Case<handshake::MergeLikeOpInterface>([&](auto) {
              OpResult mergeRes = ctrlOp->getResult(0);
              addToCtrlOps(mergeRes.getUsers());
              return updateCtrl(bb, mergeRes);
            })
            .Default([&](auto) { return success(); });
    if (failed(res))
      return failure();
  }

  return success();
}

void HandshakeCFG::findPathsTo(const mlir::SetVector<unsigned> &pathSoFar,
                               unsigned to, SmallVector<CFGPath> &paths) {
  assert(!pathSoFar.empty() && "path cannot be empty");
  for (unsigned nextBB : successors[pathSoFar.back()]) {
    if (nextBB == to) {
      CFGPath newPath;
      llvm::copy(pathSoFar, std::back_inserter(newPath));
      newPath.push_back(to);
      paths.push_back(newPath);
    } else if (!pathSoFar.contains(nextBB)) {
      mlir::SetVector<unsigned> nextPathSoFar(pathSoFar);
      nextPathSoFar.insert(nextBB);
      findPathsTo(nextPathSoFar, to, paths);
    }
  }
}

namespace {
/// Internal representation for the failure/success status of a GIID
/// backtracking process. We distinguish two types of failures:
/// - An off-path failure, meaning the function went off the CFG path provided
/// for the search before reaching the predecessor's block. This type of failure
/// is generally "recoverable", in the sense that the top-level search may still
/// succeed even if some recursive search fails in this way.
/// - An on-path failure, meaning the function failed to reach the predecessor
/// all-the-while staying on the path to it. This type of failure is generally
/// "terminating", in the sense that the top-level search will probably fail
/// as soon as some recursive search fails in this way.
enum class GIIDStatus {
  /// Denotes an off-path failure.
  FAIL_OFF_PATH,
  /// Denotes an on-path failure.
  FAIL_ON_PATH,
  /// Denotes a success.
  SUCCEED
};

using GIIDFoldFunc = const std::function<GIIDStatus(Value)> &;
} // namespace

/// Logical "or" operator for the GIID status. Combines the returns values of
/// the callback on all operands into a single GIID status.
///
/// To succeed, it is enough that one operand reaches the predecessor on the CFG
/// path. If all operands are off path, then report an off-path failure. If no
/// operand reaches the predecessor but at least one is on path, report an
/// on-path failure.
// NOLINTNEXTLINE (remove warning saying the function will not be emitted)
static GIIDStatus foldGIIDStatusOr(GIIDFoldFunc func, ValueRange operands) {
  if (operands.empty())
    return GIIDStatus::FAIL_OFF_PATH;

  GIIDStatus stat = GIIDStatus::FAIL_OFF_PATH;
  for (Value newVal : operands) {
    switch (func(newVal)) {
    case GIIDStatus::FAIL_OFF_PATH:
      // Do nothing here, so that if all operands are off path we end up
      // reporting an off-path failure
      break;
    case GIIDStatus::FAIL_ON_PATH:
      // Unless another operand reaches the predecessor, we will end up
      // reporting an on-path failure
      stat = GIIDStatus::FAIL_ON_PATH;
      break;
    case GIIDStatus::SUCCEED:
      // Early return when one of the datapaths reaches the predecessor
      return GIIDStatus::SUCCEED;
    }
  }
  return stat;
}

/// Logical "and" operator for the GIID status. Combines the returns values of
/// the callback on all operands into a single GIID status.
///
/// To succeed, at least one operand must reach the predecessor on the CFG path,
/// and none must fail to reach the predecessor on the path. If all operands are
/// off path, then report an off-path failure.
static GIIDStatus foldGIIDStatusAnd(GIIDFoldFunc func, ValueRange operands) {
  if (operands.empty())
    return GIIDStatus::FAIL_OFF_PATH;

  GIIDStatus stat = GIIDStatus::FAIL_OFF_PATH;
  for (Value newVal : operands) {
    switch (func(newVal)) {
    case GIIDStatus::FAIL_OFF_PATH:
      // Do nothing here, so that if all operands are off path we end up
      // reporting an off-path failure
      break;
    case GIIDStatus::FAIL_ON_PATH:
      // All datapaths on the CFG path must connect to the predecessor. Early
      // return when we fail on the path.
      return GIIDStatus::FAIL_ON_PATH;
    case GIIDStatus::SUCCEED:
      // Unless another operand is on the path but fails to reach the
      // predecessor, this call will succeed
      stat = GIIDStatus::SUCCEED;
      break;
    }
  }
  return stat;
}

/// Recursive version of `isGIID`. The CFG path is passed as an array reference
/// to allow us to efficiently drop elements from its back as we backtrack the
/// circuit on the CFG path. At any point, the path's last block is the one
/// which the operand's owning operation was a part of on the parent call to
/// `isGIIDRec`.
static GIIDStatus isGIIDRec(Value predecessor, OpOperand &oprd,
                            ArrayRef<unsigned> path) {
  Value val = oprd.get();
  if (predecessor == val)
    return GIIDStatus::SUCCEED;

  // The defining operation must exist, otherwise it means we have reached
  // function arguments without encountering the predecessor value. It must also
  // belong to a block
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return GIIDStatus::FAIL_ON_PATH;
  std::optional<unsigned> defBB = getLogicBB(defOp);
  if (!defBB)
    return GIIDStatus::FAIL_ON_PATH;

  size_t pathSize = path.size();
  if (isBackedge(val, oprd.getOwner())) {
    // Backedges always indicate transitions from one block on the path to its
    // predecessor

    // If we move past the first block in the path, so it's an "on-path failure"
    if (pathSize == 1)
      return GIIDStatus::FAIL_ON_PATH;

    // The previous block in the path must be the one the defining operation
    // belongs to, otherwise it's an "off-path failure"
    if (path.drop_back().back() != defBB)
      return GIIDStatus::FAIL_OFF_PATH;
    path = path.drop_back();
  } else {
    // The defining operation must be somewhere earlier in the path than before.
    // We allow the path to "jump over" BBs, since datapaths of optimized
    // circuits will sometimes skip BBs entirely

    bool foundOnPath = false;
    for (size_t idx = pathSize; idx > 0; --idx) {
      if (path[idx - 1] == *defBB) {
        foundOnPath = true;
        path = path.take_front(idx);
        break;
      }
    }
    if (!foundOnPath) {
      // If we had previously reached the block where the predecessor is defined
      // and moved past it, the failure is "on path". If we failed earlier, the
      // failure is "off" path.
      /// NOTE: Is that last part true? Is it possible to jump over the block
      /// defining the predecessor and still be on path?
      return pathSize == 1 ? GIIDStatus::FAIL_ON_PATH
                           : GIIDStatus::FAIL_OFF_PATH;
    }
  }

  // Recursively calls the function with a new value as second argument (meant
  // to be an operand of the defining operation identified above) and changing
  // the current defining operation to be the current one
  auto recurse = [&](Value newVal) -> GIIDStatus {
    for (OpOperand &oprd : defOp->getOpOperands()) {
      if (oprd.get() == newVal)
        return isGIIDRec(predecessor, oprd, path);
    }
    llvm_unreachable("recursive call should be on operand of defining op");
  };

  // The backtracking logic depends on the type of the defining operation
  return llvm::TypeSwitch<Operation *, GIIDStatus>(defOp)
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp condBrOp) {
            // The data operand or the condition operand must depend on the
            // predecessor
            return foldGIIDStatusAnd(recurse, condBrOp->getOperands());
          })
      .Case<handshake::MergeOp, handshake::ControlMergeOp>([&](auto) {
        // The data input on the path must depend on the predecessor
        return foldGIIDStatusAnd(recurse, defOp->getOperands());
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        // If the select operand depends on the predecessor, then the mux
        // depends on the predecessor
        if (recurse(muxOp.getSelectOperand()) == GIIDStatus::SUCCEED)
          return GIIDStatus::SUCCEED;

        // Otherwise, data inputs on the path must depend on the predecessor
        return foldGIIDStatusAnd(recurse, defOp->getOperands());
      })
      .Case<handshake::LoadOp>([&](handshake::LoadOp loadOp) {
        if (loadOp.getDataResult() != val)
          return GIIDStatus::FAIL_ON_PATH;

        // If the address operand depends on the predecessor then the data
        // result depends on the predecessor
        return recurse(loadOp.getAddress());
      })
      .Case<handshake::SelectOp>([&](handshake::SelectOp selectOp) {
        // Similarly to the mux, if the select operand depends on the
        // predecessor, then the select depends on the predecessor
        if (recurse(selectOp.getCondition()) == GIIDStatus::SUCCEED)
          return GIIDStatus::SUCCEED;

        // The select's true value or false value must depend on the predecessor
        llvm::SmallVector<Value> values{selectOp.getTrueValue(),
                                        selectOp.getFalseValue()};
        return foldGIIDStatusAnd(recurse, values);
      })
      .Case<handshake::ForkOp, handshake::LazyForkOp, handshake::BufferOp,
            handshake::BranchOp, handshake::AddIOp, handshake::AndIOp,
            handshake::CmpIOp, handshake::DivSIOp, handshake::DivUIOp,
            handshake::ExtSIOp, handshake::ExtUIOp, handshake::MulIOp,
            handshake::OrIOp, handshake::ShLIOp, handshake::ShRUIOp,
            handshake::SubIOp, handshake::TruncIOp, handshake::XOrIOp,
            handshake::AddFOp, handshake::CmpFOp, handshake::DivFOp,
            handshake::MulFOp, handshake::SubFOp>([&](auto) {
        // At least one operand must depend on the predecessor
        return foldGIIDStatusOr(recurse, defOp->getOperands());
      })
      .Default([&](auto) {
        // To err on the conservative side, produce the most terminating kind of
        // failure on encoutering an unsupported operation
        return GIIDStatus::FAIL_ON_PATH;
      });
}

bool dynamatic::isGIID(Value predecessor, OpOperand &oprd, CFGPath &path) {
  assert(path.size() >= 2 && "path must have at least two blocks");
  return isGIIDRec(predecessor, oprd, path) == GIIDStatus::SUCCEED;
}

bool dynamatic::isChannelOnCycle(mlir::Value channel) {
  llvm::SmallPtrSet<mlir::Value, 32> visited;

  std::function<bool(mlir::Value, bool)> dfs =
      [&](mlir::Value current, bool isStart) -> bool {
    if (!isStart && current == channel)
      return true;

    if (visited.contains(current))
      return false;
    visited.insert(current);

    for (mlir::Operation *user : current.getUsers()) {
      if (isa<handshake::MemoryControllerOp, handshake::LSQOp>(user))
        continue;
      for (mlir::Value next : user->getResults()) {
        if (dfs(next, false))
          return true;
      }
    }
    return false;
  };

  return dfs(channel, true);
}