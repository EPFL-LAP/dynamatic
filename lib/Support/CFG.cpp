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
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include <queue>

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace dynamatic;

dynamatic::LogicBBs dynamatic::getLogicBBs(handshake::FuncOp funcOp) {
  dynamatic::LogicBBs logicBBs;
  for (auto &op : funcOp.getOps())
    if (auto bbAttr = op.getAttrOfType<mlir::IntegerAttr>(BB_ATTR); bbAttr)
      logicBBs.blocks[bbAttr.getValue().getZExtValue()].push_back(&op);
    else
      logicBBs.outOfBlocks.push_back(&op);
  return logicBBs;
}

bool dynamatic::inheritBB(Operation *srcOp, Operation *dstOp) {
  if (auto bb = srcOp->getAttrOfType<mlir::IntegerAttr>(BB_ATTR)) {
    dstOp->setAttr(BB_ATTR, bb);
    return true;
  }
  return false;
}

bool dynamatic::inheritBBFromValue(Value val, Operation *dstOp) {
  if (Operation *defOp = val.getDefiningOp())
    return inheritBB(defOp, dstOp);
  dstOp->setAttr(BB_ATTR,
                 OpBuilder(val.getContext()).getUI32IntegerAttr(ENTRY_BB));
  return true;
}

std::optional<unsigned> dynamatic::getLogicBB(Operation *op) {
  if (auto bb = op->getAttrOfType<mlir::IntegerAttr>(BB_ATTR))
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

// NOLINTBEGIN(misc-no-recursion)

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
  return isa<handshake::ForkOp, arith::ExtUIOp, arith::ExtSIOp,
             arith::TruncIOp>(op);
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

// NOLINTEND(misc-no-recursion)

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

bool dynamatic::cannotBelongToCFG(Operation *op) {
  return isa<handshake::MemoryOpInterface, handshake::SinkOp>(op);
}

HandshakeCFG::HandshakeCFG(circt::handshake::FuncOp funcOp) : funcOp(funcOp) {
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

  CFGPath pathSoFar;
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
    if (auto ctrlOfBB = ctrlVals.find(bb); ctrlOfBB != ctrlVals.end()) {
      if (ctrlOfBB->second != newCtrl) {
        return funcOp->emitError()
               << "Inconsistent control value identified for basic block "
               << bb;
      }
    }
    ctrlVals[bb] = newCtrl;
    return success();
  };

  // Explore the control network one operation at a time till we've iterated
  // over all of it
  Value ctrl = funcOp.getArguments().back();
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

// NOLINTNEXTLINE(misc-no-recursion)
void HandshakeCFG::findPathsTo(const CFGPath &pathSoFar, unsigned to,
                               SmallVector<CFGPath> &paths) {
  assert(!pathSoFar.empty() && "path cannot be empty");
  for (unsigned nextBB : successors[pathSoFar.back()]) {
    if (nextBB == to) {
      CFGPath &newPath = paths.emplace_back(pathSoFar);
      newPath.insert(to);
    } else if (!pathSoFar.contains(nextBB)) {
      CFGPath nextPathSoFar(pathSoFar);
      nextPathSoFar.insert(nextBB);
      findPathsTo(nextPathSoFar, to, paths);
    }
  }
}
