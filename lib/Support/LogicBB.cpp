//===- LogicBB.cpp - Infrastructure for working w/ logical BBs --*- C++ -*-===//
//
// Implements the infrastructure useful for handling logical basic
// blocks (logical BBs) in Handshake functions.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/LogicBB.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

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
static bool areOpsInSameBlock(SmallVector<Operation *, 4> &mergeOps) {
  std::optional<unsigned> uniqueBB;
  for (Operation *op : mergeOps) {
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

Operation *dynamatic::backtrackToBranch(Operation *op) {
  do {
    if (!op)
      break;
    if (isa<handshake::BranchOp, handshake::ConditionalBranchOp>(op))
      return op;
    if (isa<handshake::ForkOp, arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp>(
            op))
      op = op->getOperand(0).getDefiningOp();
    else
      break;
  } while (true);
  return nullptr;
}

// NOLINTNEXTLINE(misc-no-recursion)
Operation *dynamatic::followToMerge(Operation *op) {
  if (isa<handshake::MergeLikeOpInterface>(op))
    return op;
  if (isa<handshake::ForkOp, arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp>(
          op)) {
    SmallVector<Operation *, 4> mergeOps;
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

bool dynamatic::isBackedge(Value val, Operation *user) {
  assert(llvm::find(val.getUsers(), user) != val.getUsers().end() &&
         "expected user to have val as operand");
  Operation *brOp = backtrackToBranch(val.getDefiningOp());
  if (!brOp)
    return false;
  Operation *mergeOp = followToMerge(user);
  if (!mergeOp)
    return false;

  std::optional<unsigned> srcBB = getLogicBB(brOp);
  std::optional<unsigned> dstBB = getLogicBB(mergeOp);
  if (!srcBB.has_value() || !dstBB.has_value())
    return false;
  return *srcBB >= *dstBB;
}

bool dynamatic::isBackedge(Value val) {
  auto users = val.getUsers();
  assert(std::distance(users.begin(), users.end()) == 1 &&
         "value must have a single user");
  return isBackedge(val, *users.begin());
}
