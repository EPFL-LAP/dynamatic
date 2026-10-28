//===- HandshakeStraightToQueue.cpp - Implement S2Q algorithm -*- C++ -*---===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass which allows to implement straight to the
// queue, a different way of allocating basic blocks in the LSQ, based on an
// ASAP approach rather than relying on the network of cmerges.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "experimental/Support/CFGAnnotation.h"
#include "experimental/Support/FtdImplementation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <map>
#include <stack>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

#define DEBUG_TYPE "handshake-straight-to-queue"

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_HANDSHAKESTRAIGHTTOQUEUE
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

namespace {

struct ProdConsMemDep {
  Block *prodBb;
  Block *consBb;

  ProdConsMemDep(Block *prod, Block *cons) : prodBb(prod), consBb(cons) {}
  /// Print the dependency stored in the current relationship
  void print();
};

/// A group represents all the memory operations belonging to the same basic
/// block which require the same LSQ. It contains a reference to the BB, a set
/// of predecessor in the dependence graph and a set of successors.
struct MemoryGroup {
  // The BB the group defines
  Block *bb;

  // List of predecessors of the group
  DenseSet<MemoryGroup *> preds;
  // List of successors of the group
  DenseSet<MemoryGroup *> succs;

  // Constructor for the group
  MemoryGroup(Block *b) : bb(b) {}

  /// Print the dependenices of the curent group
  void print();
};

void ProdConsMemDep::print() {
  llvm::dbgs() << "[PROD_CONS_MEM_DEP] Dependency from [";
  prodBb->printAsOperand(llvm::dbgs());
  llvm::dbgs() << "] to [";
  consBb->printAsOperand(llvm::dbgs());
  llvm::dbgs() << "]\n";
}

void MemoryGroup::print() {
  llvm::dbgs() << "[MEM_GROUP] Group for [";
  bb->printAsOperand(llvm::dbgs());
  llvm::dbgs() << "]; predecessors = {";
  for (auto &gp : preds) {
    gp->bb->printAsOperand(llvm::dbgs());
    llvm::dbgs() << ", ";
  }
  llvm::dbgs() << "}; successors = {";
  for (auto &gp : succs) {
    gp->bb->printAsOperand(llvm::dbgs());
    llvm::dbgs() << ", ";
  }
  llvm::dbgs() << "} \n";
}

/// Given a list of operations, return the list of memory dependencies for
/// each block. This allows to build the group graph, which allows to
/// determine the dependencies between memory access inside basic blocks.
/// Two types of hazards between the predecessors of one LSQ node:
/// (1) WAW between 2 Store operations,
/// (2) RAW and WAR between Load and Store operations
static SmallVector<ProdConsMemDep> identifyMemoryDependencies(
    handshake::FuncOp &funcOp,
    const SmallVector<handshake::MemPortOpInterface> &operations) {

  mlir::DominanceInfo domInfo;
  ftd::BlockIndexing bi(funcOp.getRegion());
  mlir::CFGLoopInfo li(domInfo.getDomTree(&funcOp.getRegion()));

  SmallVector<ProdConsMemDep> allMemDeps;

  // Returns true if there exist a path between `op1` and `op2`
  auto isThereAPath = [&](Operation *op1, Operation *op2) -> bool {
    return !findAllPaths(op1->getBlock(), op2->getBlock(), bi).empty();
  };

  // Returns true if two operations are both load
  auto areBothLoad = [](Operation *op1, Operation *op2) {
    return (isa<handshake::LoadOp>(op1) && isa<handshake::LoadOp>(op2));
  };

  // Returns true if two operations belong to the same block
  auto isSameBlock = [](Operation *op1, Operation *op2) {
    return (op1->getBlock() == op2->getBlock());
  };

  // Given all the operations which are assigned to an LSQ, loop over them
  // and skip those which are not memory operations
  for (handshake::MemPortOpInterface i : operations) {

    // Loop over all the other operations in the LSQ. There is no dependency
    // in the following cases:
    // 1. One of them is not a memory operation;
    // 2. The two operations are in the same group, thus they are in the same
    // BB;
    // 3. They are both load operations;
    // 4. The operations are mutually exclusive (i.e. there is no path which
    // goes from i to j and vice-versa);
    for (handshake::MemPortOpInterface j : operations) {

      if (isSameBlock(i, j) || areBothLoad(i, j) ||
          (!isThereAPath(i, j) && !isThereAPath(j, i)))
        continue;

      // Get the two blocks
      Block *bbI = i->getBlock(), *bbJ = j->getBlock();

      // If the relationship was already present, then skip the pairs of
      // blocks
      if (llvm::find_if(allMemDeps, [bbI, bbJ](ProdConsMemDep p) {
            return p.prodBb == bbJ && p.consBb == bbI;
          }) != allMemDeps.end())
        continue;

      // Insert a dependency only if index _j is smaller than index _i: in
      // this case i is the producer, j is the consumer. If this doesn't
      // hold, the dependency will be added when the two blocks are analyzed
      // in the opposite direction
      if (bi.isLess(bbJ, bbI)) {
        allMemDeps.push_back(ProdConsMemDep(bbJ, bbI));

        // If the two blocks are in the same loop, then bbI is also a
        // consumer, while bbJ is a producer. This relationship is backward.
        if (ftd::isSameLoopBlocks(bbI, bbJ, li))
          allMemDeps.push_back(ProdConsMemDep(bbI, bbJ));
      }
    }
  }

  // Add a self-dependency for blocks with STORE operation that are in a loop
  // and have no dependency from any producer in the same or an inner loop.
  for (handshake::MemPortOpInterface op : operations) {
    // Only consider STORE operations
    if (!isa<handshake::StoreOp>(op))
      continue;

    Block *bb = op->getBlock();

    // Skip if block is not inside a loop
    const CFGLoop *bbLoop = li.getLoopFor(bb);
    if (!bbLoop)
      continue;

    // Check if a self-dependency already exists
    bool hasSelfDep = llvm::find_if(allMemDeps, [bb](ProdConsMemDep p) {
                        return p.prodBb == bb && p.consBb == bb;
                      }) != allMemDeps.end();
    if (hasSelfDep)
      continue;

    // Check if any existing dependency for this block comes from a producer
    // in the same loop or inner loop
    bool hasPredInSameOrInnerLoop =
        llvm::any_of(allMemDeps, [bb, &li](const ProdConsMemDep &dep) {
          return dep.consBb == bb &&
                 ftd::isSameOrInnerLoopBlocks(dep.prodBb, bb, li);
        });

    // If no such producer exists, add a self-dependency
    if (!hasPredInSameOrInnerLoop) {
      allMemDeps.push_back(ProdConsMemDep(bb, bb));
    }
  }

  return allMemDeps;
}

/// Given a set of operations related to one LSQ and the memory dependency
/// information among them, create a group graph.
static DenseSet<MemoryGroup *>
constructGroupsGraph(SmallVector<handshake::MemPortOpInterface> &lsqOps,
                     SmallVector<ProdConsMemDep> &lsqMemDeps) {

  DenseSet<MemoryGroup *> groups;

  //  Given the operations related to the LSQ, create a group for each of the
  //  correspondent basic block
  for (Operation *op : lsqOps) {
    Block *b = op->getBlock();
    auto it = llvm::find_if(groups, [b](MemoryGroup *g) { return g->bb == b; });
    if (it == groups.end()) {
      MemoryGroup *g = new MemoryGroup(b);
      groups.insert(g);
    }
  }

  // If there exist a relationship O_i -> O_j, with O_i being in basic BB_i
  // (thus group i) and O_j being in BB_j (thus in group j), add G_i to the
  // predecessors of G_j, G_j to the successors of G_i
  for (ProdConsMemDep memDep : lsqMemDeps) {
    // Find the group related to the producer
    MemoryGroup *producerGroup =
        *llvm::find_if(groups, [&memDep](const MemoryGroup *group) {
          return group->bb == memDep.prodBb;
        });

    // Find the group related to the consumer
    MemoryGroup *consumerGroup =
        *llvm::find_if(groups, [&memDep](const MemoryGroup *group) {
          return group->bb == memDep.consBb;
        });

    // create edges to link the groups
    producerGroup->succs.insert(consumerGroup);
    consumerGroup->preds.insert(producerGroup);
  }

  return groups;
}

/// Minimizes the connections between groups based on dominance info. Let's
/// consider the graph
///
/// B -> C -> D
/// |         ^
/// |---------|
///
/// having B, C and D as groups, B being predecessor of both C and D, C of D.
/// Since C has to wait for B to be done, and D has to wait for C to be done,
/// there is no point in D waiting for C to be done. For this reason, the
/// graph can be simplified, saving and edge:
///
/// B -> C -> D
static void minimizeGroupsConnections(handshake::FuncOp funcOp,
                                      DenseSet<MemoryGroup *> &groupsGraph) {

  // Get the dominance info for the region
  DominanceInfo domInfo;
  ftd::BlockIndexing bi(funcOp.getRegion());

  // For each group, compare all the pairs of its predecessors. Cut the edge
  // between them iff the predecessor with the bigger index dominates the
  // whole group
  for (auto &group : groupsGraph) {
    // List of predecessors to remove
    DenseSet<MemoryGroup *> predsToRemove;
    for (auto &bp : group->preds) {

      for (auto &sp : group->preds) {

        // if we are considering the same elements, ignore them
        if (sp->bb == bp->bb || bi.isGreater(sp->bb, bp->bb))
          continue;

        // Add the small predecessors to the list of elements to remove in
        // case the big predecessor has the small one among its
        // predecessors, and the big precessor's BB properly dominates the
        // BB of the group currently under analysis
        if ((bp->preds.find(sp) != bp->preds.end()) &&
            domInfo.properlyDominates(bp->bb, group->bb)) {
          predsToRemove.insert(sp);
        }
      }
    }

    for (auto *pred : predsToRemove) {
      group->preds.erase(pred);
      pred->succs.erase(group);
    }
  }
}

/// For each element in the group, build a lazy fork and use its output to feed
/// the correspondent input of the LSQ.
static handshake::LazyForkOp
ensureLazyForkOutputCount(Block *bb, Value input, unsigned numResults,
                          DenseMap<Block *, handshake::LazyForkOp> &forksGraph,
                          PatternRewriter &rewriter) {
  auto it = forksGraph.find(bb);
  if (it == forksGraph.end()) {
    rewriter.setInsertionPointToStart(bb);
    auto forkOp = rewriter.create<handshake::LazyForkOp>(bb->front().getLoc(),
                                                         input, numResults);
    forksGraph[bb] = forkOp;
    return forkOp;
  }

  handshake::LazyForkOp oldFork = it->second;
  if (oldFork->getNumResults() >= numResults)
    return oldFork;

  rewriter.setInsertionPoint(oldFork);
  auto newFork = rewriter.create<handshake::LazyForkOp>(
      oldFork.getLoc(), oldFork.getOperand(), numResults);
  newFork->setAttrs(oldFork->getAttrs());

  for (unsigned idx = 0, e = oldFork->getNumResults(); idx < e; ++idx)
    oldFork->getResult(idx).replaceAllUsesWith(newFork->getResult(idx));

  rewriter.eraseOp(oldFork);
  forksGraph[bb] = newFork;
  return newFork;
}

static LogicalResult
connectLSQToForkGraph(handshake::FuncOp &funcOp,
                      DenseSet<MemoryGroup *> &groups, handshake::LSQOp lsqOp,
                      DenseMap<Block *, handshake::LazyForkOp> &sharedForks,
                      DenseMap<Block *, unsigned> &nextAllocPort,
                      PatternRewriter &rewriter) {

  DenseMap<Block *, unsigned> allocPort;
  auto startValue = (Value)funcOp.getArguments().back();

  // Create or reuse one fork node per basic block. Result #0 is reserved for
  // the continuation/dependency network; one extra result is allocated for each
  // LSQ group allocation control input in the block.
  for (MemoryGroup *group : groups) {
    Block *bb = group->bb;

    unsigned port = nextAllocPort.lookup(bb);
    if (port == 0)
      port = 1;
    nextAllocPort[bb] = port + 1;
    allocPort[bb] = port;

    auto forkOp = ensureLazyForkOutputCount(bb, startValue, port + 1,
                                            sharedForks, rewriter);
    sharedForks[bb] = forkOp;
  }

  // The allocated output of each lazy fork must be connected to the LSQ, so
  // that it can activate the allocation for the operations of the corresponding
  // basic block.
  //
  // Use the actual group ports, including entry-block groups whose allocation
  // signal is a function argument instead of a control merge.
  auto ports = lsqOp.getPorts();
  for (auto &group : ports.getGroups()) {
    unsigned opIdx = group->ctrlPort->getCtrlInputIndex();
    Value allocation = lsqOp->getOperand(opIdx);
    Operation *producer = allocation.getDefiningOp();
    Block *bb = producer ? producer->getBlock() : &funcOp.front();
    if (!allocPort.contains(bb))
      return lsqOp.emitError("allocation group has no corresponding S2Q fork");
    lsqOp.setOperand(opIdx, sharedForks[bb]->getResult(allocPort[bb]));
  }
  return success();
}

/// Get all the load and store operations related to a LSQ operation
static SmallVector<handshake::MemPortOpInterface>
getLsqOps(handshake::FuncOp &funcOp, handshake::LSQOp lsqOp) {
  SmallVector<handshake::MemPortOpInterface> lsqOps;

  for (auto memOp : funcOp.getOps<handshake::MemPortOpInterface>()) {
    if (llvm::any_of(memOp.getAddressOutput().getUsers(),
                     [&](Operation *user) { return user == lsqOp; }))
      lsqOps.push_back(memOp);
  }

  return lsqOps;
}

/// Given a graph of lazy forks, connect the elements together with some proper
/// SSA phi
static LogicalResult
connectForkGraphs(handshake::FuncOp &funcOp,
                  ArrayRef<DenseSet<MemoryGroup *>> allGroupsGraphs,
                  const DenseMap<Block *, handshake::LazyForkOp> &forksGraph,
                  PatternRewriter &rewriter) {

  DenseMap<OpOperand *, SmallVector<Value>> deps;
  DenseMap<Block *, Value> registeredDependencies;
  auto getAcceptedAllocation = [&](Block *bb) -> Value {
    Value &token = registeredDependencies[bb];
    if (!token) {
      Operation *producerLF = forksGraph.at(bb);
      // Result zero fires only when every allocation output is accepted.
      // Register the event to break ready/valid cycles and ensure subsequent
      // requests, including ctrlEnd, observe the updated LSQ occupancy.
      rewriter.setInsertionPointAfter(producerLF);
      token = rewriter
                  .create<handshake::BufferOp>(
                      producerLF->getLoc(), producerLF->getResult(0), 1,
                      handshake::BufferType::ONE_SLOT_BREAK_DV, 1)
                  .getResult();
    }
    return token;
  };

  for (const DenseSet<MemoryGroup *> &groupsGraph : allGroupsGraphs) {
    for (MemoryGroup *consumerGroup : groupsGraph) {
      handshake::LazyForkOp consumerLF = forksGraph.at(consumerGroup->bb);
      SmallVector<Value> &forkDeps = deps[&consumerLF->getOpOperand(0)];
      for (MemoryGroup *producerGroup : consumerGroup->preds) {
        Value token = getAcceptedAllocation(producerGroup->bb);
        if (!llvm::is_contained(forkDeps, token))
          forkDeps.push_back(token);
      }
    }
  }

  // The original exit control can outrun S2Q allocations. An empty LSQ may
  // otherwise acknowledge ctrlEnd in the very cycle its final group arrives.
  // At the function exit, join the latest accepted allocation from each group;
  // SSA/GSA forwarding supplies start on paths where a group never executes.
  // LSQs connected to the same MC share one completion dependency set.
  DenseMap<Operation *, SmallVector<Value>> completionDeps;
  unsigned graphIdx = 0;
  for (auto lsq : funcOp.getOps<handshake::LSQOp>()) {
    Operation *master = lsq;
    if (auto mc = lsq.getConnectedMC())
      master = mc;
    auto &tokens = completionDeps[master];
    for (MemoryGroup *group : allGroupsGraphs[graphIdx++]) {
      Value token = getAcceptedAllocation(group->bb);
      if (!llvm::is_contained(tokens, token))
        tokens.push_back(token);
    }
  }
  auto end = *funcOp.getOps<handshake::EndOp>().begin();
  for (auto &[master, tokens] : completionDeps) {
    OpOperand &ctrlEnd = master->getOpOperand(master->getNumOperands() - 1);
    assert(ctrlEnd.get() ==
           cast<handshake::MemoryOpInterface>(master).getCtrlEnd());
    SmallVector<Value> inputs{ctrlEnd.get()};
    llvm::append_range(inputs, tokens);
    rewriter.setInsertionPoint(end->getBlock()->getTerminator());
    auto join = rewriter.create<handshake::JoinOp>(end.getLoc(), inputs);
    for (auto [idx, token] : llvm::enumerate(tokens))
      deps[&join->getOpOperand(idx + 1)].push_back(token);
    ctrlEnd.set(join.getResult());
  }

  return ftd::createPhiNetworkDeps(funcOp.getRegion(), rewriter, deps);
}

/// Remove control-network operations that are no longer needed after LSQ
/// allocation has been replaced. Preserve the control paths that tell memory
/// interfaces when the function's accesses have finished. A returned value can
/// be constant or independent of the accesses, so it cannot establish memory
/// completion on its own.
static void removeNetworkCMerges(handshake::FuncOp &funcOp,
                                 PatternRewriter &rewriter) {

  // Get the end operation and its operands.
  handshake::EndOp endOperation = *funcOp.getOps<handshake::EndOp>().begin();
  auto operands = endOperation->getOperands();

  // Check if the function is void. This happens if all the types of the
  // arguments of the `end` operands are identical.
  bool hasReturnValue = llvm::any_of(operands, [&](Value v) -> bool {
    return v.getType() != operands[0].getType();
  });

  // Retain the existing behavior for void functions.
  if (!hasReturnValue)
    return;

  // The FTD circuit introduces no branch operations, so the ones in the circuit
  // are only related to the network of cmerges
  auto branchOps = funcOp.getOps<handshake::BranchOp>();

  if (branchOps.empty())
    return;

  // S2Q replaces allocation dependencies, but memory completion must still
  // wait for the original exit control. Replacing that control with start can
  // finish a memory transaction before its first allocation has even arrived.
  SmallVector<Value> completionWorklist;
  funcOp.walk([&](Operation *op) {
    if (auto memory = dyn_cast<handshake::MemoryOpInterface>(op))
      completionWorklist.push_back(memory.getCtrlEnd());
  });
  for (Value operand : operands)
    if (isa<handshake::ControlType>(operand.getType()))
      completionWorklist.push_back(operand);

  DenseSet<Operation *> completionNetwork;
  while (!completionWorklist.empty()) {
    Operation *op = completionWorklist.pop_back_val().getDefiningOp();
    if (!op ||
        !isa<handshake::ControlMergeOp, handshake::BranchOp,
             handshake::ConditionalBranchOp, handshake::JoinOp>(op) ||
        !completionNetwork.insert(op).second)
      continue;
    for (Value operand : op->getOperands())
      if (isa<handshake::ControlType>(operand.getType()))
        completionWorklist.push_back(operand);
  }

  // Keep traversal bookkeeping separate from the removal set: preserved
  // operations may participate in cycles too.
  DenseSet<Operation *> visited;
  DenseSet<Operation *> operationsToRemove;
  // Operations that needs to be traversed via BFS, handled in a FIFO approach
  SmallVector<Operation *> operationsToTraverse;

  // Start traversing the original control network.
  operationsToTraverse.push_back((Operation *)(*branchOps.begin()));

  auto startValue = (Value)funcOp.getArguments().back();

  // While there are operations to be traversed
  while (!operationsToTraverse.empty()) {
    Operation *toTraverse = operationsToTraverse.pop_back_val();
    if (!visited.insert(toTraverse).second)
      continue;

    // The network is made of control merges, branches and conditional branches:
    // ignore anything else.
    if (!(llvm::isa<handshake::ControlMergeOp>(toTraverse) ||
          llvm::isa<handshake::BranchOp>(toTraverse) ||
          llvm::isa<handshake::ConditionalBranchOp>(toTraverse))) {
      continue;
    }

    // Consider all the users of some results of the operations, and add them to
    // the operations to traverse
    for (auto result : toTraverse->getResults()) {
      for (auto *user : result.getUsers()) {
        if (!visited.contains(user))
          operationsToTraverse.push_back(user);
      }
    }

    if (completionNetwork.contains(toTraverse))
      continue;
    operationsToRemove.insert(toTraverse);

    // Since the operations is to be remove, its results are not needed anymore,
    // and can be substitued with `start`. This is useful to guarantee that any
    // operation is not in use once it gets removed.

    // Substitute all the control merge results with start
    if (auto cmerge = llvm::dyn_cast<handshake::ControlMergeOp>(toTraverse);
        cmerge)
      rewriter.replaceAllUsesWith(cmerge.getResult(), startValue);

    // Substitute all the branch results with start
    if (auto br = llvm::dyn_cast<handshake::BranchOp>(toTraverse); br)
      rewriter.replaceAllUsesWith(br.getResult(), startValue);

    // Substitute all the conditional branch results with start
    if (auto condBr =
            llvm::dyn_cast<handshake::ConditionalBranchOp>(toTraverse);
        condBr) {
      rewriter.replaceAllUsesWith(condBr.getTrueResult(), startValue);
      rewriter.replaceAllUsesWith(condBr.getFalseResult(), startValue);
    }
  }

  for (auto *toRemove : operationsToRemove)
    toRemove->erase();
}

/// Per-block edge information captured from the multi-block CFG before
/// flattening. Used to reconstruct a ShadowCFG afterwards.
struct CapturedEdgeInfo {
  bool isConditional = false;
  bool hasSuccessors = false;
  unsigned trueSuccIdx = 0;
  unsigned falseSuccIdx = 0;
  unsigned uncondSuccIdx = 0;
};

/// Capture the CFG topology and branch condition Values of a handshake::FuncOp
/// while it still has restored multi-block structure. Both pieces of
/// information are destroyed by removeNetworkCMerges + flattenFunction and are
/// needed later to construct the ShadowCFG for addRegen / addSupp.
static void captureCFGTopology(handshake::FuncOp funcOp, unsigned &numBlocks,
                               SmallVector<CapturedEdgeInfo> &edges,
                               DenseMap<unsigned, Value> &capturedConditions) {
  Region &region = funcOp.getRegion();
  DenseMap<Block *, unsigned> blockToIdx;
  for (auto [idx, block] : llvm::enumerate(region))
    blockToIdx[&block] = idx;

  numBlocks = blockToIdx.size();
  edges.resize(numBlocks);

  for (auto [idx, block] : llvm::enumerate(region)) {
    CapturedEdgeInfo &edge = edges[idx];
    Operation *term = block.getTerminator();
    if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
      edge.isConditional = true;
      edge.hasSuccessors = true;
      edge.trueSuccIdx = blockToIdx.lookup(condBr.getTrueDest());
      edge.falseSuccIdx = blockToIdx.lookup(condBr.getFalseDest());
      // Restored CF branches temporarily carry Handshake channel values,
      // so avoid the typed CF accessor that requires an integer SSA value.
      capturedConditions[idx] = condBr->getOperand(0);
    } else if (auto br = dyn_cast<cf::BranchOp>(term)) {
      edge.hasSuccessors = true;
      edge.uncondSuccIdx = blockToIdx.lookup(br.getDest());
    }
  }

  // Conditions come from the restored CFG, whose annotations name the original
  // branch predicates. A handshake conditional branch may instead implement
  // suppression over a different or compound predicate in the same block.
}

/// Build a ShadowCFG for the given handshake::FuncOp using previously captured
/// CFG topology and condition Values. The shadow function replicates the
/// original block structure with standard CF terminators, enabling analysis
/// passes (BlockIndexing, DominanceInfo, CFGLoopInfo) that addRegen and addSupp
/// rely on.
static ftd::ShadowCFG buildShadowFromCapturedTopology(
    OpBuilder &builder, handshake::FuncOp funcOp, unsigned numBlocks,
    const SmallVector<CapturedEdgeInfo> &edges,
    const DenseMap<unsigned, Value> &capturedConditions) {

  ftd::ShadowCFG shadow;
  Location loc = funcOp.getLoc();

  // Create a temporary func::FuncOp with blocks + CF terminators that mirror
  // the original multi-block topology.
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(funcOp);
    auto funcType = builder.getFunctionType({}, {});
    shadow.shadowFunc =
        builder.create<func::FuncOp>(loc, "__ftd_shadow_cfg__", funcType);

    Region &R = shadow.shadowFunc.getBody();
    SmallVector<Block *> blocks;
    for (unsigned i = 0; i < numBlocks; ++i)
      blocks.push_back(builder.createBlock(&R, R.end()));

    for (unsigned i = 0; i < numBlocks; ++i) {
      const CapturedEdgeInfo &edge = edges[i];
      builder.setInsertionPointToEnd(blocks[i]);

      if (edge.isConditional) {
        auto dummyCond =
            builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true));
        builder.create<cf::CondBranchOp>(
            loc, dummyCond, blocks[edge.trueSuccIdx], ValueRange{},
            blocks[edge.falseSuccIdx], ValueRange{});
      } else if (edge.hasSuccessors) {
        builder.create<cf::BranchOp>(loc, blocks[edge.uncondSuccIdx]);
      } else {
        builder.create<func::ReturnOp>(loc);
      }
    }
  }

  // Use original branch conditions captured before flattening the restored CFG.
  shadow.conditionMap = capturedConditions;

  return shadow;
}

/// Remove the cmerge network, flatten the function, and run FTD gating on the
/// flattened IR using a temporary ShadowCFG reconstructed from the original
/// multi-block topology.
static LogicalResult runPostCmergeFtd(handshake::FuncOp funcOp,
                                      MLIRContext *ctx,
                                      bool resolveCondPlaceholders) {
  ConversionPatternRewriter rewriter(ctx);

  experimental::cfg::markBasicBlocks(funcOp, rewriter);

  // Capture the original CFG before the cmerge network is removed so we can
  // rebuild a ShadowCFG after flattening for addRegen/addSupp.
  unsigned capturedNumBlocks = 0;
  SmallVector<CapturedEdgeInfo> capturedEdges;
  DenseMap<unsigned, Value> capturedConditions;
  captureCFGTopology(funcOp, capturedNumBlocks, capturedEdges,
                     capturedConditions);

  removeNetworkCMerges(funcOp, rewriter);

  if (failed(cfg::flattenFunction(funcOp)))
    return failure();

  // A logical return can precede loop latches in reverse postorder. Keep its
  // logical BB ID, but place EndOp last in the flattened function.
  auto ends = funcOp.getOps<handshake::EndOp>();
  if (!ends.empty())
    (*ends.begin())->moveBefore(&funcOp.front(), funcOp.front().end());

  if (capturedNumBlocks <= 1)
    return success();

  OpBuilder builder(ctx);
  ftd::ShadowCFG shadow = buildShadowFromCapturedTopology(
      builder, funcOp, capturedNumBlocks, capturedEdges, capturedConditions);

  if (resolveCondPlaceholders) {
    ftd::resolveCondPlaceholders(funcOp, builder, shadow);

    // Pick up negated condition signals produced by resolveCondPlaceholders.
    for (auto notOp : funcOp.getOps<handshake::NotIOp>()) {
      if (!notOp->hasAttr("ftd.cvar"))
        continue;
      auto bbAttr = notOp->getAttrOfType<IntegerAttr>("handshake.bb");
      if (!bbAttr)
        continue;
      shadow.conditionMap[bbAttr.getUInt()] = notOp.getResult();
    }
  }

  ftd::addRegen(funcOp, builder, shadow);
  ftd::addSupp(funcOp, builder, shadow);

  if (resolveCondPlaceholders)
    ftd::finalizeCondPlaceholders(funcOp);

  shadow.destroy();
  return success();
}

/// The LSQ allocator accepts one group at a time. Arbitrate the activation of
/// allocation forks, rather than their individual outputs: a fork shared by
/// multiple LSQs must obtain a consistent grant for all of them. Forks sharing
/// an LSQ form a conflict component; independent components remain parallel.
static LogicalResult arbitrateLSQAllocations(handshake::FuncOp funcOp,
                                             PatternRewriter &rewriter) {
  SmallVector<handshake::LazyForkOp> forks;
  DenseSet<Operation *> seen;
  SmallVector<SmallVector<Operation *>> conflicts;
  for (auto lsq : funcOp.getOps<handshake::LSQOp>()) {
    auto ports = lsq.getPorts();
    SmallVector<Operation *> members;
    for (auto &group : ports.getGroups()) {
      Value allocation = lsq->getOperand(group->ctrlPort->getCtrlInputIndex());
      auto fork = dyn_cast_if_present<handshake::LazyForkOp>(
          allocation.getDefiningOp());
      if (!fork)
        return lsq.emitError(
            "S2Q group allocation must be driven by a lazy fork");
      members.push_back(fork);
      if (seen.insert(fork).second)
        forks.push_back(fork);
    }
    conflicts.push_back(std::move(members));
  }
  llvm::sort(forks, [](handshake::LazyForkOp lhs, handshake::LazyForkOp rhs) {
    return lhs->getAttrOfType<IntegerAttr>("handshake.bb").getUInt() <
           rhs->getAttrOfType<IntegerAttr>("handshake.bb").getUInt();
  });
  DenseMap<Operation *, unsigned> indices;
  SmallVector<unsigned> parents;
  for (auto [idx, fork] : llvm::enumerate(forks)) {
    indices[fork] = idx;
    parents.push_back(idx);
  }
  auto findRoot = [&](unsigned idx) {
    while (parents[idx] != idx) {
      parents[idx] = parents[parents[idx]];
      idx = parents[idx];
    }
    return idx;
  };
  for (auto &members : conflicts) {
    if (members.empty())
      continue;
    unsigned root = findRoot(indices.at(members.front()));
    for (Operation *member : llvm::drop_begin(members)) {
      unsigned other = findRoot(indices.at(member));
      unsigned first = std::min(root, other);
      parents[std::max(root, other)] = first;
      root = first;
    }
  }
  std::map<unsigned, SmallVector<handshake::LazyForkOp>> components;
  for (auto [idx, fork] : llvm::enumerate(forks))
    components[findRoot(idx)].push_back(fork);
  for (auto &[_, members] : components) {
    if (members.size() < 2)
      continue;
    SmallVector<Value> requests;
    for (auto fork : members)
      requests.push_back(fork.getOperand());
    rewriter.setInsertionPoint(members.front());
    auto mark = [&](Operation *op) {
      op->setAttr(
          "handshake.bb",
          rewriter.getIntegerAttr(IntegerType::get(rewriter.getContext(), 32,
                                                   IntegerType::Unsigned),
                                  0));
    };
    auto arbiter = rewriter.create<handshake::ControlMergeOp>(
        funcOp.getLoc(), ValueRange(requests));
    mark(arbiter);
    auto source = rewriter.create<handshake::SourceOp>(funcOp.getLoc());
    mark(source);
    auto indexType = cast<handshake::ChannelType>(arbiter.getIndex().getType())
                         .getDataType();
    for (auto [memberIdx, fork] : llvm::enumerate(members)) {
      auto constant = rewriter.create<handshake::ConstantOp>(
          fork.getLoc(), rewriter.getIntegerAttr(indexType, memberIdx),
          source.getResult());
      mark(constant);
      auto selected = rewriter.create<handshake::CmpIOp>(
          fork.getLoc(), handshake::CmpIPredicate::eq, arbiter.getIndex(),
          constant.getResult());
      mark(selected);
      auto branch = rewriter.create<handshake::ConditionalBranchOp>(
          fork.getLoc(), selected.getResult(), arbiter.getResult());
      mark(branch);
      auto sink = rewriter.create<handshake::SinkOp>(fork.getLoc(),
                                                     branch.getFalseResult());
      mark(sink);
      fork->setOperand(0, branch.getTrueResult());
    }
  }
  return success();
}

/// Restore the CFG used by S2Q from its explicit node IDs. In particular, an
/// exit is an edge destination, not necessarily one past the largest source.
/// Keep original condition signals; suppression branches are not substitutes.
static LogicalResult restoreS2QCFG(handshake::FuncOp funcOp,
                                   PatternRewriter &rewriter) {
  auto annotation = funcOp->getAttrOfType<StringAttr>(cfg::CFG_EDGES);
  if (!annotation)
    return funcOp.emitError("S2Q requires the original CFG annotation");
  if (!funcOp.getBody().hasOneBlock())
    return funcOp.emitError("S2Q expects a flattened input function");

  std::map<unsigned, cfg::CFGNode> edges;
  DenseMap<unsigned, Value> conditions;
  unsigned maxID = 0;
  StringRef remaining = annotation.getValue();
  while (!remaining.empty()) {
    if (!remaining.consume_front("["))
      return funcOp.emitError("invalid S2Q CFG annotation");
    size_t close = remaining.find(']');
    if (close == StringRef::npos)
      return funcOp.emitError("unterminated S2Q CFG edge");
    SmallVector<StringRef> fields;
    remaining.take_front(close).split(fields, ',');
    remaining = remaining.drop_front(close + 1);
    if (fields.size() != 2 && fields.size() != 4)
      return funcOp.emitError("invalid S2Q CFG edge");
    unsigned source, first, second = 0;
    if (fields[0].getAsInteger(10, source) ||
        fields[1].getAsInteger(10, first) ||
        (fields.size() == 4 && fields[2].getAsInteger(10, second)))
      return funcOp.emitError("invalid S2Q CFG block ID");
    maxID = std::max({maxID, source, first, second});
    if (fields.size() == 4) {
      Value condition;
      for (Operation &op : funcOp.getOps()) {
        auto name = op.getAttrOfType<StringAttr>("handshake.name");
        if (name && name.getValue() == fields[3] && op.getNumResults()) {
          condition = op.getResult(0);
          break;
        }
      }
      if (!condition)
        return funcOp.emitError("S2Q cannot find original branch condition ")
               << fields[3];
      conditions[source] = condition;
    }
    auto edge = fields.size() == 4
                    ? cfg::CFGNode(first, second, fields[3].str())
                    : cfg::CFGNode(first);
    if (!edges.emplace(source, edge).second)
      return funcOp.emitError("duplicate S2Q CFG source block");
  }

  SmallVector<Operation *> operations;
  for (Operation &op : funcOp.getOps()) {
    operations.push_back(&op);
    maxID = std::max(maxID, getLogicBB(&op).value_or(0));
  }
  SmallVector<Block *> blocks{&funcOp.front()};
  while (blocks.size() <= maxID)
    blocks.push_back(funcOp.addBlock());
  for (Operation *op : operations) {
    unsigned block = getLogicBB(op).value_or(0);
    op->moveBefore(blocks[block], blocks[block]->end());
  }
  for (auto [id, block] : llvm::enumerate(blocks)) {
    rewriter.setInsertionPointToEnd(block);
    auto found = edges.find(id);
    if (found == edges.end()) {
      rewriter.create<func::ReturnOp>(funcOp.getLoc());
    } else if (found->second.isConditional()) {
      const auto &edge = found->second;
      rewriter.create<cf::CondBranchOp>(funcOp.getLoc(), conditions.lookup(id),
                                        blocks[edge.getTrueSuccessor()],
                                        blocks[edge.getFalseSuccessor()]);
    } else {
      rewriter.create<cf::BranchOp>(funcOp.getLoc(),
                                    blocks[found->second.getSuccessor()]);
    }
  }
  return success();
}

/// Run straight to the queue.
static LogicalResult applyStraightToQueue(handshake::FuncOp funcOp,
                                          MLIRContext *ctx) {

  ConversionPatternRewriter rewriter(ctx);

  // Return if there are no LSQs in the function
  if (funcOp.getOps<handshake::LSQOp>().empty())
    return success();

  // Restore the cf structure to work on a structured IR
  if (failed(restoreS2QCFG(funcOp, rewriter)))
    return failure();

  SmallVector<DenseSet<MemoryGroup *>> allGroupsGraphs;
  DenseMap<Block *, handshake::LazyForkOp> sharedForks;
  DenseMap<Block *, unsigned> nextAllocPort;

  // For each LSQ
  for (const handshake::LSQOp lsqOp : funcOp.getOps<handshake::LSQOp>()) {

    // Collect all the operations related to that LSQ
    auto lsqOps = getLsqOps(funcOp, lsqOp);

    // Get all the memory depdencies among the operations connected to the
    // same LSQ
    auto lsqMemDeps = identifyMemoryDependencies(funcOp, lsqOps);
    LLVM_DEBUG(for (auto &dep : lsqMemDeps) dep.print());

    // Build a group graph out of the dependencies
    auto groupsGraph = constructGroupsGraph(lsqOps, lsqMemDeps);

    // Apply group minimization techniques
    minimizeGroupsConnections(funcOp, groupsGraph);

    LLVM_DEBUG(for (auto &g : groupsGraph) g->print());

    // Build a lazy fork for each group and connect it to the related
    // activation input in the LSQ
    if (failed(connectLSQToForkGraph(funcOp, groupsGraph, lsqOp, sharedForks,
                                     nextAllocPort, rewriter)))
      return failure();

    allGroupsGraphs.push_back(std::move(groupsGraph));
  }

  // Connect the shared lazy forks together through a network of merges.
  if (failed(connectForkGraphs(funcOp, allGroupsGraphs, sharedForks, rewriter)))
    return failure();

  // Delete the groups
  for (const DenseSet<MemoryGroup *> &groupsGraph : allGroupsGraphs)
    for (auto *g : groupsGraph)
      delete g;

  // Create condition placeholders for every conditional block in the restored
  // CFG. The refactored addGsaGates (called by replaceMergeToGSA) relies on
  // these placeholders when building distribution networks and BDD circuits
  // for MU gates.  In the FTD flow, createAllCondPlaceholders is called before
  // addGsaGates; S2Q must do the same.
  ftd::createAllCondPlaceholders(funcOp.getRegion(), rewriter);

  // Replace each merge created by `createPhiNetwork` with a multiplxer
  if (failed(ftd::replaceMergeToGSA(funcOp, rewriter)))
    return failure();

  if (failed(runPostCmergeFtd(funcOp, ctx, /*resolveCondPlaceholders=*/true)))
    return failure();
  return arbitrateLSQAllocations(funcOp, rewriter);
}

struct HandshakeStraightToQueuePass
    : public dynamatic::experimental::impl::HandshakeStraightToQueueBase<
          HandshakeStraightToQueuePass> {

  void runOnOperation() override {
    // Make sure all operation names are unique and haven't changed from what is
    // cached (if anything is cached)
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    if (!namer.isAnalysisValid())
      return signalPassFailure();

    MLIRContext *ctx = &getContext();
    mlir::ModuleOp module = getOperation();

    for (auto funcOp : module.getOps<handshake::FuncOp>())
      if (failed(applyStraightToQueue(funcOp, ctx)))
        signalPassFailure();

    // Make sure all operation names are still unique and name operations that
    // do not currently have a name
    if (failed(namer.walk(NameAnalysis::UnnamedBehavior::NAME)))
      return signalPassFailure();
    markAnalysesPreserved<NameAnalysis>();
  };
};
} // namespace
