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

#include "experimental/Transforms/HandshakeStraightToQueue.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "experimental/Analysis/GSAAnalysis.h"
#include "experimental/Support/CFGAnnotation.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

namespace {

struct ProdConsMemDep {
  Block *prodBb;
  Block *consBb;
  bool isBackward;

  ProdConsMemDep(Block *prod, Block *cons, bool backward)
      : prodBb(prod), consBb(cons), isBackward(backward) {}

  /// Print the dependency stored in the current relationship
  void printDependency() {
    llvm::dbgs() << "[PROD_CONS_MEM_DEP] Dependency from [";
    prodBb->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "] to [";
    consBb->printAsOperand(llvm::dbgs());
    llvm::dbgs() << "]";
    if (isBackward)
      llvm::dbgs() << " (backward)";
    llvm::dbgs() << "\n";
  }
};

/// A group represents all the memory operations belonging to the same basic
/// block which require the same LSQ. It contains a reference to the BB, a set
/// of predecessor in the dependence graph and a set of successors.
struct Group {
  // The BB the group defines
  Block *bb;

  // List of predecessors of the group
  DenseSet<Group *> preds;
  // List of successors of the group
  DenseSet<Group *> succs;

  // Constructor for the group
  Group(Block *b) : bb(b) {}

  // Relationship operator between groups
  bool operator<(const Group &other) const { return bb < other.bb; }

  /// Print the dependenices of the curent group
  void printDependenices() {
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
};

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
      if (bi.lessIndex(bbJ, bbI)) {

        // and add it to the list of dependencies
        ProdConsMemDep oneMemDep(bbJ, bbI, false);
        allMemDeps.push_back(oneMemDep);

        // If the two blocks are in the same loop, then bbI is also a
        // consumer, while bbJ is a producer. This relationship is backward.
        if (ftd::isSameLoopBlocks(bbI, bbJ, li)) {
          ProdConsMemDep opp(bbI, bbJ, true);
          allMemDeps.push_back(opp);
        }
      }
    }
  }

  return allMemDeps;
}

/// Given a set of operations related to one LSQ and the memory dependency
/// information among them, create a group graph.
static DenseSet<Group *>
constructGroupsGraph(SmallVector<handshake::MemPortOpInterface> &lsqOps,
                     SmallVector<ProdConsMemDep> &lsqMemDeps) {

  DenseSet<Group *> groups;

  //  Given the operations related to the LSQ, create a group for each of the
  //  correspondent basic block
  for (Operation *op : lsqOps) {
    Block *b = op->getBlock();
    auto it = llvm::find_if(groups, [b](Group *g) { return g->bb == b; });
    if (it == groups.end()) {
      Group *g = new Group(b);
      groups.insert(g);
    }
  }

  // If there exist a relationship O_i -> O_j, with O_i being in basic BB_i
  // (thus group i) and O_j being in BB_j (thus in group j), add G_i to the
  // predecessors of G_j, G_j to the successors of G_i
  for (ProdConsMemDep memDep : lsqMemDeps) {
    // Find the group related to the producer
    Group *producerGroup =
        *llvm::find_if(groups, [&memDep](const Group *group) {
          return group->bb == memDep.prodBb;
        });

    // Find the group related to the consumer
    Group *consumerGroup =
        *llvm::find_if(groups, [&memDep](const Group *group) {
          return group->bb == memDep.consBb;
        });

    // create edges to link the groups
    producerGroup->succs.insert(consumerGroup);
    consumerGroup->preds.insert(producerGroup);
  }

  // Add a self dependency each time you have a group with no dependency
  for (Group *g : groups) {
    if (!g->preds.size()) {
      g->preds.insert(g);
      g->succs.insert(g);
    }
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
                                      DenseSet<Group *> &groupsGraph) {

  // Get the dominance info for the region
  DominanceInfo domInfo;
  ftd::BlockIndexing bi(funcOp.getRegion());

  // For each group, compare all the pairs of its predecessors. Cut the edge
  // between them iff the predecessor with the bigger index dominates the
  // whole group
  for (auto &group : groupsGraph) {
    // List of predecessors to remove
    DenseSet<Group *> predsToRemove;
    for (auto &bp : group->preds) {

      for (auto &sp : group->preds) {

        // if we are considering the same elements, ignore them
        if (sp->bb == bp->bb || bi.greaterIndex(sp->bb, bp->bb))
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

static DenseMap<Block *, handshake::LazyForkOp>
connectLSQToForkGraph(handshake::FuncOp &funcOp, DenseSet<Group *> &groups,
                      handshake::LSQOp lsqOp,
                      ConversionPatternRewriter &rewriter) {

  DenseMap<Block *, handshake::LazyForkOp> forksGraph;
  auto startValue = (Value)funcOp.getArguments().back();

  // Create the fork nodes: for each group among the set of groups
  for (Group *group : groups) {
    Block *bb = group->bb;
    rewriter.setInsertionPointToStart(bb);

    // Add a lazy fork with two outputs, having the start control value as
    // input and two output ports, one for the LSQ and one for the subsequent
    // buffer
    auto forkOp = rewriter.create<handshake::LazyForkOp>(bb->front().getLoc(),
                                                         startValue, 2);

    // Add the new component to the list of components create for FTD and to
    // the fork graph
    forksGraph[bb] = forkOp;
  }

  // The second output of each lazy fork must be connected to the LSQ, so that
  // they can activate the allocation for the operations of the corresponding
  // basic block
  for (auto [opIdx, op] : llvm::enumerate(lsqOp.getOperands())) {
    if (!llvm::isa_and_nonnull<handshake::ControlMergeOp>(op.getDefiningOp()))
      continue;
    auto cmerge = llvm::dyn_cast<handshake::ControlMergeOp>(op.getDefiningOp());
    Block *bb = cmerge->getBlock();
    if (!forksGraph.contains(bb))
      continue;
    lsqOp.setOperand(opIdx, forksGraph[bb]->getResult(1));
  }

  return forksGraph;
}

static LogicalResult replaceMergeToGSA(handshake::FuncOp funcOp,
                                       ConversionPatternRewriter &rewriter) {
  auto startValue = (Value)funcOp.getArguments().back();

  // Create a backedge for the start value, to be sued during the merges to
  // muxes conversion
  BackedgeBuilder edgeBuilderStart(rewriter, funcOp.getRegion().getLoc());
  Backedge startValueBackedge =
      edgeBuilderStart.get(rewriter.getType<handshake::ControlType>());

  // For each merge that was signed with the `NEW_PHI` attribute, substitute
  // it with its GSA equivalent
  for (handshake::MergeOp merge : funcOp.getOps<handshake::MergeOp>()) {
    if (!merge->hasAttr(ftd::NEW_PHI))
      continue;
    gsa::GSAAnalysis gsa(merge, funcOp.getRegion());
    if (failed(gsa::GSAAnalysis::addGsaGates(funcOp.getRegion(), rewriter, gsa,
                                             startValueBackedge, false)))
      return failure();

    // Get rid of the merge
    rewriter.eraseOp(merge);
  }

  // Replace the backedge
  startValueBackedge.setValue(startValue);

  return success();
}

/// Allocate some joins in front of each lazy fork, so that the number of
/// inputs for each of them is exactly one. The current inputs of the lazy
/// forks become inputs for the joins.
static void
joinInsertion(ConversionPatternRewriter &rewriter, DenseSet<Group *> &groups,
              DenseMap<Block *, handshake::LazyForkOp> &forksGraph) {

  // For each group
  for (Group *group : groups) {
    // Get the corresponding fork and operands
    Operation *forkNode = forksGraph[group->bb];
    ValueRange operands = forkNode->getOperands();
    // If the number of inputs is higher than one
    if (operands.size() > 1) {

      // Join all the inputs, and set the output of this new element as input
      // of the lazy fork
      rewriter.setInsertionPointToStart(forkNode->getBlock());
      auto joinOp =
          rewriter.create<handshake::JoinOp>(forkNode->getLoc(), operands);
      /// The result of the JoinOp becomes the input to the LazyFork
      forkNode->setOperands(joinOp.getResult());
    }
  }
}

LogicalResult applyStraightToQueue(handshake::FuncOp funcOp, MLIRContext *ctx) {

  ConversionPatternRewriter rewriter(ctx);

  llvm::dbgs() << "[INFO] Runnin S2Q\n";

  if (funcOp.getOps<handshake::LSQOp>().empty())
    return success();

  auto startValue = (Value)funcOp.getArguments().back();

  if (failed(cfg::restoreCfStructure(funcOp, rewriter)))
    return failure();

  for (handshake::LSQOp lsqOp : funcOp.getOps<handshake::LSQOp>()) {
    SmallVector<handshake::MemPortOpInterface> lsqOps;

    for (auto memOp : funcOp.getOps<handshake::MemPortOpInterface>()) {
      if (llvm::any_of(memOp.getAddressOutput().getUsers(),
                       [&](Operation *user) { return user == lsqOp; }))
        lsqOps.push_back(memOp);
    }

    auto lsqMemDeps = identifyMemoryDependencies(funcOp, lsqOps);
    for (auto &dep : lsqMemDeps)
      dep.printDependency();

    auto groupsGraph = constructGroupsGraph(lsqOps, lsqMemDeps);
    minimizeGroupsConnections(funcOp, groupsGraph);

    for (auto &g : groupsGraph)
      g->printDependenices();

    auto forksGraph =
        connectLSQToForkGraph(funcOp, groupsGraph, lsqOp, rewriter);

    for (Group *consumerGroup : groupsGraph) {
      SmallVector<Value> differentInputs;
      Operation *consumerLF = forksGraph[consumerGroup->bb];
      for (Group *producerGroup : consumerGroup->preds) {
        Operation *producerLF = forksGraph[producerGroup->bb];

        SmallVector<Value> forkValuesToConnect = {startValue,
                                                  producerLF->getResult(0)};

        auto phiNetworkOrFailure = ftd::createPhiNetwork(
            funcOp.getRegion(), rewriter, forkValuesToConnect);
        if (failed(phiNetworkOrFailure))
          return failure();

        auto &phiNetwork = *phiNetworkOrFailure;
        differentInputs.push_back(phiNetwork[consumerGroup->bb]);
      }

      if (differentInputs.size() == 0)
        differentInputs.push_back(startValue);

      consumerLF->setOperands(differentInputs);
    }

    joinInsertion(rewriter, groupsGraph, forksGraph);
  }

  if (failed(replaceMergeToGSA(funcOp, rewriter)))
    return failure();

  experimental::ftd::addRegen(funcOp, rewriter);
  experimental::ftd::addSupp(funcOp, rewriter);
  experimental::cfg::markBasicBlocks(funcOp, rewriter);

  if (failed(cfg::flattenFunction(funcOp)))
    return failure();

  return success();
}

struct HandshakeStraightToQueuePass
    : public dynamatic::experimental::ftd::impl::HandshakeStraightToQueueBase<
          HandshakeStraightToQueuePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    mlir::ModuleOp module = getOperation();

    for (auto funcOp : module.getOps<handshake::FuncOp>())
      if (failed(applyStraightToQueue(funcOp, ctx)))
        signalPassFailure();
  };
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::ftd::createStraightToQueue() {
  return std::make_unique<HandshakeStraightToQueuePass>();
}
