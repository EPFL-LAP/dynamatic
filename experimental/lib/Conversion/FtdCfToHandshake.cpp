//===- FtdCfToHandshake.cpp - FTD conversion cf -> handshake -----*- C++
//-*-===//
//
// Implements the fast token delivery methodology as depicted in FPGA'22,
// together with the fast LSQ allocation as depicted in FPGA'23.
//
//===----------------------------------------------------------------------===//

#include "experimental/Conversion/FtdCfToHandshake.h"
#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/CFG.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <unordered_set>
#include <utility>

using namespace mlir;
using namespace dynamatic;

namespace {

struct FtdCfToHandshakePass
    : public dynamatic::experimental::ftd::impl::FtdCfToHandshakeBase<
          FtdCfToHandshakePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp modOp = getOperation();

    // Put all non-external functions into maximal SSA form
    for (auto funcOp : modOp.getOps<func::FuncOp>()) {
      if (!funcOp.isExternal()) {
        FuncSSAStrategy strategy;
        if (failed(dynamatic::maximizeSSA(funcOp.getBody(), strategy)))
          return signalPassFailure();
      }
    }

    CfToHandshakeTypeConverter converter;
    RewritePatternSet patterns(ctx);

    patterns.add<experimental::ftd::FtdLowerFuncToHandshake>(
        getAnalysis<NameAnalysis>(), getAnalysis<ControlDependenceAnalysis>(),
        converter, ctx);

    patterns.add<ConvertConstants, ConvertCalls, ConvertUndefinedValues,
                 ConvertIndexCast<arith::IndexCastOp, handshake::ExtSIOp>,
                 ConvertIndexCast<arith::IndexCastUIOp, handshake::ExtUIOp>,
                 OneToOneConversion<arith::AddFOp, handshake::AddFOp>,
                 OneToOneConversion<arith::AddIOp, handshake::AddIOp>,
                 OneToOneConversion<arith::AndIOp, handshake::AndIOp>,
                 OneToOneConversion<arith::CmpFOp, handshake::CmpFOp>,
                 OneToOneConversion<arith::CmpIOp, handshake::CmpIOp>,
                 OneToOneConversion<arith::DivFOp, handshake::DivFOp>,
                 OneToOneConversion<arith::DivSIOp, handshake::DivSIOp>,
                 OneToOneConversion<arith::DivUIOp, handshake::DivUIOp>,
                 OneToOneConversion<arith::ExtSIOp, handshake::ExtSIOp>,
                 OneToOneConversion<arith::ExtUIOp, handshake::ExtUIOp>,
                 OneToOneConversion<arith::MaximumFOp, handshake::MaximumFOp>,
                 OneToOneConversion<arith::MinimumFOp, handshake::MinimumFOp>,
                 OneToOneConversion<arith::MulFOp, handshake::MulFOp>,
                 OneToOneConversion<arith::MulIOp, handshake::MulIOp>,
                 OneToOneConversion<arith::NegFOp, handshake::NegFOp>,
                 OneToOneConversion<arith::OrIOp, handshake::OrIOp>,
                 OneToOneConversion<arith::SelectOp, handshake::SelectOp>,
                 OneToOneConversion<arith::ShLIOp, handshake::ShLIOp>,
                 OneToOneConversion<arith::ShRSIOp, handshake::ShRSIOp>,
                 OneToOneConversion<arith::ShRUIOp, handshake::ShRUIOp>,
                 OneToOneConversion<arith::SubFOp, handshake::SubFOp>,
                 OneToOneConversion<arith::SubIOp, handshake::SubIOp>,
                 OneToOneConversion<arith::TruncIOp, handshake::TruncIOp>,
                 OneToOneConversion<arith::XOrIOp, handshake::XOrIOp>>(
        getAnalysis<NameAnalysis>(), converter, ctx);

    // All func-level functions must become handshake-level functions
    ConversionTarget target(*ctx);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<handshake::HandshakeDialect>();
    target.addIllegalDialect<func::FuncDialect, cf::ControlFlowDialect,
                             arith::ArithDialect, math::MathDialect,
                             BuiltinDialect>();

    if (failed(applyFullConversion(modOp, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

namespace dynamatic {
namespace experimental {
namespace ftd {

using namespace boolean;

// --- Helper functions ---

/// Returns true if the provided operation is either of they `LSQLoad` or
/// `LSQStore`
static bool isHandhsakeLSQOperation(Operation *op) {
  return isa<handshake::LSQStoreOp, handshake::LSQLoadOp>(op);
}

/// Returns true if two operations belong to the same block
static bool isSameBlock(Operation *op1, Operation *op2) {
  return (op1->getBlock() == op2->getBlock());
}

/// Returns true if two operations are both load
static bool areBothLoad(Operation *op1, Operation *op2) {
  return (isa<handshake::LSQLoadOp>(op1) && isa<handshake::LSQLoadOp>(op2));
}

/// Recursive function which allows to obtain all the paths from block `start`
/// to block `end` using a DFS
static void dfsAllPaths(Block *start, Block *end, std::vector<Block *> &path,
                        std::unordered_set<Block *> &visited,
                        std::vector<std::vector<Block *>> &allPaths) {

  // The current block is part of the current path
  path.push_back(start);
  // The current block has been visited
  visited.insert(start);

  // If we are at the end of the path, then add it to the list of paths
  if (start == end) {
    allPaths.push_back(path);
  } else {
    // Else, for each successor which was not visited, run DFS again
    for (Block *successor : start->getSuccessors()) {
      if (visited.find(successor) == visited.end())
        dfsAllPaths(successor, end, path, visited, allPaths);
    }
  }

  // Remove the current block from the current path and from the list of visited
  // blocks
  path.pop_back();
  visited.erase(start);
}

/// Gets all the paths from block `start` to block `end` using a dfs search
static std::vector<std::vector<Block *>> findAllPaths(Block *start,
                                                      Block *end) {
  std::vector<std::vector<Block *>> allPaths;
  std::vector<Block *> path;
  std::unordered_set<Block *> visited;
  dfsAllPaths(start, end, path, visited, allPaths);
  return allPaths;
}

/// Returns true if two operations are both load
static bool isThereAPath(Operation *op1, Operation *op2) {
  return !findAllPaths(op1->getBlock(), op2->getBlock()).empty();
}

/// Check whether the index of `block1` is less than the one of `block2`
static bool lessThanBlocks(Block *block1, Block *block2) {
  // Get the name of block 1
  std::string result1;
  llvm::raw_string_ostream os1(result1);
  block1->printAsOperand(os1);
  std::string block1id = os1.str();

  // Obtain the index of the block as integer
  int id1 = std::stoi(block1id.substr(3));

  // Get the name of block 2
  std::string result2;
  llvm::raw_string_ostream os2(result2);
  block2->printAsOperand(os2);
  std::string block2id = os2.str();

  // Obtain the index of the block as integer
  int id2 = std::stoi(block2id.substr(3));

  // Compare the two integers
  return id1 < id2;
}

/// Recursively check weather 2 blocks belong to the same loop, starting from
/// the inner-most loops
static bool isSameLoop(const CFGLoop *loop1, const CFGLoop *loop2) {
  if (!loop1 || !loop2)
    return false;
  return (loop1 == loop2 || isSameLoop(loop1->getParentLoop(), loop2) ||
          isSameLoop(loop1, loop2->getParentLoop()) ||
          isSameLoop(loop1->getParentLoop(), loop2->getParentLoop()));
}

/// checks if the source and destination are in a loop
static bool isSameLoopBlocks(Block *source, Block *dest,
                             const mlir::CFGLoopInfo &li) {
  return isSameLoop(li.getLoopFor(source), li.getLoopFor(dest));
}

/// Given an LSQ, extract the list of operations which require that same LSQ
static SmallVector<Operation *> getLSQOperations(
    const llvm::MapVector<unsigned, SmallVector<Operation *>> &lsqPorts) {

  // Result vector holding the result
  SmallVector<Operation *> combinedOperations;

  // Iterate over the MapVector and add all Operation* to the
  // combinedOperations vector
  for (const auto &entry : lsqPorts) {
    const SmallVector<Operation *> &operations = entry.second;
    combinedOperations.insert(combinedOperations.end(), operations.begin(),
                              operations.end());
  }
  return combinedOperations;
}

/// Given a set of operations related to one LSQ and the memory dependency
/// information among them, crate a group graph.
static void constructGroupsGraph(SmallVector<Operation *> &operations,
                                 SmallVector<ProdConsMemDep> &allMemDeps,
                                 DenseSet<Group *> &groups) {

  //  Given the operations related to the LSQ, create a group for each of the
  //  correspondent basic block
  for (Operation *op : operations) {
    if (isHandhsakeLSQOperation(op)) {
      Block *b = op->getBlock();
      auto it = llvm::find_if(groups, [b](Group *g) { return g->bb == b; });
      if (it == groups.end()) {
        Group *g = new Group(b);
        groups.insert(g);
      }
    }
  }

  // If there exist a relationship O_i -> O_j, with O_i being in basic BB_i
  // (thus group i) and O_j being in BB_j (thus in group j), add G_i to the
  // predecessors of G_j, G_j to the successors of G_i
  for (ProdConsMemDep memDep : allMemDeps) {
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
/// there is no point in D waiting for C to be done. For this reason, the graph
/// can be simplified, saving and edge:
///
/// B -> C -> D
static void minimizeGroupsConnections(DenseSet<Group *> &groupsGraph) {

  // Get the dominance info for the region
  DominanceInfo domInfo;

  // For each group, compare all the pairs of its predecessors. Cut the edge
  // between them iff the predecessor with the bigger index dominates the whole
  // group
  for (auto &group : groupsGraph) {
    // List of predecessors to remove
    DenseSet<Group *> predsToRemove;
    for (auto &bp : group->preds) {
      // If the big predecessor is alreay in the list to remove, ignore it
      if (llvm::find(predsToRemove, bp) != predsToRemove.end())
        continue;
      for (auto &sp : group->preds) {
        // If the small predecessor has bigger index than the big predecessor,
        // ignore it
        if (lessThanBlocks(bp->bb, sp->bb))
          continue;
        // If the small predecessor is alreay in the list to remove, ignore it
        if (llvm::find(predsToRemove, sp) != predsToRemove.end())
          continue;
        // if we are considering the same elements, ignore them
        if (sp->bb == bp->bb)
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

/// Allocate some joins in front of each lazy fork, so that the number of inputs
/// for each of them is exactly one. The current inputs of the lazy forks become
/// inputs for the joins.
static LogicalResult
joinInsertion(OpBuilder &builder, DenseSet<Group *> &groups,
              DenseMap<Block *, Operation *> &forksGraph,
              SmallVector<Operation *> &allocationNetwork) {
  // For each group
  for (Group *group : groups) {
    // Get the corresponding fork and operands
    Operation *forkNode = forksGraph[group->bb];
    ValueRange operands = forkNode->getOperands();
    // If the number of inputs is higher than one
    if (operands.size() > 1) {

      // Join all the inputs, and set the ouptut of this new element as input of
      // the lazy fork
      builder.setInsertionPointToStart(forkNode->getBlock());
      auto joinOp =
          builder.create<handshake::JoinOp>(forkNode->getLoc(), operands);
      allocationNetwork.push_back(joinOp);
      /// The result of the JoinOp becomes the input to the LazyFork
      forkNode->setOperands(joinOp.getResult());
    }
  }
  return success();
}

// -- -End helper functions-- -

using ArgReplacements = DenseMap<BlockArgument, OpResult>;

LogicalResult FtdLowerFuncToHandshake::matchAndRewrite(
    func::FuncOp lowerFuncOp, OpAdaptor /*adaptor*/,
    ConversionPatternRewriter &rewriter) const {

  FtdStoredOperations ftdOps;

  std::string funcName = (std::string)lowerFuncOp.getName();

  if (failed(cdgAnalysis.printAllBlocksDeps(funcName)))
    return failure();

  // Map all memory accesses in the matched function to the index of their
  // memref in the function's arguments
  DenseMap<Value, unsigned> memrefToArgIdx;
  for (auto [idx, arg] : llvm::enumerate(lowerFuncOp.getArguments())) {
    if (isa<mlir::MemRefType>(arg.getType()))
      memrefToArgIdx.insert({arg, idx});
  }

  // First lower the parent function itself, without modifying its body
  // (except the block arguments and terminators)
  auto funcOrFailure = lowerSignature(lowerFuncOp, rewriter);
  if (failed(funcOrFailure))
    return failure();
  handshake::FuncOp funcOp = *funcOrFailure;
  if (funcOp.isExternal())
    return success();

  // Stores mapping from each value that passes through a merge-like
  // operation to the data result of that merge operation
  ArgReplacements argReplacements;
  addMergeOps(funcOp, rewriter, argReplacements);
  addBranchOps(funcOp, rewriter);

  // The memory operations are converted to the corresponding handshake
  // counterparts. No LSQ interface is created yet.
  BackedgeBuilder edgeBuilder(rewriter, funcOp->getLoc());
  LowerFuncToHandshake::MemInterfacesInfo memInfo;
  if (failed(convertMemoryOps(funcOp, rewriter, memrefToArgIdx, edgeBuilder,
                              memInfo)))
    return failure();

  // First round of bb-tagging so that newly inserted Dynamatic memory ports
  // get tagged with the BB they belong to (required by memory interface
  // instantiation logic)
  idBasicBlocks(funcOp, rewriter);

  if (failed(
          ftdVerifyAndCreateMemInterfaces(funcOp, rewriter, memInfo, ftdOps)))
    return failure();

  idBasicBlocks(funcOp, rewriter);
  return flattenAndTerminate(funcOp, rewriter, argReplacements);
}

LogicalResult FtdLowerFuncToHandshake::ftdVerifyAndCreateMemInterfaces(
    handshake::FuncOp funcOp, ConversionPatternRewriter &rewriter,
    MemInterfacesInfo &memInfo, FtdStoredOperations &ftdOps) const {

  if (memInfo.empty())
    return success();

  // Get the CFG loop information
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));

  // Create a mapping between each block and all the other blocks it
  // properly dominates so that we can quickly determine whether LSQ groups
  // make sense
  DenseMap<Block *, DenseSet<Block *>> dominations;
  for (Block &maybeDominator : funcOp) {
    // Start with an empty set of dominated blocks for each potential
    // dominator
    dominations[&maybeDominator] = {};
    for (Block &maybeDominated : funcOp) {
      if (&maybeDominator == &maybeDominated)
        continue;
      if (domInfo.properlyDominates(&maybeDominator, &maybeDominated))
        dominations[&maybeDominator].insert(&maybeDominated);
    }
  }

  // Find the control value indicating the last control flow decision in the
  // function; it will be fed to memory interfaces to indicate that no more
  // group allocations will be coming
  Value ctrlEnd;
  auto returns = funcOp.getOps<func::ReturnOp>();
  assert(!returns.empty() && "no returns in function");
  if (std::distance(returns.begin(), returns.end()) == 1) {
    ctrlEnd = getBlockControl((*returns.begin())->getBlock());
  } else {
    // Merge the control signals of all blocks with a return to create a
    // control representing the final control flow decision
    SmallVector<Value> controls;
    func::ReturnOp lastRetOp;
    for (func::ReturnOp retOp : returns) {
      lastRetOp = retOp;
      controls.push_back(getBlockControl(retOp->getBlock()));
    }
    rewriter.setInsertionPointToStart(lastRetOp->getBlock());
    auto mergeOp =
        rewriter.create<handshake::MergeOp>(lastRetOp.getLoc(), controls);
    ctrlEnd = mergeOp.getResult();

    // The merge goes into an extra "end block" after all others, this will
    // be where the function end terminator will be located as well
    mergeOp->setAttr(BB_ATTR_NAME,
                     rewriter.getUI32IntegerAttr(funcOp.getBlocks().size()));
  }

  // Create a mapping between each block and its control value in the right
  // format for the memory interface builder
  DenseMap<unsigned, Value> ctrlVals;
  for (auto [blockIdx, block] : llvm::enumerate(funcOp))
    ctrlVals.insert({blockIdx, getBlockControl(&block)});

  // Each memory region is independent from the others
  for (auto &[memref, memAccesses] : memInfo) {
    SmallPtrSet<Block *, 4> controlBlocks;

    FtdMemoryInterfaceBuilder memBuilder(funcOp, memref, memAccesses.memStart,
                                         ctrlEnd, ctrlVals);

    // Add MC ports to the interface builder
    for (auto &[_, mcBlockOps] : memAccesses.mcPorts) {
      for (Operation *mcOp : mcBlockOps)
        memBuilder.addMCPort(mcOp);
    }

    // Determine LSQ group validity and add ports the the interface builder
    // at the same time
    for (auto &[group, groupOps] : memAccesses.lsqPorts) {
      assert(!groupOps.empty() && "group cannot be empty");

      // Group accesses by the basic block they belong to
      llvm::MapVector<Block *, SmallVector<Operation *>> opsPerBlock;
      for (Operation *op : groupOps)
        opsPerBlock[op->getBlock()].push_back(op);

      // Check whether there is a clear "linear dominance" relationship
      // between all blocks, and derive a port ordering for the group from
      // it
      SmallVector<Block *> order;
      if (failed(computeLinearDominance(dominations, opsPerBlock, order)))
        return failure();

      // Verify that no two groups have the same control signal
      if (auto [_, newCtrl] = controlBlocks.insert(order.front()); !newCtrl)
        return groupOps.front()->emitError()
               << "Inconsistent LSQ group for memory interface the "
                  "operation "
                  "references. No two groups can have the same control "
                  "signal.";

      // Add all group ports in the correct order to the builder. Within
      // each block operations are naturally in program order since we
      // always use ordered maps and iterated over the operations in program
      // order to begin with
      for (Block *block : order) {
        for (Operation *lsqOp : opsPerBlock[block])
          memBuilder.addLSQPort(group, lsqOp);
      }
    }

    // Build the memory interfaces.
    // If the memory accesses require an LSQ, then the Fast Load-Store queue
    // allocation method from FPGA'23 is used. In particular, first the
    // groups allocation is performed together with the creation of the fork
    // graph. Afterwards, the FTD methodology is used to interconnect the
    // elements correctly.
    if (memAccesses.lsqPorts.size() > 0) {
      // Get all the operations associated to an LSQ
      SmallVector<Operation *> allOperations =
          getLSQOperations(memAccesses.lsqPorts);

      // Get all the dependencies among the BBs of the related operations.
      // Two memory operations are dependant if:
      // 1. They are in different BBs;
      // 2. One of them is a write operations;
      // 3. They are not mutually exclusive.
      SmallVector<ProdConsMemDep> allMemDeps;
      identifyMemoryDependencies(allOperations, allMemDeps, loopInfo);

      for (auto &dep : allMemDeps)
        dep.printDependency();

      // Get the initial start signal, which is the last argument of the
      // function
      auto startValue = (Value)funcOp.getArguments().back();

      // Stores the Groups graph required for the allocation network
      // analysis
      DenseSet<Group *> groupsGraph;
      constructGroupsGraph(allOperations, allMemDeps, groupsGraph);
      minimizeGroupsConnections(groupsGraph);

      for (auto &g : groupsGraph)
        g->printDependenices();

      // Build the memory interfaces
      handshake::MemoryControllerOp mcOp;
      handshake::LSQOp lsqOp;

      // As we instantiate the interfaces for the LSQ for each memory operation,
      // we need to add some forks in order for the control input to be
      // propagated. In particular, we want to keep track of the control valuea
      // associated to each basic block in the region
      DenseMap<Block *, Operation *> forksGraph;

      if (failed(memBuilder.instantiateInterfacesWithForks(
              rewriter, mcOp, lsqOp, groupsGraph, forksGraph, startValue,
              ftdOps.allocationNetwork)))
        return failure();

      if (failed(joinInsertion(rewriter, groupsGraph, forksGraph,
                               ftdOps.allocationNetwork)))
        return failure();

      // [TODO] This branch is not completed yet...
    }

    handshake::MemoryControllerOp mcOp;
    handshake::LSQOp lsqOp;
    if (failed(memBuilder.instantiateInterfaces(rewriter, mcOp, lsqOp)))
      return failure();
  }

  return success();
}

void FtdLowerFuncToHandshake::identifyMemoryDependencies(
    const SmallVector<Operation *> &operations,
    SmallVector<ProdConsMemDep> &allMemDeps,
    const mlir::CFGLoopInfo &li) const {

  // Given all the operations which are assigned to an LSQ, loop over them
  // and skip those which are not memory operations
  for (Operation *i : operations) {

    if (!isHandhsakeLSQOperation(i)) {
      continue;
    }

    // Loop over all the other operations in the LSQ. There is no dependency
    // in the following cases:
    // 1. One of them is not a memory operation;
    // 2. The two operation are in the same group, thus they are in the same
    // BB;
    // 3. They are both load operations;
    // 4. The operations are mutually exclusive (i.e. there is no path which
    // goes from i to j and vice-versa);
    for (Operation *j : operations) {

      if (!isHandhsakeLSQOperation(j) || isSameBlock(i, j) ||
          areBothLoad(i, j) || (!isThereAPath(i, j) && !isThereAPath(j, i)))
        continue;

      // Get the two blocks
      Block *bbI = i->getBlock();
      Block *bbJ = j->getBlock();

      // If the relationship was already present, then skip the pairs of
      // blocks
      auto *it = llvm::find_if(allMemDeps, [bbI, bbJ](ProdConsMemDep p) {
        return p.prodBb == bbJ && p.consBb == bbI;
      });

      if (it != allMemDeps.end())
        continue;

      // Insert a dependency only if index _j is smaller than index _i: in
      // this case i is the producer, j is the consumer. If this doesn't
      // hold, the dependency will be added when the two blocks are analyzed
      // in the opposite direction
      if (lessThanBlocks(bbJ, bbI)) {

        // bbI is the producer, bbJ is the consumer: create a new dependency
        // and add it to the list of dependencies
        ProdConsMemDep oneMemDep(bbJ, bbI, false);
        allMemDeps.push_back(oneMemDep);

        // If the two blocks are in the same loop, then bbI is also a
        // consumer, while bbJ is a producer. This relationship is backward.
        if (isSameLoopBlocks(bbI, bbJ, li)) {
          ProdConsMemDep opp(bbI, bbJ, true);
          allMemDeps.push_back(opp);
        }
      }
    }
  }
}

std::unique_ptr<dynamatic::DynamaticPass> createFtdCfToHandshake() {
  return std::make_unique<FtdCfToHandshakePass>();
}

} // namespace ftd
} // namespace experimental
} // namespace dynamatic
