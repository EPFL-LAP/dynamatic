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
#include "dynamatic/Support/CFG.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/BooleanLogic/Shannon.h"
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
#include "llvm/ADT/TypeSwitch.h"
#include <unordered_set>
#include <utility>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::boolean;

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

    patterns.add<experimental::ftd::FtdLowerFuncToHandshake, ConvertCalls,
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

// --- Helper functions ---

static Type channelifyType(Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case<IndexType, IntegerType, FloatType>(
          [](auto type) { return handshake::ChannelType::get(type); })
      .Case<MemRefType>([](MemRefType memrefType) {
        if (!isa<IndexType>(memrefType.getElementType()))
          return memrefType;
        OpBuilder builder(memrefType.getContext());
        IntegerType elemType = builder.getIntegerType(32);
        return MemRefType::get(memrefType.getShape(), elemType);
      })
      .Case<handshake::ChannelType, handshake::ControlType>(
          [](auto type) { return type; })

      .Default([](auto type) { return nullptr; });
}

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

  // Remove the current block from the current path and from the list of
  // visited blocks
  path.pop_back();
  visited.erase(start);
}

/// Recursive function which allows to obtain all the paths from operation
/// `start` to operation `end` using a DFS
void dfsAllPaths(Operation *current, Operation *end,
                 std::unordered_set<Operation *> &visited,
                 std::vector<Operation *> &path,
                 std::vector<std::vector<Operation *>> &allPaths) {
  visited.insert(current);
  path.push_back(current);

  if (current == end) {
    // If the current operation is the end, add the path to allPaths
    allPaths.push_back(path);
  } else {
    // Otherwise, explore the successors
    for (auto result : current->getResults()) {
      for (auto *successor : result.getUsers()) {
        if (visited.find(successor) == visited.end()) {
          dfsAllPaths(successor, end, visited, path, allPaths);
        }
      }
    }
  }

  // Backtrack
  path.pop_back();
  visited.erase(current);
}

/// Gets all the paths from operation `start` to operation `end` using a dfs
/// search
static std::vector<std::vector<Operation *>> findAllPaths(Operation *start,
                                                          Operation *end) {
  std::vector<std::vector<Operation *>> allPaths;
  std::unordered_set<Operation *> visited;
  std::vector<Operation *> path;
  dfsAllPaths(start, end, visited, path, allPaths);
  return allPaths;
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

/// Returns true if there exist a path between `op1` and `op2`
static bool isThereAPath(Operation *op1, Operation *op2) {
  return !findAllPaths(op1->getBlock(), op2->getBlock()).empty();
}

/// Get the index of a basic block
static int getBlockIndex(Block *bb) {
  std::string result1;
  llvm::raw_string_ostream os1(result1);
  bb->printAsOperand(os1);
  std::string block1id = os1.str();
  return std::stoi(block1id.substr(3));
}

/// Check whether the index of `block1` is less than the one of `block2`
static bool lessThanBlocks(Block *block1, Block *block2) {
  // Compare the two integers
  return getBlockIndex(block1) < getBlockIndex(block2);
}

/// Check whether the index of `block1` is greater than the one of `block2`
bool greaterThanBlocks(Block *block1, Block *block2) {
  // Compare the two integers
  return getBlockIndex(block1) > getBlockIndex(block2);
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
/// there is no point in D waiting for C to be done. For this reason, the
/// graph can be simplified, saving and edge:
///
/// B -> C -> D
static void minimizeGroupsConnections(DenseSet<Group *> &groupsGraph) {

  // Get the dominance info for the region
  DominanceInfo domInfo;

  // For each group, compare all the pairs of its predecessors. Cut the edge
  // between them iff the predecessor with the bigger index dominates the
  // whole group
  for (auto &group : groupsGraph) {
    // List of predecessors to remove
    DenseSet<Group *> predsToRemove;
    for (auto &bp : group->preds) {
      // If the big predecessor is already in the list to remove, ignore it
      if (llvm::find(predsToRemove, bp) != predsToRemove.end())
        continue;
      for (auto &sp : group->preds) {
        // If the small predecessor has bigger index than the big predecessor,
        // ignore it
        if (lessThanBlocks(bp->bb, sp->bb))
          continue;
        // If the small predecessor is already in the list to remove, ignore
        // it
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

/// Allocate some joins in front of each lazy fork, so that the number of
/// inputs for each of them is exactly one. The current inputs of the lazy
/// forks become inputs for the joins.
static LogicalResult joinInsertion(OpBuilder &builder,
                                   DenseSet<Group *> &groups,
                                   DenseMap<Block *, Operation *> &forksGraph,
                                   DenseSet<Operation *> &allocationNetwork) {
  // For each group
  for (Group *group : groups) {
    // Get the corresponding fork and operands
    Operation *forkNode = forksGraph[group->bb];
    ValueRange operands = forkNode->getOperands();
    // If the number of inputs is higher than one
    if (operands.size() > 1) {

      // Join all the inputs, and set the output of this new element as input
      // of the lazy fork
      builder.setInsertionPointToStart(forkNode->getBlock());
      auto joinOp =
          builder.create<handshake::JoinOp>(forkNode->getLoc(), operands);
      allocationNetwork.insert(joinOp);
      /// The result of the JoinOp becomes the input to the LazyFork
      forkNode->setOperands(joinOp.getResult());
    }
  }
  return success();
}

/// Given two sets containing object of type `Type`, remove the common entries
template <typename Type>
void eliminateCommonEntries(DenseSet<Type> &s1, DenseSet<Type> &s2) {

  std::vector<Type> intersection;
  set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                   back_inserter(intersection));

  for (auto &bb : intersection) {
    s1.erase(bb);
    s2.erase(bb);
  }
}

/// Given a block whose name is `^BBN` (where N is an integer) return a string
/// in the format `cN`, used to identify the condition which allows the block
/// to be executed.
static std::string getBlockCondition(Block *block) {
  std::string blockCondition = "c" + std::to_string(getBlockIndex(block));
  return blockCondition;
}

/// Given a path in the DFG of the basic blocks, compute the minterm which
/// defines whether that path will be covered or not.This correspond to the
/// product of all the boolean conditions associated to each basic block.
static BoolExpression *pathToMinterm(const std::vector<Block *> &path,
                                     const DenseSet<Block *> &controlDeps) {

  // Start from 1 as minterm. If no path exist, the result will be `1` as well
  BoolExpression *exp = BoolExpression::boolOne();

  // Consider each element of the path from 0 to N-2, so that pairs of
  // consecutive blocks can be analyzed. Multiply `exp` for a new minterm if
  // the there is a conditional branch to go from element `i` to `i+1`.
  for (unsigned i = 0; i < path.size() - 1; i++) {

    // Get the first block
    Block *prod = path.at(i);

    // A condition should be taken into account only if the following block is
    // control dependent on the previous one. Otherwise, the following is
    // always executed
    if (controlDeps.contains(prod)) {

      // Get the next
      Block *cons = path.at(i + 1);

      // Get last operation of the block, also called `terminator`
      Operation *producerTerminator = prod->getTerminator();

      // Get the condition which allows the execution of the producer
      BoolExpression *prodCondition =
          BoolExpression::parseSop(getBlockCondition(prod));

      // If the terminator operation of the consumer is a conditional branch,
      // then its condition must be taken into account to know if the
      // following block will be executed or not.
      if (isa<cf::CondBranchOp>(producerTerminator)) {

        auto condOp = dyn_cast<cf::CondBranchOp>(producerTerminator);

        // If the following BB is on the FALSE side of the current BB, then
        // negate the condition of the current BB
        if (cons == condOp.getFalseDest())
          prodCondition = prodCondition->boolNegate();
      }
      // Modify the resulting expression
      exp = BoolExpression::boolAnd(exp, prodCondition);
    }
  }
  return exp;
}

/// The boolean condition to either generate or suppress a token are computed
/// by considering all the paths from the producer (`start`) to the consumer
/// (`end`). "Each path identifies a Boolean product of elementary conditions
/// expressing the reaching of the target BB from the corresponding member of
/// the set; the product of all such paths are added".
static BoolExpression *enumeratePaths(Block *start, Block *end,
                                      const DenseSet<Block *> &controlDeps) {
  // Start with a boolean expression of zero (so that new conditions can be
  // added)
  BoolExpression *sop = BoolExpression::boolZero();

  // Find all the paths from the producer to the consumer, using a DFS
  std::vector<std::vector<Block *>> allPaths = findAllPaths(start, end);

  // For each path
  for (const std::vector<Block *> &path : allPaths) {

    // Compute the product of the conditions which allow that path to be
    // executed
    BoolExpression *minterm = pathToMinterm(path, controlDeps);

    // Add the value to the result
    sop = BoolExpression::boolOr(sop, minterm);
  }
  return sop->boolMinimizeSop();
}

/// Helper recursive function for getPostDominantSuccessor
static Block *getPostDominantSuccessor(Block *prod, Block *cons,
                                       std::unordered_set<Block *> &visited,
                                       PostDominanceInfo &postDomInfo) {
  // If the producer is not valid, return, otherwise insert it among the
  // visited ones.
  if (!prod)
    return nullptr;
  visited.insert(prod);

  // For each successor of the producer
  for (Block *successor : prod->getSuccessors()) {

    // Check if the successor post-dominates cons
    if (successor != cons && postDomInfo.postDominates(successor, cons))
      return successor;

    // If not visited, recursively search successors of the current successor
    if (visited.find(successor) == visited.end()) {
      Block *result =
          getPostDominantSuccessor(successor, cons, visited, postDomInfo);
      if (result)
        return result;
    }
  }
  return nullptr;
}

/// Given a pair of consumer and producer, we are interested in a basic block
/// which is a successor of the producer and post-dominates the consumer.
/// If this block exists, the MERGE/GENERATE block can be put right after it,
/// since all paths between the producer and the consumer pass through it.
static Block *getPostDominantSuccessor(Block *prod, Block *cons) {
  std::unordered_set<Block *> visited;
  PostDominanceInfo postDomInfo;
  return getPostDominantSuccessor(prod, cons, visited, postDomInfo);
}

/// Helper recursive function for getPredecessorDominatingAndPostDominating
static Block *getPredecessorDominatingAndPostDominating(
    Block *producer, Block *consumer, std::unordered_set<Block *> &visited,
    DominanceInfo &domInfo, PostDominanceInfo &postDomInfo) {

  // If the consumer is not valid, return, otherwise insert it in the visited
  // ones
  if (!consumer)
    return nullptr;
  visited.insert(consumer);

  // For each predecessor of the consumer
  for (Block *predecessor : consumer->getPredecessors()) {

    // If the current predecessor is not the producer itself, and this block
    // both dominates the consumer and post-dominates the producer, return it
    if (predecessor != producer &&
        postDomInfo.postDominates(predecessor, producer) &&
        domInfo.dominates(predecessor, consumer))
      return predecessor;

    // If not visited, recursively search predecessors of the current
    // predecessor
    if (visited.find(predecessor) == visited.end()) {
      Block *result = getPredecessorDominatingAndPostDominating(
          producer, predecessor, visited, domInfo, postDomInfo);
      if (result)
        return result;
    }
  }
  return nullptr;
}

/// Given a pair of consumer and producer, we are interested in a basic block
/// which both dominates the consumer and post-dominates the producer. If this
/// block exists, the MERGE/GENERATE block can be put right after it, since
/// all paths between the producer and the consumer pass through it.
static Block *getPredecessorDominatingAndPostDominating(Block *prod,
                                                        Block *cons) {
  std::unordered_set<Block *> visited;
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
  return getPredecessorDominatingAndPostDominating(prod, cons, visited, domInfo,
                                                   postDomInfo);
}

LogicalResult FtdLowerFuncToHandshake::addMergeNonLoop(
    handshake::FuncOp &funcOp, OpBuilder &builder,
    SmallVector<ProdConsMemDep> &allMemDeps, DenseSet<Group *> &groups,
    DenseMap<Block *, Operation *> &forksGraph, FtdStoredOperations &ftdOps,
    Value startCtrl) const {

  // Get entry block of the function to lower
  Block *entryBlock = &funcOp.getRegion().front();

  // Stores the information related to the control dependencies ob basic
  // blocks within an handshake::funcOp object
  ControlDependenceAnalysis<dynamatic::handshake::FuncOp> cdgAnalysis(funcOp);

  // For each group within the groups graph
  for (Group *producerGroup : groups) {

    // Get the block associated to that same group
    Block *producerBlock = producerGroup->bb;

    // Compute all the forward dependencies of that block
    DenseSet<Block *> producerControlDeps;
    if (failed(cdgAnalysis.getBlockForwardControlDeps(producerBlock,
                                                      producerControlDeps)))
      return failure();

    // For each successor (which is now considered as a consumer)
    for (Group *consumerGroup : producerGroup->succs) {

      // Get its basic block
      Block *consumerBlock = consumerGroup->bb;

      // Compute all the forward dependencies of that block
      DenseSet<Block *> consumerControlDeps;
      if (failed(cdgAnalysis.getBlockForwardControlDeps(consumerBlock,
                                                        consumerControlDeps)))
        return failure();

      // Remove the common forward dependencies among the two blocks
      eliminateCommonEntries<Block *>(producerControlDeps, consumerControlDeps);

      // Compute the boolean function `fProd`, that is true when the producer
      // is going to produce the token (cfr. FPGA'22, IV.C: Generating and
      // Suppressing Tokens)
      BoolExpression *fProd =
          enumeratePaths(entryBlock, producerBlock, producerControlDeps);

      // Compute the boolean function `fCons`, that is true when the consumer
      // is going to consume the token (cfr. FPGA'22, IV.C: Generating and
      // Suppressing Tokens)
      BoolExpression *fCons =
          enumeratePaths(entryBlock, consumerBlock, consumerControlDeps);

      // A token needs to be generated when the consumer consumes  but the
      // producer does not producer. Compute the corresponding function and
      // minimize it.
      BoolExpression *fGen =
          BoolExpression::boolAnd(fCons, fProd->boolNegate())->boolMinimize();

      // If the condition for the generation is not zero a `GENERATE` block
      // needs to be inserted, which is a multiplexer/mux
      if (fGen->type != experimental::boolean::ExpressionType::Zero) {

        // Find the memory dependence related to the current producer and
        // consumer
        auto *memDepIt = llvm::find_if(
            allMemDeps,
            [producerBlock, consumerBlock](const ProdConsMemDep &dep) {
              return dep.prodBb == producerBlock && dep.consBb == consumerBlock;
            });
        if (memDepIt == allMemDeps.end())
          return failure();
        ProdConsMemDep &memDep = *memDepIt;

        // The merge needs to be inserted before the consumer and its fork
        builder.setInsertionPointToStart(consumerBlock);
        Location loc = forksGraph[consumerBlock]->getLoc();

        // Adjust insertion position: if there is a block which is always
        // traversed between the producer and the consumer, the merge/generate
        // block can be put right after it.
        //
        // If the relationship between consumer and producer is backward, we
        // are interested in a block which is a successor of the producer and
        // post-dominates the consumer. If the relationship is forward, we are
        // interested in a block which dominates the consumer and
        // post-dominates the consumer.
        Block *bbNewLoc = nullptr;
        if (memDep.isBackward) {
          bbNewLoc = getPostDominantSuccessor(producerBlock, consumerBlock);
        } else {
          bbNewLoc = getPredecessorDominatingAndPostDominating(producerBlock,
                                                               consumerBlock);
        }
        if (bbNewLoc) {
          builder.setInsertionPointToStart(bbNewLoc);
          loc = bbNewLoc->getOperations().front().getLoc();
        }

        // The possible inputs of the merge are the start value and the first
        // output of the producer fork
        SmallVector<Value> mergeOperands;
        mergeOperands.push_back(startCtrl);
        mergeOperands.push_back(forksGraph[producerBlock]->getResult(0));
        auto mergeOp = builder.create<handshake::MergeOp>(loc, mergeOperands);
        ftdOps.allocationNetwork.insert(mergeOp);

        // At this point, the output of the merge is the new producer, which
        // becomes an input for the consumer.
        forksGraph[consumerBlock]->replaceUsesOfWith(
            forksGraph[producerBlock]->getResult(0), mergeOp->getResult(0));
      }
    }
  }
  return success();
}

/// Helper recursive function to get the innermost common loop
static CFGLoop *checkInnermostCommonLoop(CFGLoop *loop1, CFGLoop *loop2) {

  // None of them is a loop
  if (!loop1 || !loop2)
    return nullptr;

  // Same loop
  if (loop1 == loop2)
    return loop1;

  // Check whether the parent loop of `loop1` is `loop2`
  if (CFGLoop *pl = checkInnermostCommonLoop(loop1->getParentLoop(), loop2); pl)
    return pl;

  // Check whether the parent loop of `loop2` is `loop1`
  if (CFGLoop *pl = checkInnermostCommonLoop(loop2->getParentLoop(), loop1); pl)
    return pl;

  // Check whether the parent loop of `loop1` is identical to the parent loop
  // of `loop1`
  if (CFGLoop *pl = checkInnermostCommonLoop(loop2->getParentLoop(),
                                             loop1->getParentLoop());
      pl)
    return pl;

  // If no common loop is found, return nullptr
  return nullptr;
}

/// Given two blocks, return a reference to the innermost common loop. The
/// result is `nullptr` if the two blocks are not within a loop
static CFGLoop *getInnermostCommonLoop(Block *block1, Block *block2,
                                       mlir::CFGLoopInfo &li) {
  return checkInnermostCommonLoop(li.getLoopFor(block1), li.getLoopFor(block2));
}

LogicalResult FtdLowerFuncToHandshake::addMergeLoop(
    handshake::FuncOp &funcOp, OpBuilder &builder,
    SmallVector<ProdConsMemDep> &allMemDeps, DenseSet<Group *> &groups,
    DenseMap<Block *, Operation *> &forksGraph, FtdStoredOperations &ftdOps,
    Value startCtrl) const {

  // Get the dominance info about the current region, in order to compute the
  // properties of loop
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));

  // For each group within the groups graph
  for (Group *consGroup : groups) {

    // Get the block associated to that same group
    Block *cons = consGroup->bb;

    // For each predecessor (which is now considered as a producer)
    for (Group *prodGroup : consGroup->preds) {

      // Get its basic block
      Block *prod = prodGroup->bb;

      // If the consumer comes before a producer, it might mean the two basic
      // blocks are within a loop
      if (greaterThanBlocks(prod, cons)) {

        if (auto *loop = getInnermostCommonLoop(prod, cons, loopInfo); loop) {

          // A merge is inserted at the beginning of the loop, getting as
          // operands both `start` and the result of the producer as operand.
          // As many merges must be added as the number of paths from the
          // producer to the consumer (corresponding to the amount of loops
          // which involve both of them).
          Block *loopHeader = loop->getHeader();
          builder.setInsertionPointToStart(loopHeader);

          std::vector<std::vector<Operation *>> allPaths =
              findAllPaths(forksGraph[prod], forksGraph[cons]);

          // For each path
          for (std::vector<Operation *> path : allPaths) {
            SmallVector<Value> operands;

            // Get the result of the operation before the consumer (this will
            // become an input of the merge)
            Value mergeOperand = path.at(path.size() - 2)->getResult(0);
            operands.push_back(startCtrl);
            operands.push_back(mergeOperand);

            // Add the merge and update the FTD data structures
            auto mergeOp = builder.create<handshake::MergeOp>(
                mergeOperand.getLoc(), operands);
            ftdOps.allocationNetwork.insert(mergeOp);
            ftdOps.memDepLoopMerges.insert(mergeOp);

            // The merge becomes the producer now, so connect the result of
            // the MERGE as an operand of the Consumer. Also remove the old
            // connection between the producer's LazyFork and the consumer's
            // LazyFork Connect the MERGE to the consumer's LazyFork
            forksGraph[cons]->replaceUsesOfWith(mergeOperand,
                                                mergeOp->getResult(0));
          }
        }
      }
    }
  }
  return success();
}

/// Checks if an operation is a Branch in a loop exit block
static bool isBranchInLoopExit(Operation *op, CFGLoopInfo &loopInfo) {
  if (isa<handshake::ConditionalBranchOp>(op)) {
    // Get the loop in which the corresponding basic block is
    CFGLoop *loop = loopInfo.getLoopFor(op->getBlock());
    if (loop) {
      llvm::SmallVector<Block *> exitBlocks;
      // check whether the basic block is in one of the exit blocks of the loop
      loop->getExitingBlocks(exitBlocks);
      return llvm::find(exitBlocks, op->getBlock()) != exitBlocks.end();
    }
  }
  return false;
}

/// Modify a merge operation related to the INIT process of a variable, so that
/// input number 0 comes from outside of the loop, while input numer 1 comes
/// from the loop itself
static void fixMergeConvention(Operation *merge, CFGLoop *loop,
                               CFGLoopInfo &li) {
  // Swap if the operand 0 comes from the loop itself
  Value firstOperand = merge->getOperand(0);
  if (li.getLoopFor(merge->getOperand(0).getParentBlock()) == loop) {
    Value secondOperand = merge->getOperand(1);
    merge->setOperand(0, secondOperand);
    merge->setOperand(1, firstOperand);
  }
}

/// Add an INIT merge, so that the initialization value of a loop is generated
/// only at the first iteration of the loop. This merge cannot be replaced by a
/// mux
static Value addInit(ConversionPatternRewriter &rewriter,
                     DenseSet<Operation *> &initMerges, Operation *oldMerge,
                     FtdStoredOperations &ftdOps, Value &startValue,
                     CFGLoopInfo &li) {

  SmallVector<Value> mergeOperands;

  // Given the merge we are currently translating, one of its inputs must be a
  // branch in a loop exit block (that is, the regeneration of the loop
  // variable)
  bool inputIsBranchInLoopExit = false;
  Value mergeOperandFromMux;
  handshake::ConditionalBranchOp branchOp;

  // For each operand
  for (Value operand : oldMerge->getOperands()) {
    // For each of its producers
    if (Operation *producer = operand.getDefiningOp()) {
      // If it is the branch on a loop exit
      if (isBranchInLoopExit(producer, li)) {
        // Get the branch, which is supposed to exist
        branchOp = dyn_cast<handshake::ConditionalBranchOp>(producer);
        mergeOperandFromMux = branchOp.getConditionOperand();
        if (isa<handshake::NotOp>(mergeOperandFromMux.getDefiningOp()))
          mergeOperandFromMux =
              mergeOperandFromMux.getDefiningOp()->getOperand(0);
        inputIsBranchInLoopExit = true;
        break;
      }
    }
  }
  assert(inputIsBranchInLoopExit &&
         "An input of the Merge must be a Branch in a loop exit block");

  // The value of the constant we need to addis either 0 or 1, depending on
  // which input of the merge we are currently translating to mux is coming from
  // outsize of the loop:
  // 1. If the first input of the merge that we are translating to mux is coming
  // from outside of the loop, the value of the constant should be 0;
  // 2. If the second input of the merge that we are translating to mux is
  // coming from the outside of the loop, the value of the constant should be 1.
  if (branchOp.getResult(0) == oldMerge->getOperand(1) ||
      branchOp.getResult(1) == oldMerge->getOperand(0)) {
    rewriter.setInsertionPointAfterValue(mergeOperandFromMux);
    auto notOp = rewriter.create<handshake::NotOp>(mergeOperandFromMux.getLoc(),
                                                   mergeOperandFromMux);
    mergeOperandFromMux = notOp->getResult(0);
  }

  // Insert a new constant in the same block as that of the new merge and feed
  // its input from the start value.

  auto cstType = rewriter.getIntegerType(1);
  auto cstAttr = IntegerAttr::get(cstType, 0);
  rewriter.setInsertionPointToStart(oldMerge->getBlock());
  auto constOp = rewriter.create<handshake::ConstantOp>(oldMerge->getLoc(),
                                                        cstAttr, startValue);

  ftdOps.networkConstants.insert(constOp);

  mergeOperands.push_back(constOp.getResult());
  mergeOperands.push_back(mergeOperandFromMux);

  // Create the merge
  auto mergeOp =
      rewriter.create<handshake::MergeOp>(oldMerge->getLoc(), mergeOperands);
  initMerges.insert(mergeOp);
  ftdOps.allocationNetwork.insert(mergeOp);
  return mergeOp->getResult(0);
}

LogicalResult FtdLowerFuncToHandshake::convertMergesToMuxes(
    ConversionPatternRewriter &rewriter, handshake::FuncOp &funcOp,
    FtdStoredOperations &ftdOps) const {

  // Some merges are the INIT components for loops, so that the initial value of
  // a phi function can be used. In this case, the merge should not be modified
  DenseSet<Operation *> initMerges;

  // Get information about the loop and start value of the function
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));
  auto startValue = (Value)funcOp.getArguments().back();

  Region &region = funcOp.getBody();

  // For each merge withing each block
  for (Block &block : region.getBlocks()) {
    for (Operation &merge : block.getOperations()) {

      // Skip if the operation is not related to the allocation network
      if (!ftdOps.allocationNetwork.contains(&merge))
        continue;

      // Skip if it is not a marge or if it is an INIT merge
      if (!isa<handshake::MergeOp>(merge) || initMerges.contains(&merge))
        continue;

      bool loopHeader = false;

      // Check whether we have a loop header, thus we need to insert an INIT
      // merge. This happens if the block is inside a loop, its block is the
      // header one and one of the inputs of the merge is not coming from the
      // same block (on the contrary, it might happen that the merge has two
      // operands from the same input)
      if (loopInfo.getLoopFor(&block))
        loopHeader =
            (loopInfo.getLoopFor(&block)->getHeader() == &block &&
             loopInfo.getLoopFor(merge.getOperand(0).getParentBlock()) !=
                 loopInfo.getLoopFor(merge.getOperand(1).getParentBlock()));

      // Case of a loop header
      if (loopHeader) {

        // We want the first operand of the merge is the one coming from outside
        // the loop.
        fixMergeConvention(&merge, loopInfo.getLoopFor(&block), loopInfo);

        // The INIT component is in charge of choosing the value 0 of the
        // mux only in case of the first loop iteration
        Value select =
            addInit(rewriter, initMerges, &merge, ftdOps, startValue, loopInfo);

        // The merge itself is transformed into a mux driven by the INIT
        // signal
        rewriter.setInsertionPointAfter(&merge);

        auto mux = rewriter.create<handshake::MuxOp>(merge.getLoc(), select,
                                                     merge.getOperands());
        ftdOps.allocationNetwork.insert(mux);
        rewriter.replaceOp(&merge, mux);

      } else {
        // Pick the select signal as output of a cmerge
        Value select = getBlockControl(&block);

        // [TODO]
        // The first input of the cmerge and the first input of the mux are in
        // the same block
        // bool firstSameBlock =
        //     select.getDefiningOp()->getOperand(0).getParentBlock() ==
        //     merge.getOperand(0).getParentBlock();

        // // The block of the mux first input dominates the block of the cmerge
        // // first input
        // bool firstMuxDominateMerge = domInfo.properlyDominates(
        //     merge.getOperand(0).getParentBlock(),
        //     select.getDefiningOp()->getOperand(0).getParentBlock());

        // // The second input of the cmerge and the second input of the mux are
        // // in the same block
        // bool secondSameBlock =
        //     select.getDefiningOp()->getOperand(1).getParentBlock() ==
        //     merge.getOperand(1).getParentBlock();

        // // The block of the mux second input dominates the block of the
        // cmerge
        // // second input
        // bool secondMuxDominateMerge = domInfo.properlyDominates(
        //     merge.getOperand(1).getParentBlock(),
        //     select.getDefiningOp()->getOperand(1).getParentBlock());

        // // In case one the two conditions do not hold at the same time, the
        // // select signal needs to undergoes a not
        // if (!((firstSameBlock || firstMuxDominateMerge) &&
        //       (secondSameBlock || secondMuxDominateMerge))) {
        //   rewriter.setInsertionPointAfterValue(select);
        //   auto notOp =
        //       rewriter.create<handshake::NotOp>(select.getLoc(), select);
        //   select = notOp->getResult(0);
        // }

        // Convert to a mux
        rewriter.setInsertionPointAfter(&merge);
        auto mux = rewriter.create<handshake::MuxOp>(merge.getLoc(), select,
                                                     merge.getOperands());
        ftdOps.allocationNetwork.insert(mux);
        rewriter.replaceOp(&merge, mux);

        llvm::dbgs() << "\n[BLOCK] ";
        block.printAsOperand(llvm::dbgs());
        llvm::dbgs() << "\n[FUNC]\n";
        funcOp.print(llvm::dbgs());
      }
    }
  }
  return success();
}

static bool
findClosestBranchPredecessor(Value input, DominanceInfo &domInfo, Block &block,
                             Value &desiredCond, bool &getTrueSuccessor,
                             std::unordered_set<Operation *> &visited) {
  Operation *defOp = input.getDefiningOp();
  if (!defOp || visited.count(defOp))
    return false;

  visited.insert(defOp);

  for (Value pred : defOp->getOperands()) {
    Operation *predOp = pred.getDefiningOp();
    if (!predOp)
      continue;

    if (isa<handshake::ConditionalBranchOp>(predOp)) {
      auto branch = dyn_cast<handshake::ConditionalBranchOp>(predOp);
      for (Value branchPred : branch->getOperands()) {
        if (domInfo.dominates(branchPred.getParentBlock(), &block)) {
          desiredCond = branch.getConditionOperand();
          if (pred == branch.getFalseResult()) {
            getTrueSuccessor = true;
          }
          return true;
        }
      }
    }

    if (findClosestBranchPredecessor(pred, domInfo, block, desiredCond,
                                     getTrueSuccessor, visited)) {
      return true;
    }
  }

  return false;
}

/// Gets the closest Branch predecessor to the input and accesses its condition
static bool findClosestBranchPredecessor(Value input, DominanceInfo &domInfo,
                                         Block &block, Value &desiredCond,
                                         bool &getTrueSuccessor) {
  std::unordered_set<Operation *> visited;
  return findClosestBranchPredecessor(input, domInfo, block, desiredCond,
                                      getTrueSuccessor, visited);
}

LogicalResult
FtdLowerFuncToHandshake::addSuppGSA(ConversionPatternRewriter &rewriter,
                                    handshake::FuncOp &funcOp,
                                    FtdStoredOperations &ftdOps) const {
  Region &region = funcOp.getBody();

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));
  auto startValue = (Value)funcOp.getArguments().back();

  for (Block &block : region.getBlocks()) {

    // Loop thorugh all the operations, looking for muxes which do not relate to
    // loop headers
    for (Operation &op : block.getOperations()) {

      if (!ftdOps.allocationNetwork.contains(&op) &&
          (!isa<handshake::ConstantOp>(op) ||
           ftdOps.networkConstants.contains(&op)))
        continue;

      // Skip if it is a mux introduced by shannon, if it is not a multiplexer
      // or if it is a loop header multiplexer
      if (ftdOps.shannonMUXes.contains(&op))
        continue;

      bool loopHeader = loopInfo.getLoopFor(&block) &&
                        (loopInfo.getLoopFor(&block)->getHeader() == &block);

      if (!isa<handshake::MuxOp>(op) || loopHeader)
        continue;

      auto mux = dyn_cast<handshake::MuxOp>(op);

      // We want to check whether one input is dominating the mux's block
      bool inputIsDominating = false;
      Value firstInputMux = mux.getOperand(1);
      Value secondInputMux = mux.getOperand(2);
      Value dominatingInput = firstInputMux;
      Value nonDominatingInput = secondInputMux;

      if (domInfo.dominates(firstInputMux.getParentBlock(), &block)) {
        inputIsDominating = true;
      } else if (domInfo.dominates(secondInputMux.getParentBlock(), &block)) {
        inputIsDominating = true;
        dominatingInput = secondInputMux;
        nonDominatingInput = firstInputMux;
      }

      if (!inputIsDominating)
        continue;

      // We don't want both the inputs to dominate the mux
      assert(!domInfo.dominates(nonDominatingInput.getParentBlock(), &block) &&
             "The BB of the other input of the Mux should not dominate the BB "
             "of the Mux");

      Value desiredCond;
      bool getTrueSuccessor = false;
      bool hasPredBranch = findClosestBranchPredecessor(
          nonDominatingInput, domInfo, block, desiredCond, getTrueSuccessor);

      assert(hasPredBranch && "At least one predecessor of the non-dominating "
                              "input must be a Branch");

      rewriter.setInsertionPointAfterValue(dominatingInput);
      auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
          dominatingInput.getLoc(), desiredCond, dominatingInput);
      ftdOps.allocationNetwork.insert(branchOp);

      Value newInput = branchOp.getFalseResult();
      if (getTrueSuccessor)
        newInput = branchOp.getTrueResult();

      mux->replaceUsesOfWith(dominatingInput, newInput);
    }
  }
  return success();
}

// -- -End helper functions-- -

using ArgReplacements = DenseMap<BlockArgument, OpResult>;

LogicalResult FtdLowerFuncToHandshake::matchAndRewrite(
    func::FuncOp lowerFuncOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  FtdStoredOperations ftdOps;

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

  // Create the memory interface according to the algorithm from FPGA'23. The
  // data dependencies which are created in this way will be then modified by
  // the FTD methodology.
  if (failed(
          ftdVerifyAndCreateMemInterfaces(funcOp, rewriter, memInfo, ftdOps)))
    return failure();

  // Convert the constants from the `arith` dialect to the `handshake`
  // dialect, while also using the start value as their control value. While
  // the usage of the `MatchAndRewrite` system is effective for most of the
  // conversions, since the FTD algorithm requires the constants to be already
  // converted before it can run effectively, this operation has to be done in
  // here.
  if (failed(convertConstants(rewriter, funcOp)))
    return failure();

  // For the same reason as above, the conversion of the undefined values
  // should happen before the FTD algorithm. Since they now are constants,
  // each value has to be activated by the function start value as well.
  if (failed(convertUndefinedValues(rewriter, funcOp)))
    return failure();

  // Add phi
  if (failed(addPhi(rewriter, funcOp, ftdOps)))
    return failure();

  // Add suppression blocks between each pair of producer and consumer
  if (failed(addSupp(rewriter, funcOp, ftdOps)))
    return failure();

  // Add supp branches
  DenseSet<Operation *> oldBranches;
  while (true) {
    size_t counter = ftdOps.suppBranches.size();
    if (failed(addSuppBranches(rewriter, funcOp, ftdOps, oldBranches)))
      return failure();
    if (ftdOps.suppBranches.size() == counter)
      break;
  }

  // Add supp for start
  if (failed(addSuppForStart(rewriter, funcOp, ftdOps)))
    return failure();

  // [TODO] Fix the transformation from merges to mux
  // Convert merges to muxes
  // if (failed(convertMergesToMuxes(rewriter, funcOp, ftdOps)))
  //   return failure();

  // Add supp GSA
  if (failed(addSuppGSA(rewriter, funcOp, ftdOps)))
    return failure();

  // id basic block
  idBasicBlocks(funcOp, rewriter);

  if (failed(flattenAndTerminate(funcOp, rewriter, argReplacements)))
    return failure();

  return success();
}

static void mapConditionsToValues(Region &region, FtdStoredOperations &ftdOps) {
  for (Block &block : region.getBlocks()) {
    Operation *terminator = block.getTerminator();
    if (terminator) {
      if (isa<cf::CondBranchOp>(terminator)) {
        auto condBranch = dyn_cast<cf::CondBranchOp>(terminator);

        ftdOps.conditionToValue[getBlockCondition(&block)] =
            condBranch.getCondition();
      }
    }
  }
}

/// Given an operation, returns true if the operation is a conditional branch
/// which terminates a for loop (it might be that the conditional branch is
/// just at the end of an if/forward branch)
static bool isBrachExitLoop(Operation *op, CFGLoopInfo &li) {
  if (isa<handshake::ConditionalBranchOp>(op)) {
    if (CFGLoop *loop = li.getLoopFor(op->getBlock()); loop) {
      llvm::SmallVector<Block *> exitBlocks;
      loop->getExitingBlocks(exitBlocks);
      return llvm::find(exitBlocks, op->getBlock()) != exitBlocks.end();
    }
  }
  return false;
}

/// Given an operation, return true if the two operands of a merge come from
/// two different loops. When this happens, the merge is connecting two loops
bool isaMergeLoop(Operation *merge, CFGLoopInfo &li) {
  return li.getLoopFor(merge->getOperand(0).getParentBlock()) !=
         li.getLoopFor(merge->getOperand(1).getParentBlock());
}

/// Get a boolean expression representing the exit condition of the current
/// loop block
BoolExpression *getBlockLoopExitCondition(Block *loopExit, CFGLoop *loop,
                                          CFGLoopInfo &li) {
  BoolExpression *blockCond =
      BoolExpression::parseSop(getBlockCondition(loopExit));
  auto *terminatorOperation = loopExit->getTerminator();
  assert(isa<cf::CondBranchOp>(terminatorOperation) &&
         "Terminator condition of a loop exit must be a conditional branch.");
  auto condBranch = dyn_cast<cf::CondBranchOp>(terminatorOperation);

  // If the destination of the false outcome is not the block, then the
  // condition must be negated
  if (li.getLoopFor(condBranch.getFalseDest()) != loop)
    blockCond->boolNegate();

  return blockCond;
}

/// Get a value out of the input boolean expression
static Value boolVariableToCircuit(ConversionPatternRewriter &rewriter,
                                   experimental::boolean::BoolExpression *expr,
                                   Block *block, FtdStoredOperations &ftdOps) {
  SingleCond *singleCond = static_cast<SingleCond *>(expr);
  auto condition = ftdOps.conditionToValue[singleCond->id];
  if (singleCond->isNegated) {
    rewriter.setInsertionPointToStart(block);
    auto notOp = rewriter.create<handshake::NotOp>(
        block->getOperations().front().getLoc(),
        channelifyType(condition.getType()), condition);

    return notOp->getResult(0);
  }
  return condition;
}

/// Forwarding declaration of `dataToCircuit`
static Value dataToCircuit(ConversionPatternRewriter &rewriter,
                           MultiplexerIn *data, Block *block,
                           FtdStoredOperations &ftdOps);

/// Given an instance of a multiplexer, convert it to a circuit by using the
/// Shannon expansion
static Value muxToCircuit(ConversionPatternRewriter &rewriter, Multiplexer *mux,
                          Block *block, FtdStoredOperations &ftdOps) {

  rewriter.setInsertionPointToStart(block);

  // Get the two operands by recursively calling `dataToCircuit` (it possibly
  // creates other muxes in a hierarchical way)
  SmallVector<Value, 4> muxOperands;
  muxOperands.push_back(dataToCircuit(rewriter, mux->in0, block, ftdOps));
  muxOperands.push_back(dataToCircuit(rewriter, mux->in1, block, ftdOps));
  Value muxCond = dataToCircuit(rewriter, mux->cond, block, ftdOps);

  // Create the multiplxer and add it to the rest of the circuit
  auto muxOp = rewriter.create<handshake::MuxOp>(
      block->getOperations().front().getLoc(), muxCond, muxOperands);
  ftdOps.shannonMUXes.insert(muxOp);
  ftdOps.allocationNetwork.insert(muxOp);
  return muxOp.getResult();
}

/// Get a circuit out a boolean expression, depending on the different kinds
/// of expressions you might have
static Value boolExpressionToCircuit(ConversionPatternRewriter &rewriter,
                                     BoolExpression *expr, Block *block,
                                     FtdStoredOperations &ftdOps) {

  // Variable case
  if (expr->type == ExpressionType::Variable)
    return boolVariableToCircuit(rewriter, expr, block, ftdOps);

  // Constant case (either 0 or 1)
  rewriter.setInsertionPointToStart(block);
  auto sourceOp = rewriter.create<handshake::SourceOp>(
      block->getOperations().front().getLoc());
  Value cnstTrigger = sourceOp.getResult();

  auto cstType = rewriter.getI1Type();
  auto cstAttr =
      IntegerAttr::get(cstType, (expr->type == ExpressionType::One ? 1 : 0));

  auto constOp = rewriter.create<handshake::ConstantOp>(
      block->getOperations().front().getLoc(), cstAttr.getType(), cstAttr,
      cnstTrigger);
  ftdOps.networkConstants.insert(constOp);

  return constOp.getResult();
}

/// Convert a `MultiplexerIn` object as obtained from Shannon expansion to a
/// circuit
static Value dataToCircuit(ConversionPatternRewriter &rewriter,
                           MultiplexerIn *data, Block *block,
                           FtdStoredOperations &ftdOps) {
  if (data->boolexpression.has_value())
    return boolExpressionToCircuit(rewriter, data->boolexpression.value(),
                                   block, ftdOps);

  return muxToCircuit(rewriter, data->mux, block, ftdOps);
}

/// Insert a branch to the corrent position, taking into account whether it
/// should work to suppress the over-production of tokens or self-regeneration
static Value insertBranchToLoop(ConversionPatternRewriter &rewriter,
                                CFGLoop *loop, Operation *consumer,
                                Value connection, bool moreProdThanCons,
                                bool selfRegeneration,
                                FtdStoredOperations &ftdOps, CFGLoopInfo &li) {

  // Case in which there is only one termination block
  if (Block *loopExit = loop->getExitingBlock(); loopExit) {

    // Get the termination operation, which is supposed to be conditional
    // branch.
    Operation *loopTerminator = loopExit->getTerminator();
    assert(isa<cf::CondBranchOp>(loopTerminator) &&
           "Terminator condition of a loop exit must be a conditional "
           "branch.");

    // A conditional branch is now to be added next to the loop terminator, so
    // that the token can be suppressed
    rewriter.setInsertionPointToStart(loopExit);
    auto conditionValue = boolVariableToCircuit(
        rewriter, getBlockLoopExitCondition(loopExit, loop, li), loopExit,
        ftdOps);

    // Since only one output is used, the other one will be optimized to a
    // sink, as we expect from a suppress branch
    auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        loopExit->getOperations().back().getLoc(), conditionValue, connection);

    // Keep track of the new component in the circuit
    ftdOps.allocationNetwork.insert(branchOp);
    if (moreProdThanCons)
      ftdOps.suppBranches.insert(branchOp);
    if (selfRegeneration)
      ftdOps.selfGenBranches.insert(branchOp);

    // Modify the usage of the previous value
    Value newConnection =
        moreProdThanCons ? branchOp.getTrueResult() : branchOp.getFalseResult();
    consumer->replaceUsesOfWith(connection, newConnection);

    // Return the output of the new component related to the regeneration
    return newConnection;
  }

  // Case in which there are more than one terminal blocks
  std::vector<std::string> cofactorList;
  SmallVector<Block *> exitBlocks;
  loop->getExitingBlocks(exitBlocks);
  Block *loopExit = exitBlocks.front();

  BoolExpression *fLoopExit = BoolExpression::boolZero();

  // Get the list of all the cofactors related to possible exit conditions
  for (Block *exitBlock : exitBlocks) {
    BoolExpression *blockCond = getBlockLoopExitCondition(exitBlock, loop, li);
    fLoopExit = BoolExpression::boolOr(fLoopExit, blockCond);
    cofactorList.push_back(getBlockCondition(exitBlock));
  }

  // Sort the cofactors alphabetically
  std::sort(cofactorList.begin(), cofactorList.end());

  // Apply shannon expansion to the loop exit expression and the list of
  // cofactors
  MultiplexerIn *shannonResult = applyShannon(fLoopExit, cofactorList);

  // Convert the boolean expression obtained through Shannon to a circuit
  Value branchCond = dataToCircuit(rewriter, shannonResult, loopExit, ftdOps);

  Operation *loopTerminator = loopExit->getTerminator();
  assert(isa<cf::CondBranchOp>(loopTerminator) &&
         "Terminator condition of a loop exit must be a conditional "
         "branch.");

  rewriter.setInsertionPointToStart(loopExit);
  auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
      loopExit->getOperations().front().getLoc(), branchCond, connection);
  ftdOps.allocationNetwork.insert(branchOp);

  if (moreProdThanCons)
    ftdOps.suppBranches.insert(branchOp);
  if (selfRegeneration)
    ftdOps.selfGenBranches.insert(branchOp);

  Value newConnection = branchOp.getTrueResult();
  if (!moreProdThanCons)
    newConnection = branchOp.getFalseResult();
  consumer->replaceUsesOfWith(connection, newConnection);
  return newConnection;
}

/// For each loop which contains the producer and does not contains a
/// consumer, we need to insert a branch to possibly suppress its value
static void manageMoreProdThanCons(ConversionPatternRewriter &rewriter,
                                   Block *producerBlock, Operation *consumer,
                                   Value connection, CFGLoopInfo &li,
                                   FtdStoredOperations &ftdOps) {

  Value con = connection;
  for (CFGLoop *loop = li.getLoopFor(producerBlock); loop;
       loop = loop->getParentLoop()) {

    // For each loop containing the producer but not the consumer, add the
    // branch
    if (!loop->contains(consumer->getBlock()))
      con = insertBranchToLoop(rewriter, loop, consumer, con, true, false,
                               ftdOps, li);
  }
}

/// Self regeneration is necessary in case a value is inside of a loop, so
/// that the value should be used more than one time. This function is in
/// charge of implementing this behaviour
static void manageSelfRegeneration(ConversionPatternRewriter &rewriter,
                                   Operation *consumer, Value connection,
                                   CFGLoopInfo &li,
                                   FtdStoredOperations &ftdOps) {

  if (CFGLoop *loop = li.getLoopFor(consumer->getBlock()); loop)
    insertBranchToLoop(rewriter, loop, consumer, connection, false, true,
                       ftdOps, li);
}

/// This kind of regeneration is for loop values which are not
/// self-regenerations (mainly backward edges in a CFG)
static void manageDifferentRegeneration(ConversionPatternRewriter &rewriter,
                                        Operation *consumer, Value connection,
                                        CFGLoopInfo &li,
                                        FtdStoredOperations &ftdOps) {
  // Regenerate only if a loop is involved
  if (li.getLoopFor(consumer->getBlock())) {
    // Get all the possible exist blocks
    SmallVector<Block *> exits;
    li.getLoopFor(consumer->getBlock())->getExitBlocks(exits);
    insertBranchToLoop(rewriter, li.getLoopFor(consumer->getBlock()), consumer,
                       connection, false, false, ftdOps, li);
  }
}

/// Apply the algorithm from FPGA'22 to handle a non-loop situation of
/// producer and consumer
static LogicalResult manageNonLoop(ConversionPatternRewriter &rewriter,
                                   handshake::FuncOp &funcOp,
                                   Block *producerBlock, Operation *consumer,
                                   Value connection,
                                   FtdStoredOperations &ftdOps) {
  Block *entryBlock = &funcOp.getBody().front();
  ControlDependenceAnalysis<dynamatic::handshake::FuncOp> cdgAnalysis(funcOp);

  // Get the control dependencies from the producer
  DenseSet<Block *> prodControlDeps;
  if (failed(cdgAnalysis.getBlockForwardControlDeps(producerBlock,
                                                    prodControlDeps)))
    return failure();

  // Get the control dependencies from the consumer
  DenseSet<Block *> consControlDeps;
  if (failed(cdgAnalysis.getBlockForwardControlDeps(consumer->getBlock(),
                                                    consControlDeps)))
    return failure();

  // Get rid of common entries in the two sets
  eliminateCommonEntries(prodControlDeps, consControlDeps);

  // Compute the activation function of producer and consumer
  BoolExpression *fProd =
      enumeratePaths(entryBlock, producerBlock, prodControlDeps);
  BoolExpression *fCons =
      enumeratePaths(entryBlock, consumer->getBlock(), consControlDeps);

  /// f_supp = f_prod and not f_cons
  BoolExpression *fSup = BoolExpression::boolAnd(fProd, fCons->boolNegate());
  fSup = fSup->boolMinimize();

  // If the activation function is not zero, then a suppress block is to be
  // inserted
  if (fSup->type != experimental::boolean::ExpressionType::Zero) {
    std::set<std::string> blocks = fSup->getVariables();

    std::vector<std::string> cofactorList(blocks.begin(), blocks.end());
    MultiplexerIn *shannonResult = applyShannon(fSup, cofactorList);
    Value branchCond =
        dataToCircuit(rewriter, shannonResult, consumer->getBlock(), ftdOps);

    rewriter.setInsertionPointToStart(consumer->getBlock());
    auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        consumer->getLoc(), branchCond, connection);
    ftdOps.allocationNetwork.insert(branchOp);
    consumer->replaceUsesOfWith(connection, branchOp.getFalseResult());
  }

  return success();
}

LogicalResult
FtdLowerFuncToHandshake::addSuppForStart(ConversionPatternRewriter &rewriter,
                                         handshake::FuncOp &funcOp,
                                         FtdStoredOperations &ftdOps) const {

  Region &region = funcOp.getBody();
  auto startValue = (Value)funcOp.getArguments().back();
  for (Block &consumerBlock : region.getBlocks()) {
    for (Operation &consumerOp : consumerBlock.getOperations()) {

      // At this point, we are only interested in applying the algorithm to
      // constants which were not inserted by the FTD algorithm
      if (!ftdOps.allocationNetwork.contains(&consumerOp) &&
          (!isa<handshake::ConstantOp>(consumerOp) ||
           ftdOps.networkConstants.contains(&consumerOp)))
        continue;

      // Skip if the consumer is a shannon's mux
      if (ftdOps.shannonMUXes.contains(&consumerOp))
        continue;

      // Skip if the consumer is a conditional branch
      if (isa<handshake::ConditionalBranchOp>(consumerOp))
        continue;

      // For all the operands of the consumer, take into account only the start
      // value if exists
      for (Value operand : consumerOp.getOperands()) {
        if (operand == startValue) {
          if (operand.getParentBlock() == &consumerBlock &&
              !isa<handshake::MergeOp>(consumerOp))
            continue;

          if (failed(manageNonLoop(rewriter, funcOp, operand.getParentBlock(),
                                   &consumerOp, operand, ftdOps)))
            return failure();
        }
      }
    }
  }
  return success();
}

LogicalResult FtdLowerFuncToHandshake::addSuppBranches(
    ConversionPatternRewriter &rewriter, handshake::FuncOp &funcOp,
    FtdStoredOperations &ftdOps, DenseSet<Operation *> &oldBranches) const {

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  DenseSet<Operation *> suppBranchesCopy = ftdOps.suppBranches;

  // For each branch inserted through FTD which was not already analyzed (not in
  // `oldBranches`)
  for (Operation *producerOp : ftdOps.suppBranches) {

    if (oldBranches.contains(producerOp))
      continue;

    Block *producerBlock = producerOp->getBlock();

    // Consider individually each result and each possible consumer
    for (Value res : producerOp->getResults()) {

      for (Operation *consumerOp : res.getUsers()) {

        Block *consumerBlock = consumerOp->getBlock();

        // Apply the FTD insertion algorithm on each pair

        if (!ftdOps.allocationNetwork.contains(producerOp) &&
            (!isa<handshake::ConstantOp>(producerOp) ||
             ftdOps.networkConstants.contains(producerOp)))
          continue;

        // Skip if the consumer and the producer are in the same block and
        // the consumer is not a merge
        if (consumerBlock == producerBlock &&
            !isa<handshake::MergeOp>(consumerOp))
          continue;

        // Skip if the current consumer is a MUX added by Shannon
        if (ftdOps.shannonMUXes.contains(consumerOp))
          continue;

        // Skip if the current consumer is a conditional branch
        if (isa<handshake::ConditionalBranchOp>(consumerOp))
          continue;

        // In case the innermost loop containing the producer does not contains
        // the consumer, then we are in a situation of more producer than
        // consumers
        bool moreProdThanCons = false;
        if (loopInfo.getLoopFor(producerBlock))
          moreProdThanCons =
              !loopInfo.getLoopFor(producerBlock)->contains(consumerBlock);

        // Scenario of more producer than consumers
        if (moreProdThanCons && !isBranchInLoopExit(producerOp, loopInfo))
          manageMoreProdThanCons(rewriter, producerBlock, consumerOp, res,
                                 loopInfo, ftdOps);

        // Scenario of backward edge
        else if (greaterThanBlocks(producerBlock, consumerBlock) ||
                 (isa<handshake::MergeOp>(consumerOp) &&
                  producerBlock == consumerBlock &&
                  isaMergeLoop(consumerOp, loopInfo)))
          manageDifferentRegeneration(rewriter, consumerOp, res, loopInfo,
                                      ftdOps);
        else {
          if (failed(manageNonLoop(rewriter, funcOp, producerBlock, consumerOp,
                                   res, ftdOps)))
            return failure();
        }
      }
    }
  }

  // Update the value of `oldBranches` so that it contains the set of branches
  // as it was before the current iteration of the algorithm
  oldBranches = suppBranchesCopy;

  return success();
}

LogicalResult
FtdLowerFuncToHandshake::addSupp(ConversionPatternRewriter &rewriter,
                                 handshake::FuncOp &funcOp,
                                 FtdStoredOperations &ftdOps) const {
  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));

  // For each block in the function, obtain a value representing its condition
  mapConditionsToValues(region, ftdOps);

  for (Block &producerBlock : region.getBlocks()) {
    // For each producer
    for (Operation &producerOp : producerBlock.getOperations()) {

      // At this point, we are only interested in applying the algorithm to
      // constants which were not inserted by the FTD algorithm
      if (!ftdOps.allocationNetwork.contains(&producerOp) &&
          (!isa<handshake::ConstantOp>(producerOp) ||
           ftdOps.networkConstants.contains(&producerOp)))
        continue;

      // Skip if: the producer comes from the generation mechanism; the
      // producer comes from the suppression mechanism; the producer
      // comes from the self generation mechanism
      if (ftdOps.shannonMUXes.contains(&producerOp) ||
          ftdOps.suppBranches.contains(&producerOp) ||
          ftdOps.selfGenBranches.contains(&producerOp))
        continue;

      // For each result coming out of the producer
      for (Value result : producerOp.getResults()) {

        // Consider all the users of that result
        for (Operation *consumerOp : result.getUsers()) {
          Block *consumerBlock = consumerOp->getBlock();

          // At this point, we are only interested in applying the algorithm
          // to constants which were not inserted by the FTD algorithm
          if (!ftdOps.allocationNetwork.contains(&producerOp) &&
              (!isa<handshake::ConstantOp>(producerOp) ||
               ftdOps.networkConstants.contains(&producerOp)))
            continue;

          // Skip if the consumer and the producer are in the same block and
          // the consumer is not a merge
          if (consumerBlock == &producerBlock &&
              !isa<handshake::MergeOp>(consumerOp))
            continue;

          // Skip if the current consumer is a MUX added by Shannon
          if (ftdOps.shannonMUXes.contains(consumerOp))
            continue;

          // Skip if the current consumer is a conditional branch
          if (isa<handshake::ConditionalBranchOp>(consumerOp))
            continue;

          // Different scenarios about the relationship between consumer and
          // producer should be handled:
          // 1. If the producer is in a loop and the consumer is not in that
          // same loop, then the token produced needs to be suppressed as long
          // as the loop is executed
          // 2. If the consumer uses its own result, then we need to handle a
          // self-regeneration;
          // 3. If the consumer and the producer are in different basic blocks
          // OR the consumer is a merge in the same block, then we need to
          // handle a loop case without self-regeneration;
          // 4. Else, the components are in the same basic block and we add
          // a normal suppression-generation mechanism.

          bool producingGtUsing =
              loopInfo.getLoopFor(&producerBlock) &&
              !loopInfo.getLoopFor(&producerBlock)->contains(consumerBlock);

          // We need to suppress all the tokens produced within a loop and
          // used outside each time a new iteration starts, excluding those
          // related to the final branches, since they are fundamental to
          // determine the loop execution
          if (producingGtUsing && !isBrachExitLoop(&producerOp, loopInfo)) {
            manageMoreProdThanCons(rewriter, &producerBlock, consumerOp, result,
                                   loopInfo, ftdOps);
          } else {

            // Self regeneration case: the token is reused by the operation
            // itself
            bool selfRegeneration =
                llvm::any_of(consumerOp->getResults(),
                             [&result](const Value &v) { return v == result; });
            if (selfRegeneration) {
              manageSelfRegeneration(rewriter, consumerOp, result, loopInfo,
                                     ftdOps);
            }

            // Regeneration in loop case
            else if (greaterThanBlocks(&producerBlock, consumerBlock) ||
                     (isa<handshake::MergeOp>(consumerOp) &&
                      &producerBlock == consumerBlock &&
                      isaMergeLoop(consumerOp, loopInfo) &&
                      !isa<handshake::ConditionalBranchOp>(producerOp))) {
              manageDifferentRegeneration(rewriter, consumerOp, result,
                                          loopInfo, ftdOps);
            } else {
              if (failed(manageNonLoop(rewriter, funcOp, &producerBlock,
                                       consumerOp, result, ftdOps)))
                return failure();
            }
          }
        }
      }
    }
  }
  return success();
}

/// Get all the loops the consumer is in while the producer is not, from the
/// outermost to the innermost
static SmallVector<CFGLoop *> getLoopsConsNotInProd(Block *cons, Block *prod,
                                                    mlir::CFGLoopInfo &li) {

  SmallVector<CFGLoop *> result;

  // Get all the loops in which the consumer is but the producer is not
  for (CFGLoop *loop = li.getLoopFor(cons); loop;
       loop = loop->getParentLoop()) {
    if (!loop->contains(prod))
      result.push_back(loop);
  }

  // Reverse to the get the loops from outermost to innermost
  std::reverse(result.begin(), result.end());
  return result;
}

LogicalResult
FtdLowerFuncToHandshake::addPhi(ConversionPatternRewriter &rewriter,
                                handshake::FuncOp &funcOp,
                                FtdStoredOperations &ftdOps) const {

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));

  for (Block &consumerBlock : region.getBlocks()) {
    for (Operation &consumerOp : consumerBlock.getOperations()) {

      // Skip if the operation was not added by the FTD algorithm AND either
      // it is not a constant or it is a constant related to the network of
      // constants (for MUX select signals)
      if (!ftdOps.allocationNetwork.contains(&consumerOp) &&
          (!isa<handshake::ConstantOp>(consumerOp) ||
           ftdOps.networkConstants.contains(&consumerOp)))
        continue;

      // Skip consumers which were were added by the `addPhi` function, to
      // avoid inifinte loops
      if (ftdOps.phiMerges.contains(&consumerOp))
        continue;

      for (Value operand : consumerOp.getOperands()) {

        // Skip the analysis of the current operand if it is produced by a
        // MERGE produced by `addPhi`, to avoid infinite loops
        if (mlir::Operation *producerOp = operand.getDefiningOp();
            ftdOps.phiMerges.contains(producerOp))
          continue;

        // Get the parent block of the operand
        Block *producerBlock = operand.getParentBlock();
        Value producerOperand = operand;

        // Get all the loops for which we need to regenerate the corresponding
        // value
        SmallVector<CFGLoop *> loops =
            getLoopsConsNotInProd(&consumerBlock, producerBlock, loopInfo);

        // For each of the loop, from the outermost to the innermost
        for (auto *it = loops.begin(); it != loops.end(); ++it) {

          // If we are in the innermost loop (thus the iterator is at its end)
          // and the consumer is a loop merge, stop
          if (std::next(it) == loops.end() &&
              ftdOps.memDepLoopMerges.contains(&consumerOp))
            break;

          // Add the merge to the network, by substituting the operand with
          // the output of the merge, and forwarding the output of the merge
          // to its inputs.
          rewriter.setInsertionPointToStart((*it)->getHeader());
          auto mergeOp = rewriter.create<handshake::MergeOp>(
              producerOperand.getLoc(), producerOperand);
          producerOperand = mergeOp.getResult();
          ftdOps.phiMerges.insert(mergeOp);
          ftdOps.allocationNetwork.insert(mergeOp);
          mergeOp->insertOperands(1, mergeOp->getResult(0));
        }
        consumerOp.replaceUsesOfWith(operand, producerOperand);
      }
    }
  }
  return success();
}

LogicalResult FtdLowerFuncToHandshake::convertUndefinedValues(
    ConversionPatternRewriter &rewriter, handshake::FuncOp &funcOp) const {

  // Get all the undefined value in the function
  auto undefinedValues = funcOp.getBody().getOps<LLVM::UndefOp>();

  // Get the start value of the current function
  auto startValue = (Value)funcOp.getArguments().back();

  // For each constant
  for (auto undefOp : llvm::make_early_inc_range(undefinedValues)) {
    // Create an attribute of the appropriate type for the constant
    auto resType = undefOp.getRes().getType();
    TypedAttr cstAttr;
    if (isa<IndexType>(resType)) {
      auto intType = rewriter.getIntegerType(32);
      cstAttr = rewriter.getIntegerAttr(intType, 0);
    } else if (isa<IntegerType>(resType)) {
      cstAttr = rewriter.getIntegerAttr(resType, 0);
    } else if (FloatType floatType = dyn_cast<FloatType>(resType)) {
      cstAttr = rewriter.getFloatAttr(floatType, 0.0);
    } else {
      return undefOp->emitError() << "operation has unsupported result type";
    }

    // Create a constant with a default value and replace the undefined value
    rewriter.setInsertionPoint(undefOp);
    auto cstOp = rewriter.create<handshake::ConstantOp>(undefOp.getLoc(),
                                                        cstAttr, startValue);
    // Move attributes and replace the usage of the value
    cstOp->setDialectAttrs(undefOp->getAttrDictionary());
    namer.replaceOp(cstOp, cstOp);
    rewriter.replaceOp(undefOp, cstOp.getResult());
  }

  return success();
}

LogicalResult
FtdLowerFuncToHandshake::convertConstants(ConversionPatternRewriter &rewriter,
                                          handshake::FuncOp &funcOp) const {

  // Get the start value of the current function
  auto startValue = (Value)funcOp.getArguments().back();

  // Get all the constants in the function
  auto constants = funcOp.getBody().getOps<mlir::arith::ConstantOp>();

  // For each constant
  for (auto cstOp : llvm::make_early_inc_range(constants)) {

    // Set the insertion point after the specified operation
    rewriter.setInsertionPoint(cstOp);

    // Get the value of the constant
    TypedAttr cstAttr = cstOp.getValue();

    // Convert IndexType'd values to equivalent signless integers
    if (isa<IndexType>(cstAttr.getType())) {
      auto intType = rewriter.getIntegerType(32);
      cstAttr = IntegerAttr::get(
          intType, cast<IntegerAttr>(cstAttr).getValue().trunc(32));
    }

    // Create the new constant with the same value and attributes, trigger
    // each of them using the start signal of the function
    auto newCstOp = rewriter.create<handshake::ConstantOp>(cstOp.getLoc(),
                                                           cstAttr, startValue);

    // Move the attributes to the new constant
    newCstOp->setDialectAttrs(cstOp->getDialectAttrs());

    // Replace the constant and the usage of its result
    namer.replaceOp(cstOp, newCstOp);
    cstOp.getResult().replaceAllUsesWith(newCstOp.getResult());
    rewriter.replaceOp(cstOp, newCstOp->getResults());
  }
  return success();
}

LogicalResult FtdLowerFuncToHandshake::ftdVerifyAndCreateMemInterfaces(
    handshake::FuncOp &funcOp, ConversionPatternRewriter &rewriter,
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

    // Determine LSQ group validity and add ports the interface builder
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

      // As we instantiate the interfaces for the LSQ for each memory
      // operation, we need to add some forks in order for the control input
      // to be propagated. In particular, we want to keep track of the control
      // value associated to each basic block in the region
      DenseMap<Block *, Operation *> forksGraph;

      if (failed(memBuilder.instantiateInterfacesWithForks(
              rewriter, mcOp, lsqOp, groupsGraph, forksGraph, startValue,
              ftdOps.allocationNetwork)))
        return failure();

      if (failed(addMergeNonLoop(funcOp, rewriter, allMemDeps, groupsGraph,
                                 forksGraph, ftdOps, startValue)))
        return failure();

      if (failed(addMergeLoop(funcOp, rewriter, allMemDeps, groupsGraph,
                              forksGraph, ftdOps, startValue)))
        return failure();

      if (failed(joinInsertion(rewriter, groupsGraph, forksGraph,
                               ftdOps.allocationNetwork)))
        return failure();
    } else {
      handshake::MemoryControllerOp mcOp;
      handshake::LSQOp lsqOp;
      if (failed(memBuilder.instantiateInterfaces(rewriter, mcOp, lsqOp)))
        return failure();
    }
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
