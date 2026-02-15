//===- FtdImplementation.cpp --- Main FTD Algorithm -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the core functions to run the Fast Token Delivery algorithm,
// according to the original FPGA'22 paper by Elakhras et al.
// (https://ieeexplore.ieee.org/document/10035134).
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/FtdImplementation.h"
#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Support/Backedge.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/FtdCycleAnalysis.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::boolean;

/// Annotation to use in the IR when an operation needs to be skipped by the FTD
/// algorithm.
constexpr llvm::StringLiteral FTD_OP_TO_SKIP("ftd.skip");
/// Annotation to to identify muxes inserted with the `addGsaGates`
/// functionalities.
constexpr llvm::StringLiteral FTD_EXPLICIT_MU("ftd.MU");
constexpr llvm::StringLiteral FTD_EXPLICIT_GAMMA("ftd.GAMMA");
/// Temporary annotation to be used with merges created with the
/// `createPhiNetwork` functionality, which will then be converted into muxes.
constexpr llvm::StringLiteral NEW_PHI("nphi");
/// Annotation to use for initial merges and initial false constants.
constexpr llvm::StringLiteral FTD_INIT_MERGE("ftd.imerge");
/// Annotation to use for regeneration multiplexers.
constexpr llvm::StringLiteral FTD_REGEN("ftd.regen");

/// Identify the block that has muxCondition as its terminator condition
/// Note that it is not necessarily the same block defining the muxCondition
static Block *returnMuxConditionBlock(Value muxCondition) {
  Block *muxConditionBlock = nullptr;

  for (auto &use : muxCondition.getUses()) {
    Operation *userOp = use.getOwner();
    Block *userBlock = userOp->getBlock();

    if (isa_and_nonnull<cf::CondBranchOp>(userOp)) {
      muxConditionBlock = userBlock;
      break;
    }
  }
  if (!muxConditionBlock)
    muxConditionBlock = muxCondition.getParentBlock();
  return muxConditionBlock;
}

/// Given a block, get its immediate dominator if exists
static Block *getImmediateDominator(Region &region, Block *bb) {

  // Avoid a situation with no blocks in the region
  if (region.getBlocks().empty())
    return nullptr;

  // The first block in the CFG has both non predecessors and no dominators
  if (bb->hasNoPredecessors())
    return nullptr;

  DominanceInfo domInfo;
  llvm::DominatorTreeBase<Block, false> &domTree = domInfo.getDomTree(&region);
  return domTree.getNode(bb)->getIDom()->getBlock();
}

/// Get the dominance frontier of each block in the region
static DenseMap<Block *, DenseSet<Block *>>
getDominanceFrontier(Region &region) {

  // This algorithm comes from the following paper:
  // Cooper, Keith D., Timothy J. Harvey and Ken Kennedy. “A Simple, Fast
  // Dominance Algorithm.” (1999).

  DenseMap<Block *, DenseSet<Block *>> result;

  // Create an empty set of reach available block
  for (Block &bb : region.getBlocks())
    result.insert({&bb, DenseSet<Block *>()});

  for (Block &bb : region.getBlocks()) {

    // Get the predecessors of the block
    auto predecessors = bb.getPredecessors();

    // Count the number of predecessors
    int numberOfPredecessors = 0;
    for (auto *pred : predecessors)
      if (pred)
        numberOfPredecessors++;

    // Skip if the node has none or only one predecessors
    if (numberOfPredecessors < 2)
      continue;

    // Run the algorithm as explained in the paper
    for (auto *pred : predecessors) {
      Block *runner = pred;
      // Runner performs a bottom up traversal of the dominator tree
      while (runner != getImmediateDominator(region, &bb)) {
        result[runner].insert(&bb);
        runner = getImmediateDominator(region, runner);
      }
    }
  }

  return result;
}

/// Get a list of all the loops in which the consumer is but the producer is
/// not, starting from the innermost.
static SmallVector<CFGLoop *> getLoopsConsNotInProd(Block *cons, Block *prod,
                                                    mlir::CFGLoopInfo &li) {
  SmallVector<CFGLoop *> result;

  // Get all the loops in which the consumer is but the producer is
  // not, starting from the innermost
  for (CFGLoop *loop = li.getLoopFor(cons); loop;
       loop = loop->getParentLoop()) {
    if (!loop->contains(prod))
      result.push_back(loop);
  }

  // Reverse to the get the loops from outermost to innermost
  std::reverse(result.begin(), result.end());
  return result;
};

/// A lightweight DFS to check if 'end' is reachable from 'start'.
static bool isReachable(Block *start, Block *end) {
  if (start == end)
    return true;

  DenseSet<Block *> visited;
  SmallVector<Block *, 8> stack;
  stack.push_back(start);
  visited.insert(start);

  while (!stack.empty()) {
    Block *curr = stack.pop_back_val();

    if (curr == end)
      return true;

    for (Block *succ : curr->getSuccessors()) {
      if (!visited.count(succ)) {
        visited.insert(succ);
        stack.push_back(succ);
      }
    }
  }
  return false;
}

/// Helper function to generate expression combining Local Logic (True/False)
/// with Original Variables (c2, c3...).
/// Note: Input is Local Deps
static boolean::BoolExpression *
getHybridPathExpression(const std::vector<Block *> &localPath,
                        const ftd::LocalCFG &lcfg, const ftd::BlockIndexing &bi,
                        const DenseSet<Block *> &localDeps) {

  // Start with 1
  auto *exp = boolean::BoolExpression::boolOne();

  unsigned pathSize = localPath.size();
  for (unsigned i = 0; i < pathSize - 1; i++) {
    // Local Block
    Block *u = localPath[i];
    // Local Block (Target)
    Block *v = localPath[i + 1];

    // 1. Check Dependency using LOCAL sets
    if (localDeps.contains(u)) {
      Operation *term = u->getTerminator();

      if (isa<cf::CondBranchOp>(term)) {
        // 2. Map to Original Block to get the Variable Index/Name
        Block *origU = lcfg.origMap.lookup(u);

        // Special handling for second visit if needed, though usually origMap
        // covers it
        if (!origU && u == lcfg.secondVisitBB) {
          origU = lcfg.origMap.lookup(lcfg.newProd);
        }

        if (origU) {
          // 3. Get Name from ORIGINAL CFG (e.g., "c2")
          auto blockIndexOptional = bi.getIndexFromBlock(origU);
          if (blockIndexOptional.has_value()) {
            std::string blockCondition = bi.getBlockCondition(origU);
            boolean::BoolExpression *cond =
                boolean::BoolExpression::parseSop(blockCondition);

            // 4. Get Polarity from LOCAL CFG (Structure)
            // Does the path go to 'v' via the False branch in the Decision
            // Graph?
            auto condOp = dyn_cast<cf::CondBranchOp>(term);
            if (condOp.getFalseDest() == v) {
              // Add '~' if Local Graph says False
              cond->boolNegate();
            }

            // Combine
            exp = boolean::BoolExpression::boolAnd(exp, cond);
          }
        }
      }
    }
  }
  return exp;
}

static BoolExpression *enumeratePaths(const ftd::LocalCFG &lcfg,
                                      const ftd::BlockIndexing &bi,
                                      const DenseSet<Block *> &controlDeps) {

  // 1. Path Finding using Iterative DFS (on Local CFG)
  std::vector<std::vector<Block *>> allPaths;

  struct StackFrame {
    Block *u;
    unsigned currIdx;
    unsigned numSuccs;
  };

  std::vector<StackFrame> dfsStack;
  std::vector<Block *> currentLocalPath;

  if (lcfg.newProd && lcfg.newCons) {
    Block *root = lcfg.newProd;
    auto *term = root->getTerminator();
    unsigned n = term ? term->getNumSuccessors() : 0;

    dfsStack.push_back({root, 0, n});
    currentLocalPath.push_back(root);
  } else {
    return BoolExpression::boolZero();
  }

  while (!dfsStack.empty()) {
    StackFrame &frame = dfsStack.back();

    // --- Case A: Reached Consumer ---
    if (frame.u == lcfg.newCons) {
      // [CRITICAL] Store the LOCAL path exactly as traversed.
      // Do NOT map to original blocks here.
      allPaths.push_back(currentLocalPath);

      currentLocalPath.pop_back();
      dfsStack.pop_back();
      continue;
    }

    // --- Case B: Traverse Successors ---
    if (frame.currIdx < frame.numSuccs) {
      auto *term = frame.u->getTerminator();
      Block *succ = term->getSuccessor(frame.currIdx);
      frame.currIdx++;

      bool isCycle = std::find(currentLocalPath.begin(), currentLocalPath.end(),
                               succ) != currentLocalPath.end();

      if (succ != lcfg.sinkBB && !isCycle) {
        auto *succTerm = succ->getTerminator();
        unsigned succN = succTerm ? succTerm->getNumSuccessors() : 0;

        dfsStack.push_back({succ, 0, succN});
        currentLocalPath.push_back(succ);
      }
    }
    // --- Case C: Backtrack ---
    else {
      currentLocalPath.pop_back();
      dfsStack.pop_back();
    }
  }

  if (allPaths.empty())
    return BoolExpression::boolZero();

  // 2. Expression Generation
  BoolExpression *sop = BoolExpression::boolZero();

  for (const std::vector<Block *> &path : allPaths) {
    // Use the hybrid helper to look up logic locally and names globally
    BoolExpression *minterm =
        getHybridPathExpression(path, lcfg, bi, controlDeps);

    sop = BoolExpression::boolOr(sop, minterm);
  }
  return sop->boolMinimizeSop();
}

/// Run the Cytron algorithm to determine, give a set of values, in which blocks
/// should we add a merge in order for those values to be merged
static DenseSet<Block *>
runCrytonAlgorithm(Region &funcRegion, DenseMap<Block *, Value> &inputBlocks) {
  // Get dominance frontier
  auto dominanceFrontier = getDominanceFrontier(funcRegion);

  // Temporary data structures to run the Cytron algorithm for phi positioning
  DenseMap<Block *, bool> work;
  DenseMap<Block *, bool> hasAlready;
  SmallVector<Block *> w;

  DenseSet<Block *> result;

  // Initialize data structures to run the Cytron algorithm
  for (auto &bb : funcRegion.getBlocks()) {
    work.insert({&bb, false});
    hasAlready.insert({&bb, false});
  }

  for (auto &[bb, val] : inputBlocks)
    w.push_back(bb), work[bb] = true;

  // Until there are no more elements in `w`
  while (w.size() != 0) {

    // Pop the top of `w`
    auto *x = w.back();
    w.pop_back();

    // Get the dominance frontier of `w`
    auto xFrontier = dominanceFrontier[x];

    // For each of its elements
    for (auto &y : xFrontier) {

      // Add the block in the dominance frontier to the list of blocks which
      // require a new phi. If it was not analyzed yet, also add it to `w`
      if (!hasAlready[y]) {
        result.insert(y);
        hasAlready[y] = true;
        if (!work[y])
          work[y] = true, w.push_back(y);
      }
    }
  }

  return result;
}

LogicalResult experimental::ftd::createPhiNetwork(
    Region &funcRegion, PatternRewriter &rewriter, SmallVector<Value> &vals,
    SmallVector<OpOperand *> &toSubstitue) {

  if (vals.empty()) {
    llvm::errs() << "Input of \"createPhiNetwork\" is empty";
    return failure();
  }

  auto *ctx = funcRegion.getContext();
  OpBuilder builder(ctx);

  mlir::DominanceInfo domInfo;
  // Type of the inputs
  Type valueType = vals[0].getType();
  // All the input values associated to one block
  DenseMap<Block *, SmallVector<Value>> valuesPerBlock;
  // Associate for each block the value that is dominated by all the others in
  // the same block
  DenseMap<Block *, Value> inputBlocks;
  // Backedge builder to insert new merges
  BackedgeBuilder edgeBuilder(builder, funcRegion.getLoc());
  // Backedge corresponding to each phi
  DenseMap<Block *, Backedge> resultPerPhi;
  // Operands of each merge
  DenseMap<Block *, SmallVector<Value>> operandsPerPhi;
  // Which value should be the input of each input value
  DenseMap<Block *, Value> inputPerBlock;

  // Check that all the values have the same type, then collet them according to
  // their input blocks
  for (auto &val : vals) {
    if (val.getType() != valueType) {
      llvm::errs() << "All values must have the same type\n";
      return failure();
    }
    auto *bb = val.getParentBlock();
    valuesPerBlock[bb].push_back(val);
  }

  // Sort the vectors of values in each block according to their dominance and
  // get only the last input value for each block. This is necessary in case in
  // the input sets there is more than one value per blocks
  for (auto &[bb, vals] : valuesPerBlock) {
    std::sort(vals.begin(), vals.end(), [&](Value a, Value b) -> bool {
      if (!a.getDefiningOp())
        return false;
      if (!b.getDefiningOp())
        return true;
      return domInfo.dominates(b.getDefiningOp(), a.getDefiningOp());
    });
    inputBlocks.insert({bb, vals[0]});
  }

  // In which block a new phi is necessary
  DenseSet<Block *> blocksToAddPhi =
      runCrytonAlgorithm(funcRegion, inputBlocks);

  // A backedge is created for each block in `blocksToAddPhi`, and it will
  // contain the value used as placeholder for the phi
  for (auto &bb : blocksToAddPhi) {
    Backedge mergeResult = edgeBuilder.get(valueType, bb->front().getLoc());
    operandsPerPhi.insert({bb, SmallVector<Value>()});
    resultPerPhi.insert({bb, mergeResult});
  }

  // For each phi, we need one input for every predecessor of the block
  for (auto &bb : blocksToAddPhi) {

    // Avoid to cover a predecessor twice
    llvm::DenseSet<Block *> coveredPred;
    auto predecessors = bb->getPredecessors();

    for (Block *pred : predecessors) {
      if (coveredPred.contains(pred))
        continue;
      coveredPred.insert(pred);

      // If the predecessor does not contains a definition of the value, we move
      // to its immediate dominator, until we have found a definition.
      Block *predecessorOrDominator = nullptr;
      Value valueToUse = nullptr;

      do {
        predecessorOrDominator =
            !predecessorOrDominator
                ? pred
                : getImmediateDominator(funcRegion, predecessorOrDominator);

        if (inputBlocks.contains(predecessorOrDominator))
          valueToUse = inputBlocks[predecessorOrDominator];
        else if (resultPerPhi.contains(predecessorOrDominator))
          valueToUse = resultPerPhi.find(predecessorOrDominator)->getSecond();

      } while (!valueToUse);

      operandsPerPhi[bb].push_back(valueToUse);
    }
  }

  // Create the merge and then replace the values
  DenseMap<Block *, handshake::MergeOp> newMergePerPhi;

  for (auto *bb : blocksToAddPhi) {
    rewriter.setInsertionPointToStart(bb);
    auto mergeOp = rewriter.create<handshake::MergeOp>(bb->front().getLoc(),
                                                       operandsPerPhi[bb]);
    mergeOp->setAttr(NEW_PHI, rewriter.getUnitAttr());
    newMergePerPhi.insert({bb, mergeOp});
  }

  for (auto *bb : blocksToAddPhi)
    resultPerPhi.find(bb)->getSecond().setValue(newMergePerPhi[bb].getResult());

  // For each block, find the incoming value of the network
  for (Block &bb : funcRegion.getBlocks()) {

    Value foundValue = nullptr;
    Block *blockOrDominator = &bb;

    if (blocksToAddPhi.contains(&bb)) {
      inputPerBlock[&bb] = newMergePerPhi[&bb].getResult();
      continue;
    }

    do {
      if (!blockOrDominator->hasNoPredecessors())
        blockOrDominator = getImmediateDominator(funcRegion, blockOrDominator);

      if (inputBlocks.contains(blockOrDominator)) {
        foundValue = inputBlocks[blockOrDominator];
      } else if (blocksToAddPhi.contains(blockOrDominator)) {
        foundValue = newMergePerPhi[blockOrDominator].getResult();
      }

    } while (!foundValue);

    inputPerBlock[&bb] = foundValue;
  }

  for (auto &op : toSubstitue)
    op->set(inputPerBlock[op->getOwner()->getBlock()]);

  return success();
}

LogicalResult ftd::createPhiNetworkDeps(
    Region &funcRegion, PatternRewriter &rewriter,
    const DenseMap<OpOperand *, SmallVector<Value>> &dependenciesMap) {

  mlir::DominanceInfo domInfo;

  // For each pair of operand and its dependencies
  for (auto &[operand, dependencies] : dependenciesMap) {

    Operation *operandOwner = operand->getOwner();
    auto startValue = (Value)funcRegion.getArguments().back();

    /// Lambda to run the SSA analysis over the pair of values {dep, startValue}
    /// and properly connect the operand `op` to the correct value in the
    /// network.
    auto connect = [&](OpOperand *op, Value dep) -> LogicalResult {
      Operation *depOwner = dep.getDefiningOp();

      // If the producer and the consumer are in the same basic block, and the
      // producer properly dominates the consumer (i.e. comes before in a linear
      // sense) then the consumer is directly connected to the producer without
      // further mechanism.
      if (dep.getParentBlock() == operandOwner->getBlock() &&
          domInfo.properlyDominates(depOwner, operandOwner)) {
        op->set(dep);
        return success();
      }

      // Otherwise, we run the SSA insertion
      SmallVector<mlir::OpOperand *> operandsToChange = {op};
      SmallVector<Value> inputValues = {startValue, dep};

      if (failed(ftd::createPhiNetwork(funcRegion, rewriter, inputValues,
                                       operandsToChange))) {
        return failure();
      }

      return success();
    };

    // If the operand has not dependencies, then it can be connected to start
    // directly.
    if (dependencies.size() == 0) {
      operand->set(startValue);
      continue;
    }

    // If the operand has one dependency only, there is no need for a join.
    if (dependencies.size() == 1) {
      if (failed(connect(operand, dependencies[0])))
        return failure();
      continue;
    }

    // If the operand has many dependencies, then each of them is singularly
    // connected with an SSA network, and then everything is joined.
    ValueRange operands = dependencies;
    rewriter.setInsertionPointToStart(operand->getOwner()->getBlock());
    auto joinOp = rewriter.create<handshake::JoinOp>(
        operand->getOwner()->getLoc(), operands);
    joinOp->moveBefore(operandOwner);

    for (unsigned i = 0; i < dependencies.size(); i++) {
      if (failed(connect(&joinOp->getOpOperand(i), dependencies[i])))
        return failure();
    }

    operand->set(joinOp.getResult());
  }

  return success();
}

/// FTD Distribution Logic
struct PathStep {
  std::string var;
  bool value;

  bool operator==(const PathStep &other) const {
    return var == other.var && value == other.value;
  }
  bool operator!=(const PathStep &other) const { return !(*this == other); }
};

using PathContext = std::vector<PathStep>;

struct VariableRequirement {
  std::string varName;
  PathContext path;
};

struct SignalRegistry {
  // Maps a variable name to its available versions across different paths.
  // Each entry stores the path context where the value is valid.
  std::map<std::string, std::vector<std::pair<PathContext, Value>>> map;

  // Registers a physical signal available at a specific path context.
  void registerSignal(StringRef var, const PathContext &path, Value val) {
    map[var.str()].push_back({path, val});
  }

  // Finds the best signal source using Longest Prefix Match.
  // Returns the value defined in the deepest matching path context.
  Value lookup(StringRef var, const PathContext &queryPath) {
    std::string v = var.str();
    if (map.find(v) == map.end())
      return nullptr;

    Value bestMatch = nullptr;
    size_t bestLen = 0;
    bool foundAny = false;

    for (auto &entry : map[v]) {
      const PathContext &regPath = entry.first;
      if (regPath.size() > queryPath.size())
        continue;

      // Filter: The registered path must be a prefix of the query path
      // to ensure the signal lies on the same control flow path.
      bool isPrefix = true;
      for (size_t i = 0; i < regPath.size(); ++i) {
        if (regPath[i] != queryPath[i]) {
          isPrefix = false;
          break;
        }
      }

      // Selection: Choose the longest matching prefix (closest definition).
      if (isPrefix) {
        if (!foundAny || regPath.size() >= bestLen) {
          bestLen = regPath.size();
          bestMatch = entry.second;
          foundAny = true;
        }
      }
    }
    return bestMatch;
  }
};

/// Retrieves the initial value from BlockIndexing.
static Value getOriginalValue(PatternRewriter &rewriter, StringRef varName,
                              const ftd::BlockIndexing &bi) {
  StringRef lookupName = varName;
  if (lookupName.startswith("~")) {
    llvm::errs() << "[FTD Error] Negated variable '" << varName << "'.\n";
    lookupName = lookupName.drop_front();
  }

  auto conditionOpt = bi.getBlockFromCondition(lookupName.str());
  if (!conditionOpt.has_value())
    return nullptr;

  Operation *term = conditionOpt.value()->getTerminator();
  if (!term || term->getNumOperands() == 0)
    return nullptr;

  return term->getOperand(0);
}

static BoolExpression *getBlockLoopExitCondition(Block *loopExit, CFGLoop *loop,
                                          CFGLoopInfo &li,
                                          const ftd::BlockIndexing &bi) {

  // Get the boolean expression associated to the block exit
  BoolExpression *blockCond =
      BoolExpression::parseSop(bi.getBlockCondition(loopExit));

  // Since we are in a loop, the terminator is a conditional branch.
  auto *terminatorOperation = loopExit->getTerminator();
  auto condBranch = dyn_cast<cf::CondBranchOp>(terminatorOperation);
  assert(condBranch && "Terminator of a loop must be `cf::CondBranchOp`");

  // If the destination of the false outcome is not the block, then the
  // condition must be negated
  if (li.getLoopFor(condBranch.getFalseDest()) != loop)
    blockCond->boolNegate();

  return blockCond;
}

static BoolExpression *
getLoopExitCondition(CFGLoop *loop, std::vector<std::string> *cofactorList,
                     mlir::CFGLoopInfo &li, const ftd::BlockIndexing &bi) {

  SmallVector<Block *> exitBlocks;
  loop->getExitingBlocks(exitBlocks);

  BoolExpression *fLoopExit = BoolExpression::boolZero();

  // Get the list of all the cofactors related to possible exit conditions
  for (Block *exitBlock : exitBlocks) {
    BoolExpression *blockCond =
        getBlockLoopExitCondition(exitBlock, loop, li, bi);
    fLoopExit = BoolExpression::boolOr(fLoopExit, blockCond);
    cofactorList->push_back(bi.getBlockCondition(exitBlock));
    fLoopExit = fLoopExit->boolMinimize();
  }

  // Sort the cofactors alphabetically
  std::sort(cofactorList->begin(), cofactorList->end());

  return fLoopExit;
}

/// Converts a boolean expression node to a circuit signal.
static Value boolExpressionToCircuit(
    PatternRewriter &rewriter, experimental::boolean::BoolExpression *expr,
    Block *block, SignalRegistry &registry, const PathContext &currentPath,
    const ftd::BlockIndexing &bi) {

  // Case 1: Variable
  if (expr->type == ExpressionType::Variable) {
    SingleCond *singleCond = static_cast<SingleCond *>(expr);
    std::string varName = singleCond->id;

    // 1. Registry Lookup
    Value val = registry.lookup(varName, currentPath);

    // 2. Fallback
    if (!val) {
      val = getOriginalValue(rewriter, varName, bi);

      if (!val) {
        llvm::errs() << "[FTD Error] Variable '" << varName
                     << "' not found in Registry or BlockIndexing.\n";
        assert(val && "Signal missing from IR");
      }

      if (!val.getType().isa<handshake::ChannelType>()) {
        val.setType(ftd::channelifyType(val.getType()));
      }
    }

    // 3. Handle Negation
    if (singleCond->isNegated) {
      auto notOp =
          rewriter.create<handshake::NotOp>(val.getLoc(), val.getType(), val);
      notOp->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());
      return notOp.getResult();
    }
    return val;
  }

  // Case 2: Constant
  auto sourceOp = rewriter.create<handshake::SourceOp>(block->front().getLoc());
  auto intType = rewriter.getIntegerType(1);
  int constVal = (expr->type == ExpressionType::One ? 1 : 0);
  auto cstAttr = rewriter.getIntegerAttr(intType, constVal);
  auto constOp = rewriter.create<handshake::ConstantOp>(
      block->front().getLoc(), cstAttr, sourceOp.getResult());
  constOp->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());

  return constOp.getResult();
}

/// Recursively converts a BDD to a Mux Tree.
static Value bddToCircuit(PatternRewriter &rewriter, BDD *bdd, Block *block,
                          SignalRegistry &registry, PathContext currentPath,
                          const ftd::BlockIndexing &bi) {
  using namespace experimental::boolean;

  // 1. Leaf Node
  if (!bdd->successors.has_value()) {
    return boolExpressionToCircuit(rewriter, bdd->boolVariable, block, registry,
                                   currentPath, bi);
  }

  // 2. Mux Node
  std::string varName = bdd->boolVariable->toString();

  Value muxCond = registry.lookup(varName, currentPath);
  if (!muxCond) {
    muxCond = getOriginalValue(rewriter, varName, bi);
    assert(muxCond && "Mux condition not found");
    if (!muxCond.getType().isa<handshake::ChannelType>())
      muxCond.setType(ftd::channelifyType(muxCond.getType()));
  }

  SmallVector<Value> muxOperands;

  // Recursion: Update PathContext so downstream lookups find distributed
  // signals
  PathContext falsePath = currentPath;
  falsePath.push_back({varName, false});
  muxOperands.push_back(bddToCircuit(rewriter, bdd->successors.value().first,
                                     block, registry, falsePath, bi));

  PathContext truePath = currentPath;
  truePath.push_back({varName, true});
  muxOperands.push_back(bddToCircuit(rewriter, bdd->successors.value().second,
                                     block, registry, truePath, bi));

  auto muxOp = rewriter.create<handshake::MuxOp>(
      muxCond.getLoc(), muxOperands[0].getType(), muxCond, muxOperands);
  muxOp->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());

  return muxOp.getResult();
}

/// Generates the Suppression Logic (Mux Tree) for a branch's select signal.
/// It constructs the logic for the "UNREACHABLE" condition.
/// F_suppress = NOT( OR( All Valid Paths ) )
static Value
generateReachabilityLogic(PatternRewriter &rewriter, Block *block,
                          const std::vector<VariableRequirement> &requirements,
                          const PathContext &currentPath,
                          SignalRegistry &registry,
                          const ftd::BlockIndexing &bi, size_t startIndex) {

  using namespace experimental::boolean;

  // 1. Construct Boolean Expression for Valid Paths
  BoolExpression *fValid = BoolExpression::boolZero();

  for (const auto &req : requirements) {
    BoolExpression *pathExpr = BoolExpression::boolOne();

    // Iterate through the path suffix starting from the current split point
    for (size_t i = startIndex; i < req.path.size(); ++i) {
      PathStep step = req.path[i];
      // Construct: SingleCond(Type, Name, Negated)
      BoolExpression *stepExpr =
          new SingleCond(ExpressionType::Variable, step.var, !step.value);
      pathExpr = BoolExpression::boolAnd(pathExpr, stepExpr);
    }
    fValid = BoolExpression::boolOr(fValid, pathExpr);
  }

  // 2. Compute Suppression Condition: F_suppress = NOT( F_valid )
  // We want the circuit to output TRUE when the path is INVALID.
  BoolExpression *fSuppress = fValid->boolNegate();
  fSuppress = fSuppress->boolMinimize();

  // 4. Build BDD and Circuit for Suppression Condition
  std::set<std::string> vars = fSuppress->getVariables();
  std::vector<std::string> cofactorList(vars.begin(), vars.end());

  // Sort cofactor list to match topological order (e.g., c0, then c1)
  // This ensures the Mux Tree structure matches the dependency order.
  std::sort(cofactorList.begin(), cofactorList.end(),
            [&](const std::string &a, const std::string &b) {
              auto idA = bi.getBlockFromCondition(a);
              auto idB = bi.getBlockFromCondition(b);
              if (!idA || !idB)
                return a < b;
              return bi.isLess(idA.value(), idB.value());
            });

  BDD *bdd = buildBDD(fSuppress, cofactorList);

  // Note: bddToCircuit uses registry.lookup. If the suppression logic involves
  // variables distributed earlier (like c3a), it will correctly find them.
  return bddToCircuit(rewriter, bdd, block, registry, currentPath, bi);
}

/// Recursively builds the Branch Tree.
static void
buildBranchTreeRecursive(PatternRewriter &rewriter, StringRef currentVar,
                         std::vector<VariableRequirement> &requirements,
                         PathContext currentPath, SignalRegistry &registry,
                         const ftd::BlockIndexing &bi) {

  // 1. Retrieve Data Signal
  // Look up the current data signal to be distributed from the registry using
  // the current path context.
  Value sourceVal = registry.lookup(currentVar, currentPath);
  assert(sourceVal && "Source value for distribution not found");

  // 2. Identify Split Variable
  // Key: {Variable Name, Value} -> Value: List of requirements.
  std::map<std::pair<std::string, bool>, std::vector<VariableRequirement>>
      groups;
  std::string splitVar = "";
  bool splitFound = false;

  size_t maxDepth = 0;
  for (const auto &req : requirements)
    maxDepth = std::max(maxDepth, req.path.size());

  // Scan forward to find the first point where requirements disagree on a
  // variable value.
  size_t scanDepth = currentPath.size();
  for (; scanDepth < maxDepth; ++scanDepth) {
    groups.clear();
    splitVar = "";

    // Simply collect the step at this depth.
    for (auto &req : requirements) {
      PathStep step = req.path[scanDepth];
      if (splitVar == "")
        splitVar = step.var;

      groups[{step.var, step.value}].push_back(req);
    }

    // Divergence found: We have both True and False branches.
    if (groups.size() > 1) {
      splitFound = true;
      llvm::errs() << "[FTD] Split Variable Found: " << splitVar
                   << " at Depth: " << scanDepth << "\n";
      break;
    }
  }

  if (!splitFound)
    return;

  // 3. Retrieve Raw Select Signal
  // We need the physical control signal corresponding to the 'splitVar' found
  // above.
  Value conditionVal = registry.lookup(splitVar, currentPath);
  if (!conditionVal) {
    // Fallback: If not in registry, get the original value from the IR
    // (BlockIndexing).
    conditionVal = getOriginalValue(rewriter, splitVar, bi);
  }
  assert(conditionVal && "Splitter condition value not found");

  // Ensure Types are compatible with Handshake channels.
  if (!conditionVal.getType().isa<handshake::ChannelType>())
    conditionVal.setType(ftd::channelifyType(conditionVal.getType()));
  if (!sourceVal.getType().isa<handshake::ChannelType>())
    sourceVal.setType(ftd::channelifyType(sourceVal.getType()));

  // 4. Register Outputs and Recurse
  // [Context Backfilling]
  // Since we might have skipped several variables (Common Prefix) to reach
  // 'splitVar', we must fill these skipped steps back into the PathContext.
  // This ensures that the recursive call has a continuous path history, keeping
  // index alignment correct.
  PathContext baseNextPath = currentPath;
  if (!groups.empty()) {
    // Take the first requirement as a template to retrieve the skipped steps.
    const auto &repReq = groups.begin()->second.front();
    for (size_t k = currentPath.size(); k < scanDepth; ++k)
      baseNextPath.push_back({repReq.path[k].var, repReq.path[k].value});
  }

  // 5. [Suppression Logic]
  // Generate the logic to identify "Unreachable" or "Invalid" paths.
  // We pass 'scanDepth' (the index of splitVar) to the generator.
  // This tells the generator to check validity starting from the current split
  // variable, effectively ignoring the "Common Prefix" variables skipped in
  // Step 2 (which are implicitly valid). The logic checks the entire future
  // path to ensure reachability.
  Value suppressCondition = generateReachabilityLogic(
      rewriter, sourceVal.getParentBlock(), requirements, baseNextPath,
      registry, bi, scanDepth);

  if (!suppressCondition.getType().isa<handshake::ChannelType>())
    suppressCondition.setType(ftd::channelifyType(suppressCondition.getType()));

  // [Suppression Branch]
  // Acts as a filter:
  // If suppressCondition is TRUE (Invalid Path) -> Output to Sink (Discard
  // Token). If suppressCondition is FALSE (Valid Path)  -> Output to
  // 'activeSelectSignal' (Pass Token).
  SmallVector<Type> suppResultTypes = {conditionVal.getType(),
                                       conditionVal.getType()};
  auto suppBranch = rewriter.create<handshake::ConditionalBranchOp>(
      conditionVal.getLoc(), suppResultTypes, suppressCondition, conditionVal);
  suppBranch->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());

  // False Output -> Active Select (Pass to Main Branch)
  Value activeSelectSignal = suppBranch.getFalseResult();

  // 6. [Distribution Logic] Main Branch
  SmallVector<Type> resultTypes = {sourceVal.getType(), sourceVal.getType()};

  // Create the branch that splits the 'sourceVal' based on the (possibly
  // filtered) 'activeSelectSignal'.
  auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
      sourceVal.getLoc(), resultTypes, activeSelectSignal, sourceVal);
  branchOp->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());

  Value trueResult = branchOp.getTrueResult();
  Value falseResult = branchOp.getFalseResult();

  // Handle the True branch recursion
  if (!groups[{splitVar, true}].empty()) {
    PathContext truePath = baseNextPath;
    truePath.push_back({splitVar, true});
    registry.registerSignal(currentVar, truePath, trueResult);
    buildBranchTreeRecursive(rewriter, currentVar, groups[{splitVar, true}],
                             truePath, registry, bi);
  }

  // Handle the False branch recursion
  if (!groups[{splitVar, false}].empty()) {
    PathContext falsePath = baseNextPath;
    falsePath.push_back({splitVar, false});
    registry.registerSignal(currentVar, falsePath, falseResult);
    buildBranchTreeRecursive(rewriter, currentVar, groups[{splitVar, false}],
                             falsePath, registry, bi);
  }
}

/// Main entry point of distribution logic.
static void buildDistributionNetwork(PatternRewriter &rewriter, 
                                     const ftd::LocalCFG &lcfg,
                                     const ftd::BlockIndexing &bi,
                                     SignalRegistry &registry) {
  using namespace experimental::boolean;

  // 1. Collect Variable Requirements
  std::map<std::string, std::vector<VariableRequirement>> varNeeds;
  std::function<void(Block *, PathContext)> collect = [&](Block *curr,
                                                          PathContext path) {

    // Stop recursion if reaching the consumer or the sink block
    if (curr == lcfg.newCons || curr == lcfg.sinkBB)
      return;

    // Find the condition variable
    Block *origBlock = lcfg.origMap.lookup(curr);
    std::string var = "";
    var = bi.getBlockCondition(origBlock);

    // Record the variable requirement
    if (!var.empty()) {
      varNeeds[var].push_back({var, path});
    }

    auto *term = curr->getTerminator();
    if (!term)
      return;

    // Handle conditional branches.
    if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
      if (!var.empty()) {
        PathContext truePath = path;
        truePath.push_back({var, true});
        collect(condBr.getTrueDest(), truePath);

        PathContext falsePath = path;
        falsePath.push_back({var, false});
        collect(condBr.getFalseDest(), falsePath);
      } else {
        llvm::errs() << "[FTD ERROR] CondBranchOp encountered with empty condition variable at block " 
                     << origBlock << ". Successors will not be traversed.\n";
      }
    } else {
      // Handle unconditional branches
      for (Block *succ : term->getSuccessors()) {
        collect(succ, path);
      }
    }
  };

  // Start collection from the producer block
  if (lcfg.newProd) {
    collect(lcfg.newProd, {});
  }

  // 2. Topological Sort
  std::vector<std::string> sortedVars;
  for (auto &kv : varNeeds)
    sortedVars.push_back(kv.first);

  std::sort(sortedVars.begin(), sortedVars.end(),
            [&](const std::string &a, const std::string &b) {
              auto idA = bi.getBlockFromCondition(a);
              auto idB = bi.getBlockFromCondition(b);
              if (!idA || !idB) {
                llvm::errs()
                    << "[FTD Warning] Variable missing from BlockIndexing: '"
                    << (idA ? b : a) << "'\n";
                return a < b;
              }
              return bi.isLess(idA.value(), idB.value());
            });

  // 3. Initial Registration and Construct Branch Trees
  for (const auto &var : sortedVars) {
    Value rawVal = getOriginalValue(rewriter, var, bi);
    if (rawVal) {
      if (!rawVal.getType().isa<handshake::ChannelType>())
        rawVal.setType(ftd::channelifyType(rawVal.getType()));
      registry.registerSignal(var, {}, rawVal);
      if (varNeeds[var].size() > 1) {
        buildBranchTreeRecursive(rewriter, var, varNeeds[var], {}, registry,
                                 bi);
      }
    } else {
      llvm::errs() << "[FTD Error] Variable '" << var
                   << "' not found in BlockIndexing during registration.\n";
      assert(rawVal && "Signal missing from IR");
    }
  }
}

/// Build a local control-flow subgraph (LocalCFG) between a producer and
/// consumer. The subgraph is reconstructed as a region with unique entry
/// (producer) and exit (sink).
static std::unique_ptr<ftd::LocalCFG>
buildLocalCFGRegion(OpBuilder &builder, Block *origProd, Block *origCons,
                    const ftd::BlockIndexing &bi) {
  auto L = std::make_unique<ftd::LocalCFG>();
  Location loc = builder.getUnknownLoc();

  // Setup Region Container
  OpBuilder::InsertionGuard guard(builder);
  auto funcType = builder.getFunctionType({}, {});
  auto dummyFunc =
      builder.create<func::FuncOp>(loc, "__ftd_local_cfg__", funcType);
  Region &R = dummyFunc.getBody();
  L->region = &R;
  L->containerOp = dummyFunc;

  // Sink Block: The unified exit for all paths (valid or suppressed).
  L->sinkBB = new Block();
  R.push_back(L->sinkBB);
  L->origMap[L->sinkBB] = nullptr;

  // Producer Block: The entry point of the local CFG.
  Block *entry = new Block();
  R.push_back(entry);
  L->newProd = entry;
  L->origMap[entry] = origProd;

  DenseMap<Block *, Block *> cloned;
  DenseSet<Block *> visited;
  // Avoid scheduling the same orig block twice
  DenseSet<Block *> enqueued;
  cloned[origProd] = entry;

  // DFS Function
  std::function<void(Block *, Block *)> dfs = [&](Block *currOrig,
                                                  Block *currNew) {
    visited.insert(currOrig);

    auto *term = currOrig->getTerminator();

    // Dead End: Implicit flow to Sink.
    if (!term || term->getNumSuccessors() == 0) {
      builder.setInsertionPointToEnd(currNew);
      builder.create<cf::BranchOp>(loc, L->sinkBB);
      return;
    }

    // LIST 1: The distinct successors in the NEW Local CFG for the current
    // block. Used to construct the BranchOp/CondBranchOp.
    SmallVector<Block *, 2> localSuccessors;

    // LIST 2: The successors that are valid and new, requiring further DFS
    // traversal. Stored as pairs: {Original Successor, New Local Block}.
    SmallVector<std::pair<Block *, Block *>, 2> successorsToVisit;

    for (auto it = term->successor_begin(), e = term->successor_end(); it != e;
         ++it) {
      Block *succOrig = *it;
      Block *nextBlockInLocalCFG =
          nullptr; // Where the edge points to in the new graph

      // Determine the edge destination based on rules

      // Case 1: Consumer Reached (Valid Delivery)
      if (succOrig == origCons) {
        if (succOrig == origProd) {
          // Self-loop delivery
          if (!L->secondVisitBB) {
            L->secondVisitBB = new Block();
            R.push_back(L->secondVisitBB);
            L->origMap[L->secondVisitBB] = nullptr;
            // Terminate SecondVisit immediately to Sink
            OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointToEnd(L->secondVisitBB);
            builder.create<cf::BranchOp>(loc, L->sinkBB);
          }
          nextBlockInLocalCFG = L->secondVisitBB;
          L->newCons = L->secondVisitBB;
        } else {
          // Standard delivery
          if (!L->newCons || L->newCons == L->secondVisitBB) {
            Block *proxy = new Block();
            R.push_back(proxy);
            L->origMap[proxy] = succOrig;
            L->newCons = proxy;
            // Terminate Proxy immediately to Sink
            OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointToEnd(proxy);
            builder.create<cf::BranchOp>(loc, L->sinkBB);
          }
          nextBlockInLocalCFG = L->newCons;
        }
      }
      // Case 2: Producer revisited, consumer not reached (Invalid)
      else if (succOrig == origProd) {
        nextBlockInLocalCFG = L->sinkBB;
      }
      // Already cloned (discovered) but may not be visited yet.
      // Reuse the existing clone to avoid duplicating blocks for the same orig.
      else if (cloned.count(succOrig)) {
        nextBlockInLocalCFG = cloned[succOrig];
        // If this node hasn't been visited yet, ensure it will be traversed
        // once.
        if (!visited.count(succOrig) && !enqueued.count(succOrig)) {
          enqueued.insert(succOrig);
          successorsToVisit.push_back({succOrig, nextBlockInLocalCFG});
        }
      }
      // Case 3: Visited
      else if (visited.count(succOrig)) {
        // Normally unreachable now because cloned.count(succOrig) should hold
        // if it was ever visited through this builder. Keep as safety.
        nextBlockInLocalCFG = L->sinkBB;
      }
      // Case 4: Invalid Back-edge
      else if (bi.isLess(succOrig, currOrig)) {
        nextBlockInLocalCFG = L->sinkBB;
      }
      // Case 5: Valid Forward Edge (Continue Traversal)
      else {
        Block *newSucc = new Block();
        R.push_back(newSucc);
        cloned[succOrig] = newSucc;
        L->origMap[newSucc] = succOrig;

        nextBlockInLocalCFG = newSucc;
        // Schedule this node for DFS visitation (once)
        if (!enqueued.count(succOrig)) {
          enqueued.insert(succOrig);
          successorsToVisit.push_back({succOrig, newSucc});
        }
      }

      // Add the determined destination to the list of local successors
      localSuccessors.push_back(nextBlockInLocalCFG);
    }

    // Create the branch instruction
    builder.setInsertionPointToEnd(currNew);
    if (localSuccessors.size() == 1) {
      builder.create<cf::BranchOp>(loc, localSuccessors[0]);
    } else if (localSuccessors.size() == 2) {
      // Placeholder condition for 2-way branches
      Value cond = builder.create<arith::ConstantIntOp>(loc, 1, 1);
      builder.create<cf::CondBranchOp>(loc, cond, localSuccessors[0],
                                       localSuccessors[1]);
    } else {
      // Default fall-through for complex control flow
      builder.create<cf::BranchOp>(loc, L->sinkBB);
    }

    // Continue DFS
    for (auto &pair : successorsToVisit) {
      dfs(pair.first, pair.second);
    }
  };

  // Start DFS
  dfs(origProd, L->newProd);

  // Finalize Sink
  builder.setInsertionPointToEnd(L->sinkBB);
  builder.create<func::ReturnOp>(loc);

  if (!L->newCons)
    L->newCons = L->sinkBB;

  // Compute Topological Order
  DenseSet<Block *> visitedTopo;
  SmallVector<Block *> order;
  std::function<void(Block *)> topo = [&](Block *u) {
    if (!u || visitedTopo.contains(u))
      return;
    visitedTopo.insert(u);
    if (auto *term = u->getTerminator())
      for (auto it = term->successor_begin(), e = term->successor_end();
           it != e; ++it)
        topo(*it);
    order.push_back(u);
  };

  topo(L->newProd);
  std::reverse(order.begin(), order.end());
  L->topoOrder = std::move(order);

  // Physical Reordering
  // Reorder blocks in the region list to match the topological order.
  // This does not change the graph structure (pointers), only the memory
  // layout/print order.
  for (Block *b : L->topoOrder) {
    if (b != L->sinkBB) {
      b->moveBefore(L->sinkBB);
    }
  }

  return L;
}

/// Constructs a NEW LocalCFG that represents the Decision Graph.
/// \param rawGraph The source LocalCFG.
/// \param dependencies The set of blocks (from rawGraph) that are relevant decision nodes.
/// \param muxConstraints A map {Block* -> bool} enforcing specific values for blocks.
///                       If a block is in this map, the branch corresponding to !value is wired to Sink.
static std::unique_ptr<ftd::LocalCFG>
buildDecisionGraph(const ftd::LocalCFG &rawGraph,
                   const DenseSet<Block *> &dependencies,
                   const DenseMap<Block *, bool> &muxConstraints) {

  if (!rawGraph.newCons)
    return nullptr;

  // NodeSet: Consumer + Sink + Dependencies
  DenseSet<Block *> nodeSet;
  nodeSet.insert(rawGraph.newCons);
  nodeSet.insert(rawGraph.sinkBB);
  for (Block *b : dependencies) {
    nodeSet.insert(b);
  }

  // 2. Setup New Container
  auto newL = std::make_unique<ftd::LocalCFG>();
  OpBuilder builder(rawGraph.containerOp->getContext());
  Location loc = builder.getUnknownLoc();

  auto funcType = builder.getFunctionType({}, {});
  auto newContainer =
      builder.create<func::FuncOp>(loc, "__ftd_decision_graph__", funcType);
  Region &newRegion = newContainer.getBody();
  newL->region = &newRegion;
  newL->containerOp = newContainer;

  // 3. Create Blocks & Map
  DenseMap<Block *, Block *> oldToNew;

  for (Block *oldBlock : rawGraph.topoOrder) {
    if (nodeSet.contains(oldBlock)) {
      Block *newBlock = new Block();
      newRegion.push_back(newBlock);
      oldToNew[oldBlock] = newBlock;

      if (Block *origIR = rawGraph.origMap.lookup(oldBlock)) {
        newL->origMap[newBlock] = origIR;
      } else {
        newL->origMap[newBlock] = nullptr;
      }

      // [CRITICAL] Set newProd to the first valid block (TopoOrder)
      if (newL->newProd == nullptr) {
        newL->newProd = newBlock;
      }

      if (oldBlock == rawGraph.newCons)
        newL->newCons = newBlock;
      if (oldBlock == rawGraph.sinkBB)
        newL->sinkBB = newBlock;
      if (oldBlock == rawGraph.secondVisitBB)
        newL->secondVisitBB = newBlock;
    }
  }

  // Fallback for newProd
  if (newL->newProd == nullptr && !newL->region->empty()) {
    newL->newProd = &newL->region->front();
  }

  // --- 4. Helper: Find Nearest using DFS with Visited Set ---
  auto findNearest = [&](Block *start) -> Block * {
    if (!start)
      return nullptr;
    DenseSet<Block *> visited;
    std::function<Block *(Block *)> dfs = [&](Block *curr) -> Block * {
      if (!curr)
        return nullptr;
      if (nodeSet.contains(curr))
        return curr;
      if (!visited.insert(curr).second)
        return nullptr; // Cycle
      for (Block *succ : curr->getSuccessors()) {
        if (Block *res = dfs(succ))
          return res;
      }
      return nullptr;
    };
    return dfs(start);
  };

  // 5. Wire the Graph
  builder.setInsertionPointToStart(&newRegion.front());

  for (auto [oldBlock, newBlock] : oldToNew) {
    // Sink Logic: Terminate
    if (oldBlock == rawGraph.sinkBB) {
      builder.setInsertionPointToEnd(newBlock);
      builder.create<func::ReturnOp>(loc);
      continue;
    }

    // Consumer Logic: MUST Branch to Sink
    if (oldBlock == rawGraph.newCons) {
      builder.setInsertionPointToEnd(newBlock);
      // In LocalCFG, Consumer always branches to Sink.
      // We replicate this connection in the new graph.
      Block *newSink = newL->sinkBB;
      if (newSink) {
        builder.create<cf::BranchOp>(loc, newSink);
      } else {
        // Should not happen if Sink is in nodeSet
        builder.create<func::ReturnOp>(loc);
      }
      continue;
    }

    // Decision Node Logic
    Operation *term = oldBlock->getTerminator();
    if (!term)
      continue;

    builder.setInsertionPointToEnd(newBlock);

    if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
      Block *oldTrue = findNearest(condBr.getTrueDest());
      Block *oldFalse = findNearest(condBr.getFalseDest());

      Block *newTrue = oldToNew.lookup(oldTrue);
      Block *newFalse = oldToNew.lookup(oldFalse);

      // [Safety Wiring] If a path is dead/looping, wire to Sink
      if (!newTrue)
        newTrue = newL->sinkBB;
      if (!newFalse)
        newFalse = newL->sinkBB;

      // [Constraint Application]
      // If this block has a constraint, wire the invalid path to Sink.
      if (muxConstraints.count(oldBlock)) {
        bool requiredVal = muxConstraints.lookup(oldBlock);
        if (requiredVal) { 
          // Require True -> Wire False to Sink
          newFalse = newL->sinkBB;
        } else {
          // Require False -> Wire True to Sink
          newTrue = newL->sinkBB;
        }
      }

      builder.create<cf::CondBranchOp>(loc, condBr.getCondition(), newTrue,
                                       newFalse);
    } else {
      // Non-CondBranch nodes in the decision set (rare)
      // Wire to nearest valid successor or Sink
      Block *oldTarget = findNearest(term->getSuccessor(0));
      Block *newTarget = oldToNew.lookup(oldTarget);
      if (!newTarget)
        newTarget = newL->sinkBB;
      builder.create<cf::BranchOp>(loc, newTarget);
    }
  }

  // 6. Compute TopoOrder
  DenseSet<Block *> visited;
  SmallVector<Block *, 8> order;
  std::function<void(Block *)> topo = [&](Block *u) {
    if (!u || visited.contains(u))
      return;
    visited.insert(u);
    if (auto *term = u->getTerminator())
      for (auto it = term->successor_begin(); it != term->successor_end(); ++it)
        topo(*it);
    order.push_back(u);
  };

  if (newL->newProd) {
    topo(newL->newProd);
    std::reverse(order.begin(), order.end());
    newL->topoOrder = std::move(order);
  }

  return newL;
}

/// Apply the algorithm from FPL'22 to handle a non-loop situation of
/// producer and consumer
static void insertDirectSuppression(
    PatternRewriter &rewriter, handshake::FuncOp &funcOp, Operation *consumer,
    Value connection, const ftd::BlockIndexing &bi,
    ControlDependenceAnalysis::BlockControlDepsMap &cdAnalysis) {

  Block *entryBlock = &funcOp.getBody().front();
  Block *producerBlock = connection.getParentBlock();
  Block *consumerBlock = consumer->getBlock();

  bool debuglog = true;
  std::string funcName = funcOp.getName().str();
  std::string dir = "/home/yuqin/dynamatic-scripts/TempOutputs/";
  std::string cfgFile = dir + funcName + "_localcfg.txt";
  std::string logFile = dir + funcName + "_debuglog.txt";
  std::error_code EC_log;
  llvm::raw_fd_ostream log(logFile, EC_log,
                           static_cast<llvm::sys::fs::OpenFlags>(0x0004));
  llvm::raw_ostream &out = EC_log ? llvm::errs() : log;

  // Account for the condition of a Mux only if it corresponds to a GAMMA GSA
  // gate and the producer is one of its data inputs
  bool deliverToGamma = llvm::isa<handshake::MuxOp>(consumer) &&
                        consumer->hasAttr(FTD_EXPLICIT_GAMMA);

  if (debuglog) {
    out << "[FTD] Producer block: ";
    if (producerBlock)
      producerBlock->printAsOperand(out);
    else
      out << "(null)";
    out << ", Consumer block: ";
    if (consumerBlock)
      consumerBlock->printAsOperand(out);
    else
      out << "(null)";
    out << "\n";
    // Debug: dump consumer block control deps
    {
      Block *consumerBlock = consumer->getBlock();
      auto &prodEntry = cdAnalysis[producerBlock];
      auto &depsEntry = cdAnalysis[consumerBlock];

      auto printBlockSet = [&](llvm::StringRef label,
                               const DenseSet<Block *> &S) {
        out << label << " = { ";
        bool first = true;
        for (Block *b : S) {
          if (!first)
            out << ", ";
          if (b)
            b->printAsOperand(out);
          else
            out << "<null>";
          first = false;
        }
        out << " }\n";
      };
      printBlockSet("[FTD] prod forwardControlDeps",
                    prodEntry.forwardControlDeps);
      printBlockSet("[FTD] cons forwardControlDeps",
                    depsEntry.forwardControlDeps);
      printBlockSet("[FTD] cons allControlDeps", depsEntry.allControlDeps);
    }
  }

  // If producer is unreachable, the suppression is not needed.
  if (!isReachable(entryBlock, producerBlock)) {
    return;
  }

  // If deliverToGamma is true, we need to trace down the mux chain to find the
  // root condition block that effectively controls the delivery.
  if (deliverToGamma) {
    // Start tracing from the current consumer Mux
    Operation *lastMuxInChain = consumer;
    bool isChainActive = true;

    // 1. Trace down the Mux chain within the same block
    while (isChainActive) {
      Operation *nextMuxOp = nullptr;
      Value currentResult = lastMuxInChain->getResult(0);

      for (auto *user : currentResult.getUsers()) {
        // Condition: User is a Gamma gate in the Same Block
        if (llvm::isa<handshake::MuxOp>(user) &&
            user->hasAttr(FTD_EXPLICIT_GAMMA) &&
            user->getBlock() == lastMuxInChain->getBlock()) {

          // Skip if both data inputs use the connection
          unsigned connectionCount = 0;
          if (user->getOperand(1) == currentResult)
            connectionCount++;
          if (user->getOperand(2) == currentResult)
            connectionCount++;

          if (connectionCount != 2) {
            // Found the next Mux in the chain
            nextMuxOp = user;
            break;
          }
        }
      }

      if (nextMuxOp) {
        lastMuxInChain = nextMuxOp;
      } else {
        // End of chain reached
        isChainActive = false;
      }
    }

    // 2. Update producerBlock to be the block defining the condition of the last Mux
    Value finalCondition = lastMuxInChain->getOperand(0);
    producerBlock = returnMuxConditionBlock(finalCondition);

    if (debuglog) {
      out << "[FTD] deliverToGamma: Traced to final Mux: " << *lastMuxInChain << "\n";
      out << "      Updated Producer Block to Condition Source: ";
      if (producerBlock)
        producerBlock->printAsOperand(out);
      else
        out << "(null)";
      out << "\n";
    }
  }

  // Create a temporary builder to isolate the LocalCFG creation from the
  // main PatternRewriter. This prevents the rewriter from tracking the
  // temporary operations which are later erased manually.
  OpBuilder tmpBuilder(funcOp.getContext());
  auto locGraph =
      buildLocalCFGRegion(tmpBuilder, producerBlock, consumerBlock, bi);

  ControlDependenceAnalysis locCDA(*locGraph->region);
  DenseSet<Block *> locConsControlDepsTmp =
      locCDA.getAllBlockDeps()[locGraph->newCons].allControlDeps;

  // Map to store specific requirements for Mux Conditions (LocalBlock -> RequiredValue)
  DenseMap<Block *, bool> muxRequirements;
  SignalRegistry registry;
  rewriter.setInsertionPointToStart(consumer->getBlock());

  // Logic specific to Gamma delivery to identify Mux dependencies
  if (deliverToGamma) {
    Operation *currentMuxOp = consumer;
    Value currentConnection = connection;
    bool isChainActive = true;

    while (isChainActive) {
      bool isDataInput = false;
      bool requiredVal = true;

      // Check how the connection enters the current Mux
      // If operand(0) == currentConnection, it is the condition input. 
      // In that case, isDataInput remains false, and we simply traverse 
      // to the output to find the next mux in the chain.
      if (!(currentMuxOp->getOperand(0) == currentConnection)) {
        if (currentMuxOp->getOperand(1) == currentConnection) {
          // Input 1 is the FALSE input
          isDataInput = true;
          requiredVal = false; 
        } else if (currentMuxOp->getOperand(2) == currentConnection) {
          // Input 2 is the TRUE input
          isDataInput = true;
          requiredVal = true;
        }
      }

      if (isDataInput) {
        // 1. Get the condition value driving this Mux
        Value muxCondition = currentMuxOp->getOperand(0);

        // 2. Identify the Original Block defining this condition variable
        // (Using the provided helper function)
        Block *muxConditionBlock = returnMuxConditionBlock(muxCondition);

        // 3. Find the corresponding Block in the Local CFG
        // Since locGraph->origMap maps Local->Original, we iterate to reverse lookup.
        Block *condBlockLocal = nullptr;
        for (auto it : locGraph->origMap) {
          if (it.second == muxConditionBlock) {
            condBlockLocal = it.first;
            break;
          }
        }

        // 4. Add to dependencies and record requirement
        if (condBlockLocal) {
          // Add this block to the dependency set so path enumeration observes it
          locConsControlDepsTmp.insert(condBlockLocal);
          
          // Record the specific value required (True/False) to pass this Mux
          muxRequirements[condBlockLocal] = requiredVal;

          if (debuglog) {
            out << "[FTD] Added Mux Condition Dependency:\n";
            out << "      Orig Block: "; muxConditionBlock->printAsOperand(out);
            out << "\n      Local Block: "; condBlockLocal->printAsOperand(out);
            out << "\n      Required Value: " << (requiredVal ? "True" : "False") << "\n";
          }
        }
      }

      // 5. Traverse Downstream (Search for cascaded Gamma Muxes)
      Operation *nextMuxOp = nullptr;
      Value currentResult = currentMuxOp->getResult(0);

      for (auto *user : currentResult.getUsers()) {
        // Check if user is a Mux in the same block marked as Gamma
        if (llvm::isa<handshake::MuxOp>(user) &&
            user->hasAttr(FTD_EXPLICIT_GAMMA) &&
            user->getBlock() == currentMuxOp->getBlock()) {
          // Count how many data inputs use currentResult
          unsigned connectionCount = 0;
          if (user->getOperand(1) == currentResult) 
            connectionCount++;
          if (user->getOperand(2) == currentResult) 
            connectionCount++;

          // We only proceed if at most ONE data input comes from the previous mux.
          // Otherwise, the user is a temporary MUX which we don't care about.
          if (connectionCount != 2) {
            nextMuxOp = user;
            // Update connection for next iteration
            currentConnection = currentResult;
            break;
          }
        }
      }

      if (nextMuxOp) {
        currentMuxOp = nextMuxOp;
      } else {
        isChainActive = false;
        if (debuglog)
          out << "[FTD] End of Gamma Mux Chain search.\n";
      }
    }
  }

  // --- Common Logic for Building Suppression ---
  // If deliverToGamma is true, we use the empty constraints to build the distribution
  // network (so it covers all paths), but we use the muxRequirements to calculate
  // the specific suppression condition for this path.
  // If deliverToGamma is false, muxRequirements will be empty, so both graphs are identical.

  DenseMap<Block *, bool> emptyConstraints;
  auto fullDecisionGraph = buildDecisionGraph(*locGraph, locConsControlDepsTmp, emptyConstraints);
  
  // Build the distribution network based on the full graph
  buildDistributionNetwork(rewriter, *fullDecisionGraph, bi, registry);

  // Build the constrained graph for logic calculation
  auto decisionGraph = buildDecisionGraph(*locGraph, locConsControlDepsTmp, muxRequirements);
  ControlDependenceAnalysis locCDA2(*decisionGraph->region);
  DenseSet<Block *> locConsControlDeps =
      locCDA2.getAllBlockDeps()[decisionGraph->newCons].allControlDeps;

  BoolExpression *fCons =
      enumeratePaths(*decisionGraph, bi, locConsControlDeps);

  fCons = fCons->boolMinimize();
  if (debuglog) {
    out << "fCons  = " << fCons->toString() << "\n";
  }
  // f_supp = f_prod and not f_cons
  BoolExpression *fSup = fCons->boolNegate();
  fSup = fSup->boolMinimize();
  if (debuglog) {
    out << "fSupmin  = " << fSup->toString() << "\n";
  }

  fullDecisionGraph->containerOp->erase();
  decisionGraph->containerOp->erase();
  locGraph->containerOp->erase();

  // --- [NEW] Upstream Suppression Logic ---
  BoolExpression *fSupUp = BoolExpression::boolZero();

  // If deliverToGamma is true, we must also ensure the path from Gamma Root to Data Source is valid.
  if (deliverToGamma) {
    Block *upstreamProd = producerBlock; // This was updated earlier to be the Gamma Root
    Block *upstreamCons = connection.getParentBlock(); // Original Data Source

    if (upstreamProd && upstreamCons && upstreamProd != upstreamCons &&
        isReachable(entryBlock, upstreamProd)) {
      
      OpBuilder tmpBuilder2(funcOp.getContext());
      auto locGraphUp = buildLocalCFGRegion(tmpBuilder2, upstreamProd, upstreamCons, bi);

      if (locGraphUp->newCons) {
        // 1. Get dependencies for upstream graph
        ControlDependenceAnalysis upCDA(*locGraphUp->region);
        auto upDepsTmp = upCDA.getAllBlockDeps()[locGraphUp->newCons].allControlDeps;

        // 2. Build Upstream Decision Graph (No constraints needed)
        DenseMap<Block *, bool> noConstraints;
        auto decisionGraphUp = buildDecisionGraph(*locGraphUp, upDepsTmp, noConstraints);

        // 3. Calculate Upstream Logic
        ControlDependenceAnalysis finalUpCDA(*decisionGraphUp->region);
        auto upDeps = finalUpCDA.getAllBlockDeps()[decisionGraphUp->newCons].allControlDeps;
        
        BoolExpression *fConsUp = enumeratePaths(*decisionGraphUp, bi, upDeps);
        // Calculate Upstream Suppress Condition (stored separately)
        fSupUp = fConsUp->boolMinimize()->boolNegate()->boolMinimize();

        if (debuglog) {
          out << "fSupUp   = " << fSupUp->toString() << "\n";
        }

        decisionGraphUp->containerOp->erase();
      }
      locGraphUp->containerOp->erase();
    }
  }

  // If the activation function is not zero, then a suppress block is to be
  // inserted
  if (fSup->type != experimental::boolean::ExpressionType::Zero) {
    std::set<std::string> blocks = fSup->getVariables();

    std::vector<std::string> cofactorList(blocks.begin(), blocks.end());
    if (debuglog) {
      llvm::errs() << "[CofactorList] ";
      for (const auto &s : cofactorList)
        llvm::errs() << s << " ";
      llvm::errs() << "\n";
    }
    BDD *bdd = buildBDD(fSup, cofactorList);
    Value branchCond =
        bddToCircuit(rewriter, bdd, consumer->getBlock(), registry, {}, bi);

    // 2. Cascaded Upstream Filter
    if (fSupUp->type != experimental::boolean::ExpressionType::Zero) {
        std::set<std::string> blocksUp = fSupUp->getVariables();
        std::vector<std::string> cofactorListUp(blocksUp.begin(), blocksUp.end());
        BDD *bddUp = buildBDD(fSupUp, cofactorListUp);
        
        // Build the Upstream Condition Circuit
        Value upBranchCond = bddToCircuit(rewriter, bddUp, consumer->getBlock(), registry, {}, bi);

        // Create the Intermediate Branch
        // Data Input: branchCond (from Downstream logic)
        // Condition: upBranchCond (from Upstream logic)
        auto upBranchOp = rewriter.create<handshake::ConditionalBranchOp>(
          consumer->getLoc(), ftd::getListTypes(branchCond.getType()), 
          upBranchCond, branchCond);
        
        upBranchOp->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());

        // We use the False result (meaning "Not Suppressed Upstream") as the condition for the next stage
        branchCond = upBranchOp.getFalseResult();
    }

    // 3. Final Data Branch
    auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        consumer->getLoc(), ftd::getListTypes(connection.getType()), branchCond,
        connection);

    // Take into account the possibility of a mux to get the condition input
    // also as data input. In this case, the data input can be optimized to
    // a constant value, since it is always selected only when its own value
    // is true or false.
    for (auto &use : llvm::make_early_inc_range(connection.getUses())) {
      if (use.getOwner() != consumer)
        continue;
      if (llvm::isa<handshake::MuxOp>(consumer) &&
          consumer->getOperand(0) == connection &&
          use.getOperandNumber() != 0) {
        auto src = rewriter.create<handshake::SourceOp>(consumer->getLoc());
        auto innerType = connection.getType().cast<handshake::ChannelType>().getDataType();
        auto attr = rewriter.getIntegerAttr(innerType, (use.getOperandNumber() == 2)); 
        auto cst = rewriter.create<handshake::ConstantOp>(
            consumer->getLoc(), connection.getType(), attr, src.getResult());
        cst->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());
        use.set(cst.getResult());
        continue;
      }
      use.set(branchOp.getFalseResult());
    }
  }
}

void ftd::addRegenOperandConsumer(PatternRewriter &rewriter,
                                  handshake::FuncOp &funcOp,
                                  Operation *consumerOp, Value operand) {

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));
  BlockIndexing bi(funcOp.getBody());
  auto startValue = (Value)funcOp.getArguments().back();

  // Skip if the consumer was added by this function, if it is an init merge, if
  // it comes from the explicit gsa gate insertion process or if it is a generic
  // operation to skip
  if (consumerOp->hasAttr(FTD_REGEN) ||
      consumerOp->hasAttr(FTD_EXPLICIT_GAMMA) ||
      consumerOp->hasAttr(FTD_EXPLICIT_MU) ||
      consumerOp->hasAttr(FTD_INIT_MERGE) ||
      consumerOp->hasAttr(FTD_OP_TO_SKIP))
    return;

  // Skip if the consumer has to do with memory operations, cmerge networks or
  // if it is a conditional branch.
  if (llvm::isa_and_nonnull<handshake::MemoryOpInterface>(consumerOp) ||
      llvm::isa_and_nonnull<handshake::ControlMergeOp>(consumerOp) ||
      llvm::isa_and_nonnull<handshake::ConditionalBranchOp>(consumerOp))
    return;

  mlir::Operation *producerOp = operand.getDefiningOp();

  // Skip if the producer was added by this function or if it is an op to skip
  if (producerOp &&
      (producerOp->hasAttr(FTD_REGEN) || producerOp->hasAttr(FTD_OP_TO_SKIP)))
    return;

  // Skip if the producer has to do with memory operations
  if (llvm::isa_and_nonnull<handshake::MemoryOpInterface>(producerOp) ||
      llvm::isa_and_nonnull<MemRefType>(operand.getType()))
    return;

  // Last regenerated value
  Value regeneratedValue = operand;

  // Get all the loops for which we need to regenerate the
  // corresponding value
  SmallVector<CFGLoop *> loops = getLoopsConsNotInProd(
      consumerOp->getBlock(), operand.getParentBlock(), loopInfo);
  unsigned numberOfLoops = loops.size();

  auto cstType = rewriter.getIntegerType(1);
  auto cstAttr = IntegerAttr::get(cstType, 0);

  auto createRegenMux = [&](CFGLoop *loop) -> handshake::MuxOp {
    rewriter.setInsertionPointToStart(loop->getHeader());
    regeneratedValue.setType(channelifyType(regeneratedValue.getType()));

    // Determine the loop exit condition:
    // - If the condition spans multiple cofactors, build a BDD and
    //   translate it into a circuit.
    // - Otherwise, use the simple terminating condition of the exiting block
    Value conditionValue;
    std::vector<std::string> cofactorList;
    BoolExpression *exitCondition =
        getLoopExitCondition(loop, &cofactorList, loopInfo, bi);
    if (size(cofactorList) > 1) {
      BDD *bdd = buildBDD(exitCondition, cofactorList);
      SignalRegistry emptyRegistry;
      conditionValue =
          bddToCircuit(rewriter, bdd, loop->getHeader(), emptyRegistry, {}, bi);
    } else
      conditionValue = loop->getExitingBlock()->getTerminator()->getOperand(0);

    // Create the false constant to feed `init`
    auto constOp = rewriter.create<handshake::ConstantOp>(consumerOp->getLoc(),
                                                          cstAttr, startValue);
    constOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());

    // Create the `init` operation
    SmallVector<Value> mergeOperands = {constOp.getResult(), conditionValue};
    auto initMergeOp = rewriter.create<handshake::MergeOp>(consumerOp->getLoc(),
                                                           mergeOperands);
    initMergeOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());

    // The multiplexer is to be fed by the init block, and takes as inputs the
    // regenerated value and the result itself (to be set after) it was created.
    auto selectSignal = initMergeOp.getResult();
    selectSignal.setType(channelifyType(selectSignal.getType()));

    SmallVector<Value> muxOperands = {regeneratedValue, regeneratedValue};
    auto muxOp = rewriter.create<handshake::MuxOp>(regeneratedValue.getLoc(),
                                                   regeneratedValue.getType(),
                                                   selectSignal, muxOperands);

    muxOp->setOperand(2, muxOp->getResult(0));
    muxOp->setAttr(FTD_REGEN, rewriter.getUnitAttr());

    return muxOp;
  };

  // For each of the loop, from the outermost to the innermost
  for (unsigned i = 0; i < numberOfLoops; i++) {

    // If we are in the innermost loop (thus the iterator is at its end)
    // and the consumer is a loop merge, stop
    if (i == numberOfLoops - 1 && consumerOp->hasAttr(NEW_PHI))
      break;

    auto muxOp = createRegenMux(loops[i]);
    regeneratedValue = muxOp.getResult();
  }

  // Final replace the usage of the operand in the consumer with the output of
  // the last regen multiplexer created.
  consumerOp->replaceUsesOfWith(operand, regeneratedValue);
}

void ftd::addSuppOperandConsumer(PatternRewriter &rewriter,
                                 handshake::FuncOp &funcOp,
                                 Operation *consumerOp, Value operand) {

  Region &region = funcOp.getBody();
  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));
  BlockIndexing bi(region);
  auto cda = ControlDependenceAnalysis(region).getAllBlockDeps();

  // Skip the prod-cons if the producer is part of the operations related to
  // the BDD expansion or INIT merges
  if (consumerOp->hasAttr(FTD_OP_TO_SKIP) ||
      consumerOp->hasAttr(FTD_INIT_MERGE))
    return;

  // Do not take into account conditional branch
  if (llvm::isa<handshake::ConditionalBranchOp>(consumerOp) &&
      consumerOp->getOperand(0) != operand)
    return;

  // The consumer block is the block which contains the consumer
  Block *consumerBlock = consumerOp->getBlock();

  // The producer block is the block which contains the producer, and it
  // corresponds to the parent block of the operand. Since the operand might
  // have no producer operation (if it is a function argument) then this is the
  // only way to get the relevant information.
  Block *producerBlock = operand.getParentBlock();

  // If the consumer and the producer are in the same block without the
  // consumer being a multiplexer skip because no delivery is needed
  if (consumerBlock == producerBlock &&
      !llvm::isa<handshake::MuxOp>(consumerOp))
    return;

  if (Operation *producerOp = operand.getDefiningOp(); producerOp) {

    // In any cases, suppressing a branch ends up with incorrect results.
    if (llvm::isa<handshake::ConditionalBranchOp>(producerOp))
      return;

    // Skip the prod-cons if the consumer is part of the operations
    // related to the BDD expansion or INIT merges
    if (producerOp->hasAttr(FTD_OP_TO_SKIP) ||
        producerOp->hasAttr(FTD_INIT_MERGE))
      return;

    // Skip if either the producer of the consumer are
    // related to memory operations, or if the consumer is a conditional
    // branch
    if (llvm::isa_and_nonnull<handshake::MemoryControllerOp>(consumerOp) ||
        llvm::isa_and_nonnull<handshake::MemoryControllerOp>(producerOp) ||
        llvm::isa_and_nonnull<handshake::LSQOp>(producerOp) ||
        llvm::isa_and_nonnull<handshake::LSQOp>(consumerOp) ||
        llvm::isa_and_nonnull<handshake::ControlMergeOp>(producerOp) ||
        llvm::isa_and_nonnull<handshake::ControlMergeOp>(consumerOp) ||
        llvm::isa_and_nonnull<handshake::ConditionalBranchOp>(consumerOp) ||
        llvm::isa_and_nonnull<cf::CondBranchOp>(consumerOp) ||
        llvm::isa_and_nonnull<cf::BranchOp>(consumerOp) ||
        (llvm::isa<memref::LoadOp>(consumerOp) &&
         !llvm::isa<handshake::LoadOp>(consumerOp)) ||
        (llvm::isa<memref::StoreOp>(consumerOp) &&
         !llvm::isa<handshake::StoreOp>(consumerOp)) ||
        llvm::isa<mlir::MemRefType>(operand.getType()))
      return;

    // Handle the suppression in all the other cases (including the operand
    // being a function argument)
    insertDirectSuppression(rewriter, funcOp, consumerOp, operand, bi, cda);
  }
}

void ftd::addSupp(handshake::FuncOp &funcOp, PatternRewriter &rewriter) {

  // Set of original operations in the IR
  std::vector<Operation *> consumersToCover;
  for (Operation &consumerOp : funcOp.getOps())
    consumersToCover.push_back(&consumerOp);

  for (auto *consumerOp : consumersToCover) {
    for (auto operand : consumerOp->getOperands())
      addSuppOperandConsumer(rewriter, funcOp, consumerOp, operand);
  }
}

void ftd::addRegen(handshake::FuncOp &funcOp, PatternRewriter &rewriter) {

  // Set of original operations in the IR
  std::vector<Operation *> consumersToCover;
  for (Operation &consumerOp : funcOp.getOps())
    consumersToCover.push_back(&consumerOp);

  // For each producer/consumer relationship
  for (Operation *consumerOp : consumersToCover) {
    for (Value operand : consumerOp->getOperands())
      addRegenOperandConsumer(rewriter, funcOp, consumerOp, operand);
  }
}

LogicalResult experimental::ftd::addGsaGates(Region &region,
                                             PatternRewriter &rewriter,
                                             const gsa::GSAAnalysis &gsa,
                                             Backedge startValue,
                                             bool removeTerminators) {

  using namespace experimental::gsa;
  BlockIndexing bi(region);

  // The function instantiates the GAMMA and MU gates as provided by the GSA
  // analysis pass. A GAMMA function is translated into a multiplexer driven by
  // single control signal and fed by two operands; a MU function is
  // translated into a multiplexer driven by an init (it is currently
  // implemented as a Merge fed by a constant triggered from Start once and
  // from the loop condition thereafter). The input of one of these functions
  // might be another GSA function, and it's possible that the function was
  // not instantiated yet. For this reason, we keep track of the missing
  // operands, and reconnect them later on.
  //
  // To simplify the way GSA functions are handled, each of them has an unique
  // index.

  struct MissingGsa {
    // Index of the GSA function to modify
    unsigned phiIndex;
    // Index of the GSA function providing the result
    unsigned edgeIndex;
    // Index of the operand to modify
    unsigned operandInput;

    MissingGsa(unsigned pi, unsigned ei, unsigned oi)
        : phiIndex(pi), edgeIndex(ei), operandInput(oi) {}
  };

  if (region.getBlocks().size() == 1)
    return success();

  // List of missing GSA functions
  SmallVector<MissingGsa> missingGsaList;
  // List of gammas with only one input
  DenseSet<Operation *> oneInputGammaList;
  // Maps the index of each GSA function to each real operation
  DenseMap<unsigned, Operation *> gsaList;

  // For each block excluding the first one, which has no GSA
  for (Block &block : llvm::drop_begin(region)) {

    // For each GSA function
    ArrayRef<Gate *> gates = gsa.getGatesPerBlock(&block);
    for (Gate *gate : gates) {

      Location loc = block.front().getLoc();
      rewriter.setInsertionPointToStart(&block);
      SmallVector<Value> operands;

      // Maintain the index of the current operand
      unsigned operandIndex = 0;
      // Checks whether one index is empty
      int nullOperand = -1;

      // For each of its operand
      for (auto *operand : gate->operands) {
        // If the input is another GSA function, then a dummy value is used as
        // operand and the operations will be reconnected later on.
        // If the input is empty, we keep track of its index.
        // In the other cases, we already have the operand of the function.
        if (operand->isTypeGate()) {
          Gate *g = std::get<Gate *>(operand->input);
          operands.emplace_back(g->result);
          missingGsaList.emplace_back(
              MissingGsa(gate->index, g->index, operandIndex));
        } else if (operand->isTypeEmpty()) {
          nullOperand = operandIndex;
          operands.emplace_back(nullptr);
        } else {
          auto val = std::get<Value>(operand->input);
          operands.emplace_back(val);
        }
        operandIndex++;
      }

      // Get the condition for the block exiting
      Value conditionValue;
      
      // Determine the gate exit condition
      if (gate->gsaGateFunction == MuGate) {
        // For MU gates, we generate the condition based on the 
        // reaching condition from the loop header back to itself.
        OpBuilder tmpBuilder(region.getContext());
        Block *loopHeader = gate->getBlock();

        // 1. Build Local CFG with Prod = Cons = Loop Header
        // This graph captures the loop-back paths.
        auto locGraph =
            buildLocalCFGRegion(tmpBuilder, loopHeader, loopHeader, bi);
        ControlDependenceAnalysis locCDATmp(*locGraph->region);
        DenseSet<Block *> locConsControlDepsTmp =
            locCDATmp.getAllBlockDeps()[locGraph->newCons].allControlDeps;
        DenseMap<Block *, bool> emptyConstraints;
        auto decisionGraph = buildDecisionGraph(*locGraph, locConsControlDepsTmp, emptyConstraints);

        // 2. Construct distribution circuit and suppression circuit on distribution
        SignalRegistry registry;
        buildDistributionNetwork(rewriter, *decisionGraph, bi, registry);
        
        // 3. Control Dependence Analysis on the Decision Graph
        ControlDependenceAnalysis locCDA(*decisionGraph->region);
        auto depsMap = locCDA.getAllBlockDeps();
        DenseSet<Block *> locConsControlDeps =
            depsMap[decisionGraph->newCons].allControlDeps;

        // 4. Enumerate paths to get the boolean expression for the backedges
        // (i.e., condition is True if we are looping back)
        BoolExpression *fBackedge =
            enumeratePaths(*decisionGraph, bi, locConsControlDeps);

        // 5. Build BDD
        std::set<std::string> vars = fBackedge->getVariables();
        std::vector<std::string> cofactorList(vars.begin(), vars.end());

        // Sort cofactors strictly to ensure consistent BDD structure
        std::sort(cofactorList.begin(), cofactorList.end(),
                  [&](const std::string &a, const std::string &b) {
                    auto idA = bi.getBlockFromCondition(a);
                    auto idB = bi.getBlockFromCondition(b);
                    if (!idA || !idB)
                      return a < b;
                    return bi.isLess(idA.value(), idB.value());
                  });

        BDD *bdd = buildBDD(fBackedge, cofactorList);
        // 6. Convert to Circuit.
        conditionValue =
            bddToCircuit(rewriter, bdd, loopHeader, registry, {}, bi);

        // Clean up temporary graphs
        decisionGraph->containerOp->erase();
        locGraph->containerOp->erase();

      } else {
        // [Gamma Logic]
        if (size(gate->cofactorList) > 1) {
          // Apply a BDD expansion to the loop exit expression and the list of
          // cofactors
          BDD *bdd = buildBDD(gate->condition, gate->cofactorList);
          // Convert the boolean expression obtained through BDD to a circuit
          // We pass an empty registry, since this is not an expression for suppression 
          // and does not require distribution.
          SignalRegistry emptyRegistry;
          conditionValue =
              bddToCircuit(rewriter, bdd, gate->getBlock(), emptyRegistry, {}, bi);
        } else {
          conditionValue =
              gate->conditionBlock->getTerminator()->getOperand(0);
          // Ensure type consistency (Channel vs i1)
          if (!conditionValue.getType().isa<handshake::ChannelType>())
             conditionValue.setType(channelifyType(conditionValue.getType()));
        }
      }

      // If the function is MU, then we create a merge
      // and use its result as condition
      if (gate->gsaGateFunction == MuGate) {
        mlir::DominanceInfo domInfo;
        mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&region));

        // The inputs of the merge are the condition value and a `false`
        // constant driven by the start value of the function. This will
        // created later on, so we use a dummy value.
        SmallVector<Value> mergeOperands;
        mergeOperands.push_back(conditionValue);
        mergeOperands.push_back(conditionValue);

        auto initMergeOp =
            rewriter.create<handshake::MergeOp>(loc, mergeOperands);

        initMergeOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());

        // Replace the new condition value
        conditionValue = initMergeOp->getResult(0);
        conditionValue.setType(channelifyType(conditionValue.getType()));

        // Add the activation constant driven by the backedge value, which will
        // be then updated with the real start value, once available
        auto cstType = rewriter.getIntegerType(1);
        auto cstAttr = IntegerAttr::get(cstType, 0);
        rewriter.setInsertionPointToStart(initMergeOp->getBlock());
        auto constOp = rewriter.create<handshake::ConstantOp>(
            initMergeOp->getLoc(), cstAttr, startValue);
        constOp->setAttr(FTD_INIT_MERGE, rewriter.getUnitAttr());
        initMergeOp->setOperand(0, constOp.getResult());
      }

      // When a single input gamma is encountered, a mux is inserted as a
      // placeholder to perform the gamma/mu allocation flow. In the end,
      // these multiplexers are erased from the IR
      if (nullOperand >= 0) {
        operands[0] = operands[1 - nullOperand];
        operands[1] = operands[1 - nullOperand];
      }

      // Create the multiplexer
      auto mux = rewriter.create<handshake::MuxOp>(loc, gate->result.getType(),
                                                   conditionValue, operands);

      // The one input gamma is marked at an operation to skip in the IR and
      // later removed
      if (nullOperand >= 0)
        oneInputGammaList.insert(mux);

      if (gate->isRoot)
        rewriter.replaceAllUsesWith(gate->result, mux.getResult());

      gsaList.insert({gate->index, mux});

      if (gate->gsaGateFunction == MuGate)
        mux->setAttr(FTD_EXPLICIT_MU, rewriter.getUnitAttr());
      else
        mux->setAttr(FTD_EXPLICIT_GAMMA, rewriter.getUnitAttr());
    }
  }

  // For each of the GSA missing inputs, perform a replacement
  for (auto &missingMerge : missingGsaList) {

    auto *operandMerge = gsaList[missingMerge.phiIndex];
    auto *resultMerge = gsaList[missingMerge.edgeIndex];

    operandMerge->setOperand(missingMerge.operandInput + 1,
                             resultMerge->getResult(0));

    // In case of a one-input gamma, the other input must be replaced as well,
    // to avoid errors when the block arguments are erased later on
    if (oneInputGammaList.contains(operandMerge))
      operandMerge->setOperand(2 - missingMerge.operandInput,
                               resultMerge->getResult(0));
  }

  // Get rid of the multiplexers adopted as place-holders of one input gamma
  for (auto &op : llvm::make_early_inc_range(oneInputGammaList)) {
    int operandToUse = llvm::isa_and_nonnull<handshake::MuxOp>(
                           op->getOperand(1).getDefiningOp())
                           ? 1
                           : 2;
    op->getResult(0).replaceAllUsesWith(op->getOperand(operandToUse));
    rewriter.eraseOp(op);
  }

  if (!removeTerminators)
    return success();

  // Remove all the block arguments for all the non starting blocks
  for (Block &block : llvm::drop_begin(region))
    block.eraseArguments(0, block.getArguments().size());

  // Each terminator must be replaced so that it does not provide any block
  // arguments (possibly only the final control argument)
  for (Block &block : region) {
    if (Operation *terminator = block.getTerminator(); terminator) {
      rewriter.setInsertionPointAfter(terminator);
      if (auto cbr = dyn_cast<cf::CondBranchOp>(terminator); cbr) {
        while (!cbr.getTrueOperands().empty())
          cbr.eraseTrueOperand(0);
        while (!cbr.getFalseOperands().empty())
          cbr.eraseFalseOperand(0);
      } else if (auto br = dyn_cast<cf::BranchOp>(terminator); br) {
        while (!br.getOperands().empty())
          br.eraseOperand(0);
      }
    }
  }

  return success();
}

LogicalResult ftd::replaceMergeToGSA(handshake::FuncOp &funcOp,
                                     PatternRewriter &rewriter) {
  auto startValue = (Value)funcOp.getArguments().back();
  auto *ctx = funcOp->getContext();
  OpBuilder builder(ctx);

  // Create a backedge for the start value, to be sued during the merges to
  // multiplexers conversion
  BackedgeBuilder edgeBuilderStart(builder, funcOp.getRegion().getLoc());
  Backedge startValueBackedge = edgeBuilderStart.get(startValue.getType());

  // For each merge that was signed with the `NEW_PHI` attribute, substitute
  // it with its GSA equivalent
  for (handshake::MergeOp merge :
       llvm::make_early_inc_range(funcOp.getOps<handshake::MergeOp>())) {
    if (!merge->hasAttr(NEW_PHI))
      continue;
    gsa::GSAAnalysis gsa(merge, funcOp.getRegion());
    if (failed(ftd::addGsaGates(funcOp.getRegion(), rewriter, gsa,
                                startValueBackedge, false)))
      return failure();

    // Get rid of the merge
    merge.erase();
  }

  // Replace the backedge
  startValueBackedge.setValue(startValue);

  return success();
}
