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
/// Annotation to use when a suppression branch is added which needs to go
/// through the suppression mechanism again.
constexpr llvm::StringLiteral FTD_NEW_SUPP("ftd.supp");
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
  assert(muxConditionBlock &&
         "Mux condition must be feeding any block terminator.");
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

/// Given two sets containing object of type `Block*`, remove the common
/// entries.
static void eliminateCommonBlocks(DenseSet<Block *> &s1,
                                  DenseSet<Block *> &s2) {

  SmallVector<Block *> intersection;
  for (auto &e1 : s1) {
    if (s2.contains(e1))
      intersection.push_back(e1);
  }

  for (auto &bb : intersection) {
    s1.erase(bb);
    s2.erase(bb);
  }
}

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

/// The boolean condition to either generate or suppress a token are computed
/// by considering all the paths from the producer (`start`) to the consumer
/// (`end`). "Each path identifies a Boolean product of elementary conditions
/// expressing the reaching of the target BB from the corresponding member of
/// the set; the product of all such paths are added".
static BoolExpression *enumeratePaths(const ftd::LocalCFG &lcfg,
                                      const ftd::BlockIndexing &bi,
                                      const DenseSet<Block *> &controlDeps) {

  // 1. Path Finding using Iterative DFS
  std::vector<std::vector<Block *>> allPaths;

  struct StackFrame {
    Block *u;
    unsigned currIdx;
    unsigned numSuccs;
  };

  std::vector<StackFrame> dfsStack;
  std::vector<Block *> currentLocalPath; // Acts as 'visited in current path'

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
      std::vector<Block *> origPath;
      bool validMapping = true;

      for (Block *lb : currentLocalPath) {
        Block *ob = lcfg.origMap.lookup(lb);
        if (!ob && lb == lcfg.secondVisitBB) {
          ob = lcfg.origMap.lookup(lcfg.newProd);
        }
        if (!ob) {
          validMapping = false;
          break;
        }
        origPath.push_back(ob);
      }

      if (validMapping)
        allPaths.push_back(origPath);

      currentLocalPath.pop_back();
      dfsStack.pop_back();
      continue;
    }

    // --- Case B: Traverse Successors ---
    if (frame.currIdx < frame.numSuccs) {
      auto *term = frame.u->getTerminator();
      Block *succ = term->getSuccessor(frame.currIdx);
      frame.currIdx++;

      // 1. Skip Sink (Suppression)
      // 2. [CRITICAL FIX] Skip Cycle: Check if succ is already in current path
      // This prevents infinite loops while allowing the structure to exist.
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
    DenseSet<unsigned> tempCofactorSet;
    BoolExpression *minterm =
        getPathExpression(path, tempCofactorSet, bi, controlDeps, false);

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

  // Finds the best signal source using Longest Prefix Match (LPM).
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

      // Filter: A signal defined in a deeper path cannot be used
      // in a shallower path.
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
                              Block *block, const ftd::BlockIndexing &bi) {
  StringRef lookupName = varName;
  if (lookupName.startswith("~")) {
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
      val = getOriginalValue(rewriter, varName, block, bi);

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
      if (auto *defOp = val.getDefiningOp())
        rewriter.setInsertionPointAfter(defOp);
      else
        rewriter.setInsertionPointToStart(val.getParentBlock());

      auto notOp =
          rewriter.create<handshake::NotOp>(val.getLoc(), val.getType(), val);
      notOp->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());
      return notOp.getResult();
    }
    return val;
  }

  // Case 2: Constant
  rewriter.setInsertionPointToStart(block);
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
    muxCond = getOriginalValue(rewriter, varName, block, bi);
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

  if (auto *term = block->getTerminator())
    rewriter.setInsertionPoint(term);
  else
    rewriter.setInsertionPointToEnd(block);

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
    conditionVal = getOriginalValue(rewriter, splitVar, nullptr, bi);
  }
  assert(conditionVal && "Splitter condition value not found");

  // Ensure Types are compatible with Handshake channels.
  if (!conditionVal.getType().isa<handshake::ChannelType>())
    conditionVal.setType(ftd::channelifyType(conditionVal.getType()));
  if (!sourceVal.getType().isa<handshake::ChannelType>())
    sourceVal.setType(ftd::channelifyType(sourceVal.getType()));

  // 4. [Suppression Logic]
  // Generate the logic to identify "Unreachable" or "Invalid" paths.
  // We pass 'scanDepth' (the index of splitVar) to the generator.
  // This tells the generator to check validity starting from the current split
  // variable, effectively ignoring the "Common Prefix" variables skipped in
  // Step 2 (which are implicitly valid). The logic checks the entire future
  // path to ensure reachability.
  Value suppressCondition = generateReachabilityLogic(
      rewriter, sourceVal.getParentBlock(), requirements, currentPath, registry,
      bi, scanDepth);

  if (!suppressCondition.getType().isa<handshake::ChannelType>())
    suppressCondition.setType(
        ftd::channelifyType(suppressCondition.getType()));

  // Insertion Point Strategy: Append Mode.
  // Always insert at the end of the block (before terminator) to maintain SSA
  // dominance.
  if (auto *term = sourceVal.getParentBlock()->getTerminator())
    rewriter.setInsertionPoint(term);
  else
    rewriter.setInsertionPointToEnd(sourceVal.getParentBlock());

  // [Suppression Branch]
  // Acts as a filter:
  // If suppressCondition is TRUE (Invalid Path) -> Output to Sink (Discard
  // Token). If suppressCondition is FALSE (Valid Path)  -> Output to
  // 'activeSelectSignal' (Pass Token).
  SmallVector<Type> suppResultTypes = {conditionVal.getType(),
                                        conditionVal.getType()};
  auto suppBranch = rewriter.create<handshake::ConditionalBranchOp>(
      conditionVal.getLoc(), suppResultTypes, suppressCondition,
      conditionVal);
  suppBranch->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());

  // False Output -> Active Select (Pass to Main Branch)
  Value activeSelectSignal = suppBranch.getFalseResult();

  // 5. [Distribution Logic] Main Branch
  // Insert the main data distribution branch at the end of the block.
  if (auto *term = sourceVal.getParentBlock()->getTerminator())
    rewriter.setInsertionPoint(term);
  else
    rewriter.setInsertionPointToEnd(sourceVal.getParentBlock());

  SmallVector<Type> resultTypes = {sourceVal.getType(), sourceVal.getType()};

  // Create the branch that splits the 'sourceVal' based on the (possibly
  // filtered) 'activeSelectSignal'.
  auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
      sourceVal.getLoc(), resultTypes, activeSelectSignal, sourceVal);
  branchOp->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());

  Value trueResult = branchOp.getTrueResult();
  Value falseResult = branchOp.getFalseResult();

  // 6. Register Outputs and Recurse
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

/// Main entry point
static void buildDistributionNetwork(PatternRewriter &rewriter, BDD *rootBDD,
                                     Block *block, const ftd::BlockIndexing &bi,
                                     SignalRegistry &registry) {
  using namespace experimental::boolean;

  // 1. Collect Variable Requirements
  std::map<std::string, std::vector<VariableRequirement>> varNeeds;
  std::function<void(BDD *, PathContext)> collect = [&](BDD *node,
                                                        PathContext path) {
    if (!node)
      return;

    std::string var;
    if (node->boolVariable->type == ExpressionType::Variable) {
      SingleCond *singleCond = static_cast<SingleCond *>(node->boolVariable);
      var = singleCond->id;
      varNeeds[var].push_back({var, path});
    }

    if (node->successors.has_value()) {
      PathContext falsePath = path;
      falsePath.push_back({var, false});
      collect(node->successors.value().first, falsePath);
      PathContext truePath = path;
      truePath.push_back({var, true});
      collect(node->successors.value().second, truePath);
    }
  };
  collect(rootBDD, {});

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
    Value rawVal = getOriginalValue(rewriter, var, block, bi);
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
      // Case 3: Visited
      else if (visited.count(succOrig)) {
        if (cloned.count(succOrig)) {
          nextBlockInLocalCFG = cloned[succOrig];
        } else {
          nextBlockInLocalCFG = L->sinkBB;
        }
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
        // Schedule this node for DFS visitation
        successorsToVisit.push_back({succOrig, newSucc});
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

/// Transform the LocalCFG into a Binary Decision Graph
static void transformToDecisionGraph(ftd::LocalCFG &lcfg,
                                     const ftd::BlockIndexing &bi) {
  ControlDependenceAnalysis cda(*lcfg.region);
  auto &depsEntry = cda.getAllBlockDeps()[lcfg.newCons];
  // Store dependencies in a set for fast lookup
  DenseSet<Block *> dependencies = depsEntry.allControlDeps;
  // Initialize the set of relevant nodes with Consumer and Sink
  DenseSet<Block *> nodeSet;
  nodeSet.insert(lcfg.newCons);
  nodeSet.insert(lcfg.sinkBB);
  // Add all dependency blocks to the relevant node set
  for (Block *b : dependencies) {
    nodeSet.insert(b);
  }

  // Find the nearest relevant successor
  auto findNearest = [&](Block *curr) -> Block * {
    while (!nodeSet.contains(curr)) {
      curr = curr->getSuccessor(0);
    }
    return curr;
  };

  // Identify the new root (Producer) for the transformed graph
  Block *newRoot = nullptr;
  for (Block *b : lcfg.topoOrder) {
    if (nodeSet.contains(b)) {
      newRoot = b;
      break;
    }
  }

  // Update the graph entry point
  lcfg.newProd = newRoot;
  // This variable is no longer meaningful in the dependency graph
  lcfg.secondVisitBB = nullptr;

  // Initialize a builder to modify terminators
  OpBuilder builder(lcfg.containerOp->getContext());

  // Rewire edges for all relevant nodes
  for (Block *curr : nodeSet) {
    // Skip Sink and Consumer as they are terminal nodes
    if (curr == lcfg.sinkBB || curr == lcfg.newCons)
      continue;

    // Get the current terminator of the block
    Operation *term = curr->getTerminator();

    Block *trueTarget = nullptr;
    Block *falseTarget = nullptr;

    // Check if the terminator is a conditional branch
    if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
      // Find the nearest relevant node for the true path
      trueTarget = findNearest(condBr.getTrueDest());
      // Find the nearest relevant node for the false path
      falseTarget = findNearest(condBr.getFalseDest());
      // Safety check for null targets
      if (!trueTarget || !falseTarget) {
        llvm::errs() << "[FTD Error] findNearest returned null target.\n";
      }
      // Create a new conditional branch connecting directly to relevant nodes
      builder.setInsertionPoint(term);
      builder.create<cf::CondBranchOp>(term->getLoc(), condBr.getCondition(),
                                       trueTarget, falseTarget);
      // Remove the old terminator
      term->erase();
    } else {
      // Dependence nodes must be binary decision points (CondBranch)
      llvm::errs()
          << "[FTD Error] Block in dependence set is not a CondBranchOp.\n";
    }
  }

  // Identify blocks that are no longer part of the graph
  std::vector<Block *> blocksToRemove;
  for (Block &b : lcfg.region->getBlocks()) {
    if (!nodeSet.contains(&b)) {
      blocksToRemove.push_back(&b);
    }
  }
  // Clear block contents to break use-def chains
  for (Block *b : blocksToRemove) {
    b->clear();
  }
  // Erase the blocks safely
  for (Block *b : blocksToRemove) {
    b->erase();
  }
}

// Context managing variable state across hierarchy levels
struct VariableContext {
  std::string name;
  Block *definingBlock;
  Value currentValue;
  unsigned level;
  Block *sourceHeader = nullptr;

  // Distinguishes variables with the same name coming from different Loop
  // Header paths
  std::string getUniqueId() const {
    if (!sourceHeader)
      return name;
    return name + "_H" +
           std::to_string(reinterpret_cast<uintptr_t>(sourceHeader));
  }
};

// Layered Signal Registry
class LayeredSignalRegistry {
public:
  void registerVar(const VariableContext &ctx) {
    storage[ctx.level].push_back(ctx);
    lookupMap[ctx.level][ctx.getUniqueId()] = ctx;
  }

  const std::vector<VariableContext> &getVariablesForLevel(unsigned level) {
    return storage[level];
  }

  // Lookup: Prioritize specific Header source, otherwise look for native
  // variable in current level
  Value lookup(unsigned level, StringRef name, Block *entryHeader = nullptr) {
    std::string queryId = name.str();
    if (entryHeader)
      queryId +=
          "_H" + std::to_string(reinterpret_cast<uintptr_t>(entryHeader));

    // 1. Try exact match (Lowered variable)
    if (lookupMap[level].count(queryId))
      return lookupMap[level][queryId].currentValue;

    // 2. Try native variable match (when entryHeader is null or specific copy
    // not found)
    if (!entryHeader && lookupMap[level].count(name.str()))
      return lookupMap[level][name.str()].currentValue;

    return nullptr;
  }

private:
  std::map<unsigned, std::vector<VariableContext>> storage;
  std::map<unsigned, std::map<std::string, VariableContext>> lookupMap;
};

/// Calculates the reachability condition BDD for a variable in a sub-loop
/// (Rooted Sub-expression). This is the source logic for the Lowering Circuit
/// Select Signal.
static BDD *calculateReachabilityBDD(const ftd::LoopScope *scope,
                                     const VariableContext &varCtx,
                                     const ftd::BlockIndexing &bi,
                                     ftd::CyclicGraphManager &manager) {
  // 1. Temporarily build sub-loop DAG for analysis
  OpBuilder builder(scope->header->getParent()->getContext());
  auto localGraph = manager.extractLayeredCFG(scope, builder);
  ControlDependenceAnalysis cda(*localGraph->region);

  // 2. Find the corresponding block of variable definition in LocalCFG
  Block *targetNewBlock = nullptr;
  for (auto [newBlock, origBlock] : localGraph->origMap) {
    if (origBlock == varCtx.definingBlock) {
      targetNewBlock = newBlock;
      break;
    }
  }

  // If variable definition is not in current Scope (possibly lowered from
  // deeper level), considered reachable as long as Scope is entered
  if (!targetNewBlock) {
    return experimental::boolean::buildBDD(
        experimental::boolean::BoolExpression::boolOne(), {});
  }

  // 3. Compute control dependencies to reach the variable definition block
  DenseSet<Block *> depsTmp =
      cda.getAllBlockDeps()[targetNewBlock].allControlDeps;
  DenseSet<Block *> deps;
  for (Block *nb : depsTmp) {
    if (auto *orig = localGraph->origMap.lookup(nb))
      deps.insert(orig);
  }

  // 4. Compute path expression (Path to Definition)
  // Trick: Temporarily redirect newCons to utilize enumeratePaths for
  // calculating path to target
  Block *originalCons = localGraph->newCons;
  localGraph->newCons = targetNewBlock;
  BoolExpression *fReach = enumeratePaths(*localGraph, bi, deps);
  localGraph->newCons = originalCons; // Restore

  // 5. Generate Suppression Condition (NOT Reachable)
  BoolExpression *fSuppress = fReach->boolNegate();
  fSuppress = fSuppress->boolMinimize();

  std::set<std::string> vars = fSuppress->getVariables();
  std::vector<std::string> cofactorList(vars.begin(), vars.end());
  // Topological Sort Cofactor
  std::sort(cofactorList.begin(), cofactorList.end(),
            [&](const std::string &a, const std::string &b) {
              StringRef nameA = a;
              if (nameA.startswith("~"))
                nameA = nameA.drop_front();
              StringRef nameB = b;
              if (nameB.startswith("~"))
                nameB = nameB.drop_front();
              auto idA = bi.getBlockFromCondition(nameA.str());
              auto idB = bi.getBlockFromCondition(nameB.str());
              if (!idA || !idB)
                return a < b;
              return bi.isLess(idA.value(), idB.value());
            });

  return buildBDD(fSuppress, cofactorList);
}

/// Construct Lowering Circuit (Suppression Branch).
/// Lowers a variable from Inner Scope to Outer Scope.
static Value buildLoweringCircuit(PatternRewriter &rewriter,
                                  const VariableContext &varCtx,
                                  const ftd::LoopScope *innerScope,
                                  const ftd::BlockIndexing &bi,
                                  ftd::CyclicGraphManager &manager,
                                  SignalRegistry &localRegistry) {

  // 1. Calculate Select Signal BDD (Based on reachability within sub-loop)
  BDD *suppressBDD = calculateReachabilityBDD(innerScope, varCtx, bi, manager);

  // 2. Generate Circuit (Mux Tree)
  // localRegistry contains all known variables in current level (used to build
  // signals for BDD nodes)
  Value selectSignal = bddToCircuit(rewriter, suppressBDD, innerScope->header,
                                    localRegistry, {}, bi);

  // 3. Construct Branch
  Value dataInput = varCtx.currentValue;

  // Type Correction
  if (!selectSignal.getType().isa<handshake::ChannelType>())
    selectSignal.setType(ftd::channelifyType(selectSignal.getType()));
  if (!dataInput.getType().isa<handshake::ChannelType>())
    dataInput.setType(ftd::channelifyType(dataInput.getType()));

  // Determine Insertion Point
  if (auto *def = selectSignal.getDefiningOp())
    rewriter.setInsertionPointAfter(def);
  else
    rewriter.setInsertionPointToStart(innerScope->header);

  auto branch = rewriter.create<handshake::ConditionalBranchOp>(
      dataInput.getLoc(),
      dataInput.getType(), // True -> Pass to Outer (Lowered Result)
      dataInput.getType(), // False -> Sink (Suppressed)
      selectSignal, dataInput);

  branch->setAttr(FTD_OP_TO_SKIP, rewriter.getUnitAttr());
  rewriter.create<handshake::SinkOp>(dataInput.getLoc(),
                                     branch.getFalseResult());

  return branch.getTrueResult();
}

/// Core recursive processing function (Post-Order Traversal)
static void processLevelRecursive(ftd::LoopScope *scope,
                                  ftd::CyclicGraphManager &manager,
                                  PatternRewriter &rewriter,
                                  const ftd::BlockIndexing &bi,
                                  LayeredSignalRegistry &globalRegistry) {

  // 1. [Recursion] Process all sub-loops (Level N+1) first
  // Ensures Distribution and Lowering are ready for Inner Loops
  for (auto &subScope : scope->subLoops) {
    processLevelRecursive(subScope.get(), manager, rewriter, bi,
                          globalRegistry);
  }

  // Prepare Local Registry for current level (for Distribution and BDD
  // construction) First, populate all native variables of this level
  SignalRegistry currentLevelRegistry;
  for (auto &v : globalRegistry.getVariablesForLevel(scope->level)) {
    currentLevelRegistry.registerSignal(v.name, {}, v.currentValue);
  }

  // 2. [Lowering] Lower variables defined in sub-loops to the current level
  for (auto &subScope : scope->subLoops) {
    // Get all available variables from next level (Level N+1)
    auto innerVars = globalRegistry.getVariablesForLevel(scope->level + 1);

    for (auto &varCtx : innerVars) {
      // Only lower variables defined inside this sub-loop Scope
      // (Note: if lowered from deeper levels, it will also be included in
      // allBlocksInclusive)
      if (!subScope->allBlocksInclusive.contains(varCtx.definingBlock))
        continue;

      // Construct Lowering Circuit
      Value loweredVal = buildLoweringCircuit(
          rewriter, varCtx, subScope.get(), bi, manager, currentLevelRegistry);

      // Register as new variable for current level (Level N)
      VariableContext newCtx = varCtx;
      newCtx.currentValue = loweredVal;
      newCtx.level = scope->level;
      newCtx.sourceHeader = subScope->header; // Key: Mark source Header to
                                              // distinguish different paths

      globalRegistry.registerVar(newCtx);

      // Also update Local Registry for subsequent Distribution
      // At this point, same variable name might have multiple Values (from
      // different Headers), buildDistributionNetwork needs to support lookup
      currentLevelRegistry.registerSignal(varCtx.name, {}, loweredVal);
    }
  }

  // 3. [Local Analysis] Extract DAG for current level and generate expressions
  // This step happens after Lowering, but logically depends only on control
  // flow structure
  auto localGraph = manager.extractLayeredCFG(scope, rewriter);
  ControlDependenceAnalysis cda(*localGraph->region);

  // Compute F_cons (Consumer Reachability)
  DenseSet<Block *> depsTmp =
      cda.getAllBlockDeps()[localGraph->newCons].allControlDeps;
  DenseSet<Block *> deps;
  for (Block *nb : depsTmp) {
    if (auto *orig = localGraph->origMap.lookup(nb))
      deps.insert(orig);
  }

  BoolExpression *fCons = enumeratePaths(*localGraph, bi, deps);
  BoolExpression *fSup = fCons->boolNegate();
  fSup = fSup->boolMinimize();

  // 4. [Distribution] Build Distribution Network at current level
  // Now currentLevelRegistry contains native variables + lowered sub-level
  // variables
  if (fSup->type != experimental::boolean::ExpressionType::Zero) {
    std::set<std::string> vars = fSup->getVariables();
    std::vector<std::string> cofactorList(vars.begin(), vars.end());
    std::sort(cofactorList.begin(), cofactorList.end(),
              [&](const std::string &a, const std::string &b) {
                StringRef nameA = a;
                if (nameA.startswith("~"))
                  nameA = nameA.drop_front();
                StringRef nameB = b;
                if (nameB.startswith("~"))
                  nameB = nameB.drop_front();
                auto idA = bi.getBlockFromCondition(nameA.str());
                auto idB = bi.getBlockFromCondition(nameB.str());
                if (!idA || !idB)
                  return a < b;
                return bi.isLess(idA.value(), idB.value());
              });

    BDD *bdd = buildBDD(fSup, cofactorList);

    // Build Distribution Tree (Branch Tree)
    // Select Signal here will be looked up from currentLevelRegistry,
    // automatically matching the correct Lowered copy
    buildDistributionNetwork(rewriter, bdd, scope->header, bi,
                             currentLevelRegistry);
  }
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
  Value muxCondition = nullptr;

  bool debuglog = true;
  std::string funcName = funcOp.getName().str();
  std::string dir = "/home/yuaqin/new2/dynamatic-scripts/TempOutputs/";
  std::string cfgFile = dir + funcName + "_localcfg.txt";
  std::string logFile = dir + funcName + "_debuglog.txt";
  std::error_code EC_log;
  llvm::raw_fd_ostream log(logFile, EC_log,
                           static_cast<llvm::sys::fs::OpenFlags>(0x0004));
  llvm::raw_ostream &out = EC_log ? llvm::errs() : log;

  // Account for the condition of a Mux only if it corresponds to a GAMMA GSA
  // gate and the producer is one of its data inputs
  bool accountMuxCondition = llvm::isa<handshake::MuxOp>(consumer) &&
                             consumer->hasAttr(FTD_EXPLICIT_GAMMA) &&
                             (consumer->getOperand(1) == connection ||
                              consumer->getOperand(2) == connection);

  // Get the control dependencies from the producer
  DenseSet<Block *> prodControlDeps =
      cdAnalysis[producerBlock].forwardControlDeps;

  // Get the control dependencies from the consumer
  DenseSet<Block *> consControlDeps =
      cdAnalysis[consumerBlock].forwardControlDeps;

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

  // If the mux condition is to be taken into account, then the control
  // dependencies of the mux conditions are to be added to the consumer control
  // dependencies
  if (accountMuxCondition) {
    muxCondition = consumer->getOperand(0);
    Block *muxConditionBlock = returnMuxConditionBlock(muxCondition);
    DenseSet<Block *> condControlDeps =
        cdAnalysis[muxConditionBlock].forwardControlDeps;
    for (auto &x : condControlDeps)
      consControlDeps.insert(x);
  }

  // Get rid of common entries in the two sets
  eliminateCommonBlocks(prodControlDeps, consControlDeps);
  // If producer is unreachable, the suppression is not needed.
  if (!isReachable(entryBlock, producerBlock)) {
    return;
  }

  // auto locGraph =
  //     buildLocalCFGRegion(rewriter, producerBlock, consumerBlock, bi);
  // ControlDependenceAnalysis locCDA(*locGraph->region);
  // DenseSet<Block *> locConsControlDepsTmp =
  //     locCDA.getAllBlockDeps()[locGraph->newCons].forwardControlDeps;

  // DenseSet<Block *> locConsControlDeps;
  // for (Block *nb : locConsControlDepsTmp) {
  //   Block *orig = locGraph->origMap.lookup(nb);
  //   if (orig)
  //     locConsControlDeps.insert(orig);
  // }

  // BoolExpression *fCons = enumeratePaths(*locGraph, bi, locConsControlDeps);

  auto locGraph =
      buildLocalCFGRegion(rewriter, producerBlock, consumerBlock, bi);
  if (debuglog) {
    std::error_code EC;
    llvm::raw_fd_ostream file(cfgFile, EC,
                              static_cast<llvm::sys::fs::OpenFlags>(0x0004));

    if (!EC) {
      file << "=============================================================\n";
      file << "Function: " << funcOp.getName() << "\n";
      file << "Producer: ";
      if (producerBlock)
        producerBlock->printAsOperand(file);
      else
        file << "<null>";
      file << "\nConsumer: ";
      if (consumerBlock)
        consumerBlock->printAsOperand(file);
      else
        file << "<null>";
      file << "\n";

      file << "newProd: ";
      if (locGraph->newProd)
        locGraph->newProd->printAsOperand(file);
      else
        file << "<null>";
      file << "\nnewCons: ";
      if (locGraph->newCons)
        locGraph->newCons->printAsOperand(file);
      else
        file << "<null>";
      file << "\n";

      file << "regionFront = ";
      locGraph->region->front().printAsOperand(file);
      file << "\n";

      file << "\nBlocks (origMap):\n";
      for (Block &b : locGraph->region->getBlocks()) {
        file << "  new=";
        b.printAsOperand(file);
        Block *orig = locGraph->origMap.lookup(&b);
        file << " -> orig=";
        if (orig)
          orig->printAsOperand(file);
        else
          file << "<null>";
        file << "\n";
      }

      file << "\nEdges:\n";
      for (Block &b : locGraph->region->getBlocks()) {
        file << "  ";
        b.printAsOperand(file);
        file << " -> { ";
        if (auto *term = b.getTerminator()) {
          bool first = true;
          for (auto it = term->successor_begin(), e = term->successor_end();
               it != e; ++it) {
            if (!first)
              file << ", ";
            (*it)->printAsOperand(file);
            first = false;
          }
        }
        file << " }\n";
      }

      file << "\nTopoOrder:\n  ";
      for (Block *b : locGraph->topoOrder) {
        if (b)
          b->printAsOperand(file);
        else
          file << "<null>";
        file << ' ';
      }
      file << "\n\n";
    }
  }
  transformToDecisionGraph(*locGraph, bi);
  ftd::CyclicGraphManager manager(*locGraph);
  LayeredSignalRegistry globalRegistry;

  // 2. Initialization: Register original variables for all levels
  for (Block &b : funcOp.getBody()) {
    std::string condName = bi.getBlockCondition(&b);
    if (condName.empty())
      continue;
    Value val = nullptr;
    if (auto *term = b.getTerminator()) {
      if (term->getNumOperands() > 0)
        val = term->getOperand(0);
    }
    if (val) {
      unsigned lvl = manager.getNestingLevel(&b);
      VariableContext ctx;
      ctx.name = condName;
      ctx.definingBlock = &b;
      ctx.currentValue = val;
      ctx.level = lvl;
      globalRegistry.registerVar(ctx);
    }
  }

  // 3. Recursive processing: Execute Lowering for all levels, converging
  // variables to Level 0
  processLevelRecursive(manager.getTopLevelScope(), manager, rewriter, bi,
                        globalRegistry);

  // 4. Final analysis on Top Level (Level 0) to generate Suppression Branch for
  // the current connection
  auto l0Graph =
      manager.extractLayeredCFG(manager.getTopLevelScope(), rewriter);
  ControlDependenceAnalysis locCDA(*l0Graph->region);

  // Compute Level 0 control dependencies
  DenseSet<Block *> locConsControlDepsTmp =
      locCDA.getAllBlockDeps()[l0Graph->newCons].allControlDeps;

  DenseSet<Block *> locConsControlDeps;
  for (Block *nb : locConsControlDepsTmp) {
    Block *orig = l0Graph->origMap.lookup(nb);
    if (orig)
      locConsControlDeps.insert(orig);
  }

  BoolExpression *fCons = enumeratePaths(*l0Graph, bi, locConsControlDeps);

  if (debuglog) {
    out << "fCons-no-mux  = " << fCons->toString() << "\n";
  }

  if (accountMuxCondition) {
    Block *muxConditionBlock = returnMuxConditionBlock(muxCondition);
    BoolExpression *selectOperandCondition =
        BoolExpression::parseSop(bi.getBlockCondition(muxConditionBlock));

    // Note: Comparing blocks in the original CFG
    if (!bi.isLess(muxConditionBlock, producerBlock)) {
      if (consumer->getOperand(1) == connection) {
        fCons = BoolExpression::boolAnd(fCons,
                                        selectOperandCondition->boolNegate());
      } else {
        fCons = BoolExpression::boolAnd(fCons, selectOperandCondition);
      }
    }
  }

  // Calculate fSup
  BoolExpression *fSup = fCons->boolNegate();
  fSup = fSup->boolMinimize();

  if (debuglog) {
    out << "fSupmin = " << fSup->toString() << "\n";
  }

  // 5. Build Final Circuit
  if (fSup->type != experimental::boolean::ExpressionType::Zero) {
    std::set<std::string> blocks = fSup->getVariables();
    std::vector<std::string> cofactorList(blocks.begin(), blocks.end());

    // Sort Cofactor
    std::sort(cofactorList.begin(), cofactorList.end(),
              [&](const std::string &a, const std::string &b) {
                StringRef nameA = a;
                if (nameA.startswith("~"))
                  nameA = nameA.drop_front();
                StringRef nameB = b;
                if (nameB.startswith("~"))
                  nameB = nameB.drop_front();
                auto idA = bi.getBlockFromCondition(nameA.str());
                auto idB = bi.getBlockFromCondition(nameB.str());
                if (!idA || !idB)
                  return a < b;
                return bi.isLess(idA.value(), idB.value());
              });

    BDD *bdd = buildBDD(fSup, cofactorList);

    // Prepare Top Level Registry (containing Lowered variables)
    SignalRegistry topLevelRegistry;
    for (const auto &v : globalRegistry.getVariablesForLevel(0)) {
      topLevelRegistry.registerSignal(v.name, {}, v.currentValue);
    }

    // Build Distribution Network (for this connection)
    buildDistributionNetwork(rewriter, bdd, consumer->getBlock(), bi,
                             topLevelRegistry);
    Value branchCond = bddToCircuit(rewriter, bdd, consumer->getBlock(),
                                    topLevelRegistry, {}, bi);

    // Insert Suppression Branch
    rewriter.setInsertionPointToStart(consumer->getBlock());
    auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        consumer->getLoc(), ftd::getListTypes(connection.getType()), branchCond,
        connection);

    for (auto &use : connection.getUses()) {
      if (use.getOwner() != consumer)
        continue;
      if (llvm::isa<handshake::MuxOp>(consumer) && use.getOperandNumber() == 0)
        continue;
      use.set(branchOp.getFalseResult());
    }
  }
  // rewriter.eraseOp(locGraph->containerOp);
}

void ftd::addRegenOperandConsumer(PatternRewriter &rewriter,
                                  handshake::FuncOp &funcOp,
                                  Operation *consumerOp, Value operand) {

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));
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

    // Get the condition for the block exiting
    Value conditionValue =
        loop->getExitingBlock()->getTerminator()->getOperand(0);

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
  if (llvm::isa<handshake::ConditionalBranchOp>(consumerOp))
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

    // A conditional branch should undergo the suppression mechanism only if it
    // has the `FTD_NEW_SUPP` annotation, set in `addMoreSuppressionInLoop`. In
    // any other cases, suppressing a branch ends up with incorrect results.
    if (llvm::isa<handshake::ConditionalBranchOp>(producerOp) &&
        !producerOp->hasAttr(FTD_NEW_SUPP))
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

      // The condition value is provided by the `condition` field of the phi
      Value conditionValue =
          gate->conditionBlock->getTerminator()->getOperand(0);

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
