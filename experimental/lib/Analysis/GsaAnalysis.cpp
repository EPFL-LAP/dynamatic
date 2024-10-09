//===- GsaAnalysis.cpp - GSA analyis utilities ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements some utility functions useful towards converting the static single
// assignment (SSA) representation into gated single assingment representation
// (GSA).
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/GsaAnalysis.h"
#include "experimental/Support/BooleanLogic/Shannon.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "unordered_set"
#include "vector"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::func;

namespace dynamatic {
namespace experimental {
namespace gsa {

/// Get the index of a basic block
static int getBlockIndex(Block *bb) {
  std::string result1;
  llvm::raw_string_ostream os1(result1);
  bb->printAsOperand(os1);
  std::string block1id = os1.str();
  return std::stoi(block1id.substr(3));
}

/// Given a block whose name is `^BBN` (where N is an integer) return a string
/// in the format `cN`, used to identify the condition which allows the block
/// to be executed.
static std::string getBlockCondition(Block *block) {
  std::string blockCondition = "c" + std::to_string(getBlockIndex(block));
  return blockCondition;
}

/// Recursive function which allows to obtain all the paths from block `start`
/// to block `end` using a DFS. We are interested in paths which go through
/// `blockToTravers` but don't go through any of `blocksToAvoid`
static void dfsPathsWithConditions(Block *start, Block *end,
                                   std::vector<Block *> &path,
                                   std::unordered_set<Block *> &visited,
                                   std::vector<std::vector<Block *>> &allPaths,
                                   Block *blockToTraverse,
                                   std::vector<Block *> &blocksToAvoid,
                                   bool blockToTraverseFound) {

  // The current block is part of the current path
  path.push_back(start);

  // The current block has been visited
  visited.insert(start);

  // Mark the target block as found if it is identical to block to traverse
  bool blockFound = (start == blockToTraverse);

  // If we are at the end of the path and we have traversed the target one, then
  // add it to the list
  if (start == end && blockToTraverseFound) {
    allPaths.push_back(path);
  } else {

    // Else, for each successor which was not visited, run DFS again
    for (Block *successor : start->getSuccessors()) {

      // Do not run DFS if the successor is in the list of blocks to traverse
      bool incorrectPath = false;
      for (auto *toAvoid : blocksToAvoid) {
        if (toAvoid == successor &&
            getBlockIndex(toAvoid) > getBlockIndex(blockToTraverse)) {
          incorrectPath = true;
          break;
        }
      }

      if (incorrectPath)
        continue;

      // Do not run DFS if the successor was already visited
      if (visited.find(successor) == visited.end()) {
        dfsPathsWithConditions(successor, end, path, visited, allPaths,
                               blockToTraverse, blocksToAvoid,
                               blockFound || blockToTraverseFound);
      }
    }
  }

  // Remove the current block from the current path and from the list of
  // visited blocks
  path.pop_back();
  visited.erase(start);
}

/// Get the boolean condition determining when a path is executed. While
/// covering each block in the path, add the cofactor of each block to the list
/// of cofactors if not already covered
static boolean::BoolExpression *
getPathCondition(const std::vector<Block *> &path,
                 std::vector<std::string> &cofactorList) {

  // Start with a boolean expression of one
  boolean::BoolExpression *exp = boolean::BoolExpression::boolOne();

  // Cover each pair of adjacent blocks
  for (int i = 0; i < (int)path.size() - 1; i++) {
    Block *firstBlock = path[i];
    Block *secondBlock = path[i + 1];

    // Skip pair if the first block has only one successor, thus no conditional
    // branch
    if (firstBlock->getSuccessors().size() == 1)
      continue;

    // Get last operation of the block, also called `terminator`
    Operation *terminatorOp = firstBlock->getTerminator();

    // TODO: use a general model for this same conditional branch
    if (isa<cf::CondBranchOp>(terminatorOp)) {
      auto blockCondition = getBlockCondition(firstBlock);

      // Get a boolean condition out of the block condition
      boolean::BoolExpression *pathCondition =
          boolean::BoolExpression::parseSop(blockCondition);

      // Possibly add the condition to the list of cofactors
      if (std::find(cofactorList.begin(), cofactorList.end(), blockCondition) ==
          cofactorList.end()) {
        cofactorList.push_back(blockCondition);
      }

      // Negate the condition if `secondBlock` is reached when the condition is
      // false
      auto condOp = dyn_cast<cf::CondBranchOp>(terminatorOp);
      if (condOp.getFalseDest() == secondBlock)
        pathCondition->boolNegate();

      // And the condition with the rest path
      exp = boolean::BoolExpression::boolAnd(exp, pathCondition);
    }
  }

  // Minimize the condition and return
  return exp->boolMinimize();
}

template <typename FunctionType>
Phi *GsaAnalysis<FunctionType>::expandExpressions(
    std::vector<std::pair<boolean::BoolExpression *, PhiInput *>> &expressions,
    std::vector<std::string> &cofactors, Phi *originalPhi) {

  // At each iteration, we want to use a cofactor that is present in all the
  // expressions in `expressions`. Since the cofactors are ordered according to
  // the basic block number, we can say that if a cofactor is present in one
  // expression, then it must be present in all the others, since they all have
  // the blocks associated to that cofactor as common dominator.
  std::string cofactorToUse;
  while (true) {
    cofactorToUse = cofactors.front();
    cofactors.erase(cofactors.begin());
    unsigned cofactorUsage = 0;
    for (auto expr : expressions) {
      if (expr.first->containsMintern(cofactorToUse))
        cofactorUsage++;
    }

    // None is using that cofactor, then go to the next one
    if (cofactorUsage == 0)
      continue;

    // All are using the cofactor
    if (cofactorUsage == expressions.size())
      break;

    assert(false && "A cofactor is used only by some expressions");
  }

  std::vector<std::pair<boolean::BoolExpression *, PhiInput *>>
      cofactorFalseExpressions;
  std::vector<std::pair<boolean::BoolExpression *, PhiInput *>>
      cofactorTrueExpressions;

  // For each expression
  for (auto expression : expressions) {

    // Substitute the cofactor with `true`
    auto *exprTrue = boolean::shannonExpansionPositive(
        expression.first->deepCopy(), cofactorToUse);
    // Substitute the cofactor with `false`
    auto *exprFalse = boolean::shannonExpansionNegative(
        expression.first->deepCopy(), cofactorToUse);

    // One of the two expressions above will be zero: add the input to the other
    // list
    if (exprTrue->toString() != "0")
      cofactorTrueExpressions.emplace_back(exprTrue, expression.second);
    if (exprFalse->toString() != "0")
      cofactorFalseExpressions.emplace_back(exprFalse, expression.second);
  }

  SmallVector<PhiInput *> operandsGamma(2);

  // If the number of non-null expressions obtained with cofactor = 1 is greater
  // than 1, then the expansions must be done again over those inputs with a new
  // cofactor. The resulting gamma is going to be used as `true` input of this
  // gamma.
  //
  // If only one element is present, that same element is used as input of the
  // gamma.
  //
  // If no elements are present, then that input will never be used, and it is
  // considered as empty.
  if (cofactorTrueExpressions.size() > 1) {
    auto *trueGamma =
        expandExpressions(cofactorTrueExpressions, cofactors, originalPhi);
    auto *phiInput = new PhiInput(trueGamma, originalPhi->blockOwner);
    operandsGamma[1] = phiInput;

  } else if (cofactorTrueExpressions.size() == 1) {
    operandsGamma[1] = cofactorTrueExpressions[0].second;
  } else {
    auto *phiInput = new PhiInput();
    operandsGamma[1] = phiInput;
  }

  // If the number of non-null expressions obtained with cofactor = 0 is greater
  // than 1, then the expansions must be done again over those inputs with a new
  // cofactor. The resulting gamma is going to be used as `false` input of this
  // gamma.
  //
  // If only one element is present, that same element is used as input of the
  // gamma.
  //
  // If no elements are present, then that input will never be used, and it is
  // considered as empty.
  if (cofactorFalseExpressions.size() > 1) {
    auto *falseGamma =
        expandExpressions(cofactorFalseExpressions, cofactors, originalPhi);
    auto *phiInput = new PhiInput(falseGamma, originalPhi->blockOwner);
    operandsGamma[0] = phiInput;
  } else if (cofactorFalseExpressions.size() == 1) {
    operandsGamma[0] = cofactorFalseExpressions[0].second;
  } else {
    auto *phiInput = new PhiInput();
    operandsGamma[0] = phiInput;
  }

  // Create a new gamma and add it to the list of phis for the original basic
  // block.
  auto *newPhi = new Phi(nullptr, originalPhi->argNumber, operandsGamma,
                         originalPhi->blockOwner, GsaGateFunction::GammaGate,
                         cofactorToUse);
  phiList[originalPhi->blockOwner].push_back(newPhi);
  newPhi->index = ++uniquePhiIndex;
  newPhi->result = originalPhi->result;

  return newPhi;
}

/// Gets all the paths from block `start` to block `end` using a dfs
/// search
static std::vector<std::vector<Block *>>
findAllPaths(Block *start, Block *end, Block *blockToTraverse,
             std::vector<Block *> blocksToAvoid) {
  std::vector<std::vector<Block *>> allPaths;
  std::vector<Block *> path;
  std::unordered_set<Block *> visited;
  dfsPathsWithConditions(start, end, path, visited, allPaths, blockToTraverse,
                         blocksToAvoid, false);
  return allPaths;
}

template <typename FunctionType>
void GsaAnalysis<FunctionType>::identifyAllPhi(FunctionType &funcOp) {

  // The input of a phi might be another phi. This is the case when the
  // input of a phi is a block argument from a block with index different
  // from 0. In this situation, we mark the phi input as `missing`, and we
  // store the necessary information (block of the block argument and
  // argument index) to later reconstruct the relationship.
  struct MissingPhi {

    // Which input is missing
    PhiInput *pi;
    // Which is the block owning the phi function which will provide the
    // input
    Block *blockOwner;
    // Argument number of the missing phi
    unsigned argNumber;

    MissingPhi(PhiInput *pi, Block *blockOwner, unsigned argNumber)
        : pi(pi), blockOwner(blockOwner), argNumber(argNumber) {}
  };

  // Initialize the index of the class
  uniquePhiIndex = 0;

  // Vector to store all the missing phi
  SmallVector<MissingPhi> phiToConnect;

  // For each block in the function
  for (Block &block : funcOp.getBlocks()) {
    // Create a list for the phi functions corresponding to the block
    llvm::SmallVector<Phi *> phiListBlock;
    // For each block argument
    for (BlockArgument &arg : block.getArguments()) {
      unsigned argNumber = arg.getArgNumber();
      // Create a set for the operands of the corresponding phi function
      SmallVector<PhiInput *> operands;
      // For each predecessor of the block, which is in charge of
      // providing the inputs of the phi functions
      for (Block *pred : block.getPredecessors()) {
        // Get the branch terminator
        auto branchOp = dyn_cast<BranchOpInterface>(pred->getTerminator());
        assert(branchOp && "Expected terminator operation in a predecessor "
                           "block feeding a block argument!");
        // For each alternative in the branch terminator
        for (auto [successorId, successorBlock] :
             llvm::enumerate(branchOp->getSuccessors())) {
          // Get the one corresponding to the block containing the phi
          if (successorBlock == &block) {
            // Get the value used on that branch
            auto successorOperands = branchOp.getSuccessorOperands(successorId);
            // Get the corresponding producer/value
            auto producer = successorOperands[argNumber];
            PhiInput *phiInput = nullptr;
            // Try to convert the producer to a block argument
            BlockArgument ba = dyn_cast<BlockArgument>(producer);
            // If the producer is a BA but its block has no predecessor,
            // then it is a function argument. Otherwise, if it is a BA,
            // it must be connected to another phi. In all the other
            // situations, it comes from an operation.
            if (ba && producer.getParentBlock()->getPredecessors().empty()) {
              bool alreadyPresent = false;
              for (auto &phi : operands) {
                if (phi->type == ArgInputType && phi->v == ba)
                  alreadyPresent = true;
              }
              if (!alreadyPresent)
                phiInput = new PhiInput(ba, pred);

            } else if (ba) {
              // TODO double chcek that arg is used only once
              phiInput = new PhiInput((Phi *)nullptr, pred);
              phiToConnect.push_back(
                  MissingPhi(phiInput, ba.getParentBlock(), ba.getArgNumber()));

            } else {
              bool alreadyPresent = false;
              for (auto &phi : operands) {
                if (phi->type == OpInputType &&
                    phi->v == dyn_cast<Value>(producer))
                  alreadyPresent = true;
              }
              if (!alreadyPresent)
                phiInput = new PhiInput(dyn_cast<Value>(producer), pred);
            }

            // Insert the value among the inputs of the phi
            if (phiInput)
              operands.push_back(phiInput);

            break;
          }
        }
      }

      // If the list of operands is not empty (i.e. the phi has at least
      // one input), add it to the phis associated to that block
      if (!operands.empty()) {
        auto *newPhi =
            new Phi(arg, argNumber, operands, &block, GsaGateFunction::PhiGate);
        newPhi->index = ++uniquePhiIndex;
        phiListBlock.push_back(newPhi);
      }
    }

    // Associate the list of phis to the basic block
    phiList.insert({&block, phiListBlock});
  }

  // For each missing phi, look for it among the phis related to the
  // marked block and connect it
  for (auto &missing : phiToConnect) {
    Phi *foundPhi = nullptr;
    for (auto &phi : phiList[missing.blockOwner]) {
      if (phi->argNumber == missing.argNumber) {
        foundPhi = phi;
        break;
      }
    }
    assert(foundPhi && "[GSA] Not found phi to reconnect");
    missing.pi->phi = foundPhi;
  }

  convertPhiToMu(funcOp);
  convertPhiToGamma(funcOp);
  printPhiList();
}

template <typename FunctionType>
void GsaAnalysis<FunctionType>::convertPhiToGamma(FunctionType &funcOp) {

  if (funcOp.getBlocks().size() == 1)
    return;

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));

  // For each block
  for (auto const &[phiBlock, phis] : phiList) {

    // For each phi
    for (auto *phi : phis) {

      // Skip if the phi is not of type `Phi`
      if (phi->gsaGateFunction != PhiGate)
        continue;

      // Get the phi operands and sort them according to the index of the basic
      // block, so that input `i` comes before input `j` if
      // `index_Bi`<`index_Bj`
      auto phiOperands = phi->operands;

      llvm::sort(
          phiOperands.begin(), phiOperands.end(), [](PhiInput *a, PhiInput *b) {
            return getBlockIndex(a->getBlock()) < getBlockIndex(b->getBlock());
          });

      // Find the nearest common dominator among all the blocks involved
      // in the phi inputs
      Block *commonDominator = phiOperands[0]->getBlock();
      for (auto &operand : llvm::drop_begin(phiOperands))
        commonDominator = domInfo.findNearestCommonDominator(
            operand->getBlock(), commonDominator);

      std::vector<Block *> blocksToAvoid;

      // Add all the inputs to the blocks to avoid
      for (auto &operand : phiOperands)
        blocksToAvoid.push_back(operand->getBlock());

      // List of cofactors present in all the expressions for all the paths
      std::vector<std::string> cofactorList;

      // Vector associating each input of the Phi to a boolean expression
      std::vector<std::pair<boolean::BoolExpression *, PhiInput *>>
          expressionsList;

      // For each input of the phi, compute the boolean expression which defines
      // its usage
      for (auto &operand : phiOperands) {

        // Remove the current operand from the list of blocks to avoid
        blocksToAvoid.erase(blocksToAvoid.begin());

        // Find all the paths from `commonDominator` to `phiBlock` which pass
        // through operand's block but not through any of the `blocksToAvoid`
        auto paths = findAllPaths(commonDominator, phiBlock,
                                  operand->getBlock(), blocksToAvoid);

        boolean::BoolExpression *phiInputCondition =
            boolean::BoolExpression::boolZero();

        // Sum all the conditions for each path
        for (auto &path : paths) {
          auto *condition = getPathCondition(path, cofactorList);
          phiInputCondition =
              boolean::BoolExpression::boolOr(condition, phiInputCondition);
          phiInputCondition = phiInputCondition->boolMinimize();
        }

        // Associate the expression to the phi operand
        expressionsList.emplace_back(phiInputCondition, operand);
      }

      // Sort the cofactors according to their index
      llvm::sort(cofactorList.begin(), cofactorList.end(),
                 [](std::string a, std::string b) {
                   a.erase(0, 1);
                   b.erase(0, 1);
                   return std::stoi(a) < std::stoi(b);
                 });

      // Expand the expressions to get the tree of gammas
      auto *newPhi = expandExpressions(expressionsList, cofactorList, phi);
      newPhi->isRoot = true;

      for (auto &[bb, phis] : phiList) {
        for (auto &phii : phis) {
          for (auto &op : phii->operands) {
            if (op->phi && op->phi == phi)
              op->phi = newPhi;
          }
        }
      }
    }
  }
}

template <typename FunctionType>
void GsaAnalysis<FunctionType>::convertPhiToMu(FunctionType &funcOp) {

  if (funcOp.getBlocks().size() == 1)
    return;

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));

  for (auto const &[phiBlock, phis] : phiList) {
    for (auto *phi : phis) {

      // A phi might be a MU iff it is inside a for loop and has exactly
      // two operands
      if (loopInfo.getLoopFor(phiBlock) && phi->operands.size() == 2) {

        auto *op0Block = phi->operands[0]->getBlock();
        auto *op1Block = phi->operands[1]->getBlock();

        // Checks whether the block of the merge is a loop header
        bool isBlockHeader =
            loopInfo.getLoopFor(phiBlock)->getHeader() == phiBlock;

        // Checks whether the two operands come from different loops (in
        // this case, one of the values is the initial definition)
        bool operandFromOutsideLoop =
            loopInfo.getLoopFor(op0Block) != loopInfo.getLoopFor(op1Block);

        // If both the conditions hold, then we have a MU gate
        if (isBlockHeader && operandFromOutsideLoop) {
          phi->gsaGateFunction = GsaGateFunction::MuGate;

          // Use the initial value of mu as first input of the gate
          if (domInfo.dominates(op1Block, phiBlock)) {
            auto *firstOperand = phi->operands[0];
            auto *secondOperand = phi->operands[1];
            phi->operands[0] = secondOperand;
            phi->operands[1] = firstOperand;
          }

          auto *terminator = loopInfo.getLoopFor(phi->blockOwner)
                                 ->getExitingBlock()
                                 ->getTerminator();
          phi->minterm = getBlockCondition(terminator->getBlock());
          phi->isRoot = true;
        }
      }
    }
  }
}

template <typename FunctionType>
SmallVector<Phi *> *GsaAnalysis<FunctionType>::getPhis(Block *bb) {
  if (!phiList.contains(bb))
    return nullptr;
  return &phiList[bb];
}

template <typename FunctionType>
void GsaAnalysis<FunctionType>::printPhiList() {
  for (auto const &[_, phis] : phiList) {
    for (auto *phi : phis) {
      phi->print();
    }
  }
}

void gsa::Phi::print() {

  auto getPhiName = [&](Phi *p) -> std::string {
    switch (p->gsaGateFunction) {
    case GammaGate:
      return "GAMMA";
    case MuGate:
      return "MU";
    case EtaGate:
      return "ETA";
    default:
      return "PHI";
    }
  };

  llvm::dbgs() << "[GSA] Block ";
  blockOwner->printAsOperand(llvm::dbgs());
  llvm::dbgs() << " arg " << argNumber << " type " << getPhiName(this) << "_"
               << index;

  if (gsaGateFunction == GammaGate || gsaGateFunction == MuGate) {
    llvm::dbgs() << " minterm " << minterm;
  }

  llvm::dbgs() << "\n";

  for (auto &op : operands) {
    switch (op->type) {
    case ArgInputType:
      llvm::dbgs() << "[GSA]\t ARG\t: ";
      op->v.print(llvm::dbgs());
      break;
    case OpInputType:
      llvm::dbgs() << "[GSA]\t OP\t: ";
      op->v.print(llvm::dbgs());
      break;
    case EmptyInputType:
      llvm::dbgs() << "[GSA]\t EMPTY";
      break;
    default:
      llvm::dbgs() << "[GSA]\t PHI\t: " << getPhiName(op->phi) << "_"
                   << op->phi->index;
      break;
    }

    if (op->type != EmptyInputType) {
      llvm::dbgs() << "\t(";
      op->getBlock()->printAsOperand(llvm::dbgs());
      llvm::dbgs() << ")";
    }
    llvm::dbgs() << "\n";
  }
}

Block *PhiInput::getBlock() { return this->blockOwner; }

} // namespace gsa
} // namespace experimental
} // namespace dynamatic

namespace dynamatic {

// Explicit template instantiation
template class experimental::gsa::GsaAnalysis<mlir::func::FuncOp>;

} // namespace dynamatic
