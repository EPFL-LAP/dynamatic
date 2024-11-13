//===- GSAAnalysis.cpp - GSA analyis utilities ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements some utility functions useful towards converting the static single
// assignment (SSA) representation into gated single assignment representation
// (GSA).
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/GSAAnalysis.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "vector"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "gsa"

using namespace mlir;
using namespace mlir::func;
using namespace dynamatic;
using namespace dynamatic::experimental::ftd;
using namespace dynamatic::experimental::boolean;

Block *experimental::gsa::GSAAnalysis::getBlockFromIndex(unsigned index) {
  for (auto const &[b, i] : indexPerBlock) {
    if (index == i)
      return b;
  }
  return nullptr;
}

experimental::gsa::GSAAnalysis::GSAAnalysis(Operation *operation) {

  // Only one function should be present in the module, excluding external
  // functions
  unsigned functionsCovered = 0;

  // The analysis can be instantiated either over a module containing one
  // function only or over a function
  if (ModuleOp modOp = dyn_cast<ModuleOp>(operation); modOp) {
    for (func::FuncOp funcOp : modOp.getOps<func::FuncOp>()) {

      // Skip if external
      if (funcOp.isExternal())
        continue;

      // Analyze the function
      if (!functionsCovered) {
        inputOp = funcOp;
        convertSSAToGSA(funcOp);
        functionsCovered++;
      } else {
        llvm::errs() << "[GSA] Too many functions to handle in the module";
      }
    }
  } else if (func::FuncOp fOp = dyn_cast<func::FuncOp>(operation); fOp) {
    convertSSAToGSA(fOp);
    functionsCovered = 1;
  }

  // report an error indicating that the analysis is instantiated over
  // an inappropriate operation
  if (functionsCovered != 1)
    llvm::errs() << "[GSA] GSAAnalysis failed due to a wrong input type\n";
};

experimental::gsa::Gate *experimental::gsa::GSAAnalysis::expandGammaTree(
    ListExpressionsPerGate &expressions, std::queue<unsigned> &conditions,
    Gate *originalPhi) {

  // At each iteration, we want to use a cofactor that is present in all the
  // expressions in `expressions`. Since the cofactors are ordered according to
  // the basic block number, we can say that if a cofactor is present in one
  // expression, then it must be present in all the others, since they all have
  // the blocks associated to that cofactor as common dominator.
  unsigned indexToUse;
  std::string conditionToUse;
  while (true) {
    indexToUse = conditions.front();
    conditionToUse = "c" + std::to_string(indexToUse);
    conditions.pop();
    unsigned cofactorUsage =
        std::count_if(expressions.begin(), expressions.end(),
                      [&](std::pair<BoolExpression *, GateInput *> g) {
                        return g.first->containsMintern(conditionToUse);
                      });

    // None is using that cofactor, go to the next one
    if (cofactorUsage == 0)
      continue;

    // All are using the cofactor, expand the expressions over it
    if (cofactorUsage == expressions.size())
      break;

    llvm_unreachable("A cofactor is used only by some expressions");
  }

  ListExpressionsPerGate conditionsFalseExpressions;
  ListExpressionsPerGate conditionsTrueExpressions;

  // For each expression
  for (const auto &expression : expressions) {

    // Restrict an expression according to a condition ans an input boolean
    // value and return it
    auto getRestrictedExpression = [&](bool value) -> BoolExpression * {
      BoolExpression *expr = expression.first->deepCopy();
      restrict(expr, conditionToUse, value);
      return expr->boolMinimize();
    };

    // Substitute a condition in the expression both with value true and false
    BoolExpression *exprTrue = getRestrictedExpression(true);
    BoolExpression *exprFalse = getRestrictedExpression(false);

    // One of the two expressions above will be zero: add the input to the
    // corresponding list
    if (exprTrue->type != ExpressionType::Zero)
      conditionsTrueExpressions.emplace_back(exprTrue, expression.second);
    if (exprFalse->type != ExpressionType::Zero)
      conditionsFalseExpressions.emplace_back(exprFalse, expression.second);
  }

  SmallVector<GateInput *> operandsGamma(2);

  // If the number of non-null expressions obtained with cofactor = X is greater
  // than 1, then the expansions must be done again over those inputs with a new
  // cofactor. The resulting gamma is going to be used as X input of this
  // gamma.
  //
  // If only one element is present, that same element is used as input of the
  // gamma.
  //
  // If no elements are present, then that input will never be used, and it is
  // considered as empty.
  auto setGammaOperand = [&](int input, ListExpressionsPerGate &list) -> void {
    if (list.size() > 1) {
      Gate *gamma = expandGammaTree(list, conditions, originalPhi);
      operandsGamma[input] = new GateInput(gamma);
      gateInputList.push_back(operandsGamma[input]);
    } else if (list.size() == 1) {
      operandsGamma[input] = list[0].second;
    } else {
      operandsGamma[input] = new GateInput();
      gateInputList.push_back(operandsGamma[input]);
    }
  };

  setGammaOperand(1, conditionsTrueExpressions);
  setGammaOperand(0, conditionsFalseExpressions);

  // Create a new gamma and add it to the list of phis for the original basic
  // block.

  // Get the index of the condition (it is associated to a basic block in
  // "indexPerBlock" mapping)
  Gate *newGate =
      new Gate(originalPhi->result, operandsGamma, GateType::GammaGate,
               ++uniqueGateIndex, getBlockFromIndex(indexToUse));
  gatesPerBlock[originalPhi->getBlock()].push_back(newGate);

  return newGate;
}

void experimental::gsa::GSAAnalysis::mapBlocksToIndex(func::FuncOp &funcOp) {
  mlir::DominanceInfo domInfo;

  // Create a vector with all the blocks
  SmallVector<Block *> allBlocks;
  for (Block &bb : funcOp.getBlocks())
    allBlocks.push_back(&bb);

  // Sort the vector according to the dominance information
  std::sort(allBlocks.begin(), allBlocks.end(),
            [&](Block *a, Block *b) { return domInfo.dominates(a, b); });

  // Associate a smalled index in the map to the blocks at higer levels of the
  // dominance tree
  unsigned bbIndex = 0;
  for (Block *bb : allBlocks)
    indexPerBlock.insert({bb, bbIndex++});
}

void experimental::gsa::GSAAnalysis::convertSSAToGSA(func::FuncOp &funcOp) {

  if (funcOp.getBlocks().size() == 1)
    return;

  // Associate an index to each basic block in "funcOp" so that if Bi
  // dominates Bj than i < j
  mapBlocksToIndex(funcOp);

  // This function works in two steps. First, all the block arguments in the
  // IR are converted into PHIs, taking care or properly extracting the
  // information about the producers of the operands. Then, the phis are
  // converted either to GAMMAs or MUs.

  // The input of a phi might be another gate. This is the case when the
  // input of a phi is a block argument from a block with index different
  // from 0. In this situation, we mark the phi input as "missing", and we
  // store the necessary information (block argument) to later reconstruct
  // the relationship.
  struct MissingPhi {

    // Which input is missing
    GateInput *pi;

    // Related block argument
    BlockArgument blockArg;

    MissingPhi(GateInput *pi, BlockArgument blockArg)
        : pi(pi), blockArg(blockArg) {}
  };

  // Initialize the index of the class
  uniqueGateIndex = 0;

  // Vector to store all the missing phi
  SmallVector<MissingPhi> phisToConnect;

  // For each block in the function
  for (Block &block : funcOp.getBlocks()) {

    // Create an empty list for the phi functions corresponding to the block
    gatesPerBlock.insert({&block, llvm::SmallVector<Gate *>()});

    // For each block argument
    for (BlockArgument &arg : block.getArguments()) {
      unsigned argNumber = arg.getArgNumber();
      // Create a set for the operands of the corresponding phi function
      SmallVector<GateInput *> operands;
      DenseSet<Block *> coveredPredecessors;
      // For each predecessor of the block, which is in charge of
      // providing the inputs of the phi functions
      for (Block *pred : block.getPredecessors()) {

        // Make sure that a predecessor is covered only once
        if (coveredPredecessors.contains(pred))
          continue;
        coveredPredecessors.insert(pred);

        // Get the branch terminator
        auto branchOp = dyn_cast<BranchOpInterface>(pred->getTerminator());
        assert(branchOp && "Expected terminator operation in a predecessor "
                           "block feeding a block argument!");

        // Check if the input value "c" of type Value is already present among
        // the operands of the phi function
        auto isAlreadyPresent = [&](Value c) -> bool {
          return std::any_of(operands.begin(), operands.end(),
                             [c](GateInput *in) {
                               return in->isTypeValue() && in->getValue() == c;
                             });
        };

        // For each alternative in the branch terminator
        for (auto [successorId, successorBlock] :
             llvm::enumerate(branchOp->getSuccessors())) {

          // Skip the successor if it is not the block under analysis
          if (successorBlock != &block)
            continue;

          // Get the values used for that branch
          auto successorOperands = branchOp.getSuccessorOperands(successorId);
          // Get the value used as input of the gate
          Value producer = successorOperands[argNumber];
          GateInput *gateInput = nullptr;

          /// If it is a block argument whose parent block has some predecessor,
          /// then the value is the output of a phi, and we add it to the list
          /// of missing phis. Otherwise, the input is a value, and it can be
          /// safely added directly.
          if (BlockArgument blockArg = dyn_cast<BlockArgument>(producer);
              blockArg && !producer.getParentBlock()->hasNoPredecessors()) {
            gateInput = new GateInput((Gate *)nullptr);
            phisToConnect.push_back(MissingPhi(gateInput, blockArg));
            gateInputList.push_back(gateInput);
          } else {
            if (!isAlreadyPresent(dyn_cast<Value>(producer))) {
              gateInput = new GateInput(producer);
              gateInputList.push_back(gateInput);
            }
          }

          // Insert the value among the inputs of the phi
          if (gateInput)
            operands.push_back(gateInput);

          break;
        }
      }

      // If the list of operands is not empty (i.e. the phi has at least
      // one input), add it to the phis associated to that block
      if (!operands.empty()) {
        Gate *newPhi =
            new Gate(arg, operands, GateType::PhiGate, ++uniqueGateIndex);
        gatesPerBlock[&block].push_back(newPhi);
      }
    }
  }

  // Find the missing phi and correct the pointers
  for (MissingPhi &missing : phisToConnect) {
    auto list = gatesPerBlock[missing.blockArg.getParentBlock()];
    Gate **foundGate = std::find_if(list.begin(), list.end(),
                                    [&](Gate *&t) {
                                      return t->getArgumentNumber() ==
                                             missing.blockArg.getArgNumber();
                                    }

    );
    assert(foundGate != list.end() && "[GSA] Not found phi to reconnect");
    missing.pi->input = *foundGate;
  }

  convertPhiToMu(funcOp);
  convertPhiToGamma(funcOp);
  printAllGates();
}

void experimental::gsa::GSAAnalysis::convertPhiToGamma(func::FuncOp &funcOp) {

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));

  // For each block
  for (auto const &[phiBlock, phis] : gatesPerBlock) {

    // For each phi
    for (Gate *phi : phis) {

      // Skip if the phi is not of type `Phi`
      if (phi->gsaGateFunction != PhiGate)
        continue;

      // Sort the operands of the phi so that Bi comes before Bj if Bi dominates
      // Bj
      SmallVector<GateInput *> phiOperands = phi->operands;
      llvm::sort(phiOperands.begin(), phiOperands.end(),
                 [&](GateInput *a, GateInput *b) {
                   return domInfo.dominates(a->getBlock(), b->getBlock());
                 });

      // Find the nearest common dominator among all the blocks involved
      // in the phi inputs. Also add all the blocks to the list of 'blocks to
      // avoid'
      std::vector<Block *> blocksToAvoid = {phiOperands[0]->getBlock()};
      Block *commonDominator = phiOperands[0]->getBlock();
      for (GateInput *operand : llvm::drop_begin(phiOperands)) {
        Block *bb = operand->getBlock();
        commonDominator =
            domInfo.findNearestCommonDominator(bb, commonDominator);
        blocksToAvoid.push_back(bb);
      }

      // When traversing a path, some blocks have a condition which steers the
      // control flow execution. This set stores the list of the blocks
      // accordingly. They will become boolean conditions afterwards
      DenseSet<unsigned> blocksWithConditionInPath;

      // Vector associating each input of the Phi to a boolean expression
      std::vector<std::pair<BoolExpression *, GateInput *>> expressionsList;

      // For each input of the phi, compute the boolean expression which defines
      // its usage
      for (GateInput *operand : phiOperands) {

        // Remove the current operand from the list of blocks to avoid
        blocksToAvoid.erase(blocksToAvoid.begin());

        // Find all the paths from `commonDominator` to `phiBlock` which pass
        // through operand's block but not through any of the `blocksToAvoid`
        auto paths = findAllPaths(commonDominator, phiBlock,
                                  operand->getBlock(), blocksToAvoid);

        BoolExpression *phiInputCondition = BoolExpression::boolZero();

        // Sum all the conditions for each path
        for (std::vector<Block *> &path : paths) {
          boolean::BoolExpression *condition =
              getPathExpression(path, blocksWithConditionInPath, indexPerBlock);
          phiInputCondition =
              BoolExpression::boolOr(condition, phiInputCondition);
          phiInputCondition = phiInputCondition->boolMinimize();
        }

        // Associate the expression to the phi operand
        expressionsList.emplace_back(phiInputCondition, operand);
      }

      // Move the indexes of the blocks associated to the conditions into a
      // queue. Since the set was ordered, the queue will contain the indexes
      // ordered in ascending order as well
      std::queue<unsigned> conditionsOrdered;
      for (unsigned &index : blocksWithConditionInPath)
        conditionsOrdered.push(index);

      // Expand the expressions to get the tree of gammas
      Gate *gammaRoot =
          expandGammaTree(expressionsList, conditionsOrdered, phi);
      gammaRoot->isRoot = true;

      // Once that a phi has been converted into a tree of gammas, all the
      // gates which used the original phi as input must be connected to the
      // root of the gamma tree
      for (auto &[bb, phis] : gatesPerBlock) {
        for (Gate *phii : phis) {
          for (GateInput *op : phii->operands) {
            if (op->isTypeGate() && op->getGate() == phi)
              op->input = gammaRoot;
          }
        }
      }
    }
  }
}

void experimental::gsa::GSAAnalysis::convertPhiToMu(func::FuncOp &funcOp) {

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));

  // For each phi
  for (const std::pair<Block *, SmallVector<Gate *>> &entry : gatesPerBlock) {
    Block *phiBlock = entry.first;
    SmallVector<Gate *> phis = entry.second;
    for (Gate *phi : phis) {

      // A phi might be a MU iff it is inside a for loop and has exactly
      // two operands
      if (!loopInfo.getLoopFor(phiBlock) || phi->operands.size() != 2)
        continue;

      Block *op0Block = phi->operands[0]->getBlock(),
            *op1Block = phi->operands[1]->getBlock();

      // Checks whether the block of the merge is a loop header
      bool isBlockHeader =
          loopInfo.getLoopFor(phiBlock)->getHeader() == phiBlock;

      // Checks whether the two operands come from different loops (in
      // this case, one of the values is the initial definition)
      bool operandFromOutsideLoop =
          loopInfo.getLoopFor(op0Block) != loopInfo.getLoopFor(op1Block);

      // If both the conditions hold, then we have a MU gate
      if (!(isBlockHeader && operandFromOutsideLoop))
        continue;

      phi->gsaGateFunction = GateType::MuGate;

      // Use the initial value of MU as first input of the gate
      if (domInfo.dominates(op1Block, phiBlock))
        std::swap(phi->operands[0], phi->operands[1]);

      // The block determining the MU condition is the exiting block of the
      // innermost loop the MU is in
      phi->conditionBlock =
          loopInfo.getLoopFor(phi->getBlock())->getExitingBlock();
      phi->isRoot = true;
    }
  }
}

ArrayRef<experimental::gsa::Gate *>
experimental::gsa::GSAAnalysis::getGates(Block *bb) const {
  auto it = gatesPerBlock.find(bb);
  return it == gatesPerBlock.end() ? ArrayRef<Gate *>() : it->getSecond();
}

void experimental::gsa::GSAAnalysis::printAllGates() {
  for (auto const &[_, gates] : gatesPerBlock) {
    for (Gate *g : gates)
      g->print();
  }
}

experimental::gsa::GSAAnalysis::~GSAAnalysis() {
  for (GateInput *gi : gateInputList)
    delete gi;
  for (auto const &[_, gates] : gatesPerBlock) {
    for (Gate *g : gates)
      delete g;
  }
}

void experimental::gsa::Gate::print() {

  LLVM_DEBUG(
      auto getPhiName = [](Gate *p) -> std::string {
        switch (p->gsaGateFunction) {
        case GammaGate:
          return "GAMMA";
        case MuGate:
          return "MU";
        default:
          return "PHI";
        }
      };

      llvm::dbgs() << "[GSA] Block "; getBlock()->printAsOperand(llvm::dbgs());
      llvm::dbgs() << " arg " << getArgumentNumber() << " type "
                   << getPhiName(this) << "_" << index;

      if (gsaGateFunction == GammaGate || gsaGateFunction == MuGate) {
        llvm::dbgs() << " condition ";
        conditionBlock->printAsOperand(llvm::dbgs());
      }

      llvm::dbgs()
      << "\n";

      for (GateInput *&op : operands) {
        if (op->isTypeValue()) {
          llvm::dbgs() << "[GSA]\t VALUE\t: ";
          op->getValue().print(llvm::dbgs());
        } else if (op->isTypeEmpty()) {
          llvm::dbgs() << "[GSA]\t EMPTY";
        } else {
          llvm::dbgs() << "[GSA]\t GATE\t: " << getPhiName(op->getGate()) << "_"
                       << op->getGate()->index;
        }

        if (!op->isTypeEmpty()) {
          llvm::dbgs() << "\t(";
          op->getBlock()->printAsOperand(llvm::dbgs());
          llvm::dbgs() << ")";
        }
        llvm::dbgs() << "\n";
      });
}

Block *experimental::gsa::GateInput::getBlock() {
  if (isTypeEmpty())
    return nullptr;
  return isTypeGate() ? getGate()->getBlock() : getValue().getParentBlock();
}
