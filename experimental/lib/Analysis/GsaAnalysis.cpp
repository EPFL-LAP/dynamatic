//===- GsaAnalysis.cpp - GSA analyis utilities ------------------*- C++ -*-===//
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

#include "experimental/Analysis/GsaAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/IR/Dominance.h"
#include "vector"
#include "llvm/Support/Debug.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::func;
using namespace dynamatic::experimental::ftd;

namespace dynamatic {
namespace experimental {
namespace gsa {

template <typename FunctionType>
Gate *
GsaAnalysis<FunctionType>::expandGammaTree(ListExpressionsPerGate &expressions,
                                           std::vector<std::string> &conditions,
                                           Gate *originalPhi) {

  // At each iteration, we want to use a cofactor that is present in all the
  // expressions in `expressions`. Since the cofactors are ordered according to
  // the basic block number, we can say that if a cofactor is present in one
  // expression, then it must be present in all the others, since they all have
  // the blocks associated to that cofactor as common dominator.
  std::string conditionToUse;
  while (true) {
    conditionToUse = conditions.front();
    conditions.erase(conditions.begin());
    unsigned cofactorUsage =
        std::count_if(expressions.begin(), expressions.end(),
                      [&](std::pair<boolean::BoolExpression *, GateInput *> g) {
                        return g.first->containsMintern(conditionToUse);
                      });

    // None is using that cofactor, go to the next one
    if (cofactorUsage == 0)
      continue;

    // All are using the cofactor, expand the expressions over it
    if (cofactorUsage == expressions.size())
      break;

    assert(false && "A cofactor is used only by some expressions");
  }

  ListExpressionsPerGate conditionsFalseExpressions;
  ListExpressionsPerGate conditionsTrueExpressions;

  // For each expression
  for (auto expression : expressions) {

    // Restrict an expression according to a condition ans an input boolean
    // value and return it
    auto getRestrictedExpression =
        [&](bool value) -> boolean::BoolExpression * {
      auto *expr = expression.first->deepCopy();
      boolean::restrict(expr, conditionToUse, value);
      return expr->boolMinimize();
    };

    // Substitute a condition in the expression both with value true and false
    auto *exprTrue = getRestrictedExpression(true);
    auto *exprFalse = getRestrictedExpression(false);

    // One of the two expressions above will be zero: add the input to the
    // corresponding list
    if (exprTrue->toString() != "0")
      conditionsTrueExpressions.emplace_back(exprTrue, expression.second);
    if (exprFalse->toString() != "0")
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
      auto *gamma = expandGammaTree(list, conditions, originalPhi);
      operandsGamma[input] = new GateInput(gamma);
    } else if (list.size() == 1) {
      operandsGamma[input] = list[0].second;
    } else {
      operandsGamma[input] = new GateInput();
    }
  };

  setGammaOperand(1, conditionsTrueExpressions);
  setGammaOperand(0, conditionsFalseExpressions);

  // Create a new gamma and add it to the list of phis for the original basic
  // block.
  auto *newGate =
      new Gate(originalPhi->result, operandsGamma, GateType::GammaGate,
               ++uniqueGateIndex, conditionToUse);
  gateList[originalPhi->getBlock()].push_back(newGate);

  return newGate;
}

template <typename FunctionType>
void GsaAnalysis<FunctionType>::identifyAllGates(FunctionType &funcOp) {

  // This function works in two steps. First, all the block arguments in the IR
  // are converted into PHIs, taking care or properly extracting the information
  // about the producers of the operands. Then, the phis ar converted either to
  // GAMMAs or MUs.

  // The input of a phi might be another . This is the case when the
  // input of a phi is a block argument from a block with index different
  // from 0. In this situation, we mark the phi input as `missing`, and we
  // store the necessary information (block argument) to later reconstruct the
  // relationship.
  struct MissingPhi {

    // Which input is missing
    GateInput *pi;

    // Related block argument
    BlockArgument ba;

    MissingPhi(GateInput *pi, BlockArgument ba) : pi(pi), ba(ba) {}
  };

  // Initialize the index of the class
  uniqueGateIndex = 0;

  // Vector to store all the missing phi
  SmallVector<MissingPhi> phisToConnect;

  // For each block in the function
  for (Block &block : funcOp.getBlocks()) {

    // Create an empty list for the phi functions corresponding to the block
    gateList.insert({&block, llvm::SmallVector<Gate *>()});

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

        // Check if the input value `c` of type Value is already present among
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
          auto producer = successorOperands[argNumber];
          GateInput *gateInput = nullptr;

          /// If it is a block argument whose parent block has some predecessor,
          /// then the value is the output of a phi, and we add it to the list
          /// of missing phis. Otherwise, the input is a value, and it can be
          /// safely added directly.
          if (auto ba = dyn_cast<BlockArgument>(producer);
              ba && !producer.getParentBlock()->hasNoPredecessors()) {
            gateInput = new GateInput((Gate *)nullptr);
            phisToConnect.push_back(MissingPhi(gateInput, ba));
          } else {
            if (!isAlreadyPresent(dyn_cast<Value>(producer)))
              gateInput = new GateInput(producer);
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
        auto *newPhi =
            new Gate(arg, operands, GateType::PhiGate, ++uniqueGateIndex);
        gateList[&block].push_back(newPhi);
      }
    }
  }

  // Find the missing phi and correct the pointers
  for (MissingPhi &missing : phisToConnect) {
    auto list = gateList[missing.ba.getParentBlock()];
    auto foundGate = std::find_if(list.begin(), list.end(),
                                  [&](auto &t) {
                                    return t->getArgumentNumber() ==
                                           missing.ba.getArgNumber();
                                  }

    );
    assert(foundGate != list.end() && "[GSA] Not found phi to reconnect");
    missing.pi->input = *foundGate;
  }

  // Convert phis to MUs and GAMMAs
  convertPhiToMu(funcOp);
  convertPhiToGamma(funcOp);
  printGateList();
}

template <typename FunctionType>
void GsaAnalysis<FunctionType>::convertPhiToGamma(FunctionType &funcOp) {

  if (funcOp.getBlocks().size() == 1)
    return;

  mlir::DominanceInfo domInfo;
  mlir::CFGLoopInfo loopInfo(domInfo.getDomTree(&funcOp.getBody()));

  // For each block
  for (auto const &[phiBlock, phis] : gateList) {

    // For each phi
    for (auto *phi : phis) {

      // Skip if the phi is not of type `Phi`
      if (phi->gsaGateFunction != PhiGate)
        continue;

      // Get the phi operands and sort them according to the index of the basic
      // block, so that input `i` comes before input `j` if
      // `index_Bi`<`index_Bj`
      auto phiOperands = phi->operands;

      llvm::sort(phiOperands.begin(), phiOperands.end(),
                 [](GateInput *a, GateInput *b) {
                   return lessThanBlocks(a->getBlock(), b->getBlock());
                 });

      // Find the nearest common dominator among all the blocks involved
      // in the phi inputs. Also add all the blocks to the list of 'blocks to
      // avoid'
      std::vector<Block *> blocksToAvoid = {phiOperands[0]->getBlock()};
      Block *commonDominator = phiOperands[0]->getBlock();
      for (auto &operand : llvm::drop_begin(phiOperands)) {
        Block *bb = operand->getBlock();
        commonDominator =
            domInfo.findNearestCommonDominator(bb, commonDominator);
        blocksToAvoid.push_back(bb);
      }

      // List of conditions present in all the expressions for all the paths
      std::vector<std::string> conditionsList;

      // Vector associating each input of the Phi to a boolean expression
      std::vector<std::pair<boolean::BoolExpression *, GateInput *>>
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
          auto *condition = getPathExpression(path, conditionsList);
          phiInputCondition =
              boolean::BoolExpression::boolOr(condition, phiInputCondition);
          phiInputCondition = phiInputCondition->boolMinimize();
        }

        // Associate the expression to the phi operand
        expressionsList.emplace_back(phiInputCondition, operand);
      }

      // Sort the conditions according to their index
      llvm::sort(conditionsList.begin(), conditionsList.end(),
                 [](std::string a, std::string b) {
                   a.erase(0, 1);
                   b.erase(0, 1);
                   return std::stoi(a) < std::stoi(b);
                 });

      // Expand the expressions to get the tree of gammas
      auto *gammaRoot = expandGammaTree(expressionsList, conditionsList, phi);
      gammaRoot->isRoot = true;

      // Once that a phi has been converted into a tree of gammas, all the
      // gates which used the original phi as input must be connected to the
      // root of the gamma tree
      for (auto &[bb, phis] : gateList) {
        for (auto &phii : phis) {
          for (auto &op : phii->operands) {
            if (op->isTypeGate() && op->getGate() == phi)
              op->input = gammaRoot;
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

  // For each phi
  for (auto const &[phiBlock, phis] : gateList) {
    for (auto *phi : phis) {

      // A phi might be a MU iff it is inside a for loop and has exactly
      // two operands
      if (!loopInfo.getLoopFor(phiBlock) || phi->operands.size() != 2)
        continue;

      auto *op0Block = phi->operands[0]->getBlock(),
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

      // Use the initial value of mu as first input of the gate
      if (domInfo.dominates(op1Block, phiBlock))
        std::swap(phi->operands[0], phi->operands[1]);

      // The MU condition is given by the condition used for the termination
      // of the loop
      auto *terminator = loopInfo.getLoopFor(phi->getBlock())
                             ->getExitingBlock()
                             ->getTerminator();
      phi->condition = getBlockCondition(terminator->getBlock());
      phi->isRoot = true;
    }
  }
}

template <typename FunctionType>
SmallVector<Gate *> *GsaAnalysis<FunctionType>::getGates(Block *bb) {
  if (!gateList.contains(bb))
    return nullptr;
  return &gateList[bb];
}

template <typename FunctionType>
void GsaAnalysis<FunctionType>::printGateList() {
  for (auto const &[_, gates] : gateList) {
    for (auto *g : gates)
      g->print();
  }
}

void gsa::Gate::print() {

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

  llvm::dbgs() << "[GSA] Block ";
  getBlock()->printAsOperand(llvm::dbgs());
  llvm::dbgs() << " arg " << getArgumentNumber() << " type " << getPhiName(this)
               << "_" << index;

  if (gsaGateFunction == GammaGate || gsaGateFunction == MuGate) {
    llvm::dbgs() << " condition " << condition;
  }

  llvm::dbgs() << "\n";

  for (auto &op : operands) {
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
  }
}

Block *GateInput::getBlock() {
  return isTypeEmpty()  ? nullptr
         : isTypeGate() ? getGate()->getBlock()
                        : getValue().getParentBlock();
}

} // namespace gsa
} // namespace experimental
} // namespace dynamatic

namespace dynamatic {

// Explicit template instantiation
template class experimental::gsa::GsaAnalysis<mlir::func::FuncOp>;

} // namespace dynamatic
