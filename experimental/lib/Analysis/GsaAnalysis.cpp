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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/IR/Dominance.h"
#include "vector"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <any>

using namespace mlir;
using namespace mlir::func;
using namespace dynamatic::experimental::ftd;

namespace dynamatic {
namespace experimental {
namespace gsa {

template <typename FunctionType>
Gate *GsaAnalysis<FunctionType>::expandExpressions(
    std::vector<std::pair<boolean::BoolExpression *, GateInput *>> &expressions,
    std::vector<std::string> &cofactors, Gate *originalPhi) {

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

  std::vector<std::pair<boolean::BoolExpression *, GateInput *>>
      cofactorFalseExpressions;
  std::vector<std::pair<boolean::BoolExpression *, GateInput *>>
      cofactorTrueExpressions;

  // For each expression
  for (auto expression : expressions) {

    // Substitute the cofactor with `true`
    auto *exprTrue = expression.first->deepCopy();
    boolean::restrict(exprTrue, cofactorToUse, true);
    exprTrue = exprTrue->boolMinimize();
    // Substitute the cofactor with `false`
    auto *exprFalse = expression.first->deepCopy();
    boolean::restrict(exprFalse, cofactorToUse, false);
    exprFalse = exprFalse->boolMinimize();

    // One of the two expressions above will be zero: add the input to the other
    // list
    if (exprTrue->toString() != "0")
      cofactorTrueExpressions.emplace_back(exprTrue, expression.second);
    if (exprFalse->toString() != "0")
      cofactorFalseExpressions.emplace_back(exprFalse, expression.second);
  }

  SmallVector<GateInput *> operandsGamma(2);

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
    auto *phiInput = new GateInput(trueGamma);
    operandsGamma[1] = phiInput;

  } else if (cofactorTrueExpressions.size() == 1) {
    operandsGamma[1] = cofactorTrueExpressions[0].second;
  } else {
    auto *phiInput = new GateInput();
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
    auto *phiInput = new GateInput(falseGamma);
    operandsGamma[0] = phiInput;
  } else if (cofactorFalseExpressions.size() == 1) {
    operandsGamma[0] = cofactorFalseExpressions[0].second;
  } else {
    auto *phiInput = new GateInput();
    operandsGamma[0] = phiInput;
  }

  // Create a new gamma and add it to the list of phis for the original basic
  // block.
  auto *newGate = new Gate(originalPhi->result, operandsGamma,
                           GateType::GammaGate, cofactorToUse);
  gateList[originalPhi->getBlock()].push_back(newGate);
  newGate->index = ++uniqueGateIndex;
  newGate->result = originalPhi->result;

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
  // store the necessary information (block owner of the operand and relative
  // argument number) to later reconstruct the relationship.
  struct MissingPhi {

    // Which input is missing
    GateInput *pi;

    // Block owner of the phi providing the result
    Block *blockOwner;

    // Argument number of the missing phi
    unsigned argNumber;

    MissingPhi(GateInput *pi, Block *blockOwner, unsigned argNumber)
        : pi(pi), blockOwner(blockOwner), argNumber(argNumber) {}
  };

  // Initialize the index of the class
  uniqueGateIndex = 0;

  // Vector to store all the missing phi
  SmallVector<MissingPhi> phisToConnect;

  // For each block in the function
  for (Block &block : funcOp.getBlocks()) {

    // Create a list for the phi functions corresponding to the block
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

        // Check if the input value `c` of type `git` is already present among
        // the operands of the phi function
        auto isAlreadyPresent = [&](GateInputType git, Value c) -> bool {
          return std::any_of(
              operands.begin(), operands.end(), [git, c](GateInput *in) {
                return in->type == git && std::get<Value>(in->input) == c;
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

          // Try to convert the producer to a block argument (BA)
          BlockArgument ba = dyn_cast<BlockArgument>(producer);

          // If the producer is a BA but its block has no predecessor,
          // then it is a function argument, and it can be inserted as argument.
          //
          // Otherwise, if it is a BA, it must be connected to another phi. In
          // all the other situations, it comes from an operation, and it can be
          // inserted directly.
          if (ba && producer.getParentBlock()->hasNoPredecessors()) {
            if (!isAlreadyPresent(ArgInput, ba))
              gateInput = new GateInput(ba);
          } else if (ba) {
            gateInput = new GateInput((Gate *)nullptr);
            phisToConnect.push_back(
                MissingPhi(gateInput, ba.getParentBlock(), ba.getArgNumber()));
          } else {
            if (!isAlreadyPresent(OpInput, dyn_cast<Value>(producer)))
              gateInput = new GateInput(dyn_cast<Value>(producer));
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
        auto *newPhi = new Gate(arg, operands, GateType::PhiGate);
        newPhi->index = ++uniqueGateIndex;
        gateList[&block].push_back(newPhi);
      }
    }
  }

  // For each missing phi, look for it among the phis related to the
  // marked block and connect it
  for (auto &missing : phisToConnect) {
    Gate *foundGate = nullptr;
    for (auto &g : gateList[missing.blockOwner]) {
      if (g->getArgumentNumber() == missing.argNumber) {
        foundGate = g;
        break;
      }
    }
    assert(foundGate && "[GSA] Not found phi to reconnect");
    missing.pi->input = foundGate;
  }

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
                   return getBlockIndex(a->getBlock()) <
                          getBlockIndex(b->getBlock());
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
          auto *condition = getPathExpression(path, cofactorList);
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

      // Once that a phi has been converted into a tree of gammas, all the
      // functions which were previously connected to the phi are now to be
      // connected to the new gamma
      for (auto &[bb, phis] : gateList) {
        for (auto &phii : phis) {
          for (auto &op : phii->operands) {
            if (op->type == GSAInput && std::get<Gate *>(op->input) == phi)
              op->input = newPhi;
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

  for (auto const &[phiBlock, phis] : gateList) {
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
          phi->gsaGateFunction = GateType::MuGate;

          // Use the initial value of mu as first input of the gate
          if (domInfo.dominates(op1Block, phiBlock)) {
            auto *firstOperand = phi->operands[0];
            auto *secondOperand = phi->operands[1];
            phi->operands[0] = secondOperand;
            phi->operands[1] = firstOperand;
          }

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

  auto getPhiName = [&](Gate *p) -> std::string {
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
    switch (op->type) {
    case ArgInput:
      llvm::dbgs() << "[GSA]\t ARG\t: ";
      std::get<Value>(op->input).print(llvm::dbgs());
      break;
    case OpInput:
      llvm::dbgs() << "[GSA]\t OP\t: ";
      std::get<Value>(op->input).print(llvm::dbgs());
      break;
    case EmptyInput:
      llvm::dbgs() << "[GSA]\t EMPTY";
      break;
    default:
      llvm::dbgs() << "[GSA]\t PHI\t: "
                   << getPhiName(std::get<Gate *>(op->input)) << "_"
                   << std::get<Gate *>(op->input)->index;
      break;
    }

    if (op->type != EmptyInput) {
      llvm::dbgs() << "\t(";
      op->getBlock()->printAsOperand(llvm::dbgs());
      llvm::dbgs() << ")";
    }
    llvm::dbgs() << "\n";
  }
}

Block *GateInput::getBlock() {
  if (type == EmptyInput)
    return nullptr;
  if (type == GSAInput)
    return std::get<Gate *>(input)->getBlock();
  return std::get<Value>(input).getParentBlock();
}

} // namespace gsa
} // namespace experimental
} // namespace dynamatic

namespace dynamatic {

// Explicit template instantiation
template class experimental::gsa::GsaAnalysis<mlir::func::FuncOp>;
template class experimental::gsa::GsaAnalysis<handshake::FuncOp>;

} // namespace dynamatic
