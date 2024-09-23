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
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::func;

namespace dynamatic {
namespace experimental {
namespace gsa {

template <typename FunctionType>
void GsaAnalysis<FunctionType>::identifyAllPhi(FunctionType &funcOp) {

  // The input of a phi might be another phi. This is the case when the input of
  // a phi is a block argument from a block with index different from 0. In this
  // situation, we mark the phi input as `missing`, and we store the necessary
  // information (block of the block argument and argument index) to later
  // reconstruct the relationship.
  struct MissingPhi {

    // Which input is missing
    PhiInput *pi;
    // Which is the block owning the phi function which will provide the input
    Block *blockOwner;
    // Argument number of the missing phi
    unsigned argNumber;

    MissingPhi(PhiInput *pi, Block *blockOwner, unsigned argNumber)
        : pi(pi), blockOwner(blockOwner), argNumber(argNumber) {}
  };

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
      DenseSet<PhiInput *> operands;
      // For each predecessor of the block, which is in charge of providing the
      // inputs of the phi functions
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
            // If the producer is a BA but its block has no predecessor, then it
            // is a function argument. Otherwise, if it is a BA, it must be
            // connected to another phi. In all the other situations, it comes
            // from an operation.
            if (ba && producer.getParentBlock()->getPredecessors().empty()) {
              phiInput = new PhiInput(ba);
            } else if (ba) {
              phiInput = new PhiInput((Phi *)nullptr);
              phiToConnect.push_back(
                  MissingPhi(phiInput, ba.getParentBlock(), ba.getArgNumber()));
            } else {
              phiInput = new PhiInput(dyn_cast<Value>(producer));
            }

            // Insert the value among the inputs of the phi
            operands.insert(phiInput);
            break;
          }
        }
      }

      // If the list of operands is not empty (i.e. the phi has at least one
      // input), add it to the phis associated to that block
      if (!operands.empty()) {
        auto *newPhi = new Phi(arg, argNumber, operands, &block);
        phiListBlock.push_back(newPhi);
      }
    }

    // Associate the list of phis to the basic block
    phiList.insert({&block, phiListBlock});
  }

  // For each missing phi, look for it among the phis related to the marked
  // block and connect it
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

  printPhiList();
}

template <typename FunctionType>
SmallVector<Phi *> *GsaAnalysis<FunctionType>::getPhis(Block *bb) {
  if (!phiList.contains(bb))
    return nullptr;
  return &phiList[bb];
}

template <typename FunctionType>
Phi *GsaAnalysis<FunctionType>::getPhi(Block *bb, unsigned argNumber) {
  if (!phiList.contains(bb))
    return nullptr;
  auto &list = phiList[bb];
  for (auto *phi : list) {
    if (phi->argNumber == argNumber)
      return phi;
  }
  return nullptr;
}

template <typename FunctionType>
void GsaAnalysis<FunctionType>::printPhiList() {
  for (auto const &[block, phis] : phiList) {
    for (auto *phi : phis) {
      llvm::dbgs() << "[GSA] Block ";
      block->printAsOperand(llvm::dbgs());
      llvm::dbgs() << " arg " << phi->argNumber << "\n";
      for (auto &op : phi->operands) {
        switch (op->type) {
        case ArgInputType:
          llvm::dbgs() << "[GSA]\t ARG\t: ";
          op->v.print(llvm::dbgs());
          break;
        case OpInputType:
          llvm::dbgs() << "[GSA]\t OP\t: ";
          op->v.print(llvm::dbgs());
          break;
        default:
          llvm::dbgs() << "[GSA]\t PHI\t: arg " << op->phi->argNumber
                       << " from ";
          op->phi->blockOwner->printAsOperand(llvm::dbgs());
          break;
        }
        llvm::dbgs() << "\n";
      }
    }
  }
}

} // namespace gsa
} // namespace experimental
} // namespace dynamatic

namespace dynamatic {

// Explicit template instantiation
template class experimental::gsa::GsaAnalysis<mlir::func::FuncOp>;

} // namespace dynamatic
