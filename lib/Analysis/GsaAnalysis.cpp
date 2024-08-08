//===- GsaAnalysis.h - Gated Single Assignment analyis utilities
//----------*- C++ -*-===//
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

#include "dynamatic/Analysis/GsaAnalysis.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::func;
using namespace dynamatic;

// Adds a new entry to the private field all_ssa_phis that is composed of SSAPhi
// objects. Each SSAPhi contains the owner Block along with a vector of the
// producer operations
void GsaAnalysis::identifySsaPhis(func::FuncOp &funcOp) {
  Region &funcReg = funcOp.getRegion();

  SmallVector<SSAPhi *, 4> ssa_phis;

  for (Block &block : funcReg.getBlocks()) {

    // llvm::errs() << "\n Printing the information of ";
    // block.printAsOperand(llvm::errs());
    // llvm::errs() << ": ";

    // looping over the block's arguments..
    for (BlockArgument arg : block.getArguments()) {
      Block *owner_block = arg.getOwner(); // the block that owns the argument
      // Create a new SSAPhi object
      SSAPhi phi;
      phi.owner_block = owner_block; // the block that the argument is inside

      // loop over the predecessor blocks of the owner_block
      for (Block *pred_block : owner_block->getPredecessors()) {

        // llvm::errs() << "\tPred of the block's argument: ";
        // pred_block->printAsOperand(llvm::errs());

        // for each block, identify its terminator branching instruction
        auto branch_op =
            dyn_cast<BranchOpInterface>(pred_block->getTerminator());
        assert(branch_op && "Expected terminator operation in a predecessor "
                            "block feeding a block argument!");

        for (auto [idx, succ_branch_block] :
             llvm::enumerate(branch_op->getSuccessors())) {
          // llvm::errs()
          //     << "\n \tPrinting the successor blocks of the Pred block: ";
          // succ_branch_block->printAsOperand(llvm::errs());
          // llvm::errs() << "\n";
          // llvm::errs() << "Printing the types of values passed to this "
          //                 "successor block: ";

          for (int u = 0; u < branch_op.getSuccessorOperands(idx).size(); u++) {
            Operation *producer_operation =
                branch_op.getSuccessorOperands(idx)[u].getDefiningOp();
            phi.producer_operations.push_back(producer_operation);

            // llvm::errs() << producer_operation->getName();
            // llvm::errs() << " in block: ";
            // branch_op.getSuccessorOperands(idx)[u]
            //     .getDefiningOp()
            //     ->getBlock()
            //     ->printAsOperand(llvm::errs());
            // llvm::errs() << ", ";
          }
          // llvm::errs() << "\n\n";
        }
      }
      ssa_phis.push_back(&phi);
    }
  }

  all_ssa_phis.push_back(ssa_phis);
}

// takes a function ID and a Block* and searches for this Block's name in the
// all_deps of this function and overwrites its pointer value
void GsaAnalysis::adjustBlockPtr(int funcOp_idx, Block *new_block) {
  // use the name to search for this block in all dependencies and update its
  // ptr
  // for (auto &one_block_deps : all_ssa_phis[funcOp_idx]) {
  //   Block *old_block = one_block_deps.first;
  //   SmallVector<Block *, 4> old_block_deps;
  //   compareNamesAndModifyBlockPtr(new_block, old_block, old_block_deps);
  // }
}

// TODO: (1) Complete the above function, (2) Create a similar function for
// operations ptr, (3) Create print functions to test the correctness, (4) CHeck
// Dana's stuff and direct her to do the additional Mux condition, (5) Implement
// the rest of the GSA