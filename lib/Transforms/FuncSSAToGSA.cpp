#include "dynamatic/Transforms/FuncSSAToGSA.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

#include "dynamatic/Analysis/ControlDependenceAnalysis.h"

using namespace mlir;
using namespace dynamatic;

namespace {

// Simple driver for the pass.
struct FuncSSAToGSAPass
    : public dynamatic::impl::FuncSSAToGSABase<FuncSSAToGSAPass> {

public:
  void runOnOperation() override { translateSSAToGSA(getOperation()); };
  void translateSSAToGSA(func::FuncOp funcOp);

private:
  SmallVector<SSAPhi, 4> ssa_phis;
};
}; // namespace

// GOAL: FILL A VECTOR OF STRUCTURE WHERE EACH STRUCTURE REPRESENTS A MERGE:
// composed of owner Block and two producer Blocks
// FOR THE ABOVE GOAL, WE CAN CALL THIS FUNCTION EXPLICIT SSA, BUT NOT YET GSA!!
void FuncSSAToGSAPass::translateSSAToGSA(func::FuncOp funcOp) {
  Region &funcReg = funcOp.getRegion();

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
      ssa_phis.push_back(phi);
    }
  }
}

std::unique_ptr<mlir::OperationPass<func::FuncOp>>
dynamatic::createFuncSSAToGSA() {
  return std::make_unique<FuncSSAToGSAPass>();
}