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
    void runOnOperation() override {
      translate_ssa_to_gsa(getOperation());
    };
  public:
    void translate_ssa_to_gsa(func::FuncOp funcOp);  // the main function of the pass
  };
}; // namespace

void FuncSSAToGSAPass::translate_ssa_to_gsa(func::FuncOp funcOp) {
  // instantiate the control dependence graph analysis 
  ControlDependenceAnalysis &cdg_analysis = getAnalysis<ControlDependenceAnalysis>();

  Region &funcReg = funcOp.getRegion();
  for (Block &block : funcReg.getBlocks()) {
    llvm::SmallVector<mlir::Block*, 4> returned_control_deps;
    cdg_analysis.returnControlDeps(&block, returned_control_deps);  // given a block, it returns (by reference) all blocks it is control dependent on
  }

  // uncomment this if you want to see control dependencies in terminal
  cdg_analysis.printBlocksDeps(funcOp);
}

std::unique_ptr<mlir::OperationPass<func::FuncOp>>
dynamatic::createFuncSSAToGSA() {
  return std::make_unique<FuncSSAToGSAPass>();
}