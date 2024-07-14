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
      translateSSAToGsa(getOperation());
    };
  public:
    void translateSSAToGsa(func::FuncOp funcOp);  
  };
}; // namespace

void FuncSSAToGSAPass::translateSSAToGsa(func::FuncOp funcOp) {
  // instantiate the control dependence graph analysis 
  ControlDependenceAnalysis &cdg_analysis = getAnalysis<ControlDependenceAnalysis>();

  Region &funcReg = funcOp.getRegion();
  for (Block &block : funcReg.getBlocks()) {
    llvm::SmallVector<mlir::Block*, 4> returned_control_deps;
    // given a block, it returns (by reference) all blocks it is control dependent on
    cdg_analysis.returnAllControlDeps(&block, returned_control_deps);  
  }

  // prints the CDG information only in debug mode
  cdg_analysis.printAllBlocksDeps(funcOp);
  cdg_analysis.printForwardBlocksDeps(funcOp);

  // TODO: use the CDG analysis to create support GSA representation
}

std::unique_ptr<mlir::OperationPass<func::FuncOp>>
dynamatic::createFuncSSAToGSA() {
  return std::make_unique<FuncSSAToGSAPass>();
}