#include "dynamatic/Transforms/FuncSSAToGSA.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"


using namespace mlir;
using namespace dynamatic;

namespace {

/// Simple driver for the pass. 
struct FuncSSAToGSAPass
    : public dynamatic::impl::FuncSSAToGSABase<FuncSSAToGSAPass> {

  public:
    void runOnOperation() override {
      llvm::outs() << "Hii from inside Aya's pass!\n";
      Region &funcReg = getOperation().getRegion(); 
      
      for (Block &block : funcReg.getBlocks()) {
        llvm::outs() << "Block!\n";
      }
    };
  };
}; // namespace


std::unique_ptr<mlir::OperationPass<func::FuncOp>>
dynamatic::createFuncSSAToGSA() {
  return std::make_unique<FuncSSAToGSAPass>();
}