//===- BitsOptimize.cpp -  Optimize bits width -------*- C++ -*-===//
//
// This file contains the implementation of the bits optimization pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BitsOptimize.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/IR/InstIterator.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"


using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

static LogicalResult bitsOptimize(handshake::FuncOp funcOp) {
  llvm::errs() << funcOp->getNumRegions() << '\n';

  llvm::errs() << "visiting regions: \n";
  for (auto &region : funcOp->getRegions()){
    // llvm::errs() << region << '\n';
    
    llvm::errs() << region.getBlocks().size()  << '\n';

  }

  llvm::errs() << "visiting op: \n";
  for (auto &op : funcOp.getOps()){
    // validate the operation is legalized or not

    // functions of determining operators and results type
    
    // determine whether success

    llvm::errs() << op << '\n';
    llvm::errs() << op.getOperandTypes() << '\n';
    for(NamedAttribute attr : op.getAttrs())
      llvm::errs() << " - '" << attr.getName() << "' : '" << attr.getValue()
                      << "'\n";
    llvm::errs() << op.getResultTypes() << "\n\n";
  }
  
  return success();
}

namespace {
struct HandshakeBitsOptimizePass
    : public HandshakeBitsOptimizeBase<HandshakeBitsOptimizePass> {

  void runOnOperation() override {
    ModuleOp m = getOperation();

    llvm::errs() << "Attemp to debug\n";
    for (auto funcOp : m.getOps<handshake::FuncOp>()){
      llvm::errs() << "Attemp to debug funcOp\n";
      if (failed(bitsOptimize(funcOp)))
        return signalPassFailure();
    }
  };

};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createBitsOptimizationPass() {
  return std::make_unique<HandshakeBitsOptimizePass>();
}