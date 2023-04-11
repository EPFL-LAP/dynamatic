//===- BitsOptimize.cpp -  Optimize bits width -------*- C++ -*-===//
//
// This file contains the implementation of the bits optimization pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BitsOptimize.h"
#include "dynamatic/Transforms/UtilsBitsUpdate.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "dynamatic/Transforms/Passes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/IndentedOstream.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

using namespace mlir::detail;


static LogicalResult rewriteBitsWidths(handshake::FuncOp funcOp, MLIRContext *ctx) {
  OpBuilder builder(ctx);
  SmallVector<Operation *> vecOp;

  using forward_func  = std::function<unsigned (mlir::Operation::operand_range vecOperands)>;

    DenseMap<StringRef, forward_func> mapOpNameWidth;
    constructFuncMap(mapOpNameWidth);

    // bool changed = true;
    // while (changed)
    // changed = false;

    // Forward process
    for (auto &op : funcOp.getOps()){
      llvm::errs() << "op : " << op << '\n';
      if (isa<handshake::ConstantOp>(op))
        continue;
      // get the name of the operator
      const auto opName = op.getName().getStringRef();

      if (0 < op.getNumResults()){
        int newWidth = 32;
        if (mapOpNameWidth.find(opName) != mapOpNameWidth.end()){
          // get the new bit width of the result operator
          newWidth = mapOpNameWidth[opName](op.getOperands());
          llvm::errs() << "change forward resultOp Type\n";
          // if the new type can be optimized, update the type
          if(Type newOpResultType = getNewType(op.getResult(0), newWidth, true);  
              newOpResultType != op.getResult(0).getType()){
                // changed |= true;
                op.getResult(0).setType(newOpResultType);
                for (auto &user : op.getResult(0).getUses())
                  updateUserType(user.getOwner(), newOpResultType, vecOp, ctx);
              }

        }
      }

    }

    // Backward Process
    
    
    return success();
}

struct HandshakeBitsOptimizePass
    : public HandshakeBitsOptimizeBase<HandshakeBitsOptimizePass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    ModuleOp m = getOperation();

    llvm::errs() << "Attemp to debug\n";
    for (auto funcOp : m.getOps<handshake::FuncOp>())
      // if(failed(rewriteBitsWidths(funcOp, ctx)))
        return signalPassFailure();

    llvm::errs() << "End of debug\n";

  };

};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createBitsOptimizationPass() {
  return std::make_unique<HandshakeBitsOptimizePass>();
}