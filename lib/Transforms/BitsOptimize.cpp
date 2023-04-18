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

      if (isa<handshake::ConstantOp>(op))
        continue;
      // get the name of the operator
      const auto opName = op.getName().getStringRef();

      if (op.getNumResults() > 0 && !isa<NoneType>(op.getResult(0).getType())){
        unsigned int newWidth = 0;
        // get the new bit width of the result operator
        if (mapOpNameWidth.find(opName) != mapOpNameWidth.end())
          newWidth = mapOpNameWidth[opName](op.getOperands());
        

        // if newWidth==0 => NoneType, skip operation
        if (newWidth==0)
          continue;

          // if the new type can be optimized, update the type
        if(Type newOpResultType = getNewType(op.getResult(0), newWidth, true);  
            newWidth < op.getResult(0).getType().getIntOrFloatBitWidth()){
          // changed |= true;
          llvm::errs() << "-------------------\n";
          llvm::errs() << "Update " << op.getResult(0).getType() <<
          " to " << newOpResultType << "\n";
          
          op.getResult(0).setType(newOpResultType);

          SmallVector<Operation *> userOps;
          for (auto &user : op.getResult(0).getUses())
            userOps.push_back(user.getOwner());

          SmallVector<Operation *> vecOp;
          for (auto updateOp : userOps){
            vecOp.insert(vecOp.end(), &op);
            updateUserType(updateOp, newOpResultType, vecOp, ctx);
            vecOp.clear();
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

    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if(failed(rewriteBitsWidths(funcOp, ctx)))
        return signalPassFailure();


  };

};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createBitsOptimizationPass() {
  return std::make_unique<HandshakeBitsOptimizePass>();
}