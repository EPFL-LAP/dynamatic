//===- BitsOptimize.cpp -  Optimize bits width -------*- C++ -*-===//
//
// This file contains the implementation of the bits optimization pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BitsOptimize.h"
#include "dynamatic/Transforms/UtilsBitsUpdate.h"
#include "dynamatic/Transforms/ForwardUpdate.h"
#include "dynamatic/Transforms/BackwardUpdate.h"
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
  using backward_func = std::function<unsigned (mlir::Operation::result_range vecResults)>;
  
  DenseMap<StringRef, forward_func> mapOpNameWidth;
  SmallVector<Operation *> containerOps;
  update::constructFuncMap(mapOpNameWidth);

  bool changed = true;
  while (changed) {
    changed = false;

    // Forward process
    for (auto &op : funcOp.getOps()){
      // store the operations in a container for backward process
      // containerOps.insert(containerOps.end(), &op);
      llvm::errs() << op << "\n";
      containerOps.push_back(&op);

      if (isa<handshake::ConstantOp>(op))
        continue;
      // get the name of the operator
      const auto opName = op.getName().getStringRef();

      if (op.getNumResults() > 0 && !isa<NoneType>(op.getResult(0).getType())){
        unsigned int newWidth = 0;
        // get the new bit width of the result operator
        if (mapOpNameWidth.find(opName) != mapOpNameWidth.end())
          newWidth = mapOpNameWidth[opName](op.getOperands());

        // if the new type can be optimized, update the type
        if(Type newOpResultType = getNewType(op.getResult(0), newWidth, true);  
            newWidth < op.getResult(0).getType().getIntOrFloatBitWidth() && newWidth > 0){
          changed |= true;

          SmallVector<Operation *> userOps;
          for (auto &user : op.getResult(0).getUses())
            userOps.push_back(user.getOwner());

          SmallVector<Operation *> vecOp;
          for (auto updateOp : userOps){
            vecOp.insert(vecOp.end(), &op);
            update::updateUserType(updateOp, newOpResultType, vecOp, ctx);
            vecOp.clear();
          }
        }
      }

      // TODO: Insert operation to pass the verification
      if (isa<handshake::DynamaticLoadOp>(op) || 
          isa<handshake::DynamaticStoreOp>(op) || 
          isa<mlir::arith::MulIOp>(op) ||
          isa<mlir::arith::AddIOp>(op))  {
        bool isLdSt = isa<handshake::DynamaticLoadOp>(op) || 
                      isa<handshake::DynamaticStoreOp>(op);
        unsigned opWidth = cpp_max_width;

        // TODO: change it to backward function map
        if (!isLdSt)
          opWidth = op.getResult(0).getType().getIntOrFloatBitWidth();
        
        for (unsigned int i = 0; i < op.getNumOperands(); ++i) {
          // width of data operand for Load and Store op 
          if (isLdSt && i==1)
            opWidth = address_width;

          // opWidth = 0 indicates the operand is none type operand, skip matched width insertion
          if (auto Operand = op.getOperand(i) ; opWidth != 0)
            auto insertOp = insertWidthMatchOp(&op, 
                                                i, 
                                                getNewType(Operand, opWidth, false), 
                                                ctx);
        }
      }
        
    }

    

    // Backward Process
    // all operations width are matched for verification, 
    // backward only update the trunc operation
    SmallVector<Operation *> truncOps;

    for (auto op=containerOps.rbegin(); op!=containerOps.rend(); ++op) {
    //  if ((*op)->getNumResults() > 0 && !isa<N'oneType>((*op)->getResult(0).getType()))
      if (isa<mlir::arith::TruncIOp>(*op)) {
          changed |= true;

        truncOps.push_back(*op);
        SmallVector<Operation *> vecOp;
        vecOp.insert(vecOp.end(), *op);

        backward::updateDefOpType((*op)->getOperand(0).getDefiningOp(),
                                  (*op)->getResult(0).getType(),
                                  vecOp,
                                  ctx);
        (*op)->getResult(0).replaceAllUsesWith((*op)->getOperand(0));
      }
    }
   
    llvm::errs() << "-----------------end one round-----------------\n";
    // remove truncOps
    for (auto op : truncOps)
      op->erase();

    // clear the operations
    containerOps.clear();
    
  }

  
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