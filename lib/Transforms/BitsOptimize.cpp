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
  
  
  SmallVector<Operation *> containerOps;
  

  bool changed = true;
  while (changed) {
    changed = false;

    // Forward process
    DenseMap<StringRef, forward_func> forMapOpNameWidth;
    forward::constructFuncMap(forMapOpNameWidth);

    for (auto &op : funcOp.getOps()){

      if (isa<handshake::ConstantOp>(op))
        continue;
      // store the operations in a container for backward process
      // containerOps.insert(containerOps.end(), &op);
      containerOps.push_back(&op);

      // propogate the type from operators to result 
      if (forward::passType(&op))
        continue;

      // get the new bit width of the result operator
      const auto opName = op.getName().getStringRef();
      unsigned int newWidth = 0;
      if (forMapOpNameWidth.find(opName) != forMapOpNameWidth.end())
        newWidth = forMapOpNameWidth[opName](op.getOperands());

      if (newWidth==0)
        continue;
      // if the new type can be optimized, update the type
      if(Type newOpResultType = getNewType(op.getResult(0), newWidth, true);  
          newWidth < op.getResult(0).getType().getIntOrFloatBitWidth() && newWidth > 0){
          changed |= true;
          op.getResult(0).setType(newOpResultType);
      }
      

      // TODO: Insert operation to pass the verification
      // forward::match
      if (isa<handshake::DynamaticLoadOp>(op) || 
          isa<handshake::DynamaticStoreOp>(op) || 
          isa<mlir::arith::MulIOp>(op) ||
          isa<mlir::arith::AddIOp>(op) ||
          isa<mlir::arith::CmpIOp>(op))  {
        bool isLdSt = isa<handshake::DynamaticLoadOp>(op) || 
                      isa<handshake::DynamaticStoreOp>(op);
        
        unsigned opWidth = cpp_max_width;

        // TODO: change it to backward function map
        if (!isLdSt)
          opWidth = op.getResult(0).getType().getIntOrFloatBitWidth();
        if (isa<mlir::arith::CmpIOp>(op))
          opWidth = std::max(op.getOperand(0).getType().getIntOrFloatBitWidth(),
                             op.getOperand(1).getType().getIntOrFloatBitWidth());
        
        for (unsigned int i = 0; i < op.getNumOperands(); ++i) {
          // width of data operand for Load and Store op 
          if (isLdSt && i==1)
            opWidth = address_width;

          // opWidth = 0 indicates the operand is none type operand, skip matched width insertion
          if (auto Operand = op.getOperand(i) ; opWidth != 0){
            auto insertOp = insertWidthMatchOp(&op, 
                                                i, 
                                                getNewType(Operand, opWidth, false), 
                                                ctx);
            // if (insertOp.has_value())
            // op.getOperand(i).replaceAllUsesWith(insertOp.value()->getResult(0));
          }
        }
      }
    }

    // Backward Process
    // all operations width are matched for verification, 
    // backward only update the trunc operation
    SmallVector<Operation *> truncOps;
    DenseMap<StringRef, backward_func> backMapOpNameWidth;
    backward::constructFuncMap(backMapOpNameWidth);

    for (auto op=containerOps.rbegin(); op!=containerOps.rend(); ++op) {
    //  if ((*op)->getNumResults() > 0 && !isa<N'oneType>((*op)->getResult(0).getType()))
      if (isa<mlir::arith::TruncIOp>(*op)) {
          changed |= true;
        llvm::errs() << (*op)->getResult(0) <<"\n";
        truncOps.push_back(*op);
        SmallVector<Operation *> vecOp;
        vecOp.insert(vecOp.end(), *op);

        backward::updateDefOpType((*op)->getOperand(0).getDefiningOp(),
                                  (*op)->getResult(0).getType(),
                                  vecOp,
                                  ctx);
        (*op)->getResult(0).replaceAllUsesWith((*op)->getOperand(0));
      }

      if (isa<handshake::ConstantOp>(**op))
        continue;
      
      if (isa<handshake::ConditionalBranchOp>(**op))
        (*op)->getOperand(1).setType((*op)->getResult(0).getType());

      if (isa<handshake::BranchOp>(**op))
        (*op)->getOperand(0).setType((*op)->getResult(0).getType());

      const auto opName = (*op)->getName().getStringRef();

      unsigned int newWidth = 0;
      // get the new bit width of the result operator
      if (backMapOpNameWidth.find(opName) != backMapOpNameWidth.end())
        newWidth = backMapOpNameWidth[opName]((*op)->getResults());

      if (newWidth==0)
        continue;
      // if the new type can be optimized, update the type
      if(Type newOpResultType = getNewType((*op)->getOperand(0), newWidth, true);  
          newWidth < (*op)->getOperand(0).getType().getIntOrFloatBitWidth() && newWidth > 0){
        changed |= true;
        llvm::errs() << "backward : " << (*op)->getResult(0) <<"\n";
        for (unsigned i=0;i<(*op)->getNumOperands();++i)
          (*op)->getOperand(i).setType(newOpResultType);
      }
      
    }

    llvm::errs() << "-----------------end one round-----------------\n";
    // remove truncOps
    for (auto op : truncOps)
      op->erase();

    // clear the operations
    containerOps.clear();
    
  }

  // for (auto &op : funcOp.getOps())
  //     llvm::errs() << op <<"\n";
    

  
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