#include "dynamatic/Transforms/UtilsBitsUpdate.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"


IntegerType getNewType(Value opType, unsigned bitswidth, bool signless){                 
  IntegerType::SignednessSemantics ifSign = 
  IntegerType::SignednessSemantics::Signless;
  if (!signless)
    if (auto validType = opType.getType() ; isa<IntegerType>(validType))
      ifSign = dyn_cast<IntegerType>(validType).getSignedness();

  return IntegerType::get(opType.getContext(), bitswidth,ifSign);
}

IntegerType getNewType(Value opType, unsigned bitswidth,  
                      IntegerType::SignednessSemantics ifSign){
  return IntegerType::get(opType.getContext(), bitswidth,ifSign);
}

void constructFuncMap(DenseMap<StringRef, 
                     std::function<unsigned (Operation::operand_range vecOperands)>> 
                     &mapOpNameWidth){
                      
  mapOpNameWidth[StringRef("arith.addi")] = [](Operation::operand_range vecOperands){
    return std::min(arith_max_width,
                std::max(vecOperands[0].getType().getIntOrFloatBitWidth(), 
                        vecOperands[1].getType().getIntOrFloatBitWidth())+1);
  };

  mapOpNameWidth[StringRef("arith.subi")] = mapOpNameWidth[StringRef("arith.addi")];

  mapOpNameWidth[StringRef("arith.muli")] = [](Operation::operand_range vecOperands){
    return std::min(arith_max_width,
                vecOperands[0].getType().getIntOrFloatBitWidth() + 
                  vecOperands[1].getType().getIntOrFloatBitWidth());
  };

  mapOpNameWidth[StringRef("arith.ceildivsi")] = [](Operation::operand_range vecOperands){
    return std::min(arith_max_width,
                vecOperands[0].getType().getIntOrFloatBitWidth() + 1);
  };

  // mapOpNameWidth[StringRef("arith.ceil

  mapOpNameWidth[StringRef("arith.andi")] = [](Operation::operand_range vecOperands){
    return std::min(arith_max_width,
                std::min(vecOperands[0].getType().getIntOrFloatBitWidth(),
                      vecOperands[1].getType().getIntOrFloatBitWidth()));
  };

  mapOpNameWidth[StringRef("arith.ori")] = [](Operation::operand_range vecOperands){
    return std::min(arith_max_width,
                std::max(vecOperands[0].getType().getIntOrFloatBitWidth(),
                      vecOperands[1].getType().getIntOrFloatBitWidth()));
  };

  mapOpNameWidth[StringRef("arith.xori")] = mapOpNameWidth[StringRef("arith.ori")];


  mapOpNameWidth[StringRef("arith.shli")] = [](Operation::operand_range vecOperands){
    int shift_bit = 0;

    if (IntegerType validType = dyn_cast<IntegerType>(vecOperands[1].getType()) )
      llvm::errs() << "shift bit: " << *(uint64_t *)validType.getAsOpaquePointer() << "\n";
          
    return std::min(arith_max_width,
                vecOperands[0].getType().getIntOrFloatBitWidth() + shift_bit);
  };

  mapOpNameWidth[StringRef("arith.shrsi")] = [](Operation::operand_range vecOperands){
    int shift_bit = 0;

    if (IntegerType validType = dyn_cast<IntegerType>(vecOperands[1].getType()) )
      llvm::errs() << "shift bit: " << *(uint64_t *)validType.getAsOpaquePointer() << "\n";
          
    return std::min(arith_max_width,
                vecOperands[0].getType().getIntOrFloatBitWidth() - shift_bit);
  };

  mapOpNameWidth[StringRef("arith.shrui")] = mapOpNameWidth[StringRef("arith.shrsi")];

  mapOpNameWidth[StringRef("handshake.mux")] = [](Operation::operand_range vecOperands){
    unsigned max_width = 2;
    for (auto oprand : vecOperands)
      if (!isa<IndexType>(oprand.getType()) && 
          oprand.getType().getIntOrFloatBitWidth() > max_width)
        max_width = oprand.getType().getIntOrFloatBitWidth();
    return std::min(cpp_max_width, max_width);
  };

  mapOpNameWidth[StringRef("handshake.merge")] = mapOpNameWidth[StringRef("handshake.mux")];

  mapOpNameWidth[StringRef("handshake.cond_br")] = [](Operation::operand_range vecOperands){
    if (isa<NoneType>(vecOperands[1].getType()))
      return unsigned(0);
    return vecOperands[1].getType().getIntOrFloatBitWidth();
  };

  mapOpNameWidth[StringRef("arith.cmpi")] = [](Operation::operand_range vecOperands){
    return 1;
  };

  // mapOpNameWidth[StringRef("handshake.d_store")] = [](Operation::operand_range vecOperands){return address_width;
  // };

  // mapOpNameWidth[StringRef("handshake.d_load")] = mapOpNameWidth[StringRef("handshake.d_store")];
  
};

void setUserType(Operation *newOp, Type newType,
                          MLIRContext *ctx, SmallVector<int> vecIndex){
    for (int i : vecIndex){
      if (newOp->getResult(i).getType() == newType)
        continue;
      newOp->getResult(i).setType(newType);
    }  
    return;                      
}

// specify which value to extend
std::optional<Operation *> insertWidthMatchOp(Operation *newOp, int opInd, Type newType, 
                          MLIRContext *ctx){
  OpBuilder builder(ctx);
  Value opVal = newOp->getOperand(opInd);

  int opWidth;
  if (isa<IndexType>(opVal.getType()))
    opWidth = 64;
  else
    opWidth = opVal.getType().getIntOrFloatBitWidth();
  
  if (isa<IntegerType>(opVal.getType()) || isa<IndexType>(opVal.getType())){
    // insert Truncation operation to match the opresult width
    if (opWidth > newType.getIntOrFloatBitWidth()){
      builder.setInsertionPoint(newOp);
      auto extOp = builder.create<mlir::arith::TruncIOp>(newOp->getLoc(), 
                                                        newType,
                                                        opVal); 
      if (!isa<IndexType>(opVal.getType()))
        newOp->setOperand(opInd, extOp.getResult());
        
      return extOp;
    } 

    // insert Extension operation to match the opresult width
    if (opWidth < newType.getIntOrFloatBitWidth()){
      builder.setInsertionPoint(newOp);
      auto extOp = builder.create<mlir::arith::ExtSIOp>(newOp->getLoc(),
                                          newType,
                                          opVal); 
      if (!isa<IndexType>(opVal.getType())) 
        newOp->setOperand(opInd, extOp.getResult());
      
      return extOp;
    }
  }
  return {};    

}

// TODO: consider a universal method
void setUpdateFlag(Operation *newResult, 
                  bool &passType, bool &oprAdapt, bool &resAdapter, bool &deleteOp){
  
  passType = false;
  oprAdapt = false;
  resAdapter = false;
  deleteOp = false;

  if (isa<handshake::BranchOp>(newResult)) {
    passType = true;
  } 
  else if (isa<handshake::ConditionalBranchOp>(newResult)) {
    // resAdapter = true;
    // oprAdapt = true;
  }
  else if (isa<handshake::MuxOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
  }
  else if (isa<handshake::MergeOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
  }
  else if (isa<handshake::DynamaticStoreOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
  }
  else if (isa<handshake::DynamaticLoadOp>(newResult)){
    resAdapter = true;
    oprAdapt = true;
  }
  else if (isa<mlir::arith::CmpIOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
  }
  else if (isa<mlir::arith::AddIOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
  }
  else if (isa<mlir::arith::MulIOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
  }
  else if (isa<mlir::arith::IndexCastOp>(newResult)){
    // TODO: should rewrite the operation
    resAdapter = true;
    oprAdapt = true;
  }
  else if (isa<mlir::arith::ExtSIOp>(newResult) &&
          newResult->getResult(0).getType().getIntOrFloatBitWidth() <= newResult->getOperand(0).getType().getIntOrFloatBitWidth()) 
    deleteOp = true;
  
  else if (isa<mlir::arith::TruncIOp>(newResult) &&
          newResult->getResult(0).getType().getIntOrFloatBitWidth() >= newResult->getOperand(0).getType().getIntOrFloatBitWidth() ) 
    deleteOp = true;
          
}

void updateUserType(Operation *newResult, Type newType, SmallVector<Operation *> &vecOp,
                          MLIRContext *ctx){
  // if the operation has been processed, return
  if (std::find(vecOp.begin(), vecOp.end(), newResult) != vecOp.end()){
    return;
  }
    
  vecOp.push_back(newResult);
  // llvm::errs() << "----------recursive--------------"  << '\n';
  // llvm::errs() << "new Result: " << newResult->getResult(0) << '\n';
  OpBuilder builder(ctx);
  DenseMap<StringRef, std::function<unsigned (mlir::Operation::operand_range vecOperands)>>  mapOpNameWidth;

  constructFuncMap(mapOpNameWidth); 
  SmallVector<int> vecIndex;

  // only update the resultOp recursively with updated UserType : newType
  bool passType = false; 
  // insert extOp|truncOp to make the multiple operands have the same width
  bool oprAdapt = false; 
  // set the OpResult type to make the result have the same width as the operands
  bool resAdapt = false; 
  // delete the extOp|TruncIOp when the width of its result and operand are not consistent
  bool deleteOp = false;
  bool isLdSt = false;
  
  // TODO : functions for determine the four flags : passType, oprAdapt, resAdapter, deleteOp
  setUpdateFlag(newResult, passType, oprAdapt, resAdapt, deleteOp);


  if (passType) {
    setUserType(newResult, newType, ctx, {0});
    for(auto &user : newResult->getResult(0).getUses()) 
      updateUserType(user.getOwner(), newType, vecOp, ctx);
  }

  // unsigned opWidth;
  if (oprAdapt) {
    int startInd = 0;
    unsigned opWidth;
    
    if (isa<handshake::MuxOp>(newResult))
      startInd = 1; // start from the second operand (i=1), as the first one is the select index

    if (isa<handshake::DynamaticLoadOp>(newResult) ||
      isa<handshake::DynamaticStoreOp>(newResult)){
        isLdSt = true;
        opWidth = cpp_max_width;
      }
    // the comparsion operation holds different bit width strategy for its operands and opresult: 
    // for operands, the width follows the same strategy as handshake::muxOp: get the widest operands; 
    // for opresult, the width is always 1.
    else  if (isa<mlir::arith::CmpIOp>(newResult))
      opWidth = mapOpNameWidth[StringRef("handshake.mux")](newResult->getOperands());
    else
      opWidth = mapOpNameWidth[newResult->getName().getStringRef()](newResult->getOperands());

    for (int i = startInd; i < newResult->getNumOperands(); ++i){
      // width of data operand for Load and Store op 
      if (isLdSt && i==1)
        opWidth = address_width;
      // opWidth = 0 means the operand is none type operand
      if (isa<handshake::MuxOp>(newResult))
        llvm::errs() << opWidth << '\n';
      if (auto Operand = newResult->getOperand(i) ; opWidth != 0){
        auto insertOp = insertWidthMatchOp(newResult, i, getNewType(Operand, opWidth, false), ctx);
        if (insertOp.has_value())
          vecOp.push_back(insertOp.value());
      }
    }
  }

  if (resAdapt) {
    unsigned opWidth;
    if (isLdSt)
      opWidth = cpp_max_width;
    else
      opWidth= mapOpNameWidth[newResult->getName().getStringRef()](newResult->getOperands());
    
    for (int i = 0; i < newResult->getNumResults(); ++i)
      if (OpResult resultOp = newResult->getResult(i)){
        if (isLdSt && i==1)
          opWidth = address_width;
        // update the passed newType w.r.t. the resultOp
        newType = getNewType(resultOp, opWidth, false);
        setUserType(newResult, newType, ctx, {i});
        if (isa<handshake::MuxOp>(newResult)){
          llvm::errs() << newResult->getOperand(0) <<  newResult->getOperand(1) << '\n';

          llvm::errs() << "result: " << opWidth << '\n';

        }
        // update the user type recursively
        for(auto &user : resultOp.getUses()) 
          updateUserType(user.getOwner(), newType, vecOp, ctx);
      }
  }

  if (deleteOp) {
    for(auto user : newResult->getResult(0).getUsers()) 
      user->replaceUsesOfWith(newResult->getResult(0), newResult->getOperand(0));
    newResult->erase();
  }

}