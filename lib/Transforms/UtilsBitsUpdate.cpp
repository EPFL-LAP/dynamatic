#include "dynamatic/Transforms/UtilsBitsUpdate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/IndentedOstream.h"


IntegerType getNewType(Value opType, 
                       unsigned bitswidth, 
                       bool signless) 
{                 
  IntegerType::SignednessSemantics ifSign = 
  IntegerType::SignednessSemantics::Signless;
  if (!signless)
    if (auto validType = opType.getType() ; isa<IntegerType>(validType))
      ifSign = dyn_cast<IntegerType>(validType).getSignedness();

  return IntegerType::get(opType.getContext(), bitswidth,ifSign);
}

IntegerType getNewType(Value opType, 
                       unsigned bitswidth,  
                       IntegerType::SignednessSemantics ifSign) 
{
  return IntegerType::get(opType.getContext(), bitswidth,ifSign);
}

void setUserType (Operation *newOp, 
                 Type newType,
                 SmallVector<int> vecIndex)
{
    for (int i : vecIndex){
      if (newOp->getResult(i).getType() == newType)
        continue;
      newOp->getResult(i).setType(newType);
    }                       
}


void setDefOpType (Operation *newOp, 
                   Type newType,
                   SmallVector<int> vecIndex)
{
  for (int i : vecIndex) {
    if (newOp->getOperand(i).getType() == newType)
      continue;
    newOp->getOperand(i).setType(newType);
  }  
} 


// specify which value to extend
std::optional<Operation *> insertWidthMatchOp (Operation *newOp, 
                                               int opInd, 
                                               Type newType, 
                                               MLIRContext *ctx)
{
  OpBuilder builder(ctx);
  Value opVal = newOp->getOperand(opInd);

  unsigned int opWidth;
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

namespace update {

void constructFuncMap(DenseMap<StringRef, 
                     std::function<unsigned (Operation::operand_range vecOperands)>> 
                     &mapOpNameWidth){
                      
  mapOpNameWidth[StringRef("arith.addi")] = [](Operation::operand_range vecOperands)
  {
    return std::min(cpp_max_width,
                std::max(vecOperands[0].getType().getIntOrFloatBitWidth(), 
                        vecOperands[1].getType().getIntOrFloatBitWidth())+1);
  };

  mapOpNameWidth[StringRef("arith.subi")] = mapOpNameWidth[StringRef("arith.addi")];

  mapOpNameWidth[StringRef("arith.muli")] = [](Operation::operand_range vecOperands)
  {
    return std::min(cpp_max_width,
                vecOperands[0].getType().getIntOrFloatBitWidth() + 
                  vecOperands[1].getType().getIntOrFloatBitWidth());
  };

  mapOpNameWidth[StringRef("arith.ceildivsi")] = [](Operation::operand_range vecOperands)
  {
    return std::min(cpp_max_width,
                vecOperands[0].getType().getIntOrFloatBitWidth() + 1);
  };

  // mapOpNameWidth[StringRef("arith.ceil

  mapOpNameWidth[StringRef("arith.andi")] = [](Operation::operand_range vecOperands)
  {
    return std::min(cpp_max_width,
                std::min(vecOperands[0].getType().getIntOrFloatBitWidth(),
                      vecOperands[1].getType().getIntOrFloatBitWidth()));
  };

  mapOpNameWidth[StringRef("arith.ori")] = [](Operation::operand_range vecOperands)
  {
    return std::min(cpp_max_width,
                std::max(vecOperands[0].getType().getIntOrFloatBitWidth(),
                      vecOperands[1].getType().getIntOrFloatBitWidth()));
  };

  mapOpNameWidth[StringRef("arith.xori")] = mapOpNameWidth[StringRef("arith.ori")];


  mapOpNameWidth[StringRef("arith.shli")] = [](Operation::operand_range vecOperands)
  {
    int shift_bit = 0;
    if (auto defOp = vecOperands[1].getDefiningOp(); isa<handshake::ConstantOp>(defOp)){
      if (handshake::ConstantOp cstOp = dyn_cast<handshake::ConstantOp>(defOp))
        if (auto IntAttr = cstOp.getValue().dyn_cast<mlir::IntegerAttr>())
          shift_bit = IntAttr.getValue().getZExtValue();
      return std::min(cpp_max_width,
                vecOperands[0].getType().getIntOrFloatBitWidth() + shift_bit);
    }
    return cpp_max_width;
  };

  mapOpNameWidth[StringRef("arith.shrsi")] = [](Operation::operand_range vecOperands)
  {
    int shift_bit = 0;
    if (auto defOp = vecOperands[1].getDefiningOp(); isa<handshake::ConstantOp>(defOp))
      if (handshake::ConstantOp cstOp = dyn_cast<handshake::ConstantOp>(defOp))
        if (auto IntAttr = cstOp.getValue().dyn_cast<mlir::IntegerAttr>())
          shift_bit = IntAttr.getValue().getZExtValue();

    return std::min(cpp_max_width,
              vecOperands[0].getType().getIntOrFloatBitWidth() - shift_bit);

  };

  mapOpNameWidth[StringRef("arith.shrui")] = mapOpNameWidth[StringRef("arith.shrsi")];
  
  mapOpNameWidth[StringRef("handshake.mux")] = [](Operation::operand_range vecOperands)
  {
    if (isa<NoneType>(vecOperands[0].getType()))
      return unsigned(0);
    unsigned max_width = 2;
    for (auto oprand : vecOperands)
      if (!isa<IndexType>(oprand.getType()) && 
          oprand.getType().getIntOrFloatBitWidth() > max_width)
        max_width = oprand.getType().getIntOrFloatBitWidth();
    return std::min(cpp_max_width, max_width);
  };

  mapOpNameWidth[StringRef("arith.cmpi")] = mapOpNameWidth[StringRef("handshake.mux")];

  mapOpNameWidth[StringRef("handshake.merge")] = mapOpNameWidth[StringRef("handshake.mux")];

  mapOpNameWidth[StringRef("handshake.cond_br")] = [](Operation::operand_range vecOperands)
  {
    if (isa<NoneType>(vecOperands[1].getType()))
      return unsigned(0);
    return vecOperands[1].getType().getIntOrFloatBitWidth();
  };

  mapOpNameWidth[StringRef("handshake.d_return")] = [](Operation::operand_range vecOperands)
  {
    return address_width;
  };

};

// TODO: consider a universal method
void setUpdateFlag(Operation *newResult, 
                  bool &passType, 
                  bool &oprAdapt, 
                  bool &resAdapter, 
                  bool &deleteOp, 
                  bool &lessThanRes,
                  SmallVector<int> &vecIndex)
{
  
  passType = false;
  oprAdapt = false;
  resAdapter = false;
  deleteOp = false;
  lessThanRes = false;

  if (isa<handshake::BranchOp>(newResult)) {
    passType = true;
    vecIndex = {0};
  } 
  else if (isa<handshake::ConditionalBranchOp>(newResult)) {
    passType = true;
    vecIndex = {0, 1};
  }
  else if (isa<handshake::MuxOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
    lessThanRes = true;
  }
  else if (isa<handshake::MergeOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
  }
  else if (isa<handshake::DynamaticStoreOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
  }
  else if (isa<handshake::DynamaticLoadOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
  }
  else if (isa<handshake::DynamaticReturnOp>(newResult)) {
    oprAdapt = true;
  }
  else if (isa<mlir::arith::CmpIOp>(newResult)) {
    oprAdapt = true;
  }
  else if (isa<mlir::arith::AddIOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
    lessThanRes = true;
  }
  else if (isa<mlir::arith::MulIOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
    lessThanRes = true;
  }
  else if (isa<mlir::arith::ShRSIOp>(newResult)) {
    resAdapter = true;
    oprAdapt = true;
    lessThanRes = true;
  }
  else if (isa<mlir::arith::ExtSIOp>(newResult) &&
          newResult->getResult(0).getType().getIntOrFloatBitWidth() <= newResult->getOperand(0).getType().getIntOrFloatBitWidth()) 
    deleteOp = true;
  
  else if (isa<mlir::arith::TruncIOp>(newResult) &&
          newResult->getResult(0).getType().getIntOrFloatBitWidth() >= newResult->getOperand(0).getType().getIntOrFloatBitWidth() ) 
    deleteOp = true;
          
}

void updateUserType(Operation *newResult, 
                    Type newType, 
                    SmallVector<Operation *> &vecOp,
                    MLIRContext *ctx)
{
  // if the operation has been processed, return
  if (std::find(vecOp.begin(), vecOp.end(), newResult) != vecOp.end()){
    return;
  }
    
  vecOp.push_back(newResult);
  OpBuilder builder(ctx);
  DenseMap<StringRef, std::function<unsigned (mlir::Operation::operand_range vecOperands)>>  mapOpNameWidth;

  constructFuncMap(mapOpNameWidth); 
  SmallVector<int> vecIndex;

  // bit width of the operand should not exceeds the result width
  bool lessThanRes = false;
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
  setUpdateFlag(newResult, 
                passType, 
                oprAdapt, 
                resAdapt, 
                deleteOp, 
                lessThanRes,
                vecIndex);

  if (passType) {
    setUserType(newResult, newType, vecIndex);

    for(auto &user : newResult->getResult(0).getUses()) 
      updateUserType(user.getOwner(), newType, vecOp, ctx);
  }

  // unsigned opWidth;setUserType
  if (oprAdapt) {
    unsigned int startInd = 0;
    unsigned opWidth;
    
    // start from the second operand (i=1), as the first one is the select index
    if (isa<handshake::MuxOp>(newResult) || 
        isa<handshake::ConditionalBranchOp>(newResult))
      startInd = 1; 

    if (isa<handshake::DynamaticLoadOp>(newResult) ||
      isa<handshake::DynamaticStoreOp>(newResult)) {
        isLdSt = true;
        opWidth = cpp_max_width;
      }
    else {
      opWidth = mapOpNameWidth[newResult->getName().getStringRef()](newResult->getOperands());
      if (lessThanRes)
        opWidth = std::min(opWidth,
                          newResult->getResult(0).getType().getIntOrFloatBitWidth());
    }

    for (unsigned int i = startInd; i < newResult->getNumOperands(); ++i) {
      // width of data operand for Load and Store op 
      if (isLdSt && i==1)
        opWidth = address_width;

      // opWidth = 0 indicates the operand is none type operand, skip matched width insertion
      if (auto Operand = newResult->getOperand(i) ; opWidth != 0){
        auto insertOp = insertWidthMatchOp(newResult, 
                                          i, 
                                          getNewType(Operand, opWidth, false), 
                                          ctx);
        if (insertOp.has_value())
          vecOp.push_back(insertOp.value());
      }
    }
  }

  if (resAdapt) {
    unsigned opWidth;
    if (isLdSt)
      opWidth = cpp_max_width;
    else {
      opWidth = mapOpNameWidth[newResult->getName().getStringRef()](newResult->getOperands());
      if (lessThanRes)
        opWidth = std::min(opWidth,
                          newResult->getResult(0).getType().getIntOrFloatBitWidth());
      }
    
    for (unsigned int i = 0; i < newResult->getNumResults(); ++i){

      if (OpResult resultOp = newResult->getResult(i)){
        if (isLdSt && i==1)
          opWidth = address_width;
        // update the passed newType w.r.t. the resultOp
        newType = getNewType(resultOp, opWidth, false);
        setUserType(newResult, newType, {static_cast<int>(i)});

        // update the user type recursively
        for(auto &user : resultOp.getUses()) 
          updateUserType(user.getOwner(), newType, vecOp, ctx);
      }
    }
  }

  if (deleteOp) {
    for(auto user : newResult->getResult(0).getUsers()) 
      user->replaceUsesOfWith(newResult->getResult(0), newResult->getOperand(0));
    newResult->erase();
  }
}
}