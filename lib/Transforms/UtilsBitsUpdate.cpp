#include "dynamatic/Transforms/UtilsBitsUpdate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/IR/Dialect.h"


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
      auto truncOp = builder.create<mlir::arith::TruncIOp>(newOp->getLoc(), 
                                                        newType,
                                                        opVal); 
      if (!isa<IndexType>(opVal.getType()))
        newOp->setOperand(opInd, truncOp.getResult());
        
      return truncOp;
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

void constructForwardFuncMap(DenseMap<StringRef, 
                     std::function<unsigned (Operation::operand_range vecOperands)>> 
                     &mapOpNameWidth){
                      
  mapOpNameWidth[StringRef("arith.addi")] = [](Operation::operand_range vecOperands)
  {
    return std::min(CPP_MAX_WIDTH,
                std::max(vecOperands[0].getType().getIntOrFloatBitWidth(), 
                        vecOperands[1].getType().getIntOrFloatBitWidth())+1);
  };

  mapOpNameWidth[StringRef("arith.subi")] = mapOpNameWidth[StringRef("arith.addi")];

  mapOpNameWidth[StringRef("arith.muli")] = [](Operation::operand_range vecOperands)
  {
    return std::min(CPP_MAX_WIDTH,
                vecOperands[0].getType().getIntOrFloatBitWidth() + 
                  vecOperands[1].getType().getIntOrFloatBitWidth());
  };

  mapOpNameWidth[StringRef("arith.divui")] = [](Operation::operand_range vecOperands)
  {
    return std::min(CPP_MAX_WIDTH,
                vecOperands[0].getType().getIntOrFloatBitWidth() + 1);
  };
  mapOpNameWidth[StringRef("arith.divsi")] = mapOpNameWidth[StringRef("arith.divui")];

  mapOpNameWidth[StringRef("arith.andi")] = [](Operation::operand_range vecOperands)
  {
    return std::min(CPP_MAX_WIDTH,
                std::min(vecOperands[0].getType().getIntOrFloatBitWidth(),
                      vecOperands[1].getType().getIntOrFloatBitWidth()));
  };

  mapOpNameWidth[StringRef("arith.ori")] = [](Operation::operand_range vecOperands)
  {
    return std::min(CPP_MAX_WIDTH,
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
      return std::min(CPP_MAX_WIDTH,
                vecOperands[0].getType().getIntOrFloatBitWidth() + shift_bit);
    }
    return CPP_MAX_WIDTH;
  };

  mapOpNameWidth[StringRef("arith.shrsi")] = [](Operation::operand_range vecOperands)
  {
    int shift_bit = 0;
    if (auto defOp = vecOperands[1].getDefiningOp(); isa<handshake::ConstantOp>(defOp))
      if (handshake::ConstantOp cstOp = dyn_cast<handshake::ConstantOp>(defOp))
        if (auto IntAttr = cstOp.getValue().dyn_cast<mlir::IntegerAttr>())
          shift_bit = IntAttr.getValue().getZExtValue();

    return std::min(CPP_MAX_WIDTH,
              vecOperands[0].getType().getIntOrFloatBitWidth() - shift_bit);

  };

  mapOpNameWidth[StringRef("arith.shrui")] = mapOpNameWidth[StringRef("arith.shrsi")];

  mapOpNameWidth[StringRef("arith.cmpi")] = [](Operation::operand_range vecOperands)
  {
    return unsigned(1);
  };

  mapOpNameWidth[StringRef("arith.extsi")] = [](Operation::operand_range vecOperands)
  {
    return vecOperands[0].getType().getIntOrFloatBitWidth();
  };

  mapOpNameWidth[StringRef("arith.extui")] = mapOpNameWidth[StringRef("arith.extsi")];

  mapOpNameWidth[StringRef("handshake.control_merge")] = [](Operation::operand_range vecOperands)
  {
    unsigned ind = 0; // record number of operators

    for (auto oprand : vecOperands) 
      ind++;

    unsigned indexWidth=1;
    if (ind>1)
      indexWidth = ceil(log2(ind));

    return indexWidth;
  };

};

void constructBackwardFuncMap(DenseMap<StringRef, 
                        std::function<unsigned (Operation::result_range vecResults)>> 
                        &mapOpNameWidth)
  {
    mapOpNameWidth[StringRef("arith.addi")] = [](Operation::result_range vecResults)
    {
      return std::min(CPP_MAX_WIDTH, 
                      vecResults[0].getType().getIntOrFloatBitWidth());
    };

    mapOpNameWidth[StringRef("arith.subi")] = mapOpNameWidth[StringRef("arith.addi")];

    mapOpNameWidth[StringRef("arith.muli")] = mapOpNameWidth[StringRef("arith.addi")];

    mapOpNameWidth[StringRef("arith.andi")] = mapOpNameWidth[StringRef("arith.addi")];

    mapOpNameWidth[StringRef("arith.ori")] = mapOpNameWidth[StringRef("arith.addi")];

    mapOpNameWidth[StringRef("arith.xori")] = mapOpNameWidth[StringRef("arith.addi")];


  }

void constructUpdateFuncMap(DenseMap<StringRef, 
                     std::function<std::vector<std::vector<unsigned int>> 
                                  (Operation::operand_range vecOperands, 
                                  Operation::result_range vecResults)>> 
                     &mapOpNameWidth){
  
  mapOpNameWidth[StringRef("arith.addi")] = 
    [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {
      std::vector<std::vector<unsigned>> widths; 
      
      unsigned int maxOpWidth = std::max(vecOperands[0].getType().getIntOrFloatBitWidth(), 
                                         vecOperands[1].getType().getIntOrFloatBitWidth());
                                
      unsigned int width = std::min(vecResults[0].getType().getIntOrFloatBitWidth(),
                           maxOpWidth+1);

      width = std::min(CPP_MAX_WIDTH, width);
      widths.push_back({width, width}); //matched widths for operators
      widths.push_back({width}); //matched widths for result
      
      return widths;
    };

  mapOpNameWidth[StringRef("arith.subi")] = mapOpNameWidth[StringRef("arith.addi")];
   
  mapOpNameWidth[StringRef("arith.muli")] = 
    [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {
      std::vector<std::vector<unsigned>> widths; 
      
      unsigned int maxOpWidth = vecOperands[0].getType().getIntOrFloatBitWidth() + 
                                vecOperands[1].getType().getIntOrFloatBitWidth();
                                
      unsigned int width = std::min(vecResults[0].getType().getIntOrFloatBitWidth(),
                           maxOpWidth);
      
      width = std::min(CPP_MAX_WIDTH, width);

      widths.push_back({width, width}); //matched widths for operators
      widths.push_back({width}); //matched widths for result
      
      return widths;
    };

    mapOpNameWidth[StringRef("arith.divsi")] = 
    [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {
      std::vector<std::vector<unsigned>> widths; 
      
      unsigned int maxOpWidth = vecOperands[0].getType().getIntOrFloatBitWidth();
                                
      unsigned int width = std::min(vecResults[0].getType().getIntOrFloatBitWidth(),
                           maxOpWidth+1);
      
      width = std::min(CPP_MAX_WIDTH, width);

      widths.push_back({width, width}); //matched widths for operators
      widths.push_back({width}); //matched widths for result
      
      return widths;
    };

    mapOpNameWidth[StringRef("arith.divui")] = mapOpNameWidth[StringRef("arith.divsi")];

    mapOpNameWidth[StringRef("arith.shli")] = 
    [&](Operation::operand_range vecOperands,
         Operation::result_range vecResults) {
      std::vector<std::vector<unsigned>> widths; 
      unsigned shift_bit = 0;
      if (auto defOp = vecOperands[1].getDefiningOp(); isa<handshake::ConstantOp>(defOp))
        if (handshake::ConstantOp cstOp = dyn_cast<handshake::ConstantOp>(defOp))
          if (auto IntAttr = cstOp.getValue().dyn_cast<mlir::IntegerAttr>())
            shift_bit = IntAttr.getValue().getZExtValue();
                                
      unsigned int width = std::min(std::min(CPP_MAX_WIDTH, vecResults[0].getType().getIntOrFloatBitWidth()), 
                                  vecOperands[0].getType().getIntOrFloatBitWidth() + shift_bit);

      width = std::min(CPP_MAX_WIDTH, width);
      widths.push_back({width, width}); //matched widths for operators
      widths.push_back({width}); //matched widths for result

      return widths;
    };

    mapOpNameWidth[StringRef("arith.shrsi")] = 
    [&](Operation::operand_range vecOperands,
         Operation::result_range vecResults) {
      std::vector<std::vector<unsigned>> widths; 
      unsigned shift_bit = 0;
      if (auto defOp = vecOperands[1].getDefiningOp(); isa<handshake::ConstantOp>(defOp))
        if (handshake::ConstantOp cstOp = dyn_cast<handshake::ConstantOp>(defOp))
          if (auto IntAttr = cstOp.getValue().dyn_cast<mlir::IntegerAttr>())
            shift_bit = IntAttr.getValue().getZExtValue();
                                
      unsigned int width = std::min(std::min(CPP_MAX_WIDTH, vecResults[0].getType().getIntOrFloatBitWidth()),
                                  vecOperands[0].getType().getIntOrFloatBitWidth() - shift_bit);

      width = std::min(CPP_MAX_WIDTH, width);
      widths.push_back({width, width}); //matched widths for operators
      widths.push_back({width}); //matched widths for result

      return widths;
    };

    mapOpNameWidth[StringRef("arith.shrui")] = mapOpNameWidth[StringRef("arith.shrsi")];

    mapOpNameWidth[StringRef("arith.cmpi")] = 
      [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {
      std::vector<std::vector<unsigned>> widths; 
      
      unsigned int maxOpWidth = std::max(vecOperands[0].getType().getIntOrFloatBitWidth(), 
                                         vecOperands[1].getType().getIntOrFloatBitWidth());
      
      unsigned int width = std::min(CPP_MAX_WIDTH, maxOpWidth);

      widths.push_back({width, width}); //matched widths for operators
      widths.push_back({unsigned(1)}); //matched widths for result
      
      return widths;
    };

    mapOpNameWidth[StringRef("handshake.mux")] = 
      [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {

      std::vector<std::vector<unsigned>> widths; 
      unsigned maxOpWidth = 2;

      unsigned ind = 0; // record number of operators

      for (auto oprand : vecOperands) {
        ind++;
        if (ind==0)
          continue; // skip the width of the index 
        
        if (!isa<NoneType>(oprand.getType()))
          if (!isa<IndexType>(oprand.getType()) && 
              oprand.getType().getIntOrFloatBitWidth() > maxOpWidth)
            maxOpWidth = oprand.getType().getIntOrFloatBitWidth();
      }
      unsigned indexWidth=2;
      if (ind>2)
        indexWidth = log2(ind-2)+2;

      widths.push_back({indexWidth}); // the bit width for the mux index result;

      if (isa<NoneType>(vecResults[0].getType())) {
        widths.push_back({});
        return widths;
      }

      unsigned int width = std::min(vecResults[0].getType().getIntOrFloatBitWidth(),
                                    std::min(CPP_MAX_WIDTH, maxOpWidth));
      // 1st operand is the index; rest of (ind -1) operands set to width
      std::vector<unsigned> opwidths(ind-1, width); 

      widths[0].insert(widths[0].end(), opwidths.begin(), opwidths.end()); //matched widths for operators
      widths.push_back({width}); //matched widths for result
      
      return widths;
    };

    mapOpNameWidth[StringRef("handshake.merge")] = 
      [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {

      std::vector<std::vector<unsigned>> widths; 
      unsigned maxOpWidth = 2;

      unsigned ind = 0; // record number of operators

      for (auto oprand : vecOperands) {
        ind++;
        if (!isa<NoneType>(vecOperands[0].getType()))
          if (!isa<IndexType>(oprand.getType()) && 
              oprand.getType().getIntOrFloatBitWidth() > maxOpWidth)
            maxOpWidth = oprand.getType().getIntOrFloatBitWidth();
      }

      if (isa<NoneType>(vecOperands[0].getType())) {
        widths.push_back({});
        widths.push_back({});
        return widths;
      }

      unsigned int width = std::min(vecResults[0].getType().getIntOrFloatBitWidth(),
                                    std::min(CPP_MAX_WIDTH, maxOpWidth));
      std::vector<unsigned> opwidths(ind, width);

      widths.push_back(opwidths); //matched widths for operators
      widths.push_back({width}); //matched widths for result
      
      return widths;
    };

    mapOpNameWidth[StringRef("handshake.constant")] = 
      [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {

          std::vector<std::vector<unsigned>> widths; 
          // Do not set the width of the input
          widths.push_back({});
          Operation* Op = vecResults[0].getDefiningOp();
          if (handshake::ConstantOp cstOp = dyn_cast<handshake::ConstantOp>(*Op))
            if (auto IntAttr = cstOp.getValueAttr().dyn_cast<mlir::IntegerAttr>())
              if (auto IntType = dyn_cast<IntegerType>(IntAttr.getType())){
                widths.push_back({IntType.getWidth()});
                return widths;
              }
            
          widths.push_back({});
          return widths;
    };

    mapOpNameWidth[StringRef("handshake.control_merge")] = 
      [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {

      std::vector<std::vector<unsigned>> widths; 
      unsigned maxOpWidth = 2;

      unsigned ind = 0; // record number of operators

      for (auto oprand : vecOperands) {
        ind++;
        if (!isa<NoneType>(oprand.getType()))
          if (!isa<IndexType>(oprand.getType()) && 
              oprand.getType().getIntOrFloatBitWidth() > maxOpWidth)
            maxOpWidth = oprand.getType().getIntOrFloatBitWidth();
      }

      unsigned indexWidth=1;
      if (ind>1)
        indexWidth = ceil(log2(ind));

      if (isa<NoneType>(vecOperands[0].getType())) {
        widths.push_back({});
        widths.push_back({0, indexWidth});
        return widths;
      }

      unsigned int width = std::min(CPP_MAX_WIDTH, maxOpWidth);
      std::vector<unsigned> opwidths(ind, width);

      widths.push_back(opwidths); //matched widths for operators
      widths.push_back({indexWidth}); //matched widths for result
      
      return widths;
    };

    mapOpNameWidth[StringRef("handshake.select")] = 
        [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {

      std::vector<std::vector<unsigned>> widths;
      
      unsigned ind=0, maxOpWidth=2 ;

      for (auto oprand : vecOperands) {
        ind++;
        if (ind==0)
          continue; // skip the width of the index 
        
        if (!isa<NoneType>(oprand.getType()))
          if (!isa<IndexType>(oprand.getType()) && 
              oprand.getType().getIntOrFloatBitWidth() > maxOpWidth)
            maxOpWidth = oprand.getType().getIntOrFloatBitWidth();
      }

      widths.push_back({1}); // bool like condition
      if (isa<NoneType>(vecOperands[1].getType())) {
        widths.push_back({});
        return widths;
      }

      std::vector<unsigned> opwidths(ind-1, maxOpWidth);

      widths[0].insert(widths[0].end(), opwidths.begin(), opwidths.end()); //matched widths for operators
      widths.push_back({maxOpWidth}); //matched widths for result

      return widths;
    };

    mapOpNameWidth[StringRef("handshake.d_return")] = 
      [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {

          std::vector<std::vector<unsigned>> widths; 
          widths.push_back({ADDRESS_WIDTH});
          if (!isa<NoneType>(vecResults[0].getType()))
            widths.push_back({ADDRESS_WIDTH});
          else 
            widths.push_back({});
          return widths;
    };

    mapOpNameWidth[StringRef("handshake.d_load")] = 
      [&] (Operation::operand_range vecOperands,
         Operation::result_range vecResults) {

          std::vector<std::vector<unsigned>> widths; 
          widths.push_back({CPP_MAX_WIDTH, ADDRESS_WIDTH});
          widths.push_back({CPP_MAX_WIDTH, ADDRESS_WIDTH});
          return widths;
    };

    mapOpNameWidth[StringRef("handshake.d_store")] = mapOpNameWidth[StringRef("handshake.d_load")];

};

  void setValidateType(Operation *Op,
                       bool &pass,
                       bool &match,
                       bool &revert) {

    pass   = false;
    match  = false;
    revert = false;
 
    if (isa<handshake::BranchOp>(*Op) ||
        isa<handshake::ConditionalBranchOp>(*Op))
        pass = true;

    if (isa<mlir::arith::AddIOp>(*Op)  ||
        isa<mlir::arith::SubIOp>(*Op)  ||
        isa<mlir::arith::MulIOp>(*Op)  ||
        isa<mlir::arith::ShLIOp>(*Op)  ||
        isa<mlir::arith::ShRSIOp>(*Op) ||
        isa<mlir::arith::ShRUIOp>(*Op) ||
        isa<mlir::arith::DivSIOp>(*Op) ||
        isa<mlir::arith::DivUIOp>(*Op) ||
        isa<mlir::arith::CmpIOp>(*Op)  ||
        isa<mlir::arith::ShRSIOp>(*Op) ||
        isa<handshake::MuxOp>(*Op)     ||
        isa<handshake::MergeOp>(*Op)   ||
        isa<handshake::SelectOp>(*Op)  ||
        isa<handshake::ConstantOp>(*Op)||
        isa<handshake::DynamaticLoadOp>(*Op) ||
        isa<handshake::DynamaticStoreOp>(*Op)||
        isa<handshake::ControlMergeOp>(*Op)  ||
        isa<handshake::DynamaticReturnOp>(*Op) )
      match = true;  

    if (isa<mlir::arith::TruncIOp>(*Op) ||
        isa<mlir::arith::ExtSIOp>(*Op) ||
        isa<mlir::arith::ExtUIOp>(*Op))
      revert = true;  
  }

  bool propType(Operation *op) {

    if (isa<handshake::ConditionalBranchOp>(*op)) {
      for (auto resOp : op->getResults())
        resOp.setType(op->getOperand(1).getType());
      return true;
    }

    if (isa<handshake::BranchOp>(*op)) {
      op->getResult(0).setType(op->getOperand(0).getType());
      return true;
    }
    return false;
  }

  void replaceWithSuccessor(Operation *Op) {
    Operation *sucNode = Op->getOperand(0).getDefiningOp();
      // llvm::errs() << "successor" << *sucNode << "\n";

      // find the index of result in vec_results
      unsigned ind = 0;
      for (auto Val : sucNode->getResults()) {

        if (Val==Op->getOperand(0))
          break;
        ind++;
      }

      SmallVector<Operation *> vecUsers;
      for (auto user : Op->getResult(0).getUsers())
        vecUsers.push_back(user);
      
      for (auto user : vecUsers)
        user->replaceUsesOfWith(Op->getResult(0), sucNode->getResult(ind));

  }

  void replaceWithSuccessor(Operation *Op, 
                            Type resType) {
    Operation *sucNode = Op->getOperand(0).getDefiningOp();

      // find the index of result in vec_results
      unsigned ind = 0;
      for (auto Val : sucNode->getResults()) {

        if (Val==Op->getOperand(0)){
          Val.setType(resType);
          break;
        }
          
        ind++;
      }
      // llvm::errs() << "successor" << *sucNode << "\n";

      SmallVector<Operation *> vecUsers;
      for (auto user : Op->getResult(0).getUsers())
        vecUsers.push_back(user);
      
      for (auto user : vecUsers)
        user->replaceUsesOfWith(Op->getResult(0), sucNode->getResult(ind));

  }


  void revertTruncOrExt(Operation *Op, MLIRContext *ctx) {
    OpBuilder builder(ctx);
    // if width(res) == width(opr) : delte the operand;

    if (Op->getResult(0).getType().getIntOrFloatBitWidth() ==
        Op->getOperand(0).getType().getIntOrFloatBitWidth()) {

      replaceWithSuccessor(Op);
      Op->erase();
      return;
    }

    // if for extension operation width(res) < width(opr),
    // change it to truncation operation
    if (isa<mlir::arith::ExtSIOp>(*Op) || isa<mlir::arith::ExtUIOp>(*Op))
      if (Op->getResult(0).getType().getIntOrFloatBitWidth() <
          Op->getOperand(0).getType().getIntOrFloatBitWidth()) {

        builder.setInsertionPoint(Op);
        Type newType = getNewType(Op->getResult(0), 
                                  Op->getResult(0).getType().getIntOrFloatBitWidth(), 
                                  false);
        auto truncOp = builder.create
                      <mlir::arith::TruncIOp>(Op->getLoc(), 
                                              newType,
                                              Op->getOperand(0));
        Op->getResult(0).replaceAllUsesWith(truncOp.getResult());
        Op->erase();
        return;

      }

    // if for truncation operation width(res) > width(opr),
    // change it to extension operation
    if (isa<mlir::arith::TruncIOp>(*Op))
      if (Op->getResult(0).getType().getIntOrFloatBitWidth() >
          Op->getOperand(0).getType().getIntOrFloatBitWidth()) {
            
        builder.setInsertionPoint(Op);
        Type newType = getNewType(Op->getResult(0), 
                                  Op->getResult(0).getType().getIntOrFloatBitWidth(), 
                                  false);
        auto truncOp = builder.create
                      <mlir::arith::ExtSIOp>(Op->getLoc(), 
                                              newType,
                                              Op->getOperand(0));
        Op->getResult(0).replaceAllUsesWith(truncOp.getResult());
        Op->erase();
      }
  }

  void matchOpResWidth (Operation *Op, 
                        MLIRContext *ctx, 
                        SmallVector<Operation *> &newMatchedOps) {

    DenseMap<mlir::StringRef,
               std::function<std::vector<std::vector<unsigned int>> 
                  (Operation::operand_range vecOperands, 
                   Operation::result_range vecResults)>> mapOpNameWidth;

    constructUpdateFuncMap(mapOpNameWidth);

    std::vector<std::vector<unsigned int> > OprsWidth = 
                                   mapOpNameWidth[Op->getName().getStringRef()]
                                   (Op->getOperands(), Op->getResults());
    // make operator matched the width
    for (size_t i = 0; i < OprsWidth[0].size(); ++i) {
      // llvm::errs() << "validate operator " << i << " : " << OprsWidth[0][i] << "\n";
      if (auto Operand = Op->getOperand(i); !isa<NoneType>(Operand.getType()) &&
          Operand.getType().getIntOrFloatBitWidth() != OprsWidth[0][i]) 
        {
        auto insertOp = insertWidthMatchOp(Op, 
                                           i, 
                                           getNewType(Operand, OprsWidth[0][i], false), 
                                           ctx); 
        if (insertOp.has_value())
          newMatchedOps.push_back(insertOp.value());
        }
        
    }
    // make result matched the width
    for (size_t i = 0; i < OprsWidth[1].size(); ++i) {
      // llvm::errs() << "validate result operator " << i << " : " << OprsWidth[1][i] << "\n";
      if (auto OpRes = Op->getResult(i); OprsWidth[1][i]!=0 &&
          OpRes.getType().getIntOrFloatBitWidth() != OprsWidth[1][i]) {
        Type newType = getNewType(OpRes, OprsWidth[1][i], false);
        Op->getResult(i).setType(newType) ;
      }
    }
  }

  void validateOp(Operation *Op, 
                  MLIRContext *ctx,
                  SmallVector<Operation *> &newMatchedOps) {
    // the operations can be divided to three types to make it validated
    // passType: branch, conditionalbranch
    // c <= op(a,b): addi, subi, mux, etc. where both a,b,c needed to be verified
    // need to be reverted or deleted : truncIOp, extIOp
    bool pass   = false;
    bool match  = false;
    bool revert = false;

    setValidateType(Op, pass, match, revert);

    if (pass)
      bool res = propType(Op);

    if (match)
      matchOpResWidth(Op, ctx, newMatchedOps);

    if (revert) 
      revertTruncOrExt(Op, ctx);


    
  }
}