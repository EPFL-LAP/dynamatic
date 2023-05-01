//===- ForwardUpdate.cpp ---------*- C++ -*-===//
//
// This file declares functions for forward pass in --optimize-bits.
//
//===----------------------------------------------------------------------===//
#include "dynamatic/Transforms/ForwardUpdate.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace forward {

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

    unsigned indexWidth=2;
    if (ind>2)
      indexWidth = log2(ind-2)+2;

    return indexWidth;
  };

  // mapOpNameWidth[StringRef("handshake.control_merge")] = mapOpNameWidth[StringRef("handshake.mux")];


  };



}