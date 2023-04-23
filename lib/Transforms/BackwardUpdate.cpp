//===- BackwardUpdate.cpp ---------*- C++ -*-===//
//
// This file declares functions for backward pass in --optimize-bits.
//
//===----------------------------------------------------------------------===//
#include "dynamatic/Transforms/BackwardUpdate.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace backward {

  void constructFuncMap(DenseMap<StringRef, 
                        std::function<unsigned (Operation::result_range vecResults)>> 
                        &mapOpNameWidth)
  {
    mapOpNameWidth[StringRef("arith.addi")] = [](Operation::result_range vecResults)
    {
      return std::min(cpp_max_width, 
                      vecResults[0].getType().getIntOrFloatBitWidth());
    };

    mapOpNameWidth[StringRef("arith.subi")] = mapOpNameWidth[StringRef("arith.addi")];

    mapOpNameWidth[StringRef("arith.muli")] = mapOpNameWidth[StringRef("arith.addi")];

    mapOpNameWidth[StringRef("arith.andi")] = mapOpNameWidth[StringRef("arith.addi")];

    mapOpNameWidth[StringRef("arith.ori")] = mapOpNameWidth[StringRef("arith.addi")];

    mapOpNameWidth[StringRef("arith.xori")] = mapOpNameWidth[StringRef("arith.addi")];

  }

}

//   void setUpdateFlag(Operation *newResult, 
//                      bool &passType, 
//                      bool &oprAdapt, 
//                      bool &resAdapt, 
//                      bool &deleteOp,
//                      SmallVector<int> &vecOpIndex,
//                      SmallVector<int> &vecResIndex)
//   {
//     passType = false;
//     oprAdapt = false;
//     resAdapt = false;
//     deleteOp = false;
//     vecResIndex = {0};
//     if (isa<handshake::BranchOp>(newResult)) {
//       passType = true;
//       resAdapt = true;
//       vecOpIndex = {0};
//     } 
//     else if (isa<handshake::ConditionalBranchOp>(newResult)) {
//       passType = true;
//       resAdapt = true;
//       vecOpIndex = {1};
//       vecResIndex.push_back(1);
//     }
//     else if (isa<handshake::MergeOp>(newResult)) {
//       passType = true;
//       resAdapt = true;
//       for (int i=0; i<newResult->getNumOperands(); ++i)
//         vecOpIndex.push_back(i);
//     }
//     else if (isa<handshake::MuxOp>(newResult)) {
//       passType = true;
//       resAdapt = true;
//       vecOpIndex = {1, 2};
//     }
//     else if (isa<handshake::DynamaticStoreOp>(newResult)) {
//       // resAdapt = true;
//       oprAdapt = true;
//     }
//     else if (isa<handshake::DynamaticLoadOp>(newResult)) {
//       // resAdapt = true;
//       oprAdapt = true;
//     }
//     else if (isa<handshake::DynamaticReturnOp>(newResult)) {
//       oprAdapt = true;
//     }
//     else if (isa<mlir::arith::AddIOp>(newResult)) {
//       passType = true;
//       resAdapt = true;
//       vecOpIndex = {0, 1};
//     }
    
//   }

//   void updateDefOpType(Operation *newResult, 
//                        Type newType, 
//                        SmallVector<Operation *> &vecOp,
//                        MLIRContext *ctx)
//   {
//     // if the operation has been processed, return
//     if (std::find(vecOp.begin(), vecOp.end(), newResult) != vecOp.end())
//       return;
    

//     vecOp.push_back(newResult);
//     OpBuilder builder(ctx);
//     DenseMap<StringRef, std::function<unsigned (mlir::Operation::result_range vecResults)>>  mapOpNameWidth;

//     constructFuncMap(mapOpNameWidth); 
//     SmallVector<int> vecOpIndex;
//     SmallVector<int> vecResIndex;
//     // bit width of the operand should not exceeds the result width
//     // bool lessThanRes = false;
//     // only update the resultOp recursively with updated UserType : newType
//     bool passType = false; 
//     // insert extOp|truncOp to make the multiple operands have the same width
//     bool oprAdapt = false; 
//     // set the OpResult type to make the result have the same width as the operands
//     bool resAdapt = false; 
//     // delete the extOp|TruncIOp when the width of its result and operand are not consistent
//     bool deleteOp = false;
//     bool isLdSt = false;
    
//     // TODO : functions for determine the four flags : passType, oprAdapt, resAdapter, deleteOp
//     setUpdateFlag(newResult, 
//                   passType, 
//                   oprAdapt, 
//                   resAdapt, 
//                   deleteOp, 
//                   vecOpIndex,
//                   vecResIndex);

//     if (passType) {
//       // setDefOpType(newResult, newType, vecOpIndex);
//       llvm::errs() << newResult->getResult(0) << "\n";

//       for (int opInd : vecOpIndex) 
//         if (auto operand = newResult->getOperand(opInd);
//           !isa<handshake::ConstantOp>(operand.getDefiningOp()) )
//           updateDefOpType(operand.getDefiningOp(),
//                           newType,
//                           vecOp,
//                           ctx);
//     }

//     if (resAdapt)
//       for (int opInd : vecResIndex)
//         newResult->getResult(opInd).setType(newType);
    

//     if (oprAdapt) {
//       unsigned int startInd = 0;
//       unsigned opWidth;

//       if (isa<handshake::DynamaticLoadOp>(newResult) ||
//       isa<handshake::DynamaticStoreOp>(newResult)) {
//         isLdSt = true;
//         opWidth = cpp_max_width;
//       }

//       for (unsigned int i = startInd; i < newResult->getNumOperands(); ++i) {
//       // width of data operand for Load and Store op 
//       if (isLdSt && i==1)
//         opWidth = address_width;

//       // opWidth = 0 indicates the operand is none type operand, skip matched width insertion
//       if (auto Operand = newResult->getOperand(i) ; opWidth != 0){
//         auto insertOp = insertWidthMatchOp(newResult, 
//                                           i, 
//                                           getNewType(Operand, opWidth, false), 
//                                           ctx);
//         if (insertOp.has_value())
//           vecOp.push_back(insertOp.value());
//       }
//     }
//     }

//   }
// }

