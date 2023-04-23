//===- UtilsBitsUpdate.h ---------*- C++ -*-===//
//
// This file declares basic functions for type updates for --optimize-bits pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_UTILSBITSUPDATE_H
#define DYNAMATIC_TRANSFORMS_UTILSBITSUPDATE_H

#include <optional>
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"


using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace dynamatic;


const unsigned cpp_max_width = 64;
// const unsigned arith_max_width = 32;
const unsigned address_width = 32;

IntegerType getNewType (Value opType, 
                        unsigned bitswidth, 
                        bool signless=false);

IntegerType getNewType (Value opType, 
                        unsigned bitswidth,  
                        IntegerType::SignednessSemantics ifSign);

std::optional<Operation *> insertWidthMatchOp (Operation *newOp, 
                                               int opInd, 
                                               Type newType, 
                                               MLIRContext *ctx);
                  
namespace update {

  void constructFuncMap(DenseMap<mlir::StringRef, 
                      std::function<unsigned (Operation::operand_range vecOperands)>> 
                      &mapOpNameWidth);
  
  void validateOp(Operation *Op, MLIRContext *ctx);

  bool propType(Operation *Op);

  void matchOpResWidth (Operation *Op, MLIRContext *ctx);

  void replaceWithSuccessor(Operation *Op);

  void revertTruncOrExt(Operation *Op, MLIRContext *ctx);

  void setValidateType(Operation *Op,
                       bool &passtype,
                       bool &match,
                       bool &revert);
                    
  // void updateUserType(Operation *newResult, 
  //                   Type newType, 
  //                   SmallVector<Operation *> &vecOp, 
  //                   MLIRContext *ctx);

}

#endif //DYNAMATIC_TRANSFORMS_UTILSBITSUPDATE_H