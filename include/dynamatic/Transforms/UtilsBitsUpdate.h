#ifndef DYNAMATIC_TRANSFORMS_UTILSBITSUPDATE_H
#define DYNAMATIC_TRANSFORMS_UTILSBITSUPDATE_H

#include <optional>
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "dynamatic/Support/LLVM.h"


using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace dynamatic;


const unsigned cpp_max_width = 64;
const unsigned arith_max_width = 32;
const unsigned address_width = 32;

IntegerType getNewType(Value opType, unsigned bitswidth, bool signless=false);
IntegerType getNewType(Value opType, unsigned bitswidth,  
                      IntegerType::SignednessSemantics ifSign);
void constructFuncMap(DenseMap<StringRef, 
                     std::function<unsigned (Operation::operand_range vecOperands)>> 
                     &mapOpNameWidth);
void setUserType(Operation *newOp, Type newType,
                          MLIRContext *ctx, SmallVector<int> vecIndex);
std::optional<Operation *> insertWidthMatchOp(Operation *newOp, int opInd, Type newType, MLIRContext *ctx);
void updateUserType(Operation *newResult, Type newType, SmallVector<Operation *> &vecOp, MLIRContext *ctx);

#endif