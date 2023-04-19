//===- BackwardUpdate.h ---------*- C++ -*-===//
//
// This file declares functions for backward pass in --optimize-bits.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BACKWARDUPDATE_H
#define DYNAMATIC_TRANSFORMS_BACKWARDUPDATE_H

#include "dynamatic/Transforms/UtilsBitsUpdate.h"

namespace backward {

void constructFuncMap(DenseMap<StringRef, 
                     std::function<unsigned (Operation::operand_range vecOperands)>> 
                     &mapOpNameWidth);

void setUpdateFlag(Operation *newResult, 
                  bool &passType, 
                  bool &oprAdapt, 
                  bool &resAdapter, 
                  bool &deleteOp);
                  
void updateDefOpType(Operation *newResult, 
                    Type newType, 
                    SmallVector<Operation *> &vecOp, 
                    MLIRContext *ctx);

}
#endif // DYNAMATIC_TRANSFORMS_BACKWARDUPDATE_H