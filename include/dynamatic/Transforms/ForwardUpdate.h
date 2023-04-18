//===- ForwardUpdate.h ---------*- C++ -*-===//
//
// This file declares functions for forward pass in --optimize-bits.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_FORWARDUPDATE_H
#define DYNAMATIC_TRANSFORMS_FORWARDUPDATE_H

#include "dynamatic/Transforms/UtilsBitsUpdate.h"

namespace forward {

void constructFuncMap(DenseMap<StringRef, 
                     std::function<unsigned (Operation::operand_range vecOperands)>> 
                     &mapOpNameWidth);
void setUpdateFlag(Operation *newResult, bool &passType, bool &oprAdapt, bool &resAdapter, bool &deleteOp);
void updateUserType(Operation *newResult, Type newType, SmallVector<Operation *> &vecOp, MLIRContext *ctx);

}
#endif // DYNAMATIC_TRANSFORMS_FORWARDUPDATE_H