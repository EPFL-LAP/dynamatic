//===- PushConstants.h - Push constants in using blocks ---------*- C++ -*-===//
//
// This file declares the --push-constants pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_PUSHCONSTANTS_H
#define DYNAMATIC_TRANSFORMS_PUSHCONSTANTS_H

#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createArithPushConstants();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_PUSHCONSTANTS_H
