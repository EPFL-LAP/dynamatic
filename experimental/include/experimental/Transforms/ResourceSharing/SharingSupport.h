//===- SharingSupport.h - Resource Sharing Utilities-----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H
#define EXPERIMENTAL_INCLUDE_DYNAMATIC_TRANSFORMS_RESOURCESHARING_SHARINGSUPPORT_H

#include "dynamatic/Support/TimingModels.h"
#include <vector>

// list of types that can be shared
#define SHARING_TARGETS                                                        \
  mlir::arith::MulFOp, mlir::arith::MulIOp, mlir::arith::AddFOp,               \
      mlir::arith::SubFOp

#endif