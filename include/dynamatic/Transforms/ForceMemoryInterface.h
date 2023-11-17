//===- ForceMemoryInterface.h - Force interface in Handshake ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --force-memory-interface pass which allows one to
// force the type of memory interface that is placed during Handshake lowering.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_FORCEMEMORYINTERFACE_H
#define DYNAMATIC_TRANSFORMS_FORCEMEMORYINTERFACE_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_FORCEMEMORYINTERFACE
#define GEN_PASS_DEF_FORCEMEMORYINTERFACE
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createForceMemoryInterface(bool forceLSQ = false, bool forceMC = false);

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_FORCEMEMORYINTERFACE_H