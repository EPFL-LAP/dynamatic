//===- HandshakeCanonicalize.h - Canonicalize Handshake ops -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --name-all-operations pass and
// --remove-operation-names pass which, respectively, adds or removes unique
// names to/from all operations in the IR.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_OPERATIONNAMES_H
#define DYNAMATIC_TRANSFORMS_OPERATIONNAMES_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_NAMEALLOPERATIONS
#define GEN_PASS_DEF_NAMEALLOPERATIONS
#define GEN_PASS_DECL_REMOVEOPERATIONNAMES
#define GEN_PASS_DEF_REMOVEOPERATIONNAMES
#include "dynamatic/Transforms/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createNameAllOperations();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRemoveOperationNames();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_OPERATIONNAMES_H