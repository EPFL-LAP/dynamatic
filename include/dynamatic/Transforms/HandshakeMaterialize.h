//===- HandshakeMaterialize.h - Materialize Handshake IR --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-materialize pass, which ensures that every
// SSA value within Handshake functions is used exactly once by inserting forks
// and sinks as needed to, respectively, "duplicate" values and "absorb" others.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_HANDSHAKEMATERIALIZE_H
#define DYNAMATIC_TRANSFORMS_HANDSHAKEMATERIALIZE_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKEMATERIALIZE
#define GEN_PASS_DEF_HANDSHAKEMATERIALIZE
#include "dynamatic/Transforms/Passes.h.inc"

/// Error message to display when the Handshake function is not materialized.
constexpr llvm::StringLiteral ERR_NON_MATERIALIZED_FUNC(
    "This step requires that all values in the Handshake functions are used "
    "exactly once. Run the --handshake-materialize pass prior to this step to "
    "ensure that the Handshake function is materialized.");

/// Verifies whether the Handshake function is materialized i.e., whether every
/// SSA value in the function (function arguments or operation result) has
/// exactly one use. Fails and emits an error if a value does not have exactly
/// one use.
LogicalResult verifyIRMaterialized(handshake::FuncOp funcOp);

/// Error message to display when the MLIR module is not materialized.
constexpr llvm::StringLiteral ERR_NON_MATERIALIZED_MOD(
    "This step requires that all values within Handshake functions of the "
    "MLIR module are used exactly once. Run the --handshake-materialize "
    "pass prior to this step to ensure that the MLIR module is "
    "materialized.");

/// Verifies whether the module is materialized i.e., whether every Handshake
/// function in the module is materialized. Fails and emits an error if a value
/// does not have exactly one use within a Handshake function.
LogicalResult verifyIRMaterialized(mlir::ModuleOp modOp);

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeMaterialize();

} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_HANDSHAKEMATERIALIZE_H