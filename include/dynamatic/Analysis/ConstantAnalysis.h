//===- ConstantAnalysis.h - Constant analyis utilities ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares some utility functions useful to analyzing and handling Handshake
// constants (dynamatic::handshake::ConstantOp) in dataflow circuits.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace dynamatic {

/// Attempts to find an equivalent constant to the one provided in the circuit,
/// that is a constant with the same control and same value attribute. Returns
/// such an equivalent constant if it finds one, nullptr otherwise.
handshake::ConstantOp findEquivalentCst(handshake::ConstantOp cstOp);

/// During std-to-handshake lowering, determines whether it is possible to
/// transform an arith-level constant into a Handsahke-level constant that is
/// triggered by an always-triggering source component without compromising the
/// circuit semantics (e.g., without triggering a memory operation before the
/// circuit "starts"). Returns false if the Handshake-level constant that
/// replaces the input must instead be connected to the control-only network;
/// returns true otherwise. This function assumes that the rest of the std-level
/// operations have already been converted to their Handshake equivalent.
bool isCstSourcable(mlir::arith::ConstantOp cstOp);

} // namespace dynamatic