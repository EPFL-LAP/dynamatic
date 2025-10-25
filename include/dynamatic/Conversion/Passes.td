//===-- Passes.td - Conversion passes definition ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for all dialect conversion and export passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_PASSES_TD
#define DYNAMATIC_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"
include "dynamatic/Support/Passes.td"

//===----------------------------------------------------------------------===//
// AffineToScf
//===----------------------------------------------------------------------===//

def AffineToScf : DynamaticPass<"lower-affine-to-scf", [
  "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect",
  "mlir::vector::VectorDialect"
]> {
  let summary =
      "Lower affine operations to a mix of SCF and standard operations";
  let description = [{
    Very close analog to the AffineToStandard pass from MLIR that preserves the
    attributes of AffineLoadOp's and AffineStoreOp's during lowering. Keeping
    these attributes is required for our memory analysis pass to pass down
    dependence information between memory accesses to lower IR levels, in
    particular to the std-to-handshake lowering pass. 
  }];
  let constructor = "dynamatic::createAffineToScfPass()";
}

//===----------------------------------------------------------------------===//
// ScfToCf
//===----------------------------------------------------------------------===//

def ScfToCf : DynamaticPass<"lower-scf-to-cf", [
    "mlir::cf::ControlFlowDialect", "mlir::arith::ArithDialect"]> {
  let summary = "Lower scf dialect to unstructured control flow (cf)";
  let description = [{
    Very close analog to the SCFToControlFlow pass from MLIR that replaces the
    structured for loop lowering pattern with an almost identical one that
    additionally attempts to insert an unsigned comparison (ult) in the IR
    instead of a signed one (lt) if the loop's iterator can be proven to be
    always positive.
  }];
  let constructor = "dynamatic::createLowerScfToCf()";
}

//===----------------------------------------------------------------------===//
// CfToHandshake
//===----------------------------------------------------------------------===//

def CfToHandshake : DynamaticPass<"lower-cf-to-handshake"> {
  let summary = "Lowers func and cf dialects to handshake.";
  let description = [{
    Lowers func-level functions whose body have unstructured control flow into
      their handshake counterparts,
      which represent dataflow circuits that can ultimately be converted to an
          RTL design.
}];
let constructor = "dynamatic::createCfToHandshake()";
}

//===----------------------------------------------------------------------===//
// HandshakeToHW
//===----------------------------------------------------------------------===//

def HandshakeToHW
    : DynamaticPass<"lower-handshake-to-hw", ["dynamatic::hw::HWDialect"]> {
  let summary = "Lowers Handshake to HW.";
  let description = [{
    Lowers Handshake IR into a "netlist-style" representation made up of HW
    instances of external modules interconnected by dataflow channels.
  }];
  let constructor = "dynamatic::createHandshakeToHWPass()";
}
  
#endif // DYNAMATIC_CONVERSION_PASSES_TD
