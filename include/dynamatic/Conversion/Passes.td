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

def AffineToScf : DynamaticPass<"lower-affine-to-scf",
                                [ "mlir::memref::MemRefDialect", 
                                  "mlir::scf::SCFDialect", 
                                  "mlir::vector::VectorDialect" ]> {
  let summary = "Lower affine operations to a mix of SCF and standard operations";
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
// StandardToHandshakeFPGA18
//===----------------------------------------------------------------------===//

def StandardToHandshakeFPGA18 : DynamaticPass<"lower-std-to-handshake-fpga18"> {
  let summary = "Lowers cf-level IR to Handshake.";
  let constructor = "dynamatic::createStandardToHandshakeFPGA18Pass()";
}

//===----------------------------------------------------------------------===//
// HandshakeToNetlist
//===----------------------------------------------------------------------===//

def HandshakeToNetlist : DynamaticPass< "lower-handshake-to-netlist",
                                        [ "circt::hw::HWDialect",  
                                          "circt::esi::ESIDialect" ]> {
  let summary = "Lower Handshake to ESI/HW (netlist)";
  let description = [{
    Lower Handshake IR into a "netlist-level" representation made up of HW
    instances of external modules interconnected by ESI channels.
  }];
  let constructor = "dynamatic::createHandshakeToNetlistPass()";
}

#endif // DYNAMATIC_CONVERSION_PASSES_TD
