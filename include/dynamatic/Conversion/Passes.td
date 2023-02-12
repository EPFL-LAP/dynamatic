//===-- Passes.td - Conversion passes definition ----------------*- C++ -*-===//
//
// This file contains definitions for all dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_PASSES_TD
#define DYNAMATIC_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// StandardToHandshakeFPGA18
//===----------------------------------------------------------------------===//

def StandardToHandshakeFPGA18 : Pass<"lower-std-to-handshake-fpga18", "mlir::ModuleOp"> {
  let summary = "Lower Standard MLIR into Handshake IR following elastic pass described in FPGA18";
  let constructor = "dynamatic::createStandardToHandshakeFPGA18Pass()";
  let dependentDialects = ["circt::handshake::HandshakeDialect"];
}

#endif // DYNAMATIC_CONVERSION_PASSES_TD
