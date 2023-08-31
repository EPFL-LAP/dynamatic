//===- Passes.td - Exp. conversion passes definition --------*- tablegen -*-===//
//
// This file contains the definition for all experimental conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_CONVERSION_PASSES_TD
#define EXPERIMENTAL_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def StandardToHandshakeFPL22 : Pass< "exp-lower-std-to-handshake-fpl22", 
                                "mlir::ModuleOp"> {
  let summary = "Lower Standard MLIR into Handshake IR following elastic pass described in FPL22";
  let constructor = "dynamatic::experimental::createStandardToHandshakeFPL22Pass()";
  let dependentDialects = ["circt::handshake::HandshakeDialect"];
  let options =
    [Option<"idBasicBlocks", "id-basic-blocks", "bool", "false",
            "If true, will attach a 'bb' attribute to each operation in the "
            "resulting IR to identify the basic block from which the "
            "operation originates.">];
}

#endif // EXPERIMENTAL_CONVERSION_PASSES_TD