//===-- HandshakePasses.td - Handshake passes definition ----*- tablegen -*-===//
//
// This file contains definitions for passes that work on the handshake dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKEPASSES_TD
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKEPASSES_TD

include "mlir/Pass/PassBase.td"

def HandshakeToDot : Pass<"handshake-to-dot", "mlir::ModuleOp"> {
  let summary = "Generate .dot graph of a handshake function.";
  let description = [{
    This pass analyses a handshake.func operation and produces a .dot graph of
    its structure. If multiple functions are present in the IR, the top level
    function will be printed, and called functions will be subgraphs within the
    main graph.
  }];
  let constructor = "dynamatic::createHandshakeToDotPass()";
}

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKEPASSES_TD