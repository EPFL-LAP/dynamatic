//===- Passes.td - Exp. transformation passes definition ---*- tablegen -*-===//
//
// This file contains the definition of all experimental transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_PASSES_TD
#define EXPERIMENTAL_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def HandshakeFixArgNames : Pass<"handshake-fix-arg-names", "mlir::ModuleOp"> {
  let summary = "Replace function argument names with original ones from C";
  let description = [{
    Extremely hacky pass that text parses the original source code from which
    the IR was obtained and replaces the default argument names given by
    Polygeist to all function arguments with the ones used in the source code.
  }];
  let constructor = "dynamatic::experimental::createHandshakeFixArgNames()";
  let options =
    [Option<"source", "source", "std::string", "",
            "Path to source C file from which the IR was generated.">];
}

#endif // EXPERIMENTAL_TRANSFORMS_PASSES_TD