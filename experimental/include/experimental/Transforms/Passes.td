//===- Passes.td - Exp. transformation passes definition ---*- tablegen -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of all experimental transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TRANSFORMS_PASSES_TD
#define EXPERIMENTAL_TRANSFORMS_PASSES_TD

include "dynamatic/Support/Passes.td"
include "mlir/Pass/PassBase.td"

def HandshakeFixArgNames : DynamaticPass<"handshake-fix-arg-names"> {
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