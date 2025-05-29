//===- Passes.td - Exp. analysis passes definition ---*- tablegen -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of all experimental analysis passes.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ANALYSIS_PASSES_TD
#define EXPERIMENTAL_ANALYSIS_PASSES_TD

include "dynamatic/Support/Passes.td"
include "mlir/Pass/PassBase.td"

def HandshakeAnnotateProperties : DynamaticPass< "handshake-annotate-properties"> {
  let summary = "Annotates Formal Properties into a json database";
  let description = [{
    This analysis pass generates a table that contains all the information needed to
    define and check a set of properties. Currently it only annotates Absence of
    Backpressure and Valid Equivalence, but it can be extended to annotate more.
  }];
  let constructor = "dynamatic::experimental::formalprop::createAnnotateProperties()";
  let options = [ Option<"jsonPath", "json-path", "std::string", "",
    "Path to JSON-formatted file where the properties' information is stored.">];
}

#endif // EXPERIMENTAL_ANALYSIS_PASSES_TD
