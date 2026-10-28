//===- Passes.td - Exp. analysis passes definition ---*- tablegen -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of all analysis passes.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_ANALYSIS_PASSES_TD
#define DYNAMATIC_ANALYSIS_PASSES_TD

include "dynamatic/Support/Passes.td"
include "mlir/Pass/PassBase.td"

def HandshakeAnnotateProperties : Pass<"handshake-annotate-properties", "mlir::ModuleOp"> {
  let summary = "Annotates Formal Properties into a json database";
  let description = [{
    This analysis pass generates a table that contains all the information needed to
    define and check a set of properties. Currently it only annotates Absence of
    Backpressure and Valid Equivalence, but it can be extended to annotate more.
  }];
  let options = [ Option<"jsonPath", "json-path", "std::string", "",
    "Path to JSON-formatted file where the properties' information is stored.">,
    Option<"annotateInvariants", "annotate-invariants", "bool", "false",
    "Annotate invariants to assist k-inductive proofs">,
    Option<"annotateProperties", "annotate-properties", "bool", "true",
    "Annotate properties for optimizations">,
    Option<"annotateList", "annotate-list", "std::string", "",
    "List of properties to annotate (overwrites annotate-properties/invariants)">
    ];
  let dependentDialects = ["dynamatic::handshake::HandshakeDialect"];
}

#endif // DYNAMATIC_ANALYSIS_PASSES_TD
