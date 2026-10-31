//===- Passes.td - Experimental test passes definition -----*- tablegen -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the experimental test passes, which
// exercise parts of the experimental work from lit tests.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TEST_TRANSFORMS_PASSES_TD
#define EXPERIMENTAL_TEST_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TestHandshakeSimulator : Pass<"exp-test-handshake-simulator",
                                  "mlir::ModuleOp"> {
  let summary = "Test the Handshake simulator";
}

def TestPrintFormalProperties : Pass<"exp-test-print-formal-properties",
                                     "mlir::ModuleOp"> {
  let summary = "Print formal properties in SMV syntax";
  let description = [{
    Loads the formal properties of a JSON file, as written by
    --handshake-annotate-properties, and prints each one on a line of the
    standard output as its kind followed by its expression in SMV syntax.
  }];
  let options = [
    Option<"jsonPath", "json-path", "std::string", "\"\"",
           "Path to the JSON-formatted formal properties">
  ];
}

#endif // EXPERIMENTAL_TEST_TRANSFORMS_PASSES_TD
