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

#endif // EXPERIMENTAL_TEST_TRANSFORMS_PASSES_TD
