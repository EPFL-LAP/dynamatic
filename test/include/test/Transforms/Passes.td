//===- Passes.td - Test passes definition ------------------*- tablegen -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the test passes, which exercise parts of
// Dynamatic from lit tests.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TRANSFORMS_PASSES_TD
#define TEST_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TestRTLSupport : Pass<"test-rtl-support", "mlir::ModuleOp"> {
  let summary = "Test RTL support (RTL configuration file parsing)";
  let description = [{
    Parses an RTL configuration file and checks that every external hardware
    module in the IR has a matching component in it.
  }];
  let options = [
    Option<"rtlConfigPath", "rtl-config-path", "std::string", "\"\"",
           "Path to the JSON-formatted RTL configuration">
  ];
}

#endif // TEST_TRANSFORMS_PASSES_TD
