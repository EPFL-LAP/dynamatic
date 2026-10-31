//===- TestHandshakeSimulator.cpp - Handshake simulator tests  ---- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass for Handhake simulator. Run with --exp-test-handshake-simulator.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"

// [START Boilerplate code for the MLIR pass]
#include "experimental/test/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
namespace test {
#define GEN_PASS_DEF_TESTHANDSHAKESIMULATOR
#include "experimental/test/Transforms/Passes.h.inc"
} // namespace test
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;

namespace {

struct TestHandshakeSimulator
    : public dynamatic::experimental::test::impl::TestHandshakeSimulatorBase<
          TestHandshakeSimulator> {
  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();

    // Retrieve the single Handshake function
    auto allFunctions = modOp.getOps<handshake::FuncOp>();
    if (std::distance(allFunctions.begin(), allFunctions.end()) != 1) {
      llvm::errs() << "Expected single Handshake function\n";
      return signalPassFailure();
    }
  }
};
} // namespace
