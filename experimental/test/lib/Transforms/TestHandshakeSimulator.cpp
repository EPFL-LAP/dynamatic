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
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace dynamatic;

namespace {

struct TestHandshakeSimulatorOptions {};

struct TestHandshakeSimulator
    : public PassWrapper<TestHandshakeSimulator,
                         OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHandshakeSimulator)
  using Base =
      PassWrapper<TestHandshakeSimulator, OperationPass<mlir::ModuleOp>>;

  TestHandshakeSimulator() : Base() {};
  TestHandshakeSimulator(const TestHandshakeSimulator &other) = default;

  StringRef getArgument() const final { return "exp-test-handshake-simulator"; }

  StringRef getDescription() const final {
    return "Test the Handshake simulator";
  }

  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();

    // Retrieve the single Handshake function
    auto allFunctions = modOp.getOps<handshake::FuncOp>();
    if (std::distance(allFunctions.begin(), allFunctions.end()) != 1) {
      llvm::errs() << "Expected single Handshake function\n";
      return signalPassFailure();
    }

    handshake::FuncOp funcOp = *allFunctions.begin();
  }

  TestHandshakeSimulator(const TestHandshakeSimulatorOptions &options)
      : TestHandshakeSimulator() {}
};
} // namespace

namespace dynamatic {
namespace experimental {
namespace test {
void registerTestHandshakeSimulator() {
  PassRegistration<TestHandshakeSimulator>();
}
} // namespace test
} // namespace experimental
} // namespace dynamatic
