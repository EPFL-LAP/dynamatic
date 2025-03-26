//===- TestRigidificationSupport.cpp - rigidification support ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declaration of the rigidification functionality that eliminates some control
// signals to reduce the handshake overhead
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/RigidificationSupport.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace dynamatic;

namespace {

struct TestRigidificationSupportOptions {};

struct TestRigidificationSupport
    : public PassWrapper<TestRigidificationSupport,
                         OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRigidificationSupport)
  using Base =
      PassWrapper<TestRigidificationSupport, OperationPass<mlir::ModuleOp>>;

  TestRigidificationSupport() : Base() {};
  TestRigidificationSupport(const TestRigidificationSupport &other) = default;

  StringRef getArgument() const final { return "exp-test-handshake-simulator"; }

  StringRef getDescription() const final {
    return "Test the Handshake simulator";
  }

  void runOnOperation() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Retrieve the single Handshake function
    auto allFunctions = modOp.getOps<handshake::FuncOp>();
    if (std::distance(allFunctions.begin(), allFunctions.end()) != 1) {
      llvm::errs() << "Expected single Handshake function\n";
      return signalPassFailure();
    }

    // handshake::FuncOp funcOp = *allFunctions.begin();

    getOperation()->walk([&](Operation *op) {
      for (auto ch : op->getResults()) {
        Type opType = ch.getType();
        if (llvm::dyn_cast<handshake::ChannelType>(opType))
          rigidifyChannel(ch, ctx);
      }
    });
  }

  TestRigidificationSupport(const TestRigidificationSupportOptions &options)
      : TestRigidificationSupport() {}
};
} // namespace

namespace dynamatic {
namespace experimental {
namespace test {
void registerTestRigidificationSupport() {
  PassRegistration<TestRigidificationSupport>();
}
} // namespace test
} // namespace experimental
} // namespace dynamatic
