//===- TestRTLSupport.cpp - Test pass for RTL support  ------------ C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass for RTL support. Run with --test-rtl-support.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/RTL/RTL.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace dynamatic;

namespace {

struct TestRTLSupportOptions {
  std::string rtlConfigPath;
};

struct TestRTLSupport
    : public PassWrapper<TestRTLSupport, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestRTLSupport)
  using Base = PassWrapper<TestRTLSupport, OperationPass<mlir::ModuleOp>>;

  TestRTLSupport() : Base() {};
  TestRTLSupport(const TestRTLSupport &other) : Base(other) {};

  StringRef getArgument() const final { return "test-rtl-support"; }

  StringRef getDescription() const final {
    return "Test RTL support (RTL configuration file parsing)";
  }

  void runOnOperation() override {
    RTLConfiguration config;
    if (failed(config.addComponentsFromJSON(rtlConfigPath)))
      return signalPassFailure();

    for (auto modOp : getOperation().getOps<hw::HWModuleExternOp>()) {
      RTLRequestFromHWModule request(modOp);
      if (!config.hasMatchingComponent(request)) {
        modOp->emitError() << "no matching component";
        return signalPassFailure();
      }
    }
  }

  TestRTLSupport(const TestRTLSupportOptions &options) : TestRTLSupport() {
    rtlConfigPath = options.rtlConfigPath;
  }

protected:
  Pass::Option<std::string> rtlConfigPath{
      *this, "rtl-config-path",
      ::llvm::cl::desc("Pass to JSON-formatted RTL configuration"),
      ::llvm::cl::init("")};
};
} // namespace

namespace dynamatic {
namespace test {
void registerTestRTLSuppport() { PassRegistration<TestRTLSupport>(); }
} // namespace test
} // namespace dynamatic