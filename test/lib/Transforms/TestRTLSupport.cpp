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

// [START Boilerplate code for the MLIR pass]
#include "test/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace test {
#define GEN_PASS_DEF_TESTRTLSUPPORT
#include "test/Transforms/Passes.h.inc"
} // namespace test
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;

namespace {

struct TestRTLSupport
    : public dynamatic::test::impl::TestRTLSupportBase<TestRTLSupport> {
  using TestRTLSupportBase::TestRTLSupportBase;

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
};
} // namespace
