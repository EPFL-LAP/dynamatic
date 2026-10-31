//===- TestPrintFormalProperties.cpp - Print formal properties --*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass that prints formal properties in SMV syntax, so that lit tests can
// check the generated properties one line each. Run with
// --exp-test-print-formal-properties.
//
//===----------------------------------------------------------------------===//

#include "experimental/Support/FormalProperty.h"
#include "mlir/IR/BuiltinOps.h"

// [START Boilerplate code for the MLIR pass]
#include "experimental/test/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
namespace test {
#define GEN_PASS_DEF_TESTPRINTFORMALPROPERTIES
#include "experimental/test/Transforms/Passes.h.inc"
} // namespace test
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;

namespace {

struct TestPrintFormalProperties
    : public dynamatic::experimental::test::impl::TestPrintFormalPropertiesBase<
          TestPrintFormalProperties> {
  using TestPrintFormalPropertiesBase::TestPrintFormalPropertiesBase;

  void runOnOperation() override {
    FormalPropertyTable table;
    if (failed(table.addPropertiesFromJSON(jsonPath)))
      return signalPassFailure();
    for (const FormalProperty &property : table.getProperties())
      llvm::outs() << property.kind << ": " << property.expr.toSMV() << "\n";
    markAllAnalysesPreserved();
  }
};
} // namespace
