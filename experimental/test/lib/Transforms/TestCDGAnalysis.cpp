//===- TestCDGAnalaysis.cpp - Pass to test CDG analysis ----------- C++ -*-===//
//
// Test pass for the experimental CDG analysis utilities. Run with
// --exp-test-cdg-analysis.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace dynamatic;

namespace {
struct TestCDGAnalysisPass
    : public PassWrapper<TestCDGAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCDGAnalysisPass)

  StringRef getArgument() const final { return "exp-test-cdg-analysis"; }
  StringRef getDescription() const final {
    return "Test CDG analysis utilities";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    llvm::outs() << "Your test pass starts here!\n";
  }
};
} // namespace

namespace experimental {
namespace test {
void registerTestCDGAnalysisPass() { PassRegistration<TestCDGAnalysisPass>(); }
} // namespace test
} // namespace experimental