//===- TestCDGAnalaysis.cpp - Pass to test CDG analysis ----------- C++ -*-===//
//
// Test pass for the experimental CDG analysis utilities. Run with
// --exp-test-cdg-analysis.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "experimental/Support/CDGAnalysis.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

namespace {
struct TestCDGAnalysisPass
    : public PassWrapper<TestCDGAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCDGAnalysisPass)

  StringRef getArgument() const final { return "exp-test-cdg-analysis"; }
  StringRef getDescription() const final {
    return "Test CDG analysis utilities";
  }

  void runOnOperation() override {
    // Get the MLIR context for the current operation
    MLIRContext *ctx = &getContext();
    // Get the operation (the top level module)
    ModuleOp mod = getOperation();

    // Iterate over all functions in the module
    for (func::FuncOp funcOp : mod.getOps<func::FuncOp>()) {
      CDGNode<Block>* entryCDGNode = CDGAnalysis(funcOp, ctx);
      if (!entryCDGNode) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

namespace experimental {
namespace test {
void registerTestCDGAnalysisPass() { PassRegistration<TestCDGAnalysisPass>(); }
} // namespace test
} // namespace experimental