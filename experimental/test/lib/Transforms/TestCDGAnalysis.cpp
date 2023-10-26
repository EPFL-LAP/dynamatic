//===- TestCDGAnalaysis.cpp - Pass to test CDG analysis ----------- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass for the experimental CDG analysis utilities. Run with
// --exp-test-cdg-analysis.
//
//===----------------------------------------------------------------------===//

#include <set>
#include <stack>

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "experimental/Support/CDGAnalysis.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

namespace {

/// Traverses the control-dependence graph (CDG) and attaches attributes to each
/// basic block's terminator Operation. These attributes are needed for testing
/// purposes.
void cdgTraversal(DenseMap<Block *, BlockNeighbors> &cdg, MLIRContext &ctx) {

  for (auto &[block, blockNeighbors] : cdg) {
    std::string result;
    llvm::raw_string_ostream ss(result);

    block->printAsOperand(ss);

    ss << " [";
    for (Block *successor : blockNeighbors.successors) {
      successor->printAsOperand(ss);
      ss << " ";
    }
    ss << "]";

    Operation *termOp = block->getTerminator();
    OpBuilder builder(&ctx);
    termOp->setAttr("CD", builder.getStringAttr(ss.str()));
  }
}

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
      DenseMap<Block *, BlockNeighbors> cdg = cdgAnalysis(funcOp, *ctx);

      // Attach attributes to each BB terminator Operation, needed for testing.
      cdgTraversal(cdg, *ctx);
    }
  }
};
} // namespace

namespace dynamatic {
namespace experimental {
namespace test {
void registerTestCDGAnalysisPass() { PassRegistration<TestCDGAnalysisPass>(); }
} // namespace test
} // namespace experimental
} // namespace dynamatic