//===- TestCDGAnalaysis.cpp - Pass to test CDG analysis ----------- C++ -*-===//
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

/// @brief CDG traversal function.
///
/// Traverses the control-dependence graph (CDG) and attaches attributes to each
/// basic block's terminator Operation. These attributes are needed for testing
/// purposes.
void cdgTraversal(DenseMap<Block *, BlockNeighbors *> &cdg, MLIRContext &ctx) {

  std::set<Block *> visitedSet;
  std::stack<Block *> blockStack;

  for (auto &[block, blockNeighbours] : cdg)
    // Push the blocks with no predecessor to the stack.
    if (blockNeighbours->predecessors.empty())
      blockStack.push(block);

  while (!blockStack.empty()) {
    Block *currBlock = blockStack.top();
    blockStack.pop();

    // visit node

    visitedSet.insert(currBlock);

    std::string result;
    llvm::raw_string_ostream ss(result);

    currBlock->printAsOperand(ss);

    ss << " [";
    for (Block *successor : cdg[currBlock]->successors) {
      successor->printAsOperand(ss);
      ss << " ";
    }
    ss << "]";

    Operation *termOp = currBlock->getTerminator();
    OpBuilder builder(&ctx);
    termOp->setAttr("CD", builder.getStringAttr(ss.str()));

    // end visit

    for (Block *successor : cdg[currBlock]->successors) {
      // Check if successor is already visited.
      if (visitedSet.find(successor) != visitedSet.end())
        continue;
      // Push unvisited successors to the stack.
      blockStack.push(successor);
    }
  } // end while
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
      DenseMap<Block *, BlockNeighbors *> *cdg = cdgAnalysis(funcOp, *ctx);

      // Attach attributes to each BB terminator Operation, needed for testing.
      cdgTraversal(*cdg, *ctx);
    }
  }
};
} // namespace

namespace experimental {
namespace test {
void registerTestCDGAnalysisPass() { PassRegistration<TestCDGAnalysisPass>(); }
} // namespace test
} // namespace experimental
