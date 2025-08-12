#include "CfGateBinarization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <iterator>
#include <queue>

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;

namespace dynamatic {
namespace experimental {

// Implement the base class and auto-generated create functions.
// Must be called from the .cpp file to avoid multiple definitions
#define GEN_PASS_DEF_CFGATEBINARIZATION
#include "experimental/Transforms/Passes.h.inc"

} // namespace experimental
} // namespace dynamatic

struct CfGateBinarizationPass
    : public dynamatic::experimental::impl::CfGateBinarizationBase<
          CfGateBinarizationPass> {
  using CfGateBinarizationBase<CfGateBinarizationPass>::CfGateBinarizationBase;
  void runDynamaticPass() override;
};

void CfGateBinarizationPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();
  OpBuilder builder(modOp.getContext());
  // builder.createBlock()

  for (auto funcOp : llvm::make_early_inc_range(modOp.getOps<func::FuncOp>())) {
    // Perform BFS
    std::queue<std::pair<Block *, Block *>> bfsQueue;
    bfsQueue.push({nullptr, &funcOp.getBlocks().front()});
    DenseSet<Block *> visited;

    while (!bfsQueue.empty()) {
      auto [fromBlock, toBlock] = bfsQueue.front();
      bfsQueue.pop();

      if (visited.contains(toBlock)) {
        // backedge
        auto predecessors = toBlock->getPredecessors();
        auto distance = std::distance(predecessors.begin(), predecessors.end());
        if (distance > 2) {
          SmallVector<Location> locs;
          for (auto arg : toBlock->getArguments()) {
            locs.push_back(arg.getLoc());
          }
          Block *newBlock =
              builder.createBlock(toBlock, toBlock->getArgumentTypes(), locs);
          for (auto *pred : llvm::make_early_inc_range(predecessors)) {
            if (pred == fromBlock)
              continue;

            for (auto &oprd : pred->getTerminator()->getBlockOperands()) {
              if (oprd.get() == toBlock) {
                oprd.set(newBlock);
              }
            }
          }
          builder.setInsertionPointToStart(newBlock);
          builder.create<cf::BranchOp>(builder.getUnknownLoc(),
                                       newBlock->getArguments(), toBlock);
          visited.insert(newBlock);
        }
      } else {
        visited.insert(toBlock);
        // queue successors using block->getSuccessors()
        for (auto *succ : toBlock->getSuccessors()) {
          bfsQueue.emplace(toBlock, succ);
        }
      }
    }

    // for (auto &block : funcOp.getBlocks()) {
    //   // for (auto *pred : block.getPredecessors()) {
    //   //   pred->in
    //   // }
    //   block.ge
    //   block.getTerminator()->dump();
    // }
  }
}
