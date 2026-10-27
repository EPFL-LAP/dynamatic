#include "dynamatic/Analysis/NameAnalysis.h" // needed
#include "dynamatic/Dialect/CFExtra/CFExtraOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // needed
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <fstream>

// NOTE: The code wrapped in LLVM_DEBUG(...) is executed when
// - Dynamatic is built in debug mode
// - dynamatic-opt is called with `--debug` or `--debug-only=<DEBUG_TYPE>`.
#define DEBUG_TYPE "loop-prediction"

using namespace llvm;
using namespace dynamatic;
using namespace mlir;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_LOOPPREDICTION
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]
namespace {

struct LoopPredictionPass
    : public dynamatic::experimental::impl::LoopPredictionBase<
          LoopPredictionPass> {

  using LoopPredictionBase::LoopPredictionBase;

  void runOnOperation() override;

private:
};

} // namespace

// Collect all blocks in the natural loop (supports single or multi-block loops)
void collectLoopBlocks(Block *header, Block *latch,
                       SmallPtrSetImpl<Block *> &loopBlocks) {
  loopBlocks.insert(header);
  loopBlocks.insert(latch);
  SmallVector<Block *, 8> worklist;
  worklist.push_back(latch);

  while (!worklist.empty()) {
    Block *curr = worklist.pop_back_val();
    if (curr == header)
      continue;
    for (Block *pred : curr->getPredecessors()) {
      if (loopBlocks.insert(pred).second) {
        worklist.push_back(pred);
      }
    }
  }
}

void LoopPredictionPass::runOnOperation() {
  mlir::ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  NameAnalysis &namer = getAnalysis<NameAnalysis>();

  auto funcOps = modOp.getBody()->getOps<mlir::func::FuncOp>();
  assert(!funcOps.empty() && "No funcOp found!");
  mlir::func::FuncOp funcOp = *funcOps.begin();
  DominanceInfo domInfo(funcOp);

  // find the loop
  SmallVector<cf::CondBranchOp, 4> loopLatches;
  for (Block &block : funcOp) {
    auto condBr = dyn_cast<cf::CondBranchOp>(block.getTerminator());
    if (!condBr)
      continue;

    Block *header = condBr.getTrueDest();
    if (domInfo.dominates(header, &block)) {
      loopLatches.push_back(condBr);
    }
  }
  cf::CondBranchOp latchBr = loopLatches[0];

  Location loc = latchBr.getLoc();

  Block *latchBlock = latchBr->getBlock();
  Block *innerHeader = latchBr.getTrueDest();
  Block *exitBlock = latchBr.getFalseDest();
  llvm::errs() << "Applying loop prediction.\n";

  // Ensure the loop header has the 2 arguments: (%i : i32, %sum : f32)
  // Index 1 is our target block argument (`sum`).
  unsigned targetArgIdx = 1;
  if (innerHeader->getNumArguments() <= targetArgIdx)
    return;

  SmallPtrSet<Block *, 8> loopBlocks;
  collectLoopBlocks(innerHeader, latchBlock, loopBlocks);

  // Create `^outer_header` right before `^innerHeader` with identical
  // argument types
  Block *outerHeader = builder.createBlock(innerHeader);
  for (BlockArgument arg : innerHeader->getArguments()) {
    outerHeader->addArgument(arg.getType(), arg.getLoc());
  }

  // Redirect outside predecessors of `innerHeader` to `outerHeader`
  for (Block *pred :
       llvm::make_early_inc_range(innerHeader->getPredecessors())) {
    if (pred == outerHeader || loopBlocks.contains(pred))
      continue;
    Operation *term = pred->getTerminator();
    for (unsigned i = 0, e = term->getNumSuccessors(); i < e; ++i) {
      if (term->getSuccessor(i) == innerHeader) {
        term->setSuccessor(outerHeader, i);
      }
    }
  }

  // Populate `outerHeader`:
  builder.setInsertionPointToEnd(outerHeader);
  builder.create<cf::BranchOp>(loc, innerHeader, outerHeader->getArguments());

  // 4. `innerHeader->getArgument(targetArgIdx)` IS `pred_sum` (%4).
  //    All existing uses in the loop body already point to it!
  Value predSum = innerHeader->getArgument(targetArgIdx);
  Type sumVarType = predSum.getType();

  // 5. Grab the real slow update `%9` (`updatedSum`) and the original loop
  //    bound condition `%11` (`origBoundCond`) from `latchBr` BEFORE modifying
  //    it.
  SmallVector<Value, 4> outerRecoveryOperands(
      latchBr.getTrueDestOperands().begin(),
      latchBr.getTrueDestOperands().end());
  Value updatedSum = outerRecoveryOperands[targetArgIdx]; // %9 (addf0)
  Value origBoundCond = latchBr.getCondition();

  // In `latchBlock`, right before `latchBr`:
  //    - Easy update: `pred_next` stays `predSum` (no addition!)
  //    - Compare `predNext == updatedSum` (%14)
  //    - Combine with original loop bound check: `innerCond = predOk &&
  //    origBoundCond`
  builder.setInsertionPoint(latchBr);

  Value predNext = predSum; // Easy update: keep it the same!

  Value predOk;
  if (isa<FloatType>(sumVarType)) {
    predOk = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                           predNext, updatedSum);
  } else {
    predOk = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           predNext, updatedSum);
  }

  Value innerCond = builder.create<arith::AndIOp>(loc, predOk, origBoundCond);

  // Create `^latch_outer_check` right before `exitBlock` to route either
  // back to `outerHeader` (on misprediction when i < N) or to `exitBlock` (when
  // i >= N).
  Block *latchOuterCheck = builder.createBlock(exitBlock);
  builder.setInsertionPointToEnd(latchOuterCheck);
  builder.create<cf::CondBranchOp>(
      loc, origBoundCond,
      /*trueDest=*/outerHeader, latchBr.getTrueDestOperands(),
      /*falseDest=*/exitBlock, latchBr.getFalseDestOperands());

  // Replace `latchBr` in `latchBlock` with the inner loop conditional branch
  builder.setInsertionPoint(latchBr);
  SmallVector<Value, 4> innerBackedgeOperands = outerRecoveryOperands;
  innerBackedgeOperands[targetArgIdx] = predNext;

  builder.create<cf::CondBranchOp>(loc, innerCond,
                                   /*trueDest=*/innerHeader,
                                   innerBackedgeOperands,
                                   /*falseDest=*/latchOuterCheck, ValueRange{});

  latchBr.erase();
}
