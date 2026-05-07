#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

using namespace llvm;
using namespace dynamatic;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_SHORTCUTPATH
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]
namespace {

struct ShortcutPathPass
    : public dynamatic::experimental::impl::ShortcutPathBase<ShortcutPathPass> {

  using ShortcutPathBase::ShortcutPathBase;
  /*
    void runDynamaticPass() override {
      mlir::ModuleOp modOp = getOperation();
      MLIRContext *ctx = &getContext();
      OpBuilder builder(ctx);

      auto funcOp = *modOp.getOps<mlir::func::FuncOp>().begin();
      if (!funcOp)
        return;

      // Get all the blocks
      auto &blocks = funcOp.getBlocks();
      auto it = blocks.begin();
      mlir::Block *bb0 = &*it;
      mlir::Block *bb1 = &*(++it);
      mlir::Block *bb2 = &*(++it);
      mlir::Block *bb3 = &*(++it);

      // Move load2 (a[i-1]) to block1 to calculate it with 25
      NameAnalysis &namer = getAnalysis<NameAnalysis>();
      Operation *load2Op = namer.getOp("load2");

      if (load2Op) {
        mlir::Value indexVal = load2Op->getOperand(1); // The index operand
        mlir::Operation *castOp = indexVal.getDefiningOp();
        if (castOp) {
          mlir::Value extuiVal = castOp->getOperand(0);
          mlir::Operation *extuiOp = extuiVal.getDefiningOp();

          if (extuiOp) {
            mlir::Value addiVal = extuiOp->getOperand(0);
            mlir::Operation *addiOp = addiVal.getDefiningOp();

            // 3. Move the entire chain into bb1 in the correct order
            // They must be moved before the load so they dominate it
            if (addiOp)
              addiOp->moveBefore(bb1, bb1->end());
            extuiOp->moveBefore(bb1, bb1->end());
            castOp->moveBefore(bb1, bb1->end());
          }
        }
        load2Op->moveBefore(bb1, bb1->begin());
      } else
        return signalPassFailure();

      builder.setInsertionPointAfter(load2Op);
      Location loc = load2Op->getLoc();
      auto cnst25 = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getI32IntegerAttr(25));
      auto shortcutVal =
          builder.create<mlir::arith::MulIOp>(loc, load2Op->getResult(0),
    cnst25);

      // create new block
      mlir::Block *bbShortcut = builder.createBlock(&funcOp.getBody());
      builder.setInsertionPointToStart(bbShortcut);

      // Cast i to index type for the store
      auto iCast = builder.create<mlir::arith::IndexCastOp>(
          loc, builder.getIndexType(), bb1->getArgument(0));
      builder.create<mlir::memref::StoreOp>(
          loc, shortcutVal, funcOp.getArgument(1), iCast.getResult());

      // Branch back to the loop increment (the original end of ^bb3)
      builder.create<mlir::cf::BranchOp>(loc, bb3, bb1->getArgument(0));

      // update switch condition in bb1 to point to our shortcut
      auto condBr = mlir::cast<mlir::cf::CondBranchOp>(bb1->getTerminator());
      condBr.setSuccessor(bbShortcut, 0);
      condBr.getTrueDestOperandsMutable().clear();
    }
  */
  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    auto funcOp = *modOp.getOps<mlir::func::FuncOp>().begin();
    if (!funcOp)
      return;

    auto &blocks = funcOp.getBlocks();
    auto it = blocks.begin();
    mlir::Block *bb1 = &*(++it);
    mlir::Block *bb2 = &*(++it);
    mlir::Block *bb3 = &*(++it);

    Location loc = funcOp.getLoc();
    mlir::Value currentI = bb1->getArgument(0);

    // 1. Create fresh ops in bb1 for the prediction logic
    builder.setInsertionPoint(bb1->getTerminator());
    auto cMinus1 = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(-1));
    auto newIdxI32 =
        builder.create<mlir::arith::AddIOp>(loc, currentI, cMinus1);
    auto newExtui = builder.create<mlir::arith::ExtUIOp>(
        loc, builder.getI64Type(), newIdxI32);
    auto newCast = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getIndexType(), newExtui);
    auto newLoad = builder.create<mlir::memref::LoadOp>(
        loc, funcOp.getArgument(0), newCast.getResult());
    mlir::Value sharedLoadVal = newLoad.getResult();

    // 2. Create the 'Exit' block for loop bookkeeping
    mlir::Block *bbExit = builder.createBlock(&funcOp.getBody());
    mlir::Value exitArg = bbExit->addArgument(builder.getI32Type(), loc);
    builder.setInsertionPointToStart(bbExit);

    // 3. Define FRESH constants in bbExit for the increment and comparison
    // This satisfies dominance because these are defined inside the block they
    // are used in
    auto exitConst1 = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(1));
    auto exitConst100 = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(100));

    // Identify the existing operations we want to move from bb3
    Operation *loopCondBr = bb3->getTerminator();
    Operation *loopCmpi = loopCondBr->getPrevNode();
    Operation *loopAddi = loopCmpi->getPrevNode();

    // Move the logic operations to bbExit
    loopAddi->moveAfter(exitConst100);
    loopCmpi->moveAfter(loopAddi);
    loopCondBr->moveAfter(loopCmpi);

    // Re-wire the moved operations to use our NEW constants and the block
    // argument
    loopAddi->setOperand(0, exitArg); // Use index passed to bbExit
    loopAddi->setOperand(1, exitConst1.getResult());   // Use new constant 1
    loopCmpi->setOperand(1, exitConst100.getResult()); // Use new constant 100

    // 4. Create the Shortcut Block
    mlir::Block *bbShortcut = builder.createBlock(bbExit);
    builder.setInsertionPointToStart(bbShortcut);

    auto cnst25 = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(25));
    auto shortcutVal =
        builder.create<mlir::arith::MulIOp>(loc, sharedLoadVal, cnst25);
    auto iStoreCast = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getIndexType(), currentI);
    builder.create<mlir::memref::StoreOp>(
        loc, shortcutVal, funcOp.getArgument(1), iStoreCast.getResult());

    // Branch to exit, providing the current loop index
    builder.create<mlir::cf::BranchOp>(loc, bbExit, currentI);

    // 5. Update Successors and Redirections
    auto condBr1 = mlir::cast<mlir::cf::CondBranchOp>(bb1->getTerminator());
    condBr1.setSuccessor(bbShortcut, 0);
    condBr1.getTrueDestOperandsMutable().clear();

    // Slow Path (bb3) now also jumps to the exit logic
    builder.setInsertionPointToEnd(bb3);
    builder.create<mlir::cf::BranchOp>(loc, bbExit, currentI);
  }
};

} // namespace