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

    // duplicate operations in bb1 for the prediction
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
    // TODO: put multiplication with 25 also in here!

    // create the 'Exit' block for the loop logic (increment)
    mlir::Block *bbExit = builder.createBlock(&funcOp.getBody());
    mlir::Value exitArg = bbExit->addArgument(builder.getI32Type(), loc);
    builder.setInsertionPointToStart(bbExit);

    // define new constants in bbExit for the increment and comparison
    // TODO: check whether these constants can be also just moved
    auto exitConst1 = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(1));
    auto exitConst100 = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(100));

    // identify the existing operations for increment in bb3
    Operation *loopCondBr = bb3->getTerminator();
    Operation *loopCmpi = loopCondBr->getPrevNode();
    Operation *loopAddi = loopCmpi->getPrevNode();

    // Move the logic operations from bb3 to bbExit and rewire them
    loopAddi->moveAfter(exitConst100);
    loopCmpi->moveAfter(loopAddi);
    loopCondBr->moveAfter(loopCmpi);
    loopAddi->setOperand(0, exitArg); // Use index passed to bbExit
    loopAddi->setOperand(1, exitConst1.getResult());   // Use new constant 1
    loopCmpi->setOperand(1, exitConst100.getResult()); // Use new constant 100

    // create shortcut block
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

    // branch to exit, providing the current loop index
    builder.create<mlir::cf::BranchOp>(loc, bbExit, currentI);

    // update everything
    auto condBr1 = mlir::cast<mlir::cf::CondBranchOp>(bb1->getTerminator());
    condBr1.setSuccessor(bbShortcut, 0);
    condBr1.getTrueDestOperandsMutable().clear();

    // bb3 now also jumps to the increment / exit block
    builder.setInsertionPointToEnd(bb3);
    builder.create<mlir::cf::BranchOp>(loc, bbExit, currentI);
  }
};

} // namespace