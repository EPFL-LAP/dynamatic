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

    // find select0 operation
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    Operation *rawOp = namer.getOp("select0");
    if (!rawOp) {
      llvm::errs() << "No operation named \"addf0\" exists\n";
      return signalPassFailure();
    }

    auto selectOp = dyn_cast<mlir::arith::SelectOp>(rawOp);
    mlir::Block *bb1 = selectOp->getBlock();
    mlir::func::FuncOp funcOp =
        dyn_cast<mlir::func::FuncOp>(bb1->getParentOp());
    Location loc = selectOp.getLoc();

    // create branch condition
    builder.setInsertionPointAfter(selectOp);
    Value selectRes = selectOp.getResult();
    Value cst5 = selectOp.getTrueValue();
    Value branchCond = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OEQ, selectRes, cst5);
    mlir::Block *bb2 = funcOp.addBlock(); // true path
    mlir::Block *bb3 = funcOp.addBlock(); // false path

    // restructure the blocks
    mlir::Block *bb4 = bb1->splitBlock(
        Block::iterator(branchCond.getDefiningOp())->getNextNode());
    Operation *c1 = namer.getOp("constant6");
    c1->moveBefore(bb4, bb4->begin());
    Value c100 = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(100));
    Operation *cmpi1 = namer.getOp("cmpi1");
    cmpi1->setOperand(1, c100);
    builder.setInsertionPointToEnd(bb1);
    builder.create<mlir::cf::CondBranchOp>(loc, branchCond, bb2, bb3);

    Operation *addfOp = namer.getOp("addf1");
    Operation *mulfOp = namer.getOp("mulf1");
    Operation *storeOp = namer.getOp("store1");

    if (!addfOp || !mulfOp || !storeOp) {
      llvm::errs() << "Required operations for branching not found\n";
      return signalPassFailure();
    }

    // start of true branch (y = 5.0f)
    builder.setInsertionPointToStart(bb2);
    mlir::Value arg2 = funcOp.getArgument(2);
    Value newCst5 = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getFloatAttr(builder.getF32Type(), 5.0));
    Value newCst10 = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getFloatAttr(builder.getF32Type(), 10.0));
    Value specAdd = builder.create<mlir::arith::AddFOp>(loc, arg2, newCst5);
    Value specMul = builder.create<mlir::arith::MulFOp>(loc, specAdd, newCst10);

    // clone the address calculation for the store
    Operation *extui1 = namer.getOp("extui1");
    Operation *idxCast1 = namer.getOp("index_cast1");
    Operation *newExt = builder.clone(*extui1);
    newExt->setAttr("handshake.name", builder.getStringAttr("extui2"));
    Operation *newIdx = builder.clone(*idxCast1);
    newIdx->setAttr("handshake.name", builder.getStringAttr("index_cast2"));
    newIdx->setOperand(0, newExt->getResult(0));

    auto store2 = builder.create<mlir::memref::StoreOp>(
        loc, specMul, storeOp->getOperand(1), newIdx->getResult(0));
    store2->setAttr("handshake.name", builder.getStringAttr("store2"));

    builder.create<mlir::cf::BranchOp>(loc, bb4);

    // start of false branch (y = x)
    builder.setInsertionPointToStart(bb3);
    Value newCst10False = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getFloatAttr(builder.getF32Type(), 10.0));
    addfOp->moveBefore(bb3, bb3->end());
    mulfOp->moveBefore(bb3, bb3->end());
    mulfOp->setOperand(1, newCst10False);
    extui1->moveBefore(bb3, bb3->end());
    idxCast1->moveBefore(bb3, bb3->end());
    storeOp->moveBefore(bb3, bb3->end());
    storeOp->setAttr("handshake.name", builder.getStringAttr("store1"));

    builder.create<mlir::cf::BranchOp>(loc, bb4);
  }
};

} // namespace