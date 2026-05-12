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
#define GEN_PASS_DEF_NDUPLICATION
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]
namespace {

struct NDuplicationPass
    : public dynamatic::experimental::impl::NDuplicationBase<NDuplicationPass> {

  using NDuplicationBase::NDuplicationBase;

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // find operation which contains the selected value
    // Names need to be checked!! might be different
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    Operation *rawOp = namer.getOp("select0"); // might be different
    if (!rawOp) {
      llvm::errs() << "No operation named \"select0\" exists\n";
      return signalPassFailure();
    }

    auto selectOp = dyn_cast<mlir::arith::SelectOp>(rawOp);
    mlir::Block *currentBlock = selectOp->getBlock();
    mlir::func::FuncOp funcOp =
        dyn_cast<mlir::func::FuncOp>(currentBlock->getParentOp());
    Location loc = selectOp.getLoc();
    builder.setInsertionPointAfter(selectOp);
    Value selectRes = selectOp.getResult();

    // move loop logic into the last block
    builder.setInsertionPointAfter(selectOp);
    mlir::Block *exitBlock =
        currentBlock->splitBlock(builder.getInsertionPoint());
    Operation *c1 = namer.getOp("constant6");
    c1->moveBefore(exitBlock, exitBlock->begin());
    Value c100 = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(100));
    Operation *cmpi1 = namer.getOp("cmpi1");
    cmpi1->setOperand(1, c100);
    // mlir::Block *bb4 = bb1->splitBlock(
    // Block::iterator(branchCond.getDefiningOp())->getNextNode());

    Operation *addfOp = namer.getOp("addf1");
    Operation *mulfOp = namer.getOp("mulf1");
    Operation *storeOp = namer.getOp("store1");
    Operation *extui1 = namer.getOp("extui1");
    Operation *idxCast1 = namer.getOp("index_cast1");

    if (!addfOp || !mulfOp || !storeOp) {
      llvm::errs() << "Required operations for branching not found\n";
      return signalPassFailure();
    }

    std::vector<float> listOfComp = {5.0f, 6.0f, 7.0f};
    int counter = 2;
    for (float compVal : listOfComp) {
      builder.setInsertionPointToEnd(currentBlock);
      Value constant = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getFloatAttr(builder.getF32Type(), compVal));
      Value branchCond = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::OEQ, selectRes, constant);
      mlir::Block *trueBlock = funcOp.addBlock();     // true path
      mlir::Block *nextElseBlock = funcOp.addBlock(); // false path
      builder.create<mlir::cf::CondBranchOp>(loc, branchCond, trueBlock,
                                             nextElseBlock);

      // start true path
      builder.setInsertionPointToStart(trueBlock);
      mlir::Value arg2 = funcOp.getArgument(2); // c
      Value newCst5 = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getFloatAttr(builder.getF32Type(), 5.0));
      Value newCst10 = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getFloatAttr(builder.getF32Type(), 10.0));
      Value specAdd = builder.create<mlir::arith::AddFOp>(loc, arg2, newCst5);
      Value specMul =
          builder.create<mlir::arith::MulFOp>(loc, specAdd, newCst10);

      // clone the address calculation for the store
      Operation *newExt = builder.clone(*extui1);
      std::string extName = "extui" + std::to_string(counter);
      newExt->setAttr("handshake.name", builder.getStringAttr(extName));
      Operation *newIdx = builder.clone(*idxCast1);
      std::string idxName = "index_cast" + std::to_string(counter);
      newIdx->setAttr("handshake.name", builder.getStringAttr(idxName));
      newIdx->setOperand(0, newExt->getResult(0));
      counter++;

      auto store2 = builder.create<mlir::memref::StoreOp>(
          loc, specMul, storeOp->getOperand(1), newIdx->getResult(0));

      // all true paths jump to the final exit block
      builder.create<mlir::cf::BranchOp>(loc, exitBlock);

      // update
      currentBlock = nextElseBlock;
    }

    // start of last branch (y = x)
    builder.setInsertionPointToStart(currentBlock);
    Value newCst10False = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getFloatAttr(builder.getF32Type(), 10.0));
    addfOp->moveBefore(currentBlock, currentBlock->end());
    mulfOp->moveBefore(currentBlock, currentBlock->end());
    mulfOp->setOperand(1, newCst10False);
    extui1->moveBefore(currentBlock, currentBlock->end());
    idxCast1->moveBefore(currentBlock, currentBlock->end());
    storeOp->moveBefore(currentBlock, currentBlock->end());

    builder.create<mlir::cf::BranchOp>(loc, exitBlock);
  }
};

} // namespace