// Include some other useful headers.
#include "dynamatic/Analysis/NameAnalysis.h" // needed
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // needed
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

using namespace llvm;
using namespace dynamatic;

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_PIPELINEDUPLICATION
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]
namespace {

struct PipelineDuplicationPass
    : public dynamatic::experimental::impl::PipelineDuplicationBase<
          PipelineDuplicationPass> {

  using PipelineDuplicationBase::PipelineDuplicationBase;

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // Find select0 operation
    // TODO: take from a .json file
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    Operation *rawOp = namer.getOp("select0");
    if (!rawOp) {
      llvm::errs() << "No operation named \"select0\" exists\n";
      return signalPassFailure();
    }
    // TODO: make this mlir::arith::xxx changeable
    auto op = dyn_cast<mlir::arith::SelectOp>(rawOp);
    mlir::Block *currentBlock = op->getBlock();
    mlir::func::FuncOp funcOp =
        dyn_cast<mlir::func::FuncOp>(currentBlock->getParentOp());
    Location loc = op.getLoc();

    // create branch condition
    builder.setInsertionPointAfter(op);
    // TODO: get this info from the json file
    Value selectRes = op.getResult();
    Value cst5 = op.getTrueValue();
    Value branchCond = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OEQ, selectRes, cst5);

    // restructure the blocks
    // splitblock maybe works differently for other operations
    mlir::Block *exitBlock = currentBlock->splitBlock(
        Block::iterator(branchCond.getDefiningOp())->getNextNode());
    mlir::Block *trueBlock = funcOp.addBlock();  // true path
    mlir::Block *falseBlock = funcOp.addBlock(); // false path

    builder.setInsertionPointToEnd(currentBlock);
    builder.create<mlir::cf::CondBranchOp>(loc, branchCond, trueBlock,
                                           falseBlock);

    // somehow get all of the operations that I need to clone / duplicate
    // this could also be done earlier or later than the splitting i think

    // i will probably always have to add a result of some kind of calculation
    // to my next block, TODO: make this not hardcoded
    // could this all also be added at the end? or right before the end of the
    // true path?
    Operation *mulfOp = namer.getOp("mulf1");
    Type resultType = mulfOp->getResult(0).getType();
    Value mulfResultExit = exitBlock->addArgument(resultType, loc);
    mulfOp->getResult(0).replaceAllUsesWith(mulfResultExit);

    // TRUE PATH
    // clone the necessary operations to here as well as the (TODO:) constant
    mlir::IRMapping mapper;
    // we only care about values derived from our starting operation which in
    // this hardcoded case is the selectop
    llvm::DenseSet<Value> trackedValues;
    for (Value res : op->getResults()) {
      trackedValues.insert(res);
    }

    builder.setInsertionPointToStart(trueBlock);

    for (Operation &blockOp : exitBlock->getOperations()) {
      // stop if we hit a store (or what else?)
      if (isa<mlir::memref::StoreOp, mlir::BranchOpInterface>(blockOp)) {
        break;
      }

      // check if the op uses a value we are tracking
      bool isDependent =
          llvm::any_of(blockOp.getOperands(), [&](Value operand) {
            return trackedValues.count(operand);
          });

      if (isDependent) {
        Operation *cloned = builder.clone(blockOp, mapper);
        llvm::StringRef originalName = namer.getName(&blockOp);
        std::string newName = originalName.str() + "_dup";
        cloned->setAttr("handshake.name", builder.getStringAttr(newName));
        // track the new results so we can find the next operations in the chain
        for (auto it : llvm::enumerate(cloned->getResults())) {
          size_t index = it.index();
          Value clonedRes = it.value();
          Value originalRes = blockOp.getResult(index);

          // track the original result so we can find its users later in the
          // block
          trackedValues.insert(originalRes);
          mapper.map(originalRes, clonedRes);
        }
      }
    }

    llvm::errs() << "--- Mapper Contents ---\n";

    // Print Value mappings (Original Value -> Cloned Value)
    for (auto &pair : mapper.getValueMap()) {
      mlir::Value original = pair.first;
      mlir::Value cloned = pair.second;

      llvm::errs() << "Value Mapping:\n";
      llvm::errs() << "  From: " << original << "\n";
      llvm::errs() << "  To:   " << cloned << "\n";
    }

    // print mappings
    for (auto &pair : mapper.getBlockMap()) {
      mlir::Block *original = pair.first;
      mlir::Block *cloned = pair.second;

      llvm::errs() << "Block Mapping:\n";
      original->printAsOperand(llvm::errs());
      llvm::errs() << " -> ";
      cloned->printAsOperand(llvm::errs());
      llvm::errs() << "\n";
    }

    llvm::errs() << "-----------------------\n";

    Value clonedMulfRes = mapper.lookup(mulfOp->getResult(0));
    builder.setInsertionPointToEnd(trueBlock);
    builder.create<mlir::cf::BranchOp>(loc, exitBlock, clonedMulfRes);

    // FALSE PATH
    // move all of the stuff from above here
    // TODO: make this not hardcoded
    Operation *addfOp = namer.getOp("addf1");
    builder.setInsertionPointToStart(falseBlock);
    Value newCst10False = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getFloatAttr(builder.getF32Type(), 10.0));
    addfOp->moveBefore(falseBlock, falseBlock->end());
    mulfOp->moveBefore(falseBlock, falseBlock->end());
    mulfOp->setOperand(1, newCst10False);
    builder.create<mlir::cf::BranchOp>(loc, exitBlock, mulfOp->getResult(0));
  }
};

} // namespace
