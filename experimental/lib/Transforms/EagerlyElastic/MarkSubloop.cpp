#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace dynamatic;
using namespace mlir;

struct EntryOpInfo {
  Operation *op;
  unsigned operandIdx;
};

// [START Boilerplate code for the MLIR pass]
#include "experimental/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_MARKSUBLOOP
#include "experimental/Transforms/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

struct MarkSubloopPass
    : public dynamatic::experimental::impl::MarkSubloopBase<
          MarkSubloopPass> {
  using MarkSubloopBase::MarkSubloopBase;

  void runOnOperation() override;

private:
  void getRealConsumers(Value val, Block *targetBlock, llvm::SmallVectorImpl<EntryOpInfo> &results);

};


// Helper function to trace past cast operations while keeping track of the operand index
void MarkSubloopPass::getRealConsumers(Value val, Block *targetBlock, llvm::SmallVectorImpl<EntryOpInfo> &results) {
  for (OpOperand &use : val.getUses()) {
    Operation *user = use.getOwner();
    if (user->getBlock() != targetBlock) continue;

    // If it's a transparent cast/ext, trace past it recursively
    if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::IndexCastOp>(user)) {
      getRealConsumers(user->getResult(0), targetBlock, results);
    } else {
      // Record the actual operation and the specific operand index being used
      results.push_back({user, use.getOperandNumber()});
    }
  }
}


void MarkSubloopPass::runOnOperation() {
  ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  auto funcOps = modOp.getBody()->getOps<mlir::func::FuncOp>();
  assert(funcOp && "No funcOp found!");

  mlir::func::FuncOp funcOp = *funcOps.begin();
  mlir::Region &region = funcOp.getBody();

  // Compute Dominance Info for the region
  DominanceInfo domInfo(funcOp);
  mlir::Block *outerHeader = nullptr, *bb2 = nullptr, *bb3 = nullptr;

  // find the first subloop (bb2) and its outer loop header
  for (mlir::Block &block : region) {
    for (mlir::Block *pred : block.getPredecessors()) {
      if (domInfo.dominates(&block, pred)) {
        if (!outerHeader) {
          outerHeader = &block;
        } else if (domInfo.properlyDominates(outerHeader, &block)) {
          bb2 = &block;
          break;
        }
      }
    } if (bb2) break;
  }

  if (!bb2) {
    llvm::errs() << "No subloops found in function!\n";
    return;
  } else {
    llvm::errs() << "Could identify a subloop header\n";
  }
  int subloopBbIdx = std::distance(region.begin(), bb2->getIterator());
  
  // find the subloop exit block
  if (auto condBr = dyn_cast<cf::CondBranchOp>(bb2->getTerminator())) {
    bb3 = (condBr.getTrueDest() != bb2) ? condBr.getTrueDest() : condBr.getFalseDest();
  }

  // Check if bb3 is dominated by the outer loop header
  if (!bb3 || !domInfo.dominates(outerHeader, bb3)) {
    llvm::errs() << "Invalid or non-returning subloop exit block!\n";
    return;
  }

  // Collect the names of all store operations inside bb2
  llvm::SmallVector<Attribute> storeOpAttrs;
  
  for (Operation &op : bb2->getOperations()) {
    if (isa<memref::StoreOp>(&op)) {
      if (auto nameAttr = op.getAttrOfType<StringAttr>("handshake.name")) {
        storeOpAttrs.push_back(nameAttr);
      } else {
        storeOpAttrs.push_back(builder.getStringAttr(op.getName().getStringRef()));
      }
    }
  }

  
  // Collect the names and operandIDs of entry operations in bb3
  llvm::SmallVector<Attribute> entryOpAttrs;

  // Helper lambda to format entry info into a DictionaryAttr
  auto addEntryOpAttr = [&](Operation *op, unsigned idx) {
    if (isa<memref::StoreOp>(op)) {
      return; }
    StringAttr nameAttr = op->getAttrOfType<StringAttr>("handshake.name");
    NamedAttribute entries[] = {
        builder.getNamedAttr("op", nameAttr),
        builder.getNamedAttr("operand_idx", builder.getI64IntegerAttr(idx))
    };
    entryOpAttrs.push_back(builder.getDictionaryAttr(entries));
  };

  // trace values defined in bb2 used directly in bb3
  for (Operation &op : bb3->getOperations()) {
    // Skip constants
    if (isa<arith::ConstantOp>(&op)) continue;
    
    for (OpOperand &use : op.getOpOperands()) {
      Value operand = use.get();
      if (operand.getType().isa<MemRefType>()) continue;
      Operation *defOp = operand.getDefiningOp();

      bool comesFromBB2 = false;
      if (defOp && defOp->getBlock() == bb2) {
        comesFromBB2 = true;
      } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (blockArg.getOwner() == bb2)
          comesFromBB2 = true;
      }
      if (comesFromBB2) {
        if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::IndexCastOp>(&op)) {
          // Trace past casts
          llvm::SmallVector<EntryOpInfo, 2> consumers;
          getRealConsumers(op.getResult(0), bb3, consumers);
          for (const auto &info : consumers) {
            addEntryOpAttr(info.op, info.operandIdx);
          }
        } else {
          // Direct non-cast user inside bb3
          addEntryOpAttr(&op, use.getOperandNumber());
        }
      }
    }
  }

  // What happens if no operations are found? - nothing!
  if (entryOpAttrs.empty()) {
    llvm::errs() << "no operations found!\n";
    // return;
  }

  // construct the Metadata Dictionary
  NamedAttribute metadata[] = {
      builder.getNamedAttr("subloop_bb", builder.getI64IntegerAttr(subloopBbIdx)),
      builder.getNamedAttr("store_ops", builder.getArrayAttr(storeOpAttrs)),
      builder.getNamedAttr("entry_ops", builder.getArrayAttr(entryOpAttrs))
  };

  DictionaryAttr regionInfoAttr = builder.getDictionaryAttr(metadata);

  // attach the attribute directly to modOp
  modOp->setAttr("handshake.subatomic_region_info", regionInfoAttr);

  // Find the outside predecessor (e.g., ^bb1), ignoring the loop back-edge (^bb2)
  mlir::Block *bb1 = nullptr;
  for (mlir::Block *pred : bb2->getPredecessors()) {
    if (pred != bb2) {
      bb1 = pred;
      break;
    }
  }

  if (!bb1) {
    llvm::errs() << "Could not find outside predecessor for subloop!\n";
    return;
  }
  bb1->dump();
  // 1. Get the branch operation in bb1 targeting bb2
  auto brOp = cast<cf::BranchOp>(bb1->getTerminator());

  // 2. Set insertion point inside bb1 right before the branch
  builder.setInsertionPoint(brOp);
  Location loc = builder.getUnknownLoc();

  // 3. Create pseudo-constant in bb1
  auto pseudoCondOp = builder.create<arith::ConstantOp>(
      loc, 
      builder.getI1Type(), 
      builder.getBoolAttr(false)
  );
  pseudoCondOp->setAttr("handshake.pseudo_cond", builder.getUnitAttr());
  pseudoCondOp->setAttr("handshake.name", builder.getStringAttr("pseudo_cond0"));

  // 4. Add a new i1 block argument to bb2
  Value tempCondArg = bb2->addArgument(builder.getI1Type(), loc);

  // 5. Pass pseudoCondOp as a branch operand from bb1 into bb2
  brOp.getDestOperandsMutable().append(pseudoCondOp.getResult());

  // 6. ALSO update the loop back-edge (inside bb2) so bb2 passes a dummy/recirculated i1 value
  for (Operation &op : bb2->getOperations()) {
    if (auto condBr = dyn_cast<cf::CondBranchOp>(&op)) {
      if (condBr.getTrueDest() == bb2) {
        condBr.getTrueDestOperandsMutable().append(tempCondArg);
        llvm::errs() << "went into the if\n";
      } else if (condBr.getFalseDest() == bb2) {
        condBr.getFalseDestOperandsMutable().append(tempCondArg);
        llvm::errs() << "went into the else if\n";
      }
    }
  }
}