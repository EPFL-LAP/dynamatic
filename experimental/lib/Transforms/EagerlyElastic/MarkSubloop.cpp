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

struct LoopInfo {
  mlir::Block *outerHeader;
  mlir::Block *subloop;
  mlir::Block *exitBlock;
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
  llvm::SmallVector<LoopInfo> validLoops;

  // find all loops with exactly one subloop
  for (mlir::Block &block : region) {
    mlir::Block *outerLatch = nullptr;
    for (mlir::Block *pred : block.getPredecessors()) {
      if (domInfo.dominates(&block, pred)) {
        outerLatch = pred;
        break;
      }
    }
    if (!outerLatch) continue;

    llvm::SmallVector<mlir::Block *> innerLoops;
    for (mlir::Block &potentialSub : region) {
      if (&potentialSub != &block && domInfo.properlyDominates(&block, &potentialSub)) {
        for (mlir::Block *pred : potentialSub.getPredecessors()) {
          if (domInfo.dominates(&potentialSub, pred)) {
            innerLoops.push_back(&potentialSub);
            break;
          }
        }
      }
    }

    if (innerLoops.size() == 1) {
      mlir::Block *subloop = innerLoops.front();
      mlir::Block *exitSubloop = nullptr;
      if (auto condBr = dyn_cast<cf::CondBranchOp>(subloop->getTerminator())) {
        exitSubloop = (condBr.getTrueDest() != subloop) ? condBr.getTrueDest() : condBr.getFalseDest();
      }
      
      if (exitSubloop && exitSubloop == outerLatch) {
        validLoops.push_back({&block, subloop, outerLatch});
      }
    }
  }

  if (validLoops.empty()) {
    llvm::errs() << "No loops with exactly one subloop found!\n";
    return;
  }

  llvm::SmallVector<Attribute> allLoopsMetadata;

  // Process all discovered loops
  for (auto &loop : validLoops) {
    mlir::Block *outerHeader = loop.outerHeader;
    mlir::Block *subloop = loop.subloop;
    mlir::Block *exitBlock = loop.exitBlock;

    int subloopBbIdx = std::distance(region.begin(), subloop->getIterator());

    // Collect the names of all store operations inside the subblock
    llvm::SmallVector<Attribute> storeOpAttrs;  
    for (Operation &op : subloop->getOperations()) {
      if (isa<memref::StoreOp>(&op)) {
        storeOpAttrs.push_back(op.getAttrOfType<StringAttr>("handshake.name"));
      }
    }  
    
    // Collect the names and operandIDs of entry operations in exitBlock
    llvm::SmallVector<Attribute> entryOpAttrs;

    // Helper lambda to format entry info into a DictionaryAttr
    auto addEntryOpAttr = [&](Operation *op, unsigned idx) {
      if (isa<memref::StoreOp>(op)) { // TODO: fix?
        return; }
      StringAttr nameAttr = op->getAttrOfType<StringAttr>("handshake.name");
      NamedAttribute entries[] = {
          builder.getNamedAttr("op", nameAttr),
          builder.getNamedAttr("operand_idx", builder.getI64IntegerAttr(idx))
      };
      entryOpAttrs.push_back(builder.getDictionaryAttr(entries));
    };

    // trace values defined in the subloop used directly in the exitBlock
    for (Operation &op : exitBlock->getOperations()) {
      // Skip constants
      if (isa<arith::ConstantOp>(&op)) continue;
      
      for (OpOperand &use : op.getOpOperands()) {
        Value operand = use.get();
        if (operand.getType().isa<MemRefType>()) continue;
        
        Operation *defOp = operand.getDefiningOp();

        // check if the defining operation or block args comes form any subblock
        bool comesFromSubloop = false;
        if (defOp && defOp->getBlock() == subloop) {
          comesFromSubloop = true;
        } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          if (blockArg.getOwner() == subloop)
            comesFromSubloop = true;
        }

        if (comesFromSubloop) {
          if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::IndexCastOp>(&op)) {
            // Trace past casts
            llvm::SmallVector<EntryOpInfo, 2> consumers;
            getRealConsumers(op.getResult(0), exitBlock, consumers);
            for (const auto &info : consumers) {
              addEntryOpAttr(info.op, info.operandIdx);
            }
          } else {
            addEntryOpAttr(&op, use.getOperandNumber());
          }
        }
      }
    }

    // construct the Metadata Dictionary
    NamedAttribute metadata[] = {
        builder.getNamedAttr("subloop_bb", builder.getI64IntegerAttr(subloopBbIdx)),
        builder.getNamedAttr("store_ops", builder.getArrayAttr(storeOpAttrs)),
        builder.getNamedAttr("entry_ops", builder.getArrayAttr(entryOpAttrs))
    };
    allLoopsMetadata.push_back(builder.getDictionaryAttr(metadata)); 

    // if storeOpAttrs isn't empty we need to add a pseudoconstant
    auto brOp = dyn_cast<cf::BranchOp>(outerHeader->getTerminator());
    builder.setInsertionPoint(brOp);
    Location loc = builder.getUnknownLoc();

    // create pseudo-constant in outerHeader
    auto pseudoCondOp = builder.create<arith::ConstantOp>(
        loc, 
        builder.getI1Type(), 
        builder.getBoolAttr(false)
    );
    std::string pseudoName = "pseudo_cond" + std::to_string(subloopBbIdx);
    pseudoCondOp->setAttr("handshake.name", builder.getStringAttr(pseudoName));

    // add a new block argument to the subblock
    Value condArg = subloop->addArgument(builder.getI1Type(), loc);

    // pass pseudoCondOp as a branch operand from outerHeader into subloop
    brOp.getDestOperandsMutable().append(pseudoCondOp.getResult());

    // update the loop back-edge (inside subloop) so it passes the pseudoconstant
    for (Operation &op : subloop->getOperations()) {
      if (auto condBr = dyn_cast<cf::CondBranchOp>(&op)) {
        if (condBr.getTrueDest() == subloop) {
          condBr.getTrueDestOperandsMutable().append(condArg);
        } else if (condBr.getFalseDest() == subloop) {
          condBr.getFalseDestOperandsMutable().append(condArg);
        }
      }
    }
    
  }

  // attach the attribute directly to modOp
  modOp->setAttr("handshake.subatomic_region_info", builder.getArrayAttr(allLoopsMetadata));
}