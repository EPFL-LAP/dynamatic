#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "dynamatic/Dialect/CFExtra/CFExtraOps.h"

using namespace dynamatic;
using namespace mlir;

struct EntryOpInfo {
  Operation *op;
  unsigned operandIdx;
};

struct LoopInfo {
  mlir::Block *outerHeader;
  llvm::SetVector<mlir::Block *> subblocks;
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
  Block *getTrueDefiningBlock(Value val);

};


// Helper function to move past cast operations 
// keeps track of the operand index
void MarkSubloopPass::getRealConsumers(Value val, Block *targetBlock, llvm::SmallVectorImpl<EntryOpInfo> &results) {
  for (OpOperand &use : val.getUses()) {
    Operation *user = use.getOwner();
    if (user->getBlock() != targetBlock) continue;

    // if it's a cast/ext, trace past it recursively
    if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::IndexCastOp>(user)) {
      getRealConsumers(user->getResult(0), targetBlock, results);
    } else {
      // Record the actual operation and the specific operand index used
      results.push_back({user, use.getOperandNumber()});
    }
  }
}

// Traces a value backwards through BlockArgs to find its defining block
Block *MarkSubloopPass::getTrueDefiningBlock(Value val) {
  // if it's not a blockarg, its defining block is just the block
  if (Operation *defOp = val.getDefiningOp())
    return defOp->getBlock();

  // If it's a blockArg, inspect predecessor branches
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    Block *ownerBlock = blockArg.getOwner();
    unsigned argNum = blockArg.getArgNumber();

    // look at all predecessors that branch into this block
    for (Block *pred : ownerBlock->getPredecessors()) {
      Operation *terminator = pred->getTerminator();

      if (auto branchOp = dyn_cast<BranchOpInterface>(terminator)) {
        // Find which successor corresponds to ownerBlock
        for (unsigned i = 0, e = branchOp->getNumSuccessors(); i < e; ++i) {
          if (branchOp->getSuccessor(i) == ownerBlock) {
            SuccessorOperands succArgs = branchOp.getSuccessorOperands(i);
            if (argNum < succArgs.size()) {
              return getTrueDefiningBlock(succArgs[argNum]);
            }
          }
        }
      }
    }
    // fallback if no predecessor could be resolved
    return ownerBlock;
  }

  return nullptr;
}


void MarkSubloopPass::runOnOperation() {
  ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  getContext().loadDialect<dynamatic::cf_extra::CFExtraDialect>();

  auto funcOps = modOp.getBody()->getOps<mlir::func::FuncOp>();
  assert(funcOp && "No funcOp found!");

  mlir::func::FuncOp funcOp = *funcOps.begin();
  mlir::Region &region = funcOp.getBody();

  // Compute Dominance Info for the region
  DominanceInfo domInfo(funcOp);
  PostDominanceInfo postDomInfo(funcOp);
  llvm::SmallVector<LoopInfo> validLoops;

  // find all loops and collect their subblocks
  for (mlir::Block &header : region) {
    for (mlir::Block *exitBlock : header.getPredecessors()) {
      if (!domInfo.dominates(&header, exitBlock)) continue;
      
      llvm::SetVector<mlir::Block *> subBlocksSet;      
      for (mlir::Block &subblock : region) {
        // subblock is dominated by the header and postdominated by the exitblock
        if (domInfo.properlyDominates(&header, &subblock) &&
            postDomInfo.properlyPostDominates(exitBlock, &subblock)) {
          subBlocksSet.insert(&subblock);
        }
      }

      // check that subblocks stay within the loop boundary
      bool validSubblocks = true;
      for (mlir::Block *subblock : subBlocksSet) {
        for (mlir::Block *succ : subblock->getSuccessors()) {
          if (!subBlocksSet.contains(succ) && succ != exitBlock) {
            validSubblocks = false;
            break;
          }
        }
        if (!validSubblocks) break;
      }

      if (validSubblocks && !subBlocksSet.empty()) {
        validLoops.push_back({&header, subBlocksSet, exitBlock});
      }
    }
  }

  if (validLoops.empty()) {
    llvm::errs() << "No loops with subblocks found!\n";
    return;
  }
  llvm::errs() << "size of validLoops: " << validLoops.size() << '\n';

  llvm::SmallVector<Attribute> allLoopsMetadata;

  // Process all discovered loops
  for (auto &loop : validLoops) {
    mlir::Block *outerHeader = loop.outerHeader;
    llvm::SetVector<mlir::Block *> &subblocks = loop.subblocks;
    mlir::Block *exitBlock = loop.exitBlock;
    for (mlir::Block *b : subblocks) {
      b->printAsOperand(llvm::errs());
      llvm::errs() << ", ";
    }

    int outerHeaderBbIdx = std::distance(region.begin(), outerHeader->getIterator());
    
    // Collect the names and operandIDs of entry operations in exitBlock
    llvm::SmallVector<Attribute> entryOpAttrs;

    // Helper to format entry info into a DictionaryAttr
    auto addEntryOpAttr = [&](Operation *op, unsigned idx) {
      StringAttr nameAttr = op->getAttrOfType<StringAttr>("handshake.name");
      NamedAttribute entries[] = {
          builder.getNamedAttr("op", nameAttr),
          builder.getNamedAttr("operand_idx", builder.getI64IntegerAttr(idx))
      };
      entryOpAttrs.push_back(builder.getDictionaryAttr(entries));
    };

    // trace values defined in the subblocks used directly in the exitBlock
    for (Operation &op : exitBlock->getOperations()) {
      // Skip constants
      if (isa<arith::ConstantOp>(&op)) continue;
      if (isa<dynamatic::cf_extra::PredicateOp>(&op)) continue;

      for (OpOperand &use : op.getOpOperands()) {
        Value operand = use.get();
        if (isa<MemRefType>(operand.getType())) continue;
        
        // determine the defining block whether it's an OpResult or a BlockArgument
        Block *definingBlock = getTrueDefiningBlock(operand);

        // Check if the defining block belongs to the subloop
        if (subblocks.contains(definingBlock)) {
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

    // find all the stores in the subblocks
    llvm::SmallVector<memref::StoreOp> storesToModify;
    bool storesInSubblock = false;
    for (mlir::Block *block : subblocks) {
      for (auto storeOp : block->getOps<memref::StoreOp>()) {
        storesInSubblock = true;
        storesToModify.push_back(storeOp);
      }
    }
    llvm::errs() << "found " << storesToModify.size() << " stores in the subblocks\n";

    // insert one pseudo constant and a predicate in front of all the stores
    if (storesInSubblock) {
      builder.setInsertionPoint(outerHeader->getTerminator());
      Location loc = builder.getUnknownLoc();

      // create pseudo-constant in outerHeader
      auto pseudoCondOp = builder.create<arith::ConstantOp>(
          loc, 
          builder.getI1Type(), 
          builder.getBoolAttr(false)
      );
      std::string pseudoName = "pseudo_cond" + std::to_string(outerHeaderBbIdx);
      pseudoCondOp->setAttr("handshake.name", builder.getStringAttr(pseudoName));
        
      for (memref::StoreOp storeOp : storesToModify) {
        builder.setInsertionPoint(storeOp);

        Value incomingData = storeOp.getValueToStore();

        auto predOp = builder.create<dynamatic::cf_extra::PredicateOp>(
            storeOp.getLoc(),
            incomingData.getType(),
            pseudoCondOp.getResult(),
            incomingData
        );

        // Rewire the data operand safely using setOperand on its specific index
        storeOp.setOperand(0, predOp->getResult(0));
      }
    }

    // construct the Metadata Dictionary
    allLoopsMetadata.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("header_bb",
                             builder.getI64IntegerAttr(outerHeaderBbIdx)),
        builder.getNamedAttr("stores", builder.getBoolAttr(storesInSubblock)),
        builder.getNamedAttr("entry_ops", builder.getArrayAttr(entryOpAttrs)),
    }));
  }

  // attach the attribute directly to modOp
  modOp->setAttr("handshake.subloop_info", builder.getArrayAttr(allLoopsMetadata));
  llvm::errs() << "finished mark subloop\n";
}