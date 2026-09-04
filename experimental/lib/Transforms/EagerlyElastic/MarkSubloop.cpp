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

struct RegionInfo {
  mlir::Block *entryBlock;
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
  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  auto funcOps = modOp.getBody()->getOps<mlir::func::FuncOp>();
  assert(!funcOps.empty() && "No funcOp found!");

  mlir::func::FuncOp funcOp = *funcOps.begin();
  mlir::Region &cfg = funcOp.getBody();

  // precompute block indices
  llvm::DenseMap<Block *, unsigned> blockIndices;
  for (auto [idx, block] : llvm::enumerate(cfg))
    blockIndices[&block] = idx;

  // Compute Dominance Info for the cfg
  DominanceInfo domInfo(funcOp);
  PostDominanceInfo postDomInfo(funcOp);
  llvm::DenseMap<mlir::Block *, RegionInfo> validRegions;

  // iterate over all dom / postdom pairs
  for (mlir::Block &entry : cfg) {
    for (mlir::Block &exit : cfg) {
      if (&entry == &exit) continue;

      // must be a valid dom / postdom pair
      if (!domInfo.dominates(&entry, &exit)) continue;
      if (!postDomInfo.postDominates(&exit, &entry)) continue;
      
      // collect all subblocks
      llvm::SetVector<mlir::Block *> subBlocksSet;      
      for (mlir::Block &subblock : cfg) {
        // subblock is dominated by the entry and postdominated by the exit
        if (domInfo.properlyDominates(&entry, &subblock) &&
            postDomInfo.properlyPostDominates(&exit, &subblock)) {
          subBlocksSet.insert(&subblock);
        }
      }
      if (subBlocksSet.empty()) continue;

      // verify that subblocks stay within the exit boundary
      bool validSuccessors = true;
      for (mlir::Block *subblock : subBlocksSet) {
        for (mlir::Block *succ : subblock->getSuccessors()) {
          if (!subBlocksSet.contains(succ) && succ != &exit) {
            validSuccessors = false;
            break;
          }
        }
        if (!validSuccessors) break;
      }

      // verify that no control enters subblocks from anywhere except entry
      bool validPredecessors = true;
      for (mlir::Block *subblock : subBlocksSet) {
        for (mlir::Block *pred : subblock->getPredecessors()) {
          if (!subBlocksSet.contains(pred) && pred != &entry) {
            validPredecessors = false;
            break;
          }
        }
        if (!validPredecessors) break;
      }

      if (validSuccessors && validPredecessors) {
        // If we already have a region for this entry that is smaller, skip
        auto it = validRegions.find(&entry);
        if (it != validRegions.end() && it->second.subblocks.size() <= subBlocksSet.size())
          continue;
        // Insert or overwrite with the shorter region
        validRegions[&entry] = {&entry, std::move(subBlocksSet), &exit};
      }
    }
  }

  if (validRegions.empty()) {
    llvm::errs() << "No valid dom / postdom pairs with subblocks found!\n";
    return;
  }
  llvm::errs() << "size of validRegions: " << validRegions.size() << '\n';

  llvm::SmallVector<Attribute> allRegionsMetadata;

  // Process all discovered regions
  for (auto &[entryBlock, region] : validRegions) {
    llvm::SetVector<mlir::Block *> &subblocks = region.subblocks;
    mlir::Block *exitBlock = region.exitBlock;
    for (mlir::Block *b : subblocks) {
      b->printAsOperand(llvm::errs());
      llvm::errs() << ", ";
    }
    llvm::errs() << "\n";

    int entryBBIdx = blockIndices[entryBlock];
    bool storesInSubblock = false;

    // collect the block indices of all successors
    llvm::SmallVector<Attribute> successorAttrs;
    for (mlir::Block *successor : entryBlock->getSuccessors()) {
      int succIdx = blockIndices[successor];
      successorAttrs.push_back(builder.getI64IntegerAttr(succIdx));
    }

    // Collect the names and operandIDs of entry operations in exitBlock
    llvm::SmallVector<Attribute> entryOpAttrs;

    // Helper to format entry info into a DictionaryAttr
    auto addEntryOpAttr = [&](Operation *op, unsigned idx) {
      StringAttr nameAttr = op->getAttrOfType<StringAttr>("handshake.name");
      std::string nameStr = nameAttr.getValue().str();

      if (isa<dynamatic::cf_extra::PredicateOp>(op)) {
        nameStr = "cond_br_" + nameStr;
      }

      NamedAttribute entries[] = {
          builder.getNamedAttr("op", builder.getStringAttr(nameStr)),
          builder.getNamedAttr("operand_idx", builder.getI64IntegerAttr(idx))
      };
      entryOpAttrs.push_back(builder.getDictionaryAttr(entries));
    };

    // trace values defined in the subblocks used directly in the exitBlock
    for (Operation &op : exitBlock->getOperations()) {
      // Skip constants
      if (isa<arith::ConstantOp>(&op)) continue;

      for (OpOperand &use : op.getOpOperands()) {
        Value operand = use.get();
        if (isa<MemRefType>(operand.getType())) continue;

        // determine the defining block whether it's an OpResult or a BlockArgument
        Block *definingBlock = getTrueDefiningBlock(operand);

        // Check if the defining block belongs to subblocks
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
    for (mlir::Block *block : subblocks) {
      for (auto storeOp : block->getOps<memref::StoreOp>()) {
        storesInSubblock = true;
        storesToModify.push_back(storeOp);
      }
    }
    llvm::errs() << "found " << storesToModify.size() << " stores in the subblocks\n";

    // insert one pseudo constant and a predicate in front of all the stores
    if (storesInSubblock) {
      builder.setInsertionPoint(entryBlock->getTerminator());
      Location loc = builder.getUnknownLoc();

      // create pseudo-constant in entryBlock
      auto pseudoCondOp = builder.create<arith::ConstantOp>(
          loc, 
          builder.getI1Type(), 
          builder.getBoolAttr(false)
      );
      std::string pseudoName = "pseudo_cond" + std::to_string(entryBBIdx);
      pseudoCondOp->setAttr("handshake.name", builder.getStringAttr(pseudoName));
      pseudoCondOp->setAttr("ftd.pseudo_cond", builder.getUnitAttr());
        
      for (memref::StoreOp storeOp : storesToModify) {
        builder.setInsertionPoint(storeOp);

        // predicate data
        Value incomingData = storeOp.getValueToStore();
        auto predOp = builder.create<dynamatic::cf_extra::PredicateOp>(
            storeOp.getLoc(),
            incomingData.getType(),
            pseudoCondOp.getResult(),
            incomingData
        );
        namer.setName(predOp);
        // Rewire the data operand safely using setOperand on its specific index
        storeOp.setOperand(0, predOp->getResult(0));
      }
    }

    // construct the Metadata Dictionary
    allRegionsMetadata.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("entry_bb",
                             builder.getI64IntegerAttr(entryBBIdx)),
        builder.getNamedAttr("successor_bbs", 
                             builder.getArrayAttr(successorAttrs)),
        builder.getNamedAttr("stores", builder.getBoolAttr(storesInSubblock)),
        builder.getNamedAttr("entry_ops", builder.getArrayAttr(entryOpAttrs)),
    }));
  }

  // attach the attribute directly to modOp
  modOp->setAttr("handshake.subregion_info", builder.getArrayAttr(allRegionsMetadata));
  llvm::errs() << "finished mark subregion\n";
}
