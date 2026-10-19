#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "dynamatic/Dialect/CFExtra/CFExtraOps.h"

#define DEBUG_TYPE "mark-subloop"

using namespace dynamatic;
using namespace mlir;

struct EntryOpInfo {
  Operation *op;
  unsigned operandIdx;
};

struct RegionInfo {
  Block *entryBlock;
  llvm::SetVector<Block *> subblocks;
  Block *exitBlock;
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

  /// Discovers valid single-entry/single-exit and loop subregions.
  llvm::SmallVector<std::pair<Block *, RegionInfo>>
  discoverRegions(Region &cfg, DominanceInfo &domInfo,
                  PostDominanceInfo &postDomInfo,
                  const llvm::DenseMap<Block *, unsigned> &blockIndices);

  /// Processes a single region, predicating stores and returning its metadata.
  DictionaryAttr processRegion(RegionInfo &region,
                               const llvm::DenseMap<Block *, unsigned> &blockIndices,
                               OpBuilder &builder, NameAnalysis &namer);

  /// Inserts RegionInputOp markers for values entering multi-successor regions.
  void markExternalInputs(ArrayRef<std::pair<Block *, RegionInfo>> sortedRegions,
                          llvm::DenseSet<Block *> &multiPredEntryBlocks,
                          OpBuilder &builder, NameAnalysis &namer);

  /// Wires BlockControlOps and control tokens across the function's CFG.
  void insertBlockControlNetwork(func::FuncOp funcOp,
                                 const llvm::DenseSet<Block *> &multiPredBlocks,
                                 OpBuilder &builder, NameAnalysis &namer);

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

llvm::SmallVector<std::pair<Block *, RegionInfo>>
MarkSubloopPass::discoverRegions(Region &cfg, DominanceInfo &domInfo,
                                 PostDominanceInfo &postDomInfo,
                                 const llvm::DenseMap<Block *, unsigned> &blockIndices) {
  llvm::DenseMap<Block *, RegionInfo> validRegions;

  // iterate over all dom / postdom pairs
  for (mlir::Block &entry : cfg) {
    if (entry.getNumSuccessors() == 1) {
      // entry has a single successor that is the header of a loop
      Block *loopHeader = entry.getSuccessor(0);

      // find the end of the loop
      Block *latch = nullptr;
      for (Block *pred : loopHeader->getPredecessors()) {
        if (domInfo.dominates(loopHeader, pred)) {
          latch = pred;
          break;
        }
      }

      if (!latch) continue;

      // collect all subblocks
      llvm::SetVector<mlir::Block *> subBlocksSet;
      subBlocksSet.insert(loopHeader);
      subBlocksSet.insert(latch);    
      for (mlir::Block &subblock : cfg) {
        // subblock is dominated by the entry and postdominated by the exit
        if (domInfo.properlyDominates(loopHeader, &subblock) &&
            postDomInfo.properlyPostDominates(latch, &subblock)) {
          subBlocksSet.insert(&subblock);
        }
      }
      
      // find the exitBlock after the latch
      Block *exit = nullptr;
      for (mlir::Block *succ : latch->getSuccessors()) {
        if (succ != loopHeader) {
          exit = succ;
          break;
        }
      }

      // add to validRegions
      validRegions[&entry] = {&entry, std::move(subBlocksSet), exit};
      continue;
    }

    // latches cannot be entries
    bool isLatch = llvm::any_of(entry.getSuccessors(), [&](Block *succ) {
      return domInfo.dominates(succ, &entry);
    });
    if (isLatch)
      continue;

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

      // If we already have a region for this entry that is smaller, skip
      auto it = validRegions.find(&entry);
      if (it != validRegions.end() && it->second.subblocks.size() <= subBlocksSet.size())
        continue;
      // Insert or overwrite with the shorter region
      validRegions[&entry] = {&entry, std::move(subBlocksSet), &exit};
    }
  }

  // Sort deterministically by block order in the parent region
  llvm::SmallVector<std::pair<Block *, RegionInfo>> sortedRegions(validRegions.begin(),
                                              validRegions.end());
  llvm::sort(sortedRegions, [&](const auto &lhs, const auto &rhs) {
    return blockIndices.lookup(lhs.first) < blockIndices.lookup(rhs.first);
  });

  return sortedRegions;
}

/// Processes a single region, predicating stores and returning its metadata.
/// Collects all entry operations = operations that are defined in the subblock
/// and used directly in the exitBlock. Finds all store operation and adds
/// a pseudo constant and a predicate in front of every store.
DictionaryAttr MarkSubloopPass::processRegion(RegionInfo &region,
                              const llvm::DenseMap<Block *, unsigned> &blockIndices,
                              OpBuilder &builder, NameAnalysis &namer) {

  Block *entryBlock = region.entryBlock;
  Block *exitBlock = region.exitBlock;
  llvm::SetVector<Block *> &subblocks = region.subblocks;
  for (mlir::Block *b : subblocks) {
    b->printAsOperand(llvm::errs());
    llvm::errs() << ", ";
  }
  llvm::errs() << "\n";

  unsigned entryBBIdx = blockIndices.lookup(entryBlock);

  // collect the block indices of all successors
  llvm::SmallVector<Attribute> successorAttrs;
  for (mlir::Block *successor : entryBlock->getSuccessors()) {
    int succIdx = blockIndices.lookup(successor);
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
  
  // add the entry operation for the control
  unsigned exitBlockIdx = blockIndices.lookup(exitBlock);
  entryOpAttrs.push_back(builder.getDictionaryAttr({
      builder.getNamedAttr("op", builder.getStringAttr("fake_control")),
      builder.getNamedAttr("operand_idx",
                           builder.getI64IntegerAttr(exitBlockIdx)),
  }));

  // find all the stores in the subblocks
  llvm::SmallVector<memref::StoreOp> storesToModify;
  for (mlir::Block *block : subblocks) {
    for (auto storeOp : block->getOps<memref::StoreOp>()) {
      storesToModify.push_back(storeOp);
    }
  }
  llvm::errs() << "found " << storesToModify.size() << 
    " stores in the subblocks\n";

  // insert one pseudo constant and a predicate in front of all the stores
  bool hasStores = !storesToModify.empty();
  if (hasStores) {
    builder.setInsertionPoint(entryBlock->getTerminator());

    // create pseudo-constant in entryBlock
    auto pseudoCondOp = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(), 
        builder.getI1Type(), 
        builder.getBoolAttr(false)
    );
    std::string pseudoName = "pseudo_cond" + std::to_string(entryBBIdx);
    pseudoCondOp->setAttr("handshake.name", builder.getStringAttr(pseudoName));

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

  return builder.getDictionaryAttr({
      builder.getNamedAttr("entry_bb", builder.getI64IntegerAttr(entryBBIdx)),
      builder.getNamedAttr("successor_bbs",
                           builder.getArrayAttr(successorAttrs)),
      builder.getNamedAttr("stores", builder.getBoolAttr(hasStores)),
      builder.getNamedAttr("entry_ops", builder.getArrayAttr(entryOpAttrs)),
  });
}

/// Inserts RegionInputOp markers for values entering multi-successor regions.
void MarkSubloopPass::markExternalInputs(ArrayRef<std::pair<Block *, RegionInfo>> sortedRegions,
                        llvm::DenseSet<Block *> &multiPredEntryBlocks,
                        OpBuilder &builder, NameAnalysis &namer) {

  for (const auto &[entryBlock, region] : sortedRegions) {
    // Only look at if/else regions
    if (entryBlock->getNumSuccessors() <= 1)
      continue;

    multiPredEntryBlocks.insert(entryBlock);
    const llvm::SetVector<Block *> &subblocks = region.subblocks;

    // Map each external Value to its marker in this specific entryBlock
    llvm::DenseMap<Value, Value> markedValues;
    Operation *terminator = entryBlock->getTerminator();

    auto processOperand = [&](OpOperand &use) {
      Value val = use.get();
      if (isa<MemRefType>(val.getType())) return;

      Block *defBlock = getTrueDefiningBlock(val);
      // we only care about values originating from outside the subblocks
      if (!defBlock || subblocks.contains(defBlock)) return;

      // Check if we already created a marker in entryBlock for this value
      auto markedIt = markedValues.find(val);
      if (markedIt != markedValues.end()) {
        use.set(markedIt->second);
        return;
      }

      // insert marker directly above entryBlock's terminator
      builder.setInsertionPoint(terminator);
      auto markerOp = builder.create<dynamatic::cf_extra::RegionInputOp>(
          terminator->getLoc(), val.getType(), val);
      namer.setName(markerOp);

      Value markedVal = markerOp.getResult();
      markedValues[val] = markedVal;

      // Rewire use to read from the marker output
      use.set(markedVal);
    };

    // mark the condition of entryBlock's conditional branch terminator (e.g., cmpi0)
    if (auto condBr = dyn_cast<cf::CondBranchOp>(terminator)) {
      processOperand(condBr.getConditionMutable()[0]);
    }

    // iterate over all operations in the subblocks
    for (Block *block : subblocks) {
      for (Operation &op : block->getOperations()) {
        for (OpOperand &use : op.getOpOperands()) {
          processOperand(use);
        }
      }
    }
  }
}

/// Wires BlockControlOps and control tokens across the function's CFG.
void MarkSubloopPass::insertBlockControlNetwork(func::FuncOp funcOp,
                                const llvm::DenseSet<Block *> &multiPredBlocks,
                                OpBuilder &builder, NameAnalysis &namer) {

  Type controlType = builder.getI32Type();
  Location loc = funcOp.getLoc();
  llvm::DenseMap<Block *, Value> blockControlValueMap; 

  // Insert BlockControlOp at the start of every block
  for (Block &block : funcOp.getBlocks()) {
    builder.setInsertionPointToStart(&block);
    Location bLoc = block.empty() ? loc : block.front().getLoc();

    if (block.isEntryBlock()) {
      // create constant `true` at the very beginning of the entry block
      auto initCst = builder.create<arith::ConstantOp>(
          bLoc, builder.getI32IntegerAttr(1));
      initCst->setAttr("handshake.name", builder.getStringAttr("control_constant"));

      // feed the constant directly into the first blockControlOp
      auto ctrlOp = builder.create<dynamatic::cf_extra::BlockControlOp>(
          bLoc, controlType, ValueRange{initCst.getResult()});
      namer.setName(ctrlOp);
      blockControlValueMap[&block] = ctrlOp.getResult();
      continue;
    }

    // Non-entry block: add incoming control taken argument and wire directly
    BlockArgument ctrlArg = block.addArgument(controlType, bLoc);
    auto ctrlOp = builder.create<dynamatic::cf_extra::BlockControlOp>(
        bLoc, controlType, ValueRange{ctrlArg});
    namer.setName(ctrlOp);

    // Default control output is the BlockControlOp result
    Value currentControlVal = ctrlOp.getResult();

    // Check if this block is an entry basic block with multiple predecessors/successors
    if (multiPredBlocks.contains(&block)) {
      builder.setInsertionPointAfter(ctrlOp);
      // Create the RegionInputOp right after the BlockControlOp
      auto regionInputOp = builder.create<dynamatic::cf_extra::RegionInputOp>(
          bLoc, controlType, ctrlOp.getResult());
      namer.setName(regionInputOp);
      
      // Update the control output to point to the RegionInputOp instead
      currentControlVal = regionInputOp.getResult();
    }

    blockControlValueMap[&block] = currentControlVal;
  }

  // Update all branch terminators to pass the source block's token
  // along every branch edge to the successor's new argument slot.
  for (Block &block : funcOp.getBlocks()) {
    Value srcToken = blockControlValueMap[&block];
    Operation *term = block.getTerminator();
    if (!term)
      continue;

    // Update branches and conditional branches separately
    if (auto brOp = dyn_cast<cf::BranchOp>(term)) {
      brOp.getDestOperandsMutable().append(srcToken);
    } else if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
      condBr.getTrueDestOperandsMutable().append(srcToken);
      condBr.getFalseDestOperandsMutable().append(srcToken);
    }
  }
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

  // Discover and sort valid regions
  llvm::SmallVector<std::pair<mlir::Block *, RegionInfo>> sortedRegions =
    discoverRegions(cfg, domInfo, postDomInfo, blockIndices);
  if (sortedRegions.empty()) {
    llvm::errs() << "No valid dom / postdom pairs with subblocks found!\n";
    return;
  }
  llvm::errs() << "size of validRegions: " << sortedRegions.size() << '\n';

  // Process regions and attach metadata to the modOp
  llvm::SmallVector<Attribute> allRegionsMetadata;
  for (auto &[entryBlock, region] : sortedRegions) {
    allRegionsMetadata.push_back(
        processRegion(region, blockIndices, builder, namer));
  }
  // attach the attribute directly to modOp
  modOp->setAttr("handshake.subregion_info", builder.getArrayAttr(allRegionsMetadata));

  // mark external inputs entering if/else regions
  llvm::DenseSet<Block *> multiPredEntryBlocks;
  markExternalInputs(sortedRegions, multiPredEntryBlocks, builder, namer);

  // Build the BlockControl network across all basic blocks
  insertBlockControlNetwork(funcOp, multiPredEntryBlocks, builder, namer);

  llvm::errs() << "finished mark subregion\n";
}
