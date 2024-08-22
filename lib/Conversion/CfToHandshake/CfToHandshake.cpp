//===- CfToHandhsake.cpp - Convert func/cf to handhsake dialect -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the elastic pass, as introduced in
// https://dl.acm.org/doi/abs/10.1145/3174243.3174264.
//
// Pars of the implementation are taken from CIRCT's cf-to-handshake conversion
// pass with modifications. Other parts of the implementation are significantly
// different, in particular those related to memory interface management and
// return network creation.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Analysis/ConstantAnalysis.h"
#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/FuncMaximizeSSA.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/BooleanLogic/Shannon.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <alloca.h>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <unordered_set>
#include <utility>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::affine;
using namespace mlir::memref;
using namespace dynamatic;
using namespace dynamatic::experimental::boolean;

//===-----------------------------------------------------------------------==//
// Helper functions
//===-----------------------------------------------------------------------==//

/// Determines whether an operation is akin to a load or store memory operation.
static bool isMemoryOp(Operation *op) {
  return isa<memref::LoadOp, memref::StoreOp, AffineReadOpInterface,
             AffineWriteOpInterface>(op);
}

/// Determines whether an operation is akin to a memory allocation operation.
static bool isAllocOp(Operation *op) {
  return isa<memref::AllocOp, memref::AllocaOp>(op);
}

/// Determines whether a memref type is suitable for covnersion in the context
/// of this pass.
static bool isValidMemrefType(Location loc, mlir::MemRefType type) {
  if (type.getNumDynamicDims() != 0 || type.getShape().size() != 1) {
    emitError(loc) << "memref's must be both statically sized and "
                      "unidimensional.";
    return false;
  }
  return true;
}

/// Extracts the memref argument to a memory operation and puts it in out.
/// Returns an error whenever the passed operation is not a memory operation.
static LogicalResult getOpMemRef(Operation *op, Value &out) {
  out = Value();
  if (auto memOp = dyn_cast<memref::LoadOp>(op))
    out = memOp.getMemRef();
  else if (auto memOp = dyn_cast<memref::StoreOp>(op))
    out = memOp.getMemRef();
  else if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
    affine::MemRefAccess access(op);
    out = access.memref;
  }
  if (out != Value())
    return success();
  return op->emitOpError() << "Unknown operation type.";
}

/// Returns the list of data inputs to be passed as operands to the
/// handshake::EndOp of a handshake::FuncOp. In the case of a single return
/// statement, this is simply the return's outputs. In the case of multiple
/// returns, this is the list of individually merged outputs of all returns.
/// In the latter case, the function inserts the required handshake::MergeOp's
/// in the region.
static SmallVector<Value, 8>
mergeFunctionResults(Region &r, ConversionPatternRewriter &rewriter,
                     SmallVector<Operation *, 4> &newReturnOps,
                     std::optional<size_t> endNetworkId) {
  Block *entryBlock = &r.front();
  if (newReturnOps.size() == 1) {
    // No need to merge results in case of single return
    return SmallVector<Value, 8>(newReturnOps[0]->getResults());
  }

  // Return values from multiple returns need to be merged together
  SmallVector<Value, 8> results;
  Location loc = entryBlock->getOperations().back().getLoc();
  rewriter.setInsertionPointToEnd(entryBlock);
  for (unsigned i = 0, e = newReturnOps[0]->getNumResults(); i < e; i++) {
    SmallVector<Value, 4> mergeOperands;
    for (auto *retOp : newReturnOps) {
      mergeOperands.push_back(retOp->getResult(i));
    }
    auto mergeOp = rewriter.create<handshake::MergeOp>(loc, mergeOperands);
    results.push_back(mergeOp.getResult());
    // Merge operation inherits from the bb atttribute of the latest (in program
    // order) return operation
    if (endNetworkId.has_value()) {
      mergeOp->setAttr(BB_ATTR_NAME,
                       rewriter.getUI32IntegerAttr(endNetworkId.value()));
    }
  }
  return results;
}

/// Returns a vector of control signals, one from each memory interface in the
/// circuit, to be passed as operands to the `handshake::EndOp` operation.
static SmallVector<Value, 8> getFunctionEndControls(Region &r) {
  SmallVector<Value, 8> controls;
  for (auto memOp : r.getOps<handshake::MemoryOpInterface>())
    controls.push_back(memOp->getResults().back());
  return controls;
}

/// Checks whether the blocks in `opsPerBlock`'s keys exhibit a "linear
/// dominance relationship" i.e., whether the execution of the "most dominant"
/// block necessarily triggers the execution of all others in a deterministic
/// order. This verification happens in linear time thanks to the cached
/// dominator/dominated relationships in `dominations`. On success, stores the
/// blocks' execution order in `dominanceOrder` ("most dominant" block first,
/// then "second most dominant", etc.). Fails when the blocks do not exhibit
/// that property.
static LogicalResult computeLinearDominance(
    DenseMap<Block *, DenseSet<Block *>> &dominations,
    llvm::MapVector<Block *, SmallVector<Operation *>> &opsPerBlock,
    SmallVector<Block *> &dominanceOrder) {
  // Initialize the dominance order to the proper size, setting each element to
  // nullptr initially
  size_t numBlocks = opsPerBlock.size();
  dominanceOrder.assign(numBlocks, nullptr);

  for (auto &[dominator, _] : opsPerBlock) {
    // Count the number of blocks among those of interest that it dominates
    size_t countDominated = 0;
    for (auto &[dominated, _] : opsPerBlock) {
      if (dominator != dominated && dominations[dominator].contains(dominated))
        ++countDominated;
    }

    // Figure out at which index in the dominance order the block should be
    // stored. The count is in (0, numBlocks - 1] and the index should be in the
    // same range, but in reverse order
    size_t idx = numBlocks - 1 - countDominated;

    if (dominanceOrder[idx]) {
      // This is not the first block which dominates this number of other
      // blocks, so there is no linear dominance relationship
      return failure();
    }
    dominanceOrder[idx] = dominator;
  }

  // At this point the dominanceOrder vector is necessarily completely filled
  return success();
}

/// Returns the value from the predecessor block that should be used as the data
/// operand of the merge-like operation under consideration.
static Value getMergeOperand(HandshakeLowering::MergeOpInfo &mergeInfo,
                             Block *predBlock, bool isFirstOperand) {
  // The input value to the merge operations
  Value srcVal = mergeInfo.blockArg;
  // The block the merge operation belongs to
  Block *block = mergeInfo.mergeLikeOp->getBlock();

  // The block terminator is either a cf-level branch or cf-level conditional
  // branch. In either case, identify the value passed to the block using its
  // index in the list of block arguments
  unsigned index = srcVal.cast<BlockArgument>().getArgNumber();
  Operation *termOp = predBlock->getTerminator();
  if (mlir::cf::CondBranchOp br = dyn_cast<mlir::cf::CondBranchOp>(termOp)) {
    // Block should be one of the two destinations of the conditional branch
    auto *trueDest = br.getTrueDest(), *falseDest = br.getFalseDest();
    if (block == trueDest) {
      if (!isFirstOperand && trueDest == falseDest)
        return br.getFalseOperand(index);
      return br.getTrueOperand(index);
    }
    assert(block == falseDest);
    return br.getFalseOperand(index);
  }
  if (isa<mlir::cf::BranchOp>(termOp))
    return termOp->getOperand(index);
  return nullptr;
}

/// Returns the first occurance within the block of an operation of the template
/// type. If none exists, returns nullptr.
template <typename Op>
static Op getFirstOp(Block *block) {
  auto ops = block->getOps<Op>();
  if (ops.empty())
    return nullptr;
  return *ops.begin();
}

/// Returns the number of predecessors of the block.
static unsigned getBlockPredecessorCount(Block *block) {
  auto predecessors = block->getPredecessors();
  return std::distance(predecessors.begin(), predecessors.end());
}

/// Replaces all backedges temporarily used as merge-like operation operands
/// with actual SSA values coming from predecessor blocks.
static void reconnectMergeOps(Region &region,
                              HandshakeLowering::BlockOps &blockMerges,
                              DenseMap<Value, Value> &mergePairs) {
  for (Block &block : region) {
    for (HandshakeLowering::MergeOpInfo &mergeInfo : blockMerges[&block]) {
      size_t operandIdx = 0;
      // Set appropriate operand from each predecessor block
      for (Block *predBlock : block.getPredecessors()) {
        Value mgOperand =
            getMergeOperand(mergeInfo, predBlock, operandIdx == 0);
        assert(mgOperand != nullptr);
        if (!mgOperand.getDefiningOp()) {
          assert(mergePairs.count(mgOperand));
          mgOperand = mergePairs[mgOperand];
        }
        mergeInfo.dataEdges[operandIdx].setValue(mgOperand);
        operandIdx++;
      }

      // Reconnect all operands originating from livein defining value through
      // corresponding merge of that block
      for (Operation &opp : block) {
        if (!isa<handshake::MergeLikeOpInterface>(&opp)) {
          opp.replaceUsesOfWith(mergeInfo.blockArg,
                                mergeInfo.mergeLikeOp->getResult(0));
        }
      }
    }
  }

  // Connect  select operand of muxes to control merge's index result in all
  // blocks with more than one predecessor
  for (Block &block : region) {
    if (getBlockPredecessorCount(&block) > 1) {
      auto ctrlMergeOp = getFirstOp<handshake::ControlMergeOp>(&block);
      assert(ctrlMergeOp != nullptr);

      for (HandshakeLowering::MergeOpInfo &mergeInfo : blockMerges[&block]) {
        if (mergeInfo.mergeLikeOp != ctrlMergeOp) {
          // If the block has multiple predecessors, merge-like operation that
          // are not the block's control merge must have an index operand (at
          // this point, an index backedge)
          assert(mergeInfo.indexEdge.has_value());
          (*mergeInfo.indexEdge).setValue(ctrlMergeOp->getResult(1));
        }
      }
    }
  }
}

/// Returns the branch result of the new handshake-level branch operation that
/// goes to the successor block of the old cf-level branch result.
static Value getSuccResult(Operation *brOp, Operation *newBrOp,
                           Block *succBlock) {
  // For conditional block, check if result goes to true or to false successor
  if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(brOp)) {
    if (condBranchOp.getTrueDest() == succBlock)
      return dyn_cast<handshake::ConditionalBranchOp>(newBrOp).getTrueResult();
    assert(condBranchOp.getFalseDest() == succBlock);
    return dyn_cast<handshake::ConditionalBranchOp>(newBrOp).getFalseResult();
  }
  // If the block is unconditional, newOp has only one result
  return newBrOp->getResult(0);
}

/// Returns the unique data operands of a cf-level branch-like operation.
static SetVector<Value> getBranchOperands(Operation *termOp) {
  if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(termOp)) {
    OperandRange oprds = condBranchOp.getOperands().drop_front();
    return SetVector<Value>{oprds.begin(), oprds.end()};
  }
  assert(isa<mlir::cf::BranchOp>(termOp) && "unsupported block terminator");
  OperandRange oprds = termOp->getOperands();
  return SetVector<Value>{oprds.begin(), oprds.end()};
}

//===-----------------------------------------------------------------------==//
// HandshakeLowering
//===-----------------------------------------------------------------------==//

LogicalResult
HandshakeLowering::createControlNetwork(ConversionPatternRewriter &rewriter) {

  // Add start point of the control-only path to the entry block's arguments
  Block *entryBlock = &region.front();
  startCtrl =
      entryBlock->addArgument(rewriter.getNoneType(), rewriter.getUnknownLoc());
  setBlockEntryControl(entryBlock, startCtrl);

  // Add a control-only argument to each block
  for (auto &block : region.getBlocks())
    if (!block.isEntryBlock())
      setBlockEntryControl(&block, block.addArgument(startCtrl.getType(),
                                                     rewriter.getUnknownLoc()));
  // Modify branch-like block terminators to forward control value through
  // all blocks
  for (auto &block : region.getBlocks())
    if (auto op = dyn_cast<BranchOpInterface>(block.getTerminator()); op)
      for (unsigned i = 0, e = op->getNumSuccessors(); i < e; i++)
        op.getSuccessorOperands(i).append(getBlockEntryControl(&block));

  return success();
}

HandshakeLowering::MergeOpInfo
HandshakeLowering::insertMerge(BlockArgument blockArg,
                               BackedgeBuilder &edgeBuilder,
                               ConversionPatternRewriter &rewriter) {
  Block *block = blockArg.getOwner();
  unsigned numPredecessors = getBlockPredecessorCount(block);
  Location insertLoc = block->front().getLoc();
  SmallVector<Backedge> dataEdges;
  SmallVector<Value> operands;

  // Every block (except the entry block) needs to feed it's entry control into
  // a control merge
  if (blockArg == getBlockEntryControl(block)) {
    Operation *mergeOp;
    if (block == &region.front()) {
      // For consistency within the entry block, replace the latter's entry
      // control with the output of a merrge that takes the control-only
      // network's start point as input. This makes it so that only the
      // merge's output is used as a control within the entry block, instead
      // of a combination of the MergeOp's output and the function/block control
      // argument. Taking this step out should have no impact on functionality
      // but would make the resulting IR less "regular"
      operands.push_back(blockArg);
      mergeOp = rewriter.create<handshake::MergeOp>(insertLoc, operands);
    } else {
      for (unsigned i = 0; i < numPredecessors; i++) {
        Backedge edge = edgeBuilder.get(rewriter.getNoneType());
        dataEdges.push_back(edge);
        operands.push_back(Value(edge));
      }
      mergeOp = rewriter.create<handshake::ControlMergeOp>(insertLoc, operands);
    }
    setBlockEntryControl(block, mergeOp->getResult(0));
    return MergeOpInfo{dyn_cast<handshake::MergeLikeOpInterface>(mergeOp),
                       blockArg, dataEdges};
  }

  // Every live-in value to a block is passed through a merge-like operation,
  // even when it's not required for circuit correctness (useless merge-like
  // operations are removed down the line during Handshake canonicalization)

  // Insert "dummy" merges for blocks with less than two predecessors
  if (numPredecessors <= 1) {
    if (numPredecessors == 0) {
      // All of the entry block's block arguments get passed through a dummy
      // merge. There is no need for a backedge here as the unique operand can
      // be resolved immediately
      operands.push_back(blockArg);
    } else {
      // The value incoming from the single block predecessor will be resolved
      // later during merge reconnection
      Backedge edge = edgeBuilder.get(blockArg.getType());
      dataEdges.push_back(edge);
      operands.push_back(Value(edge));
    }
    auto mergeOp = rewriter.create<handshake::MergeOp>(insertLoc, operands);
    return MergeOpInfo{mergeOp, blockArg, dataEdges};
  }

  // Create a backedge for the index operand, and another one for each data
  // operand. The index operand will eventually resolve to the current block's
  // control merge index output, while data operands will resolve to their
  // respective values from each block predecessor
  Backedge indexEdge = edgeBuilder.get(rewriter.getIndexType());
  for (unsigned i = 0; i < numPredecessors; i++) {
    Backedge edge = edgeBuilder.get(blockArg.getType());
    dataEdges.push_back(edge);
    operands.push_back(Value(edge));
  }
  handshake::MuxOp muxOp =
      rewriter.create<handshake::MuxOp>(insertLoc, Value(indexEdge), operands);
  return MergeOpInfo{muxOp, blockArg, dataEdges, indexEdge};
}

LogicalResult
HandshakeLowering::addMergeOps(ConversionPatternRewriter &rewriter) {
  // Stores mapping from each value that passes through a merge-like operation
  // to the data result of that merge operation
  DenseMap<Value, Value> mergePairs;

  // Create backedge builder to manage operands of merge operations between
  // insertion and reconnection
  BackedgeBuilder edgeBuilder{rewriter, region.front().front().getLoc()};

  // Insert merge operations (with backedges instead of actual operands)
  BlockOps blockMerges;
  for (Block &block : region) {
    rewriter.setInsertionPointToStart(&block);

    // All of the block's live-ins are passed explictly through block arguments
    // thanks to prior SSA maximization
    for (BlockArgument arg : block.getArguments()) {
      // No merges on memref block arguments; these are handled separately
      if (arg.getType().isa<mlir::MemRefType>())
        continue;

      MergeOpInfo mergeInfo = insertMerge(arg, edgeBuilder, rewriter);
      blockMerges[&block].push_back(mergeInfo);
      mergePairs[arg] = mergeInfo.mergeLikeOp->getResult(0);
    }
  }

  // Reconnect merge operations with values incoming from predecessor blocks
  // and resolve all backedges that were created during merge insertion
  reconnectMergeOps(region, blockMerges, mergePairs);

  // Remove all block arguments, which are no longer used
  for (Block &block : region) {
    if (!block.isEntryBlock()) {
      for (unsigned idx = block.getNumArguments(); idx > 0; --idx)
        block.eraseArgument(idx - 1);
    }
  }

  return success();
}

LogicalResult
HandshakeLowering::addBranchOps(ConversionPatternRewriter &rewriter) {
  for (Block &block : region) {
    Operation *termOp = block.getTerminator();
    Location loc = termOp->getLoc();
    rewriter.setInsertionPoint(termOp);

    Value cond = nullptr;
    if (cf::CondBranchOp condBranchOp = dyn_cast<cf::CondBranchOp>(termOp))
      cond = condBranchOp.getCondition();
    else if (isa<func::ReturnOp>(termOp))
      continue;

    // Insert a branch-like operation for each live-out and replace the original
    // branch operand value in successor blocks with the result(s) of the new
    // operation
    for (Value val : getBranchOperands(termOp)) {
      // Create a branch-like operation for the branch operand
      Operation *newOp = nullptr;
      if (cond)
        newOp = rewriter.create<handshake::ConditionalBranchOp>(loc, cond, val);
      else
        newOp = rewriter.create<handshake::BranchOp>(loc, val);

      // Connect the newly created branch's outputs with their successors by
      // looking for merge-like operations in successor blocks that take as
      // input the original branch operand, and replace the latter with a result
      // of the newly inserted branch operation
      for (Block *succ : block.getSuccessors()) {
        for (Operation *user : llvm::make_early_inc_range(val.getUsers())) {
          Block *userBlock = user->getBlock();
          if (userBlock == succ && isa<handshake::MergeLikeOpInterface>(user))
            user->replaceUsesOfWith(val, getSuccResult(termOp, newOp, succ));
        }
      }
    }
  }

  return success();
}

LogicalResult HandshakeLowering::replaceMemoryOps(
    ConversionPatternRewriter &rewriter,
    HandshakeLowering::MemInterfacesInfo &memInfo) {

  // Make sure to record external memories passed as function arguments, even if
  // they aren't used by any memory operation
  for (BlockArgument arg : region.getArguments()) {
    if (mlir::MemRefType memref = dyn_cast<mlir::MemRefType>(arg.getType())) {
      // Ensure that this is a valid memref-typed value.
      if (!isValidMemrefType(arg.getLoc(), memref))
        return failure();
      memInfo[arg] = {};
    }
  }

  // Used to keep consistency betweeen memory access names referenced by memory
  // dependencies and names of replaced memory operations
  MemoryOpLowering memOpLowering(nameAnalysis);

  // Replace load and store operations with their corresponding Handshake
  // equivalent. Traverse and store memory operations in program order (required
  // by memory interface placement later)
  for (Operation &op : llvm::make_early_inc_range(region.getOps())) {
    if (!isMemoryOp(&op))
      continue;

    // For now we don't support memory allocations within the kernels
    if (isAllocOp(&op))
      return op.emitOpError()
             << "Allocation operations are not supported during "
                "cf-to-handshake lowering.";

    // Extract the reference to the memory region from the memory operation
    rewriter.setInsertionPoint(&op);
    Value memref;
    if (getOpMemRef(&op, memref).failed())
      return failure();
    Operation *newOp = nullptr;
    Location loc = op.getLoc();

    // The memory operation must have a MemInterfaceAttr attribute attached
    StringRef attrName = handshake::MemInterfaceAttr::getMnemonic();
    auto memAttr = op.getAttrOfType<handshake::MemInterfaceAttr>(attrName);
    if (!memAttr)
      return op.emitError()
             << "Memory operation must have attribute " << attrName
             << " of type dynamatic::handshake::MemInterfaceAttr to decide "
                "which memory interface it should connect to.";
    bool connectToMC = memAttr.connectsToMC();

    // Replace memref operation with corresponding handshake operation
    LogicalResult res =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<memref::LoadOp>([&](memref::LoadOp loadOp) {
              OperandRange indices = loadOp.getIndices();
              assert(indices.size() == 1 && "load must be unidimensional");
              Value addr = indices.front();
              MemRefType type = cast<MemRefType>(memref.getType());

              if (connectToMC)
                newOp = rewriter.create<handshake::MCLoadOp>(loc, type, addr);
              else
                newOp = rewriter.create<handshake::LSQLoadOp>(loc, type, addr);

              // Replace uses of old load result with data result of new load
              op.getResult(0).replaceAllUsesWith(
                  dyn_cast<handshake::LoadOpInterface>(newOp).getDataOutput());
              return success();
            })
            .Case<memref::StoreOp>([&](memref::StoreOp storeOp) {
              OperandRange indices = storeOp.getIndices();
              assert(indices.size() == 1 && "store must be unidimensional");
              Value addr = indices.front();
              Value data = storeOp.getValueToStore();

              if (connectToMC)
                newOp = rewriter.create<handshake::MCStoreOp>(loc, addr, data);
              else
                newOp = rewriter.create<handshake::LSQStoreOp>(loc, addr, data);
              return success();
            })
            .Default([&](auto) {
              return op.emitError() << "Memory operation type unsupported.";
            });
    if (failed(res))
      return failure();

    // Record the memory access replacement
    memOpLowering.recordReplacement(&op, newOp, false);

    // Associate the new operation with the memory region it references and
    // information about the memory interface it should connect to
    if (memAttr.connectsToMC())
      memInfo[memref].mcPorts[op.getBlock()].push_back(newOp);
    else
      memInfo[memref].lsqPorts[*memAttr.getLsqGroup()].push_back(newOp);

    // Erase the original operation
    rewriter.eraseOp(&op);
  }

  // Change the name of destination memory acceses in all stored memory
  // dependencies to reflect the new access names
  memOpLowering.renameDependencies(region.getParentOp());

  return success();
}

std::vector<Operation *> getLSQPredecessors(
    const llvm::MapVector<unsigned, SmallVector<Operation *>> &lsqPorts) {
  // Create a vector to hold all the Operation* pointers
  std::vector<Operation *> combinedOperations;
  // Iterate over the MapVector and add all Operation* to the combinedOperations
  // vector
  for (const auto &entry : lsqPorts) {
    const SmallVector<Operation *> &operations = entry.second;
    combinedOperations.insert(combinedOperations.end(), operations.begin(),
                              operations.end());
  }
  return combinedOperations;
}

LogicalResult HandshakeLowering::verifyAndCreateMemInterfaces(
    ConversionPatternRewriter &rewriter, MemInterfacesInfo &memInfo) {
  // Create a mapping between each block and all the other blocks it properly
  // dominates so that we can quickly determine whether LSQ groups make sense
  DominanceInfo domInfo;
  DenseMap<Block *, DenseSet<Block *>> dominations;
  for (Block &maybeDominator : region) {
    // Start with an empty set of dominated blocks for each potential dominator
    dominations[&maybeDominator] = {};
    for (Block &maybeDominated : region) {
      if (&maybeDominator == &maybeDominated)
        continue;
      if (domInfo.properlyDominates(&maybeDominator, &maybeDominated))
        dominations[&maybeDominator].insert(&maybeDominated);
    }
  }

  // Create a mapping between each block and its control value in the right
  // format for the memory interface builder
  DenseMap<unsigned, Value> ctrlVals;
  for (auto [blockIdx, block] : llvm::enumerate(region))
    ctrlVals[blockIdx] = getBlockEntryControl(&block);

  // Each memory region is independent from the others
  for (auto &[memref, memAccesses] : memInfo) {
    SmallPtrSet<Block *, 4> controlBlocks;

    MemoryInterfaceBuilder memBuilder(
        region.getParentOfType<handshake::FuncOp>(), memref, ctrlVals);

    // Add MC ports to the interface builder
    for (const auto &[_, mcBlockOps] : memAccesses.mcPorts) {
      for (Operation *mcOp : mcBlockOps)
        memBuilder.addMCPort(mcOp);
    }

    // Determine LSQ group validity and add ports the the interface builder at
    // the same time
    for (auto &[group, groupOps] : memAccesses.lsqPorts) {
      assert(!groupOps.empty() && "group cannot be empty");

      // Group accesses by the basic block they belong to
      llvm::MapVector<Block *, SmallVector<Operation *>> opsPerBlock;
      for (Operation *op : groupOps)
        opsPerBlock[op->getBlock()].push_back(op);

      // Check whether there is a clear "linear dominance" relationship between
      // all blocks, and derive a port ordering for the group from it
      SmallVector<Block *> order;
      if (failed(computeLinearDominance(dominations, opsPerBlock, order)))
        return failure();

      // Verify that no two groups have the same control signal
      if (auto [_, newCtrl] = controlBlocks.insert(order.front()); !newCtrl)
        return groupOps.front()->emitError()
               << "Inconsistent^bb1  LSQ group for memory interface the "
                  "operation "
                  "references. No two groups can have the same control signal.";

      // Add all group ports in the correct order to the builder. Within each
      // block operations are naturally in program order since we always use
      // ordered maps and iterated over the operations in program order to begin
      // with
      for (Block *block : order) {
        for (Operation *lsqOp : opsPerBlock[block])
          memBuilder.addLSQPort(group, lsqOp);
      }
    }

    if (memAccesses.lsqPorts.size() > 0) {
      /// Construction of Allocation Network
      std::vector<Operation *> allOperations =
          getLSQPredecessors(memAccesses.lsqPorts);

      std::vector<ProdConsMemDep> allMemDeps;
      identifyMemDeps(allOperations, allMemDeps);

      /// Stores the Groups graph required for the allocation network analysis
      std::set<Group *, GroupsComparator> groups;
      constructGroupsGraph(allOperations, allMemDeps, groups);

      minimizeGroupsConnections(groups);

      // Build the memory interfaces
      handshake::MemoryControllerOp mcOp;
      handshake::LSQOp lsqOp;
      // if (failed(memBuilder.instantiateInterfaces(rewriter, mcOp, lsqOp)))
      //   return failure();

      /// Associates basic blocks of the region being lowered to their
      /// respective control value.
      DenseMap<Block *, Operation *> forksGraph;

      if (failed(memBuilder.instantiateInterfacesWithForks(
              rewriter, mcOp, lsqOp, groups, forksGraph, startCtrl,
              allocationNetwork)))
        return failure();

      if (failed(addMergeNonLoop(rewriter, allMemDeps, groups, forksGraph)))
        return failure();

      if (failed(addMergeLoop(rewriter, groups, forksGraph)))
        return failure();

      if (failed(joinInsertion(rewriter, groups, forksGraph)))
        return failure();
    } else {
      handshake::MemoryControllerOp mcOp;
      handshake::LSQOp lsqOp;
      if (failed(memBuilder.instantiateInterfaces(rewriter, mcOp, lsqOp)))
        return failure();
    }
  }

  return success();
}

LogicalResult
HandshakeLowering::convertCalls(ConversionPatternRewriter &rewriter) {
  auto modOp = region.getParentOfType<mlir::ModuleOp>();
  for (Block &block : region) {
    for (auto callOp : block.getOps<func::CallOp>()) {
      // The instance's operands are the same as the call plus an extra
      // control-only start coming from the call's parent basic block
      SmallVector<Value> operands(callOp.getOperands());
      operands.push_back(getBlockEntryControl(&block));

      // Retrieve the Handshake function that the call references to determine
      // the instance's result types (may be different from the call's result
      // types)
      SymbolRefAttr symbol = callOp->getAttrOfType<SymbolRefAttr>("callee");
      assert(symbol && "call symbol does not exist");
      Operation *lookup = modOp.lookupSymbol(symbol);
      if (!lookup)
        return callOp->emitError() << "call references unknown function";
      auto funcOp = dyn_cast<handshake::FuncOp>(lookup);
      if (!funcOp)
        return callOp->emitError() << "call does not reference a function";
      TypeRange resultTypes = funcOp.getFunctionType().getResults();

      // Replace the call with the Handshake instance
      rewriter.setInsertionPoint(callOp);
      auto instOp = rewriter.create<handshake::InstanceOp>(
          callOp.getLoc(), callOp.getCallee(), resultTypes, operands);
      if (callOp->getNumResults() == 0)
        rewriter.eraseOp(callOp);
      else
        rewriter.replaceOp(callOp, instOp->getResults());
    }
  }
  return success();
}

LogicalResult
HandshakeLowering::connectConstants(ConversionPatternRewriter &rewriter) {
  auto constants = region.getOps<mlir::arith::ConstantOp>();
  for (auto cstOp : llvm::make_early_inc_range(constants)) {
    rewriter.setInsertionPoint(cstOp);
    TypedAttr cstAttr = cstOp.getValue();
    Value controlVal;
    if (isCstSourcable(cstOp)) {
      auto sourceOp = rewriter.create<handshake::SourceOp>(
          cstOp.getLoc(), rewriter.getNoneType());
      controlVal = sourceOp.getResult();
    } else {
      controlVal = getBlockEntryControl(cstOp->getBlock());
    }
    rewriter.replaceOpWithNewOp<handshake::ConstantOp>(cstOp, cstAttr.getType(),
                                                       cstAttr, controlVal);
  }
  return success();
}

LogicalResult
HandshakeLowering::replaceUndefinedValues(ConversionPatternRewriter &rewriter) {
  for (auto &block : region) {
    for (auto undefOp : block.getOps<mlir::LLVM::UndefOp>()) {
      // Create an attribute of the appropriate type for the constant
      auto resType = undefOp.getRes().getType();
      TypedAttr cstAttr;
      if (isa<IndexType>(resType))
        cstAttr = rewriter.getIndexAttr(0);
      else if (isa<IntegerType>(resType))
        cstAttr = rewriter.getIntegerAttr(resType, 0);
      else if (FloatType floatType = dyn_cast<FloatType>(resType))
        cstAttr = rewriter.getFloatAttr(floatType, 0.0);
      else
        return undefOp->emitError() << "operation has unsupported result type";

      // Create a constant with a default value and replace the undefined value
      rewriter.setInsertionPoint(undefOp);
      auto cstOp = rewriter.create<handshake::ConstantOp>(
          undefOp.getLoc(), resType, cstAttr, startCtrl);
      rewriter.replaceOp(undefOp, cstOp.getResult());
    }
  }
  return success();
}

LogicalResult
HandshakeLowering::idBasicBlocks(ConversionPatternRewriter &rewriter) {
  for (auto [blockID, block] : llvm::enumerate(region)) {
    for (Operation &op : block) {
      if (!isa<handshake::MemoryOpInterface>(op)) {
        // Memory interfaces do not naturally belong to any block, so they do
        // not get an attribute
        op.setAttr(BB_ATTR_NAME, rewriter.getUI32IntegerAttr(blockID));
      }
    }
  }
  return success();
}

LogicalResult
HandshakeLowering::createReturnNetwork(ConversionPatternRewriter &rewriter) {
  Block *entryBlock = &region.front();
  auto &entryBlockOps = entryBlock->getOperations();

  // Move all operations to entry block. While doing so, delete all block
  // terminators and create a handshake-level return operation for each
  // existing
  // func-level return operation
  SmallVector<Operation *> terminatorsToErase;
  SmallVector<Operation *, 4> newReturnOps;
  for (Block &block : region) {
    Operation &termOp = block.back();
    if (isa<func::ReturnOp>(termOp)) {
      SmallVector<Value, 8> operands(termOp.getOperands());
      // When the enclosing function only returns a control value (no data
      // results), return statements must take exactly one control-only input
      if (operands.empty())
        operands.push_back(getBlockEntryControl(&block));

      // Insert new return operation next to the old one
      rewriter.setInsertionPoint(&termOp);
      auto newRet =
          rewriter.create<handshake::ReturnOp>(termOp.getLoc(), operands);
      newReturnOps.push_back(newRet);

      // New return operation belongs in the same basic block as the old one
      inheritBB(&termOp, newRet);
    }
    terminatorsToErase.push_back(&termOp);
    entryBlockOps.splice(entryBlockOps.end(), block.getOperations());
  }
  assert(!newReturnOps.empty() && "function must have at least one return");

  // When identifying basic blocks, the end node is either put in the same
  // block as the function's single return statement or, in the case of
  // multiple return statements, it is put in a "fake block" along with the
  // merges that feed it its data inputs
  std::optional<size_t> endNetworkID{};
  endNetworkID = (newReturnOps.size() > 1)
                     ? region.getBlocks().size()
                     : newReturnOps[0]
                           ->getAttrOfType<mlir::IntegerAttr>(BB_ATTR_NAME)
                           .getValue()
                           .getZExtValue();

  // Erase all blocks except the entry block
  for (Block &block : llvm::make_early_inc_range(llvm::drop_begin(region, 1))) {
    block.clear();
    block.dropAllDefinedValueUses();
    block.eraseArguments(0, block.getNumArguments());
    block.erase();
  }

  // Erase all leftover block terminators
  for (auto *op : terminatorsToErase)
    op->erase();

  // Insert an end node at the end of the function that merges results from
  // all handshake-level return operations and wait for all memory controllers
  // to signal completion
  SmallVector<Value, 8> endOperands;
  endOperands.append(
      mergeFunctionResults(region, rewriter, newReturnOps, endNetworkID));
  endOperands.append(getFunctionEndControls(region));
  rewriter.setInsertionPointToEnd(entryBlock);
  handshake::EndOp endOp = rewriter.create<handshake::EndOp>(
      entryBlockOps.back().getLoc(), endOperands);
  if (endNetworkID.has_value())
    endOp->setAttr(BB_ATTR_NAME,
                   rewriter.getUI32IntegerAttr(endNetworkID.value()));

  return success();
}

//===-----------------------------------------------------------------------==//
// Construction of Allocation Network
//===-----------------------------------------------------------------------==/

// checks wether an operation is a mmory operation
bool checkMemOp(Operation *op) {
  return isa<handshake::LSQStoreOp, handshake::LSQLoadOp>(op);
}

// checks wether 2 operations belong to the same BB
bool checkSameBB(Operation *op1, Operation *op2) {
  return (op1->getBlock() == op2->getBlock());
}

// checks wether 2 operations are both load operations
bool checkBothLd(Operation *op1, Operation *op2) {
  return (isa<handshake::LSQLoadOp>(op1) && isa<handshake::LSQLoadOp>(op2));
}

// Recursively check weather 2 block belong to the same loop, starting from
// the inner-most loops
bool checkSameLoop(CFGLoop *loop1, CFGLoop *loop2) {
  if (!loop1 || !loop2)
    return false;
  return (loop1 == loop2 || checkSameLoop(loop1->getParentLoop(), loop2) ||
          checkSameLoop(loop1, loop2->getParentLoop()) ||
          checkSameLoop(loop1->getParentLoop(), loop2->getParentLoop()));
}

// checks if the source and destination are in a loop
bool HandshakeLowering::sameLoop(Block *source, Block *dest) {
  return checkSameLoop(li.getLoopFor(source), li.getLoopFor(dest));
}

CFGLoop *checkInnermostCommonLoop(CFGLoop *loop1, CFGLoop *loop2) {
  if (!loop1 || !loop2)
    return nullptr;

  if (loop1 == loop2)
    return loop1;

  // Recursively check for common loops in parent loops
  CFGLoop *parentCommonLoop1 =
      checkInnermostCommonLoop(loop1->getParentLoop(), loop2);
  CFGLoop *parentCommonLoop2 =
      checkInnermostCommonLoop(loop1, loop2->getParentLoop());
  CFGLoop *parentCommonLoop3 =
      checkInnermostCommonLoop(loop1->getParentLoop(), loop2->getParentLoop());

  // Return the deepest common loop found
  if (parentCommonLoop1)
    return parentCommonLoop1;
  if (parentCommonLoop2)
    return parentCommonLoop2;
  if (parentCommonLoop3)
    return parentCommonLoop3;

  // If no common loop is found, return null
  return nullptr;
}

CFGLoop *HandshakeLowering::getInnermostCommonLoop(Block *block1,
                                                   Block *block2) {
  return checkInnermostCommonLoop(li.getLoopFor(block1), li.getLoopFor(block2));
}

///----------Overriding Operators for Blocks----------

bool lessThanBlocks(Block *block1, Block *block2) {
  std::string result1;
  llvm::raw_string_ostream os1(result1);
  block1->printAsOperand(os1);
  std::string block1id = os1.str();
  int id1 = std::stoi(block1id.substr(3));

  std::string result2;
  llvm::raw_string_ostream os2(result2);
  block2->printAsOperand(os2);
  std::string block2id = os2.str();
  int id2 = std::stoi(block2id.substr(3));

  return id1 < id2;
}

bool greaterThanBlocks(Block *block1, Block *block2) {
  std::string result1;
  llvm::raw_string_ostream os1(result1);
  block1->printAsOperand(os1);
  std::string block1id = os1.str();
  int id1 = std::stoi(block1id.substr(3));

  std::string result2;
  llvm::raw_string_ostream os2(result2);
  block2->printAsOperand(os2);
  std::string block2id = os2.str();
  int id2 = std::stoi(block2id.substr(3));

  return id1 > id2;
}

void dfsAllPaths(Block *start, Block *end, std::vector<Block *> &path,
                 std::unordered_set<Block *> &visited,
                 std::vector<std::vector<Block *>> &allPaths) {
  path.push_back(start);
  visited.insert(start);

  if (start == end) {
    allPaths.push_back(path);
  } else {
    for (Block *successor : start->getSuccessors()) {
      if (visited.find(successor) == visited.end())
        dfsAllPaths(successor, end, path, visited, allPaths);
    }
  }

  path.pop_back();
  visited.erase(start);
}

// Gets all the paths from a start block to and end block
std::vector<std::vector<Block *>> findAllPaths(Block *start, Block *end) {
  std::vector<std::vector<Block *>> allPaths;
  std::vector<Block *> path;
  std::unordered_set<Block *> visited;
  dfsAllPaths(start, end, path, visited, allPaths);
  return allPaths;
}

// Two types of hazards between the predecessors of one LSQ node:
// (1) WAW between 2 Store operations,
// (2) RAW and WAR between Load and Store operations
void HandshakeLowering::identifyMemDeps(
    std::vector<Operation *> &operations,
    std::vector<ProdConsMemDep> &allMemDeps) {
  for (Operation *i : operations) {
    // i: loop over the predecessors of the lsq_enode.. Skip those that are
    // not memory operations (i.e., not load and not store)
    if (!checkMemOp(i)) {
      continue;
    }

    for (Operation *j : operations) {
      // j: loop over every other predecessor of the lsq_enode.. Skip (1)
      // those that are not memory operations, (2) those in the same BB as
      // the one currently in hand, (3) both preds are load if(BB_i > BB_j), (4)
      // those that are mutually exclusive

      if (!checkMemOp(j) || checkSameBB(i, j) || checkBothLd(i, j) ||
          (findAllPaths(i->getBlock(), j->getBlock()).empty() &&
           findAllPaths(j->getBlock(), i->getBlock()).empty()))
        continue;

      // Comparing BB_i and BB_j
      Block *bbI = i->getBlock();
      Block *bbJ = j->getBlock();

      if (lessThanBlocks(bbJ, bbI)) {
        // prod = bbI, cons = bbJ;
        ProdConsMemDep oneMemDep(bbJ, bbI, false);
        allMemDeps.push_back(oneMemDep);

        if (sameLoop(bbI, bbJ)) {
          ProdConsMemDep opp(bbI, bbJ, true);
          allMemDeps.push_back(opp);
        }
      }
    }
  }
}

void HandshakeLowering::constructGroupsGraph(
    std::vector<Operation *> &operations,
    std::vector<ProdConsMemDep> &allMemDeps,
    std::set<Group *, GroupsComparator> &groups) {
  //  loop over the preds of the LSQ that are memory operations,
  //  create a Group object for each of them with the BB of the operation
  for (Operation *op : operations) {
    if (checkMemOp(op)) {
      Block *b = op->getBlock();
      Group *g = new Group(b);
      auto it = llvm::find_if(groups, [b](Group *g) { return g->bb == b; });
      if (it == groups.end()) {
        groups.insert(g);
      }
    }
  }

  /// After creating all of the Group objects, it is the time to connet
  /// their preds andProdConsMemDep succs Each entry in allMemDeps should
  /// represent an edge in the graph of groups
  for (ProdConsMemDep memDep : allMemDeps) {
    // Find group correspondig to the producer block
    auto itProd = llvm::find_if(groups, [&memDep](const Group *group) {
      return group->bb == memDep.prodBb;
    });

    Group *prodGroup = *itProd;

    // Find group correspondig to the consumer block
    auto itCons = llvm::find_if(groups, [&memDep](const Group *group) {
      return group->bb == memDep.consBb;
    });

    Group *consGroup = *itCons;

    // create edges to link the groups
    prodGroup->succs.insert(consGroup);
    consGroup->preds.insert(prodGroup);
  }
}

void HandshakeLowering::minimizeGroupsConnections(
    std::set<Group *, GroupsComparator> &groups) {
  /// Get the dominance info for the region
  DominanceInfo domInfo;

  /// For every group, compare every 2 of its preds, Cut the edge only if
  /// the pred with the bigger idx dominates your group
  for (auto group = groups.rbegin(); group != groups.rend(); ++group) {
    std::set<Group *> predsToRemove;

    for (auto bigPred = (*group)->preds.rbegin();
         bigPred != (*group)->preds.rend(); ++bigPred) {
      if (llvm::find(predsToRemove, *bigPred) != predsToRemove.end())
        continue;
      for (auto smallPred = (*group)->preds.rbegin();
           smallPred != (*group)->preds.rend(); ++smallPred) {
        if (llvm::find(predsToRemove, *smallPred) != predsToRemove.end())
          continue;
        if ((*bigPred != *smallPred) &&
            ((*bigPred)->preds.find(*smallPred) != (*bigPred)->preds.end()) &&
            domInfo.properlyDominates((*bigPred)->bb, (*group)->bb)) {
          predsToRemove.insert(*smallPred);
        }
      }
    }

    for (auto *pred : predsToRemove) {
      (*group)->preds.erase(pred);
      pred->succs.erase(*group);
    }
  }
}
// DFS to return all nodes in the path between the start_node and end_node
// (not including start_node and end_node) in the postDom tree
void traversePostDomTreeUtil(
    DominanceInfoNode *startNode, DominanceInfoNode *endNode,
    llvm::DenseMap<DominanceInfoNode *, bool> isVisited,
    llvm::SmallVector<mlir::DominanceInfoNode *, 4> path, int pathIndex,
    llvm::SmallVector<llvm::SmallVector<mlir::DominanceInfoNode *, 4>, 4>
        *traversedNodes) {
  isVisited[startNode] = true;
  path[pathIndex] = startNode;
  pathIndex++;

  // if start is same as end, we have completed one path so push it to
  // traversed_nodes
  if (startNode == endNode) {
    // slice of the path from its beginning until the path_index
    llvm::SmallVector<mlir::DominanceInfoNode *, 4> actualPath;
    for (auto i = 0; i < pathIndex; i++) {
      actualPath.push_back(path[i]);
    }
    traversedNodes->push_back(actualPath);

  } else {
    // loop over the children of start_node
    for (DominanceInfoNode::iterator iter = startNode->begin();
         iter < startNode->end(); iter++) {
      if (!isVisited[*iter]) {
        traversePostDomTreeUtil(*iter, endNode, isVisited, path, pathIndex,
                                traversedNodes);
      }
    }
  }

  // remove this node from path and mark it as unvisited
  pathIndex--;
  isVisited[startNode] = false;
}

// Gets the condition name of a block (^bb0 -> c0)
std::string getBlockCondition(Block *block) {
  std::string result;
  llvm::raw_string_ostream os(result);
  block->printAsOperand(os);
  std::string blockName = os.str();
  std::string blockCondition = "c" + blockName.substr(3);
  return blockCondition;
}

/// Converts a path to a product of conditions while checking if all the blocks
/// at the path are in the control dependency
BoolExpression *pathToMinterm(const std::vector<Block *> &path,
                              const SmallVector<Block *, 4> &controlDeps) {
  BoolExpression *exp = BoolExpression::parseSop("1");
  for (unsigned i = 0; i < path.size() - 1; i++) {
    Block *prod = path.at(i);
    auto *it = llvm::find(controlDeps, prod);
    if (it != controlDeps.end()) {
      Block *cons = path.at(i + 1);
      Operation *producerTerminator = prod->getTerminator();
      BoolExpression *prodCondition =
          BoolExpression::parseSop(getBlockCondition(prod));
      if (isa<cf::CondBranchOp>(producerTerminator)) {
        auto condOp = dyn_cast<cf::CondBranchOp>(producerTerminator);
        // If the following BB is on the FALSE side of the current BB, then
        // negate the condition of the current BB
        if (cons == condOp.getFalseDest())
          prodCondition = prodCondition->boolNegate();
      }
      exp = BoolExpression::boolAnd(exp, prodCondition);
    }
  }
  return exp;
}

// Function to eliminate common dependencies between the producer and the
// consumer
void eliminateCommonEntries(SmallVector<Block *, 4> &prodControlDeps,
                            SmallVector<Block *, 4> &consControlDeps) {

  std::vector<Block *> inter;
  set_intersection(prodControlDeps.begin(), prodControlDeps.end(),
                   consControlDeps.begin(), consControlDeps.end(),
                   back_inserter(inter));

  prodControlDeps.erase(set_difference(prodControlDeps.begin(),
                                       prodControlDeps.end(), inter.begin(),
                                       inter.end(), prodControlDeps.begin()),
                        prodControlDeps.end());

  consControlDeps.erase(set_difference(consControlDeps.begin(),
                                       consControlDeps.end(), inter.begin(),
                                       inter.end(), consControlDeps.begin()),
                        consControlDeps.end());
}

BoolExpression *
HandshakeLowering::enumeratePaths(Block *start, Block *end,
                                  const SmallVector<Block *, 4> &controlDeps) {
  BoolExpression *sop = BoolExpression::boolZero();
  std::vector<std::vector<Block *>> allPaths = findAllPaths(start, end);
  for (const std::vector<Block *> &path : allPaths) {
    BoolExpression *minterm = pathToMinterm(path, controlDeps);
    sop = BoolExpression::boolOr(sop, minterm);
  }
  return sop->boolMinimizeSop();
}

Block *getPostDominantSuccessor(Block *prod, Block *cons,
                                std::unordered_set<Block *> &visited,
                                PostDominanceInfo &postDomInfo) {
  if (!prod)
    return nullptr;
  visited.insert(prod);

  for (Block *successor : prod->getSuccessors()) {
    // Check if the successor post-dominates cons
    if (successor != cons && postDomInfo.postDominates(successor, cons))
      return successor;

    // If not visited, recursively search successors of the current successor
    if (visited.find(successor) == visited.end()) {
      Block *result =
          getPostDominantSuccessor(successor, cons, visited, postDomInfo);
      if (result)
        return result;
    }
  }
  return nullptr;
}

// Gets the first successor of the producer that post-dominates the consumer
Block *getPostDominantSuccessor(Block *prod, Block *cons) {
  std::unordered_set<Block *> visited;
  PostDominanceInfo postDomInfo;
  return getPostDominantSuccessor(prod, cons, visited, postDomInfo);
}

Block *getPredecessorDominatingAndPostDominating(
    Block *prod, Block *cons, std::unordered_set<Block *> &visited,
    DominanceInfo &domInfo, PostDominanceInfo &postDomInfo) {
  if (!cons)
    return nullptr;
  visited.insert(cons);

  for (Block *predecessor : cons->getPredecessors()) {
    // Check if the predecessor post-dominates prod and dominates cons
    if (predecessor != prod && postDomInfo.postDominates(predecessor, prod) &&
        domInfo.dominates(predecessor, cons))
      return predecessor;

    // If not visited, recursively search predecessors of the current
    // predecessor
    if (visited.find(predecessor) == visited.end()) {
      Block *result = getPredecessorDominatingAndPostDominating(
          prod, predecessor, visited, domInfo, postDomInfo);
      if (result)
        return result;
    }
  }
  return nullptr;
}

// Gets the first predecessor of the consumer that post-dominates the producer
// and dominates the consumer
Block *getPredecessorDominatingAndPostDominating(Block *prod, Block *cons) {
  std::unordered_set<Block *> visited;
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
  return getPredecessorDominatingAndPostDominating(prod, cons, visited, domInfo,
                                                   postDomInfo);
}

LogicalResult
HandshakeLowering::addMergeNonLoop(OpBuilder &builder,
                                   std::vector<ProdConsMemDep> &allMemDeps,
                                   std::set<Group *, GroupsComparator> &groups,
                                   DenseMap<Block *, Operation *> &forksGraph) {
  Block *entryBlock = &region.front();
  for (Group *prodGroup : groups) {
    Block *prod = prodGroup->bb;
    SmallVector<Block *, 4> prodControlDeps;
    cdgAnalysis.calculateBlockForwardControlDeps(prod, funcOpIdx,
                                                 prodControlDeps);
    for (Group *consGroup : prodGroup->succs) {
      Block *cons = consGroup->bb;
      SmallVector<Block *, 4> consControlDeps;
      cdgAnalysis.calculateBlockForwardControlDeps(cons, funcOpIdx,
                                                   consControlDeps);
      eliminateCommonEntries(prodControlDeps, consControlDeps);

      BoolExpression *fProd = enumeratePaths(entryBlock, prod, prodControlDeps);

      BoolExpression *fCons = enumeratePaths(entryBlock, cons, consControlDeps);
      BoolExpression *fGen =
          BoolExpression::boolAnd(fCons, fProd->boolNegate());
      fGen = fGen->boolMinimize();

      if (fGen->type != experimental::boolean::ExpressionType::Zero) {
        auto memDepIt =
            llvm::find_if(allMemDeps, [prod, cons](const ProdConsMemDep &dep) {
              return dep.prodBb == prod && dep.consBb == cons;
            });
        if (memDepIt == allMemDeps.end())
          return failure();
        ProdConsMemDep &memDep = *memDepIt;

        builder.setInsertionPointToStart(cons);
        Location loc = forksGraph[cons]->getLoc();

        if (memDep.isBackward) {
          builder.setInsertionPointToStart(
              getPostDominantSuccessor(prod, cons));
          loc = getPostDominantSuccessor(prod, cons)
                    ->getOperations()
                    .front()
                    .getLoc();
        } else {
          Block *block = getPredecessorDominatingAndPostDominating(prod, cons);
          if (block) {
            builder.setInsertionPointToStart(block);
            loc = block->getOperations().front().getLoc();
          }
        }

        SmallVector<Value> mergeOperands;
        mergeOperands.push_back(startCtrl);
        mergeOperands.push_back(forksGraph[prod]->getResult(0));
        auto mergeOp = builder.create<handshake::MergeOp>(loc, mergeOperands);
        allocationNetwork.push_back(mergeOp);

        /// The merge becomes the producer now, so connect the result of the
        /// MERGE as an operand of the Consumer
        // Remove the old connection between the producer's LazyFork and the
        // consumer's LazyFork
        // Connect the MERGE to the consumer's LazyFork
        forksGraph[cons]->replaceUsesOfWith(forksGraph[prod]->getResult(0),
                                            mergeOp->getResult(0));
      }
    }
  }
  return success();
}

bool mergeFeedingCurrentConsumer(Operation *merge, Operation *consumer) {
  return llvm::any_of(
      merge->getResult(0).getUses(),
      [consumer](OpOperand &use) { return use.getOwner() == consumer; });
}

// Helper function to perform DFS
void dfs(Operation *current, Operation *end,
         std::unordered_set<Operation *> &visited,
         std::vector<Operation *> &path,
         std::vector<std::vector<Operation *>> &allPaths) {
  visited.insert(current);
  path.push_back(current);

  if (current == end) {
    // If the current operation is the end, add the path to allPaths
    allPaths.push_back(path);
  } else {
    // Otherwise, explore the successors
    for (auto result : current->getResults()) {
      for (auto *successor : result.getUsers()) {
        if (visited.find(successor) == visited.end()) {
          dfs(successor, end, visited, path, allPaths);
        }
      }
    }
  }

  // Backtrack
  path.pop_back();
  visited.erase(current);
}

// Finds all paths from start operation to end operation
std::vector<std::vector<Operation *>> findAllPaths(Operation *start,
                                                   Operation *end) {
  std::vector<std::vector<Operation *>> allPaths;
  std::unordered_set<Operation *> visited;
  std::vector<Operation *> path;
  dfs(start, end, visited, path, allPaths);
  return allPaths;
}

LogicalResult
HandshakeLowering::addMergeLoop(OpBuilder &builder,
                                std::set<Group *, GroupsComparator> &groups,
                                DenseMap<Block *, Operation *> &forksGraph) {
  for (Group *consGroup : groups) {
    Block *cons = consGroup->bb;
    for (Group *prodGroup : consGroup->preds) {
      Block *prod = prodGroup->bb;
      /// For every loop containing both prod and cons, insert a MERGE in
      /// the loop header block
      if (greaterThanBlocks(prod, cons)) {
        CFGLoop *loop = getInnermostCommonLoop(prod, cons);
        if (loop) {
          /// Insert the MERGE at the beginning of the loop header with
          /// START and the result of the producer as operands
          Block *loopHeader = loop->getHeader();
          builder.setInsertionPointToStart(loopHeader);

          /// For each path from the producer to the consumer, add a MERGE
          /// right before the consumer
          std::vector<std::vector<Operation *>> allPaths =
              findAllPaths(forksGraph[prod], forksGraph[cons]);
          for (std::vector<Operation *> path : allPaths) {
            SmallVector<Value> operands;
            /// Get the result of the Operation right before the consumer
            Value mergeOperand = path.at(path.size() - 2)->getResult(0);
            operands.push_back(startCtrl);
            operands.push_back(mergeOperand);
            auto mergeOp = builder.create<handshake::MergeOp>(
                mergeOperand.getLoc(), operands);
            allocationNetwork.push_back(mergeOp);
            memDepLoopMerges.push_back(mergeOp);

            /// The merge becomes the producer now, so connect the result of
            /// the MERGE as an operand of the Consumer
            // Remove the old connection between the producer's LazyFork and
            // the consumer's LazyFork Connect the MERGE to the consumer's
            // LazyFork
            forksGraph[cons]->replaceUsesOfWith(mergeOperand,
                                                mergeOp->getResult(0));
          }
        }
      }
    }
  }
  return success();
}

LogicalResult
HandshakeLowering::joinInsertion(OpBuilder &builder,
                                 std::set<Group *, GroupsComparator> &groups,
                                 DenseMap<Block *, Operation *> &forksGraph) {
  for (Group *group : groups) {
    Operation *forkNode = forksGraph[group->bb];
    ValueRange operands = forkNode->getOperands();
    if (operands.size() > 1) {
      /// Join all the results of the predecessors of the LazyFork
      builder.setInsertionPointToStart(forkNode->getBlock());
      auto joinOp =
          builder.create<handshake::JoinOp>(forkNode->getLoc(), operands);
      allocationNetwork.push_back(joinOp);
      /// The result of the JoinOp becomes the input to the LazyFork
      forkNode->setOperands(joinOp.getResult());
    }
  }
  return success();
}

///----------Fast Token Delivery Algorithm----------

SmallVector<CFGLoop *> HandshakeLowering::getLoopsConsNotInProd(Block *cons,
                                                                Block *prod) {

  SmallVector<CFGLoop *> result;

  // Traverse the loops of cons from innermost to outermost
  CFGLoop *loop = li.getLoopFor(cons);
  while (loop) {
    if (!loop->contains(prod)) {
      result.push_back(loop);
    }
    loop = loop->getParentLoop();
  }

  // Reverse to the get the loops from outermost to innermost
  std::reverse(result.begin(), result.end());
  return result;
}

LogicalResult HandshakeLowering::addPhi(ConversionPatternRewriter &rewriter) {
  for (Block &cons : region.getBlocks()) {
    for (Operation &consOp : cons.getOperations()) {
      if (llvm::find(allocationNetwork, &consOp) == allocationNetwork.end() &&
          (!isa<handshake::ConstantOp>(consOp) ||
           llvm::find(networkConstants, &consOp) != networkConstants.end()))
        continue;
      if (llvm::find(allocationNetwork, &consOp) == allocationNetwork.end() &&
          (!isa<handshake::ConstantOp>(consOp) ||
           llvm::find(networkConstants, &consOp) != networkConstants.end()))
        continue;
      /// If the current consumer is a MERGE that was added in addPhi, then
      /// skip it to avoid an infinite loop
      if (llvm::find(phiMerges, &consOp) != phiMerges.end())
        continue;
      for (Value operand : consOp.getOperands()) {
        if (mlir::Operation *prodOp = operand.getDefiningOp()) {
          /// If the operand is a result of a producer operation (not
          /// start), then check if the producer is a MERGE that was added
          /// in addPhi If that is the case, then skip it to avoid an
          /// infinite loop
          if (llvm::find(phiMerges, prodOp) != phiMerges.end())
            continue;
        }
        Block *prod = operand.getParentBlock();
        Value input = operand;
        SmallVector<Operation *> merges;
        SmallVector<CFGLoop *> loops = getLoopsConsNotInProd(&cons, prod);
        for (auto *it = loops.begin(); it != loops.end(); ++it) {
          // If we are at the innermost loop and the consumer is a loop
          // merge, stop
          if (std::next(it) == loops.end() &&
              llvm::find(memDepLoopMerges, &consOp) != memDepLoopMerges.end())
            break;
          rewriter.setInsertionPointToStart((*it)->getHeader());
          auto mergeOp =
              rewriter.create<handshake::MergeOp>(input.getLoc(), input);
          input = mergeOp.getResult();
          merges.push_back(mergeOp);
        }
        consOp.replaceUsesOfWith(operand, input);
        for (auto *mergeOp : merges)
          mergeOp->insertOperands(1, mergeOp->getResult(0));

        phiMerges.insert(phiMerges.end(), merges.begin(), merges.end());
      }
    }
  }
  allocationNetwork.insert(allocationNetwork.end(), phiMerges.begin(),
                           phiMerges.end());
  return success();
}

/// Check whether 2 merges are being fed by teh same producer
bool sameProducer(Operation *merge1, Operation *merge2) {
  if (merge1->getOperand(0).getDefiningOp() &&
      merge2->getOperand(0).getDefiningOp())
    return merge1->getOperand(0).getDefiningOp() ==
           merge2->getOperand(0).getDefiningOp();

  return false;
}

LogicalResult
HandshakeLowering::removeRedundantPhis(ConversionPatternRewriter &rewriter) {
  for (Operation *merge : phiMerges) {
    for (Operation &op : merge->getBlock()->getOperations()) {
      // if the other operation is a Merge and is having the same producer
      if (&op != merge && isa<handshake::MergeOp>(op) &&
          sameProducer(merge, &op)) {
        auto it = llvm::find(allocationNetwork, &op);
        if (it != allocationNetwork.end())
          allocationNetwork.erase(it);
        rewriter.replaceOp(&op, merge);
      }
    }
  }
  return success();
}

bool HandshakeLowering::isaMergeLoop(Operation *merge) {
  return li.getLoopFor(merge->getOperand(0).getParentBlock()) !=
         li.getLoopFor(merge->getOperand(1).getParentBlock());
}

LogicalResult HandshakeLowering::addSupp(ConversionPatternRewriter &rewriter) {
  fillConditionToValueMapping();

  for (Block &prod : region.getBlocks()) {
    for (Operation &prodOp : prod.getOperations()) {
      if (llvm::find(allocationNetwork, &prodOp) == allocationNetwork.end() &&
          (!isa<handshake::ConstantOp>(prodOp) ||
           llvm::find(networkConstants, &prodOp) != networkConstants.end()))
        continue;

      /// (2) If the current producer is a MUX added by Shannon, then skip
      /// it
      if (llvm::find(shannonMUXes, &prodOp) != shannonMUXes.end())
        continue;

      if (llvm::find(suppBranches, &prodOp) != suppBranches.end())
        continue;

      if (llvm::find(selfGenBranches, &prodOp) != selfGenBranches.end())
        continue;

      for (Value res : prodOp.getResults()) {
        res.getUsers();
        std::vector<Operation *> users(res.getUsers().begin(),
                                       res.getUsers().end());
        for (Operation *consOp : users) {
          Block *cons = consOp->getBlock();

          if (llvm::find(allocationNetwork, consOp) ==
                  allocationNetwork.end() &&
              (!isa<handshake::ConstantOp>(consOp) ||
               llvm::find(networkConstants, consOp) != networkConstants.end()))
            continue;

          /// (1) Skip if the consumer and the producer are in the same block
          /// and the consumer is not a Merge
          if (consOp->getBlock() == &prod && !isa<handshake::MergeOp>(consOp))
            continue;

          /// (2) If the current consumer is a MUX added by Shannon, then skip
          /// it
          if (llvm::find(shannonMUXes, consOp) != shannonMUXes.end())
            continue;

          ///(3) Skip if the consumer is a Branch
          if (isa<handshake::ConditionalBranchOp>(consOp))
            continue;

          llvm::errs() << "Prod: " << prodOp << "\n";
          llvm::errs() << "Cons: " << *consOp << "\n";

          /// Innermost loop containig the producer doesn't contain the
          /// consumer
          bool moreProdThanCons = false;
          if (li.getLoopFor(&prod))
            moreProdThanCons = !li.getLoopFor(&prod)->contains(cons);

          // and prod is not a branch in loopExit
          if (moreProdThanCons && !isBranchInLoopExit(&prodOp)) {
            manageMoreProdThanCons(rewriter, &prod, consOp, res);
          } else {
            bool selfRegeneration =
                llvm::any_of(consOp->getResults(),
                             [&res](const Value &v) { return v == res; });

            if (selfRegeneration)
              manageSelfRegeneration(rewriter, consOp, res);
            else if (greaterThanBlocks(&prod, cons) ||
                     (isa<handshake::MergeOp>(consOp) && &prod == cons &&
                      isaMergeLoop(consOp) &&
                      !isa<handshake::ConditionalBranchOp>(prodOp)))
              manageDifferentRegeneration(rewriter, consOp, res);
            else {
              llvm::errs() << "Calling manageNonLoop";
              manageNonLoop(rewriter, &prod, consOp, res);
            }
          }
        }
      }
    }
  }
  return success();
}

LogicalResult
HandshakeLowering::addSuppBranches(ConversionPatternRewriter &rewriter,
                                   std::set<Operation *> &oldBranches,
                                   size_t &count) {
  std::vector<Operation *> suppBranchesCopy = suppBranches;

  for (Operation *prodOp : suppBranchesCopy) {
    if (llvm::find(oldBranches, prodOp) != oldBranches.end())
      continue;

    Block *prod = prodOp->getBlock();

    for (Value res : prodOp->getResults()) {
      std::vector<Operation *> users(res.getUsers().begin(),
                                     res.getUsers().end());

      for (Operation *consOp : users) {

        Block *cons = consOp->getBlock();

        if (llvm::find(allocationNetwork, consOp) == allocationNetwork.end() &&
            (!isa<handshake::ConstantOp>(consOp) ||
             llvm::find(networkConstants, consOp) != networkConstants.end()))
          continue;

        /// (1) Skip if the consumer and the producer are in the same
        /// block and the consumer is not a Merge
        if (consOp->getBlock() == prod && !isa<handshake::MergeOp>(consOp))
          continue;

        /// (2) If the current consumer is a MUX added by Shannon, then
        /// skip it
        if (llvm::find(shannonMUXes, consOp) != shannonMUXes.end())
          continue;

        ///(3) Skip if the consumer is a Branch
        if (isa<handshake::ConditionalBranchOp>(consOp))
          continue;

        /// Innermost loop containig the producer doesn't contain the
        /// consumer
        bool moreProdThanCons = false;
        if (li.getLoopFor(prod))
          moreProdThanCons = !li.getLoopFor(prod)->contains(cons);

        // and prod is not a branch in loopExit
        if (moreProdThanCons && !isBranchInLoopExit(prodOp))
          manageMoreProdThanCons(rewriter, prod, consOp, res);
        else if (greaterThanBlocks(prod, cons) ||
                 (isa<handshake::MergeOp>(consOp) && prod == cons &&
                  isaMergeLoop(consOp)))
          manageDifferentRegeneration(rewriter, consOp, res);
        else
          manageNonLoop(rewriter, prod, consOp, res);
      }
    }
  }
  std::copy(suppBranchesCopy.begin(), suppBranchesCopy.end(),
            std::inserter(oldBranches, oldBranches.end()));

  count = suppBranches.size();

  return success();
}

LogicalResult
HandshakeLowering::addSuppForStart(ConversionPatternRewriter &rewriter) {
  for (Block &cons : region.getBlocks()) {
    for (Operation &consOp : cons.getOperations()) {
      if (llvm::find(allocationNetwork, &consOp) == allocationNetwork.end() &&
          (!isa<handshake::ConstantOp>(consOp) ||
           llvm::find(networkConstants, &consOp) != networkConstants.end()))
        continue;

      /// (2) If the current consumer is a MUX added by Shannon, then skip it
      if (llvm::find(shannonMUXes, &consOp) != shannonMUXes.end())
        continue;

      ///(3) Skip if the consumer is a Branch
      if (isa<handshake::ConditionalBranchOp>(consOp))
        continue;
      for (Value operand : consOp.getOperands()) {
        if (operand == startCtrl) {
          /// (1) Skip if the consumer and the producer are in the same block
          /// and the consumer is not a Merge
          if (operand.getParentBlock() == &cons &&
              !isa<handshake::MergeOp>(consOp))
            continue;

          Block *prod = operand.getParentBlock();

          manageNonLoop(rewriter, prod, &consOp, operand);
        }
      }
    }
  }
  return success();
}

BoolExpression *HandshakeLowering::getBlockLoopExitCondition(Block *loopExit,
                                                             CFGLoop *loop) {
  BoolExpression *blockCond =
      BoolExpression::parseSop(getBlockCondition(loopExit));
  auto *terminatorOperation = loopExit->getTerminator();
  assert(isa<cf::CondBranchOp>(terminatorOperation) &&
         "Terminator condition of a loop exit must be a conditional branch.");
  auto condBranch = dyn_cast<cf::CondBranchOp>(terminatorOperation);
  Block *falseDest = condBranch.getFalseDest();

  /// If the termiator branch has false successor outside the loop, negate
  if (li.getLoopFor(falseDest) != loop)
    blockCond->boolNegate();
  return blockCond;
}

//----------BooleanExpression to Circuit----------
void HandshakeLowering::fillConditionToValueMapping() {
  for (Block &block : region.getBlocks()) {
    Operation *terminator = block.getTerminator();
    if (terminator) {
      if (isa<cf::CondBranchOp>(terminator)) {
        auto condBranch = dyn_cast<cf::CondBranchOp>(terminator);
        conditionToValue[getBlockCondition(&block)] = condBranch.getCondition();
      }
    }
  }
}

Value HandshakeLowering::dataToCircuit(ConversionPatternRewriter &rewriter,
                                       Data *data, Block *block) {
  if (data->boolexpression.has_value())
    return boolExpressionToCircuit(rewriter, data->boolexpression.value(),
                                   block);

  return muxToCircuit(rewriter, data->mux, block);
}

Value HandshakeLowering::muxToCircuit(ConversionPatternRewriter &rewriter,
                                      MUX *mux, Block *block) {
  rewriter.setInsertionPointToStart(block);

  /// Recursive calls to the inputs and the condition of the MUX
  SmallVector<Value, 4> muxOperands;
  muxOperands.push_back(dataToCircuit(rewriter, mux->in0, block));
  muxOperands.push_back(dataToCircuit(rewriter, mux->in1, block));
  Value muxCond = dataToCircuit(rewriter, mux->cond, block);

  /// Create the MUX
  auto muxOp = rewriter.create<handshake::MuxOp>(
      block->getOperations().front().getLoc(), muxCond, muxOperands);
  shannonMUXes.push_back(muxOp);
  allocationNetwork.push_back(muxOp);
  return muxOp.getResult();
}

Value HandshakeLowering::boolExpressionToCircuit(
    ConversionPatternRewriter &rewriter, BoolExpression *expr, Block *block) {
  if (expr->type == ExpressionType::Variable)
    return boolVariableToCircuit(rewriter, expr, block);

  if (expr->type == ExpressionType::One) {
    rewriter.setInsertionPointToStart(block);
    auto sourceOp = rewriter.create<handshake::SourceOp>(
        block->getOperations().front().getLoc(), rewriter.getNoneType());
    Value cnstTrigger = sourceOp.getResult();
    TypedAttr constantAttr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(1), 1);

    auto constOp = rewriter.create<handshake::ConstantOp>(
        block->getOperations().front().getLoc(), constantAttr.getType(),
        constantAttr, cnstTrigger);
    networkConstants.push_back(constOp);
    return constOp.getResult();
  }
  rewriter.setInsertionPointToStart(block);
  auto sourceOp = rewriter.create<handshake::SourceOp>(
      block->getOperations().front().getLoc(), rewriter.getNoneType());
  Value cnstTrigger = sourceOp.getResult();
  TypedAttr constantAttr =
      rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0);
  auto constOp = rewriter.create<handshake::ConstantOp>(
      block->getOperations().front().getLoc(), constantAttr.getType(),
      constantAttr, cnstTrigger);
  networkConstants.push_back(constOp);
  return constOp.getResult();
}

Value HandshakeLowering::boolVariableToCircuit(
    ConversionPatternRewriter &rewriter,
    experimental::boolean::BoolExpression *expr, Block *block) {
  SingleCond *singleCond = static_cast<SingleCond *>(expr);
  if (singleCond->isNegated) {
    rewriter.setInsertionPointToStart(block);
    auto notOp = rewriter.create<handshake::NotOp>(
        block->getOperations().front().getLoc(),
        conditionToValue[singleCond->id]);
    return notOp->getResult(0);
  }
  return conditionToValue[singleCond->id];
}

Value HandshakeLowering::insertBranchToLoop(ConversionPatternRewriter &rewriter,
                                            CFGLoop *loop, Operation *consumer,
                                            Value connection,
                                            bool moreProdThanCons,
                                            bool selfRegeneration) {
  Block *loopExit = loop->getExitingBlock();
  if (loopExit) {
    Operation *loopTerminator = loopExit->getTerminator();
    assert(isa<cf::CondBranchOp>(loopTerminator) &&
           "Terminator condition of a loop exit must be a conditional "
           "branch.");

    rewriter.setInsertionPointToStart(loopExit);
    auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        loopExit->getOperations().back().getLoc(),
        boolVariableToCircuit(
            rewriter, getBlockLoopExitCondition(loopExit, loop), loopExit),
        connection);
    allocationNetwork.push_back(branchOp);
    if (moreProdThanCons)
      suppBranches.push_back(branchOp);
    if (selfRegeneration)
      selfGenBranches.push_back(branchOp);

    Value newConnection = branchOp.getTrueResult();
    if (!moreProdThanCons)
      newConnection = branchOp.getFalseResult();
    consumer->replaceUsesOfWith(connection, newConnection);
    return newConnection;
  }
  std::vector<std::string> cofactorList;
  SmallVector<Block *> exitBlocks;
  loop->getExitingBlocks(exitBlocks);
  loopExit = exitBlocks.front();
  BoolExpression *fLoopExit = BoolExpression::boolZero();

  for (Block *exitBlock : exitBlocks) {
    BoolExpression *blockCond = getBlockLoopExitCondition(exitBlock, loop);
    fLoopExit = BoolExpression::boolOr(fLoopExit, blockCond);
    cofactorList.push_back(getBlockCondition(exitBlock));
  }

  std::sort(cofactorList.begin(), cofactorList.end());
  Data *shannonResult = applyShannon(fLoopExit, cofactorList);

  /// Convert the result of shannon from BoolExpression to circuitry
  Operation *loopTerminator = loopExit->getTerminator();
  assert(isa<cf::CondBranchOp>(loopTerminator) &&
         "Terminator condition of a loop exit must be a conditional "
         "branch.");
  Value branchCond = dataToCircuit(rewriter, shannonResult, loopExit);

  rewriter.setInsertionPointToStart(loopExit);
  auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
      loopExit->getOperations().front().getLoc(), branchCond, connection);
  allocationNetwork.push_back(branchOp);

  if (moreProdThanCons)
    suppBranches.push_back(branchOp);
  if (selfRegeneration)
    selfGenBranches.push_back(branchOp);

  /// moreProdThanCons flag is used to discern between the caller of the
  /// function (manageMoreProdThanCons or manageSelfRegeneration /
  /// manageDifferentRegeneration)
  /// It is used to connect either the:
  /// (1) manageSelfRegeneration / manageDifferentRegeneration: True side
  /// of the BRANCH to sink and False side to the consumer (2)
  /// manageMoreProdThanCons: True side of the BRANCH to the consumer and
  /// False side to sink
  Value newConnection = branchOp.getTrueResult();
  if (!moreProdThanCons)
    newConnection = branchOp.getFalseResult();
  consumer->replaceUsesOfWith(connection, newConnection);
  return newConnection;
}

void HandshakeLowering::manageMoreProdThanCons(
    ConversionPatternRewriter &rewriter, Block *producerBlock,
    Operation *consumer, Value connection) {
  /// Identify every loop containing the producer and not the consumer
  Value con = connection;
  // std::set<CFGLoop *> producerLoopsWithoutConsumer;
  CFGLoop *loop = li.getLoopFor(producerBlock);
  while (loop) {
    if (!loop->contains(consumer->getBlock()))
      con = insertBranchToLoop(rewriter, loop, consumer, con, true, false);

    loop = loop->getParentLoop();
  }
}

void HandshakeLowering::manageSelfRegeneration(
    ConversionPatternRewriter &rewriter, Operation *consumer,
    Value connection) {
  /// Identify every loop containing the producer is in
  CFGLoop *loop = li.getLoopFor(consumer->getBlock());
  if (loop)
    insertBranchToLoop(rewriter, loop, consumer, connection, false, true);
}

void HandshakeLowering::manageNonLoop(ConversionPatternRewriter &rewriter,
                                      Block *producerBlock, Operation *consumer,
                                      Value connection) {
  Block *entryBlock = &region.front();
  SmallVector<Block *, 4> prodControlDeps;
  cdgAnalysis.calculateBlockForwardControlDeps(producerBlock, funcOpIdx,
                                               prodControlDeps);

  llvm::errs() << "prodControlDeps\n";
  for (Block *block : prodControlDeps) {
    block->printAsOperand(llvm::errs());
    llvm::errs() << "\n";
  }

  SmallVector<Block *, 4> consControlDeps;
  cdgAnalysis.calculateBlockForwardControlDeps(consumer->getBlock(), funcOpIdx,
                                               consControlDeps);

  llvm::errs() << "consControlDeps\n";
  for (Block *block : consControlDeps) {
    block->printAsOperand(llvm::errs());
    llvm::errs() << "\n";
  }

  eliminateCommonEntries(prodControlDeps, consControlDeps);

  BoolExpression *fProd =
      enumeratePaths(entryBlock, producerBlock, prodControlDeps);
  BoolExpression *fCons =
      enumeratePaths(entryBlock, consumer->getBlock(), consControlDeps);

  /// f_supp = f_prod ano not f_cons
  BoolExpression *fSup = BoolExpression::boolAnd(fProd, fCons->boolNegate());
  fSup = fSup->boolMinimize();

  if (fSup->type != experimental::boolean::ExpressionType::Zero) {
    std::set<std::string> blocks = fSup->getVariables();

    std::vector<std::string> cofactorList(blocks.begin(), blocks.end());
    Data *shannonResult = applyShannon(fSup, cofactorList);
    Value branchCond =
        dataToCircuit(rewriter, shannonResult, consumer->getBlock());

    rewriter.setInsertionPointToStart(consumer->getBlock());
    auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
        consumer->getLoc(), branchCond, connection);
    allocationNetwork.push_back(branchOp);
    consumer->replaceUsesOfWith(connection, branchOp.getFalseResult());
  }
}

void HandshakeLowering::manageDifferentRegeneration(
    ConversionPatternRewriter &rewriter, Operation *consumer,
    Value connection) {
  // loop exit of cons
  if (li.getLoopFor(consumer->getBlock())) {
    SmallVector<Block *> exits;
    li.getLoopFor(consumer->getBlock())->getExitBlocks(exits);
    insertBranchToLoop(rewriter, li.getLoopFor(consumer->getBlock()), consumer,
                       connection, false, false);
  }
}

///----------Fast Token Delivery Cleanup----------
void HandshakeLowering::fixConvention(Operation *merge, CFGLoop *loop) {
  Value firstOperand = merge->getOperand(0);
  if (li.getLoopFor(firstOperand.getParentBlock()) == loop) {
    Value secondOperand = merge->getOperand(1);
    merge->setOperand(0, secondOperand);
    merge->setOperand(1, firstOperand);
  }
}

LogicalResult
HandshakeLowering::convertMergesToMuxes(ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *> initMerges;
  DominanceInfo domInfo;

  for (Block &block : region.getBlocks()) {
    for (Operation &merge : block.getOperations()) {
      if (llvm::find(allocationNetwork, &merge) == allocationNetwork.end())
        continue;
      /// If the current operation is a Merge but is not an INIT Merge
      if (isa<handshake::MergeOp>(merge) &&
          llvm::find(initMerges, &merge) == initMerges.end()) {

        bool loopHeader = false;
        if (li.getLoopFor(&block))
          loopHeader =
              (li.getLoopFor(&block)->getHeader() == &block &&
               li.getLoopFor(merge.getOperand(0).getParentBlock()) !=
                   li.getLoopFor(merge.getOperand(1).getParentBlock()));

        /// If the operation block is loop header && one input is outside the
        /// loop and the other input is inside the loop
        if (loopHeader) {
          fixConvention(&merge, li.getLoopFor(&block));
          Value select = addInit(rewriter, initMerges, &merge);

          // Convert to Mux with the calculated select
          rewriter.setInsertionPointAfter(&merge);
          auto mux = rewriter.create<handshake::MuxOp>(merge.getLoc(), select,
                                                       merge.getOperands());
          allocationNetwork.push_back(mux);
          rewriter.replaceOp(&merge, mux);

        } else {
          Value select =
              getBlockEntryControl(&block).getDefiningOp()->getResult(1);

          /// The first input of the CMerge and the first input of the Mux are
          /// in the same block
          bool firstSameBlock =
              select.getDefiningOp()->getOperand(0).getParentBlock() ==
              merge.getOperand(0).getParentBlock();

          /// The block of the Mux's first input dominates the block
          /// of the CMerge's first input
          bool firstMuxDominateMerge = domInfo.properlyDominates(
              merge.getOperand(0).getParentBlock(),
              select.getDefiningOp()->getOperand(0).getParentBlock());

          /// The second input of the CMerge and the second input of
          /// the Mux are in the same block
          bool secondSameBlock =
              select.getDefiningOp()->getOperand(1).getParentBlock() ==
              merge.getOperand(1).getParentBlock();

          /// The block of the Mux's second input dominates the block
          /// of the CMerge's second input
          bool secondMuxDominateMerge = domInfo.properlyDominates(
              merge.getOperand(1).getParentBlock(),
              select.getDefiningOp()->getOperand(1).getParentBlock());

          if (!((firstSameBlock || firstMuxDominateMerge) &&
                (secondSameBlock || secondMuxDominateMerge))) {
            rewriter.setInsertionPointAfterValue(select);
            auto notOp =
                rewriter.create<handshake::NotOp>(select.getLoc(), select);
            select = notOp->getResult(0);
          }

          // Convert to Mux with the calculated select
          rewriter.setInsertionPointAfter(&merge);
          auto mux = rewriter.create<handshake::MuxOp>(merge.getLoc(), select,
                                                       merge.getOperands());
          allocationNetwork.push_back(mux);

          rewriter.replaceOp(&merge, mux);
        }
      }
    }
  }
  return success();
}
bool HandshakeLowering::isBranchInLoopExit(Operation *op) {
  if (isa<handshake::ConditionalBranchOp>(op)) {
    CFGLoop *loop = li.getLoopFor(op->getBlock());
    if (loop) {
      llvm::SmallVector<Block *> exitBlocks;
      loop->getExitingBlocks(exitBlocks);
      return llvm::find(exitBlocks, op->getBlock()) != exitBlocks.end();
    }
  }
  return false;
}

Value HandshakeLowering::addInit(ConversionPatternRewriter &rewriter,
                                 SmallVector<Operation *> &initMerges,
                                 Operation *oldMerge) {
  SmallVector<Value, 4> mergeOperands;

  // assert that one of the inputs of the Merge that we are currently
  // translating to Mux is a Branch in a loop exit block
  bool inputIsBranchInLoopExit = false;
  Value mergeOperandFromMux;
  handshake::ConditionalBranchOp branchOp;
  for (Value operand : oldMerge->getOperands()) {
    if (Operation *producer = operand.getDefiningOp()) {
      if (isBranchInLoopExit(producer)) {
        branchOp = dyn_cast<handshake::ConditionalBranchOp>(producer);
        mergeOperandFromMux = branchOp.getConditionOperand();
        if (isa<handshake::NotOp>(mergeOperandFromMux.getDefiningOp()))
          mergeOperandFromMux =
              mergeOperandFromMux.getDefiningOp()->getOperand(0);
        inputIsBranchInLoopExit = true;
      }
    }
  }
  assert(inputIsBranchInLoopExit &&
         "An input of the Merge must be a Branch in a loop exit block");

  if (branchOp.getResult(0) == oldMerge->getOperand(1) ||
      branchOp.getResult(1) == oldMerge->getOperand(0)) {
    rewriter.setInsertionPointAfterValue(mergeOperandFromMux);
    auto notOp = rewriter.create<handshake::NotOp>(mergeOperandFromMux.getLoc(),
                                                   mergeOperandFromMux);
    mergeOperandFromMux = notOp->getResult(0);
  }

  mergeOperands.push_back(mergeOperandFromMux);

  /// Insert a new constant in the same block as that of the new Merge and
  /// feed
  // its input from START.

  /// Decide if the value of the constant should be 1 or 0 by, checking
  /// which input of the Merge that we are currently translating to Mux is
  /// coming from outside of the loop (1) If the first input of the Merge
  /// that we are currently translating to Mux is coming from outside of
  /// the loop, the value of the constant should be 0 (2) If the second
  /// input of the Merge that we are currently translating to Mux is
  /// coming from outside of the loop, the value of the constant should be
  /// 1
  rewriter.setInsertionPointToStart(oldMerge->getBlock());
  auto constOp = rewriter.create<handshake::ConstantOp>(
      oldMerge->getLoc(), rewriter.getIntegerType(1),
      rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0), startCtrl);
  networkConstants.push_back(constOp);
  mergeOperands.push_back(constOp.getResult());

  auto mergeOp =
      rewriter.create<handshake::MergeOp>(oldMerge->getLoc(), mergeOperands);
  initMerges.push_back(mergeOp);
  allocationNetwork.push_back(mergeOp);
  return mergeOp->getResult(0);
}

LogicalResult HandshakeLowering::triggerConstantsFromStart(
    ConversionPatternRewriter &rewriter) {
  auto constants = region.getOps<mlir::arith::ConstantOp>();
  for (auto cstOp : llvm::make_early_inc_range(constants)) {
    rewriter.setInsertionPoint(cstOp);
    TypedAttr cstAttr = cstOp.getValue();
    rewriter.replaceOpWithNewOp<handshake::ConstantOp>(cstOp, cstAttr.getType(),
                                                       cstAttr, startCtrl);
  }
  return success();
}

bool findClosestBranchPredecessor(Value input, DominanceInfo &domInfo,
                                  Block &block, Value &desiredCond,
                                  bool &getTrueSuccessor,
                                  std::unordered_set<Operation *> &visited) {
  Operation *defOp = input.getDefiningOp();
  if (!defOp || visited.count(defOp))
    return false;

  visited.insert(defOp);

  for (Value pred : defOp->getOperands()) {
    Operation *predOp = pred.getDefiningOp();
    if (!predOp)
      continue;

    if (isa<handshake::ConditionalBranchOp>(predOp)) {
      auto branch = dyn_cast<handshake::ConditionalBranchOp>(predOp);
      for (Value branchPred : branch->getOperands()) {
        if (domInfo.dominates(branchPred.getParentBlock(), &block)) {
          desiredCond = branch.getConditionOperand();
          if (pred == branch.getFalseResult()) {
            getTrueSuccessor = true;
          }
          return true;
        }
      }
    }

    if (findClosestBranchPredecessor(pred, domInfo, block, desiredCond,
                                     getTrueSuccessor, visited)) {
      return true;
    }
  }

  return false;
}

bool findClosestBranchPredecessor(Value input, DominanceInfo &domInfo,
                                  Block &block, Value &desiredCond,
                                  bool &getTrueSuccessor) {
  std::unordered_set<Operation *> visited;
  return findClosestBranchPredecessor(input, domInfo, block, desiredCond,
                                      getTrueSuccessor, visited);
}

LogicalResult
HandshakeLowering::addSuppGSA(ConversionPatternRewriter &rewriter) {
  for (Block &block : region.getBlocks()) {
    /// (1) Loop through all operations searching for Muxes not at loop
    /// headers
    for (Operation &op : block.getOperations()) {
      if (llvm::find(allocationNetwork, &op) == allocationNetwork.end() &&
          (!isa<handshake::ConstantOp>(op) ||
           llvm::find(networkConstants, &op) != networkConstants.end()))
        continue;

      if (llvm::find(shannonMUXes, &op) != shannonMUXes.end())
        continue;

      bool loopHeader = false;
      if (li.getLoopFor(&block))
        loopHeader = li.getLoopFor(&block)->getHeader() == &block;

      if (!isa<handshake::MuxOp>(op) || loopHeader)
        continue;

      auto mux = dyn_cast<handshake::MuxOp>(op);
      DominanceInfo domInfo;

      llvm::errs() << mux << "\n";
      for (Value v : mux->getOperands())
        llvm::errs() << v << "\n";

      bool inputDominting = false;
      auto it = mux.getDataOperands().begin();
      Value firstInput = *it;
      ++it;
      Value secondInput = *it;
      Value dominatingInput = firstInput;
      Value nonDominatingInput = secondInput;

      if (domInfo.dominates(firstInput.getParentBlock(), &block))
        inputDominting = true;
      else if (domInfo.dominates(secondInput.getParentBlock(), &block)) {
        inputDominting = true;
        dominatingInput = secondInput;
        nonDominatingInput = firstInput;
      }

      if (inputDominting) {
        Value desiredCond;
        /// Assert that the BB of the other input is not dominating the BB
        /// of the Mux
        assert(
            !domInfo.dominates(nonDominatingInput.getParentBlock(), &block) &&
            "The BB of the other input of the Mux should not dominate the BB "
            "of the Mux");

        /// Assert that the predecessors of the other input must all be
        /// Branches
        /*
              bool hasPredBranch = false;
              bool getTrueSuccessor = false;

              if (nonDominatingInput.getDefiningOp()) {
                for (Value pred :
           nonDominatingInput.getDefiningOp()->getOperands()) { Operation
           *predOp = pred.getDefiningOp(); if (predOp) { if
           (isa<handshake::ConditionalBranchOp>(predOp)) { auto branch =
           dyn_cast<handshake::ConditionalBranchOp>(predOp); for (Value
           branchPred : branch->getOperands()) { if
           (domInfo.dominates(branchPred.getParentBlock(), &block)) {
                          hasPredBranch = true;
                          desiredCond = branch.getConditionOperand();
                          if (pred == branch.getFalseResult())
                            getTrueSuccessor = true;
                        }
                      }
                    }
                  }
                }
              }
              */

        bool getTrueSuccessor = false;
        bool hasPredBranch = findClosestBranchPredecessor(
            nonDominatingInput, domInfo, block, desiredCond, getTrueSuccessor);

        assert(hasPredBranch &&
               "At least one predecessor of the non-dominating "
               "input must be a Branch");

        rewriter.setInsertionPointAfterValue(dominatingInput);
        auto branchOp = rewriter.create<handshake::ConditionalBranchOp>(
            dominatingInput.getLoc(), desiredCond, dominatingInput);
        allocationNetwork.push_back(branchOp);

        Value newInput = branchOp.getFalseResult();
        if (getTrueSuccessor)
          newInput = branchOp.getTrueResult();

        mux->replaceUsesOfWith(dominatingInput, newInput);
      }
    }
  }
  return success();
}

//===-----------------------------------------------------------------------==//
// Lowering strategy
//===-----------------------------------------------------------------------==//

namespace {

/// Conversion target for lowering a region.
struct LowerRegionTarget : public ConversionTarget {
  explicit LowerRegionTarget(MLIRContext &context, Region &region)
      : ConversionTarget(context), region(region) {
    // The root operation is marked dynamically legal to ensure
    // the pattern on its region is only applied once.
    markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (op != region.getParentOp())
        return true;
      return regionLowered;
    });
  }

  /// Whether the region's parent operation was lowered.
  bool regionLowered = false;
  /// The region being lowered.
  Region &region;
};

/// Allows to partially lower a region by matching on the parent operation
/// to then call the provided partial lowering function with the region
/// and the rewriter.
struct PartialLowerRegion : public ConversionPattern {
  using PartialLoweringFunc =
      std::function<LogicalResult(Region &, ConversionPatternRewriter &)>;

  PartialLowerRegion(LowerRegionTarget &target, MLIRContext *context,
                     LogicalResult &loweringResRef,
                     const PartialLoweringFunc &fun)
      : ConversionPattern(target.region.getParentOp()->getName().getStringRef(),
                          1, context),
        target(target), loweringRes(loweringResRef), fun(fun) {}
  using ConversionPattern::ConversionPattern;
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    // Dialect conversion scheme requires the matched root operation to be
    // replaced or updated if the match was successful; this ensures that
    // happens even if the lowering function does not modify the root
    // operation
    rewriter.updateRootInPlace(
        op, [&] { loweringRes = fun(target.region, rewriter); });

    // Signal to the conversion target that the conversion pattern ran
    target.regionLowered = true;

    // Success status of conversion pattern determined by success of
    // partial lowering function
    return loweringRes;
  };

private:
  LowerRegionTarget &target;
  LogicalResult &loweringRes;
  PartialLoweringFunc fun;
};

/// Strategy class for SSA maximization during std-to-handshake
/// conversion. Block arguments of type MemRefType and allocation
/// operations are not considered for SSA maximization.
class HandshakeLoweringSSAStrategy : public dynamatic::SSAMaximizationStrategy {
  /// Filters out block arguments of type MemRefType
  bool maximizeArgument(BlockArgument arg) override {
    return !arg.getType().isa<mlir::MemRefType>();
  }

  /// Filters out allocation operations
  bool maximizeOp(Operation &op) override { return !isAllocOp(&op); }
};
} // namespace

LogicalResult
dynamatic::partiallyLowerRegion(const RegionLoweringFunc &loweringFunc,
                                Region &region) {
  Operation *op = region.getParentOp();
  MLIRContext *ctx = region.getContext();
  RewritePatternSet patterns(ctx);
  LowerRegionTarget target(*ctx, region);
  LogicalResult partialLoweringSuccessfull = success();
  patterns.add<PartialLowerRegion>(target, ctx, partialLoweringSuccessfull,
                                   loweringFunc);
  return success(
      applyPartialConversion(op, target, std::move(patterns)).succeeded() &&
      partialLoweringSuccessfull.succeeded());
}

/// Lowers the region referenced by the handshake lowering strategy
/// following a fixed sequence of steps.
static LogicalResult lowerRegion(HandshakeLowering &hl) {

  if (failed(runPartialLowering(hl, &HandshakeLowering::createControlNetwork)))
    return failure();

  //===--------------------------------------------------------------------===//
  // Merges and branches instantiation
  //===--------------------------------------------------------------------===//

  if (failed(runPartialLowering(hl, &HandshakeLowering::addMergeOps)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLowering::addBranchOps)))
    return failure();

  //===--------------------------------------------------------------------===//
  // Create, analyze, and connect memory ports and interfaces
  //===--------------------------------------------------------------------===//

  HandshakeLowering::MemInterfacesInfo memInfo;
  if (failed(runPartialLowering(hl, &HandshakeLowering::replaceMemoryOps,
                                memInfo)))
    return failure();

  // First round of bb-tagging so that newly inserted Dynamatic memory
  // ports get tagged with the BB they belong to (required by memory
  // interface instantiation logic)
  if (failed(runPartialLowering(hl, &HandshakeLowering::idBasicBlocks)))
    return failure();

  if (failed(runPartialLowering(
          hl, &HandshakeLowering::verifyAndCreateMemInterfaces, memInfo)))
    return failure();

  //===--------------------------------------------------------------------===//
  // Fast Token Delivery
  //===--------------------------------------------------------------------===//

  if (failed(runPartialLowering(hl,
                                &HandshakeLowering::triggerConstantsFromStart)))
    return failure();

  if (failed(
          runPartialLowering(hl, &HandshakeLowering::replaceUndefinedValues)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLowering::addPhi)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLowering::addSupp)))
    return failure();

  std::set<Operation *> oldBranches;
  size_t oldCount = 0;
  size_t count = 0;
  while (true) {
    if (failed(runPartialLowering(hl, &HandshakeLowering::addSuppBranches,
                                  oldBranches, count)))
      return failure();
    if (count == oldCount)
      break;
    oldCount = count;
  }

  if (failed(runPartialLowering(hl, &HandshakeLowering::addSuppForStart)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLowering::convertMergesToMuxes)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLowering::addSuppGSA)))
    return failure();

  //===--------------------------------------------------------------------===//
  //    Simple final transformations
  //===--------------------------------------------------------------------===//

  if (failed(runPartialLowering(hl, &HandshakeLowering::convertCalls)))
    return failure();

  // if (failed(runPartialLowering(hl, &HandshakeLowering::connectConstants)))
  //   return failure();

  if (failed(runPartialLowering(hl, &HandshakeLowering::idBasicBlocks)))
    return failure();

  //===--------------------------------------------------------------------===//
  // Create return/end logic and flatten IR (delete actual basic blocks)
  //===--------------------------------------------------------------------===//

  return runPartialLowering(hl, &HandshakeLowering::createReturnNetwork);
}

namespace {

/// Converts a func-level function into a handshake-level function,
/// without modifying the function's body. The function signature gets an
/// extra control-only argument to represent the starting point of the
/// control network. If the function did not return any result, a
/// control-only result is added to signal function completion.
struct ConvertFuncToHandshake : OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Put the function into maximal SSA form if it is not external
    if (!funcOp.isExternal()) {
      HandshakeLoweringSSAStrategy strategy;
      if (failed(dynamatic::maximizeSSA(funcOp.getBody(), strategy)))
        return failure();
    }

    // Derive attribute for the new function
    SmallVector<NamedAttribute, 4> attributes;
    MLIRContext *ctx = getContext();
    for (const NamedAttribute &attr : funcOp->getAttrs()) {
      StringAttr attrName = attr.getName();

      // The symbol and function type attributes are set directly by the
      // Handshake function constructor, all others are forwarded directly
      if (attrName == SymbolTable::getSymbolAttrName() ||
          attrName == funcOp.getFunctionTypeAttrName())
        continue;

      // Argument names need to be augmented with the additional start
      // argument
      if (attrName == funcOp.getArgAttrsAttrName()) {
        // Extracts the name key's value from the dictionary attribute
        // corresponding to each function's argument.
        auto extractNames = [&](Attribute argAttr) -> Attribute {
          DictionaryAttr argDict = cast<DictionaryAttr>(argAttr);
          std::optional<NamedAttribute> name =
              argDict.getNamed("handshake.arg_name");
          assert(name && "missing name key in arg attribute");
          return name->getValue();
        };

        SmallVector<Attribute> argNames;
        llvm::transform(funcOp.getArgAttrsAttr(), std::back_inserter(argNames),
                        extractNames);
        argNames.push_back(StringAttr::get(ctx, "start"));
        attributes.emplace_back(StringAttr::get(ctx, "argNames"),
                                ArrayAttr::get(ctx, argNames));
        continue;
      }

      // All other attributes are forwarded without changes
      attributes.push_back(attr);
    }

    // Derive function argument and result types
    NoneType noneType = rewriter.getNoneType();
    SmallVector<Type, 8> argTypes(funcOp.getArgumentTypes());
    SmallVector<Type, 8> resTypes(funcOp.getResultTypes());
    if (resTypes.empty()) {
      resTypes.push_back(noneType);
      // The only result should be named "end"
      auto resNames = ArrayAttr::get(ctx, {StringAttr::get(ctx, "end")});
      attributes.emplace_back(StringAttr::get(ctx, "resNames"), resNames);
    }
    argTypes.push_back(noneType);
    FunctionType funcType = rewriter.getFunctionType(argTypes, resTypes);

    // Replace the func-level function with a corresponding
    // handshake-level function
    rewriter.setInsertionPoint(funcOp);
    auto newFuncOp = rewriter.create<handshake::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), funcType, attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    newFuncOp.resolveArgAndResNames();

    rewriter.eraseOp(funcOp);
    return success();
  }
};

/// FPGA18's elastic pass. Runs elastic pass on every function
/// (func::FuncOp) of the module it is applied on. Succeeds whenever all
/// functions in the module were succesfully lowered to handshake.
struct CfToHandshakePass
    : public dynamatic::impl::CfToHandshakeBase<CfToHandshakePass> {

  void runDynamaticPass() override {

    MLIRContext *ctx = &getContext();
    ModuleOp modOp = getOperation();

    // First convert functions from func-level to handshake-level, without
    // altering their bodies yet
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet patterns{ctx};
    patterns.add<ConvertFuncToHandshake>(ctx);

    // All func-level functions must become handshake-level functions
    ConversionTarget funcTarget(*ctx);
    funcTarget.addIllegalOp<func::FuncOp>();
    funcTarget.addLegalOp<handshake::FuncOp>();

    // Call the analysis hee and fill the structures from Func::FuncOp
    ControlDependenceAnalysis cdgAnalysis =
        getAnalysis<ControlDependenceAnalysis>();

    if (failed(applyPartialConversion(modOp, funcTarget, std::move(patterns))))
      return signalPassFailure();

    // Loop over the structures to change the pointer addresses to those
    // from the handshake:FuncOp  (AYA: might need to add write functions
    // to your)

    // Lower every function individually
    auto funcOps = modOp.getOps<handshake::FuncOp>();
    int funcOpIdx = 0;
    for (handshake::FuncOp funcOp : llvm::make_early_inc_range(funcOps)) {
      // Loop over the Blocks of this funcOp to update the ptrs of every
      // Block in the control dependency strucute of this funcOp_idx
      mlir::Region &funcReg = funcOp.getBody();
      for (mlir::Block &block : funcReg.getBlocks()) {
        cdgAnalysis.adjustBlockPtr(funcOpIdx, &block);
      }

      // Lower the region inside the function if it is not external
      if (!funcOp.isExternal()) {
        mlir::DominanceInfo domInfo;
        HandshakeLowering hl(funcOp.getBody(), funcOpIdx,
                             getAnalysis<NameAnalysis>(),
                             getAnalysis<ControlDependenceAnalysis>(), domInfo);

        if (failed(lowerRegion(hl)))
          return signalPassFailure();
        funcOpIdx++;
      }
    }
  }
};

} // namespace

std::unique_ptr<dynamatic::DynamaticPass> dynamatic::createCfToHandshake() {
  return std::make_unique<CfToHandshakePass>();
}
