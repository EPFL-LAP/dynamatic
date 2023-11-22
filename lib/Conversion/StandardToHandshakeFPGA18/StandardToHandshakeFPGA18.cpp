//===- StandardToHandshakeFPGA18.cpp - FPGA18's elastic pass ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the elastic pass, as described in
// https://www.epfl.ch/labs/lap/wp-content/uploads/2018/11/JosipovicFeb18_DynamicallyScheduledHighLevelSynthesis_FPGA18.pdf.
// The implementation relies for some parts on CIRCT's standard-to-handshake
// conversion pass, but brings siginificant changes related to memory interface
// management and return network creation.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/StandardToHandshakeFPGA18.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/ConstantAnalysis.h"
#include "dynamatic/Conversion/PassDetails.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <iterator>
#include <utility>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::affine;
using namespace mlir::memref;
using namespace dynamatic;

#define returnOnError(logicalResult)                                           \
  if (failed(logicalResult))                                                   \
    return failure();

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

/// Returns load/store results which are to be given as operands to a memory
/// interface.
static SmallVector<Value, 2> getResultsToMemory(Operation *op) {

  if (auto loadOp = dyn_cast<handshake::LoadOpInterface>(op)) {
    // For load, get address output
    SmallVector<Value, 2> results;
    results.push_back(loadOp.getAddressOutput());
    return results;
  }
  // For store, all outputs (address and data) go to memory
  auto storeOp = dyn_cast<handshake::StoreOpInterface>(op);
  assert(storeOp && "input operation must either be load or store");
  SmallVector<Value, 2> results(storeOp->getResults());
  return results;
}

/// Adds the data input (from memory interface) to the list of load operands.
static void addLoadDataOperand(handshake::LoadOpInterface loadOp,
                               Value dataIn) {
  assert(loadOp->getNumOperands() == 1 &&
         "load must have single address operand at this point");
  SmallVector<Value, 2> operands;
  operands.push_back(loadOp->getOperand(0));
  operands.push_back(dataIn);
  loadOp->setOperands(operands);
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
    if (endNetworkId.has_value())
      mergeOp->setAttr(BB_ATTR,
                       rewriter.getUI32IntegerAttr(endNetworkId.value()));
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

/// Returns the index of the block in its enclosing region (its position in the
/// region's block list).
static unsigned getBlockNumber(Block *block) {
  for (auto [idx, blockIt] : llvm::enumerate(block->getParent()->getBlocks())) {
    if (&blockIt == block)
      return idx;
  }
  llvm_unreachable("block does not exist");
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

//===-----------------------------------------------------------------------==//
// Concrete lowering steps
//===-----------------------------------------------------------------------==//

LogicalResult HandshakeLoweringFPGA18::createControlOnlyNetwork(
    ConversionPatternRewriter &rewriter) {

  // Add start point of the control-only path to the entry block's arguments
  Block *entryBlock = &r.front();
  startCtrl =
      entryBlock->addArgument(rewriter.getNoneType(), rewriter.getUnknownLoc());
  setBlockEntryControl(entryBlock, startCtrl);

  // Add a control-only argument to each block
  for (auto &block : r.getBlocks())
    if (!block.isEntryBlock())
      setBlockEntryControl(&block, block.addArgument(startCtrl.getType(),
                                                     rewriter.getUnknownLoc()));
  // Modify branch-like block terminators to forward control value through
  // all blocks
  for (auto &block : r.getBlocks())
    if (auto op = dyn_cast<BranchOpInterface>(block.getTerminator()); op)
      for (unsigned i = 0, e = op->getNumSuccessors(); i < e; i++)
        op.getSuccessorOperands(i).append(getBlockEntryControl(&block));

  return success();
}

LogicalResult HandshakeLoweringFPGA18::replaceMemoryOps(
    ConversionPatternRewriter &rewriter,
    HandshakeLoweringFPGA18::MemInterfacesInfo &memInfo) {

  // Make sure to record external memories passed as function arguments, even if
  // they aren't used by any memory operation
  for (BlockArgument arg : r.getArguments()) {
    if (mlir::MemRefType memref = dyn_cast<mlir::MemRefType>(arg.getType())) {
      // Ensure that this is a valid memref-typed value.
      if (!isValidMemrefType(arg.getLoc(), memref))
        return failure();
      memInfo[arg] = {};
    }
  }

  // Replace load and store operations with their corresponding Handshake
  // equivalent. Traverse and store memory operations in program order (required
  // by memory interface placement later)
  for (Operation &op : llvm::make_early_inc_range(r.getOps())) {
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

    // The memory operation must have a MemInterfaceAttr attribute attached
    StringRef attrName = MemInterfaceAttr::getMnemonic();
    MemInterfaceAttr memAttr = op.getAttrOfType<MemInterfaceAttr>(attrName);
    if (!memAttr)
      return op.emitError()
             << "Memory operation must have attribute " << attrName
             << " of type circt::handshake::MemInterfaceAttr to decide which "
                "memory interface it should connect to.";
    bool connectToMC = memAttr.connectsToMC();

    // Replace memref operation with corresponding handshake operation
    llvm::TypeSwitch<Operation *>(&op)
        .Case<memref::LoadOp>([&](memref::LoadOp loadOp) {
          OperandRange indices = loadOp.getIndices();
          assert(indices.size() == 1 && "load must be unidimensional");
          if (connectToMC)
            newOp = rewriter.create<handshake::MCLoadOp>(
                op.getLoc(), cast<MemRefType>(memref.getType()), indices[0]);
          else
            newOp = rewriter.create<handshake::LSQLoadOp>(
                op.getLoc(), cast<MemRefType>(memref.getType()), indices[0]);
          // Replace uses of old load result with data result of new load
          op.getResult(0).replaceAllUsesWith(
              dyn_cast<handshake::LoadOpInterface>(newOp).getDataOutput());
        })
        .Case<memref::StoreOp>([&](memref::StoreOp storeOp) {
          OperandRange indices = storeOp.getIndices();
          assert(indices.size() == 1 && "load must be unidimensional");
          if (connectToMC)
            newOp = rewriter.create<handshake::MCStoreOp>(
                op.getLoc(), indices[0], storeOp.getValueToStore());
          else
            newOp = rewriter.create<handshake::LSQStoreOp>(
                op.getLoc(), indices[0], storeOp.getValueToStore());
        })
        .Default([&](auto) {
          return op.emitOpError() << "Memory operation type is not supported.";
        });

    // Associate the new operation with the memory region it references and
    // information about the memory interface it should connect to
    if (memAttr.connectsToMC())
      memInfo[memref].mcPorts[op.getBlock()].push_back(newOp);
    else
      memInfo[memref].lsqPorts[*memAttr.getLsqGroup()].push_back(newOp);

    // Delete the now unused old memory operation
    rewriter.eraseOp(&op);
  }

  return success();
}

LogicalResult
HandshakeLoweringFPGA18::connectConstants(ConversionPatternRewriter &rewriter) {

  for (auto cstOp :
       llvm::make_early_inc_range(r.getOps<mlir::arith::ConstantOp>())) {

    rewriter.setInsertionPointAfter(cstOp);
    auto cstVal = cstOp.getValue();

    if (isCstSourcable(cstOp))
      rewriter.replaceOpWithNewOp<handshake::ConstantOp>(
          cstOp, cstVal.getType(), cstVal,
          rewriter.create<handshake::SourceOp>(cstOp.getLoc(),
                                               rewriter.getNoneType()));
    else
      rewriter.replaceOpWithNewOp<handshake::ConstantOp>(
          cstOp, cstVal.getType(), cstVal,
          getBlockEntryControl(cstOp->getBlock()));
  }
  return success();
}

LogicalResult HandshakeLoweringFPGA18::verifyAndCreateLSQGroups(
    ConversionPatternRewriter &rewriter, MemInterfacesInfo &memInfo,
    MemInterfacesInputs &memInputs) {
  // Create a mapping between each block and all the other blocks it properly
  // dominates
  DominanceInfo domInfo;
  DenseMap<Block *, DenseSet<Block *>> dominations;
  for (Block &maybeDominator : r) {
    // Start with an empty set of dominated blocks for each potential dominator
    dominations[&maybeDominator] = {};
    for (Block &maybeDominated : r) {
      if (&maybeDominator == &maybeDominated)
        continue;
      if (domInfo.properlyDominates(&maybeDominator, &maybeDominated))
        dominations[&maybeDominator].insert(&maybeDominated);
    }
  }

  // Each memory region is independent from the others. Verify group validity
  // and derive LSQ inputs at the same time
  for (auto &[memref, memAcesses] : memInfo) {
    MemInputs &allInputs = memInputs[memref];
    SmallPtrSet<Block *, 4> controlBlocks;

    for (auto &[_, group] : memAcesses.lsqPorts) {
      assert(!group.empty() && "group cannot be empty");

      // Group accesses by the basic block they belong to
      llvm::MapVector<Block *, SmallVector<Operation *>> opsPerBlock;
      for (Operation *op : group)
        opsPerBlock[op->getBlock()].push_back(op);

      // Check whether there is a clear "linear dominance" relationship between
      // all blocks, and derive a port ordering for the group from it
      SmallVector<Block *> order;
      if (failed(computeLinearDominance(dominations, opsPerBlock, order)))
        return failure();

      // Verify that no two groups have the same control signal
      if (auto [_, newCtrl] = controlBlocks.insert(order.front()); !newCtrl)
        return group.front()->emitError()
               << "Inconsistent LSQ group for memory interface the operation "
                  "references. No two groups can have the same control signal.";

      // Append all group inputs in the correct order. Within each block
      // operations are naturally in program order since we always use ordered
      // maps and iterated over the operations in program order to begin with
      allInputs.lsqInputs.push_back(getBlockEntryControl(order.front()));
      for (Block *inputBlock : order) {
        for (Operation *memOp : opsPerBlock[inputBlock]) {
          if (auto loadOp = dyn_cast<handshake::LSQLoadOp>(memOp)) {
            // Accumulate the number of loads and store the load order to
            // connect LSQ interfaces to load ports later
            ++allInputs.lsqNumLoads;
            allInputs.lsqLoadOrder.push_back(loadOp);
          }
          llvm::copy(getResultsToMemory(memOp),
                     std::back_inserter(allInputs.lsqInputs));
        }
      }
      allInputs.lsqGroupSizes.push_back(group.size());
    }
  }
  return success();
}

/// For simple memory controllers the control signal is fed through a constant
/// indicating the number of stores in the block (to eventually indicate block
/// completion to the end node). Returns that constant signal.
static inline Value getMCControlSignal(Value blockCtrl, unsigned numStores,
                                       ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(blockCtrl.getDefiningOp());
  return rewriter
      .create<handshake::ConstantOp>(blockCtrl.getLoc(), rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(numStores),
                                     blockCtrl)
      .getResult();
}

LogicalResult
HandshakeLoweringFPGA18::createMCBlocks(ConversionPatternRewriter &rewriter,
                                        MemInterfacesInfo &memInfo,
                                        MemInterfacesInputs &memInputs) {
  // Each memory region is independent from the others. Derive MC inputs from
  // the MC and LSQ ports (the latter is required because it may create
  // additional control signals to the MC)
  for (auto &[memref, memAccesses] : memInfo) {
    if (memAccesses.mcPorts.empty())
      continue;

    SmallVector<Value> &mcInputs = memInputs[memref].mcInputs;
    SmallVector<unsigned> &mcBlocks = memInputs[memref].mcBlocks;
    unsigned &mcNumLoads = memInputs[memref].mcNumLoads;

    // The MC also needs control signals from blocks containing store ports
    // connected to an LSQ, since these requests are forwarded to the MC. Count
    // the number of LSQ stores per block
    DenseMap<Block *, unsigned> lsqStores;
    for (auto [_, lsqGroupMemOps] : memAccesses.lsqPorts) {
      for (Operation *lsqMemOp : lsqGroupMemOps) {
        if (isa<handshake::LSQStoreOp>(lsqMemOp))
          lsqStores[lsqMemOp->getBlock()] += 1;
      }
    }

    // First, iterate over blocks that have at least one direct load/store
    // access port to the MC
    for (auto &[block, blockMemOps] : memAccesses.mcPorts) {
      mcBlocks.push_back(getBlockNumber(block));

      // Count the number of stores in the block, and accumulate the total
      // number of loads to the interface
      unsigned numStoresInBlock = lsqStores.lookup(block);
      for (Operation *memOp : blockMemOps) {
        if (isa<handshake::MCLoadOp>(memOp))
          ++mcNumLoads;
        else
          ++numStoresInBlock;
      }

      if (numStoresInBlock > 0) {
        mcInputs.push_back(getMCControlSignal(getBlockEntryControl(block),
                                              numStoresInBlock, rewriter));
      }

      // Traverse the list of memory operations in the block once more and
      // accumulate memory inputs coming from the block
      for (Operation *memOp : blockMemOps)
        llvm::copy(getResultsToMemory(memOp), std::back_inserter(mcInputs));
    }

    // Second, iterate over blocks that an LSQ will forward memory requests
    // from, and add a single control signal for these blocks
    for (auto &[lsqBlock, numStores] : lsqStores) {
      // We only need to do something if the block's potential stores have not
      // yet been accounted for
      if (memAccesses.mcPorts.contains(lsqBlock) || numStores == 0)
        continue;

      mcBlocks.push_back(getBlockNumber(lsqBlock));
      mcInputs.push_back(getMCControlSignal(getBlockEntryControl(lsqBlock),
                                            numStores, rewriter));
    }
  }

  return success();
}

LogicalResult HandshakeLoweringFPGA18::connectToMemInterfaces(
    ConversionPatternRewriter &rewriter, MemInterfacesInfo &memInfo,
    MemInterfacesInputs &memInputs) {

  // Connect memories (externally defined by memref block argument) to their
  // respective loads and stores
  for (auto &[memref, allMemOps] : memInfo) {
    MemInputs &allInputs = memInputs[memref];

    // Check whether we need any interface at all
    if (allInputs.mcInputs.empty() && allInputs.lsqInputs.empty())
      continue;

    // Prepare to insert memory interfaces
    Block *entryBlock = &r.front();
    Location loc = entryBlock->front().getLoc();
    rewriter.setInsertionPointToStart(entryBlock);
    handshake::MemoryControllerOp mcOp = nullptr;
    handshake::LSQOp lsqOp = nullptr;

    if (!allInputs.mcInputs.empty() && allInputs.lsqInputs.empty()) {
      // We only need a memory controller
      mcOp = rewriter.create<handshake::MemoryControllerOp>(
          loc, memref, allInputs.mcInputs, allInputs.mcBlocks,
          allInputs.mcNumLoads);
    } else if (allInputs.mcInputs.empty() && !allInputs.lsqInputs.empty()) {
      // We only need an LSQ
      lsqOp = rewriter.create<handshake::LSQOp>(
          loc, memref, allInputs.lsqInputs, allInputs.lsqGroupSizes,
          allInputs.lsqNumLoads);
    } else {
      // We need a MC and an LSQ. They need to be connected with 4 new channels
      // so that the LSQ can forward its loads and stores to the MC. We need
      // load address, store address, and store data channels from the LSQ to
      // the MC and a load data channel from the MC to the LSQ
      MemRefType memrefType = memref.getType().cast<MemRefType>();

      // Create 3 backedges (load address, store address, store data) for the MC
      // inputs that will eventually come from the LSQ.
      BackedgeBuilder edgeBuilder(rewriter, loc);
      Backedge ldAddr = edgeBuilder.get(rewriter.getIndexType());
      Backedge stAddr = edgeBuilder.get(rewriter.getIndexType());
      Backedge stData = edgeBuilder.get(memrefType.getElementType());
      allInputs.mcInputs.push_back(ldAddr);
      allInputs.mcInputs.push_back(stAddr);
      allInputs.mcInputs.push_back(stData);

      // Create the memory controller, adding 1 to its load count so that it
      // generates a load data result for the LSQ
      mcOp = rewriter.create<handshake::MemoryControllerOp>(
          loc, memref, allInputs.mcInputs, allInputs.mcBlocks,
          allInputs.mcNumLoads + 1);

      // Add the MC's load data result to the LSQ's inputs and create the LSQ,
      // passing a flag to the builder so that it generates the necessary
      // outputs that will go to the MC
      allInputs.lsqInputs.push_back(mcOp.getMemOutputs().back());
      lsqOp = rewriter.create<handshake::LSQOp>(loc, mcOp, allInputs.lsqInputs,
                                                allInputs.lsqGroupSizes,
                                                allInputs.lsqNumLoads);

      // Resolve the backedges to fully connect the MC and LSQ
      ValueRange lsqMemResults = lsqOp.getMemOutputs().take_back(3);
      ldAddr.setValue(lsqMemResults[0]);
      stAddr.setValue(lsqMemResults[1]);
      stData.setValue(lsqMemResults[2]);
    }

    // At this point, all load operations are missing their second operand
    // which is the data value coming from a memory interface back to the port.
    // These are the first results of each memory interface, in program order
    unsigned mcResultIdx = 0;
    for (auto &[_, blockMemoryOps] : allMemOps.mcPorts) {
      for (Operation *memOp : blockMemoryOps) {
        if (auto loadOp = dyn_cast<handshake::LoadOpInterface>(memOp)) {
          addLoadDataOperand(loadOp, mcOp->getResult(mcResultIdx++));
        }
      }
    }

    // Same for the LSQ, but here we have the load order already stored
    for (auto [resIdx, loadOp] : llvm::enumerate(allInputs.lsqLoadOrder))
      addLoadDataOperand(loadOp, lsqOp->getResult(resIdx));
  }

  // If we added constant controls, they must be labeled with a basic block
  return idBasicBlocks(rewriter);
}

LogicalResult HandshakeLoweringFPGA18::replaceUndefinedValues(
    ConversionPatternRewriter &rewriter) {
  for (auto &block : r) {
    for (auto undefOp : block.getOps<mlir::LLVM::UndefOp>()) {
      // Create an attribute of the appropriate type for the constant
      auto resType = undefOp.getRes().getType();
      TypedAttr cstAttr = llvm::TypeSwitch<Type, TypedAttr>(resType)
                              .Case<IndexType>([&](auto type) {
                                return rewriter.getIndexAttr(0);
                              })
                              .Case<IntegerType>([&](auto type) {
                                return rewriter.getIntegerAttr(type, 0);
                              })
                              .Case<FloatType>([&](auto type) {
                                return rewriter.getFloatAttr(type, 0.0);
                              })
                              .Default([&](auto type) { return nullptr; });
      if (!cstAttr)
        return undefOp->emitError() << "operation has unsupported result type";

      // Create a constant with a default value and replace the undefined value
      rewriter.setInsertionPoint(undefOp);
      auto cstOp = rewriter.create<handshake::ConstantOp>(
          undefOp.getLoc(), resType, cstAttr, getBlockEntryControl(&block));
      rewriter.replaceOp(undefOp, cstOp.getResult());
    }
  }
  return success();
}

LogicalResult
HandshakeLoweringFPGA18::idBasicBlocks(ConversionPatternRewriter &rewriter) {
  for (auto [blockID, block] : llvm::enumerate(r)) {
    for (Operation &op : block) {
      if (!isa<handshake::MemoryOpInterface>(op)) {
        // Memory interfaces do not naturally belong to any block, so they do
        // not get an attribute
        op.setAttr(BB_ATTR, rewriter.getUI32IntegerAttr(blockID));
      }
    }
  }
  return success();
}

LogicalResult HandshakeLoweringFPGA18::createReturnNetwork(
    ConversionPatternRewriter &rewriter) {

  auto *entryBlock = &r.front();
  auto &entryBlockOps = entryBlock->getOperations();

  // Move all operations to entry block. While doing so, delete all block
  // terminators and create a handshake-level return operation for each
  // existing
  // func-level return operation
  SmallVector<Operation *> terminatorsToErase;
  SmallVector<Operation *, 4> newReturnOps;
  for (auto &block : r) {
    Operation &termOp = block.back();
    if (isa<func::ReturnOp>(termOp)) {
      SmallVector<Value, 8> operands(termOp.getOperands());
      // When the enclosing function only returns a control value (no data
      // results), return statements must take exactly one control-only input
      if (operands.empty())
        operands.push_back(getBlockEntryControl(&block));

      // Insert new return operation next to the old one
      rewriter.setInsertionPoint(&termOp);
      auto newRet = rewriter.create<handshake::DynamaticReturnOp>(
          termOp.getLoc(), operands);
      newReturnOps.push_back(newRet);

      // New return operation belongs in the same basic block as the old one
      newRet->setAttr(BB_ATTR, termOp.getAttr(BB_ATTR));
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
                     ? r.getBlocks().size()
                     : newReturnOps[0]
                           ->getAttrOfType<mlir::IntegerAttr>(BB_ATTR)
                           .getValue()
                           .getZExtValue();

  // Erase all blocks except the entry block
  for (auto &block : llvm::make_early_inc_range(llvm::drop_begin(r, 1))) {
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
      mergeFunctionResults(r, rewriter, newReturnOps, endNetworkID));
  endOperands.append(getFunctionEndControls(r));
  rewriter.setInsertionPointToEnd(entryBlock);
  auto endOp = rewriter.create<handshake::EndOp>(entryBlockOps.back().getLoc(),
                                                 endOperands);
  if (endNetworkID.has_value())
    endOp->setAttr(BB_ATTR, rewriter.getUI32IntegerAttr(endNetworkID.value()));

  return success();
}

//===-----------------------------------------------------------------------==//
// Lowering strategy
//===-----------------------------------------------------------------------==//

namespace {

/// Conversion target for lowering a func::FuncOp to a handshake::FuncOp
class LowerFuncOpTarget : public ConversionTarget {
public:
  explicit LowerFuncOpTarget(MLIRContext &context) : ConversionTarget(context) {
    loweredFuncs.clear();
    addLegalDialect<handshake::HandshakeDialect>();
    addLegalDialect<func::FuncDialect>();
    addLegalDialect<arith::ArithDialect>();
    addIllegalDialect<scf::SCFDialect>();
    addIllegalDialect<affine::AffineDialect>();

    // The root operation to be replaced is marked dynamically legal based on
    // the lowering status of the given operation, see PartialLowerOp. This is
    // to make the operation go from illegal to legal after partial lowering
    addDynamicallyLegalOp<func::FuncOp>(
        [&](const auto &op) { return loweredFuncs[op]; });
  }
  DenseMap<Operation *, bool> loweredFuncs;
};

/// Conversion pattern for partially lowering a func::FuncOp to a
/// handshake::FuncOp. Lowering is achieved by a provided partial lowering
/// function.
struct PartialLowerFuncOp : public OpConversionPattern<func::FuncOp> {
  using PartialLoweringFunc =
      std::function<LogicalResult(func::FuncOp, ConversionPatternRewriter &)>;

public:
  PartialLowerFuncOp(LowerFuncOpTarget &target, MLIRContext *context,
                     const PartialLoweringFunc &fun)
      : OpConversionPattern<func::FuncOp>(context), target(target),
        loweringFunc(fun) {}
  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    // Dialect conversion scheme requires the matched root operation to be
    // replaced or updated if the match was successful. Calling
    // updateRootInPlace ensures that happens even if loweringFUnc doesn't
    // modify the root operation
    LogicalResult res = failure();
    rewriter.updateRootInPlace(op, [&] { res = loweringFunc(op, rewriter); });

    // Signal to the conversion target that the function was successfully
    // partially lowered
    target.loweredFuncs[op] = true;

    // Success status of conversion pattern determined by success of partial
    // lowering function
    return res;
  };

private:
  /// The conversion target for this pattern
  LowerFuncOpTarget &target;
  /// The rewrite function
  PartialLoweringFunc loweringFunc;
};

/// Strategy class for SSA maximization during std-to-handshake conversion.
/// Block arguments of type MemRefType and allocation operations are not
/// considered for SSA maximization.
class HandshakeLoweringSSAStrategy : public SSAMaximizationStrategy {
  /// Filters out block arguments of type MemRefType
  bool maximizeArgument(BlockArgument arg) override {
    return !arg.getType().isa<mlir::MemRefType>();
  }

  /// Filters out allocation operations
  bool maximizeOp(Operation *op) override { return !isAllocOp(op); }
};
} // namespace

/// Converts every value in the region into maximal SSA form, unless the value
/// is a block argument of type MemRefType or the result of an allocation
/// operation.
static LogicalResult maximizeSSANoMem(Region &r,
                                      ConversionPatternRewriter &rewriter) {
  HandshakeLoweringSSAStrategy strategy;
  return maximizeSSA(r, strategy, rewriter);
}

/// Convenience function for running lowerToHandshake with a partial
/// handshake::FuncOp lowering function.
static LogicalResult
partiallyLowerOp(const PartialLowerFuncOp::PartialLoweringFunc &loweringFunc,
                 MLIRContext *ctx, func::FuncOp op) {

  RewritePatternSet patterns(ctx);
  auto target = LowerFuncOpTarget(*ctx);
  patterns.add<PartialLowerFuncOp>(target, ctx, loweringFunc);
  return applyPartialConversion(op, target, std::move(patterns));
}

/// Lowers the region referenced by the handshake lowering strategy following
/// a fixed sequence of steps, some implemented in this file and some in
/// CIRCT's standard-to-handshake conversion pass.
static LogicalResult lowerRegion(HandshakeLoweringFPGA18 &hl) {
  HandshakeLowering &baseHl = static_cast<HandshakeLowering &>(hl);

  if (failed(runPartialLowering(
          hl, &HandshakeLoweringFPGA18::createControlOnlyNetwork)))
    return failure();

  //===--------------------------------------------------------------------===//
  // Merges and branches instantiation
  //===--------------------------------------------------------------------===//

  if (failed(runPartialLowering(baseHl, &HandshakeLowering::addMergeOps)))
    return failure();

  if (failed(runPartialLowering(baseHl, &HandshakeLowering::addBranchOps)))
    return failure();

  //===--------------------------------------------------------------------===//
  // Create, analyze, and connect memory ports and interfaces
  //===--------------------------------------------------------------------===//

  HandshakeLoweringFPGA18::MemInterfacesInfo memInfo;
  if (failed(runPartialLowering(hl, &HandshakeLoweringFPGA18::replaceMemoryOps,
                                memInfo)))
    return failure();

  // First round of bb-tagging so that Dynamatic memory ports get tagged
  if (failed(runPartialLowering(hl, &HandshakeLoweringFPGA18::idBasicBlocks)))
    return failure();

  HandshakeLoweringFPGA18::MemInterfacesInputs memInputs;
  if (failed(runPartialLowering(
          hl, &HandshakeLoweringFPGA18::verifyAndCreateLSQGroups, memInfo,
          memInputs)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLoweringFPGA18::createMCBlocks,
                                memInfo, memInputs)))
    return failure();

  if (failed(runPartialLowering(
          hl, &HandshakeLoweringFPGA18::connectToMemInterfaces, memInfo,
          memInputs)))
    return failure();

  //===--------------------------------------------------------------------===//
  // Simple final transformations
  //===--------------------------------------------------------------------===//

  if (failed(
          runPartialLowering(hl, &HandshakeLoweringFPGA18::connectConstants)))
    return failure();

  if (failed(runPartialLowering(
          hl, &HandshakeLoweringFPGA18::replaceUndefinedValues)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLoweringFPGA18::idBasicBlocks)))
    return failure();

  //===--------------------------------------------------------------------===//
  // Create return/end logic and flatten IR (delete actual basic blocks)
  //===--------------------------------------------------------------------===//

  return runPartialLowering(hl, &HandshakeLoweringFPGA18::createReturnNetwork);
}

/// Fully lowers a func::FuncOp to a handshake::FuncOp.
static LogicalResult lowerFuncOp(func::FuncOp funcOp, MLIRContext *ctx) {
  // Only retain those attributes that are not constructed by build
  SmallVector<NamedAttribute, 4> attributes;
  for (const auto &attr : funcOp->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == funcOp.getFunctionTypeAttrName())
      continue;
    attributes.push_back(attr);
  }

  // Get function arguments
  llvm::SmallVector<mlir::Type, 8> argTypes;
  for (auto &argType : funcOp.getArgumentTypes())
    argTypes.push_back(argType);

  // Get function results
  llvm::SmallVector<mlir::Type, 8> resTypes;
  for (auto resType : funcOp.getResultTypes())
    resTypes.push_back(resType);

  handshake::FuncOp newFuncOp;

  bool funcIsExternal = funcOp.isExternal();

  // Add control input/output to function arguments/results and create a
  // handshake::FuncOp of appropriate type
  returnOnError(partiallyLowerOp(
      [&](func::FuncOp funcOp, PatternRewriter &rewriter) {
        auto noneType = rewriter.getNoneType();
        if (resTypes.empty())
          resTypes.push_back(noneType);
        argTypes.push_back(noneType);
        auto func_type = rewriter.getFunctionType(argTypes, resTypes);
        newFuncOp = rewriter.create<handshake::FuncOp>(
            funcOp.getLoc(), funcOp.getName(), func_type, attributes);
        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                    newFuncOp.end());
        if (!funcIsExternal)
          newFuncOp.resolveArgAndResNames();
        return success();
      },
      ctx, funcOp));

  // Delete the original function
  funcOp->erase();

  // Apply SSA maximization
  returnOnError(
      partiallyLowerRegion(maximizeSSANoMem, ctx, newFuncOp.getBody()));

  if (!funcIsExternal) {
    // Lower the region inside the function
    HandshakeLoweringFPGA18 hl(newFuncOp.getBody());
    returnOnError(lowerRegion(hl));
  }

  return success();
}

namespace {
/// FPGA18's elastic pass. Runs elastic pass on every function (func::FuncOp)
/// of the module it is applied on. Succeeds whenever all functions in the
/// module were succesfully lowered to handshake.
struct StandardToHandshakeFPGA18Pass
    : public StandardToHandshakeFPGA18Base<StandardToHandshakeFPGA18Pass> {

  void runDynamaticPass() override {
    ModuleOp m = getOperation();

    // Lower every function individually
    for (auto funcOp : llvm::make_early_inc_range(m.getOps<func::FuncOp>()))
      if (failed(lowerFuncOp(funcOp, &getContext())))
        return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createStandardToHandshakeFPGA18Pass() {
  return std::make_unique<StandardToHandshakeFPGA18Pass>();
}
