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
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include <utility>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::affine;
using namespace mlir::memref;
using namespace dynamatic;

#define returnOnError(logicalResult)                                           \
  if (failed(logicalResult))                                                   \
    return failure();

// ============================================================================
// Helper functions
// ============================================================================

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

  if (auto loadOp = dyn_cast<handshake::DynamaticLoadOp>(op)) {
    // For load, get address output
    SmallVector<Value, 2> results;
    results.push_back(loadOp.getAddressResult());
    return results;
  }
  // For store, all outputs (data and address) go to memory
  auto storeOp = dyn_cast<handshake::DynamaticStoreOp>(op);
  assert(storeOp && "input operation must either be load or store");
  SmallVector<Value, 2> results(storeOp.getResults());
  return results;
}

/// Adds the data input (from memory interface) to the list of load operands.
static void addLoadDataOperand(Operation *op, Value dataIn) {
  assert(op->getNumOperands() == 1 &&
         "load must have single address operand at this point");
  SmallVector<Value, 2> operands;
  operands.push_back(op->getOperand(0));
  operands.push_back(dataIn);
  op->setOperands(operands);
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

// ============================================================================
// Concrete lowering steps
// ============================================================================

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

    // For now we don´t support memory allocations within the kernels
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

    // Replace memref operation with corresponding handshake operation
    llvm::TypeSwitch<Operation *>(&op)
        .Case<memref::LoadOp>([&](memref::LoadOp loadOp) {
          OperandRange indices = loadOp.getIndices();
          assert(indices.size() == 1 && "load must be unidimensional");
          newOp = rewriter.create<handshake::DynamaticLoadOp>(
              op.getLoc(), cast<MemRefType>(memref.getType()), indices[0]);
          copyAttr<handshake::NoLSQAttr>(loadOp, newOp);
          // Replace uses of old load result with data result of new load
          op.getResult(0).replaceAllUsesWith(
              dyn_cast<handshake::DynamaticLoadOp>(newOp).getDataResult());
        })
        .Case<memref::StoreOp>([&](memref::StoreOp storeOp) {
          OperandRange indices = storeOp.getIndices();
          assert(indices.size() == 1 && "load must be unidimensional");
          newOp = rewriter.create<handshake::DynamaticStoreOp>(
              op.getLoc(), indices[0], storeOp.getValueToStore());
          copyAttr<handshake::NoLSQAttr>(storeOp, newOp);
        })
        .Default([&](auto) {
          return op.emitOpError("Load/store operation cannot be handled.");
        });

    // Record new operation along the memory interface it uses and delete the
    // now unused old operation
    memInfo[memref][newOp->getBlock()].push_back(newOp);
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

/// Determines whether this memory port should connect to an LSQ or a memory
/// controller.
static inline bool goesToLSQ(Operation *memOp) {
  return !memOp->hasAttrOfType<handshake::NoLSQAttr>(
      handshake::NoLSQAttr::getMnemonic());
}

/// Determines whether we must place a memory controller for the provided memory
/// accesses.
static bool shouldPlaceMC(HandshakeLoweringFPGA18::MemBlockOps &allMemOps) {
  for (auto &[_, blockMemoryOps] : allMemOps) {
    for (Operation *memOp : blockMemoryOps) {
      if (!goesToLSQ(memOp))
        return true;
    }
  }
  return false;
}

std::pair<unsigned, unsigned> HandshakeLoweringFPGA18::deriveMemInterfaceInputs(
    MemBlockOps &allMemOps, ConversionPatternRewriter &rewriter,
    SmallVector<Value> &mcInputs, SmallVector<Value> &lsqInputs) {

  // Figure out whether a simple memory controller is needed
  bool placeMC = shouldPlaceMC(allMemOps);

  unsigned numLoadsMC = 0, numLoadsLSQ = 0;
  for (auto &[block, blockMemoryOps] : allMemOps) {
    // Traverse the list of operations once to determine, for the block:
    // - whether we need to connect an LSQ to at least one acceess of the
    // block
    // - the total number of stores in the block
    // - the number of stores in the block that should go to an LSQ
    // - the total number of loads in the function (accumulate)
    unsigned numStores = 0, numStoresLSQ = 0;
    for (Operation *memOp : blockMemoryOps) {
      bool lsq = goesToLSQ(memOp);
      if (isa<handshake::DynamaticStoreOp>(memOp)) {
        ++numStores;
        if (lsq)
          ++numStoresLSQ;
      } else {
        if (lsq)
          ++numLoadsLSQ;
        else
          ++numLoadsMC;
      }
    }

    // Add a control signal if the block has at least one store
    if (numStores > 0) {
      Value blockCtrl = getBlockEntryControl(block);

      // If there is at least one store to the LSQ in the block, add block
      // control signal to the interface
      if (numStoresLSQ > 0)
        lsqInputs.push_back(blockCtrl);

      if (placeMC) {
        // For simple memory controllers the control signal is fed through a
        // constant indicating the number of stores in the block (to
        // eventually indicate block completion to the end node). That's true
        // even if the stores all go through the LSQ before going to the MC
        rewriter.setInsertionPointAfter(blockCtrl.getDefiningOp());
        handshake::ConstantOp cstOp = rewriter.create<handshake::ConstantOp>(
            blockCtrl.getLoc(), rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(numStores), blockCtrl);
        mcInputs.push_back(cstOp.getResult());
      }
    }

    // Traverse the list of memory operations in the block once more and
    // accumulate memory inputs coming from the block for the correct
    // interface
    for (Operation *memOp : blockMemoryOps) {
      SmallVector<Value, 2> results = getResultsToMemory(memOp);
      // Add results of memory operation to operands of a memory interface
      SmallVector<Value> &ifaceInputs = goesToLSQ(memOp) ? lsqInputs : mcInputs;
      llvm::copy(results, std::back_inserter(ifaceInputs));
    }
  }
  return std::make_pair(numLoadsMC, numLoadsLSQ);
}

LogicalResult HandshakeLoweringFPGA18::connectToMemInterfaces(
    ConversionPatternRewriter &rewriter, MemInterfacesInfo &memInfo) {

  // Connect memories (externally defined by memref block argument) to their
  // respective loads and stores
  for (auto &[memref, allMemOps] : memInfo) {
    // Derive memory interface inputs from operations interacting with it
    SmallVector<Value> mcInputs, lsqInputs;
    auto [loadsMC, loadsLSQ] =
        deriveMemInterfaceInputs(allMemOps, rewriter, mcInputs, lsqInputs);

    // Check whether we need any interface at all
    if (mcInputs.empty() && lsqInputs.empty())
      continue;

    // Prepare to insert memory interfaces
    Block *entryBlock = &r.front();
    Location loc = entryBlock->front().getLoc();
    rewriter.setInsertionPointToStart(entryBlock);
    handshake::MemoryControllerOp mcOp = nullptr;
    handshake::LSQOp lsqOp = nullptr;

    if (!mcInputs.empty() && lsqInputs.empty()) {
      // We only need a memory controller
      mcOp = rewriter.create<handshake::MemoryControllerOp>(loc, memref,
                                                            mcInputs, loadsMC);
    } else if (mcInputs.empty() && !lsqInputs.empty()) {
      // We only need an LSQ
      lsqOp = rewriter.create<handshake::LSQOp>(loc, memref, lsqInputs,
                                                loadsLSQ, false);
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
      mcInputs.push_back(ldAddr);
      mcInputs.push_back(stAddr);
      mcInputs.push_back(stData);

      // Create the memory controller, adding 1 to its load count so that it
      // generates a load data result for the LSQ
      mcOp = rewriter.create<handshake::MemoryControllerOp>(
          loc, memref, mcInputs, loadsMC + 1);

      // Add the MC's load data result to the LSQ's inputs and create the LSQ,
      // passing a flag to the builder so that it generates the necessary
      // outputs that will go to the MC
      lsqInputs.push_back(mcOp.getMemOutputs().back());
      lsqOp = rewriter.create<handshake::LSQOp>(loc, memref, lsqInputs,
                                                loadsLSQ, true);

      // Resolve the backedges to fully connect the MC and LSQ
      ValueRange lsqMemResults = lsqOp.getMemOutputs().take_back(3);
      ldAddr.setValue(lsqMemResults[0]);
      stAddr.setValue(lsqMemResults[1]);
      stData.setValue(lsqMemResults[2]);
    }

    // At this point, all load operations are missing their second operand
    // which is the data value coming from a memory interface back to the port.
    // These are the first results of each memory interface, in program order
    unsigned mcResultIdx = 0, lsqResultIdx = 0;
    for (auto &[_, blockMemoryOps] : allMemOps) {
      for (Operation *memOp : blockMemoryOps) {
        if (isa<handshake::DynamaticLoadOp>(memOp)) {
          if (goesToLSQ(memOp))
            addLoadDataOperand(memOp, lsqOp->getResult(lsqResultIdx++));
          else
            addLoadDataOperand(memOp, mcOp->getResult(mcResultIdx++));
        }
      }
    }
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

// ============================================================================
// Lowering strategy
// ============================================================================
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

  auto &baseHl = static_cast<HandshakeLowering &>(hl);

  HandshakeLoweringFPGA18::MemInterfacesInfo memInfo;
  if (failed(runPartialLowering(hl, &HandshakeLoweringFPGA18::replaceMemoryOps,
                                memInfo)))
    return failure();

  if (failed(runPartialLowering(
          hl, &HandshakeLoweringFPGA18::createControlOnlyNetwork)))
    return failure();

  if (failed(runPartialLowering(baseHl, &HandshakeLowering::addMergeOps)))
    return failure();

  if (failed(runPartialLowering(baseHl, &HandshakeLowering::replaceCallOps)))
    return failure();

  if (failed(runPartialLowering(baseHl, &HandshakeLowering::addBranchOps)))
    return failure();

  // First round of bb-tagging so that Dynamatic memory operations are already
  // tagged
  if (failed(runPartialLowering(hl, &HandshakeLoweringFPGA18::idBasicBlocks)))
    return failure();

  if (failed(runPartialLowering(
          hl, &HandshakeLoweringFPGA18::connectToMemInterfaces, memInfo)))
    return failure();

  if (failed(
          runPartialLowering(hl, &HandshakeLoweringFPGA18::connectConstants)))
    return failure();

  if (failed(runPartialLowering(
          hl, &HandshakeLoweringFPGA18::replaceUndefinedValues)))
    return failure();

  if (failed(runPartialLowering(hl, &HandshakeLoweringFPGA18::idBasicBlocks)))
    return failure();

  if (failed(runPartialLowering(hl,
                                &HandshakeLoweringFPGA18::createReturnNetwork)))
    return failure();

  return success();
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

std::unique_ptr<dynamatic::DynamaticPass<false>>
dynamatic::createStandardToHandshakeFPGA18Pass() {
  return std::make_unique<StandardToHandshakeFPGA18Pass>();
}
