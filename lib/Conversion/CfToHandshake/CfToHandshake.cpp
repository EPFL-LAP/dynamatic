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
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <utility>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::affine;
using namespace mlir::memref;
using namespace dynamatic;

//===-----------------------------------------------------------------------==//
// Helper functions
//===-----------------------------------------------------------------------==//

/// Determines whether an operation is akin to a load or store memory operation.
static bool isMemoryOp(Operation *op) {
  return isa<memref::LoadOp, memref::StoreOp>(op);
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
/// Returns an error whenever the passed operation is not a supported memory
/// operation.
static Value getOpMemRef(Operation *op) {
  if (auto memOp = dyn_cast<memref::LoadOp>(op))
    return memOp.getMemRef();
  if (auto memOp = dyn_cast<memref::StoreOp>(op))
    return memOp.getMemRef();
  return nullptr;
}

/// Returns the list of data inputs to be passed as operands to the
/// `handshake::EndOp` of a `handshake::FuncOp`. In the case of a single return
/// statement, this is simply the return's inputs. In the case of multiple
/// returns, this is the list of individually merged inputs of all returns.
/// In the latter case, the function inserts the required `handshake::MergeOp`'s
/// in the region.
static SmallVector<Value>
mergeFuncResults(handshake::FuncOp funcOp, ConversionPatternRewriter &rewriter,
                 ArrayRef<SmallVector<Value>> returnsOperands,
                 size_t exitBlockID) {
  // No need to merge results in case of a single return
  if (returnsOperands.size() == 1)
    return returnsOperands.front();
  unsigned numReturnValues = returnsOperands.front().size();
  if (numReturnValues == 0)
    return {};

  // Return values from multiple returns need to be merged together
  SmallVector<Value, 4> results;
  Block *entryBlock = funcOp.getBodyBlock();
  Location loc = entryBlock->getOperations().back().getLoc();
  rewriter.setInsertionPointToEnd(entryBlock);
  for (unsigned i = 0, e = numReturnValues; i < e; i++) {
    SmallVector<Value, 4> mergeOperands;
    for (ValueRange operands : returnsOperands)
      mergeOperands.push_back(operands[i]);
    auto mergeOp = rewriter.create<handshake::MergeOp>(loc, mergeOperands);
    results.push_back(mergeOp.getResult());
    mergeOp->setAttr(BB_ATTR_NAME, rewriter.getUI32IntegerAttr(exitBlockID));
  }
  return results;
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
// CfToHandshakeTypeConverter
//===-----------------------------------------------------------------------==//

static std::optional<Value> oneToOneVoidMaterialization(OpBuilder &builder,
                                                        Type /*resultType*/,
                                                        ValueRange inputs,
                                                        Location /*loc*/) {
  if (inputs.size() != 1)
    return std::nullopt;
  return inputs[0];
}

static Type channelifyType(Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case<IndexType, IntegerType, FloatType>(
          [](auto type) { return handshake::ChannelType::get(type); })
      .Case<MemRefType>([](MemRefType memrefType) {
        if (!isa<IndexType>(memrefType.getElementType()))
          return memrefType;
        OpBuilder builder(memrefType.getContext());
        IntegerType elemType = builder.getIntegerType(32);
        return MemRefType::get(memrefType.getShape(), elemType);
      })
      .Case<handshake::ChannelType, handshake::ControlType>(
          [](auto type) { return type; })

      .Default([](auto type) { return nullptr; });
}

CfToHandshakeTypeConverter::CfToHandshakeTypeConverter() {
  addConversion(channelifyType);
  addArgumentMaterialization(oneToOneVoidMaterialization);
  addSourceMaterialization(oneToOneVoidMaterialization);
  addTargetMaterialization(oneToOneVoidMaterialization);
}

//===-----------------------------------------------------------------------==//
// LowerFuncToHandshake
//===-----------------------------------------------------------------------==//

using ArgReplacements = DenseMap<BlockArgument, OpResult>;

LogicalResult LowerFuncToHandshake::matchAndRewrite(
    func::FuncOp lowerFuncOp, OpAdaptor /*adaptor*/,
    ConversionPatternRewriter &rewriter) const {

  // Map all memory accesses in the matched function to the index of their
  // memref in the function's arguments
  DenseMap<Value, unsigned> memrefToArgIdx;
  for (auto [idx, arg] : llvm::enumerate(lowerFuncOp.getArguments())) {
    if (isa<mlir::MemRefType>(arg.getType()))
      memrefToArgIdx.insert({arg, idx});
  }

  // First lower the parent function itself, without modifying its body (except
  // the block arguments and terminators)
  auto funcOrFailure = lowerSignature(lowerFuncOp, rewriter);
  if (failed(funcOrFailure))
    return failure();
  handshake::FuncOp funcOp = *funcOrFailure;
  if (funcOp.isExternal())
    return success();

  // Stores mapping from each value that passes through a merge-like operation
  // to the data result of that merge operation
  ArgReplacements argReplacements;
  addMergeOps(funcOp, rewriter, argReplacements);
  addBranchOps(funcOp, rewriter);

  LowerFuncToHandshake::MemInterfacesInfo memInfo;
  if (failed(convertMemoryOps(funcOp, rewriter, memrefToArgIdx, memInfo)))
    return failure();

  // First round of bb-tagging so that newly inserted Dynamatic memory ports get
  // tagged with the BB they belong to (required by memory interface
  // instantiation logic)
  idBasicBlocks(funcOp, rewriter);
  if (failed(verifyAndCreateMemInterfaces(funcOp, rewriter, memInfo)))
    return failure();

  idBasicBlocks(funcOp, rewriter);
  return flattenAndTerminate(funcOp, rewriter, argReplacements);
}

SmallVector<NamedAttribute>
LowerFuncToHandshake::deriveNewAttributes(func::FuncOp funcOp) const {
  SmallVector<NamedAttribute, 4> attributes;
  MLIRContext *ctx = getContext();
  SmallVector<std::string> memRegionNames;
  bool hasArgNames = false;

  for (const NamedAttribute &attr : funcOp->getAttrs()) {
    StringAttr attrName = attr.getName();

    // The symbol and function type attributes are set directly by the
    // Handshake function constructor, all others are forwarded directly
    if (attrName == SymbolTable::getSymbolAttrName() ||
        attrName == funcOp.getFunctionTypeAttrName())
      continue;

    // Argument names need to be augmented with the additional start argument
    if (attrName == funcOp.getArgAttrsAttrName()) {
      hasArgNames = true;

      // Extracts the name key's value from the dictionary attribute
      // corresponding to each function's argument.
      SmallVector<Attribute> argNames;
      for (auto [argType, argAttr] :
           llvm::zip(funcOp.getArgumentTypes(), funcOp.getArgAttrsAttr())) {
        DictionaryAttr argDict = cast<DictionaryAttr>(argAttr);
        std::optional<NamedAttribute> name =
            argDict.getNamed("handshake.arg_name");
        assert(name && "missing name key in arg attribute");
        if (isa<mlir::MemRefType>(argType))
          memRegionNames.push_back(cast<StringAttr>(name->getValue()).str());
        argNames.push_back(name->getValue());
      }
      for (StringRef mem : memRegionNames)
        argNames.push_back(StringAttr::get(ctx, mem + "_start"));
      argNames.push_back(StringAttr::get(ctx, "start"));
      attributes.emplace_back(StringAttr::get(ctx, "argNames"),
                              ArrayAttr::get(ctx, argNames));
      continue;
    }

    // All other attributes are forwarded without changes
    attributes.push_back(attr);
  }

  // Regular function results have default names
  SmallVector<Attribute> resNames;
  unsigned numFuncResults = funcOp.getFunctionType().getNumResults();
  for (size_t idx = 0; idx < numFuncResults; ++idx)
    resNames.push_back(StringAttr::get(ctx, "out" + std::to_string(idx)));

  if (!hasArgNames) {
    SmallVector<Attribute> argNames;

    // Create attributes for arguments and result names
    unsigned argNum = 0, memNum = 0;
    for (BlockArgument arg : funcOp.getArguments()) {
      if (isa<mlir::MemRefType>(arg.getType())) {
        std::string name = "mem" + std::to_string(memNum++);
        memRegionNames.push_back(name);
        argNames.push_back(StringAttr::get(ctx, name));
      } else {
        argNames.push_back(
            StringAttr::get(ctx, "in" + std::to_string(argNum++)));
      }
    }
    for (StringRef memName : memRegionNames)
      argNames.push_back(StringAttr::get(ctx, memName + "_start"));
    argNames.push_back(StringAttr::get(ctx, "start"));
    attributes.emplace_back(StringAttr::get(ctx, "argNames"),
                            ArrayAttr::get(ctx, argNames));
  }

  // Use the same memory names as the arguments as the base for the result names
  for (StringRef memName : memRegionNames)
    resNames.push_back(StringAttr::get(ctx, memName + "_end"));
  resNames.push_back(StringAttr::get(ctx, "end"));
  attributes.emplace_back(StringAttr::get(ctx, "resNames"),
                          ArrayAttr::get(ctx, resNames));
  return attributes;
}

static void
setupEntryBlockConversion(Block *entryBlock, unsigned numMemories,
                          PatternRewriter &rewriter,
                          TypeConverter::SignatureConversion &conv) {
  // All func-level function arguments map one-to-one to the handshake-level
  // function arguments and get channelified in the process
  for (auto [idx, type] : llvm::enumerate(entryBlock->getArgumentTypes()))
    conv.addInputs(idx, channelifyType(type));

  // Add a new control argument for each memory and one for the start signal
  Type ctrlType = handshake::ControlType::get(rewriter.getContext());
  conv.addInputs(SmallVector<Type>{numMemories + 1, ctrlType});
}

static void setupBlockConversion(Block *block, PatternRewriter &rewriter,
                                 TypeConverter::SignatureConversion &conv) {
  // All func-level block arguments map one-to-one to the handshake-level
  // arguments and get channelified in the process
  for (auto [idx, type] : llvm::enumerate(block->getArgumentTypes()))
    conv.addInputs(idx, channelifyType(type));

  // Add a new argument for the start in each block
  conv.addInputs(handshake::ControlType::get(rewriter.getContext()));
}

FailureOr<handshake::FuncOp> LowerFuncToHandshake::lowerSignature(
    func::FuncOp funcOp, ConversionPatternRewriter &rewriter) const {
  // The Handshake function's first inputs and outputs match the original
  // function's arguments and results
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 2> resTypes;
  unsigned numMemories = 0;
  for (Type ogArgType : funcOp.getArgumentTypes()) {
    if (isa<mlir::MemRefType>(ogArgType))
      ++numMemories;
    argTypes.push_back(channelifyType(ogArgType));
  }
  for (Type ogResType : funcOp.getResultTypes())
    resTypes.push_back(channelifyType(ogResType));

  // In addition to the original function's arguments and results, the Handshake
  // function has an extra control-only output port for each memory region and
  // one for the global start/end signals
  auto ctrlType = handshake::ControlType::get(rewriter.getContext());
  argTypes.append(numMemories + 1, ctrlType);
  resTypes.append(numMemories + 1, ctrlType);

  // Create a handshake-level function corresponding to the cf-level function
  rewriter.setInsertionPoint(funcOp);
  FunctionType funTy = rewriter.getFunctionType(argTypes, resTypes);
  SmallVector<NamedAttribute> attrs = deriveNewAttributes(funcOp);
  auto newFuncOp = rewriter.create<handshake::FuncOp>(
      funcOp.getLoc(), funcOp.getName(), funTy, attrs);
  Region *oldBody = &funcOp.getBody();

  const TypeConverter *typeConv = getTypeConverter();

  // Convert the entry block's signature
  Block *entryBlock = &funcOp.getBody().front();
  TypeConverter::SignatureConversion entryConversion(
      entryBlock->getNumArguments());
  setupEntryBlockConversion(entryBlock, numMemories, rewriter, entryConversion);
  rewriter.applySignatureConversion(oldBody, entryConversion, typeConv);

  // Convert the non entry blocks' signatures
  SmallVector<TypeConverter::SignatureConversion> nonEntryConversions;
  for (Block &block : llvm::drop_begin(funcOp)) {
    auto &conv = nonEntryConversions.emplace_back(block.getNumArguments());
    setupBlockConversion(&block, rewriter, conv);
  }
  if (failed(rewriter.convertNonEntryRegionTypes(oldBody, *typeConv,
                                                 nonEntryConversions)))
    return failure();

  // Modify branch-like terminators to forward the new control value through
  // all blocks
  for (Block &block : funcOp) {
    Operation *termOp = block.getTerminator();
    Value blockCtrl = block.getArguments().back();
    rewriter.setInsertionPointToEnd(&block);
    if (auto condBrOp = dyn_cast<cf::CondBranchOp>(termOp)) {
      SmallVector<Value> trueOperands, falseOperands;
      if (failed(rewriter.getRemappedValues(condBrOp.getTrueDestOperands(),
                                            trueOperands)) ||
          failed(rewriter.getRemappedValues(condBrOp.getFalseDestOperands(),
                                            falseOperands)))
        return failure();

      trueOperands.push_back(blockCtrl);
      falseOperands.push_back(blockCtrl);
      rewriter.replaceOp(termOp,
                         rewriter.create<cf::CondBranchOp>(
                             condBrOp->getLoc(), condBrOp.getCondition(),
                             condBrOp.getTrueDest(), trueOperands,
                             condBrOp.getFalseDest(), falseOperands));

    } else if (auto brOp = dyn_cast<cf::BranchOp>(termOp)) {
      SmallVector<Value> operands;
      if (failed(rewriter.getRemappedValues(brOp.getDestOperands(), operands)))
        return failure();
      operands.push_back(blockCtrl);
      rewriter.replaceOp(termOp, rewriter.create<cf::BranchOp>(
                                     brOp->getLoc(), brOp.getDest(), operands));
    }
  }

  // Move the entire func-level body to the handhsake-level function
  Region *newBody = &newFuncOp.getBody();
  rewriter.inlineRegionBefore(*oldBody, *newBody, newFuncOp.end());
  rewriter.eraseOp(funcOp);
  return newFuncOp;
}

/// Returns the value from the predecessor block that should be used as the data
/// operand for the merge that will eventually replace the block argument.
static Value getMergeOperand(BlockArgument blockArg, Block *predBlock,
                             bool firstTimePred) {
  // The block the merge operation belongs to
  Block *block = blockArg.getOwner();

  // The block terminator is either a cf-level branch or cf-level conditional
  // branch. In either case, identify the value passed to the block using its
  // index in the list of block arguments
  unsigned argIdx = blockArg.getArgNumber();
  Operation *termOp = predBlock->getTerminator();
  if (auto condBr = dyn_cast<cf::CondBranchOp>(termOp)) {
    // Block should be one of the two destinations of the conditional branch
    Block *trueDest = condBr.getTrueDest(), *falseDest = condBr.getFalseDest();
    if (block == trueDest) {
      if (!firstTimePred) {
        assert(trueDest == falseDest && "expected same branch target");
        return condBr.getFalseOperand(argIdx);
      }
      return condBr.getTrueOperand(argIdx);
    }
    assert(block == falseDest && "expected false branch target");
    return condBr.getFalseOperand(argIdx);
  }
  if (isa<cf::BranchOp>(termOp))
    return termOp->getOperand(argIdx);
  return nullptr;
}

/// Determines the list of predecessors of the block by iterating over all block
/// terminators in the parent function. If the terminator is a conditional
/// branch whose branches both point to the target block, then the owning block
/// is added twice to the list and the branhc's "false destinatiob" is
/// associated with a false boolean value; in all other situatuions predecessor
/// blocks are associated a true boolean value.
static SmallVector<std::pair<Block *, bool>>
getRealBlockPredecessors(Block *targetBlock) {
  SmallVector<std::pair<Block *, bool>> predecessors;
  for (Block &block : cast<handshake::FuncOp>(targetBlock->getParentOp())) {
    Operation *termOp = block.getTerminator();
    if (auto condBrOp = dyn_cast<cf::CondBranchOp>(termOp)) {
      if (condBrOp.getTrueDest() == targetBlock)
        predecessors.push_back({&block, true});
      if (condBrOp.getFalseDest() == targetBlock)
        predecessors.push_back({&block, false});
    } else if (auto brOp = dyn_cast<cf::BranchOp>(termOp)) {
      if (brOp.getDest() == targetBlock)
        predecessors.push_back({&block, true});
    }
  }
  return predecessors;
}

void LowerFuncToHandshake::insertMerge(BlockArgument blockArg,
                                       ConversionPatternRewriter &rewriter,
                                       BackedgeBuilder &edgeBuilder,
                                       MergeOpInfo &iMerge) const {
  Block *block = blockArg.getOwner();
  SmallVector<std::pair<Block *, bool>> predecessors =
      getRealBlockPredecessors(block);
  assert(!predecessors.empty() && "block argument must have predecessors");
  Location loc = block->front().getLoc();
  SmallVector<Value> operands;

  // Every live-in value to a non-entry block is passed through a merge-like
  // operation, even when it's not required for circuit correctness (useless
  // merge-like operations are removed down the line during Handshake
  // canonicalization)

  auto addFromAllPredecessors = [&](Type dataType) -> void {
    for (auto &[predBlock, isFirst] : predecessors) {
      Backedge edge = edgeBuilder.get(dataType);
      iMerge.operands.emplace_back(edge, predBlock, isFirst);
      operands.push_back(Value(edge));
    }
  };

  // Every block needs to feed it's entry control into a control merge
  if (blockArg == getBlockControl(block)) {
    addFromAllPredecessors(handshake::ControlType::get(rewriter.getContext()));
    iMerge.op = rewriter.create<handshake::ControlMergeOp>(loc, operands);
  } else if (predecessors.size() == 1) {
    addFromAllPredecessors(blockArg.getType());
    iMerge.op = rewriter.create<handshake::MergeOp>(loc, operands);
  } else {
    // Create a backedge for the index operand, and another one for each data
    // operand. The index operand will eventually resolve to the current block's
    // control merge index output (which will have the optimized index width),
    // while data operands will resolve to their respective values from each
    // block predecessor
    Type idxType =
        handshake::getOptimizedIndexValType(rewriter, predecessors.size());
    iMerge.indexEdge = edgeBuilder.get(handshake::ChannelType::get(idxType));
    addFromAllPredecessors(blockArg.getType());
    Value index = *iMerge.indexEdge;
    iMerge.op = rewriter.create<handshake::MuxOp>(loc, index, operands);
  }
}

void LowerFuncToHandshake::addMergeOps(handshake::FuncOp funcOp,
                                       ConversionPatternRewriter &rewriter,
                                       ArgReplacements &argReplacements) const {
  // Create backedge builder to manage operands of merge operations between
  // insertion and reconnection
  BackedgeBuilder edgeBuilder(rewriter, funcOp.getLoc());

  // Insert merge-like operations in all non-entry blocks (with backedges
  // instead as data operands)
  DenseMap<Block *, std::vector<MergeOpInfo>> blockMerges;
  for (Block &block : llvm::drop_begin(funcOp)) {
    rewriter.setInsertionPointToStart(&block);

    // All of the block's live-ins are passed explictly through block arguments
    // thanks to prior SSA maximization
    for (BlockArgument arg : block.getArguments()) {
      MergeOpInfo &iMerge = blockMerges[&block].emplace_back(arg);
      insertMerge(arg, rewriter, edgeBuilder, iMerge);
      argReplacements.insert({arg, iMerge.op.getDataResult()});
    }
  }

  // Reconnect merge operations with values incoming from predecessor blocks
  // and resolve all backedges that were created during merge insertion
  for (Block &block : llvm::drop_begin(funcOp)) {
    // Find the control merge in the block, its index output provides the
    // index to other merge-like operations in the block
    Value indexInput = nullptr;
    for (MergeOpInfo &iMerge : blockMerges[&block]) {
      Operation *mergeLikeOp = iMerge.op.getOperation();
      if (auto cMergeOp = dyn_cast<handshake::ControlMergeOp>(mergeLikeOp)) {
        indexInput = cMergeOp.getIndex();
        break;
      }
    }
    assert(indexInput && "no control merge in the block");

    // Resolve all backedge operands to all merge-like operations in the block
    for (MergeOpInfo &iMerge : blockMerges[&block]) {
      for (auto &[dataEdge, predBlock, isFirst] : iMerge.operands) {
        Value mgOperand = getMergeOperand(iMerge.blockArg, predBlock, isFirst);
        assert(mgOperand && "failed to find merge operand");
        dataEdge.setValue(mgOperand);
      }
      if (iMerge.indexEdge)
        iMerge.indexEdge->setValue(indexInput);
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

void LowerFuncToHandshake::addBranchOps(
    handshake::FuncOp funcOp, ConversionPatternRewriter &rewriter) const {
  for (Block &block : funcOp) {
    Operation *termOp = block.getTerminator();
    Location loc = termOp->getLoc();
    rewriter.setInsertionPoint(termOp);

    Value cond = nullptr;
    if (cf::CondBranchOp condBranchOp = dyn_cast<cf::CondBranchOp>(termOp)) {
      cond = condBranchOp.getCondition();
      cond = rewriter.getRemappedValue(cond);
      assert(cond && "Failed to remap branch operand");
    } else if (isa<func::ReturnOp>(termOp)) {
      continue;
    }

    // Insert a branch-like operation for each live-out and replace the original
    // branch operand value in successor blocks with the result(s) of the new
    // operation
    for (Value branchOprd : getBranchOperands(termOp)) {
      // Create a branch-like operation for the branch operand
      Operation *newOp;
      if (cond) {
        newOp = rewriter.create<handshake::ConditionalBranchOp>(loc, cond,
                                                                branchOprd);
      } else {
        newOp = rewriter.create<handshake::BranchOp>(loc, branchOprd);
      }

      // Group users by the block which they belong to, which inform the result
      // of the branch that they will then connect to
      DenseMap<Block *, SmallPtrSet<Operation *, 4>> branchUsers;
      auto succ = block.getSuccessors();
      SmallPtrSet<Block *, 2> successors(succ.begin(), succ.end());
      for (Operation *user : branchOprd.getUsers()) {
        // Only merges in successor blocks must connect to the branch output
        if (!isa<handshake::MergeLikeOpInterface>(user) ||
            !successors.contains(user->getBlock()))
          continue;
        branchUsers[user->getBlock()].insert(user);
      }
      assert(branchUsers.size() <= 2 && "too many branch successors");

      // Connect users of the branch to the appropriate branch result
      for (const auto &userGroup : branchUsers) {
        rewriter.replaceUsesWithIf(
            branchOprd, getSuccResult(termOp, newOp, userGroup.first),
            [&](OpOperand &oprd) {
              return userGroup.second.contains(oprd.getOwner());
            });
      }
    }
  }
}

LowerFuncToHandshake::MemAccesses::MemAccesses(BlockArgument memStart)
    : memStart(memStart) {}

LogicalResult LowerFuncToHandshake::convertMemoryOps(
    handshake::FuncOp funcOp, ConversionPatternRewriter &rewriter,
    const DenseMap<Value, unsigned> &memrefIndices,
    LowerFuncToHandshake::MemInterfacesInfo &memInfo) const {
  // Count the number of memory regions in the function, and derive the starting
  // index of memory start arguments
  auto funcArgs = funcOp.getArguments();
  unsigned numMemories = llvm::count_if(
      funcArgs, [](auto arg) { return isa<mlir::MemRefType>(arg.getType()); });
  unsigned memStartOffset = funcArgs.size() - numMemories - 1;

  // Make sure to record external memories passed as function arguments,
  // even if they aren't used by any memory operation
  unsigned memIdx = 0;
  for (BlockArgument arg : funcArgs) {
    if (auto memrefTy = dyn_cast<mlir::MemRefType>(arg.getType())) {
      if (!isValidMemrefType(arg.getLoc(), memrefTy))
        return failure();
      unsigned memStartIdx = memStartOffset + (memIdx++);
      memInfo.insert({arg, {funcArgs[memStartIdx]}});
    }
  }

  // Used to keep consistency betweeen memory access names referenced by memory
  // dependencies and names of replaced memory operations
  MemoryOpLowering memOpLowering(namer);

  // Replace load and store operations with their corresponding Handshake
  // equivalent. Traverse and store memory operations in program order (required
  // by memory interface placement later)
  for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
    if (!isMemoryOp(&op))
      continue;

    // Extract the reference to the memory region from the memory operation
    rewriter.setInsertionPoint(&op);
    Value memref = getOpMemRef(&op);
    if (!memref)
      continue;
    Location loc = op.getLoc();
    Block *block = op.getBlock();

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
    Operation *newOp =
        llvm::TypeSwitch<Operation *, Operation *>(&op)
            .Case<memref::LoadOp>([&](memref::LoadOp loadOp) {
              OperandRange indices = loadOp.getIndices();
              assert(indices.size() == 1 && "load must be unidimensional");
              MemRefType type = cast<MemRefType>(memref.getType());

              Value addr = rewriter.getRemappedValue(indices.front());
              assert(addr && "failed to remap address");

              Operation *newOp;
              if (connectToMC)
                newOp = rewriter.create<handshake::MCLoadOp>(loc, type, addr);
              else
                newOp = rewriter.create<handshake::LSQLoadOp>(loc, type, addr);

              // Record the memory access replacement
              memOpLowering.recordReplacement(loadOp, newOp, false);
              Value dataOut =
                  cast<handshake::LoadOpInterface>(newOp).getDataOutput();
              rewriter.replaceOp(loadOp, dataOut);
              return newOp;
            })
            .Case<memref::StoreOp>([&](memref::StoreOp storeOp) {
              OperandRange indices = storeOp.getIndices();
              assert(indices.size() == 1 && "store must be unidimensional");

              Value addr = rewriter.getRemappedValue(indices.front());
              Value data = rewriter.getRemappedValue(storeOp.getValueToStore());
              assert((addr && data) && "failed to remap address or data");

              Operation *newOp;
              if (connectToMC)
                newOp = rewriter.create<handshake::MCStoreOp>(loc, addr, data);
              else
                newOp = rewriter.create<handshake::LSQStoreOp>(loc, addr, data);

              // Record the memory access replacement
              memOpLowering.recordReplacement(storeOp, newOp, false);
              rewriter.eraseOp(storeOp);
              return newOp;
            })
            .Default([&](auto) { return nullptr; });
    if (!newOp)
      return op.emitError() << "Memory operation type unsupported.";

    // Associate the new operation with the memory region it references and
    // the memory interface it should connect to
    auto *accessesIt = memInfo.find(funcArgs[memrefIndices.at(memref)]);
    assert(accessesIt != memInfo.end() && "unknown memref");
    if (memAttr.connectsToMC())
      accessesIt->second.mcPorts[block].push_back(newOp);
    else
      accessesIt->second.lsqPorts[*memAttr.getLsqGroup()].push_back(newOp);
  }

  memOpLowering.renameDependencies(funcOp);
  return success();
}

LogicalResult LowerFuncToHandshake::verifyAndCreateMemInterfaces(
    handshake::FuncOp funcOp, ConversionPatternRewriter &rewriter,
    MemInterfacesInfo &memInfo) const {

  if (memInfo.empty())
    return success();

  // Create a mapping between each block and all the other blocks it properly
  // dominates so that we can quickly determine whether LSQ groups make sense
  DominanceInfo domInfo;
  DenseMap<Block *, DenseSet<Block *>> dominations;
  for (Block &maybeDominator : funcOp) {
    // Start with an empty set of dominated blocks for each potential dominator
    dominations[&maybeDominator] = {};
    for (Block &maybeDominated : funcOp) {
      if (&maybeDominator == &maybeDominated)
        continue;
      if (domInfo.properlyDominates(&maybeDominator, &maybeDominated))
        dominations[&maybeDominator].insert(&maybeDominated);
    }
  }

  // Find the control value indicating the last control flow decision in the
  // function; it will be fed to memory interfaces to indicate that no more
  // group allocations will be coming
  Value ctrlEnd;
  auto returns = funcOp.getOps<func::ReturnOp>();
  assert(!returns.empty() && "no returns in function");
  if (std::distance(returns.begin(), returns.end()) == 1) {
    ctrlEnd = getBlockControl((*returns.begin())->getBlock());
  } else {
    // Merge the control signals of all blocks with a return to create a control
    // representing the final control flow decision
    SmallVector<Value> controls;
    func::ReturnOp lastRetOp;
    for (func::ReturnOp retOp : returns) {
      lastRetOp = retOp;
      controls.push_back(getBlockControl(retOp->getBlock()));
    }
    rewriter.setInsertionPointToStart(lastRetOp->getBlock());
    auto mergeOp =
        rewriter.create<handshake::MergeOp>(lastRetOp.getLoc(), controls);
    ctrlEnd = mergeOp.getResult();

    // The merge goes into an extra "end block" after all others, this will be
    // where the function end terminator will be located as well
    mergeOp->setAttr(BB_ATTR_NAME,
                     rewriter.getUI32IntegerAttr(funcOp.getBlocks().size()));
  }

  // Create a mapping between each block and its control value in the right
  // format for the memory interface builder
  DenseMap<unsigned, Value> ctrlVals;
  for (auto [blockIdx, block] : llvm::enumerate(funcOp))
    ctrlVals.insert({blockIdx, getBlockControl(&block)});

  // Each memory region is independent from the others
  for (auto &[memref, memAccesses] : memInfo) {
    SmallPtrSet<Block *, 4> controlBlocks;

    MemoryInterfaceBuilder memBuilder(funcOp, memref, memAccesses.memStart,
                                      ctrlEnd, ctrlVals);

    // Add MC ports to the interface builder
    for (auto &[_, mcBlockOps] : memAccesses.mcPorts) {
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
               << "Inconsistent LSQ group for memory interface the operation "
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

    // Build the memory interfaces
    handshake::MemoryControllerOp mcOp;
    handshake::LSQOp lsqOp;
    if (failed(memBuilder.instantiateInterfaces(rewriter, mcOp, lsqOp)))
      return failure();
  }

  return success();
}

void LowerFuncToHandshake::idBasicBlocks(
    handshake::FuncOp funcOp, ConversionPatternRewriter &rewriter) const {
  for (auto [blockID, block] : llvm::enumerate(funcOp)) {
    for (Operation &op : block) {
      if (!isa<handshake::MemoryOpInterface>(op)) {
        // Memory interfaces do not naturally belong to any block, so they do
        // not get an attribute
        op.setAttr(BB_ATTR_NAME, rewriter.getUI32IntegerAttr(blockID));
      }
    }
  }
}

LogicalResult LowerFuncToHandshake::flattenAndTerminate(
    handshake::FuncOp funcOp, ConversionPatternRewriter &rewriter,
    const ArgReplacements &argReplacements) const {
  // Erase all cf-level terminators, accumulating operands to func-level returns
  // as we go
  SmallVector<SmallVector<Value>> returnsOperands;
  for (Block &block : funcOp) {
    Operation *termOp = &block.back();
    if (auto retOp = dyn_cast<func::ReturnOp>(termOp)) {
      auto &retOperands = returnsOperands.emplace_back();
      if (failed(rewriter.getRemappedValues(retOp->getOperands(), retOperands)))
        return failure();
    }
    rewriter.eraseOp(termOp);
  }
  assert(!returnsOperands.empty() && "function must have at least one return");

  // When identifying basic blocks, the end node is either put in the same
  // block as the function's single return statement or, in the case of
  // multiple return statements, it is put in a "fake block" along with the
  // merges that feed it its data inputs
  size_t exitBlockID = funcOp.getBlocks().size();
  if (returnsOperands.size() == 1)
    exitBlockID -= 1;

  // Inline all non-entry blocks into the entry block, erasing them as we go
  Operation *lastOp = &funcOp.front().back();
  for (Block &block : llvm::make_early_inc_range(funcOp)) {
    if (block.isEntryBlock())
      continue;

    // Replace all block arguments with the data result of merge-like
    // operations; this effectively connects all merges to the rest of the
    // circuit
    SmallVector<Value> replacements;
    for (BlockArgument blockArg : block.getArguments()) {
      Value mergeRes = argReplacements.at(blockArg);
      replacements.push_back(mergeRes);
      rewriter.replaceAllUsesWith(blockArg, mergeRes);
    }
    rewriter.inlineBlockBefore(&block, lastOp, replacements);
  }

  // The terminator's operands are, in order
  // 1. the original function's results
  // 2. a control for each memory region signaling completion
  // 3. a control signaling eventual function completion (the function's start)
  SmallVector<Value, 8> endOprds;
  endOprds.append(
      mergeFuncResults(funcOp, rewriter, returnsOperands, exitBlockID));

  rewriter.setInsertionPointToEnd(funcOp.getBodyBlock());
  for (BlockArgument arg : funcOp.getArguments()) {
    if (!isa<mlir::MemRefType>(arg.getType()))
      continue;

    if (arg.getUsers().empty()) {
      // When the memory region is not accessed, just a create a constant source
      // of valid "memory end" tokens for ir
      auto sourceOp = rewriter.create<handshake::SourceOp>(lastOp->getLoc());
      sourceOp->setAttr(BB_ATTR_NAME, rewriter.getUI32IntegerAttr(exitBlockID));
      endOprds.push_back(sourceOp.getResult());
    } else {
      for (Operation *userOp : arg.getUsers()) {
        auto memOp = cast<handshake::MemoryOpInterface>(userOp);
        if (memOp.isMasterInterface()) {
          endOprds.push_back(memOp.getMemEnd());
          break;
        }
      }
    }
  }
  endOprds.push_back(getBlockControl(funcOp.getBodyBlock()));

  auto endOp = rewriter.create<handshake::EndOp>(lastOp->getLoc(), endOprds);
  endOp->setAttr(BB_ATTR_NAME, rewriter.getUI32IntegerAttr(exitBlockID));
  return success();
}

Value LowerFuncToHandshake::getBlockControl(Block *block) const {
  return block->getArguments().back();
}

//===-----------------------------------------------------------------------==//
// Simple transformations
//===-----------------------------------------------------------------------==//

/// In the operation's parent Handshake function, looks for a control merge
/// tagged with the same basic block as the operation and returns its dara
/// result. The operation must be nested inside a Handshake function and should
/// be tagged with a basic block ID. The control merge is expected to exist; the
/// function will assert if it does not.
static Value getBlockControl(Operation *op) {
  auto funcOp = op->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "operation should have parent function");
  std::optional<unsigned> bb = getLogicBB(op);
  assert(bb && "operation should be tagged with associated basic block");

  if (bb == ENTRY_BB)
    return funcOp.getArguments().back();
  for (auto cMergeOp : funcOp.getOps<handshake::ControlMergeOp>()) {
    if (auto cMergeBB = getLogicBB(cMergeOp); cMergeBB && cMergeBB == *bb)
      return cMergeOp.getResult();
  }
  llvm_unreachable("cannot find cmerge in block");
  return nullptr;
}

namespace {

template <typename SrcOp, typename DstOp>
struct OneToOneConversion : public OpConversionPattern<SrcOp> {
public:
  using OpAdaptor = typename SrcOp::Adaptor;

  OneToOneConversion(NameAnalysis &namer, const TypeConverter &typeConverter,
                     MLIRContext *ctx)
      : OpConversionPattern<SrcOp>(typeConverter, ctx), namer(namer) {}

  LogicalResult
  matchAndRewrite(SrcOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

protected:
  /// Reference to the running pass's naming analysis.
  NameAnalysis &namer;
};

template <typename CastOp, typename ExtOp>
struct ConvertIndexCast : public OpConversionPattern<CastOp> {
public:
  using OpAdaptor = typename CastOp::Adaptor;

  ConvertIndexCast(NameAnalysis &namer, const TypeConverter &typeConverter,
                   MLIRContext *ctx)
      : OpConversionPattern<CastOp>(typeConverter, ctx), namer(namer) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

protected:
  /// Reference to the running pass's naming analysis.
  NameAnalysis &namer;
};

/// Converts each `func::CallOp` operation to an equivalent
/// `handshake::InstanceOp` operation.
struct ConvertCalls : public DynOpConversionPattern<func::CallOp> {
public:
  using DynOpConversionPattern<func::CallOp>::DynOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Convers arith-level constants to handshake-level constants. Constants are
/// triggered by a source if their successor is not a branch/return or memory
/// operation. Otherwise they are triggered by the control-only network.
struct ConvertConstants : public DynOpConversionPattern<arith::ConstantOp> {
public:
  using DynOpConversionPattern<arith::ConstantOp>::DynOpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp cstOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts undefined operations (LLVM::UndefOp) with a default "0" constant
/// triggered by the control merge of the block associated to the matched
/// operation.
struct ConvertUndefinedValues : public DynOpConversionPattern<LLVM::UndefOp> {
public:
  using DynOpConversionPattern<LLVM::UndefOp>::DynOpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::UndefOp undefOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

template <typename SrcOp, typename DstOp>
LogicalResult OneToOneConversion<SrcOp, DstOp>::matchAndRewrite(
    SrcOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  rewriter.setInsertionPoint(srcOp);
  SmallVector<Type> newTypes;
  for (Type resType : srcOp->getResultTypes())
    newTypes.push_back(channelifyType(resType));
  auto newOp =
      rewriter.create<DstOp>(srcOp->getLoc(), newTypes, adaptor.getOperands(),
                             srcOp->getAttrDictionary().getValue());
  namer.replaceOp(srcOp, newOp);
  rewriter.replaceOp(srcOp, newOp);
  return success();
}

template <typename CastOp, typename ExtOp>
LogicalResult ConvertIndexCast<CastOp, ExtOp>::matchAndRewrite(
    CastOp castOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto getWidth = [](Type type) -> unsigned {
    if (isa<IndexType>(type))
      return 32;
    return type.getIntOrFloatBitWidth();
  };

  unsigned srcWidth = getWidth(castOp.getOperand().getType());
  unsigned dstWidth = getWidth(castOp.getResult().getType());
  Type dstType = handshake::ChannelType::get(rewriter.getIntegerType(dstWidth));
  Operation *newOp;
  if (srcWidth < dstWidth) {
    // This is an extension
    newOp =
        rewriter.create<ExtOp>(castOp.getLoc(), dstType, adaptor.getOperands(),
                               castOp->getAttrDictionary().getValue());
  } else {
    // This is a truncation
    newOp = rewriter.create<handshake::TruncIOp>(
        castOp.getLoc(), dstType, adaptor.getOperands(),
        castOp->getAttrDictionary().getValue());
  }
  namer.replaceOp(castOp, newOp);
  rewriter.replaceOp(castOp, newOp);
  return success();
}

LogicalResult
ConvertCalls::matchAndRewrite(func::CallOp callOp, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  auto modOp = callOp->getParentOfType<mlir::ModuleOp>();
  assert(modOp && "call should have parent module");

  // The instance's operands are the same as the call plus an extra
  // control-only start coming from the call's logical basic block
  SmallVector<Value> operands(adaptor.getOperands());
  operands.push_back(getBlockControl(callOp));

  // Retrieve the Handshake function that the call references to determine
  // the instance's result types (may be different from the call's result
  // types)
  SymbolRefAttr symbol = callOp->getAttrOfType<SymbolRefAttr>("callee");
  assert(symbol && "call symbol does not exist");
  Operation *lookup = modOp.lookupSymbol(symbol);
  if (!lookup)
    return callOp->emitError() << "call references unknown function";
  auto calledFuncOp = dyn_cast<handshake::FuncOp>(lookup);
  if (!calledFuncOp)
    return callOp->emitError() << "call does not reference a function";
  TypeRange resultTypes = calledFuncOp.getFunctionType().getResults();

  // Replace the call with the Handshake instance
  rewriter.setInsertionPoint(callOp);
  auto instOp = rewriter.create<handshake::InstanceOp>(
      callOp.getLoc(), callOp.getCallee(), resultTypes, operands);
  instOp->setDialectAttrs(callOp->getDialectAttrs());
  namer.replaceOp(callOp, instOp);
  if (callOp->getNumResults() == 0)
    rewriter.eraseOp(callOp);
  else
    rewriter.replaceOp(callOp, instOp->getResults());
  return success();
}

/// Determines whether it is possible to transform an arith-level constant into
/// a Handshake-level constant that is triggered by an always-triggering source
/// component without compromising the circuit semantics (e.g., without
/// triggering a memory operation before the circuit "starts"). Returns false if
/// the Handshake-level constant that replaces the input must instead be
/// connected to the control-only network; returns true otherwise. This function
/// assumes that the rest of the std-level operations have already been
/// converted to their Handshake equivalent.
/// NOTE: I doubt this works in half-degenerate cases, but this is the logic
/// that legacy Dynamatic follows.
static bool isCstSourcable(arith::ConstantOp cstOp) {
  std::function<bool(Operation *)> isValidUser = [&](Operation *user) -> bool {
    if (isa<UnrealizedConversionCastOp>(user))
      return llvm::all_of(user->getUsers(), isValidUser);
    return !isa<handshake::BranchOp, handshake::ConditionalBranchOp,
                handshake::LoadOpInterface, handshake::StoreOpInterface>(user);
  };

  return llvm::all_of(cstOp->getUsers(), isValidUser);
}

LogicalResult
ConvertConstants::matchAndRewrite(arith::ConstantOp cstOp,
                                  OpAdaptor /*adaptor*/,
                                  ConversionPatternRewriter &rewriter) const {
  rewriter.setInsertionPoint(cstOp);

  // Determine the new constant's control input
  Value controlVal;
  if (isCstSourcable(cstOp)) {
    auto sourceOp = rewriter.create<handshake::SourceOp>(cstOp.getLoc());
    inheritBB(cstOp, sourceOp);
    controlVal = sourceOp.getResult();
  } else {
    controlVal = getBlockControl(cstOp);
  }

  TypedAttr cstAttr = cstOp.getValue();
  // Convert IndexType'd values to equivalent signless integers
  if (isa<IndexType>(cstAttr.getType())) {
    auto intType = rewriter.getIntegerType(32);
    cstAttr = IntegerAttr::get(intType,
                               cast<IntegerAttr>(cstAttr).getValue().trunc(32));
  }
  auto newCstOp = rewriter.create<handshake::ConstantOp>(cstOp.getLoc(),
                                                         cstAttr, controlVal);
  newCstOp->setDialectAttrs(cstOp->getDialectAttrs());
  namer.replaceOp(cstOp, newCstOp);
  rewriter.replaceOp(cstOp, newCstOp->getResults());
  return success();
}

LogicalResult ConvertUndefinedValues::matchAndRewrite(
    LLVM::UndefOp undefOp, OpAdaptor /*adaptor*/,
    ConversionPatternRewriter &rewriter) const {
  // Create an attribute of the appropriate type for the constant
  auto resType = undefOp.getRes().getType();
  TypedAttr cstAttr;
  if (isa<IndexType>(resType)) {
    auto intType = rewriter.getIntegerType(32);
    cstAttr = rewriter.getIntegerAttr(intType, 0);
  } else if (isa<IntegerType>(resType)) {
    cstAttr = rewriter.getIntegerAttr(resType, 0);
  } else if (FloatType floatType = dyn_cast<FloatType>(resType)) {
    cstAttr = rewriter.getFloatAttr(floatType, 0.0);
  } else {
    return undefOp->emitError() << "operation has unsupported result type";
  }

  // Create a constant with a default value and replace the undefined value
  rewriter.setInsertionPoint(undefOp);
  auto cstOp = rewriter.create<handshake::ConstantOp>(undefOp.getLoc(), cstAttr,
                                                      getBlockControl(undefOp));
  cstOp->setDialectAttrs(undefOp->getAttrDictionary());
  namer.replaceOp(cstOp, cstOp);
  rewriter.replaceOp(undefOp, cstOp.getResult());
  return success();
}

//===-----------------------------------------------------------------------==//
// Pass driver
//===-----------------------------------------------------------------------==//

/// Filters out block arguments of type MemRefType
bool FuncSSAStrategy::maximizeArgument(BlockArgument arg) {
  return !arg.getType().isa<mlir::MemRefType>();
}

namespace {

/// FPGA18's elastic pass. Runs elastic pass on every function (func::FuncOp)
/// of the module it is applied on. Succeeds whenever all functions in the
/// module were succesfully lowered to handshake.
struct CfToHandshakePass
    : public dynamatic::impl::CfToHandshakeBase<CfToHandshakePass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    ModuleOp modOp = getOperation();

    // Put all non-external functions into maximal SSA form
    for (auto funcOp : modOp.getOps<func::FuncOp>()) {
      if (!funcOp.isExternal()) {
        FuncSSAStrategy strategy;
        if (failed(dynamatic::maximizeSSA(funcOp.getBody(), strategy)))
          return signalPassFailure();
      }
    }

    CfToHandshakeTypeConverter converter;
    RewritePatternSet patterns(ctx);
    patterns.add<LowerFuncToHandshake, ConvertConstants, ConvertCalls,
                 ConvertUndefinedValues,
                 ConvertIndexCast<arith::IndexCastOp, handshake::ExtSIOp>,
                 ConvertIndexCast<arith::IndexCastUIOp, handshake::ExtUIOp>,
                 OneToOneConversion<arith::AddFOp, handshake::AddFOp>,
                 OneToOneConversion<arith::AddIOp, handshake::AddIOp>,
                 OneToOneConversion<arith::AndIOp, handshake::AndIOp>,
                 OneToOneConversion<arith::CmpFOp, handshake::CmpFOp>,
                 OneToOneConversion<arith::CmpIOp, handshake::CmpIOp>,
                 OneToOneConversion<arith::DivFOp, handshake::DivFOp>,
                 OneToOneConversion<arith::DivSIOp, handshake::DivSIOp>,
                 OneToOneConversion<arith::DivUIOp, handshake::DivUIOp>,
                 OneToOneConversion<arith::ExtSIOp, handshake::ExtSIOp>,
                 OneToOneConversion<arith::ExtUIOp, handshake::ExtUIOp>,
                 OneToOneConversion<arith::MaximumFOp, handshake::MaximumFOp>,
                 OneToOneConversion<arith::MinimumFOp, handshake::MinimumFOp>,
                 OneToOneConversion<arith::MulFOp, handshake::MulFOp>,
                 OneToOneConversion<arith::MulIOp, handshake::MulIOp>,
                 OneToOneConversion<arith::NegFOp, handshake::NegFOp>,
                 OneToOneConversion<arith::OrIOp, handshake::OrIOp>,
                 OneToOneConversion<arith::SelectOp, handshake::SelectOp>,
                 OneToOneConversion<arith::ShLIOp, handshake::ShLIOp>,
                 OneToOneConversion<arith::ShRSIOp, handshake::ShRSIOp>,
                 OneToOneConversion<arith::ShRUIOp, handshake::ShRUIOp>,
                 OneToOneConversion<arith::SubFOp, handshake::SubFOp>,
                 OneToOneConversion<arith::SubIOp, handshake::SubIOp>,
                 OneToOneConversion<arith::TruncIOp, handshake::TruncIOp>,
                 OneToOneConversion<arith::XOrIOp, handshake::XOrIOp>>(
        getAnalysis<NameAnalysis>(), converter, ctx);

    // All func-level functions must become handshake-level functions
    ConversionTarget target(*ctx);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<handshake::HandshakeDialect>();
    target.addIllegalDialect<func::FuncDialect, cf::ControlFlowDialect,
                             arith::ArithDialect, math::MathDialect,
                             BuiltinDialect>();

    if (failed(applyFullConversion(modOp, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass> dynamatic::createCfToHandshake() {
  return std::make_unique<CfToHandshakePass>();
}
