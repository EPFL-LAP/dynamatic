//===- CfToHandshakeTranslator.cpp - Translate func/cf to handshake -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the translation behind the --lower-cf-to-handshake pass, i.e., the
// elastic pass introduced in
// https://dl.acm.org/doi/abs/10.1145/3174243.3174264. Instead of rewriting the
// func-level function in place with the dialect conversion framework, the
// translation creates a fresh Handshake function with an OpBuilder (in the
// style of tools/translate-llvm-to-std):
//
// 1. The Handshake function is created with the lowered signature. During the
//    translation, its body has one staging block per func-level block; the
//    staging blocks are merged into a single block at the end (Handshake
//    functions are graph regions).
// 2. For every non-entry block, merge-like operations are created upfront for
//    each block argument (and for the block's control), with one backedge per
//    incoming CFG edge.
// 3. Blocks are translated in reverse post-order; each operation is
//    translated using a value mapping from func-level to Handshake values.
//    Block terminators create branch-like operations, whose results are
//    recorded per CFG edge and later used to resolve the merges' backedges.
// 4. Memory interfaces are instantiated, the body is flattened, and the
//    function's end is instantiated.
// 5. The func-level function is erased.
//
// The source IR is never modified by the translation (except for the
// signature of placeholder functions, see `translateCall`). Each Handshake
// operation takes the location (i.e., the position in the source code) of the
// func-level operation it replaces, so that diagnostics and debug information
// still refer to the original source code once the func-level function is
// erased.
//
//===----------------------------------------------------------------------===//

#include "CfToHandshakeTranslator.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Transforms/FuncMaximizeSSA.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <map>

using namespace mlir;
using namespace dynamatic;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Converts a func-level type to its Handshake-level equivalent. Returns
/// `nullptr` for unsupported types.
static Type channelifyType(Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case<IndexType, IntegerType, FloatType>(
          [](auto type) { return handshake::ChannelType::get(type); })
      .Case<MemRefType>([](MemRefType memrefType) -> Type {
        if (!isa<IndexType>(memrefType.getElementType()))
          return memrefType;
        IntegerType elemType = IntegerType::get(memrefType.getContext(), 32);
        return MemRefType::get(memrefType.getShape(), elemType);
      })
      .Case<handshake::ChannelType, handshake::ControlType>(
          [](auto type) { return type; })
      .Default([](auto type) { return nullptr; });
}

/// Determines whether a memref type is suitable for conversion.
static bool isValidMemrefType(Location loc, MemRefType type) {
  if (type.getNumDynamicDims() != 0 || type.getShape().size() != 1) {
    emitError(loc) << "memref's must be both statically sized and "
                      "unidimensional.";
    return false;
  }
  return true;
}

/// Returns the Handshake-level function type corresponding to the func-level
/// function. On top of the original arguments and results, the Handshake
/// function has an extra control-only argument and result for each memory
/// region and for the global start/end signals.
static FunctionType lowerFunctionType(func::FuncOp funcOp) {
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 2> resTypes;
  unsigned numMemories = 0;
  for (Type argType : funcOp.getArgumentTypes()) {
    if (isa<MemRefType>(argType))
      ++numMemories;
    argTypes.push_back(channelifyType(argType));
  }
  for (Type resType : funcOp.getResultTypes())
    resTypes.push_back(channelifyType(resType));

  auto ctrlType = handshake::ControlType::get(funcOp.getContext());
  argTypes.append(numMemories + 1, ctrlType);
  resTypes.append(numMemories + 1, ctrlType);
  return FunctionType::get(funcOp.getContext(), argTypes, resTypes);
}

/// Produces the list of named attributes of the Handshake function
/// corresponding to the func-level function (argument and result names in
/// particular).
static SmallVector<NamedAttribute> deriveNewAttributes(func::FuncOp funcOp) {
  SmallVector<NamedAttribute> attributes;
  MLIRContext *ctx = funcOp.getContext();
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
      SmallVector<Attribute> argNames;
      for (auto [argType, argAttr] :
           llvm::zip(funcOp.getArgumentTypes(), funcOp.getArgAttrsAttr())) {
        DictionaryAttr argDict = cast<DictionaryAttr>(argAttr);
        std::optional<NamedAttribute> name =
            argDict.getNamed("handshake.arg_name");
        assert(name && "missing name key in arg attribute");
        if (isa<MemRefType>(argType))
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

    attributes.push_back(attr);
  }

  // Regular function results have default names
  SmallVector<Attribute> resNames;
  unsigned numFuncResults = funcOp.getFunctionType().getNumResults();
  for (size_t idx = 0; idx < numFuncResults; ++idx)
    resNames.push_back(StringAttr::get(ctx, "out" + std::to_string(idx)));

  if (!hasArgNames) {
    SmallVector<Attribute> argNames;
    unsigned argNum = 0, memNum = 0;
    for (BlockArgument arg : funcOp.getArguments()) {
      if (isa<MemRefType>(arg.getType())) {
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

/// A constant can be triggered by a source if none of its users becomes a
/// branch-like or memory port operation.
bool CfToHandshakeTranslator::isCstSourcable(arith::ConstantOp cstOp) {
  return llvm::none_of(cstOp->getUsers(), [](Operation *user) {
    return isa<cf::BranchOp, cf::CondBranchOp, memref::LoadOp, memref::StoreOp>(
        user);
  });
}

/// Returns a dense elements attribute of the shaped type where every element is
/// zero. Used as the initial contents of RAMs replacing allocas.
static DenseElementsAttr getZeroFilledDenseAttr(ShapedType type) {
  Type elemType = type.getElementType();
  if (auto intTy = dyn_cast<IntegerType>(elemType))
    return DenseElementsAttr::get(type, APInt(intTy.getWidth(), 0));
  if (auto floatTy = dyn_cast<FloatType>(elemType))
    return DenseElementsAttr::get(
        type, APFloat::getZero(floatTy.getFloatSemantics(), false));
  llvm::report_fatal_error("Unknown base element type!");
}

SmallVector<Value> CfToHandshakeTranslator::lookup(ValueRange values) {
  return llvm::map_to_vector(values,
                             [&](Value val) { return valueMap.lookup(val); });
}

void CfToHandshakeTranslator::transferName(Operation *cfOp, Operation *newOp) {
  if (namer.hasName(cfOp))
    namer.replaceOp(cfOp, newOp);
}

void CfToHandshakeTranslator::setBlock(Operation *op, Block *block) {
  setBB(op, blockID.at(block));
}

LogicalResult CfToHandshakeTranslator::translate() {
  // Create a Handshake-level function corresponding to the func-level one
  builder.setInsertionPoint(cfFunc);
  funcOp = builder.create<handshake::FuncOp>(cfFunc.getLoc(), cfFunc.getName(),
                                             lowerFunctionType(cfFunc),
                                             deriveNewAttributes(cfFunc));
  if (cfFunc.isExternal())
    return success();

  if (failed(translateBody())) {
    // The translation stopped midway, so some backedges may still be in use;
    // only report the error that caused the failure
    edgeBuilder.abandon();
    return failure();
  }
  return edgeBuilder.clearOrEmitError();
}

LogicalResult CfToHandshakeTranslator::translateBody() {
  // The arguments of the Handshake function's entry block are the original
  // function's arguments followed by the memory start signals and the global
  // start signal
  Block *body = funcOp.addEntryBlock();
  Block *cfEntry = &cfFunc.front();
  for (BlockArgument arg : cfEntry->getArguments())
    valueMap.map(arg, body->getArgument(arg.getArgNumber()));
  blockCtrl[cfEntry] = body->getArguments().back();

  // The entry block is the entry's staging block
  for (auto [idx, block] : llvm::enumerate(cfFunc)) {
    blockID[&block] = idx;
    if (block.isEntryBlock()) {
      stagingBlock[&block] = body;
    } else {
      stagingBlock[&block] = new Block();
      funcOp.getBody().push_back(stagingBlock[&block]);
    }
  }
  computePredecessors();
  if (failed(initMemories()) || failed(initTranslation()))
    return failure();

  // Translate blocks in reverse post-order so that values are translated
  // before their uses (unreachable blocks go last)
  SetVector<Block *> order;
  for (Block *block : llvm::ReversePostOrderTraversal<Block *>(cfEntry))
    order.insert(block);
  for (Block &block : cfFunc)
    order.insert(&block);

  for (Block *block : order) {
    if (failed(translateBlock(block)))
      return failure();
  }
  resolveMerges();

  if (failed(createMemInterfaces()))
    return failure();
  finalizeBlocks();
  flattenBody();
  createEnd();
  return finalizeTranslation();
}

void CfToHandshakeTranslator::flattenBody() {
  Block *body = &funcOp.getBody().front();
  for (Block &block :
       llvm::make_early_inc_range(llvm::drop_begin(funcOp.getBody()))) {
    body->getOperations().splice(body->end(), block.getOperations());
    block.erase();
  }
}

void CfToHandshakeTranslator::computePredecessors() {
  // Each successor of a terminator is an incoming edge of the successor block.
  // If a conditional branch's two successors are the same block, that block
  // has two incoming edges from the branch's block. The edges are ordered by
  // predecessor block, then by successor index (rather than in the order of
  // the block's uses), which determines the order of the merges' operands
  for (Block &block : cfFunc) {
    for (auto [succIdx, succ] : llvm::enumerate(block.getSuccessors()))
      preds[succ].emplace_back(&block, succIdx);
  }
}

LogicalResult CfToHandshakeTranslator::initMemories() {
  // Record memories passed as function arguments, even if they aren't used by
  // any memory operation. Each one has a dedicated start signal in the function
  // arguments
  auto funcArgs = funcOp.getArguments();
  unsigned numMemories = llvm::count_if(
      funcArgs, [](auto arg) { return isa<MemRefType>(arg.getType()); });
  unsigned memStartIdx = funcArgs.size() - numMemories - 1;
  for (BlockArgument arg : cfFunc.getArguments()) {
    if (auto memrefTy = dyn_cast<MemRefType>(arg.getType())) {
      if (!isValidMemrefType(arg.getLoc(), memrefTy))
        return failure();
      memInfo.insert({arg,
                      {/*mcPorts=*/{}, /*lsqPorts=*/{},
                       /*memStart=*/funcArgs[memStartIdx++]}});
    }
  }

  // Internally allocated memories use the function's start as their memory
  // start signal
  Value start = funcArgs.back();
  cfFunc.walk([&](memref::AllocaOp op) {
    memInfo.insert({op.getResult(),
                    {/*mcPorts=*/{}, /*lsqPorts=*/{}, /*memStart=*/start}});
  });
  cfFunc.walk([&](memref::GetGlobalOp op) {
    memInfo.insert({op.getResult(),
                    {/*mcPorts=*/{}, /*lsqPorts=*/{}, /*memStart=*/start}});
  });
  return success();
}

/// Creates a placeholder operand for each incoming edge, replaced by the value
/// flowing along the edge once all predecessor terminators are translated.
static SmallVector<Value> makeMergeOperands(BackedgeBuilder &edgeBuilder,
                                            unsigned numPreds, Type type) {
  SmallVector<Value> operands;
  for (unsigned i = 0; i < numPreds; ++i)
    operands.push_back(edgeBuilder.get(type));
  return operands;
}

handshake::ControlMergeOp
CfToHandshakeTranslator::createControlMerge(Block *block) {
  // Every block needs to feed its entry control into a control merge
  auto ctrlType = handshake::ControlType::get(builder.getContext());
  auto cmergeOp = builder.create<handshake::ControlMergeOp>(
      block->front().getLoc(),
      makeMergeOperands(edgeBuilder, preds[block].size(), ctrlType));
  setBlock(cmergeOp, block);
  blockCtrl[block] = cmergeOp.getResult();
  mergeOps[block].push_back(cmergeOp);
  return cmergeOp;
}

LogicalResult CfToHandshakeTranslator::createMerges(Block *block) {
  ArrayRef<CFGEdge> blockPreds = preds[block];
  if (blockPreds.empty())
    return block->front().emitError() << "block must have predecessors";
  Location loc = block->front().getLoc();
  SmallVector<handshake::MergeLikeOpInterface> &ops = mergeOps[block];
  auto makeOperands = [&](Type type) {
    return makeMergeOperands(edgeBuilder, blockPreds.size(), type);
  };

  // Every live-in value to a non-entry block is passed through a merge-like
  // operation, even when it's not required for circuit correctness (useless
  // merge-like operations are removed down the line during Handshake
  // canonicalization)
  SmallVector<handshake::MuxOp> muxes;
  for (BlockArgument arg : block->getArguments()) {
    Type type = channelifyType(arg.getType());
    if (!isa_and_present<handshake::ChannelType>(type))
      return emitError(arg.getLoc()) << "unsupported block argument type";

    handshake::MergeLikeOpInterface mergeOp;
    if (blockPreds.size() == 1) {
      mergeOp = builder.create<handshake::MergeOp>(loc, makeOperands(type));
    } else {
      // The select operand is the block's control merge index output, which is
      // created last
      Type idxType =
          handshake::getOptimizedIndexValType(builder, blockPreds.size());
      Value index = edgeBuilder.get(handshake::ChannelType::get(idxType));
      mergeOp = muxes.emplace_back(builder.create<handshake::MuxOp>(
          loc, type, index, makeOperands(type)));
    }
    setBlock(mergeOp, block);
    valueMap.map(arg, mergeOp.getDataResult());
    ops.push_back(mergeOp);
  }

  handshake::ControlMergeOp cmergeOp = createControlMerge(block);
  for (handshake::MuxOp mux : muxes)
    mux.getSelectOperandMutable().assign(cmergeOp.getIndex());
  return success();
}

LogicalResult CfToHandshakeTranslator::translateBlock(Block *block) {
  // Hooks may move the insertion point (e.g., to build circuits at the start
  // of a block), so it is reset before each operation
  Block *staging = stagingBlock[block];
  builder.setInsertionPointToEnd(staging);
  if (!block->isEntryBlock() && failed(createMerges(block)))
    return failure();
  for (Operation &op : block->without_terminator()) {
    builder.setInsertionPointToEnd(staging);
    if (failed(translateOp(&op)))
      return failure();
  }
  builder.setInsertionPointToEnd(staging);
  return translateTerminator(block->getTerminator());
}

LogicalResult CfToHandshakeTranslator::translateOp(Operation *op) {
  // clang-format off
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<arith::ConstantOp>([&](auto op) { return translateConstant(op); })
      .Case<LLVM::UndefOp>([&](auto op) { return translateUndef(op); })
      .Case<memref::LoadOp>([&](auto op) { return translateLoad(op); })
      .Case<memref::StoreOp>([&](auto op) { return translateStore(op); })
      .Case<memref::AllocaOp>([&](auto op) { return translateAlloca(op); })
      .Case<memref::GetGlobalOp>([&](auto op) { return translateGetGlobal(op); })
      .Case<func::CallOp>([&](auto op) { return translateCall(op); })
      .Case<arith::IndexCastOp>([&](auto op) { return translateIndexCast<handshake::ExtSIOp>(op); })
      .Case<arith::IndexCastUIOp>([&](auto op) { return translateIndexCast<handshake::ExtUIOp>(op); })
      .Case<arith::AddFOp>([&](auto op) { return translateOneToOne<handshake::AddFOp>(op); })
      .Case<arith::AddIOp>([&](auto op) { return translateOneToOne<handshake::AddIOp>(op); })
      .Case<arith::AndIOp>([&](auto op) { return translateOneToOne<handshake::AndIOp>(op); })
      .Case<arith::CmpFOp>([&](auto op) { return translateOneToOne<handshake::CmpFOp>(op); })
      .Case<arith::CmpIOp>([&](auto op) { return translateOneToOne<handshake::CmpIOp>(op); })
      .Case<arith::DivFOp>([&](auto op) { return translateOneToOne<handshake::DivFOp>(op); })
      .Case<arith::DivSIOp>([&](auto op) { return translateOneToOne<handshake::DivSIOp>(op); })
      .Case<arith::DivUIOp>([&](auto op) { return translateOneToOne<handshake::DivUIOp>(op); })
      .Case<arith::RemSIOp>([&](auto op) { return translateOneToOne<handshake::RemSIOp>(op); })
      .Case<arith::RemUIOp>([&](auto op) { return translateOneToOne<handshake::RemUIOp>(op); })
      .Case<arith::ExtSIOp>([&](auto op) { return translateOneToOne<handshake::ExtSIOp>(op); })
      .Case<arith::ExtUIOp>([&](auto op) { return translateOneToOne<handshake::ExtUIOp>(op); })
      .Case<arith::MaximumFOp>([&](auto op) { return translateOneToOne<handshake::MaximumFOp>(op); })
      .Case<arith::MinimumFOp>([&](auto op) { return translateOneToOne<handshake::MinimumFOp>(op); })
      .Case<arith::MaxSIOp>([&](auto op) { return translateOneToOne<handshake::MaxSIOp>(op); })
      .Case<arith::MaxUIOp>([&](auto op) { return translateOneToOne<handshake::MaxUIOp>(op); })
      .Case<arith::MinSIOp>([&](auto op) { return translateOneToOne<handshake::MinSIOp>(op); })
      .Case<arith::MinUIOp>([&](auto op) { return translateOneToOne<handshake::MinUIOp>(op); })
      .Case<arith::MulFOp>([&](auto op) { return translateOneToOne<handshake::MulFOp>(op); })
      .Case<arith::MulIOp>([&](auto op) { return translateOneToOne<handshake::MulIOp>(op); })
      .Case<arith::NegFOp>([&](auto op) { return translateOneToOne<handshake::NegFOp>(op); })
      .Case<arith::OrIOp>([&](auto op) { return translateOneToOne<handshake::OrIOp>(op); })
      .Case<arith::SelectOp>([&](auto op) { return translateOneToOne<handshake::SelectOp>(op); })
      .Case<arith::ShLIOp>([&](auto op) { return translateOneToOne<handshake::ShLIOp>(op); })
      .Case<arith::ShRSIOp>([&](auto op) { return translateOneToOne<handshake::ShRSIOp>(op); })
      .Case<arith::ShRUIOp>([&](auto op) { return translateOneToOne<handshake::ShRUIOp>(op); })
      .Case<arith::SubFOp>([&](auto op) { return translateOneToOne<handshake::SubFOp>(op); })
      .Case<arith::SubIOp>([&](auto op) { return translateOneToOne<handshake::SubIOp>(op); })
      .Case<arith::TruncIOp>([&](auto op) { return translateOneToOne<handshake::TruncIOp>(op); })
      .Case<arith::TruncFOp>([&](auto op) { return translateOneToOne<handshake::TruncFOp>(op); })
      .Case<arith::XOrIOp>([&](auto op) { return translateOneToOne<handshake::XOrIOp>(op); })
      .Case<arith::SIToFPOp>([&](auto op) { return translateOneToOne<handshake::SIToFPOp>(op); })
      .Case<arith::UIToFPOp>([&](auto op) { return translateOneToOne<handshake::UIToFPOp>(op); })
      .Case<arith::FPToSIOp>([&](auto op) { return translateOneToOne<handshake::FPToSIOp>(op); })
      .Case<arith::ExtFOp>([&](auto op) { return translateOneToOne<handshake::ExtFOp>(op); })
      .Case<math::AbsFOp>([&](auto op) { return translateOneToOne<handshake::AbsFOp>(op); })
      .Case<math::AbsIOp>([&](auto op) { return translateOneToOne<handshake::AbsIOp>(op); })
      .Default([&](Operation *op) { return translateGeneric(op); });
  // clang-format on
}

LogicalResult CfToHandshakeTranslator::translateTerminator(Operation *termOp) {
  if (!isa<cf::BranchOp, cf::CondBranchOp, func::ReturnOp>(termOp))
    return termOp->emitError() << "unsupported block terminator";
  Block *block = termOp->getBlock();
  if (isa<func::ReturnOp>(termOp)) {
    returns.emplace_back(blockID[block], lookup(termOp->getOperands()));
    return success();
  }

  // The values flowing along each outgoing edge: the values forwarded to the
  // successor's block arguments, followed by the block's control. Edges are
  // identified by the successor's index in the terminator rather than by the
  // successor block, since both successors of a conditional branch can be the
  // same block. For example, in the base translation,
  //   ^bb0: cf.cond_br %c, ^bb1(%x : i32), ^bb2(%y, %x : i32, i32)
  // maps successor 0 (^bb1) to [%x, ctrl] and successor 1 (^bb2) to
  // [%y, %x, ctrl], where ctrl is ^bb0's control
  std::map<unsigned, SmallVector<Value>> valuesPerSuccIdx;
  for (unsigned succIdx = 0, e = termOp->getNumSuccessors(); succIdx < e;
       ++succIdx) {
    SmallVector<Value> &values = valuesPerSuccIdx[succIdx];
    values = getForwardedValues(termOp, succIdx);
    values.push_back(blockCtrl[block]);
  }

  // Insert a branch-like operation for each unique value flowing out of the
  // block
  Value cond;
  if (auto condBrOp = dyn_cast<cf::CondBranchOp>(termOp))
    cond = getBranchCondition(condBrOp);
  Location loc = termOp->getLoc();
  DenseMap<Value, Operation *> branches;
  for (auto &[_, values] : valuesPerSuccIdx) {
    for (Value val : values) {
      if (branches.contains(val))
        continue;
      Operation *brOp;
      if (cond)
        brOp = builder.create<handshake::ConditionalBranchOp>(loc, cond, val);
      else
        brOp = builder.create<handshake::BranchOp>(loc, val);
      setBlock(brOp, block);
      branches[val] = brOp;
    }
  }

  // Record the branch-like operations' results flowing along each edge (the
  // successor index determines the conditional branch's result)
  for (auto &[succIdx, values] : valuesPerSuccIdx) {
    SmallVector<Value> &outValues = edgeValues[{block, succIdx}];
    for (Value val : values)
      outValues.push_back(branches[val]->getResult(succIdx));
  }
  return success();
}

SmallVector<Value>
CfToHandshakeTranslator::getForwardedValues(Operation *termOp,
                                            unsigned succIdx) {
  auto succOperands =
      cast<BranchOpInterface>(termOp).getSuccessorOperands(succIdx);
  return lookup(succOperands.getForwardedOperands());
}

void CfToHandshakeTranslator::resolveMerges() {
  // The placeholder operands become unused and are erased with the other
  // backedges
  for (auto &[block, ops] : mergeOps) {
    for (auto [edgeIdx, edge] : llvm::enumerate(preds[block])) {
      for (auto [mergeOp, val] : llvm::zip_equal(ops, edgeValues.at(edge)))
        mergeOp.setDataOperand(edgeIdx, val);
    }
  }
}

template <typename DstOp>
LogicalResult CfToHandshakeTranslator::translateOneToOne(Operation *op) {
  SmallVector<Type> resTypes;
  for (Type resType : op->getResultTypes())
    resTypes.push_back(channelifyType(resType));
  // Copy the discardable attributes, and the inherent attributes that the
  // Handshake operation also has (e.g., the predicate of comparisons). Other
  // inherent attributes (e.g., overflow flags) have no meaning in Handshake
  SmallVector<NamedAttribute> attrs(op->getDiscardableAttrs());
  for (StringRef name : DstOp::getAttributeNames()) {
    if (std::optional<Attribute> attr = op->getInherentAttr(name);
        attr && *attr)
      attrs.emplace_back(builder.getStringAttr(name), *attr);
  }
  auto newOp = builder.create<DstOp>(op->getLoc(), resTypes,
                                     lookup(op->getOperands()), attrs);
  setBlock(newOp, op->getBlock());
  transferName(op, newOp);
  valueMap.map(op->getResults(), newOp->getResults());
  return success();
}

template <typename ExtOp>
LogicalResult CfToHandshakeTranslator::translateIndexCast(Operation *op) {
  auto getWidth = [](Type type) -> unsigned {
    if (isa<IndexType>(type))
      return 32;
    return type.getIntOrFloatBitWidth();
  };
  unsigned srcWidth = getWidth(op->getOperand(0).getType());
  unsigned dstWidth = getWidth(op->getResult(0).getType());
  Type dstType = handshake::ChannelType::get(builder.getIntegerType(dstWidth));

  SmallVector<Value> operands = lookup(op->getOperands());
  ArrayRef<NamedAttribute> attrs = op->getAttrDictionary().getValue();
  Operation *newOp;
  if (srcWidth < dstWidth) {
    newOp = builder.create<ExtOp>(op->getLoc(), dstType, operands, attrs);
  } else {
    newOp = builder.create<handshake::TruncIOp>(op->getLoc(), dstType, operands,
                                                attrs);
  }
  setBlock(newOp, op->getBlock());
  transferName(op, newOp);
  valueMap.map(op->getResult(0), newOp->getResult(0));
  return success();
}

LogicalResult
CfToHandshakeTranslator::translateConstant(arith::ConstantOp cstOp) {
  // Constants are triggered by a source if possible, otherwise by the block's
  // trigger
  Block *block = cstOp->getBlock();
  Value ctrl;
  if (isCstSourcable(cstOp)) {
    auto sourceOp = builder.create<handshake::SourceOp>(cstOp.getLoc());
    setBlock(sourceOp, block);
    ctrl = sourceOp.getResult();
  } else {
    ctrl = getBlockControlChannel(block);
  }

  // Convert IndexType'd values to equivalent signless integers
  TypedAttr cstAttr = cstOp.getValue();
  if (isa<IndexType>(cstAttr.getType())) {
    cstAttr = IntegerAttr::get(builder.getIntegerType(32),
                               cast<IntegerAttr>(cstAttr).getValue().trunc(32));
  }
  auto newCstOp =
      builder.create<handshake::ConstantOp>(cstOp.getLoc(), cstAttr, ctrl);
  newCstOp->setDialectAttrs(cstOp->getDialectAttrs());
  setBlock(newCstOp, block);
  transferName(cstOp, newCstOp);
  valueMap.map(cstOp.getResult(), newCstOp.getResult());
  return success();
}

LogicalResult CfToHandshakeTranslator::translateUndef(LLVM::UndefOp undefOp) {
  // Undefined values become a "0" constant triggered by the block's trigger
  Type resType = undefOp.getRes().getType();
  TypedAttr cstAttr;
  if (isa<IndexType>(resType)) {
    cstAttr = builder.getIntegerAttr(builder.getIntegerType(32), 0);
  } else if (isa<IntegerType>(resType)) {
    cstAttr = builder.getIntegerAttr(resType, 0);
  } else if (FloatType floatType = dyn_cast<FloatType>(resType)) {
    cstAttr = builder.getFloatAttr(floatType, 0.0);
  } else {
    return undefOp->emitError() << "operation has unsupported result type";
  }

  Block *block = undefOp->getBlock();
  auto cstOp = builder.create<handshake::ConstantOp>(
      undefOp.getLoc(), cstAttr, getBlockControlChannel(block));
  cstOp->setDialectAttrs(undefOp->getAttrDictionary());
  setBlock(cstOp, block);
  transferName(undefOp, cstOp);
  valueMap.map(undefOp.getRes(), cstOp.getResult());
  return success();
}

LogicalResult
CfToHandshakeTranslator::addMemPort(Operation *cfOp, Value memref,
                                    handshake::MemPortOpInterface portOp) {
  if (!memInfo.count(memref))
    return cfOp->emitError() << "unknown memory region";
  memPorts.push_back({cfOp, memref, portOp});
  return success();
}

LogicalResult CfToHandshakeTranslator::translateLoad(memref::LoadOp loadOp) {
  if (!getDialectAttr<handshake::MemInterfaceAttr>(loadOp)) {
    return loadOp->emitError()
           << "memory operation must have attribute of type "
              "'handshake::MemInterfaceAttr' to encode which memory interface "
              "it should connect to.";
  }
  OperandRange indices = loadOp.getIndices();
  if (indices.size() != 1)
    return loadOp->emitError() << "load must be unidimensional";

  // The data input comes from the memory interface, which is created later
  Type dataTy = channelifyType(loadOp.getMemRefType().getElementType());
  Value data = edgeBuilder.get(dataTy);
  auto newOp = builder.create<handshake::LoadOp>(
      loadOp.getLoc(), valueMap.lookup(indices.front()), data);
  copyDialectAttr<handshake::MemDependenceArrayAttr>(loadOp, newOp);
  setBlock(newOp, loadOp->getBlock());
  transferName(loadOp, newOp);
  valueMap.map(loadOp.getResult(), newOp.getDataResult());
  return addMemPort(loadOp, loadOp.getMemRef(), newOp);
}

LogicalResult CfToHandshakeTranslator::translateStore(memref::StoreOp storeOp) {
  if (!getDialectAttr<handshake::MemInterfaceAttr>(storeOp)) {
    return storeOp->emitError()
           << "memory operation must have attribute of type "
              "'handshake::MemInterfaceAttr' to encode which memory interface "
              "it should connect to.";
  }
  OperandRange indices = storeOp.getIndices();
  if (indices.size() != 1)
    return storeOp->emitError() << "store must be unidimensional";

  auto newOp = builder.create<handshake::StoreOp>(
      storeOp.getLoc(), valueMap.lookup(indices.front()),
      valueMap.lookup(storeOp.getValueToStore()));
  copyDialectAttr<handshake::MemDependenceArrayAttr>(storeOp, newOp);
  setBlock(newOp, storeOp->getBlock());
  transferName(storeOp, newOp);
  return addMemPort(storeOp, storeOp.getMemRef(), newOp);
}

LogicalResult
CfToHandshakeTranslator::translateAlloca(memref::AllocaOp allocaOp) {
  // HACK: By default, we initialize the memory with all zeros. According to the
  // C standard, this only happens for arrays.
  MemRefType type = allocaOp.getType();
  auto ramOp = builder.create<handshake::RAMOp>(allocaOp.getLoc(), type,
                                                getZeroFilledDenseAttr(type));
  valueMap.map(allocaOp.getResult(), ramOp.getResult());
  return success();
}

LogicalResult
CfToHandshakeTranslator::translateGetGlobal(memref::GetGlobalOp getGlobalOp) {
  // The global's initial value is attached to the RAM replacing the global
  auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
      getGlobalOp, getGlobalOp.getNameAttr());
  if (!global)
    return getGlobalOp->emitError() << "reference to unknown global";
  auto denseAttr =
      dyn_cast_if_present<DenseElementsAttr>(global.getInitialValueAttr());
  if (!denseAttr) {
    return global->emitError()
           << "the initial value must be denoted in DenseElementsAttr";
  }
  auto ramOp = builder.create<handshake::RAMOp>(
      getGlobalOp.getLoc(), getGlobalOp.getType(), denseAttr);
  valueMap.map(getGlobalOp.getResult(), ramOp.getResult());
  return success();
}

/// Returns the function called by the call operation if it is an __init
/// function (used to declare the outputs of placeholder functions).
static bool isInitCall(func::CallOp callOp) {
  auto calledFn = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr()));
  return calledFn && calledFn.getSymName().starts_with("__init");
}

/// Checks that the results of an __init call, which declare the outputs of a
/// placeholder function, are first used by a (placeholder) call in the same
/// block. Translating that call maps them to the placeholder's outputs, so
/// they cannot be used before.
static LogicalResult verifyInitCall(func::CallOp initCallOp) {
  Block *block = initCallOp->getBlock();
  for (Value res : initCallOp.getResults()) {
    Operation *firstUser = nullptr;
    for (Operation *user : res.getUsers()) {
      if (user->getBlock() != block)
        return user->emitError() << "uses the result of an __init call "
                                    "defined in another block";
      if (!firstUser || user->isBeforeInBlock(firstUser))
        firstUser = user;
    }
    if (!isa_and_present<func::CallOp>(firstUser)) {
      return initCallOp->emitError()
             << "result of __init call must first be passed to a placeholder "
                "call";
    }
  }
  return success();
}

LogicalResult CfToHandshakeTranslator::translateCall(func::CallOp callOp) {
  // __init calls are translated along with the placeholder call that uses them
  if (isInitCall(callOp))
    return verifyInitCall(callOp);

  Operation *callee =
      SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr());
  if (!callee)
    return callOp->emitError() << "call references unknown function";

  Block *block = callOp->getBlock();
  SmallVector<Type> resultTypes;
  SmallVector<Value> operands;
  // Indices of the placeholder's arguments that are outputs, in order
  SmallVector<unsigned> outputIndices;
  // Parameter name to value
  SmallVector<NamedAttribute> parameters;

  if (auto calledHandshakeFuncOp = dyn_cast<handshake::FuncOp>(callee)) {
    // Regular call to an already translated function
    operands = lookup(callOp.getOperands());
    llvm::append_range(resultTypes,
                       calledHandshakeFuncOp.getFunctionType().getResults());
  } else {
    // The callee is a placeholder function whose arguments are inputs, outputs,
    // or parameters, as indicated by their name
    auto calledFuncOp = dyn_cast<func::FuncOp>(callee);
    if (!calledFuncOp)
      return callOp->emitError() << "call does not reference a function";

    FunctionType calledType = calledFuncOp.getFunctionType();
    SmallVector<Type> newInputs, newResults;
    for (unsigned i = 0, e = calledFuncOp.getNumArguments(); i < e; ++i) {
      auto nameAttr =
          calledFuncOp.getArgAttrOfType<StringAttr>(i, "handshake.arg_name");
      if (!nameAttr || nameAttr.getValue().empty()) {
        return callOp->emitError()
               << "placeholder argument " << i << " has no name";
      }
      StringRef name = nameAttr.getValue();
      Value operand = callOp.getOperand(i);
      if (name.starts_with("input_")) {
        auto initCallOp = operand.getDefiningOp<func::CallOp>();
        if (initCallOp && isInitCall(initCallOp)) {
          return callOp->emitError()
                 << "placeholder input " << i
                 << " cannot be initialized by an __init call";
        }
        newInputs.push_back(calledType.getInput(i));
        operands.push_back(valueMap.lookup(operand));
      } else if (name.starts_with("output_")) {
        newResults.push_back(calledType.getInput(i));
        outputIndices.push_back(i);
        auto initCallOp = operand.getDefiningOp<func::CallOp>();
        if (!initCallOp || !isInitCall(initCallOp)) {
          return callOp->emitError() << "placeholder output " << i
                                     << " must be initialized by an __init "
                                        "call";
        }
      } else if (name.starts_with("parameter_")) {
        auto cstOp = operand.getDefiningOp<arith::ConstantOp>();
        if (!cstOp) {
          return callOp->emitError()
                 << "placeholder parameter " << i << " must be a constant";
        }
        parameters.emplace_back(
            builder.getStringAttr(name.drop_front(strlen("parameter_"))),
            cstOp.getValue());
      } else {
        return callOp->emitError() << "placeholder argument " << i
                                   << " does not follow the naming convention";
      }
    }
    if (outputIndices.empty()) {
      return callOp->emitError()
             << "placeholder functions must have at least one output_ "
                "argument";
    }

    // The placeholder's signature only keeps its inputs, and its outputs
    // become results
    calledFuncOp.setType(
        FunctionType::get(builder.getContext(), newInputs, newResults));
    resultTypes = newResults;
  }

  // The instance has an extra control-only start input coming from the call's
  // block, and an extra control-only end output
  operands.push_back(blockCtrl[block]);
  for (Type &type : resultTypes)
    type = channelifyType(type);
  resultTypes.push_back(handshake::ControlType::get(builder.getContext()));

  auto instOp = builder.create<handshake::InstanceOp>(
      callOp.getLoc(), callOp.getCallee(), resultTypes, operands);
  instOp->setDialectAttrs(callOp->getDialectAttrs());
  for (NamedAttribute param : parameters)
    instOp->setAttr(param.getName(), param.getValue());
  setBlock(instOp, block);
  transferName(callOp, instOp);

  if (outputIndices.empty()) {
    valueMap.map(callOp.getResults(),
                 instOp.getResults().take_front(callOp.getNumResults()));
    return success();
  }

  // All users of the values produced by the __init calls now use the outputs
  // of the instance
  for (auto [resIdx, argIdx] : llvm::enumerate(outputIndices))
    valueMap.map(callOp.getOperand(argIdx), instOp.getResult(resIdx));
  if (callOp.getNumResults() != 0)
    valueMap.map(callOp.getResult(0), instOp.getResult(0));
  return success();
}

LogicalResult CfToHandshakeTranslator::translateGeneric(Operation *op) {
  // Operations from dialects that must be fully lowered cannot be kept as is
  if (isa<arith::ArithDialect, math::MathDialect, func::FuncDialect,
          cf::ControlFlowDialect, BuiltinDialect>(op->getDialect()))
    return op->emitError() << "unsupported operation";

  Operation *newOp = builder.clone(*op, valueMap);
  setBlock(newOp, op->getBlock());
  return success();
}

LogicalResult CfToHandshakeTranslator::createMemInterfaces() {
  if (memInfo.empty())
    return success();

  // Associate each memory port with the interface it should connect to. Ports
  // are considered in program order (blocks were translated in reverse
  // post-order), which determines their order in the memory interfaces
  llvm::stable_sort(memPorts, [&](const MemPort &lhs, const MemPort &rhs) {
    return blockID[lhs.cfOp->getBlock()] < blockID[rhs.cfOp->getBlock()];
  });
  for (MemPort &port : memPorts) {
    auto memAttr = getDialectAttr<handshake::MemInterfaceAttr>(port.cfOp);
    Block *block = port.cfOp->getBlock();
    portBlock[port.portOp] = block;
    MemAccesses &accesses = memInfo.find(port.memref)->second;
    if (memAttr.connectsToMC())
      accesses.mcPorts[block].push_back(port.portOp);
    else
      accesses.lsqPorts[*memAttr.getLsqGroup()].push_back(port.portOp);
  }

  // Find the control value indicating the last control flow decision in the
  // function; it will be fed to memory interfaces to indicate that no more
  // group allocations will be coming
  assert(!returns.empty() && "no returns in function");
  SmallVector<Block *> retBlocks;
  for (Block &block : cfFunc) {
    if (isa<func::ReturnOp>(block.getTerminator()))
      retBlocks.push_back(&block);
  }
  Value ctrlEnd;
  if (retBlocks.size() == 1) {
    ctrlEnd = blockCtrl[retBlocks.front()];
  } else {
    // Merge the control signals of all blocks with a return, at the beginning
    // of the last of these blocks
    SmallVector<Value> controls;
    for (Block *block : retBlocks)
      controls.push_back(blockCtrl[block]);
    Block *lastRetBlock = retBlocks.back();
    builder.setInsertionPointToStart(stagingBlock[lastRetBlock]);
    auto mergeOp = builder.create<handshake::MergeOp>(
        lastRetBlock->getTerminator()->getLoc(), controls);
    setBB(mergeOp, blockID[lastRetBlock]);
    ctrlEnd = mergeOp.getResult();
  }

  // Create a mapping between each block and its control value in the right
  // format for the memory interface builder
  DenseMap<unsigned, Value> ctrlVals;
  for (auto [block, id] : blockID)
    ctrlVals.insert({id, blockCtrl[block]});

  // Each memory region is independent from the others
  for (auto &[memref, memAccesses] : memInfo) {
#ifndef NDEBUG
    SmallPtrSet<Block *, 4> controlBlocks;
#endif
    MemoryInterfaceBuilder memBuilder(funcOp, valueMap.lookup(memref),
                                      memAccesses.memStart, ctrlEnd, ctrlVals);

    for (auto &[_, mcBlockOps] : memAccesses.mcPorts)
      for (handshake::MemPortOpInterface portOp : mcBlockOps)
        memBuilder.addMCPort(portOp);

    // An LSQ group is allocated by the control signal of a single block, so
    // all of its ports must belong to that block. Ports are already in program
    // order, which determines their order in the group
    for (auto &[group, groupOps] : memAccesses.lsqPorts) {
      Block *groupBlock = portBlock[groupOps.front()];
      for (handshake::MemPortOpInterface portOp : groupOps) {
        if (portBlock[portOp] != groupBlock) {
          return portOp->emitError()
                 << "LSQ group " << group
                 << " has memory accesses in multiple basic blocks; all "
                    "accesses of an LSQ group must belong to the same block";
        }
      }

#ifndef NDEBUG
      // Verify that no two groups have the same control signal
      if (auto [_, newCtrl] = controlBlocks.insert(groupBlock); !newCtrl)
        return groupOps.front()->emitError()
               << "Inconsistent LSQ group for memory interface the operation "
                  "references. No two groups can have the same control signal.";
#endif

      for (handshake::MemPortOpInterface portOp : groupOps)
        memBuilder.addLSQPort(group, portOp);
    }

    handshake::MemoryControllerOp mcOp;
    handshake::LSQOp lsqOp;
    if (failed(memBuilder.instantiateInterfaces(builder, mcOp, lsqOp)))
      return failure();
  }
  return success();
}

void CfToHandshakeTranslator::createEnd() {
  Block *body = funcOp.getBodyBlock();
  builder.setInsertionPointToEnd(body);
  Location loc = cfFunc.front().getTerminator()->getLoc();

  // Returns are ordered by their block in the func-level function. When
  // identifying basic blocks, the end node is either put in the same block as
  // the function's single return statement or, in the case of multiple return
  // statements, it is put in a "fake block" along with the merges that feed it
  // its data inputs
  llvm::stable_sort(returns, llvm::less_first());
  size_t exitBlockID = blockID.size();
  if (returns.size() == 1)
    exitBlockID -= 1;

  // The terminator's operands are, in order
  // 1. the original function's results
  // 2. a control for each memory region signaling completion
  // 3. a control signaling eventual function completion (the function's start)
  SmallVector<Value, 8> endOprds;
  if (returns.size() == 1) {
    endOprds.append(returns.front().second);
  } else {
    // Return values from multiple returns need to be merged together
    Location mergeLoc = body->empty() ? loc : body->back().getLoc();
    for (size_t i = 0, e = returns.front().second.size(); i < e; ++i) {
      SmallVector<Value, 4> mergeOperands;
      for (auto &[_, operands] : returns)
        mergeOperands.push_back(operands[i]);
      auto mergeOp =
          builder.create<handshake::MergeOp>(mergeLoc, mergeOperands);
      setBB(mergeOp, exitBlockID);
      endOprds.push_back(mergeOp.getResult());
    }
  }

  for (BlockArgument arg : funcOp.getArguments()) {
    if (!isa<MemRefType>(arg.getType()))
      continue;
    if (arg.use_empty()) {
      // When the memory region is not accessed, just a create a constant source
      // of valid "memory end" tokens for it
      auto sourceOp = builder.create<handshake::SourceOp>(loc);
      setBB(sourceOp, exitBlockID);
      endOprds.push_back(sourceOp.getResult());
      continue;
    }
    for (Operation *userOp : arg.getUsers()) {
      auto memOp = cast<handshake::MemoryOpInterface>(userOp);
      if (memOp.isMasterInterface()) {
        endOprds.push_back(memOp.getMemEnd());
        break;
      }
    }
  }
  endOprds.push_back(funcOp.getArguments().back());

  auto endOp = builder.create<handshake::EndOp>(loc, endOprds);
  setBB(endOp, exitBlockID);
}

//===----------------------------------------------------------------------===//
// lowerCfToHandshake
//===----------------------------------------------------------------------===//

LogicalResult dynamatic::lowerFuncsToHandshake(
    ModuleOp modOp, function_ref<LogicalResult(func::FuncOp)> translateFunc) {
  // Translate all functions in order. __init functions only declare the
  // outputs of placeholder functions and are removed afterwards
  for (auto funcOp : llvm::make_early_inc_range(modOp.getOps<func::FuncOp>())) {
    if (funcOp.getSymName().starts_with("__init"))
      continue;
    if (failed(translateFunc(funcOp)))
      return failure();
    funcOp.erase();
  }

  // Initial values of globals are attached to the RAMs referencing them, and
  // calls to __init functions were removed along with their callers
  for (Operation &op : llvm::make_early_inc_range(*modOp.getBody())) {
    if (isa<memref::GlobalOp>(op)) {
      op.erase();
    } else if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      assert(funcOp.getSymName().starts_with("__init") &&
             "all other functions should have been translated");
      funcOp.erase();
    }
  }
  return success();
}

LogicalResult dynamatic::lowerCfToHandshake(ModuleOp modOp,
                                            NameAnalysis &namer) {
  // The translation relies on all live-ins of each block being passed through
  // block arguments, which become merge-like operations
  SSAMaximizationStrategy strategy;
  for (auto funcOp : modOp.getOps<func::FuncOp>()) {
    if (failed(verifySSAMaximized(funcOp.getBody(), strategy)))
      return failure();
  }

  OpBuilder builder(modOp.getContext());
  return lowerFuncsToHandshake(modOp, [&](func::FuncOp funcOp) {
    return CfToHandshakeTranslator(funcOp, builder, namer).translate();
  });
}
