//===- FtdCfToHandshakeTranslator.h - FTD translation to handshake -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Translation of func-level functions into Handshake functions with the fast
// token delivery (FTD) methodology, without the dialect conversion framework.
// Used by the --ftd-lower-cf-to-handshake pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_LIB_CONVERSION_UTILS_FTDCFTOHANDSHAKETRANSLATOR_H
#define DYNAMATIC_LIB_CONVERSION_UTILS_FTDCFTOHANDSHAKETRANSLATOR_H

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Analysis/GSAAnalysis.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"

namespace dynamatic {

/// Lowers all func-level functions of the module to Handshake functions, in
/// order, with `translateFunc`, which must leave the func-level function in
/// place. Memory globals and __init functions (which only declare the outputs
/// of placeholder functions) are removed.
LogicalResult ftdLowerFuncsToHandshake(
    mlir::ModuleOp modOp,
    llvm::function_ref<LogicalResult(mlir::func::FuncOp)> translateFunc);

/// Translates a func-level function into a Handshake function with the FTD
/// methodology:
/// - block arguments become the GSA gates (muxes) computed by the GSA
///   analysis, instead of merge-like operations fed by the predecessors;
/// - terminators only branch the block's control: data tokens are delivered
///   by the regeneration and suppression circuits, which are added once the
///   function is translated;
/// - constants and undefined values are triggered by the function's start.
///
/// The func-level function is the CFG that the FTD analyses run on: it is left
/// untouched until the end of the translation.
///
/// The translation uses an OpBuilder and a mapping between func-level and
/// Handshake-level values. The Handshake function's body is a single block; the
/// func-level blocks only determine the "bb" attribute of the operations and
/// the merge-like and branch-like operations that implement the control flow.
class FtdCfToHandshakeTranslator {
public:
  FtdCfToHandshakeTranslator(mlir::func::FuncOp cfFunc,
                             mlir::OpBuilder &builder, NameAnalysis &namer,
                             const experimental::gsa::GSAAnalysis &gsaAnalysis)
      : cfFunc(cfFunc), builder(builder), namer(namer),
        edgeBuilder(builder, cfFunc.getLoc()), gsaAnalysis(gsaAnalysis) {}

  /// Creates the Handshake function before the func-level one. The func-level
  /// function is left untouched and must be erased by the caller.
  LogicalResult translate();

protected:
  /// An incoming CFG edge to a block, identified by the predecessor block and
  /// the index of the successor in the predecessor's terminator.
  using CFGEdge = std::pair<Block *, unsigned>;

  /// Groups memory operations by interface and group for a given memory
  /// region.
  struct MemAccesses {
    llvm::MapVector<Block *, SmallVector<handshake::MemPortOpInterface>>
        mcPorts;
    llvm::MapVector<unsigned, SmallVector<handshake::MemPortOpInterface>>
        lsqPorts;
    Value memStart;
  };

  /// A Handshake memory port, along with the memory operation it replaces and
  /// the memory region it references.
  struct MemPort {
    /// The memory operation (memref.load or memref.store) in the func-level
    /// function that the port replaces.
    Operation *cfOp;
    /// The memory region (in the func-level function) the port accesses.
    Value memref;
    handshake::MemPortOpInterface portOp;
  };

  /// The func-level function being translated, whose control flow is
  /// expressed with cf operations.
  mlir::func::FuncOp cfFunc;
  /// The Handshake function the translation creates.
  handshake::FuncOp funcOp;
  mlir::OpBuilder &builder;
  NameAnalysis &namer;
  BackedgeBuilder edgeBuilder;

  /// Maps func-level values to Handshake-level values.
  mlir::IRMapping valueMap;
  /// Index of each func-level block (used as the "bb" attribute).
  DenseMap<Block *, unsigned> blockID;
  /// Control value of each func-level block.
  DenseMap<Block *, Value> blockCtrl;
  /// Handshake block holding the operations of each func-level block until
  /// the translation flattens the Handshake function's body. The Handshake
  /// blocks are in the same order as the func-level ones, so the position of
  /// a Handshake block is the "bb" attribute of its operations.
  DenseMap<Block *, Block *> stagingBlock;
  /// Incoming CFG edges of each block. The order of the edges determines the
  /// order of the operands of the block's merge-like operations.
  DenseMap<Block *, SmallVector<CFGEdge>> preds;
  /// Merge-like operations of each block, one per block argument followed by
  /// the control merge. Their data operands are placeholders, one per incoming
  /// edge, until all predecessor terminators are translated.
  DenseMap<Block *, SmallVector<handshake::MergeLikeOpInterface>> mergeOps;
  /// Values flowing along each CFG edge: one per successor block argument,
  /// followed by the control.
  DenseMap<CFGEdge, SmallVector<Value>> edgeValues;
  /// Maps func-level memory regions to their memory accesses.
  llvm::MapVector<Value, MemAccesses> memInfo;
  /// Maps internal array memrefs to a unique number
  llvm::MapVector<Value, unsigned> memrefToInternalArrayID;
  /// Memory ports with the func-level memory operations they replace and the
  /// memory regions they reference, in translation order.
  SmallVector<MemPort> memPorts;
  /// Func-level block of each memory port.
  DenseMap<Operation *, Block *> portBlock;
  /// Operands of the func-level returns, with the ID of their block.
  SmallVector<std::pair<unsigned, SmallVector<Value>>> returns;

  const experimental::gsa::GSAAnalysis &gsaAnalysis;
  /// Indexing of the func-level blocks used by the FTD analyses.
  std::optional<experimental::ftd::BlockIndexing> blockIndexing;
  /// The CFG that the FTD functions (e.g., ftd::addSupp) take as input. Their
  /// interface (ftd::ShadowCFG) bundles the CFG with the Handshake condition of
  /// each conditional block, which the circuits they create read. The CFG is
  /// the func-level function itself, which stays alive until the end of the
  /// translation, so no shadow copy of it is built; only the conditions are
  /// tracked here. Until the blocks are translated, the conditions are
  /// placeholders.
  experimental::ftd::ShadowCFG cfgView;
  /// Placeholder for the condition of each conditional block, replaced by the
  /// block's condition proxy once the blocks are translated.
  SmallVector<std::pair<Block *, Backedge>> condPlaceholders;
  /// Placeholders for the func-level values that the GSA gates take as
  /// operands, which may not be translated yet when the gates are created.
  SmallVector<std::pair<Value, Backedge>> gateOperandPlaceholders;

  /// Transfers the name of the func-level operation (`cfOp`), if it has one, to
  /// the Handshake operation. Unnamed operations are named at the end of the
  /// pass in IR order.
  void transferName(Operation *cfOp, Operation *newOp);

  /// Returns the Handshake equivalent of the func-level values, which must
  /// already be translated.
  SmallVector<Value> lookup(ValueRange values);

  /// Sets the "bb" attribute of a newly created operation.
  void setBlock(Operation *op, Block *block);

  /// Translates the func-level function's body into the Handshake function's
  /// body.
  LogicalResult translateBody();
  void computePredecessors();
  LogicalResult translateBlock(Block *block);
  LogicalResult translateOp(Operation *op);
  LogicalResult translateTerminator(Operation *termOp);
  void resolveMerges();
  /// Moves the operations of all staging blocks into the entry block and
  /// erases the other blocks.
  void flattenBody();

  /// Creates the control merge of a non-entry block, with one placeholder
  /// operand per incoming edge, and records it as the block's control.
  handshake::ControlMergeOp createControlMerge(Block *block);

  /// Called before the blocks are translated.
  LogicalResult initTranslation();
  LogicalResult createMerges(Block *block);
  SmallVector<Value> getForwardedValues(Operation *termOp, unsigned succIdx);

  /// Returns the condition of the conditional branch terminating a block.
  Value getBranchCondition(mlir::cf::CondBranchOp condBrOp);

  /// Returns whether an arith-level constant can be triggered by a source
  /// operation without compromising the circuit semantics.
  bool isCstSourcable(mlir::arith::ConstantOp cstOp);

  /// Returns the control channel that triggers the constants (that cannot be
  /// triggered by a source) and undefined values of a block: the function's
  /// start.
  Value getBlockControlChannel(Block *block);

  /// Called after the memory interfaces are created, before the staging blocks
  /// are flattened.
  void finalizeBlocks();

  /// Called after the Handshake function's body is flattened and terminated.
  LogicalResult finalizeTranslation();

  template <typename DstOp>
  LogicalResult translateOneToOne(Operation *op);
  template <typename ExtOp>
  LogicalResult translateIndexCast(Operation *op);
  LogicalResult translateConstant(mlir::arith::ConstantOp cstOp);
  LogicalResult translateUndef(mlir::LLVM::UndefOp undefOp);
  LogicalResult translateLoad(mlir::memref::LoadOp loadOp);
  LogicalResult translateStore(mlir::memref::StoreOp storeOp);
  LogicalResult translateAlloca(mlir::memref::AllocaOp allocaOp);
  LogicalResult translateGetGlobal(mlir::memref::GetGlobalOp getGlobalOp);
  LogicalResult translateCall(mlir::func::CallOp callOp);
  LogicalResult translateGeneric(Operation *op);

  /// Records a memory port referencing a memory region.
  LogicalResult addMemPort(Operation *cfOp, Value memref,
                           handshake::MemPortOpInterface portOp);
  LogicalResult createMemInterfaces();
  void createEnd();

  /// Creates the muxes implementing the GSA gates of all non-entry blocks, and
  /// maps the block arguments to them.
  LogicalResult createGsaGates();
};

/// Lowers all func-level functions of the module to Handshake functions with
/// a `FtdCfToHandshakeTranslator`.
LogicalResult ftdLowerCfToHandshake(mlir::ModuleOp modOp, NameAnalysis &namer,
                                    const experimental::gsa::GSAAnalysis &gsa);

} // namespace dynamatic

#endif // DYNAMATIC_LIB_CONVERSION_UTILS_FTDCFTOHANDSHAKETRANSLATOR_H
