//===- CfToHandshakeTranslator.h - Translate func/cf to handshake -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Translation of func-level functions with unstructured control flow into
// Handshake functions, without the dialect conversion framework. Used by the
// --lower-cf-to-handshake pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_LIB_CONVERSION_UTILS_CFTOHANDSHAKETRANSLATOR_H
#define DYNAMATIC_LIB_CONVERSION_UTILS_CFTOHANDSHAKETRANSLATOR_H

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/LLVM.h"
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
LogicalResult lowerFuncsToHandshake(
    mlir::ModuleOp modOp,
    llvm::function_ref<LogicalResult(mlir::func::FuncOp)> translateFunc);

/// Lowers all func-level functions of the module to Handshake functions with
/// a `CfToHandshakeTranslator`. The functions must be in maximal SSA form (see
/// --func-maximize-ssa).
LogicalResult lowerCfToHandshake(mlir::ModuleOp modOp, NameAnalysis &namer);

/// Translates a single func-level function (in maximal SSA form) into a
/// Handshake function, using an OpBuilder and a mapping between func-level and
/// Handshake-level values. The Handshake function's body is a single block; the
/// func-level blocks only determine the "bb" attribute of the operations and
/// the merge-like and branch-like operations that implement the control flow.
///
/// Subclasses can change how the control flow is implemented by overriding the
/// virtual hooks (e.g., see FtdCfToHandshakeTranslator).
class CfToHandshakeTranslator {
public:
  CfToHandshakeTranslator(mlir::func::FuncOp srcFunc, mlir::OpBuilder &builder,
                          NameAnalysis &namer)
      : srcFunc(srcFunc), builder(builder), namer(namer),
        edgeBuilder(builder, srcFunc.getLoc()) {}
  virtual ~CfToHandshakeTranslator() = default;

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

  /// A memory port, along with the func-level memory operation it replaces and
  /// the memory region it references.
  struct MemPort {
    Operation *srcOp;
    Value memref;
    handshake::MemPortOpInterface portOp;
  };

  mlir::func::FuncOp srcFunc;
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
  /// Memory ports with the func-level memory operations they replace and the
  /// memory regions they reference, in translation order.
  SmallVector<MemPort> memPorts;
  /// Func-level block of each memory port.
  DenseMap<Operation *, Block *> portBlock;
  /// Operands of the func-level returns, with the ID of their block.
  SmallVector<std::pair<unsigned, SmallVector<Value>>> returns;

  /// Transfers the name of the func-level operation, if it has one, to the
  /// Handshake operation. Unnamed operations are named at the end of the pass
  /// in IR order.
  void transferName(Operation *srcOp, Operation *newOp);

  /// Returns the Handshake equivalent of the func-level values, which must
  /// already be translated.
  SmallVector<Value> lookup(ValueRange values);

  /// Sets the "bb" attribute of a newly created operation.
  void setBlock(Operation *op, Block *block);

  /// Translates the func-level function's body into the Handshake function's
  /// body.
  LogicalResult translateBody();
  LogicalResult initMemories();
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

  //===--------------------------------------------------------------------===//
  // Hooks
  //===--------------------------------------------------------------------===//

  /// Called before the blocks are translated.
  virtual LogicalResult initTranslation() { return success(); }

  /// Creates the merge-like operations at the beginning of a non-entry block:
  /// one per block argument and the control merge. Their data operands are
  /// placeholders (one per incoming edge) that `resolveMerges` replaces with
  /// the values in `edgeValues`.
  virtual LogicalResult createMerges(Block *block);

  /// Returns the values that a block terminator forwards to the successor's
  /// block arguments; the terminator branches each of them, along with the
  /// block's control, to the successor.
  virtual SmallVector<Value> getForwardedValues(Operation *termOp,
                                                unsigned succIdx);

  /// Returns the condition of the conditional branch terminating a block.
  virtual Value getBranchCondition(mlir::cf::CondBranchOp condBrOp) {
    return valueMap.lookup(condBrOp.getCondition());
  }

  /// Returns whether an arith-level constant can be triggered by a source
  /// operation without compromising the circuit semantics.
  virtual bool isCstSourcable(mlir::arith::ConstantOp cstOp);

  /// Returns the control value that triggers the constants (that cannot be
  /// triggered by a source) and undefined values of a block.
  virtual Value getTrigger(Block *block) { return blockCtrl[block]; }

  /// Called after the memory interfaces are created, before the staging blocks
  /// are flattened.
  virtual void finalizeBlocks() {}

  /// Called after the Handshake function's body is flattened and terminated.
  virtual LogicalResult finalizeTranslation() { return success(); }

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
  LogicalResult addMemPort(Operation *srcOp, Value memref,
                           handshake::MemPortOpInterface portOp);
  LogicalResult createMemInterfaces();
  void createEnd();
};

} // namespace dynamatic

#endif // DYNAMATIC_LIB_CONVERSION_UTILS_CFTOHANDSHAKETRANSLATOR_H
