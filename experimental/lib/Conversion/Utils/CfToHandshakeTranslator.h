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
// --exp-lower-cf-to-handshake pass.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_LIB_CONVERSION_UTILS_CFTOHANDSHAKETRANSLATOR_H
#define EXPERIMENTAL_LIB_CONVERSION_UTILS_CFTOHANDSHAKETRANSLATOR_H

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"

namespace dynamatic {
namespace experimental {

/// Lowers all func-level functions of the module to Handshake functions. The
/// functions must be in maximal SSA form (see --func-maximize-ssa); they are
/// translated in order by a `CfToHandshakeTranslator`. Memory globals and
/// __init functions (which only declare the outputs of placeholder functions)
/// are removed.
LogicalResult lowerCfToHandshake(mlir::ModuleOp modOp, NameAnalysis &namer);

/// Translates a single func-level function (in maximal SSA form) into a
/// Handshake function, using an OpBuilder and a mapping between func-level and
/// Handshake-level values. The Handshake function's body is a single block; the
/// func-level blocks only determine the "bb" attribute of the operations and
/// the merge-like and branch-like operations that implement the control flow.
class CfToHandshakeTranslator {
public:
  CfToHandshakeTranslator(mlir::func::FuncOp srcFunc, mlir::OpBuilder &builder,
                          NameAnalysis &namer)
      : srcFunc(srcFunc), builder(builder), namer(namer),
        edgeBuilder(builder, srcFunc.getLoc()) {}

  /// Creates the Handshake function before the func-level one. The func-level
  /// function is left untouched and must be erased by the caller.
  LogicalResult translate();

private:
  /// An incoming CFG edge to a block, identified by the predecessor block and
  /// the index of the successor in the predecessor's terminator.
  using Edge = std::pair<Block *, unsigned>;

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
  /// First Handshake operation created for each func-level block. Blocks are
  /// translated in reverse post-order, but their operations are laid out in
  /// the Handshake function's body in the func-level block order.
  DenseMap<Block *, Operation *> blockStart;
  /// Incoming CFG edges of each block. The order of the edges determines the
  /// order of the operands of the block's merge-like operations.
  DenseMap<Block *, SmallVector<Edge>> preds;
  /// Merge-like operations of each block, one per block argument followed by
  /// the control merge. Their data operands are placeholders, one per incoming
  /// edge, until all predecessor terminators are translated.
  DenseMap<Block *, SmallVector<handshake::MergeLikeOpInterface>> mergeOps;
  /// Values flowing along each CFG edge: one per successor block argument,
  /// followed by the control.
  DenseMap<Edge, SmallVector<Value>> edgeValues;
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
  LogicalResult createMerges(Block *block);
  LogicalResult translateBlock(Block *block);
  LogicalResult translateOp(Operation *op);
  LogicalResult translateTerminator(Operation *termOp);
  void resolveMerges();

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

} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_LIB_CONVERSION_UTILS_CFTOHANDSHAKETRANSLATOR_H
