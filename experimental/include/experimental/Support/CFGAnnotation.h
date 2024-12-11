//===- CFGAnnotation.h --- CFG Annotation -----------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares a set of utilites to annotate the CFG in an handshake function,
// parse the information back, re-build the cf structure and flatten it back.
// The following steps should be taken into account to use this library (related
// to the dependencies with the func and cf dialects)
//
// 1. When declaring the pass in `Passes.td`, add the following two dialects as
// dependencies: mlir::cf::ControlFlowDialect, mlir::func::FuncDialect";
// 2. In the header file of the same pass, add the following inclusions:
//    #include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
//    #include "mlir/Dialect/Func/IR/FuncOps.h"
// 3. In the CMake file of the same pass, add the following dependencies:
//    MLIRControlFlowDialect
//    MLIRFuncDialect
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_CFG_ANNOTATION_H
#define DYNAMATIC_SUPPORT_CFG_ANNOTATION_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"

namespace dynamatic {
namespace experimental {
namespace cfg {

/// Name used to store the edge annotation on the handshake function
constexpr llvm::StringLiteral CFG_EDGES("cfg.edges");

constexpr char OPEN_EDGE = '[';
constexpr char CLOSE_EDGE = ']';
constexpr char DELIMITER = ',';

/// Class to store information related to an edge in a CFG. Each node is
/// identified by its index as expressed in the MLIR format.
class CFGNode {

  /// Variant to store the successors of a node in the CFG. It's either one
  /// single successor or two of them.
  std::variant<unsigned, std::pair<unsigned, unsigned>> successors;

  /// Optional string to store the name of the operation driving a conditional
  /// branch from a node.
  std::optional<std::string> conditionName;

public:
  /// Constructor of a CFG edge with one successor only.
  CFGNode(unsigned e) : successors(e), conditionName(std::nullopt) {}

  // Constructor of a CFG edge with a true and a false successor.
  CFGNode(unsigned t, unsigned f, std::string c)
      : successors(std::pair<unsigned, unsigned>(t, f)), conditionName(c) {}

  /// Copy constructor.
  CFGNode(const CFGNode &c) = default;

  /// Default constructor.
  CFGNode() : successors(0u), conditionName(std::nullopt) {}

  /// Returns true if the edge is unconditional.
  bool isUnconditional() const;

  /// Returns true if the edge is conditional.
  bool isConditional() const;

  /// Returns the successor if the edge is unconditional.
  unsigned getSuccessor() const;

  /// Returns the true successor if the edge is conditional.
  unsigned getTrueSuccessor() const;

  /// Returns the false successor if the edge is conditional.
  unsigned getFalseSuccessor() const;

  /// Returns the string condition if the edge is conditional.
  std::string getCondition() const;
};

/// Define an object to contain the annotation, so that the index of each node
/// is associated to each edge information.
using CFGAnnotation = llvm::DenseMap<unsigned, CFGNode>;

/// Get a string version out of the edge annotation, with the following BNF
/// grammar:
///
/// List = {Edges}
/// Edges = Edge Edges | null
/// Edge = CondtionalEdge | UnconditionalEdge
/// ConditionalEdge = [source,destTrue,destFalse,condName]
/// UnconditionalEdge = [source,dest]
///
/// Add the string as annotation of the handshake function.
void annotateCFG(handshake::FuncOp &funcOp, PatternRewriter &rewriter);

/// Use an handshake function ot build the cf structure again, thanks to the
/// information in `edges`.
LogicalResult restoreCfStructure(handshake::FuncOp &funcOp,
                                 PatternRewriter &rewriter);

/// Get rid of the cf structure by moving all the operations in the initial
/// block and removing all the cf terminators.
LogicalResult flattenFunction(handshake::FuncOp &funcOp);

/// Sets an integer "bb" attribute on each operation to identify the basic
/// block from which the operation originates in the std-level IR.
void markBasicBlocks(handshake::FuncOp &funcOp, PatternRewriter &rewriter);

}; // namespace cfg
}; // namespace experimental
}; // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_FTD_SUPPORT_H
