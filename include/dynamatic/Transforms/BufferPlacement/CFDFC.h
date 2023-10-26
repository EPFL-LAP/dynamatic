//===- CFDFC.h - Control-Free DataFlow Circuit ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares data structures and functions to extract and create CFDFCs
// (Choice-Free DataFlow Circuits) from a description of a Handshake function's
// archs and basic blocks.
//
// In this context, an arch is understood as an "edge" between two basic blocks.
// There exists an arch in a function between basic block X and Y if and only if
// there exists an operation in X that returns a result used by an operation in
// Y.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/StdProfiler.h"

namespace dynamatic {
namespace buffer {

/// A set of ArchBB's pointers with a deterministic iteration order.
using ArchSet = mlir::SetVector<dynamatic::experimental::ArchBB *>;
/// A set of basic block IDs with a deterministic iteration order.
using BBSet = mlir::SetVector<unsigned>;

/// Represents a CFDFC i.e., a set of control-free units and channels from a
/// dataflow circuit accompanied by the number of times it was executed.
struct CFDFC {
  /// The list of basic blocks that make up the CFDFC.
  mlir::SetVector<unsigned> cycle;
  /// Units (i.e., MLIR operations) in the CFDFC.
  mlir::SetVector<Operation *> units;
  /// Channels (i.e., MLIR values) in the CFDFC.
  mlir::SetVector<Value> channels;
  /// Backedges in the CFDFC.
  mlir::SetVector<Value> backedges;
  /// Number of executions of the CFDFC.
  unsigned numExecs;

  /// Constructs a CFDFC from a set of selected archs and basic blocks in the
  /// function. Assumes that every value in the function is used exactly once.
  CFDFC(circt::handshake::FuncOp funcOp, ArchSet &archs, unsigned numExec);

private:
  // Determines whether the channel is a "CFDFC backedge" i.e., the first
  // channel along a sequence of backedges from a source block to a destination
  // block. The distinction is important for the buffer placement MILP, which
  // uses backedges to determine where to insert "tokens" in the circuit.
  static bool isCFDFCBackedge(Value val);
};

/// Represents a union of CFDFCs. Its blocks, units, channels, and backedges are
/// unions of the corresponding elements from all the CFDFCs that make it up. A
/// CFDFC union, unlike a regular CFDFC, does not contain a number of executions
/// since it does not make sense in the context of a union.
struct CFDFCUnion {
  /// The individual CFDFCs the union is made up of.
  SmallVector<CFDFC *> cfdfcs;
  /// The set basic blocks that make up the CFDFC union.
  mlir::SetVector<unsigned> blocks;
  /// Units (i.e., MLIR operations) in the CFDFC union.
  mlir::SetVector<Operation *> units;
  /// Channels (i.e., MLIR values) in the CFDFC union.
  mlir::SetVector<Value> channels;
  /// Backedges in the CFDFC union.
  mlir::SetVector<Value> backedges;

  /// Constructs the CFDFC union from an array of individual CFDFCs. The CFDFCs
  /// must outlive the union, which stores pointers to them.
  CFDFCUnion(ArrayRef<CFDFC *> cfdfcs);
};

/// Computes a set of CFDFC unions from a list of CFDFCs where each CFDFC in
/// each union shares at least one block with at least one other CFDFC of the
/// same union, and any two CFDFCs in different unions are completely disjoint.
/// Internally uses a disjoint-set data-structure for fast computation times and
/// low memory footprint.
void getDisjointBlockUnions(ArrayRef<CFDFC *> cfdfcs,
                            std::vector<CFDFCUnion> &unions);

/// Extracts the most frequently executed CFDFC from the Handshake function
/// described by the provided archs and basic blocks. The function internally
/// expresses the CFDFC extraction problem as an MILP that is solved bu Gurobi
/// (hence building the project with Gurobi is required to use this function).
/// On successfull extraction, succeeds and sets the last two arguments with,
/// respectively, the set of archs included in the extracted CFDFC and the
/// number of executions of the latter. When no CFDFC could be extracted,
/// succeeds but sets the number of executions to 0. On failure, and if
/// `milpStat` is not nullptr, the Gurobi status is saved in it.
LogicalResult extractCFDFC(circt::handshake::FuncOp funcOp, ArchSet &archs,
                           BBSet &bbs, ArchSet &selectedArchs,
                           unsigned &numExec, const std::string &logPath = "",
                           int *milpStat = nullptr);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_CFDFC_H
