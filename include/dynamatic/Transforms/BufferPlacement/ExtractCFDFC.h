//===- ExtractCFDFC.h - Extract CFDFCs from dataflow circuits ---*- C++ -*-===//
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

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_EXTRACTCFDFC_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_EXTRACTCFDFC_H

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
  /// Units (i.e., MLIR operations) in the CFDFC.
  mlir::SetVector<Operation *> units;
  /// Channels (i.e., MLIR values) in the CFDFC.
  mlir::SetVector<Value> channels;
  /// Number of executions of the CFDFC.
  unsigned numExec = 0;

  /// Constructs a CFDFC from a set of selected archs and basic blocks in the
  /// function. Assumes that every value in the function is used exactly once.
  CFDFC(circt::handshake::FuncOp funcOp, ArchSet &archs, BBSet &bbs,
        unsigned numExec);
};

/// Determines whether the edge between a source and destination operation is a
/// backedge in the context of buffer placement. The function assumes that the
/// source operation produces a value that the destination operation consumes.
bool isBackEdge(Operation *src, Operation *dst);

/// Extracts the most frequently executed CFDFC from the Handshake function
/// described by the provided archs and basic blocks. The function internally
/// expresses the CFDFC extraction problem as an MILP that is solved bu Gurobi
/// (hence building the project with Gurobi is required to use this function).
/// On successfull extraction, succeeds and sets the last three argument with,
/// respectively, the set of archs and basic blocks that are included in the
/// extracted CFDFC and the number of executions of the latter. When no CFDFC
/// could be extracted, succeeds but sets the number of executions to 0.
LogicalResult extractCFDFC(circt::handshake::FuncOp funcOp, ArchSet &archs,
                           BBSet &bbs, ArchSet &selectedArchs,
                           BBSet &selectedBBs, unsigned &numExec);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_EXTRACTCFDFC_H
