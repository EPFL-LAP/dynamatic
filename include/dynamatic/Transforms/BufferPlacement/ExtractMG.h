//===- UtilsForExtractMG.h - utils for extracting marked graph *- C++ ---*-===//
//
// This file declaresfunction supports for CFDFCircuit extraction.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_EXTRACTMG_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_EXTRACTMG_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace dynamatic {
namespace buffer {

/// Data structure for archs between the two basic blocks
/// srcBB: index of the source basic block;
/// dstBB: index of the destination basic block;
/// execFreq: execution frequency of the arch;
/// isBackEdge: whether the arch is a back edge.
struct ArchBB {
  unsigned srcBB, dstBB, execFreq;
  bool isBackEdge;
};

/// Read the simulation file of standard level execution and store the results
/// in the map.
LogicalResult readSimulateFile(const std::string &fileName,
                               std::map<ArchBB *, bool> &archs,
                               std::map<unsigned, bool> &bbs);

/// Define the MILP CFDFC extraction models, and write the optimization results
/// to the map.
LogicalResult extractCFDFCircuit(std::map<ArchBB *, bool> &archs,
                                 std::map<unsigned, bool> &bbs, unsigned &freq);

/// Get the index of the basic block of an operation.
int getBBIndex(Operation *op);

/// Identify whether the channel is in selected the basic block.
bool isSelect(std::map<unsigned, bool> &bbs, Value val);

/// Identify whether the channel is in selected archs between the basic block.
bool isSelect(std::map<ArchBB *, bool> &archs, Value val);

/// Identify whether the connection between the source operation and
/// the destination operation is a back edge.
bool isBackEdge(Operation *opSrc, Operation *opDst);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_EXTRACTMG_H
