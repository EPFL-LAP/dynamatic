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
#include <fstream>

namespace dynamatic {
namespace buffer {

struct ArchBB {
  unsigned srcBB, dstBB;
  int execFreq = -1;
  bool isBackEdge;
  void print() {
    llvm::errs() << "srcBB: " << srcBB << ", dstBB: " << dstBB
                 << ", execFreq: " << execFreq << ", isBackEdge: " << isBackEdge
                 << "\n";
  }
};

/// Define the MILP CFDFC extraction models, and write the optimization results
/// to the map.
int extractCFDFCircuit(std::map<ArchBB *, bool> &archs, std::map<unsigned, bool> &bbs);


/// Get the index of the basic block of an operation.
int getBBIndex(Operation *op);

///Identify whether the channel is in selected the basic block.
bool isSelect(std::map<unsigned, bool> &bbs, Value *val);

/// Identify whether the channel is in selected archs between the basic block.
bool isSelect(std::map<ArchBB *, bool> &archs, Value *val);

/// Identify whether an operation is a start point of the function block.
bool isEntryOp(Operation *op);

/// Identify whether the connection between the source operation and
/// the destination operation is a back edge.
bool isBackEdge(Operation *opSrc, Operation *opDst);

bool isBackEdge(Value *val);

/// Read the simulation file of standard level execution and store the results
/// in the map.
LogicalResult readSimulateFile(const std::string &fileName,
                      std::map<ArchBB *, bool> &archs, std::map<unsigned, bool> &bbs);
} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_EXTRACTMG_H