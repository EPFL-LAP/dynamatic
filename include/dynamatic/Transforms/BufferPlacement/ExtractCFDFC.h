//===- ExtractCFDFC.h - Extract CFDFCs from dataflow circuits ---*- C++ -*-===//
//
// This file declaresfunction supports for CFDFCircuit extraction.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_EXTRACTCFDFC_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_EXTRACTCFDFC_H

#include "dynamatic/Support/LLVM.h"
#include "experimental/Support/StdProfiler.h"
#include <map>

namespace dynamatic {
namespace buffer {

/// Data structure for control-free dataflow circuits which
/// stores the units and channels inside it.
/// The CFDFC has properties for the buffer placement optimization, including
/// the target period, and maximum period.
struct CFDFC {
  double targetCP, maxCP;
  std::vector<Operation *> units;
  std::vector<Value> channels;

  unsigned execN = 0; // execution times of CFDFC
};

using SelectedArchs = DenseMap<dynamatic::experimental::ArchBB *, bool>;
using SelectedBBs = DenseMap<unsigned, bool>;

/// Define the MILP CFDFC extraction models, and write the optimization results
/// to the map.
LogicalResult extractCFDFC(SelectedArchs &archs, SelectedBBs &bbs,
                           unsigned &freq);

/// Get the index of the basic block of an operation.
int getBBIndex(Operation *op);

/// Identify whether the channel is in selected the basic block.
bool isSelect(SelectedBBs &bbs, Value val);

/// Identify whether the channel is in selected archs between the basic block.
bool isSelect(SelectedArchs &archs, Value val);

/// Identify whether the connection between the source operation and
/// the destination operation is a back edge.
bool isBackEdge(Operation *opSrc, Operation *opDst);

/// Get the total execution frequency of a channel in the circuit
unsigned getChannelFreq(Value channel, std::vector<CFDFC> &cfdfcList);
} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_EXTRACTCFDFC_H
