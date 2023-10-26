//===- StdProfiler.h - std-level profiler -----------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declaration of std-level profiler that works in tandem with the std-level
// simulator to collect statistics during simulation.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_SUPPORT_STDPROFILER_H
#define EXPERIMENTAL_SUPPORT_STDPROFILER_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/IndentedOstream.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace dynamatic {
namespace experimental {

/// Represents an arch between two basic blocks that is traversed a specific
/// number of times during simulation. This is used to read back the profiler's
/// CSV formatted results into memory.
struct ArchBB {
  /// Source basic block ID.
  unsigned srcBB;
  /// Destination basic block ID.
  unsigned dstBB;
  /// Number of transitions recoded between the two blocks.
  unsigned numTrans;
  /// Is the arch a backedge?
  bool isBackEdge;

  /// Simple field-by-field constructor.
  ArchBB(unsigned srcBB, unsigned dstBB, unsigned numTrans, bool isBackEdge);
};

/// Data structure to hold information about the simulation.
struct StdProfiler {

  /// Holds the number of transitions between each block pair.
  mlir::DenseMap<std::pair<mlir::Block *, mlir::Block *>, unsigned> transitions;

  /// Constructs a profiler on a given function.
  StdProfiler(mlir::func::FuncOp funcOp);

  /// Prints gathered statistics on standard output.
  void writeStats(bool printDOT);

  /// Prints statistics on an output stream as a DOT formatted output compatible
  /// with legacy Dynamatic's smart buffer placement pass.
  void writeDOT(mlir::raw_indented_ostream &os);

  /// Prints statistics on an output stream as a CSV formatted output compatible
  /// with Dynamatic's smart buffer placement pass.
  void writeCSV(mlir::raw_indented_ostream &os);

  /// Reads the profiler's CSV formatted output and stores its content into an
  /// in-memory data-structure (one vector element per CSV line).
  static mlir::LogicalResult readCSV(std::string &filename,
                                     mlir::SmallVector<ArchBB> &archs);

private:
  /// The function being profiled.
  mlir::func::FuncOp funcOp;

  /// Produces a DOT formatted string corresponding to a transition between two
  /// blocks.
  std::string getDOTTransitionString(std::string &srcBlock,
                                     std::string &dstBlock, unsigned freq);

  /// Produces a CSV formatted string corresponding to a transition between two
  /// blocks.
  std::string getCSVTransitionString(unsigned srcBlock, unsigned dstBlock,
                                     unsigned freq, bool isBackedge);
};
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_SUPPORT_STDPROFILER_H