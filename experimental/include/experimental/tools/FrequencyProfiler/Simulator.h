//===- Simulator.h - std-level simulator ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations of profiler and top level simulator function that executes a
// restricted form of the standard dialect (inherited from CIRCT).
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_TOOLS_FREQUENCYPROFILER_H
#define EXPERIMENTAL_TOOLS_FREQUENCYPROFILER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace dynamatic {
namespace experimental {

/// Data structure to hold information about the simulation.
struct Profiler {

  /// Holds the number of transitions between each block pair.
  mlir::DenseMap<std::pair<mlir::Block *, mlir::Block *>, unsigned> transitions;

  /// Constructs a profiler on a given function.
  Profiler(mlir::func::FuncOp funcOp);

  /// Prints gathered statistics on standard output.
  void writeStats(bool printDOT);

  /// Prints statistics on an output stream as a DOT formatted output compatible
  /// with legacy Dynamatic's smart buffer placement pass.
  void writeDOT(mlir::raw_indented_ostream &os);

  /// Prints statistics on an output stream as a CSV formatted output compatible
  /// with Dynamatic++'s smart buffer placement pass.
  void writeCSV(mlir::raw_indented_ostream &os);

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

/// Simulates a std-level function on a specific set of inputs.
mlir::LogicalResult simulate(mlir::func::FuncOp funcOp,
                             mlir::ArrayRef<std::string> inputArgs,
                             Profiler &prof);
} // namespace experimental
} // namespace dynamatic

#endif // EXPERIMENTAL_TOOLS_FREQUENCYPROFILER_H
