//===- MAPBUFBuffers.h - MAPBUF buffer placement ---------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
// This mainly file declares the `MAPBUFPlacement` class, which inherits the
// abstract `BufferPlacementMILP` class to setup and solve a real MILP from
// which buffering decisions can be made. Every public member declared in this
// file is under the `dynamatic::buffer::mapbuf` namespace, as to not create
// name conflicts for common structs with other implementors of
// `BufferPlacementMILP`.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_MAPBUFBUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_MAPBUFBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "experimental/Support/BlifReader.h"
#include "experimental/Support/CutlessMapping.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace dynamatic {
namespace buffer {
namespace mapbuf {

using pathMap =
    std::unordered_map<std::pair<experimental::Node *, experimental::Node *>,
                       std::vector<experimental::Node *>,
                       experimental::NodePairHash>;

/// Holds the state and logic for MapBuf smart buffer placement.
class MAPBUFBuffers : public BufferPlacementMILP {
public:
  /// Setups the entire MILP that buffers the input dataflow circuit for the
  /// target clock period, after which (absent errors) it is ready for
  /// optimization. If a channel's buffering properties are provably
  /// unsatisfiable, the MILP will not be marked ready for optimization,
  /// ensuring that further calls to `optimize` fail.
  MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB,
                double targetPeriod, StringRef blifFiles, double lutDelay,
                int lutSize, bool acyclicType);

  /// Achieves the same as the other constructor but additionally logs placement
  /// decisions and achieved throughputs using the provided logger, and dumps
  /// the MILP model and solution at the provided name next to the log file.
  MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB,
                double targetPeriod, StringRef blifFiles, double lutDelay,
                int lutSize, bool acyclicType, Logger &logger,
                StringRef milpName = "placement");

protected:
  /// The same extractResult function used in FPL22Buffers.
  void extractResult(BufferPlacement &placement) override;

private:
  // Method for creating acyclic graphs from cyclic dataflow graph. If false,
  // cut loopbacks method is used. If true,
  // findMinimumFeedbackArcSet() method is used.
  bool acyclicType;
  // Maximum LUT input size of the target FPGA.
  int lutSize;
  // Average delay in nanoseconds for Look-Up Table (LUT) in the target FPGA.
  double lutDelay;
  // LogicNetwork of the circuit
  experimental::LogicNetwork *blifData;
  // Map that allows quick lookups from leaf to root nodes
  pathMap leafToRootPaths;
  // Path of BLIF files
  StringRef blifFiles;

  /// Adds channel-specific buffering constraints that were parsed from IR
  /// annotations to the Gurobi model.
  void addCustomChannelConstraints(Value channel);

  /// Setups the entire MILP, creating all variables, constraints, and setting
  /// the system's objective. Called by the constructor in the absence of prior
  /// failures, after which the MILP is ready to be optimized.
  void setup();
};

} // namespace mapbuf
} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_MAPBUFBUFFERS_H
