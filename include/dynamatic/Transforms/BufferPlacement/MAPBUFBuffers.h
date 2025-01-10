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
#include "experimental/Support/CutEnumeration.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace dynamatic {
namespace buffer {
namespace mapbuf {

using pathMap = std::unordered_map<
    std::pair<experimental::Node *, experimental::Node *>,
    std::vector<experimental::Node *>,
    boost::hash<std::pair<experimental::Node *, experimental::Node *>>>;

/// Holds the state and logic for MAPBUF smart buffer placement. To buffer a
/// dataflow circuit, this MILP-based algorithm creates:
/// 1. custom channel constraints derived from channel-specific buffering
///    properties
/// 2. path constraints for all non-memory channels and units
/// 3. elasticity constraints for all non-memory channels and units
/// 4. throughput constraints for all channels and units parts of CFDFCs that
///    were extracted from the function
/// 5. a maximixation objective, that rewards high CFDFC throughputs and
///    penalizes the placement of many large buffers in the circuit
class MAPBUFBuffers : public BufferPlacementMILP {
public:
  /// Setups the entire MILP that buffers the input dataflow circuit for the
  /// target clock period, after which (absent errors) it is ready for
  /// optimization. The `legacyPlacemnt` controls the interpretation of the
  /// MILP's results (non-legacy placement should yield faster circuits in
  /// general). If a channel's buffering properties are provably unsatisfiable,
  /// the MILP will not be marked ready for optimization, ensuring that further
  /// calls to `optimize` fail.
  MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB,
                double targetPeriod, StringRef blifFiles);

  /// Achieves the same as the other constructor but additionally logs placement
  /// decisions and achieved throughputs using the provided logger, and dumps
  /// the MILP model and solution at the provided name next to the log file.
  MAPBUFBuffers(GRBEnv &env, FuncInfo &funcInfo, const TimingDatabase &timingDB,
                double targetPeriod, StringRef blifFiles, Logger &logger,
                StringRef milpName = "placement");

protected:
  /// The same extractResult function used in FPL22Buffers.
  void extractResult(BufferPlacement &placement) override;

private:
  float lutDelay = 0.55;
  int bigConstant = 100;
  experimental::BlifData *blifData;
  pathMap leafToRootPaths;
  StringRef blifFiles;

  // Adds Blackbox Constraints for the Data Signals of blackbox ADDI, SUBI and
  // CMPI modules. These delays are retrieved from Vivado Timing Reports. Ready
  // and Valid signals are not blackboxed.
  void addBlackboxConstraints(Value channel);

  /// Adds channel-specific buffering constraints that were parsed from IR
  /// annotations to the Gurobi model.
  void addCustomChannelConstraints(Value channel);

  // Adds Cut Selection Constraints, ensuring that only 1 cut is selected per
  // node
  void addCutSelectionConstraints(std::vector<experimental::Cut> &cutVector);

  // Adds Cut Selection Conflict Constraints. These constraints ensure that
  // either a buffer is placed on a DFG edge or the cut that containts that edge
  // is selected.
  void addCutSelectionConflicts(experimental::Node *root,
                                experimental::Node *leaf,
                                GRBVar &cutSelectionVar);

  // Inserts Buffers on Back Edges. This is done by adding constraints to the
  // Gurobi Model, and inserting buffers into Subject Graph. The function loops
  // over all the channels and checks if the channel is a back edge. If it is a
  // back edge, then a buffer is inserted.
  void addCutLoopbackBuffers();

  // Converts the Cyclic Graph into an Acyclic Graph by determining the Minimum
  // Feedback Arc Set (MFAS). A graph can only be acyclic if a topological
  // ordering can be found. An additional MILP is used here, which enforces a
  // topological ordering. Since our graph is cyclic, a topological ordering
  // cannot be found without removing some edges. The MILP formulated here
  // minimizes the number of edges that needs to be removed in order to make the
  // graph acyclic. Then, buffers are inserted on the edges that needs to be
  // removed to make the graph acyclic.
  void findMinimumFeedbackArcSet();

  // Add clock period constraints for subject graph edges. For subject graph
  // edges, only a single timing variable is requires, as opposed to data flow
  // graph edges where two timing variables are required. Also adds constraints
  // for primary inputs and constants.
  void addClockPeriodConstraintsNodes();

  // Adds Clock Period Constraints, Buffer Insertion and Channel Constraints to
  // the Dataflow Graph Edges.
  void addClockPeriodConstraintsChannels(Value channel, SignalType signal);

  // Adds Delay Propagation Constraints for all the cuts by looping over cuts
  // map. If a node has only one fanin, delay is propagated from the fanin.
  // Otherwise, delay is propagated from the leaves of the cut. Loops over the
  // leaves of the cut and adds delay propagation constraints for each leaf.
  // Also adds cut selection conflict constraints.
  void
  addDelayPropagationConstraints(experimental::Node *root,
                                 std::vector<experimental::Cut> &cutVector);

  // After initializing the individual Subject Graphs, connects the Subject
  // Graphs by connecting the input and output nodes of the adjacent modules.
  // Then generates the merged Subject Graph, which is the overall Subject Graph
  // of the entire circuit. blifData variable is assigned using this Subject
  // Graph.
  void connectSubjectGraphs();

  /// Setups the entire MILP, creating all variables, constraints, and
  /// setting the system's objective. Called by the constructor in the
  /// absence of prior failures, after which the MILP is ready to be
  /// optimized.
  void setup();
};

} // namespace mapbuf
} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_MAPBUFBUFFERS_H