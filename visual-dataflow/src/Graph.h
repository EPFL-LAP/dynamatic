//===- Graph.h - Represents a graph -----------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Graph class represents a graph composed of nodes and edges with the
// purpose of being displayed in Godot.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_VISUAL_DATAFLOW_GRAPH_H
#define DYNAMATIC_VISUAL_DATAFLOW_GRAPH_H

#include "dynamatic/Support/DOT.h"
#include "dynamatic/Support/LLVM.h"
#include <string>
#include <utility>
#include <vector>

namespace dynamatic {
namespace visual {

/// All possible dataflow states. The four combinations of valid/ready wires
/// plus one state for undefined states.
enum DataflowState { UNDEFINED, ACCEPT, IDLE, STALL, TRANSFER };

struct EdgeState {
  DataflowState state;
  std::string data;
};

/// Implements the logic to create and update a Graph
class GodotGraph {
public:
  using Transitions = DenseMap<const DOTGraph::Edge *, EdgeState>;

  struct NodeProps {
    /// Position of the node in the Graph
    std::pair<float, float> position;
    /// Width of the node
    float width;
    /// Color of the Node
    std::string color = "white";
    /// Shape of the Node
    std::string shape = "rectangle";
    /// Style of the borders
    bool isDotted = false;
  };

  struct EdgeProps {
    /// Positions of the edge in the graph
    std::vector<std::pair<float, float>> positions;
    /// Arrowhead style.
    std::string arrowhead;
    // Source port index.
    size_t fromIdx;
    // Destination port index.
    size_t toIdx;
    /// Style of the edge
    bool isDotted = false;
  };

  struct SubgraphProps {
    std::vector<float> boundaries;
    std::string label;
    std::pair<float, float> labelPosition;
    std::pair<float, float> labelSize;
  };

  LogicalResult fromDOTAndCSV(StringRef dotFilePath, StringRef csvFilePath);

  void addEdgeState(unsigned cycle, const DOTGraph::Edge *edge,
                    DataflowState state, StringRef data);

  size_t getLastCycleIdx() const {
    if (transitions.empty())
      return 0;
    return transitions.size() - 1;
  }

  const Transitions &getChanges(unsigned cycle) const {
    return transitions.at(cycle);
  }

  const DOTGraph &getGraph() const { return graph; }

  const NodeProps &getNodeProperties(const DOTGraph::Node *node) const {
    return nodes.at(node);
  }

  const EdgeProps &getEdgeProperties(const DOTGraph::Edge *edge) const {
    return edges.at(edge);
  }

  const SubgraphProps &
  getSubgraphProperties(const DOTGraph::Subgraph *subgraph) const {
    return subgraphs.at(subgraph);
  }

private:
  DOTGraph graph;
  std::vector<Transitions> transitions;

  DenseMap<const DOTGraph::Node *, GodotGraph::NodeProps> nodes;
  DenseMap<const DOTGraph::Edge *, GodotGraph::EdgeProps> edges;
  DenseMap<const DOTGraph::Subgraph *, GodotGraph::SubgraphProps> subgraphs;

  LogicalResult parseDOT(StringRef filepath);

  LogicalResult parseCSV(StringRef filepath);
};

} // namespace visual
} // namespace dynamatic

#endif // DYNAMATIC_VISUAL_DATAFLOW_GRAPH_H
