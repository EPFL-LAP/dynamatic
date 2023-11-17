//===- GraphEdge.h - Represents an edge in a graph --------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The GraphEdge class represents an edge in a graph with the purpose of being
// displayed in Godot.
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_VISUAL_DATAFLOW_GRAPHEDGE_H
#define DYNAMATIC_VISUAL_DATAFLOW_GRAPHEDGE_H

#include "GraphNode.h"
#include <map>
#include <string>
#include <vector>

namespace dynamatic {
namespace experimental {
namespace visual_dataflow {

using EdgeId = int;

/// Implements the logic to create an Edge
class GraphEdge {

public:
  /// Constructs an edge
  GraphEdge(EdgeId id = -1, GraphNode src = GraphNode(),
            GraphNode dst = GraphNode(), unsigned inPort = 0,
            unsigned outPort = 0,
            std::vector<std::pair<float, float>> position =
                std::vector<std::pair<float, float>>());
  /// Sets the EdgeId
  void setId(EdgeId id);
  /// Adds a position to the vector
  void addPosition(std::pair<float, float> position);
  /// Sets the source Node
  void setSrc(GraphNode src);
  /// Sets the destination Node
  void setDst(GraphNode dst);
  /// Sets inPort
  void setInPort(unsigned inPort);
  /// Sets outPort
  void setOutPort(unsigned outPort);
  /// Returns the source node of the edge
  GraphNode getSrcNode();
  /// Returns the desitnation node of the edge
  GraphNode getDstNode();
  /// Returns the source out port of the edge
  unsigned getOutPort();
  /// Returns the desitnation in port of the edge
  unsigned getInPort();
  /// Returns the edge identifier
  EdgeId getEdgeId();
  /// Returns the edge positions
  std::vector<std::pair<float, float>> getPositions();

private:
  /// Edge identifier
  EdgeId id;
  /// Source node
  GraphNode src;
  /// Destination node
  GraphNode dst;
  /// Port number of the destination node
  unsigned inPort;
  /// Port number of the source node
  unsigned outPort;
  /// Positions of the edge in the graph
  std::vector<std::pair<float, float>> position;
};

} // namespace visual_dataflow
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_VISUAL_DATAFLOW_GRAPHEDGE_H
