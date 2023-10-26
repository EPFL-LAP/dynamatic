//===- GraphEdge.h - Represents an edge in a graph ------------*- C++ -*-===//
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
  GraphEdge(EdgeId id, GraphNode *src, GraphNode *dst, int inPort, int outPort,
            std::vector<std::pair<float, float>> position);
  /// Returns the source node of the edge
  GraphNode *getSrcNode();
  /// Returns the desitnation node of the edge
  GraphNode *getDstNode();
  /// Returns the source out port of the edge
  int getOutPort();
  /// Returns the desitnation in port of the edge
  int getInPort();
  /// Returns the edge identifier
  EdgeId getEdgeId();

private:
  /// Edge identifier
  EdgeId id;
  /// Source node
  GraphNode *src;
  /// Destination node
  GraphNode *dst;
  /// Port number of the destination node
  int inPort;
  /// Port number of the source node
  int outPort;
  /// Positions of the edge in the graph
  std::vector<std::pair<float, float>> position;
};

} // namespace visual_dataflow
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_VISUAL_DATAFLOW_GRAPHEDGE_H