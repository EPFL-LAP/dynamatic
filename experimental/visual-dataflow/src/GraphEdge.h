//===- GraphEdge.h - Represents an edge in a graph ------------*- C++ -*-===//
//
// The GraphEdge class represents an edge in a graph with the purpose of being
// displayed in Godot.
//===----------------------------------------------------------------------===//

#ifndef VISUAL_DATAFLOW_GRAPHEDGE_H
#define VISUAL_DATAFLOW_GRAPHEDGE_H

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
  /// Default Edge constructor
  GraphEdge();
  /// Constructs an edge
  GraphEdge(EdgeId id, GraphNode *src, GraphNode *dst, int inPort, int outPort,
            std::vector<std::pair<float, float>> position);
  /// Sets the EdgeId
  void setId(EdgeId id);
  /// Adds a position to the vector
  void addPosition(std::pair<float, float> position);
  /// Sets the source Node
  void setSrc(GraphNode* src);
  /// Sets the destination Node
  void setDst(GraphNode* dst);
  /// Sets inPort
  void setInPort(int inPort);
  /// Sets outPort
  void setOutPort(int outPort);
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
  /// Returns the edge positions
  std::vector<std::pair<float, float>> getPositions();

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

#endif // VISUAL_DATAFLOW_GRAPHEDGE_H