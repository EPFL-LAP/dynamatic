//===- GraphNode.h - Represents a node in a graph ------------*- C++ -*-===//
//
// The GraphNode class represents a node in a graph with the purpose of being
// displayed in Godot.
//===----------------------------------------------------------------------===//

#ifndef VISUAL_DATAFLOW_GRAPHNODE_H
#define VISUAL_DATAFLOW_GRAPHNODE_H

#include <map>
#include <string>
#include <vector>

namespace dynamatic {
namespace experimental {
namespace visual_dataflow {

using NodeId = std::string;

/// Implements the logic to create and update a Node
class GraphNode {

public:
  /// Constructs a node
  GraphNode(NodeId id, std::pair<int, int> position);
  /// Adds a port to the Node
  void addPort(std::string &port, bool isInputPort);
  /// Returns the node identifier
  NodeId getNodeId();

private:
  /// Node identifier
  NodeId id;
  /// List of all the input ports of the node
  std::vector<std::string> inPorts;
  /// List of all the output ports of the node
  std::vector<std::string> outPorts;
  /// Position of the node in the Graph
  std::pair<float, float> position;
};

} // namespace visual_dataflow
} // namespace experimental
} // namespace dynamatic

#endif // VISUAL_DATAFLOW_GRAPHNODE_H