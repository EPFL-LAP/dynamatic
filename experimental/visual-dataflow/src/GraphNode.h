//===- GraphNode.h - Represents a node in a graph ---------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The GraphNode class represents a node in a graph with the purpose of being
// displayed in Godot.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_VISUAL_DATAFLOW_GRAPHNODE_H
#define DYNAMATIC_VISUAL_DATAFLOW_GRAPHNODE_H

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
  /// Default constructor to create an empty node
  GraphNode();
  /// Constructs a node
  GraphNode(NodeId id, std::pair<int, int> position);
  /// Sets the NodeId
  void setId(NodeId id);
  /// Sets the positon of the Node
  void setPosition(std::pair<float, float> pos);
  /// Adds a port to the Node
  void addPort(std::string &port, bool isInputPort);
  /// Returns the node identifier
  NodeId getNodeId();
  /// Return the position of the Node
  std::pair<float, float> getPosition();
  // Returns the in/out ports of the Node
  std::vector<std::string> getPorts(bool isInputPort);

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

#endif // DYNAMATIC_VISUAL_DATAFLOW_GRAPHNODE_H
