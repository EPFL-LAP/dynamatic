//===- GraphNode.cpp - Represents a node in a graph ------------*- C++ -*-===//
//
// This file contains the implementation of the a GraphNode.
//
//===----------------------------------------------------------------------===//
#include "GraphNode.h"
#include <utility>

using namespace dynamatic::experimental::visual_dataflow;


GraphNode::GraphNode() {
    id = "default";
    position = std::make_pair(0, 0);
    inPorts = std::vector<std::string>();
    outPorts = std::vector<std::string>();
}

GraphNode::GraphNode(NodeId id, std::pair<int, int> position)
    : id(std::move(id)), position(position) {
  inPorts = std::vector<std::string>();
  outPorts = std::vector<std::string>();
}

void GraphNode::setId(NodeId id){
  id = id;
}

void GraphNode::setPosition(std::pair<float, float> position){
  position = position;
}

void GraphNode::addPort(std::string &port, bool isInputPort) {
  if (isInputPort)
    inPorts.push_back(port);
  else
    outPorts.push_back(port);
}

NodeId GraphNode::getNodeId() { return id; }
