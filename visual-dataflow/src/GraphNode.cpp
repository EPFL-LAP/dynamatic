//===- GraphNode.cpp - Represents a node in a graph -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
  inPorts = std::vector<size_t>();
  outPorts = std::vector<size_t>();
  width = 0.0;
}

GraphNode::GraphNode(NodeId id, std::pair<int, int> position)
    : id(std::move(id)), position(position) {
  inPorts = std::vector<size_t>();
  outPorts = std::vector<size_t>();
}

void GraphNode::setId(NodeId id) { this->id = std::move(id); }

void GraphNode::setPosition(std::pair<float, float> position) {
  this->position = position;
}

void GraphNode::addPort(size_t port, bool isInputPort) {
  if (isInputPort)
    inPorts.push_back(port);
  else
    outPorts.push_back(port);
}

NodeId GraphNode::getNodeId() { return id; }

std::pair<float, float> GraphNode::getPosition() { return position; }

std::vector<size_t> GraphNode::getPorts(bool isInputPort) {
  if (isInputPort)
    return inPorts;
  return outPorts;
}

void GraphNode::setWidth(float width) { this->width = width; }

float GraphNode::getWidth() { return width; }

void GraphNode::setColor(Color color) { this->color = color; }

Color GraphNode::getColor() { return color; }

void GraphNode::setShape(Shape shape) { this->shape = shape; }

Shape GraphNode::getShape() { return shape; }

void GraphNode::setDashed(bool dashed) { this->isDashed = dashed; }

bool GraphNode::getDashed() { return isDashed; }
