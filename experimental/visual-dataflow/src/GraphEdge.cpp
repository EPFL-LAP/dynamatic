//===- GraphEdge.cpp - Represents an edge in a graph ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of a GraphEdge.
//
//===----------------------------------------------------------------------===//
#include "GraphEdge.h"
#include <utility>

using namespace dynamatic::experimental::visual_dataflow;

void GraphEdge::setId(EdgeId id) { this->id = id; }

void GraphEdge::addPosition(std::pair<float, float> pos) {
  position.push_back(pos);
}

void GraphEdge::setSrc(GraphNode src) { this->src = std::move(src); }

void GraphEdge::setDst(GraphNode dst) { this->dst = std::move(dst); }

void GraphEdge::setInPort(unsigned inPort) { this->inPort = inPort; }

void GraphEdge::setOutPort(unsigned outPort) { this->outPort = outPort; }

GraphEdge::GraphEdge(EdgeId id, GraphNode src, GraphNode dst, unsigned inPort,
                     unsigned outPort,
                     std::vector<std::pair<float, float>> position)
    : id(id), src(std::move(src)), dst(std::move(dst)), inPort(inPort),
      outPort(outPort), position(std::move(position)) {}

GraphNode GraphEdge::getSrcNode() { return src; }
GraphNode GraphEdge::getDstNode() { return dst; }

unsigned GraphEdge::getOutPort() { return outPort; }
unsigned GraphEdge::getInPort() { return inPort; }

EdgeId GraphEdge::getEdgeId() { return id; }

std::vector<std::pair<float, float>> GraphEdge::getPositions() {
  return position;
}
