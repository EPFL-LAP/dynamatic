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

void GraphEdge::setInPort(size_t inPort) { this->inPort = inPort; }

void GraphEdge::setOutPort(size_t outPort) { this->outPort = outPort; }

GraphEdge::GraphEdge(EdgeId id, GraphNode src, GraphNode dst, size_t inPort,
                     size_t outPort,
                     std::vector<std::pair<float, float>> position)
    : id(id), src(std::move(src)), dst(std::move(dst)), inPort(inPort),
      outPort(outPort), position(std::move(position)) {}

GraphNode GraphEdge::getSrcNode() { return src; }
GraphNode GraphEdge::getDstNode() { return dst; }

size_t GraphEdge::getOutPort() { return outPort; }
size_t GraphEdge::getInPort() { return inPort; }

EdgeId GraphEdge::getEdgeId() { return id; }

std::vector<std::pair<float, float>> GraphEdge::getPositions() {
  return position;
}

void GraphEdge::setDashed(bool dashed) { this->isDashed = dashed; }

bool GraphEdge::getDashed() { return isDashed; }
