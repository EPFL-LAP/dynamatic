//===- GraphEdge.cpp - Represents an edge in a graph ------------*- C++ -*-===//
//
// This file contains the implementation of a GraphEdge.
//
//===----------------------------------------------------------------------===//
#include "GraphEdge.h"
#include <utility>

using namespace dynamatic::experimental::visual_dataflow;

GraphEdge::GraphEdge(){
  id = -1;
  src = nullptr;
  dst = nullptr;
  inPort = 0;
  outPort = 0;
  position = std::vector<std::pair<float, float>>();
}

void GraphEdge::setId(EdgeId id){
  id = id;
}

void GraphEdge::addPosition(std::pair<float, float> pos){
  position.push_back(pos);
}

void GraphEdge::setSrc(GraphNode* src){
  src = src;
}

void GraphEdge::setDst(GraphNode* dst){
  dst = dst;
}

void GraphEdge::setInPort(int inPort){
  inPort = inPort;
}

void GraphEdge::setOutPort(int outPort){
  outPort = outPort;
}

GraphEdge::GraphEdge(EdgeId id, GraphNode *src, GraphNode *dst, int inPort,
                     int outPort, std::vector<std::pair<float, float>> position)
    : id(id), src(src), dst(dst), inPort(inPort), outPort(outPort),
      position(std::move(position)) {}

GraphNode *GraphEdge::getSrcNode() { return src; }
GraphNode *GraphEdge::getDstNode() { return dst; }

int GraphEdge::getOutPort() { return outPort; }
int GraphEdge::getInPort() { return inPort; }

EdgeId GraphEdge::getEdgeId() { return id; }