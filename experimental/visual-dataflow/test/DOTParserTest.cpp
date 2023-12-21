//===- DOTParserTest.h - Executable test for DOT parser ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Run the DOT reformatter/parser on a valid input DOT to check whether it
// manages to be transformed into a graph.
//
//===----------------------------------------------------------------------===//

#include "../src/DOTParser.h"
#include "../src/DOTReformat.h"
#include "../src/Graph.h"
#include "../src/GraphEdge.h"
#include "../src/GraphNode.h"
#include "mlir/Support/LogicalResult.h"
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace dynamatic::experimental::visual_dataflow;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input_dot_file> <output_dot_file>"
              << std::endl;
    return 1;
  }

  const std::string inputDotFile = argv[1];
  const std::string outputDotFile = argv[2];

  if (failed(reformatDot(inputDotFile, outputDotFile)))
    return 1;

  std::ifstream f;
  f.open(outputDotFile);

  Graph graph;

  if (failed(processDOT(f, graph)))
    return 1;

  std::cout << "---------------------------------" << std::endl;

  for (const auto &bb : graph.getBBs()) {
    std::cout << "Boundries: " << bb.boundries.at(0) << ", "
              << bb.boundries.at(1) << ", " << bb.boundries.at(2) << ", "
              << bb.boundries.at(3) << ", " << std::endl;
    std::cout << "Label: " << bb.label << std::endl;
    std::cout << "Label position: " << bb.labelPosition.first << ", "
              << bb.labelPosition.second << std::endl;
    std::cout << "Label size: " << bb.labelSize.first << ", "
              << bb.labelSize.second << std::endl;

    std::cout << "---------------------------------" << std::endl;
  }

  std::cout << "---------------------------------" << std::endl;

  for (const auto &pair : graph.getNodes()) {
    NodeId nodeId = pair.first;
    GraphNode node = pair.second;

    std::cout << "Node ID: " << nodeId << std::endl;
    std::cout << "Position: " << node.getPosition().first << " "
              << node.getPosition().second << std::endl;
    std::cout << "InPorts: " << node.getPorts(true).size()
              << " OutPorts: " << node.getPorts(false).size() << std::endl;
    std::cout << "Width: " << node.getWidth() << std::endl;
    std::cout << "Color: " << node.getColor() << std::endl;
    std::cout << "Shape: " << node.getShape() << std::endl;
    std::cout << "Style: " << node.getDashed() << std::endl;

    std::cout << "---------------------------------" << std::endl;
  }

  std::cout << "---------------------------------" << std::endl;

  for (GraphEdge &edge : graph.getEdges()) {
    std::cout << "Edge ID: " << edge.getEdgeId() << std::endl;
    std::cout << "Positions: " << std::endl;
    for (const auto &pos : edge.getPositions()) {
      std::cout << pos.first << " " << pos.second << std::endl;
    }
    std::cout << "Source Node: " << edge.getSrcNode().getNodeId() << std::endl;
    std::cout << "Port: " << edge.getOutPort() << std::endl;
    std::cout << "Destination Node: " << edge.getDstNode().getNodeId()
              << std::endl;
    std::cout << "Port: " << edge.getInPort() << std::endl;
    std::cout << "Style: " << edge.getDashed() << std::endl;

    std::cout << "---------------------------------" << std::endl;
  }
}
