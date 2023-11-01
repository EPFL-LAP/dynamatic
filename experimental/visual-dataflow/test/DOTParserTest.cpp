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

  reformatDot(inputDotFile, outputDotFile);

  std::ifstream f;
  f.open(outputDotFile);

  Graph graph(1);

  LogicalResult result = processDOT(f, graph);

  std::cout << "---------------------------------" << std::endl;
  for (const auto &pair : graph.getNodes()) {
    NodeId nodeId = pair.first;
    GraphNode node = pair.second;

    std::cout << "Node ID: " << nodeId << std::endl;
    std::cout << "Position: " << node.getPosition().first << " "
              << node.getPosition().second << std::endl;
    std::cout << "InPorts: " << node.getPorts(true).size()
              << " OutPorts: " << node.getPorts(false).size() << std::endl;

    std::cout << "---------------------------------" << std::endl;
  }

  std::cout << "---------------------------------" << std::endl;

  std::vector<GraphEdge> edges = graph.getEdges(); // Get a copy of the edges

  for (size_t i = 0; i < edges.size(); ++i) {
    GraphEdge edge = edges[i]; // Create a non-const copy of the edge

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

    std::cout << "---------------------------------" << std::endl;
  }
}
