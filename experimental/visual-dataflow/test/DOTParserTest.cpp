#include "../src/DOTParser.h"
#include "../src/Graph.h"
#include "../src/GraphEdge.h"
#include "../src/GraphNode.h"
#include "../src/dotReformat.h"
#include "mlir/Support/LogicalResult.h"
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace dynamatic::experimental::visual_dataflow;

int main() {

  reformatDot("experimental/visual-dataflow/src/gcd.dot", "outputFinal.dot");

  std::ifstream f;
  f.open("outputFinal.dot");

  Graph graph(1);

  LogicalResult result = DOTParser::processDOT(f, &graph);

  std::cout << "---------------------------------" << std::endl;

  for (const auto &pair : graph.getNodes()) {
    NodeId nodeId = pair.first;
    GraphNode *node = pair.second;

    std::cout << "Node ID: " << nodeId << std::endl;
    std::cout << "Position: " << node->getPosition().first << " "
              << node->getPosition().second << std::endl;
    std::cout << "InPorts: " << node->getPort(true).size()
              << " OutPorts: " << node->getPort(false).size() << std::endl;

    std::cout << "---------------------------------" << std::endl;
  }

  std::cout << "---------------------------------" << std::endl;

  for (const auto &edge : graph.getEdges()) {

    std::cout << "Edge ID: " << edge->getEdgeId() << std::endl;
    std::cout << "Positions: " << std::endl;
    for (const auto &pos : edge->getPositions()) {
      std::cout << pos.first << " " << pos.second << std::endl;
    }
    std::cout << "Source Node: " << edge->getSrcNode()->getNodeId()
              << std::endl;
    std::cout << "Port: " << edge->getOutPort() << std::endl;
    std::cout << "Destination Node: " << edge->getDstNode()->getNodeId()
              << std::endl;
    std::cout << "Port: " << edge->getInPort() << std::endl;

    std::cout << "---------------------------------" << std::endl;
  }
}
