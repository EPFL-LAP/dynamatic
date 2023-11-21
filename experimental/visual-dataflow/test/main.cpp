#include "../src/Graph.h"
#include "../src/GraphEdge.h"
#include "../src/GraphNode.h"
#include "../src/GraphParser.h"
#include "../src/MLIRMapper.h"
#include "dynamatic/Support/DOTPrinter.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <fstream>
#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::visual_dataflow;

/*void saveGraphToFile(const Graph *graph) {
  std::ofstream outFile(
      "./output.txt"); // Ouvrir le fichier en mode écriture

  if (!outFile.is_open()) {
    std::cout << "could not open output file\n";
    return;
  }

  for (const auto &pair : graph->nodes) {
    std::string key = pair.first;
    const GraphNode nodePtr = pair.second;

    outFile << "Node label: " << nodePtr.id
            << ", Node pos: " << nodePtr.position.first << " "
            << nodePtr.position.second
            << ", Node in/out:" << nodePtr.inPorts.size() << " "
            << nodePtr.outPorts.size() << std::endl;
  }

  outFile << " " << std::endl;

  // Iterate through the map and print each pair
  for (const auto &pair : graph->edges) {
    const GraphEdge edgePtr = pair;

    outFile << "src:" << edgePtr.src.id << ", dst:" << edgePtr.dst.id
            << ", from:" << edgePtr.outPort << ", to:" << edgePtr.inPort
            << std::endl;

    for (const std::pair<float, float> &pos : edgePtr.position) {
      outFile << "  Pos(x=" << pos.first << ", y=" << pos.second << ")"
              << std::endl;
    }
  }

  outFile.close(); // Fermer le fichier
}*/

void saveTransitionsToFile(Graph *graph) {
  std::ofstream outFile("./output.txt");

  if (!outFile.is_open()) {
    std::cout << "could not open output file\n";
    return;
  }

  for (const auto &pair : graph->getCycleEdgeStates()) {
    CycleNb cycleNb = pair.first;
    const std::map<EdgeId, State> &edgeStates = pair.second;

    outFile << "Cycle: " << cycleNb << std::endl;

    for (const auto &edgeStatePair : edgeStates) {
      EdgeId edgeId = edgeStatePair.first;
      State state = edgeStatePair.second;

      outFile << "  Edge: " << edgeId << ", State: " << state << std::endl;
    }
  }

  outFile.close();
}

int main() {
  std::string inputDOT = "../../../experimental/"
                         "visual-dataflow/test/bicg.dot";
  std::string inputTransitions = "../../../experimental/visual-dataflow/test/"
                                 "transitions.csv";

  Graph graph = Graph();
  GraphParser parser(&graph);

  if (failed(parser.parse(inputDOT))) {
    std::cout << "could not parse the graph\n";
    return -1;
  }

  if (failed(parser.parse(inputTransitions))) {
    std::cout << "could not parse the transitions\n";
    return -1;
  }

  // Save the graph to a file
  // saveGraphToFile(&graph);

  // Save transitions to a file
  saveTransitionsToFile(&graph);
}