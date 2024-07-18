#include "experimental/Transforms/LSQSizing/LSQSizingSupport.h"
#include "experimental/Transforms/LSQSizing/HandshakeSizeLSQs.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"


using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::lsqsizing;


void AdjListGraph::addNode(std::string op_name, int latency, mlir::Operation *op) {
    nodes.insert({op_name, AdjListNode{latency, op, {}}});
}

void AdjListGraph::addNode(std::string op_name, int latency) {
    nodes.insert({op_name, AdjListNode{latency, std::nullopt, {}}});
}

void AdjListGraph::addEdge(std::string src, std::string dest) {
    nodes.at(src).adjList.push_back(dest); // Add edge from node u to node v
}

void AdjListGraph::printGraph() {
    for (const auto& pair : nodes) {
        std::string op_name = pair.first;
        const AdjListNode& node = pair.second;
        llvm::dbgs() << "Node " << op_name << " (latency: " << node.latency << "): ";
        for (std::string adj : node.adjList) {
            llvm::dbgs() << adj << " ";
        }
        llvm::dbgs() << "\n";
    }
}