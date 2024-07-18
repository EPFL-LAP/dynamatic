#include "experimental/Transforms/LSQSizing/LSQSizingSupport.h"
#include "experimental/Transforms/LSQSizing/HandshakeSizeLSQs.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"


using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::lsqsizing;


void AdjListGraph::addNode(mlir::Operation *op, int latency) {
    nodes.insert({std::string(op->getName().getStringRef()), AdjListNode{latency, op, {}}});
}

void AdjListGraph::addArtificialNode(std::string name, int latency) {
    nodes.insert({name, AdjListNode{latency, std::nullopt, {}}});
}

// TODO get handshake.name attribute instead of op name
void AdjListGraph::addEdge(mlir::Operation * src, mlir::Operation * dest) {
    nodes.at(std::string(src->getName().getStringRef())).adjList.push_back(std::string(dest->getName().getStringRef())); // Add edge from node u to node v
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