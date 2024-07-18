#include "experimental/Transforms/LSQSizing/LSQSizingSupport.h"
#include "experimental/Transforms/LSQSizing/HandshakeSizeLSQs.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"


using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::lsqsizing;


void AdjListGraph::addNode(mlir::Operation *op, int latency) {
    nodes.insert({op->getAttrOfType<StringAttr>("handshake.name").str(), AdjListNode{latency, op, {}}});
}

void AdjListGraph::addEdge(mlir::Operation * src, mlir::Operation * dest) {
    nodes.at(src->getAttrOfType<StringAttr>("handshake.name").str()).adjList.push_back(dest->getAttrOfType<StringAttr>("handshake.name").str()); // Add edge from node u to node v
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

void AdjListGraph::insertArtificialNodeOnEdge(mlir::Operation* src, mlir::Operation* dest, int latency) {
  // create new node name from src and dest name
  std::string src_name = src->getAttrOfType<StringAttr>("handshake.name").str();
  std::string dest_name = dest->getAttrOfType<StringAttr>("handshake.name").str();
  std::string new_node_name = "backedge_" + src_name + "_" + dest_name;

  // remove edge if exists TODO what happens if edge does not exist?
  nodes.at(src_name).adjList.remove(dest_name);

  // create node and add edge from src to new node and new node to dest
  nodes.insert({new_node_name, AdjListNode{latency, nullptr, {dest_name}}});
  nodes.at(src_name).adjList.push_back(new_node_name);
}
