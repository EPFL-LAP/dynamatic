#include "experimental/Transforms/LSQSizing/LSQSizingSupport.h"
#include "experimental/Transforms/LSQSizing/HandshakeSizeLSQs.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"

#include <unordered_set>
#include <stack>
#include <set>

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
        llvm::dbgs()  << op_name << " (lat: " << node.latency << "): ";
        for (std::string adj : node.adjList) {
            llvm::dbgs() << adj << ", ";
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


std::vector<std::vector<std::string>> AdjListGraph::findPaths(std::string start, std::string end) {

  std::vector<std::vector<std::string>> paths;
  std::stack<std::pair<std::vector<std::string>, std::set<std::string>>> pathStack;

  // Initialize the stack with the path containing the source node
  pathStack.push({{start}, {start}});

  while (!pathStack.empty()) {
    // Get the current path and visited set from the stack
    auto [currentPath, visited] = pathStack.top();
    pathStack.pop();
    // Get the last node in the current path
    std::string currentNode = currentPath.back();
    // If the current node is the target, add the path to allPaths
    if (currentNode == end) {
        paths.push_back(currentPath);
        continue;
    }
    // Get all adjacent nodes of the current node
    for (const std::string& neighbor : nodes.at(currentNode).adjList) {
        // If the neighbor has not been visited in the current path, extend the path
        if (visited.find(neighbor) == visited.end()) {
            std::vector<std::string> newPath = currentPath;
            newPath.push_back(neighbor);
            std::set<std::string> newVisited = visited;
            newVisited.insert(neighbor);
            // Push the new path and updated visited set onto the stack
            pathStack.push({newPath, newVisited});
        }
    }
  }
  return paths;
}

std::vector<std::vector<std::string>> AdjListGraph::findPaths(mlir::Operation *start_op, mlir::Operation *end_op) {
  return findPaths(start_op->getAttrOfType<StringAttr>("handshake.name").str(), end_op->getAttrOfType<StringAttr>("handshake.name").str());
}


int AdjListGraph::getPathLatency(std::vector<std::string> path) {
  int latency = 0;
  for(auto &node: path) {
    latency += nodes.at(node).latency;
  }
  return latency;
}

std::vector<mlir::Operation*> AdjListGraph::getOperationsWithOpName(std::string op_name) {
  std::vector<mlir::Operation*> ops;
  for(auto &node: nodes) {
    if(node.second.op && std::string(node.second.op.value()->getName().getStringRef()) == op_name)
    {
      ops.push_back(node.second.op.value());
    }
  }
  return ops;
}


int AdjListGraph::findMaxPathLatency(mlir::Operation *start_op, mlir::Operation *end_op) {
  std::vector<std::vector<std::string>> paths = findPaths(start_op, end_op);
  int max_latency = 0;
  for(auto &path: paths)
  {
    max_latency = std::max(max_latency, getPathLatency(path));
  }
  return max_latency;
}

int AdjListGraph::findMinPathLatency(mlir::Operation *start_op, mlir::Operation *end_op) {
  std::vector<std::vector<std::string>> paths = findPaths(start_op, end_op);
  int min_latency = INT_MAX;
  for(auto &path: paths)
  {
    min_latency = std::min(min_latency, getPathLatency(path));
  }
  return min_latency;
}

