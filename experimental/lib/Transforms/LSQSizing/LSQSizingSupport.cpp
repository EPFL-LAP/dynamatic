//===- LSQSizingSupport.cpp - Support functions for LSQ Sizing -*-- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements Helper Classes and Functions for the LSQ sizing pass.
// The Helper functions mainly consist of the CFDFCGraph class which is used
// to represent the CFDFC as an adjacency list graph and provides functions to
// find paths, calculate latencies and start times of nodes.
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/LSQSizing/LSQSizingSupport.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Transforms/LSQSizing/HandshakeSizeLSQs.h"

#include <set>
#include <stack>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::lsqsizing;

/// Extracts the latency for each operation
/// This is done in 3 ways:
/// 1. If the operation is in the timingDB, the latency is extracted from the
/// timingDB
/// 2. If the operation is a buffer operation, the latency is extracted from the
/// timing attribute
/// 3. If the operation is neither, then its latency is set to 0
static int extractNodeLatency(mlir::Operation *op, TimingDatabase timingDB,
                              double targetCP) {
  double latency = 0;

  if (!failed(timingDB.getLatency(op, SignalType::DATA, latency, targetCP))) {
    return latency;
  }

  if (isa<handshake::BufferOp>(op)) {
    auto params = op->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
    if (!params)
      return 0;

    auto optTiming = params.getNamed(handshake::BufferOp::TIMING_ATTR_NAME);
    if (!optTiming)
      return 0;

    if (auto timing = dyn_cast<handshake::TimingAttr>(optTiming->getValue())) {
      handshake::TimingInfo info = timing.getInfo();
      return info.getLatency(SignalType::DATA).value_or(0);
    }
  }

  return 0;
}

CFDFCGraph::CFDFCGraph(handshake::FuncOp funcOp,
                       llvm::SetVector<unsigned> cfdfcBBs,
                       TimingDatabase timingDB, unsigned II, double targetCP) {

  for (Operation &op : funcOp.getOps()) {
    // Get operation's basic block
    unsigned srcBB;
    if (auto optBB = getLogicBB(&op); !optBB.has_value())
      continue;
    else
      srcBB = *optBB;

    // The basic block the operation belongs to must be selected
    if (!cfdfcBBs.contains(srcBB))
      continue;

    // Add the unit and valid outgoing channels to the CFDFC
    addNode(&op, extractNodeLatency(&op, timingDB, targetCP));

    for (OpResult res : op.getResults()) {
      assert(std::distance(res.getUsers().begin(), res.getUsers().end()) == 1 &&
             "value must have unique user");

      // Get the value's unique user and its basic block
      Operation *user = *res.getUsers().begin();
      unsigned dstBB;
      if (std::optional<unsigned> optBB = getLogicBB(user); !optBB.has_value())
        continue;
      else
        dstBB = *optBB;

      if (srcBB != dstBB) {
        // The channel is in the CFDFC if it belongs belong to a selected arch
        // between two basic blocks
        for (size_t i = 0; i < cfdfcBBs.size(); ++i) {
          unsigned nextBB = i == cfdfcBBs.size() - 1 ? 0 : i + 1;
          if (srcBB == cfdfcBBs[i] && dstBB == cfdfcBBs[nextBB]) {
            addChannelEdges(res);
            if (buffer::CFDFC::isCFDFCBackedge(res))
              addChannelBackedges(res, (II * -1));
            break;
          }
        }
      } else if (cfdfcBBs.size() == 1) {
        // The channel is in the CFDFC if its producer/consumer belong to the
        // same basic block and the CFDFC is just a block looping to itself
        addChannelEdges(res);
        if (buffer::CFDFC::isCFDFCBackedge(res))
          addChannelBackedges(res, (II * -1));
      } else if (!isBackedge(res)) {
        // The channel is in the CFDFC if its producer/consumer belong to the
        // same basic block and the channel is not a backedge
        addChannelEdges(res);
      }
    }
  }
}

void CFDFCGraph::addNode(mlir::Operation *op, int latency) {
  nodes.insert(
      {getUniqueName(op).str(), CFDFCNode{latency, -1, op, {}, {}, {}}});
}

void CFDFCGraph::addEdge(mlir::Operation *src, mlir::Operation *dest) {
  nodes.at(getUniqueName(src).str())
      .edges.insert(
          getUniqueName(dest).str()); // Add edge from node u to node v
}

void CFDFCGraph::addChannelEdges(mlir::Value res) {
  mlir::Operation *srcOp = res.getDefiningOp();
  for (Operation *destOp : res.getUsers())
    addEdge(srcOp, destOp);
}

void CFDFCGraph::addChannelBackedges(mlir::Value res, int latency) {
  mlir::Operation *srcOp = res.getDefiningOp();
  for (Operation *destOp : res.getUsers())
    addBackedge(srcOp, destOp, latency);
}

void CFDFCGraph::printGraph() {
  for (const auto &pair : nodes) {
    std::string opName = pair.first;
    const CFDFCNode &node = pair.second;
    llvm::errs() << opName << " (lat: " << node.latency
                 << ", est: " << node.earliestStartTime << "): ";
    for (std::string edge : node.edges)
      llvm::errs() << edge << ", ";

    if (node.backedges.size() > 0) {
      llvm::errs() << " || ";
      for (std::string backedge : node.backedges)
        llvm::errs() << backedge << ", ";
    }
    llvm::errs() << "\n";
  }
}

void CFDFCGraph::printPath(std::vector<std::string> path) {
  for (std::string node : path)
    llvm::errs() << node << "(" << nodes.at(node).latency << ") - ";

  llvm::errs() << "\n";
}

void CFDFCGraph::addBackedge(mlir::Operation *src, mlir::Operation *dest,
                             int latency) {
  // create new node name from src and dest name
  std::string srcName = getUniqueName(src).str();
  std::string destName = getUniqueName(dest).str();
  std::string newNodeName = backedgePrefix + srcName + "_" + destName;

  // remove existing edges from src to dest
  nodes.at(srcName).edges.erase(destName);
  nodes.at(srcName).backedges.erase(destName);

  // create node and add edge from src to new node and new node to dest
  nodes.insert(
      {newNodeName, CFDFCNode{latency, -1, nullptr, {}, {destName}, {}}});
  nodes.at(srcName).backedges.insert(newNodeName);
}

void CFDFCGraph::addShiftingEdge(mlir::Operation *src, mlir::Operation *dest,
                                 int shiftingLatency) {
  // create new node name from src and dest name
  std::string srcName = getUniqueName(src).str();
  std::string destName = getUniqueName(dest).str();
  std::string newNodeName = "shifting_" + srcName + "_" + destName;

  // remove existing edges from src to
  assert(nodes.at(srcName).edges.find(destName) ==
             nodes.at(srcName).edges.end() &&
         "Nodes are already connected -> should not add shifting edge");

  // create node and add edge from src to new node and new node to dest
  nodes.insert({newNodeName,
                CFDFCNode{shiftingLatency, -1, nullptr, {}, {}, {destName}}});
  nodes.at(srcName).shiftingedges.insert(newNodeName);
}

void CFDFCGraph::dfsAllPaths(std::string &currentNode, std::string &end,
                             std::vector<std::string> &currentPath,
                             std::set<std::string> &visited,
                             std::vector<std::vector<std::string>> &paths,
                             bool ignoreBackedges, bool ignoreShiftingEdge) {
  // If the current node is the target, add the current path to paths and
  // return.
  if (currentNode == end) {
    paths.push_back(currentPath);
    return;
  }

  // Iterate over all adjacent nodes
  for (auto neighbor : nodes.at(currentNode).edges) {
    // If the neighbor has not been visited, visit it
    if (visited.find(neighbor) == visited.end()) {
      visited.insert(neighbor);        // Mark as visited
      currentPath.push_back(neighbor); // Add to the current pat
      // Recursively visit the neighbor
      dfsAllPaths(neighbor, end, currentPath, visited, paths, ignoreBackedges,
                  ignoreShiftingEdge);
      // Backtrack: remove the neighbor from the current path and visited set
      // for other paths
      currentPath.pop_back();
      visited.erase(neighbor);
    }
  }

  if (!ignoreBackedges) {
    for (auto neighbor : nodes.at(currentNode).backedges) {
      // If the neighbor has not been visited, visit it
      if (visited.find(neighbor) == visited.end()) {
        visited.insert(neighbor);        // Mark as visited
        currentPath.push_back(neighbor); // Add to the current pat
        // Recursively visit the neighbor
        dfsAllPaths(neighbor, end, currentPath, visited, paths, ignoreBackedges,
                    ignoreShiftingEdge);
        // Backtrack: remove the neighbor from the current path and visited set
        // for other paths
        currentPath.pop_back();
        visited.erase(neighbor);
      }
    }
  }

  if (!ignoreShiftingEdge) {
    for (auto neighbor : nodes.at(currentNode).shiftingedges) {
      // If the neighbor has not been visited, visit it
      if (visited.find(neighbor) == visited.end()) {
        visited.insert(neighbor);        // Mark as visited
        currentPath.push_back(neighbor); // Add to the current pat
        // Recursively visit the neighbor
        dfsAllPaths(neighbor, end, currentPath, visited, paths, ignoreBackedges,
                    ignoreShiftingEdge);
        // Backtrack: remove the neighbor from the current path and visited set
        // for other paths
        currentPath.pop_back();
        visited.erase(neighbor);
      }
    }
  }
}

std::vector<std::vector<std::string>>
CFDFCGraph::findPaths(std::string start, std::string end, bool ignoreBackedge,
                      bool ignoreShiftingEdge) {
  // Initialize the vectors for DFS
  std::vector<std::vector<std::string>> paths;
  std::vector<std::string> currentPath{start};
  std::set<std::string> visited{start};

  // Call DFS to find all paths
  dfsAllPaths(start, end, currentPath, visited, paths, ignoreBackedge,
              ignoreShiftingEdge);
  return paths;
}

std::vector<std::vector<std::string>>
CFDFCGraph::findPaths(mlir::Operation *startOp, mlir::Operation *endOp,
                      bool ignoreBackedge, bool ignoreShiftingEdge) {
  assert(startOp && endOp && "Start and end operations must not be null");
  return findPaths(getUniqueName(startOp).str(), getUniqueName(endOp).str(),
                   ignoreBackedge, ignoreShiftingEdge);
}

// Recursive helper function
void CFDFCGraph::dfsLongestAcyclicPath(const std::string &currentNode,
                                       std::set<std::string> &visited,
                                       std::vector<std::string> &currentPath,
                                       int &maxLatency,
                                       std::vector<std::string> &bestPath) {
  visited.insert(currentNode);
  currentPath.push_back(currentNode);
  // Update the best path if current path latency is greater than maxLatency
  int currentLatency = getPathLatency(currentPath);
  if (currentLatency > maxLatency) {
    maxLatency = currentLatency;
    bestPath = currentPath;
  }
  // Recursively explore neighbors
  for (const std::string &neighbor : nodes[currentNode].edges) {
    if (visited.find(neighbor) == visited.end()) {
      dfsLongestAcyclicPath(neighbor, visited, currentPath, maxLatency,
                            bestPath);
    }
  }
  // Backtrack: remove the current node from the path and visited set
  visited.erase(currentNode);
  currentPath.pop_back();
}

// Main function to find the longest non-cyclic path
std::vector<std::string>
CFDFCGraph::findLongestNonCyclicPath(mlir::Operation *startOp) {
  std::string start = getUniqueName(startOp).str();
  std::vector<std::string> bestPath;
  std::vector<std::string> currentPath;
  std::set<std::string> visited;
  int maxLatency = 0;
  // Start DFS from the start node
  dfsLongestAcyclicPath(start, visited, currentPath, maxLatency, bestPath);
  return bestPath;
}

int CFDFCGraph::getPathLatency(std::vector<std::string> path) {
  // Sum up the latencies of all nodes in the path
  int latency = 0;
  for (auto &node : path)
    latency += nodes.at(node).latency;

  return latency;
}

int CFDFCGraph::findMaxPathLatency(mlir::Operation *startOp,
                                   mlir::Operation *endOp, bool ignoreBackedge,
                                   bool ignoreShiftingEdge,
                                   bool excludeLastNodeLatency) {

  std::vector<std::vector<std::string>> paths;

  // Find all paths between the start and end node
  if (!ignoreBackedge && !ignoreShiftingEdge) {
    std::vector<std::vector<std::string>> paths1 =
        findPaths(startOp, endOp, false, true);
    std::vector<std::vector<std::string>> paths2 =
        findPaths(startOp, endOp, true, false);
    paths.insert(paths.end(), paths1.begin(), paths1.end());
    paths.insert(paths.end(), paths2.begin(), paths2.end());
  } else {
    std::vector<std::vector<std::string>> paths1 =
        findPaths(startOp, endOp, ignoreBackedge, ignoreShiftingEdge);
    paths.insert(paths.end(), paths1.begin(), paths1.end());
  }

  int maxLatency = 0;

  // Iterate over all paths and keep track of the path with the highest latency
  for (auto &path : paths) {
    if (excludeLastNodeLatency)
      path.pop_back();

    maxLatency = std::max(maxLatency, getPathLatency(path));
  }
  return maxLatency;
}

int CFDFCGraph::findMinPathLatency(mlir::Operation *startOp,
                                   mlir::Operation *endOp, bool ignoreBackedge,
                                   bool ignoreShiftingEdge) {
  // Find all paths between the start and end node
  std::vector<std::vector<std::string>> paths =
      findPaths(startOp, endOp, ignoreBackedge, ignoreShiftingEdge);
  int minLatency = INT_MAX;
  // Iterate over all paths and keep track of the lowest latency
  for (auto &path : paths)
    minLatency = std::min(minLatency, getPathLatency(path));

  return minLatency;
}

std::vector<mlir::Operation *>
CFDFCGraph::getConnectedOps(mlir::Operation *op) {
  std::vector<mlir::Operation *> connectedOps;
  std::string opName = getUniqueName(op).str();

  // Get all Ops which are connected via a regular edge
  for (auto &node : nodes.at(opName).edges)
    connectedOps.push_back(nodes.at(node).op);

  // Get all Ops which are connected via a backedge, by skipping the artificial
  // nodes and going over the nodes connected to the artificial nodes
  for (auto &aritificalNode : nodes.at(opName).backedges)
    for (auto &node : nodes.at(aritificalNode).backedges)
      connectedOps.push_back(nodes.at(node).op);

  return connectedOps;
}

std::vector<mlir::Operation *> CFDFCGraph::getOperations() {
  std::vector<mlir::Operation *> ops;
  for (auto &node : nodes)
    if (node.second.op)
      ops.push_back(node.second.op);

  return ops;
}

void CFDFCGraph::setNewII(unsigned II) {
  for (auto &node : nodes)
    if (node.first.find(backedgePrefix) != std::string::npos)
      node.second.latency = II * -1;
}

// Example of how the worst case II is calculated
// When there is a memory dependecy, the II is the distance between load (L) and
// store (S). The Load of the next iteration (L') is dependent on the store of
// the current iteration (S), therefore L' can only start after S has finished.
//
// IT1: #L#####S#
// Dist: |<--->|
// IT2:        #L'#####S'#
unsigned CFDFCGraph::getWorstCaseII() {

  unsigned maxLatency = 0;

  // Iterate through all nodes to find LSQOps and their associated loads and
  // stores
  for (auto &node : nodes) {
    mlir::Operation *lsqOp = nullptr;

    if (node.second.op) {
      // Find the associated LSQOp for the current node operation
      for (mlir::Operation *destOp : node.second.op->getUsers()) {
        if (isa<handshake::LSQOp>(destOp)) {
          lsqOp = destOp;
          break;
        }
      }

      if (!lsqOp)
        continue;

      unsigned tempMaxLatency = 0;

      // Check if the node operation is a load or store and compute latencies
      if (isa<handshake::LoadOp>(node.second.op)) {
        // Iterate through nodes again to find corresponding stores for this
        // LSQOp
        for (auto &otherNode : nodes) {
          if (otherNode.second.op &&
              isa<handshake::StoreOp>(otherNode.second.op) &&
              isConnectedToLSQ(otherNode.second.op)) {
            // Calculate path latency between the load and store
            unsigned latency = findMaxPathLatency(
                node.second.op, otherNode.second.op, true, true);
            tempMaxLatency = std::max(tempMaxLatency, latency);
          }
        }
      }
      maxLatency = std::max(maxLatency, tempMaxLatency);
    }
  }

  // The maximal latency between any load and store of the same LSQ is the worst
  // case II
  return maxLatency;
}

void CFDFCGraph::setEarliestStartTimes(mlir::Operation *startOp) {
  std::string startNode = getUniqueName(startOp).str();
  // initialize the earliest start time of the start node to 0, this is the
  // reference for all other nodes
  nodes.at(startNode).earliestStartTime = 0;
  std::set<std::string> visited;
  setEarliestStartTimes(startNode, visited);
}

void CFDFCGraph::setEarliestStartTimes(std::string prevNode,
                                       std::set<std::string> &visited) {

  // If the current node has already been visited, skip to avoid cycles
  if (visited.find(prevNode) != visited.end())
    return;

  // Mark this node as visited
  visited.insert(prevNode);

  // Traverse the edges of the current node
  for (auto &node : nodes.at(prevNode).edges)
    if (updateStartTimeForNode(node, prevNode) || true)
      setEarliestStartTimes(node, visited);

  for (auto &node : nodes.at(prevNode).shiftingedges)
    if (updateStartTimeForNode(node, prevNode) || true)
      setEarliestStartTimes(node, visited);

  // Remove node from visited set, to allow for other paths to visit it
  visited.erase(prevNode);
}

bool CFDFCGraph::updateStartTimeForNode(std::string node,
                                        std::string prevNode) {

  mlir::Operation *op = nodes.at(node).op;
  bool startTimeUpdated = false;

  int prevNodePathTime =
      nodes.at(prevNode).earliestStartTime + nodes.at(prevNode).latency;
  assert(nodes.at(prevNode).earliestStartTime != -1 &&
         "Previous start time must be defined");

  // Update the edgeMinLatencies map
  // This map holds the earliest argument arrival time for each incoming edge
  edgeMinLatencies[node][prevNode] =
      std::max(edgeMinLatencies[node][prevNode], prevNodePathTime);

  // Mux, merge and cmerge only need one of their inputs to be ready, all
  // other nodes need all of their inputs to be ready
  // Therefore the latency for mux, merge and cmerge is the minimum of all
  // incoming edges
  if (op && isa<handshake::MergeLikeOpInterface>(op)) {
    int minLatency = INT_MAX;
    for (auto &incomgingEdge : edgeMinLatencies[node])
      if (minLatency > incomgingEdge.second)
        minLatency = incomgingEdge.second;

    if (minLatency != nodes.at(node).earliestStartTime) {
      startTimeUpdated = true;
      nodes.at(node).earliestStartTime = minLatency;
    }
  } else {
    // For all other nodes, the operation needs to wait for all arguments to
    // arrive. The latency is the maximum of all incoming edges
    int maxLatency = 0;
    for (auto &incomgingEdge : edgeMinLatencies[node])
      if (maxLatency < incomgingEdge.second)
        maxLatency = incomgingEdge.second;

    if (maxLatency != nodes.at(node).earliestStartTime) {
      startTimeUpdated = true;
      nodes.at(node).earliestStartTime = maxLatency;
    }
  }
  return startTimeUpdated;
}

int CFDFCGraph::getEarliestStartTime(mlir::Operation *op) {
  std::string opName = getUniqueName(op).str();
  // If earliestStartTime of node has not been initialized (is -1), it is not
  // reachable by the start node. This should only happen for nodes that are
  // e.g. a source. The not reachable muxes, merges and cmerges are handled by
  // adding shifting edges for their start times and should be accounted fo
  return nodes.at(opName).earliestStartTime == -1
             ? 0
             : nodes.at(opName).earliestStartTime;
}

bool CFDFCGraph::isConnectedToLSQ(mlir::Operation *op) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    auto memOp = findMemInterface(loadOp.getAddressResult());
    if (isa_and_present<handshake::LSQOp>(memOp))
      return true;
  }
  if (auto storeOp = dyn_cast<handshake::StoreOp>(op)) {
    auto memOp = findMemInterface(storeOp.getAddressResult());
    if (isa_and_present<handshake::LSQOp>(memOp))
      return true;
  }
  return false;
}