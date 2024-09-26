#include "experimental/Transforms/LSQSizing/LSQSizingSupport.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Transforms/LSQSizing/HandshakeSizeLSQs.h"

#include <set>
#include <stack>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::lsqsizing;

// Extracts the latency for each operation
// This is done in 3 ways:
// 1. If the operation is in the timingDB, the latency is extracted from the
// timingDB
// 2. If the operation is a buffer operation, the latency is extracted from the
// timing attribute
// 3. If the operation is neither, then its latency is set to 0
int extractNodeLatency(mlir::Operation *op, TimingDatabase timingDB) {
  double latency = 0;

  if (!failed(timingDB.getLatency(op, SignalType::DATA, latency)))
    return latency;

  if (op->getName().getStringRef() == "handshake.buffer") {
    auto params = op->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
    if (!params) {
      return 0;
    }

    auto optTiming = params.getNamed(handshake::BufferOp::TIMING_ATTR_NAME);
    if (!optTiming) {
      return 0;
    }

    if (auto timing = dyn_cast<handshake::TimingAttr>(optTiming->getValue())) {
      handshake::TimingInfo info = timing.getInfo();
      return info.getLatency(SignalType::DATA).value_or(0);
    }
  }

  return 0;
}

AdjListGraph::AdjListGraph(handshake::FuncOp funcOp,
                           llvm::SetVector<unsigned> cfdfcBBs,
                           TimingDatabase timingDB, unsigned II) {

  llvm::dbgs() << "Creating AdjListGraph for CFDFC: ";
  for (auto &bb : cfdfcBBs) {
    llvm::dbgs() << bb << " ";
  }
  llvm::dbgs() << "with II: " << II << "\n";

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
    addNode(&op, extractNodeLatency(&op, timingDB));

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

void AdjListGraph::addNode(mlir::Operation *op, int latency) {
  nodes.insert(
      {getUniqueName(op).str(), AdjListNode{latency, -1, op, {}, {}, {}}});
}

void AdjListGraph::addEdge(mlir::Operation *src, mlir::Operation *dest) {
  nodes.at(getUniqueName(src).str())
      .edges.insert(
          getUniqueName(dest).str()); // Add edge from node u to node v
}

void AdjListGraph::addChannelEdges(mlir::Value res) {
  mlir::Operation *srcOp = res.getDefiningOp();
  for (Operation *destOp : res.getUsers()) {
    addEdge(srcOp, destOp);
  }
}

void AdjListGraph::addChannelBackedges(mlir::Value res, int latency) {
  mlir::Operation *srcOp = res.getDefiningOp();
  for (Operation *destOp : res.getUsers()) {
    addBackedge(srcOp, destOp, latency);
  }
}

void AdjListGraph::printGraph() {
  for (const auto &pair : nodes) {
    std::string opName = pair.first;
    const AdjListNode &node = pair.second;
    llvm::dbgs() << opName << " (lat: " << node.latency
                 << ", est: " << node.earliestStartTime << "): ";
    for (std::string edge : node.edges) {
      llvm::dbgs() << edge << ", ";
    }
    if (node.backedges.size() > 0) {
      llvm::dbgs() << " || ";
      for (std::string backedge : node.backedges) {
        llvm::dbgs() << backedge << ", ";
      }
    }
    llvm::dbgs() << "\n";
  }
}

void AdjListGraph::printPath(std::vector<std::string> path) {
  for (std::string node : path) {
    llvm::dbgs() << node << "(" << nodes.at(node).latency << ") - ";
  }
  llvm::dbgs() << "\n";
}

void AdjListGraph::addBackedge(mlir::Operation *src, mlir::Operation *dest,
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
      {newNodeName, AdjListNode{latency, -1, nullptr, {}, {destName}, {}}});
  nodes.at(srcName).backedges.insert(newNodeName);
}

void AdjListGraph::addShiftingEdge(mlir::Operation *src, mlir::Operation *dest,
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
                AdjListNode{shiftingLatency, -1, nullptr, {}, {}, {destName}}});
  nodes.at(srcName).shiftingEdges.insert(newNodeName);
}

void AdjListGraph::dfsAllPaths(std::string &currentNode, std::string &end,
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
    for (auto neighbor : nodes.at(currentNode).shiftingEdges) {
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
AdjListGraph::findPaths(std::string start, std::string end, bool ignoreBackedge,
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
AdjListGraph::findPaths(mlir::Operation *startOp, mlir::Operation *endOp,
                        bool ignoreBackedge, bool ignoreShiftingEdge) {
  assert(startOp && endOp && "Start and end operations must not be null");
  // llvm::dbgs() << "Finding paths from " << getUniqueName(startOp).str() << "
  // to " << getUniqueName(endOp).str() << "\n";
  return findPaths(getUniqueName(startOp).str(), getUniqueName(endOp).str(),
                   ignoreBackedge, ignoreShiftingEdge);
}

// Recursive helper function
void AdjListGraph::dfsLongestAcyclicPath(const std::string &currentNode,
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
AdjListGraph::findLongestNonCyclicPath(mlir::Operation *startOp) {
  std::string start = getUniqueName(startOp).str();
  std::vector<std::string> bestPath;
  std::vector<std::string> currentPath;
  std::set<std::string> visited;
  int maxLatency = 0;
  // Start DFS from the start node
  dfsLongestAcyclicPath(start, visited, currentPath, maxLatency, bestPath);
  return bestPath;
}

int AdjListGraph::getPathLatency(std::vector<std::string> path) {
  // Sum up the latencies of all nodes in the path
  int latency = 0;
  for (auto &node : path) {
    latency += nodes.at(node).latency;
  }
  return latency;
}

std::vector<mlir::Operation *>
AdjListGraph::getOperationsWithOpName(std::string opName) {
  std::vector<mlir::Operation *> ops;
  // Iterate over all nodes and return the operations with the specified
  // operation name
  for (auto &node : nodes) {
    if (node.second.op &&
        std::string(node.second.op->getName().getStringRef()) == opName) {
      ops.push_back(node.second.op);
    }
  }
  return ops;
}

int AdjListGraph::findMaxPathLatency(mlir::Operation *startOp,
                                     mlir::Operation *endOp,
                                     bool ignoreBackedge,
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
  std::vector<std::string> maxPath;

  // Iterate over all paths and keep track of the path with the highest latency
  for (auto &path : paths) {
    if (excludeLastNodeLatency) {
      path.pop_back();
    }
    // TODO cleanup: remove maxPath and use std::max
    int latency = getPathLatency(path);
    if (maxLatency < latency) {
      maxLatency = latency;
      maxPath = path;
    }
  }

  llvm::dbgs() << "latency: " << maxLatency << " path: ";
  printPath(maxPath);

  return maxLatency;
}

int AdjListGraph::findMinPathLatency(mlir::Operation *startOp,
                                     mlir::Operation *endOp,
                                     bool ignoreBackedge,
                                     bool ignoreShiftingEdge) {
  // Find all paths between the start and end node
  std::vector<std::vector<std::string>> paths =
      findPaths(startOp, endOp, ignoreBackedge, ignoreShiftingEdge);
  int minLatency = INT_MAX;
  std::vector<std::string> minPath;
  // Iterate over all paths and keep track of the path with the lowest latency
  // TODO use std::min and remove minPath
  for (auto &path : paths) {
    int latency = getPathLatency(path);
    if (latency < minLatency) {
      minLatency = latency;
      minPath = path;
    }
    // minLatency = std::min(minLatency, getPathLatency(path));
  }
  printPath(minPath);
  return minLatency;
}

std::vector<mlir::Operation *>
AdjListGraph::getConnectedOps(mlir::Operation *op) {
  std::vector<mlir::Operation *> connectedOps;
  std::string opName = getUniqueName(op).str();

  // Get all Ops which are connected via a regular edge
  for (auto &node : nodes.at(opName).edges) {
    connectedOps.push_back(nodes.at(node).op);
  }

  // Get all Ops which are connected via a backedge, by skipping the artificial
  // nodes and going over the nodes connected to the artificial nodes
  for (auto &aritificalNode : nodes.at(opName).backedges) {
    for (auto &node : nodes.at(aritificalNode).backedges) {
      connectedOps.push_back(nodes.at(node).op);
    }
  }

  return connectedOps;
}

std::vector<mlir::Operation *> AdjListGraph::getOperations() {
  std::vector<mlir::Operation *> ops;
  for (auto &node : nodes) {
    if (node.second.op) {
      ops.push_back(node.second.op);
    }
  }
  return ops;
}

void AdjListGraph::setNewII(unsigned II) {
  for (auto &node : nodes) {
    if (node.first.find(backedgePrefix) != std::string::npos) {
      node.second.latency = II * -1;
    }
  }
}

unsigned AdjListGraph::getWorstCaseII() {

  std::unordered_map<mlir::Operation *,
                     std::tuple<std::vector<mlir::Operation *>,
                                std::vector<mlir::Operation *>>>
      loadStoreOpsPerLSQ;
  for (auto &node : nodes) {
    mlir::Operation *lsqOp = nullptr;
    if (node.second.op) {
      for (Operation *destOp : node.second.op->getUsers()) {
        if (destOp->getName().getStringRef().str() == "handshake.lsq") {
          lsqOp = destOp;
          break;
        }
      }

      if (loadStoreOpsPerLSQ.find(lsqOp) == loadStoreOpsPerLSQ.end()) {
        loadStoreOpsPerLSQ.insert(
            {lsqOp, std::make_tuple(std::vector<mlir::Operation *>(),
                                    std::vector<mlir::Operation *>())});
      }

      if (node.second.op->getName().getStringRef() == "handshake.lsq_load") {
        std::get<0>(loadStoreOpsPerLSQ[lsqOp]).push_back(node.second.op);
      } else if (node.second.op->getName().getStringRef() ==
                 "handshake.lsq_store") {
        std::get<1>(loadStoreOpsPerLSQ[lsqOp]).push_back(node.second.op);
      }
    }
  }

  int maxLatency = 0;

  for (auto &lsq : loadStoreOpsPerLSQ) {
    for (auto &load : std::get<0>(lsq.second)) {
      for (auto &store : std::get<1>(lsq.second)) {
        maxLatency =
            std::max(findMaxPathLatency(load, store, true, true), maxLatency);
      }
    }
  }
  // The maximal Latency between any load and store of the same LSQ is the worst
  // case II
  return (unsigned)maxLatency;
}

void AdjListGraph::setEarliestStartTimes(mlir::Operation *startOp) {
  std::string startNode = getUniqueName(startOp).str();
  nodes.at(startNode).earliestStartTime = 0;
  std::set<std::string> visited;
  setEarliestStartTimes(startNode, visited);
}

void AdjListGraph::setEarliestStartTimes(std::string prevNode,
                                         std::set<std::string> &visited) {

  // If the current node has already been visited, skip to avoid cycles
  if (visited.find(prevNode) != visited.end()) {
    return;
  }

  // Mark this node as visited
  visited.insert(prevNode);

  // Traverse the edges of the current node
  for (auto &node : nodes.at(prevNode).edges) {
    if (updateStartTimeForNode(node, prevNode) || true) {
      setEarliestStartTimes(node, visited);
    }
  }

  for (auto &node : nodes.at(prevNode).shiftingEdges) {
    if (updateStartTimeForNode(node, prevNode) || true) {
      setEarliestStartTimes(node, visited);
    }
  }

  visited.erase(prevNode);
}

bool AdjListGraph::updateStartTimeForNode(std::string node,
                                          std::string prevNode) {

  mlir::Operation *op = nodes.at(node).op;
  bool startTimeUpdated = false;

  int prevNodePathTime =
      nodes.at(prevNode).earliestStartTime + nodes.at(prevNode).latency;
  assert(nodes.at(prevNode).earliestStartTime != -1 &&
         "Previous start time must be defined");

  edgeMinLatencies[node][prevNode] =
      std::max(edgeMinLatencies[node][prevNode], prevNodePathTime);

  // Mux, merge and cmerge only need one of their inputs to be ready, all
  // other nodes need all of their inputs to be ready
  if (op && (op->getName().getStringRef() == "handshake.mux" ||
             op->getName().getStringRef() == "handshake.merge" ||
             op->getName().getStringRef() == "handshake.cmerge")) {

    int minLatency = INT_MAX;
    for (auto &incomgingEdge : edgeMinLatencies[node]) {
      if (minLatency > incomgingEdge.second) {
        minLatency = incomgingEdge.second;
      }
    }
    if (minLatency != nodes.at(node).earliestStartTime) {
      startTimeUpdated = true;
      nodes.at(node).earliestStartTime = minLatency;
      llvm::dbgs() << "Setting " << node << " to " << minLatency
                   << "(prev: " << prevNode << ")\n";
    }
  } else {
    int maxLatency = 0;
    for (auto &incomgingEdge : edgeMinLatencies[node]) {
      if (maxLatency < incomgingEdge.second) {
        maxLatency = incomgingEdge.second;
      }
    }
    if (maxLatency != nodes.at(node).earliestStartTime) {
      startTimeUpdated = true;
      nodes.at(node).earliestStartTime = maxLatency;
      llvm::dbgs() << "Setting " << node << " to " << maxLatency
                   << "(prev: " << prevNode << ")\n";
    }
  }
  return startTimeUpdated;
}

int AdjListGraph::getEarliestStartTime(mlir::Operation *op) {
  std::string opName = getUniqueName(op).str();
  // If earliestStartTime of node has not been initialized (is -1), it is not
  // reachable by the start node. This should only happen for nodes that are
  // e.g. a source. The not reachable muxes, merges and cmerges are handled by
  // adding shifting edges for their start times and should be accounted for
  if (nodes.at(opName).earliestStartTime == -1) {
    return 0;
  } else {
    return nodes.at(opName).earliestStartTime;
  }
}