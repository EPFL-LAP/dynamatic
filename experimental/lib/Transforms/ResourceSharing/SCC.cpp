//===- SCC.cpp - Strongly Connected Components ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//     This file contains an algorithm that takes a graph with n nodes
//            and returns the Strongly connected components in this graph
//     The algorithm works well with low edge to node ratio
//            If this is not the case one might consider usng an other algorithm
//     Implementation of Kosaraju's algorithm
//     Explanatory video: https://www.youtube.com/watch?v=Qdh6-a_2MxE&t=328s
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/ResourceSharing/SCC.h"

namespace dynamatic {
namespace experimental {
namespace sharing {

/*
 * Dumps content of vector of lists to the console
 */
void printList(std::vector<std::list<int>> &adjacencyList) {
  int nodes = adjacencyList.size();
  for (int i = 0; i < nodes; i++) {
    llvm::errs() << i << ": ";
    for (auto item : adjacencyList[i]) {
      llvm::errs() << item << ", ";
    }
    llvm::errs() << "\n";
  }
}

/*
 * Dumps content of stack to the console
 */
void printStack(std::stack<int> dfsStack) {
  llvm::errs() << "Printing stack: ";
  while (!dfsStack.empty()) {
    llvm::errs() << dfsStack.top() << " ";
    dfsStack.pop();
  }
  llvm::errs() << "\n";
}

/*
 * Gets the number of Basic Blocks in the IR
 */
unsigned int getNumberOfBBs(const SmallVector<ArchBB>& archs) {
  unsigned int maximum = 0;
  for (auto archItem : archs) {
    maximum = std::max(maximum, std::max(archItem.srcBB, archItem.dstBB));
  }
  // as we have BB0, we need to add one at the end
  return maximum + 1;
}

/*
 * Input: Container with source/destination Basic Blocks of each Edge between
 Basic Blocks
 * Output: vector of lists, vector[n] stores all destination Basic Blocks, where
 the
 *         source Basic Block is the n.th Basic Block
 * Example: (In: (0,1),(1,2),(2,2),(2,3) )
             Out:  0: 1
                   1: 2
                   2: 2,3
                   3: -
 */
std::vector<std::list<int>> createAdjacencyListBbl(const SmallVector<ArchBB>& archs,
                                                      int nodes) {
  std::vector<std::list<int>> result(nodes);
  for (auto archItem : archs) {
    result[archItem.srcBB].push_front(archItem.dstBB);
  }
  return result;
}

/*
 * This function simply converts a vector to a list
 * Use: This function is a version of create_adjacency_list_bbl to test
 *      the algorithm (see top of this file) using "geeks for geeks"
 */
std::vector<std::list<int>>
createAdjacencyListGfg(std::vector<std::vector<int>> &adj, int v) {
  std::vector<std::list<int>> result(v);
  for (unsigned long i = 0; i < adj.size(); i++) {
    for (auto item : adj[i]) {
      result[i].push_front(item);
    }
  }
  return result;
}

/*
 * Type: Recursive depth first search travel
 * Use:  Creates a stack with the last finishing nodes at the top
 * Remarks: first node chosen by function firstDFStravel
 * Example:    1 - 2 - 4   We are starting at Node 1, we get to Node 2
 *                 |       then 4, where we can no longer travel, we push
 *                 3       Node 4 on the stack, return to 2, travel to 3,
 *                         where we are again stuck, we push 3.
 *                         We return to 2 and as we already visited 1,3,4
 *                         we are stuck again and push 2, we finally return
 *                         to Node 1 and push 1 to the stack
 *                         Output: 4,3,2,1
 */
void firstRecursiveDFStravel(std::stack<int> &dfsStack,
                             std::vector<bool> &nodeVisited,
                             std::vector<std::list<int>> &adjacencyList,
                             int nodes, int currentNode) {
  std::list<int> currentList = adjacencyList[currentNode];
  for (auto item : currentList) {
    if (!nodeVisited[item]) {
      nodeVisited[item] = true;
      firstRecursiveDFStravel(dfsStack, nodeVisited, adjacencyList, nodes,
                              item);
    }
  }
  dfsStack.push(currentNode);
}

/*
 * First we choose the starting node for our algorithm. As we are working with
 * an IR, all nodes are reachable from node 0, aka start. Then we are just
 * calling function firstRecursiveDFStravel
 */
void firstDFStravel(std::stack<int> &dfsStack,
                    std::vector<std::list<int>> &adjacencyList, int nodes) {
  std::vector<bool> nodeVisited(nodes, false);
  // Every BB can inevitably be reached from BB0
  int currentNode = 0;
  // As we start with node 0, we mark it as visited
  nodeVisited[0] = true;
  firstRecursiveDFStravel(dfsStack, nodeVisited, adjacencyList, nodes,
                          currentNode);
  /*
  //This code part is only used for algorithm verification using gfg
  //The assumption here is, that not all nodes can be reached through node 0
  bool continue_it = true;
  while(continue_it) {
      continue_it = false;
      for(int i = 0; i < Nodes; i++) {
          if(!node_visited[i]) {
              node_visited[i] = true;
              firstRecursiveDFStravel(DFSstack, node_visited, adjacency_list,
  Nodes, i); continue_it = true; break;
          }
      }
  }
  */
}

/*
 * This function takes a directed graph and inverts the edges
 * Example: The edge 1->2 gets replaced with 2->1
 */
std::vector<std::list<int>>
converseGraph(std::vector<std::list<int>> &adjacencyList, int nodes) {
  std::vector<std::list<int>> result(nodes);
  for (int i = 0; i < nodes; i++) {
    for (auto item : adjacencyList[i]) {
      result[item].push_front(i);
    }
  }
  return result;
}

/*
 * This function does the same as function firstRecursiveDFStravel with the key
 * difference that the wanted result here is a list instead of a stack
 */
void secondRecursiveDFStravel(std::vector<std::list<int>> &transposeGraph,
                              int nodes, std::vector<bool> &nodeVisited,
                              int currentNode, std::list<int> &currSCC) {
  std::list<int> currentList = transposeGraph[currentNode];
  for (auto item : currentList) {
    if (!nodeVisited[item]) {
      nodeVisited[item] = true;
      secondRecursiveDFStravel(transposeGraph, nodes, nodeVisited, item,
                               currSCC);
    }
  }
  currSCC.push_front(currentNode);
}

/*
 * using the stack built in function firstDFStravel, we do a second DFS travel
 * and store the first seen Nodes in a list which we finally push onto a other
 * container. This is done till all nodes are marked visited.
 */
std::vector<std::list<int>>
secondDFStravel(std::vector<std::list<int>> transposeGraph,
                std::stack<int> dfsStack, int nodes) {
  std::vector<std::list<int>> result;
  std::vector<bool> nodeVisited(nodes, false);
  while (!dfsStack.empty()) {
    int currentNode = dfsStack.top();
    dfsStack.pop();
    if (nodeVisited[currentNode]) {
      continue;
    }
    nodeVisited[currentNode] = true;
    std::list<int> currSCC;
    secondRecursiveDFStravel(transposeGraph, nodes, nodeVisited, currentNode,
                             currSCC);
    result.push_back(currSCC);
  }
  return result;
}

/*
 * Input: Container with source/destination Basic Blocks of each Edge between
 * Basic Blocks Output: vector of the size of nodes, vector[n] returns the SCC
 * id the n.th node belongs to.
 */
std::vector<int> kosarajusAlgorithmBbl(const SmallVector<ArchBB>& archs) {
  int nodes = getNumberOfBBs(archs);
  std::vector<int> result(nodes);
  std::vector<std::list<int>> adjacencyList =
      createAdjacencyListBbl(archs, nodes);
  std::stack<int> dfsStack;
  firstDFStravel(dfsStack, adjacencyList, nodes);
  std::vector<std::list<int>> transposeGraph =
      converseGraph(adjacencyList, nodes);
  std::vector<std::list<int>> scc =
      secondDFStravel(transposeGraph, dfsStack, nodes);
  int position = 0;
  for (auto component : scc) {
    bool isLoopNest = true;
    if (component.size() == 1) {
      // may be not a CFG loop nest
      isLoopNest = false;
      unsigned long bb = *component.begin();
      for (unsigned long item : adjacencyList[bb]) {
        if (item == bb) {
          // CFG loop nest
          isLoopNest = true;
        }
      }
    }
    for (auto number : component) {
      if (isLoopNest) {
        result[number] = position;
      } else {
        result[number] = -1;
      }
    }
    ++position;
  }
  return result;
}

void recursiveListCreatorOpl(
    mlir::Operation *currOp,
    std::map<Operation *, std::list<Operation *>> &adjacencyList,
    std::set<mlir::Operation *> &nodeVisited,
    std::stack<Operation *> &dfsStack) {
  nodeVisited.insert(currOp);
  for (auto &u : currOp->getResults().getUses()) {
    // get child operation
    Operation *childOp = u.getOwner();

    adjacencyList[childOp].push_back(currOp);

    // traverse child operation if not yet done
    auto it = nodeVisited.find(childOp);
    if (it == nodeVisited.end()) {
      // not visited yet
      recursiveListCreatorOpl(childOp, adjacencyList, nodeVisited,
                                 dfsStack);
    }
  }
  dfsStack.push(currOp);
}

void createAdjacencyListOpl(
    mlir::Operation *startOp,
    std::map<Operation *, std::list<Operation *>> &adjacencyList,
    std::stack<Operation *> &dfsStack, handshake::FuncOp *funcOp) {
  std::set<mlir::Operation *> nodeVisited;
  recursiveListCreatorOpl(startOp, adjacencyList, nodeVisited, dfsStack);
  for (Operation &op : funcOp->getOps()) {
    auto it = nodeVisited.find(&op);
    if (it == nodeVisited.end()) {
      recursiveListCreatorOpl(&op, adjacencyList, nodeVisited, dfsStack);
    }
  }
}

void secondRecursiveDFStravel(
    std::map<Operation *, std::list<Operation *>> &adjacencyList,
    std::set<mlir::Operation *> &nodeVisited, Operation *currentNode,
    std::list<Operation *> &currSCC) {
  std::list<Operation *> currentList = adjacencyList[currentNode];
  for (auto *item : currentList) {
    auto it = nodeVisited.find(item);
    if (it == nodeVisited.end()) {
      nodeVisited.insert(item);
      secondRecursiveDFStravel(adjacencyList, nodeVisited, item, currSCC);
    }
  }
  currSCC.push_front(currentNode);
}

std::vector<std::list<Operation *>>
secondDFStravel(std::map<Operation *, std::list<Operation *>> &adjacencyList,
                std::stack<Operation *> &dfsStack) {
  std::vector<std::list<Operation *>> result;
  std::set<mlir::Operation *> nodeVisited;
  while (!dfsStack.empty()) {
    auto *currentNode = dfsStack.top();
    dfsStack.pop();
    auto it = nodeVisited.find(currentNode);
    if (it != nodeVisited.end()) {
      continue;
    }
    nodeVisited.insert(currentNode);
    std::list<Operation *> currSCC;
    secondRecursiveDFStravel(adjacencyList, nodeVisited, currentNode,
                             currSCC);
    result.push_back(currSCC);
  }
  return result;
}

void getNoncyclicOperations(
    std::map<Operation *, std::list<Operation *>> &adjacencyList,
    std::vector<std::list<Operation *>> &scc,
    std::set<mlir::Operation *> &result) {
  for (auto list : scc) {
    if (list.size() == 1) {
      // may be noncyclic
      bool isNotOnLoop = true;
      auto *op = *list.begin();
      for (auto &u : op->getResults().getUses()) {
        // get child operation
        if (u.getOwner() == op) {
          isNotOnLoop = false;
          break;
        }
      }
      if (isNotOnLoop) {
        result.insert(op);
      }
    }
  }
}

void kosarajusAlgorithmOpl(mlir::Operation *startOp,
                             handshake::FuncOp *funcOp,
                             std::set<mlir::Operation *> &result) {
  std::map<Operation *, std::list<Operation *>> adjacencyList;
  std::stack<Operation *> dfsStack;
  createAdjacencyListOpl(startOp, adjacencyList, dfsStack, funcOp);
  std::vector<std::list<Operation *>> scc =
      secondDFStravel(adjacencyList, dfsStack);
  getNoncyclicOperations(adjacencyList, scc, result);
}

} // namespace sharing
} // namespace experimental
} // namespace dynamatic
