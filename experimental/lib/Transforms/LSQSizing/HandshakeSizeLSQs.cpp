//===- HandshakeSizeLSQs.cpp - LSQ Sizing --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-size-lsqs pass
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/LSQSizing/HandshakeSizeLSQs.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "llvm/Support/Debug.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Value.h"
#include "experimental/Transforms/LSQSizing/LSQSizingSupport.h"
#include "dynamatic/Support/CFG.h"

#define DEBUG_TYPE "handshake-size-lsqs"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::lsqsizing;

using LSQSizingResult = std::unordered_map<mlir::Operation*, std::tuple<unsigned, unsigned>>; //TUPLE: <load_size, store_size>

namespace {

struct HandshakeSizeLSQsPass
    : public dynamatic::experimental::lsqsizing::impl::HandshakeSizeLSQsBase<
          HandshakeSizeLSQsPass> {

  HandshakeSizeLSQsPass(StringRef timingModels) {
    this->timingModels = timingModels.str();
  }

  void runDynamaticPass() override;

private:

  // There is a offset between the arrival time of the arguments and the actual allocation/deallocation time inside the LSQ
  // For allocation it is the same for both load and stores, for deallocation it is different between loads and stores
  static const int allocEntryLatency = 1;
  static const int storeDeallocEntryLatency = 2;
  static const int loadDeallocEntryLatency = 1;

  // Determines the LSQ sizes, given a CFDFC and its II
  std::optional<LSQSizingResult> sizeLSQsForGraph(AdjListGraph graph, unsigned II);

  // Finds the Start Node in a CFDFC
  // The start node, is the node with the longest non-cyclic path to any other node
  mlir::Operation *findStartNode(AdjListGraph graph);

  // Finds the Phi Node for each Basic Block in a CFDFC
  // Checks all operations which get their input from a different BB and chooses the one with the lowest latency from the start node
  std::unordered_map<unsigned, mlir::Operation *> getPhiNodes(AdjListGraph graph, mlir::Operation *startNode);

  // Inserts edges to make sure that sizing is done correctly
  // These edges make sure that in the timing analysis, the allocation precedes the memory access
  // Edges are inserted from the Phi Node of a BB to the LSQ operations in the BB
  // Therefore the LSQ operations will be allocated at the start of the BB
  void insertAllocPrecedesMemoryAccessEdges(AdjListGraph &graph, std::vector<mlir::Operation *> ops, std::unordered_map<unsigned, mlir::Operation *> phiNodes);

  // Finds the allocation time of each operation, which is the latency to the Phi Node of the BB plus a fixed additional latency
  std::unordered_map<mlir::Operation *, int> getAllocTimes(AdjListGraph graph, mlir::Operation *startNode, std::vector<mlir::Operation *> ops,
                                                           std::unordered_map<unsigned, mlir::Operation *> phiNodes);

  // Finds the deallocation time of each store operation, which is the the latency of the last argument arriving plus a fixed additional latency
  std::unordered_map<mlir::Operation *, int> getStoreDeallocTimes(AdjListGraph graph, mlir::Operation *startNode, std::vector<mlir::Operation *> storeOps);

  // Finds the deallocation time of each load operation, which is the latest argument arriving at the operation succeding the load operation plus a fixed additional latency
  // This is due to to the fact that the load operation frees the queue entry as soon as the load result is passed on to the succeeding operation                                                     
  std::unordered_map<mlir::Operation *, int> getLoadDeallocTimes(AdjListGraph graph, mlir::Operation *startNode, std::vector<mlir::Operation *> loadOps);

  // Given the alloc and dealloc times of each operation, calculates the maximum queue size needed for each LSQ
  std::unordered_map<mlir::Operation*, unsigned> calcQueueSize(std::unordered_map<mlir::Operation *, int> allocTimes, std::unordered_map<mlir::Operation *, int> deallocTimes, unsigned II);
};
} // namespace


void HandshakeSizeLSQsPass::runDynamaticPass() {
  llvm::dbgs() << "\t [DBG] LSQ Sizing Pass Called!\n";
  llvm::SmallVector<LSQSizingResult> sizingResults;

  // Read component latencies
  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    signalPassFailure();


  mlir::ModuleOp mod = getOperation();
  for (handshake::FuncOp funcOp : mod.getOps<handshake::FuncOp>()) {
    llvm::dbgs() << "\t [DBG] Function: " << funcOp.getName() << "\n";

    std::unordered_map<unsigned,llvm::SetVector<unsigned>> cfdfcBBLists;
    std::unordered_map<unsigned, float> IIs;

    // Extract CFDFCs and II from the Attributes
    handshake::CFDFCThroughputAttr throughputAttr = getDialectAttr<handshake::CFDFCThroughputAttr>(funcOp);
    handshake::CFDFCToBBListAttr cfdfcAttr = getDialectAttr<handshake::CFDFCToBBListAttr>(funcOp);

    if(throughputAttr == nullptr || cfdfcAttr == nullptr) {
      llvm::dbgs() << "\t [DBG] No CFDFCThroughputAttr or CFDFCToBBListAttr found\n";
      continue;
    }

    DictionaryAttr throughputDict = throughputAttr.getThroughputMap();
    DictionaryAttr cfdfcDict = cfdfcAttr.getCfdfcMap();

    // Convert Attribute into better usable data structure
    for(auto &attr: cfdfcDict) {
      ArrayAttr bbList = llvm::dyn_cast<ArrayAttr>(attr.getValue());
      llvm::SetVector<unsigned> cfdfcBBs; 
      for(auto &bb: bbList) {
        cfdfcBBs.insert(bb.cast<IntegerAttr>().getUInt());
      }
      cfdfcBBLists.insert({std::stoi(attr.getName().str()),cfdfcBBs});
    }


    // Extract II from Attribute
    for (const NamedAttribute attr : throughputDict) {
      FloatAttr throughput = llvm::dyn_cast<FloatAttr>(attr.getValue());
      IIs.insert({std::stoi(attr.getName().str()), round(1 / throughput.getValueAsDouble())});
    }

    // Size LSQs for each CFDFC
    for(auto &entry: cfdfcBBLists) {
      llvm::dbgs() << "\n\n ==========================\n";
      std::optional<LSQSizingResult> result = sizeLSQsForGraph(AdjListGraph(funcOp, entry.second, timingDB, IIs[entry.first]), IIs[entry.first]);
      if(result) {
        sizingResults.push_back(result.value());
      }
    }
    
    // Extract maximum Queue sizes for each LSQ
    std::map<mlir::Operation*, std::tuple<unsigned, unsigned>> maxLoadStoreSizes;
    for(auto &result: sizingResults) {
      for(auto &entry: result) {
        int newMaxLoadSize = std::max(std::get<0>(maxLoadStoreSizes[entry.first]), std::get<0>(entry.second));
        int newMaxStoreSize = std::max(std::get<1>(maxLoadStoreSizes[entry.first]), std::get<1>(entry.second));
        maxLoadStoreSizes[entry.first] = std::make_tuple(newMaxLoadSize, newMaxStoreSize);
      }
    }

    // Set the maximum Queue sizes as attributes for backend
    for(auto &maxLoadStoreSize: maxLoadStoreSizes) {
      mlir::Operation *lsqOp = maxLoadStoreSize.first;
      unsigned maxLoadSize = std::get<0>(maxLoadStoreSize.second);
      unsigned maxStoreSize = std::get<1>(maxLoadStoreSize.second);

      // A LSQ with size 1 or 0 does not make sense and cant be generated by the LSQ Generator, a minimum size of 2 is needed
      if(maxLoadSize < 2) {
        llvm::dbgs() << " [DBG] LSQ " << getUniqueName(lsqOp).str() << " Load Size: " << maxLoadSize << " is too small, setting to 2\n";
      }
      if(maxStoreSize < 2) {
        llvm::dbgs() << " [DBG] LSQ " << getUniqueName(lsqOp).str() << " Store Size: " << maxStoreSize << " is too small, setting to 2\n";
      }

      maxLoadSize = std::max(maxLoadSize, (unsigned)2);
      maxStoreSize = std::max(maxStoreSize, (unsigned)2);

      llvm::dbgs() << " [DBG] final LSQ " << getUniqueName(lsqOp).str() << " Max Load Size: " << maxLoadSize << " Max Store Size: " << maxStoreSize << "\n";
      handshake::LSQSizeAttr lsqSizeAttr = handshake::LSQSizeAttr::get(mod.getContext(), maxLoadSize, maxStoreSize);
      setDialectAttr(lsqOp, lsqSizeAttr);
    }
  }
}


std::optional<LSQSizingResult> HandshakeSizeLSQsPass::sizeLSQsForGraph(AdjListGraph graph, unsigned II) {

  std::vector<mlir::Operation *> loadOps = graph.getOperationsWithOpName("handshake.lsq_load");
  std::vector<mlir::Operation *> storeOps = graph.getOperationsWithOpName("handshake.lsq_store");

  //graph.printGraph();


  if(loadOps.size() == 0 && storeOps.size() == 0) {
    llvm::dbgs() << "\t [DBG] No LSQ Ops found in CFDFC\n";
    return std::nullopt;
  }

  // Find starting node, which will be the reference to the rest
  mlir::Operation * startNode = findStartNode(graph);
  llvm::dbgs() << "\t [DBG] Start Node: " << getUniqueName(startNode).str() << "\n";
  
  // Find Phi node of each BB
  std::unordered_map<unsigned, mlir::Operation *> phiNodes = getPhiNodes(graph, startNode);

  // connect all phi nodes to the lsq ps in their BB
  insertAllocPrecedesMemoryAccessEdges(graph, loadOps, phiNodes);  

  llvm::dbgs() << "----------------------------\n";
  graph.printGraph();
  llvm::dbgs() << "----------------------------\n";

  // Get Alloc Time of each Op (Start time of BB) 
  std::unordered_map<mlir::Operation *, int> loadAllocTimes = getAllocTimes(graph, startNode, loadOps, phiNodes);
  std::unordered_map<mlir::Operation *, int> storeAllocTimes = getAllocTimes(graph, startNode, storeOps, phiNodes);

  // Get Dealloc Time of each Op
  std::unordered_map<mlir::Operation *, int> loadDeallocTimes = getLoadDeallocTimes(graph, startNode, loadOps);
  std::unordered_map<mlir::Operation *, int> storeDeallocTimes = getStoreDeallocTimes(graph, startNode, storeOps);

  // Get Load and Store Sizes
  std::unordered_map<mlir::Operation*, unsigned> loadSizes = calcQueueSize(loadAllocTimes, loadDeallocTimes, II);
  std::unordered_map<mlir::Operation*, unsigned> storeSizes = calcQueueSize(storeAllocTimes, storeDeallocTimes, II);

  LSQSizingResult result;
  for(auto &entry: loadSizes) {
    unsigned storeSize = storeSizes.find(entry.first) != storeSizes.end() ? storeSizes[entry.first] : 0;
    result.insert({entry.first, std::make_tuple(entry.second, storeSize)});
  }

  for(auto &entry: storeSizes) {
    if(result.find(entry.first) == result.end()) {
      result.insert({entry.first, std::make_tuple(0, entry.second)});
    }
  }

  for(auto &entry: result) {
    llvm::dbgs() << "\t [DBG] LSQ " << getUniqueName(entry.first).str() << " Load Size: " << std::get<0>(entry.second) << " Store Size: " << std::get<1>(entry.second) << "\n";
  }

  llvm::dbgs() << "==========================\n";
  return result;
}


mlir::Operation * HandshakeSizeLSQsPass::findStartNode(AdjListGraph graph) {

  // Find all mux and control_merge ops as candidates for start node
  std::vector<mlir::Operation *> muxOps = graph.getOperationsWithOpName("handshake.mux");
  std::vector<mlir::Operation *> cmergeOps = graph.getOperationsWithOpName("handshake.control_merge");
  std::vector<mlir::Operation *> startNodeCandidates = std::vector<mlir::Operation *>(muxOps.size() + cmergeOps.size());
  std::merge(muxOps.begin(), muxOps.end(), cmergeOps.begin(), cmergeOps.end(), startNodeCandidates.begin());


  std::unordered_map<mlir::Operation *, int> maxLatencies;
  std::unordered_map<mlir::Operation *, int> nodeCounts;

  // Go trough all candidates and save the longest path, its latency and node count
  for(auto &op: startNodeCandidates) {
    std::vector<std::string> path = graph.findLongestNonCyclicPath2(op);
    maxLatencies.insert({op, graph.getPathLatency(path)});
    nodeCounts.insert({op, path.size()});
  }

  // Find the node with the highest latency, if there are multiple, choose the one with the most nodes
  mlir::Operation *maxLatencyNode = nullptr;
  int maxLatency = 0;
  int maxNodeCount = 0;
  for(auto &node: maxLatencies) {
    if(node.second > maxLatency) {
      maxLatency = node.second;
      maxLatencyNode = node.first;
      maxNodeCount = nodeCounts[node.first];
    } else if(node.second == maxLatency && nodeCounts[node.first] > maxNodeCount) {
      maxLatencyNode = node.first;
      maxNodeCount = nodeCounts[node.first];
    }
  }

  return maxLatencyNode;
}


std::unordered_map<unsigned, mlir::Operation *> HandshakeSizeLSQsPass::getPhiNodes(AdjListGraph graph, mlir::Operation *startNode) {
  std::unordered_map<unsigned, std::vector<mlir::Operation *>> phiNodeCandidates;
  std::unordered_map<unsigned, mlir::Operation *> phiNodes;

  // Insert start_node as a candidate for cases where there is only 1 bb (will be choosen anyway for other cases, but looks cleaner then special handling)
  std::optional<unsigned> startNodeBB = getLogicBB(startNode);
  assert(startNodeBB && "Start Node must belong to basic block");
  phiNodeCandidates.insert({*startNodeBB, {startNode}});
  //llvm::dbgs() << "\t [DBG] Inserted Start Node: " << startNode->getAttrOfType<StringAttr>("handshake.name").str() << " for BB " << startNode->getAttrOfType<IntegerAttr>("handshake.bb").getUInt() << "\n";

  // Go trought all branch and fork ops and find connected ops
  for(auto &srcOp: graph.getOperations()) {
    std::optional<unsigned> srcBB = getLogicBB(srcOp);
    assert(srcBB && "Src Op must belong to basic block");    
    //llvm::dbgs() << "\t [DBG] Branch Op: " << branchOp->getAttrOfType<StringAttr>("handshake.name").str() << " of BB " << srcBB <<"\n";

    // For each connected Op, check if its in a different BB and add it to the candidates
    for(auto &destOp: graph.getConnectedOps(srcOp)) {
      std::optional<unsigned> destBB = getLogicBB(destOp);
      assert(destBB && "Dest Op must belong to basic block");  
      //llvm::dbgs() << "\t\t [DBG] connected to: " << destOp->getAttrOfType<StringAttr>("handshake.name").str() << " of BB" << destBB << "\n";
      if(*destBB != *srcBB) {
        //llvm::dbgs() << "\t [DBG] Found Phi Node Candidate: " << destOp->getAttrOfType<StringAttr>("handshake.name").str() << " for BB " << destBB << "\n";
        if(phiNodeCandidates.find(*destBB) == phiNodeCandidates.end()) {
          phiNodeCandidates.insert({*destBB, std::vector<mlir::Operation *>()});
        }
        phiNodeCandidates.at(*destBB).push_back(destOp);
      }
    }
  }

  // Go trough all candidates and choose the one with the lowest latency to the start node, ignore backedges for the path
  for(auto &entry: phiNodeCandidates) {
    mlir::Operation *phiNode = nullptr;
    int minLatency = INT_MAX;
    for(auto &op: entry.second) {
      int latency = graph.findMinPathLatency(startNode, op, true); //TODO think about if backedges should be ignored or not
      //llvm::dbgs() << "\t\t [DBG] Latency from " << startNode->getAttrOfType<StringAttr>("handshake.name").str() << " to " << op->getAttrOfType<StringAttr>("handshake.name").str() << " is " << latency << "\n";
      if (latency < minLatency)
      {
        phiNode = op;
        minLatency = latency;
      }
    }
    phiNodes.insert({entry.first, phiNode});
  }

  for(auto &nodes: phiNodes) {
    llvm::dbgs() << "\t [DBG] Phi Node for BB " << nodes.first << ": " << getUniqueName(nodes.second).str() << "\n";
  }

  return phiNodes;
}

std::unordered_map<mlir::Operation *, int> HandshakeSizeLSQsPass::getAllocTimes(AdjListGraph graph, mlir::Operation *startNode, std::vector<mlir::Operation *> ops,
                                                           std::unordered_map<unsigned, mlir::Operation *> phiNodes) {
  std::unordered_map<mlir::Operation *, int> allocTimes;
  
  // Go trough all ops and find the latency to the phi node of the ops BB
  for(auto &op: ops) {
    std::optional<unsigned> bb = getLogicBB(op);
    assert(bb && "Load/Store Op must belong to basic block");
    mlir::Operation *phiNode = phiNodes[*bb];
    assert(phiNode && "Phi node not found for BB");
    int latency = graph.findMinPathLatency(startNode, phiNode, true) + allocEntryLatency;
    allocTimes.insert({op, latency});
    llvm::dbgs() << "\t\t [DBG] " << getUniqueName(op).str() << " alloc time: " << latency << "\n";
  }
  return allocTimes;
}

std::unordered_map<mlir::Operation *, int> HandshakeSizeLSQsPass::getStoreDeallocTimes(AdjListGraph graph, mlir::Operation *startNode, std::vector<mlir::Operation *> ops) {
  std::unordered_map<mlir::Operation *, int> deallocTimes;
  
  // Go trough all ops and find the maximum latency to the op node
  for(auto &op: ops) {
    int latency = graph.findMaxPathLatency(startNode, op) + storeDeallocEntryLatency;
    deallocTimes.insert({op, latency});
    llvm::dbgs() << "\t\t [DBG] " << getUniqueName(op).str() << " dealloc time: " << latency << "\n";
  }
  return deallocTimes;
}

std::unordered_map<mlir::Operation *, int> HandshakeSizeLSQsPass::getLoadDeallocTimes(AdjListGraph graph, mlir::Operation *startNode, std::vector<mlir::Operation *> ops) {
  std::unordered_map<mlir::Operation *, int> deallocTimes;
  
  // Go trough all ops which directly succeed the load op and therefore need its result
  // Check every succeeding nodes for the maximum latency (latest argument arrival), excluding the latency of the operation itself
  for(auto &op: ops) {
    int maxLatency = 0;
    for(auto &succedingOp: graph.getConnectedOps(op)) {
      maxLatency = std::max(graph.findMaxPathLatency(startNode, succedingOp, false, true) + loadDeallocEntryLatency, maxLatency);
    }
    deallocTimes.insert({op, maxLatency});
    llvm::dbgs() << "\t\t [DBG] " << getUniqueName(op).str() << " dealloc time: " << maxLatency << "\n";
  }
  return deallocTimes;
}


std::unordered_map<mlir::Operation*, unsigned> HandshakeSizeLSQsPass::calcQueueSize(std::unordered_map<mlir::Operation *, int> allocTimes, std::unordered_map<mlir::Operation *, int> deallocTimes, unsigned II) {
  std::unordered_map<mlir::Operation*, unsigned> queueSizes;

  int endTime = 0;
  
  // Choose the maxiumm time of all dealloc times
  for(auto &entry : deallocTimes) {
    if(entry.second > endTime) {
      endTime = entry.second;
    }
  }

  // Go trough all alloc ops find the corresponding LSQ and save the alloc times
  std::unordered_map<mlir::Operation*, std::tuple<std::vector<int>, std::vector<int>>> allocDeallocTimesPerLSQ;
  for(auto &allocTime: allocTimes) {
    mlir::Operation *lsqOp = nullptr;
    for(Operation *destOp: allocTime.first->getUsers()) {
      if(destOp->getName().getStringRef().str() == "handshake.lsq") {
        lsqOp = destOp;
        break;
      }
    }
    if(allocDeallocTimesPerLSQ.find(lsqOp) == allocDeallocTimesPerLSQ.end()) {
      allocDeallocTimesPerLSQ.insert({lsqOp, std::make_tuple(std::vector<int>(), std::vector<int>())});
    }
    std::get<0>(allocDeallocTimesPerLSQ[lsqOp]).push_back(allocTime.second);
  }

  // Go trough all dealloc times, find the corresponding LSQ and add all dealloc times
  for(auto &deallocTime: deallocTimes) {
    mlir::Operation *lsqOp = nullptr;
    for(Operation *destOp: deallocTime.first->getUsers()) {
      if(destOp->getName().getStringRef().str() == "handshake.lsq") {
        lsqOp = destOp;
        break;
      }
    }
    if(allocDeallocTimesPerLSQ.find(lsqOp) == allocDeallocTimesPerLSQ.end()) {
      allocDeallocTimesPerLSQ.insert({lsqOp, std::make_tuple(std::vector<int>(), std::vector<int>())});
    }
    std::get<1>(allocDeallocTimesPerLSQ[lsqOp]).push_back(deallocTime.second);
  }

  // Go trough all LSQs and calculate the maximum amount of slots needed
  for(auto &entry: allocDeallocTimesPerLSQ) {
    unsigned iterMax = std::ceil((float)endTime / II);
    std::vector<int> allocPerCycle(endTime);


    // Build array for how many slots are allocated and deallocated per cycle
    for(unsigned iter = 0; iter < iterMax; iter++) {
      
      for(auto &allocTime: std::get<0>(entry.second)) {
        int t = allocTime + II * iter;
        if(t >= 0 && t < endTime) {
          allocPerCycle[t]++;
        }
      }
      for(auto &deallocTime: std::get<1>(entry.second)) {
        int t = deallocTime + II * iter;
        if(t >= 0 && t < endTime) {
          allocPerCycle[t]--;
        }
      }
    }
    // build array for many slots are actively allocated at which cycle
    std::vector<int> slotsPerCycle(endTime);
    slotsPerCycle[0] = allocPerCycle[0];
    for(int i=1; i < endTime; i++) {
      slotsPerCycle[i] = slotsPerCycle[i - 1] + allocPerCycle[i];
    }

    llvm::dbgs() << " \t [DBG] slots[" << (endTime) << "]:";
    for(auto &slot: slotsPerCycle) {
      llvm::dbgs() << slot << " ";
    }
    llvm::dbgs() << "\n";


    // get highest amount of slots from the array
    unsigned maxSlots = *std::max_element(slotsPerCycle.begin(), slotsPerCycle.end());
    queueSizes.insert({entry.first, maxSlots});
  }

  return queueSizes;
}

void HandshakeSizeLSQsPass::insertAllocPrecedesMemoryAccessEdges(AdjListGraph &graph, std::vector<mlir::Operation *> ops, std::unordered_map<unsigned, mlir::Operation *> phiNodes) {
  //Iterate over all provided ops and add an edge from the phi node of the ops BB to the op
  for(auto &op: ops) {
    std::optional<unsigned> bb = getLogicBB(op);
    assert(bb && "Load/Store Op must belong to basic block");
    mlir::Operation *phiNode = phiNodes[*bb];
    graph.addEdge(phiNode, op);
  }
}


std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::lsqsizing::createHandshakeSizeLSQs(StringRef timingModels) {
  return std::make_unique<HandshakeSizeLSQsPass>(timingModels);
}


