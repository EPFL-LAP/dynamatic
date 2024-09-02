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
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "llvm/Support/Debug.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Support/TimingModels.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Value.h"
#include "experimental/Transforms/LSQSizing/LSQSizingSupport.h"
#include "dynamatic/Support/CFG.h"

#define DEBUG_TYPE "handshake-size-lsqs"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::lsqsizing;

using LSQSizingResult = DenseMap<mlir::Operation*, std::tuple<unsigned, unsigned>>; //TUPLE: <load_size, store_size>

namespace {

struct HandshakeSizeLSQsPass
    : public dynamatic::experimental::lsqsizing::impl::HandshakeSizeLSQsBase<
          HandshakeSizeLSQsPass> {

  HandshakeSizeLSQsPass(StringRef timingModels) {
    this->timingModels = timingModels.str();
  }

  void runDynamaticPass() override;

private:

  LSQSizingResult sizeLSQsForCFDFC(buffer::CFDFC cfdfc, unsigned II, TimingDatabase timingDB);
  mlir::Operation *findStartNode(AdjListGraph graph);
  std::unordered_map<unsigned, mlir::Operation *> getPhiNodes(AdjListGraph graph, mlir::Operation *startNode);
  void insertAllocPrecedesMemoryAccessEdges(AdjListGraph &graph, std::vector<mlir::Operation *> ops, std::unordered_map<unsigned, mlir::Operation *> phiNodes);
  void insertLoadStoreEdge(AdjListGraph &graph, std::vector<mlir::Operation *> loadOps, std::vector<mlir::Operation *> storeOps);
  std::unordered_map<mlir::Operation *, int> getAllocTimes(AdjListGraph graph, mlir::Operation *startNode, std::vector<mlir::Operation *> ops,
                                                           std::unordered_map<unsigned, mlir::Operation *> phiNodes);
  std::unordered_map<mlir::Operation *, int> getDeallocTimes(AdjListGraph graph, mlir::Operation *startNode, std::vector<mlir::Operation *> ops);
  std::unordered_map<mlir::Operation*, unsigned> calcQueueSize(std::unordered_map<mlir::Operation *, int> allocTimes, std::unordered_map<mlir::Operation *, int> deallocTimes, unsigned II);
};
} // namespace


void HandshakeSizeLSQsPass::runDynamaticPass() {
  llvm::dbgs() << "\t [DBG] LSQ Sizing Pass Called!\n";

  std::map<unsigned,buffer::CFDFC> cfdfcs;
  llvm::SmallVector<LSQSizingResult> sizingResults;

  // Read component latencies
  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    signalPassFailure();


  mlir::ModuleOp mod = getOperation();
  for (handshake::FuncOp funcOp : mod.getOps<handshake::FuncOp>()) {
    llvm::dbgs() << "\t [DBG] Function: " << funcOp.getName() << "\n";

    std::unordered_map<unsigned, float> IIs;
    DictionaryAttr troughputAttr = getUniqueAttr<handshake::CFDFCThroughputAttr>(funcOp).getThroughputMap();
    DictionaryAttr cfdfcAttr = getUniqueAttr<handshake::CFDFCToBBListAttr>(funcOp).getCfdfcMap();

    //DictionaryAttr troughputAttr = dyn_cast<DictionaryAttr>(funcOp->getAttr("handshake.throughput"));
    //DictionaryAttr cfdfcAttr = dyn_cast<DictionaryAttr>(funcOp->getAttr("handshake.cfdfc"));

    // TODO this part will be rewritten to not use the CFDFC constructor, ArchSet and ArchBB class from buffer placement (line 96 - 114)
    // Some of the features are not needed and it would look more clean to make my own constructor instead to directly build
    // the data strucuture i work with instead of converting it from the buffer placement data structure
      
    // Extract Arch sets
    for(auto &entry: cfdfcAttr) {
      SmallVector<experimental::ArchBB> archStore;

      ArrayAttr bbList = llvm::dyn_cast<ArrayAttr>(entry.getValue());
      auto it = bbList.begin();
      int firstBBId = it->cast<IntegerAttr>().getUInt();
      int currBBId, prevBBId = firstBBId;      
      for(std::advance(it, 1); it != bbList.end(); it++) {
        currBBId = it->cast<IntegerAttr>().getUInt();
        archStore.push_back(experimental::ArchBB(prevBBId, currBBId, 0, false));
        prevBBId = currBBId;
      }
      archStore.push_back(experimental::ArchBB(prevBBId, firstBBId, 0, false));

      llvm::dbgs() << "\t [DBG] CFDFC: " << entry.getName() << " with " << archStore.size() << " arches\n";
      buffer::ArchSet archSet;
      for(auto &arch: archStore) {
        llvm::dbgs() << "\t [DBG] Arch: " << arch.srcBB << " -> " << arch.dstBB << "\n";
        archSet.insert(&arch);
      }

      cfdfcs.insert_or_assign(std::stoi(entry.getName().str()), buffer::CFDFC(funcOp, archSet, 0));
    }


    // Extract II
    for (const NamedAttribute attr : troughputAttr) {
      FloatAttr throughput = llvm::dyn_cast<FloatAttr>(attr.getValue());
      IIs.insert({std::stoi(attr.getName().str()), round(1 / throughput.getValueAsDouble())});
    }

    // Size LSQs for each CFDFC
    for(auto &cfdfc : cfdfcs) {
      sizingResults.push_back(sizeLSQsForCFDFC(cfdfc.second, IIs[cfdfc.first], timingDB));
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
      llvm::dbgs() << " [DBG] final LSQ " << lsqOp->getAttrOfType<StringAttr>("handshake.name").str() << " Max Load Size: " << maxLoadSize << " Max Store Size: " << maxStoreSize << "\n";

      handshake::LSQSizeAttr lsqSizeAttr = handshake::LSQSizeAttr::get(mod.getContext(), maxLoadSize, maxStoreSize);
      setUniqueAttr(lsqOp, lsqSizeAttr);
    }
  }
}

LSQSizingResult HandshakeSizeLSQsPass::sizeLSQsForCFDFC(buffer::CFDFC cfdfc, unsigned II, TimingDatabase timingDB) {
  llvm::dbgs() << " [DBG] sizeLSQsForCFDFC called for CFDFC with " << cfdfc.cycle.size() << " BBs and II=" << II << "\n";

  AdjListGraph graph(cfdfc, timingDB, II);
  std::vector<mlir::Operation *> loadOps = graph.getOperationsWithOpName("handshake.lsq_load");
  std::vector<mlir::Operation *> storeOps = graph.getOperationsWithOpName("handshake.lsq_store");

  //graph.printGraph();

  //TODO return empty or make result std::optional?
  /*if(loadOps.size() == 0 && storeOps.size() == 0) {
    llvm::dbgs() << "\t [DBG] No LSQ Ops found in CFDFC\n";
    return DenseMap<unsigned, std::tuple<unsigned, unsigned>>();
  }*/

  // Find starting node, which will be the reference to the rest
  mlir::Operation * startNode = findStartNode(graph);
  llvm::dbgs() << "\t [DBG] Start Node: " << startNode->getAttrOfType<StringAttr>("handshake.name").str()<< "\n";
  
  // Find Phi node of each BB
  std::unordered_map<unsigned, mlir::Operation *> phiNodes = getPhiNodes(graph, startNode);

  // connect all phi nodes to the lsq ps in their BB
  insertAllocPrecedesMemoryAccessEdges(graph, loadOps, phiNodes);  
  //insertLoadStoreEdge(graph, loadOps, storeOps);

  llvm::dbgs() << "============================\n";
  graph.printGraph();
  llvm::dbgs() << "============================\n";

  // Get Alloc Time of each Op (Start time of BB) 
  std::unordered_map<mlir::Operation *, int> loadAllocTimes = getAllocTimes(graph, startNode, loadOps, phiNodes);
  std::unordered_map<mlir::Operation *, int> storeAllocTimes = getAllocTimes(graph, startNode, storeOps, phiNodes);

  // Get Dealloc Time of each Op
  std::unordered_map<mlir::Operation *, int> loadDeallocTimes = getDeallocTimes(graph, startNode, loadOps);
  std::unordered_map<mlir::Operation *, int> storeDeallocTimes = getDeallocTimes(graph, startNode, storeOps);

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
    llvm::dbgs() << "\t [DBG] LSQ " << entry.first->getAttrOfType<StringAttr>("handshake.name").str() << " Load Size: " << std::get<0>(entry.second) << " Store Size: " << std::get<1>(entry.second) << "\n";
  }

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
    std::vector<std::string> path = graph.findLongestNonCyclicPath(op);
    maxLatencies.insert({op, graph.getPathLatency(path)});
    nodeCounts.insert({op, path.size()});

    /*llvm::dbgs() << "\t [DBG] Longest path from " << op->getAttrOfType<StringAttr>("handshake.name").str() << " lat: " << graph.getPathLatency(path) << " : ";
    for(auto &node: path) {
      llvm::dbgs() << node << " ";
    }
    llvm::dbgs() << "\n";*/
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

  // Find all branch and fork ops as candidates for phi nodes
  std::vector<mlir::Operation *> branchOps = graph.getOperationsWithOpName("handshake.cond_br");
  std::vector<mlir::Operation *> forkOps = graph.getOperationsWithOpName("handshake.fork");

  std::vector<mlir::Operation *> srcOps = std::vector<mlir::Operation *>(branchOps.size() + forkOps.size());
  std::merge(branchOps.begin(), branchOps.end(), forkOps.begin(), forkOps.end(), srcOps.begin());

  // Insert start_node as a candidate for cases where there is only 1 bb (will be choosen anyway for other cases, but looks cleaner then special handling)
  phiNodeCandidates.insert({startNode->getAttrOfType<IntegerAttr>("handshake.bb").getUInt(), {startNode}});
  //llvm::dbgs() << "\t [DBG] Inserted Start Node: " << startNode->getAttrOfType<StringAttr>("handshake.name").str() << " for BB " << startNode->getAttrOfType<IntegerAttr>("handshake.bb").getUInt() << "\n";

  // Go trought all branch and fork ops and find connected ops
  for(auto &srcOp: srcOps) {
    unsigned srcBB =srcOp->getAttrOfType<IntegerAttr>("handshake.bb").getUInt();
    //llvm::dbgs() << "\t [DBG] Branch Op: " << branchOp->getAttrOfType<StringAttr>("handshake.name").str() << " of BB " << srcBB <<"\n";

    // For each connected Op, check if its in a different BB and add it to the candidates
    for(auto &destOp: graph.getConnectedOps(srcOp)) {
      unsigned destBB = destOp->getAttrOfType<IntegerAttr>("handshake.bb").getUInt();
      //llvm::dbgs() << "\t\t [DBG] connected to: " << destOp->getAttrOfType<StringAttr>("handshake.name").str() << " of BB" << destBB << "\n";
      if(destBB != srcBB) {
        //llvm::dbgs() << "\t [DBG] Found Phi Node Candidate: " << destOp->getAttrOfType<StringAttr>("handshake.name").str() << " for BB " << destBB << "\n";
        if(phiNodeCandidates.find(destBB) == phiNodeCandidates.end()) {
          phiNodeCandidates.insert({destBB, std::vector<mlir::Operation *>()});
        }
        phiNodeCandidates.at(destBB).push_back(destOp);
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
    llvm::dbgs() << "\t [DBG] Phi Node for BB " << nodes.first << ": " << nodes.second->getAttrOfType<StringAttr>("handshake.name").str() << "\n";
  }

  return phiNodes;
}

std::unordered_map<mlir::Operation *, int> HandshakeSizeLSQsPass::getAllocTimes(AdjListGraph graph, mlir::Operation *startNode, std::vector<mlir::Operation *> ops,
                                                           std::unordered_map<unsigned, mlir::Operation *> phiNodes) {
  std::unordered_map<mlir::Operation *, int> allocTimes;
  
  // Go trough all ops and find the latency to the phi node of the ops BB
  for(auto &op: ops) {
    int bb = op->getAttrOfType<IntegerAttr>("handshake.bb").getUInt();
    llvm::dbgs() << "\t\t [DBG] " << op->getAttrOfType<StringAttr>("handshake.name").str() << " BB: " << bb << "\n";
    mlir::Operation *phiNode = phiNodes[op->getAttrOfType<IntegerAttr>("handshake.bb").getUInt()];
    assert(phiNode && "Phi node not found for BB");
    int latency = graph.findMinPathLatency(startNode, phiNode, true); //TODO ignore backedges?
    allocTimes.insert({op, latency});
    llvm::dbgs() << "\t\t [DBG] " << op->getAttrOfType<StringAttr>("handshake.name").str() << " alloc time: " << latency << "\n";
  }
  return allocTimes;
}

std::unordered_map<mlir::Operation *, int> HandshakeSizeLSQsPass::getDeallocTimes(AdjListGraph graph, mlir::Operation *startNode, std::vector<mlir::Operation *> ops) {
  std::unordered_map<mlir::Operation *, int> deallocTimes;
  
  // Go trough all ops and find the maximum latency to the op node
  for(auto &op: ops) {
    int latency = graph.findMaxPathLatency(startNode, op);
    deallocTimes.insert({op, latency});
    llvm::dbgs() << "\t\t [DBG] " << op->getAttrOfType<StringAttr>("handshake.name").str() << " dealloc time: " << latency << "\n";
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
    std::vector<int> allocPerCycle(endTime + 1);


    // Build array for how many slots are allocated and deallocated per cycle
    for(unsigned iter = 0; iter < iterMax; iter++) {
      
      for(auto &allocTime: std::get<0>(entry.second)) {
        int t = allocTime + II * iter;
        if(t >= 0 && t <= endTime) {
          allocPerCycle[t]++;
        }
      }
      for(auto &deallocTime: std::get<1>(entry.second)) {
        int t = deallocTime + II * iter;
        if(t >= 0 && t <= endTime) {
          allocPerCycle[t]--;
        }
      }
    }
    // build array for many slots are actively allocated at which cycle
    std::vector<int> slotsPerCycle(endTime + 1);
    slotsPerCycle[0] = allocPerCycle[0];
    for(int i=1; i <= endTime; i++) {
      slotsPerCycle[i] = slotsPerCycle[i - 1] + allocPerCycle[i];
    }

    llvm::dbgs() << " \t [DBG] slots[" << (endTime + 1) << "]:";
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
  for(auto &op: ops) {
    unsigned bb = op->getAttrOfType<IntegerAttr>("handshake.bb").getUInt();
    mlir::Operation *phiNode = phiNodes[bb];
    graph.addEdge(phiNode, op);
    llvm::dbgs() << " [DBG] Added edge from " << phiNode->getAttrOfType<StringAttr>("handshake.name").str() << " to " << op->getAttrOfType<StringAttr>("handshake.name").str() << "\n";
  }
}


std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::lsqsizing::createHandshakeSizeLSQs(StringRef timingModels) {
  return std::make_unique<HandshakeSizeLSQsPass>(timingModels);
}


void HandshakeSizeLSQsPass::insertLoadStoreEdge(AdjListGraph &graph, std::vector<mlir::Operation *> loadOps, std::vector<mlir::Operation *> storeOps) {
  for(Operation *storeOp: storeOps) {
    mlir::Operation *LsqOp = nullptr;
    for(Operation *destOp: storeOp->getUsers()) {
      if(destOp->getName().getStringRef().str() == "handshake.lsq") {
        LsqOp = destOp;
        break;
      }
    }
    for(Operation *loadOp: loadOps) {
      for(Operation *destOp: storeOp->getUsers()) {
        if(destOp == LsqOp) {
          graph.addEdge(storeOp, loadOp);
          llvm::dbgs() << " [DBG] Added edge from " << storeOp->getAttrOfType<StringAttr>("handshake.name").str() << " to " << loadOp->getAttrOfType<StringAttr>("handshake.name").str()
          << " for LSQ:" << LsqOp->getAttrOfType<StringAttr>("handshake.name") << "\n";
          break;
        }
      }
    }
  }
}

