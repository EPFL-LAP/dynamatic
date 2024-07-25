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
//#include "experimental/Support/StdProfiler.h"
#include "experimental/Transforms/LSQSizing/LSQSizingSupport.h"
#include "dynamatic/Support/CFG.h"

#define DEBUG_TYPE "handshake-size-lsqs"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::lsqsizing;

using LSQSizingResult = DenseMap<unsigned, std::tuple<unsigned, unsigned>>; //TUPLE: <load_size, store_size>

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
};
} // namespace


void HandshakeSizeLSQsPass::runDynamaticPass() {
  llvm::dbgs() << "\t [DBG] LSQ Sizing Pass Called!\n";

  std::map<unsigned,buffer::CFDFC> cfdfcs; //TODO chane to DenseMap?
  llvm::SmallVector<LSQSizingResult> sizingResults; //TODO datatype?

  // 1. Read Attributes
  // 2. Reconstruct CFDFCs
  // 3. ???
  // 4. Profit

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

    // Extract Arch sets
    for(auto &entry: cfdfcAttr) {
      SmallVector<experimental::ArchBB> arch_store;

      ArrayAttr bbList = llvm::dyn_cast<ArrayAttr>(entry.getValue());
      auto it = bbList.begin();
      int firstBBId = (*it++).cast<IntegerAttr>().getUInt();
      int currBBId, prevBBId = firstBBId;      
      for(; it != bbList.end(); it++) {
        currBBId = (*it).cast<IntegerAttr>().getUInt();
        arch_store.push_back(experimental::ArchBB(prevBBId, currBBId, 0, false));
        prevBBId = currBBId;
      }
      arch_store.push_back(experimental::ArchBB(prevBBId, firstBBId, 0, false));

      llvm::dbgs() << "\t [DBG] CFDFC: " << entry.getName() << " with " << arch_store.size() << " arches\n";
      buffer::ArchSet archSet;
      for(auto &arch: arch_store) {
        llvm::dbgs() << "\t [DBG] Arch: " << arch.srcBB << " -> " << arch.dstBB << "\n";
        archSet.insert(&arch);
      }

      cfdfcs.insert_or_assign(std::stoi(entry.getName().str()), buffer::CFDFC(funcOp, archSet, 0));
    }

    //Extract II
    for (const NamedAttribute attr : troughputAttr) {
      FloatAttr throughput = llvm::dyn_cast<FloatAttr>(attr.getValue());
      IIs.insert({std::stoi(attr.getName().str()), round(1 / throughput.getValueAsDouble())});
    }

    llvm::dbgs() << "\t [DBG] CFDFCs: " << cfdfcs.size() << "\n";
    for(auto &cfdfc : cfdfcs) {
      sizingResults.push_back(sizeLSQsForCFDFC(cfdfc.second, IIs[cfdfc.first], timingDB));
    }
    
    std::map<unsigned, unsigned> maxStoreSizes;
    std::map<unsigned, unsigned> maxLoadSizes;
    for(auto &result: sizingResults) {
      for(auto &entry: result) {
        maxStoreSizes[entry.first] = std::max(maxStoreSizes[entry.first], std::get<1>(entry.second));
        maxLoadSizes[entry.first] = std::max(maxLoadSizes[entry.first], std::get<0>(entry.second));
      }
    }

    //TODO Add Sizing to Attributes
  }
}

LSQSizingResult HandshakeSizeLSQsPass::sizeLSQsForCFDFC(buffer::CFDFC cfdfc, unsigned II, TimingDatabase timingDB) {
  //TODO implement algo
  llvm::dbgs() << "\t [DBG] sizeLSQsForCFDFC called for CFDFC with " << cfdfc.cycle.size() << " BBs and II of " << II << "\n";

  AdjListGraph graph(cfdfc, timingDB, II);
  graph.printGraph();

  // Find starting node, which will be the reference to the rest
  mlir::Operation * startNode = findStartNode(graph);
  llvm::dbgs() << "\t [DBG] Start Node: " << startNode->getAttrOfType<StringAttr>("handshake.name").str()<< "\n";
  
  // Find Phi node of each BB
  std::unordered_map<unsigned, mlir::Operation *> phiNodes = getPhiNodes(graph, startNode);

  // Get Start Times of each BB (Alloc Times) 


  // Get Dealloc Times and End Times
  std::vector<mlir::Operation *> loadOps = graph.getOperationsWithOpName("handshake.lsq_load");
  std::unordered_map<mlir::Operation *, int> loadDeallocTimes;
  int loadEndTime = 0;

  for(auto &op: loadOps) {
    int latency = graph.findMaxPathLatency(startNode, op);
    loadDeallocTimes.insert({op, latency});
    loadEndTime = std::max(loadEndTime, latency);
  }

  std::vector<mlir::Operation *> storeOps = graph.getOperationsWithOpName("handshake.lsq_store");
  std::unordered_map<mlir::Operation *, int> storeDeallocTimes;
  int storeEndTime = 0;

  for(auto &op: storeOps) {
    int latency = graph.findMaxPathLatency(startNode, op);
    storeDeallocTimes.insert({op, latency});
    storeEndTime = std::max(storeEndTime, latency);
  }

  // Get Load and Store Sizes

  return DenseMap<unsigned, std::tuple<unsigned, unsigned>>();
}


mlir::Operation * HandshakeSizeLSQsPass::findStartNode(AdjListGraph graph) {
  std::vector<mlir::Operation *> muxOps = graph.getOperationsWithOpName("handshake.mux");
  std::vector<mlir::Operation *> cmergeOps = graph.getOperationsWithOpName("handshake.control_merge");

  std::vector<mlir::Operation *> startNodeCandidates = std::vector<mlir::Operation *>(muxOps.size() + cmergeOps.size());
  std::merge(muxOps.begin(), muxOps.end(), cmergeOps.begin(), cmergeOps.end(), startNodeCandidates.begin());

  llvm::dbgs() << "\t [DBG] Start Node Candidates: ";
  for(auto &op: startNodeCandidates) {
    llvm::dbgs() << op->getAttrOfType<StringAttr>("handshake.name").str() << ", ";
  }
  llvm::dbgs() << "\n";


  std::unordered_map<mlir::Operation *, int> maxLatencies;
  std::unordered_map<mlir::Operation *, int> nodeCounts;

  for(auto &op: startNodeCandidates) {
    std::vector<std::string> path = graph.findLongestNonCyclicPath(op);
    maxLatencies.insert({op, graph.getPathLatency(path)});
    nodeCounts.insert({op, path.size()});

    llvm::dbgs() << "\t [DBG] Longest path from " << op->getAttrOfType<StringAttr>("handshake.name").str() << " lat: " << graph.getPathLatency(path) << " : ";
    for(auto &node: path) {
      llvm::dbgs() << node << " ";
    }
    llvm::dbgs() << "\n";
  }

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


// TODO identify phi node if there is only 1 bb and identify correctly if there are multiple cond_br pointing to a single bb
std::unordered_map<unsigned, mlir::Operation *> HandshakeSizeLSQsPass::getPhiNodes(AdjListGraph graph, mlir::Operation *startNode) {
  std::unordered_map<unsigned, std::vector<mlir::Operation *>> phiNodeCandidates;
  std::unordered_map<unsigned, mlir::Operation *> phiNodes;
  std::vector<mlir::Operation *> branchOps = graph.getOperationsWithOpName("handshake.cond_br");
  std::vector<mlir::Operation *> forkOps = graph.getOperationsWithOpName("handshake.fork");

  std::vector<mlir::Operation *> srcOps = std::vector<mlir::Operation *>(branchOps.size() + forkOps.size());
  std::merge(branchOps.begin(), branchOps.end(), forkOps.begin(), forkOps.end(), srcOps.begin());

  // Insert start_node as a candidate (will be choosen anyway, but looks cleaner then special handling)
  phiNodeCandidates.insert({startNode->getAttrOfType<IntegerAttr>("handshake.bb").getUInt(), {startNode}});
  //llvm::dbgs() << "\t [DBG] Inserted Start Node: " << startNode->getAttrOfType<StringAttr>("handshake.name").str() << " for BB " << startNode->getAttrOfType<IntegerAttr>("handshake.bb").getUInt() << "\n";

  for(auto &srcOp: srcOps) {
    unsigned srcBB =srcOp->getAttrOfType<IntegerAttr>("handshake.bb").getUInt();
    //llvm::dbgs() << "\t [DBG] Branch Op: " << branchOp->getAttrOfType<StringAttr>("handshake.name").str() << " of BB " << srcBB <<"\n";

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


  for(auto &entry: phiNodeCandidates) {
    mlir::Operation *phiNode = nullptr;
    int minLatency = INT_MAX;
    llvm::dbgs() << "\t [DBG] Phi Node Candidates for BB " << entry.first << ":\n";
    for(auto &op: entry.second) {
      int latency = graph.findMinPathLatency(startNode, op, true); //TODO think about if backedges should be ignored or not
      llvm::dbgs() << "\t\t [DBG] Latency from " << startNode->getAttrOfType<StringAttr>("handshake.name").str() << " to " << op->getAttrOfType<StringAttr>("handshake.name").str() << " is " << latency << "\n";
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



std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::lsqsizing::createHandshakeSizeLSQs(StringRef timingModels) {
  return std::make_unique<HandshakeSizeLSQsPass>(timingModels);
}
