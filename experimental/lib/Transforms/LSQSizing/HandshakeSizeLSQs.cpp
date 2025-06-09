//===- HandshakeSizeLSQs.cpp - LSQ Sizing -----------------------*- C++ -*-===//
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
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Transforms/LSQSizing/LSQSizingSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "handshake-size-lsqs"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::lsqsizing;

/// TUPLE: <load_size, store_size>
using LSQSizingResult =
    std::unordered_map<mlir::Operation *, std::tuple<unsigned, unsigned>>;

/// TUPLE: <Operation, time>
using StartTimes = std::vector<std::tuple<mlir::Operation *, int>>;
using SizePerOpMap = std::unordered_map<mlir::Operation *, unsigned>;
using TimePerOpMap = std::unordered_map<mlir::Operation *, int>;
using AllocDeallocTimesPerII =
    std::unordered_map<unsigned,
                       std::tuple<std::vector<int>, std::vector<int>>>;
namespace {

struct HandshakeSizeLSQsPass
    : public dynamatic::experimental::lsqsizing::impl::HandshakeSizeLSQsBase<
          HandshakeSizeLSQsPass> {

  HandshakeSizeLSQsPass(StringRef timingModels, StringRef collisions,
                        double targetCP) {
    this->targetCP = targetCP;
    this->timingModels = timingModels.str();
    this->collisions = collisions.str();
  }

  void runDynamaticPass() override;

private:
  // There is a offset between the arrival time of the arguments and the actual
  // allocation/deallocation time inside the LSQ For allocation it is the same
  // for both load and stores, for deallocation it is different between loads
  // and stores
  static const unsigned allocEntryLatency = 1;
  static const unsigned storeDeallocEntryLatency = 2;
  static const unsigned loadDeallocEntryLatency = 1;

  /// Determines the LSQ sizes, given a CFDFC and its II
  std::optional<LSQSizingResult>
  sizeLSQsForCFDFC(handshake::FuncOp funcOp, llvm::SetVector<unsigned> cfdfcBBs,
                   TimingDatabase timingDB, unsigned initialII,
                   std::string collisions, double targetCP);

  /// Finds the Start Node in a CFDFC
  /// The start node, is the node with the longest non-cyclic path to any other
  /// node
  std::tuple<mlir::Operation *, StartTimes> findStartTimes(CFDFCGraph graph);

  /// Inserts edges to handle an edge case where the start node is not connected
  /// to other "potential start nodes", which do not start at the same time, but
  /// could supply an argument to a node in the graph. This argument would
  /// impact the timing of the node, but would not be considered in timing
  /// anlysis. By adding an edge which has a aritificial node with a latency,
  /// that models the difference of this other "potential start node" in start
  /// time to the chosen start node, the timing analysis can be done correctly
  void insertStartnodeShiftingEdges(CFDFCGraph &graph,
                                    mlir::Operation *startNode,
                                    StartTimes startTimes);

  /// Finds the Phi Node for each Basic Block in a CFDFC
  /// Checks all operations which get their input from a different BB and
  /// chooses the one with the lowest latency from the start node
  std::unordered_map<unsigned, mlir::Operation *>
  getPhiNodes(CFDFCGraph graph, mlir::Operation *startNode);

  /// Inserts edges to make sure that sizing is done correctly
  /// These edges make sure that in the timing analysis, the allocation precedes
  /// the memory access Edges are inserted from the Phi Node of a BB to the LSQ
  /// operations in the BB Therefore the LSQ operations will be allocated at the
  /// start of the BB
  void insertAllocPrecedesMemoryAccessEdges(
      CFDFCGraph &graph, std::vector<mlir::Operation *> ops,
      std::unordered_map<unsigned, mlir::Operation *> phiNodes);

  /// Finds the allocation time of each operation, which is the earliest start
  /// time of the Phi Node of the BB plus a fixed additional latency
  TimePerOpMap
  getAllocTimes(CFDFCGraph graph, mlir::Operation *startNode,
                std::vector<mlir::Operation *> ops,
                std::unordered_map<unsigned, mlir::Operation *> phiNodes);

  /// Finds the deallocation time of each store operation, which is the the
  /// latency of the last argument arriving plus a fixed additional latency
  TimePerOpMap getStoreDeallocTimes(CFDFCGraph graph,
                                    mlir::Operation *startNode,
                                    std::vector<mlir::Operation *> storeOps);

  /// Finds the deallocation time of each load operation, which is the latest
  /// argument arriving at the operation succeding the load operation plus a
  /// fixed additional latency This is due to to the fact that the load
  /// operation frees the queue entry as soon as the load result is passed on to
  /// the succeeding operation
  TimePerOpMap getLoadDeallocTimes(CFDFCGraph graph, mlir::Operation *startNode,
                                   std::vector<mlir::Operation *> loadOps);

  /// Given the alloc and dealloc times of each operation, calculates the
  /// maximum queue size needed for each LSQ
  SizePerOpMap
  calcQueueSize(std::unordered_map<unsigned, TimePerOpMap> allocTimes,
                std::unordered_map<unsigned, TimePerOpMap> deallocTimes,
                std::vector<unsigned> IIs);
};
} // namespace

void HandshakeSizeLSQsPass::runDynamaticPass() {
  llvm::SmallVector<LSQSizingResult> sizingResults;

  // Read component latencies
  TimingDatabase timingDB(&getContext());
  if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
    signalPassFailure();

  mlir::ModuleOp mod = getOperation();
  for (handshake::FuncOp funcOp : mod.getOps<handshake::FuncOp>()) {

    std::unordered_map<unsigned, llvm::SetVector<unsigned>> cfdfcBBLists;
    std::unordered_map<unsigned, float> IIs;

    // Extract CFDFCs and II from the Attributes
    auto throughputAttr =
        getDialectAttr<handshake::CFDFCThroughputAttr>(funcOp);
    auto cfdfcAttr = getDialectAttr<handshake::CFDFCToBBListAttr>(funcOp);

    if (throughputAttr == nullptr || cfdfcAttr == nullptr)
      continue;

    DictionaryAttr throughputDict = throughputAttr.getThroughputMap();
    DictionaryAttr cfdfcDict = cfdfcAttr.getCfdfcMap();

    // Convert Attribute into better usable data structure
    for (const mlir::NamedAttribute attr : cfdfcDict) {
      ArrayAttr bbList = llvm::dyn_cast<ArrayAttr>(attr.getValue());
      llvm::SetVector<unsigned> cfdfcBBs;
      for (auto bb : bbList)
        cfdfcBBs.insert(bb.cast<IntegerAttr>().getUInt());

      unsigned index;
      if (attr.getName().getValue().getAsInteger(10, index))
        signalPassFailure();

      cfdfcBBLists.insert({index, cfdfcBBs});
    }

    // Extract II from Attribute
    for (const NamedAttribute attr : throughputDict) {
      FloatAttr throughput = llvm::dyn_cast<FloatAttr>(attr.getValue());
      IIs.insert({std::stoi(attr.getName().str()),
                  round(1 / throughput.getValueAsDouble())});
    }

    std::map<mlir::Operation *, std::tuple<unsigned, unsigned>>
        maxLoadStoreSizes;
    // Size LSQs for each CFDFC
    // And extract maximum Queue sizes for each LSQ
    for (auto &entry : cfdfcBBLists) {
      // If there is no II corresponding to the CFDFC skip it
      // Something is wrong with the attributes in that case
      if (IIs.find(entry.first) == IIs.end())
        continue;

      std::optional<LSQSizingResult> result =
          sizeLSQsForCFDFC(funcOp, entry.second, timingDB, IIs.at(entry.first),
                           collisions, targetCP);

      if (result) {
        for (auto &entry : result.value()) {
          int newMaxLoadSize =
              std::max(std::get<0>(maxLoadStoreSizes[entry.first]),
                       std::get<0>(entry.second));
          int newMaxStoreSize =
              std::max(std::get<1>(maxLoadStoreSizes[entry.first]),
                       std::get<1>(entry.second));
          maxLoadStoreSizes[entry.first] =
              std::make_tuple(newMaxLoadSize, newMaxStoreSize);
        }
      }
    }

    // Set the maximum Queue sizes as attributes for backend

    for (auto &[lsqOp, maxSizes] : maxLoadStoreSizes) {
      unsigned maxLoadSize = std::get<0>(maxSizes);
      unsigned maxStoreSize = std::get<1>(maxSizes);

      // A LSQ with size 1 or 0 does not make sense and cant be generated by the
      // LSQ Generator, a minimum size of 2 is needed
      maxLoadSize = std::max(maxLoadSize, (unsigned)2);
      maxStoreSize = std::max(maxStoreSize, (unsigned)2);

      handshake::LSQDepthAttr lsqDepthAttr = handshake::LSQDepthAttr::get(
          mod.getContext(), maxLoadSize, maxStoreSize);
      setDialectAttr(lsqOp, lsqDepthAttr);
    }
  }
}

std::optional<LSQSizingResult> HandshakeSizeLSQsPass::sizeLSQsForCFDFC(
    handshake::FuncOp funcOp, llvm::SetVector<unsigned> cfdfcBBs,
    TimingDatabase timingDB, unsigned initialII, std::string collisions,
    double targetCP) {

  CFDFCGraph graph(funcOp, cfdfcBBs, timingDB, initialII, targetCP);

  // We only want LSQ loads and stores (not MC loads and stores), therefore we
  // need to check if they are connected to an LSQ
  std::vector<mlir::Operation *> loadOps;
  std::vector<mlir::Operation *> storeOps;
  for (auto op : graph.getOperationsWithOpType<handshake::LoadOp>()) {
    if (graph.isConnectedToLSQ(op))
      loadOps.push_back(op);
  }

  for (auto op : graph.getOperationsWithOpType<handshake::StoreOp>()) {
    if (graph.isConnectedToLSQ(op))
      storeOps.push_back(op);
  }

  if (loadOps.size() == 0 && storeOps.size() == 0)
    return std::nullopt;

  // Find starting node, which will be the reference to the rest
  std::tuple<mlir::Operation *, StartTimes> startNodeAndTimes =
      findStartTimes(graph);
  mlir::Operation *startNode = get<0>(startNodeAndTimes);
  StartTimes startTimes = get<1>(startNodeAndTimes);

  // Find Phi node of each BB
  std::unordered_map<unsigned, mlir::Operation *> phiNodes =
      getPhiNodes(graph, startNode);

  // connect all phi nodes to the lsq ops in their BB
  insertAllocPrecedesMemoryAccessEdges(graph, loadOps, phiNodes);

  // connect all start node candidates, with a latency of their shifting time,
  // for the case that there is no path from the start node to the start node
  // candidate and that the start time of the candidate is not zero
  insertStartnodeShiftingEdges(graph, startNode, startTimes);

  // Calculate the earliest start times for each operation in the graph. This is
  // needed to find the allocation times
  graph.setEarliestStartTimes(startNode);

  std::vector<unsigned> IIs;

  // The collisions argument can be passed in the compile script
  // For now there are 3 defined cases: full, half, none
  // full: determines the worst case II, for the case that all lsq ops have
  // memory collisions
  // half: determines the worst case II and alternate it with
  // the best case II (from buffer placement), for the case that half of the lsq
  // ops have memory collisions
  // none: gets the best case II(from buffer
  // placement), for the case that no lsq ops have memory collisions
  //
  // It would also easily be possible to add more cases, by just pushing the IIs
  // into the IIs vector in the order they are occuring

  // Important side note: The II observed in simulation does not match the II
  // received from buffer placement in some cases. This is also the case when
  // there are no memory collisions. It is not clear why this is the case and
  // would need further investigation by someone familiar with the buffer
  // placement pass

  if (collisions == "full") {
    IIs.push_back(graph.getWorstCaseII());
  } else if (collisions == "half") {
    IIs.push_back(initialII);
    IIs.push_back(graph.getWorstCaseII());
  } else {
    // none is the default case
    IIs.push_back(initialII);
  }

  std::unordered_map<unsigned, TimePerOpMap> loadAllocTimes;
  std::unordered_map<unsigned, TimePerOpMap> storeAllocTimes;
  std::unordered_map<unsigned, TimePerOpMap> loadDeallocTimes;
  std::unordered_map<unsigned, TimePerOpMap> storeDeallocTimes;

  for (auto &II : IIs) {
    graph.setNewII(II);
    loadAllocTimes.insert_or_assign(
        II, getAllocTimes(graph, startNode, loadOps, phiNodes));
    storeAllocTimes.insert_or_assign(
        II, getAllocTimes(graph, startNode, storeOps, phiNodes));
    loadDeallocTimes.insert_or_assign(
        II, getLoadDeallocTimes(graph, startNode, loadOps));
    storeDeallocTimes.insert_or_assign(
        II, getStoreDeallocTimes(graph, startNode, storeOps));
  }
  // Get Load and Store Sizes
  SizePerOpMap loadSizes = calcQueueSize(loadAllocTimes, loadDeallocTimes, IIs);
  SizePerOpMap storeSizes =
      calcQueueSize(storeAllocTimes, storeDeallocTimes, IIs);

  LSQSizingResult result;
  for (auto &entry : loadSizes) {
    unsigned storeSize = storeSizes.find(entry.first) != storeSizes.end()
                             ? storeSizes[entry.first]
                             : 0;
    result.insert({entry.first, std::make_tuple(entry.second, storeSize)});
  }

  for (auto &entry : storeSizes) {
    if (result.find(entry.first) == result.end())
      result.insert({entry.first, std::make_tuple(0, entry.second)});
  }

  return result;
}

std::tuple<mlir::Operation *, StartTimes>
HandshakeSizeLSQsPass::findStartTimes(CFDFCGraph graph) {
  StartTimes startTimes;

  std::vector<mlir::Operation *> startNodeCandidates =
      graph.getOperationsWithOpType<handshake::MergeLikeOpInterface>();

  std::unordered_map<mlir::Operation *, unsigned> maxLatencies;
  std::unordered_map<mlir::Operation *, unsigned> nodeCounts;

  // Go trough all candidates and save the longest path, its latency and node
  // count
  for (auto &op : startNodeCandidates) {
    std::vector<std::string> path = graph.findLongestNonCyclicPath(op);
    maxLatencies.insert({op, graph.getPathLatency(path)});
    nodeCounts.insert({op, path.size()});
  }

  // Find the node with the highest latency, if there are multiple, choose the
  // one with the most nodes
  mlir::Operation *maxLatencyNode = nullptr;
  unsigned maxLatency = 0;
  unsigned maxNodeCount = 0;
  for (auto &node : maxLatencies) {
    if (node.second > maxLatency) {
      maxLatency = node.second;
      maxLatencyNode = node.first;
      maxNodeCount = nodeCounts[node.first];
    } else if (node.second == maxLatency &&
               nodeCounts[node.first] > maxNodeCount) {
      maxLatencyNode = node.first;
      maxNodeCount = nodeCounts[node.first];
    }
  }

  for (auto &entry : maxLatencies)
    startTimes.push_back({entry.first, maxLatency - entry.second});

  return {maxLatencyNode, startTimes};
}

void HandshakeSizeLSQsPass::insertStartnodeShiftingEdges(
    CFDFCGraph &graph, mlir::Operation *startNode, StartTimes startTimes) {
  for (auto &entry : startTimes) {
    // Ignore if it is the startNode itself
    if (get<0>(entry) != startNode) {
      // Check if there is already a Path connecting the Nodes, if not insert
      // an edge with an artifical node The artificial node has the shifting
      // latency as latency
      std::vector<std::vector<std::string>> paths =
          graph.findPaths(startNode, get<0>(entry), true);
      if (paths.size() == 0)
        graph.addShiftingEdge(startNode, get<0>(entry), get<1>(entry));
    }
  }
}

std::unordered_map<unsigned, mlir::Operation *>
HandshakeSizeLSQsPass::getPhiNodes(CFDFCGraph graph,
                                   mlir::Operation *startNode) {
  std::unordered_map<unsigned, std::vector<mlir::Operation *>>
      phiNodeCandidates;
  std::unordered_map<unsigned, mlir::Operation *> phiNodes;

  // Insert start_node as a candidate for cases where there is only 1 bb (will
  // be choosen anyway for other cases, but looks cleaner then special
  // handling)
  std::optional<unsigned> startNodeBB = getLogicBB(startNode);
  assert(startNodeBB && "Start Node must belong to basic block");
  phiNodeCandidates.insert({*startNodeBB, {startNode}});

  // Go trought ops and find connected ops
  for (auto &srcOp : graph.getOperations()) {
    std::optional<unsigned> srcBB = getLogicBB(srcOp);
    assert(srcBB && "Src Op must belong to basic block");
    // For each connected Op, check if its in a different BB and add it to the
    // candidates
    for (auto &destOp : graph.getConnectedOps(srcOp)) {
      std::optional<unsigned> destBB = getLogicBB(destOp);
      assert(destBB && "Dest Op must belong to basic block");
      if (*destBB != *srcBB) {
        if (phiNodeCandidates.find(*destBB) == phiNodeCandidates.end())
          phiNodeCandidates.insert({*destBB, std::vector<mlir::Operation *>()});
        phiNodeCandidates.at(*destBB).push_back(destOp);
      }
    }
  }

  // Go trough all candidates and choose the one with the lowest latency to
  // the start node, ignore backedges for the path
  for (auto &entry : phiNodeCandidates) {
    mlir::Operation *phiNode = nullptr;
    int minLatency = INT_MAX;
    for (auto &op : entry.second) {
      int latency = graph.findMinPathLatency(startNode, op, true);

      if (latency < minLatency) {
        phiNode = op;
        minLatency = latency;
      }
    }
    phiNodes.insert({entry.first, phiNode});
  }

  return phiNodes;
}

std::unordered_map<mlir::Operation *, int> HandshakeSizeLSQsPass::getAllocTimes(
    CFDFCGraph graph, mlir::Operation *startNode,
    std::vector<mlir::Operation *> ops,
    std::unordered_map<unsigned, mlir::Operation *> phiNodes) {
  std::unordered_map<mlir::Operation *, int> allocTimes;

  // Go trough all ops and find the latency to the phi node of the ops BB
  for (auto &op : ops) {
    std::optional<unsigned> bb = getLogicBB(op);
    assert(bb && "Load/Store Op must belong to basic block");
    mlir::Operation *phiNode = phiNodes[*bb];
    assert(phiNode && "Phi node not found for BB");
    int latency = graph.getEarliestStartTime(phiNode) + allocEntryLatency;
    allocTimes.insert({op, latency});
  }
  return allocTimes;
}

// For a store operation, the deallocation time is the time when the last
// argument arrives at the store operation plus a fixed additional latency.
// This is because the store operation can deallocate the slot as soon as the
// data and address are known and therefore can issue the memory operation.
std::unordered_map<mlir::Operation *, int>
HandshakeSizeLSQsPass::getStoreDeallocTimes(
    CFDFCGraph graph, mlir::Operation *startNode,
    std::vector<mlir::Operation *> ops) {
  std::unordered_map<mlir::Operation *, int> deallocTimes;

  // Go trough all ops and find the maximum latency to the op node
  for (auto &op : ops) {
    int latency = graph.findMaxPathLatency(startNode, op, false, false, true) +
                  storeDeallocEntryLatency;
    deallocTimes.insert({op, latency});
  }
  return deallocTimes;
}

// Go trough all ops which directly succeed the load op. These ops are waiting
// for the result of the load operation. Check every succeeding node for the
// maximum latency (latest argument arrival), which is the time it can accept
// the results from the load operation. As soon as the succeeding operation
// has accepted the result, the LSQ can deallocate the slot. The latency of
// the succeding operation itself does not matter, because only the time when
// the succeeding operation can accept the result is relevant for the
// deallocation time.
std::unordered_map<mlir::Operation *, int>
HandshakeSizeLSQsPass::getLoadDeallocTimes(CFDFCGraph graph,
                                           mlir::Operation *startNode,
                                           std::vector<mlir::Operation *> ops) {
  std::unordered_map<mlir::Operation *, int> deallocTimes;

  for (auto &op : ops) {
    int maxLatency = 0;
    for (auto &succedingOp : graph.getConnectedOps(op)) {

      maxLatency = std::max(
          graph.findMaxPathLatency(startNode, succedingOp, false, false, true) +
              (int)loadDeallocEntryLatency,
          maxLatency);

      // If the node is a buffer, check if it is a tehb buffer and if so,
      // check the latency of the nodes connected to the buffer
      if (isa<handshake::BufferOp>(succedingOp)) {
        auto params = succedingOp->getAttrOfType<DictionaryAttr>(
            RTL_PARAMETERS_ATTR_NAME);

        if (!params)
          continue;

        auto optTiming = params.getNamed(handshake::BufferOp::TIMING_ATTR_NAME);
        if (!optTiming)
          continue;

        auto timing = dyn_cast<handshake::TimingAttr>(optTiming->getValue());
        if (!timing)
          continue;

        handshake::TimingInfo info = timing.getInfo();

        if (info == TimingInfo::break_r() || info == TimingInfo::break_none()) {
          for (auto &succedingOp2 : graph.getConnectedOps(succedingOp)) {
            // -1 because buffer can get the load result 1 cycle earlier
            // Maybe it could also be earlier for a buffer with multiple slots
            // But its not clear for now
            maxLatency = std::max(
                maxLatency, graph.findMaxPathLatency(startNode, succedingOp2,
                                                     false, false, true) +
                                (int)loadDeallocEntryLatency - 1);
          }
        }
      }
    }
    deallocTimes.insert({op, maxLatency});
  }
  return deallocTimes;
}

std::unordered_map<mlir::Operation *, unsigned>
HandshakeSizeLSQsPass::calcQueueSize(
    std::unordered_map<unsigned, TimePerOpMap> allocTimes,
    std::unordered_map<unsigned, TimePerOpMap> deallocTimes,
    std::vector<unsigned> IIs) {
  std::unordered_map<mlir::Operation *, unsigned> queueSizes;

  std::unordered_map<mlir::Operation *, AllocDeallocTimesPerII>
      allocDeallocTimesPerIIPerLSQ;

  // Go trough all alloc times and sort them by LSQ and II
  for (auto &entry : allocTimes) {
    TimePerOpMap allocTimeMapForII = entry.second;
    for (auto &timeForOp : allocTimeMapForII) {
      mlir::Operation *lsqOp = nullptr;
      for (Operation *destOp : timeForOp.first->getUsers()) {
        if (isa<handshake::LSQOp>(destOp)) {
          lsqOp = destOp;
          break;
        }
      }
      if (allocDeallocTimesPerIIPerLSQ.find(lsqOp) ==
          allocDeallocTimesPerIIPerLSQ.end()) {
        allocDeallocTimesPerIIPerLSQ.insert({lsqOp, AllocDeallocTimesPerII()});
      }
      if (allocDeallocTimesPerIIPerLSQ.at(lsqOp).find(entry.first) ==
          allocDeallocTimesPerIIPerLSQ.at(lsqOp).end()) {
        allocDeallocTimesPerIIPerLSQ.at(lsqOp).insert(
            {entry.first,
             std::make_tuple(std::vector<int>(), std::vector<int>())});
      }
      std::get<0>(allocDeallocTimesPerIIPerLSQ.at(lsqOp).at(entry.first))
          .push_back(timeForOp.second);
    }
  }

  // Go trough all dealloc times and sort them by LSQ and II
  for (auto &entry : deallocTimes) {
    TimePerOpMap deallocTimeMapForII = entry.second;

    for (auto &timeForOp : deallocTimeMapForII) {
      mlir::Operation *lsqOp = nullptr;
      for (Operation *destOp : timeForOp.first->getUsers()) {
        if (isa<handshake::LSQOp>(destOp)) {
          lsqOp = destOp;
          break;
        }
      }
      if (allocDeallocTimesPerIIPerLSQ.find(lsqOp) ==
          allocDeallocTimesPerIIPerLSQ.end()) {
        allocDeallocTimesPerIIPerLSQ.insert({lsqOp, AllocDeallocTimesPerII()});
      }
      if (allocDeallocTimesPerIIPerLSQ.at(lsqOp).find(entry.first) ==
          allocDeallocTimesPerIIPerLSQ.at(lsqOp).end()) {
        allocDeallocTimesPerIIPerLSQ.at(lsqOp).insert(
            {entry.first,
             std::make_tuple(std::vector<int>(), std::vector<int>())});
      }
      std::get<1>(allocDeallocTimesPerIIPerLSQ.at(lsqOp).at(entry.first))
          .push_back(timeForOp.second);
    }
  }

  int maxEndTime = 0;
  // Choose the maxiumm time of all dealloc times for analysis time scope
  for (auto &II : IIs)
    for (auto &entry : deallocTimes.at(II))
      maxEndTime = std::max(maxEndTime, entry.second);

  // Double the time for the analysis scope to make sure that all
  // deallocations are included (necessary scope actually is lower, but not
  // trivial to determine)
  maxEndTime = maxEndTime * 2;

  // Go trough all LSQs and calculate the maximum amount of slots needed
  for (auto &entry : allocDeallocTimesPerIIPerLSQ) {
    std::vector<int> allocPerCycle(maxEndTime);
    int startOffset = 0;
    unsigned iter = 0;
    // Build array for how many slots are allocated and deallocated per cycle
    // Alternate trough the different IIs in the order they are in the array
    while (startOffset < maxEndTime) {
      unsigned II = IIs[iter % IIs.size()];

      for (auto &allocTime : std::get<0>(entry.second.at(II))) {
        int t = allocTime + startOffset;
        if (t >= 0 && t < maxEndTime)
          allocPerCycle[t]++;
      }
      for (auto &deallocTime : std::get<1>(entry.second.at(II))) {
        int t = deallocTime + startOffset;
        if (t >= 0 && t < maxEndTime)
          allocPerCycle[t]--;
      }
      // Increase the start offset for the next iteration by the II
      startOffset += II;
      iter++;
    }

    // build array for many slots are actively allocated at which cycle
    std::vector<int> slotsPerCycle(maxEndTime);
    slotsPerCycle[0] = allocPerCycle[0];
    for (int i = 1; i < maxEndTime; i++)
      slotsPerCycle[i] = slotsPerCycle[i - 1] + allocPerCycle[i];

    // get highest amount of slots from the array
    unsigned maxSlots =
        *std::max_element(slotsPerCycle.begin(), slotsPerCycle.end());
    queueSizes.insert({entry.first, maxSlots});
  }

  return queueSizes;
}

void HandshakeSizeLSQsPass::insertAllocPrecedesMemoryAccessEdges(
    CFDFCGraph &graph, std::vector<mlir::Operation *> ops,
    std::unordered_map<unsigned, mlir::Operation *> phiNodes) {
  // Iterate over all provided ops and add an edge from the phi node of the
  // ops BB to the op
  for (auto &op : ops) {
    std::optional<unsigned> bb = getLogicBB(op);
    assert(bb && "Load/Store Op must belong to basic block");
    mlir::Operation *phiNode = phiNodes[*bb];
    graph.addEdge(phiNode, op);
  }
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::lsqsizing::createHandshakeSizeLSQs(
    StringRef timingModels, StringRef collisions, double targetCP) {
  return std::make_unique<HandshakeSizeLSQsPass>(timingModels, collisions,
                                                 targetCP);
}