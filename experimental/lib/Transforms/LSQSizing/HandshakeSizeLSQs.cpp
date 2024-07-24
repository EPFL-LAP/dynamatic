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
  std::unordered_map<unsigned, mlir::Operation *> getPhiNodes(AdjListGraph graph, mlir::Operation *start_node);
};
} // namespace


void HandshakeSizeLSQsPass::runDynamaticPass() {
  llvm::dbgs() << "\t [DBG] LSQ Sizing Pass Called!\n";

  std::map<unsigned,buffer::CFDFC> cfdfcs; //TODO chane to DenseMap?
  llvm::SmallVector<LSQSizingResult> sizing_results; //TODO datatype?

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

    // Read Attributes -> hardcoded for bicg
    
    // BICG
    DenseMap<unsigned, SmallVector<unsigned>> cfdfc_attribute = {{0, {2}}, {1, {3, 1, 2}}}; // = funcOp.getCFDFCs();
    DenseMap<unsigned, float> troughput_attribute = {{0, 3.333333e-01}, {1, 2.000000e-01}}; // = funcOp.getThroughput();      

    // FIR
    //DenseMap<unsigned, SmallVector<unsigned>> cfdfc_attribute = {{0, {1}}}; // = funcOp.getCFDFCs();
    //DenseMap<unsigned, float> troughput_attribute = {{0, 3.333333e-01}};   

    // Extract Arch sets
    for(auto &entry: cfdfc_attribute) {
      SmallVector<experimental::ArchBB> arch_store;
      auto it = entry.second.begin();
      //TODO Implement more clean?
      int first_bb_id = *it++;
      int curr_bb_id, prev_bb_id = first_bb_id;      
      for(; it != entry.second.end(); it++) {
        curr_bb_id = *it;
        arch_store.push_back(experimental::ArchBB(prev_bb_id, curr_bb_id, 0, false));
        prev_bb_id = curr_bb_id;
      }
      arch_store.push_back(experimental::ArchBB(prev_bb_id, first_bb_id, 0, false));

      llvm::dbgs() << "\t [DBG] CFDFC: " << entry.first << " with " << arch_store.size() << " arches\n";
      buffer::ArchSet arch_set;
      for(auto &arch: arch_store) {
        llvm::dbgs() << "\t [DBG] Arch: " << arch.srcBB << " -> " << arch.dstBB << "\n";
        arch_set.insert(&arch);
      }

      cfdfcs.insert_or_assign(entry.first, buffer::CFDFC(funcOp, arch_set, 0));
    }

    llvm::dbgs() << "\t [DBG] CFDFCs: " << cfdfcs.size() << "\n";
    for(auto &cfdfc : cfdfcs) {
      unsigned II = round(1 / troughput_attribute[cfdfc.first]);
      sizing_results.push_back(sizeLSQsForCFDFC(cfdfc.second, II, timingDB));
    }
    
    std::map<unsigned, unsigned> max_store_sizes;
    std::map<unsigned, unsigned> max_load_sizes;
    for(auto &result: sizing_results) {
      for(auto &entry: result) {
        max_store_sizes[entry.first] = std::max(max_store_sizes[entry.first], std::get<1>(entry.second));
        max_load_sizes[entry.first] = std::max(max_load_sizes[entry.first], std::get<0>(entry.second));
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
  mlir::Operation * start_node = findStartNode(graph);
  llvm::dbgs() << "\t [DBG] Start Node: " << start_node->getAttrOfType<StringAttr>("handshake.name").str()<< "\n";

  // Find Phi node of each BB
  std::unordered_map<unsigned, mlir::Operation *> phi_nodes = getPhiNodes(graph, start_node);

  // Get Start Times of each BB (Alloc Times) 


  // Get Dealloc Times and End Times
  std::vector<mlir::Operation *> load_ops = graph.getOperationsWithOpName("handshake.lsq_load");
  std::unordered_map<mlir::Operation *, int> load_dealloc_times;
  int load_end_time = 0;

  for(auto &op: load_ops) {
    int latency = graph.findMaxPathLatency(start_node, op);
    load_dealloc_times.insert({op, latency});
    load_end_time = std::max(load_end_time, latency);
  }

  std::vector<mlir::Operation *> store_ops = graph.getOperationsWithOpName("handshake.lsq_store");
  std::unordered_map<mlir::Operation *, int> store_dealloc_times;
  int store_end_time = 0;

  for(auto &op: store_ops) {
    int latency = graph.findMaxPathLatency(start_node, op);
    store_dealloc_times.insert({op, latency});
    store_end_time = std::max(store_end_time, latency);
  }

  // Get Load and Store Sizes

  return DenseMap<unsigned, std::tuple<unsigned, unsigned>>();
}


mlir::Operation * HandshakeSizeLSQsPass::findStartNode(AdjListGraph graph) {
  std::vector<mlir::Operation *> mux_ops = graph.getOperationsWithOpName("handshake.mux");
  std::vector<mlir::Operation *> cmerge_ops = graph.getOperationsWithOpName("handshake.control_merge");

  std::vector<mlir::Operation *> potential_start_nodes = std::vector<mlir::Operation *>(mux_ops.size() + cmerge_ops.size());
  std::merge(mux_ops.begin(), mux_ops.end(), cmerge_ops.begin(), cmerge_ops.end(), potential_start_nodes.begin());

  llvm::dbgs() << "\t [DBG] Potential Start Nodes: ";
  for(auto &op: potential_start_nodes) {
    llvm::dbgs() << op->getAttrOfType<StringAttr>("handshake.name").str() << ", ";
  }
  llvm::dbgs() << "\n";


  std::unordered_map<mlir::Operation *, int> max_latencies;

  for(auto &op: potential_start_nodes) {
    max_latencies.insert({op, graph.findMaxLatencyFromStart(op)});
  }

  return std::max_element(max_latencies.begin(), max_latencies.end(), 
    [](const std::pair<mlir::Operation *, int> &a, const std::pair<mlir::Operation *, int> &b) {
      return a.second < b.second;
    })->first;
}


std::unordered_map<unsigned, mlir::Operation *> HandshakeSizeLSQsPass::getPhiNodes(AdjListGraph graph, mlir::Operation *start_node) {
  std::unordered_map<unsigned, mlir::Operation *> phi_nodes;
  std::vector<mlir::Operation *> branch_ops = graph.getOperationsWithOpName("handshake.cond_br");
  for(auto &branch_op: branch_ops) {
    unsigned src_bb =branch_op->getAttrOfType<IntegerAttr>("handshake.bb").getUInt();
    llvm::dbgs() << "\t [DBG] Branch Op: " << branch_op->getAttrOfType<StringAttr>("handshake.name").str() << " of BB " << src_bb <<"\n";

    for(auto &dest_op: graph.getConnectedOps(branch_op)) {
      llvm::dbgs() << "\t\t [DBG] connected to: " << dest_op->getAttrOfType<StringAttr>("handshake.name").str() << "\n";
      unsigned dest_bb = dest_op->getAttrOfType<IntegerAttr>("handshake.bb").getUInt();
      if(dest_bb != src_bb) {
        llvm::dbgs() << "\t [DBG] Found Phi Node: " << dest_op->getAttrOfType<StringAttr>("handshake.name").str() << " for BB " << dest_bb << "\n";
        phi_nodes.insert({dest_bb, dest_op});
      }
    }
  }
  return phi_nodes;
}



std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::lsqsizing::createHandshakeSizeLSQs(StringRef timingModels) {
  return std::make_unique<HandshakeSizeLSQsPass>(timingModels);
}
