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
  AdjListGraph createAdjacencyList(buffer::CFDFC cfdfc, unsigned II, TimingDatabase timingDB);
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
    DenseMap<unsigned, SmallVector<unsigned>> cfdfc_attribute = {{0, {2}}, {1, {3, 1, 2}}}; // = funcOp.getCFDFCs();
    DenseMap<unsigned, float> troughput_attribute = {{0, 3.333333e-01}, {1, 2.000000e-01}}; // = funcOp.getThroughput();      
    

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

  AdjListGraph graph = createAdjacencyList(cfdfc, II, timingDB);
  

  // Add additional edges for Allocation preceding Memory access
  // Add additional nodes for backededge with -II latency
  
  
  // Get Start Times of each BB 
  // 

  return DenseMap<unsigned, std::tuple<unsigned, unsigned>>();
}


AdjListGraph HandshakeSizeLSQsPass::createAdjacencyList(buffer::CFDFC cfdfc, unsigned II, TimingDatabase timingDB) {
  AdjListGraph graph;

  for(auto &unit: cfdfc.units) {
    double latency;
    //llvm::dbgs() << "unit: " << unit->getAttrOfType<StringAttr>("handshake.name") << "\n";
    if(failed(timingDB.getLatency(unit, SignalType::DATA, latency))) {
      //llvm::dbgs() << "No latency found for unit: " << unit->getName().getStringRef() << " found \n";
      graph.addNode(unit, 0);
    } 
    else {
      graph.addNode(unit, latency);
    }
  }

  for(auto &channel: cfdfc.channels) {
    mlir::Operation *src_op = channel.getDefiningOp();
    for(Operation *dest_op: channel.getUsers()) {
      graph.addEdge(src_op, dest_op);
    }
  }

  for(auto &backedge: cfdfc.backedges) {
    mlir::Operation *src_op = backedge.getDefiningOp();
    for(Operation *dest_op: backedge.getUsers()) {
      llvm::dbgs() << "backedge: " << src_op->getAttrOfType<StringAttr>("handshake.name") << " -> " << dest_op->getAttrOfType<StringAttr>("handshake.name") << "\n";
      //graph.addEdge(std::string(src_op->getName().getStringRef()), std::string(dest_op->getName().getStringRef()));
    }
  }

  //TODO add backedge extra nodes with latency
  //TODO add extra vertices for "allocation precedes memory access"  

  return graph;

}


std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::lsqsizing::createHandshakeSizeLSQs(StringRef timingModels) {
  return std::make_unique<HandshakeSizeLSQsPass>(timingModels);
}

