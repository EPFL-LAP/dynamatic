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

#define DEBUG_TYPE "handshake-size-lsqs"

using namespace mlir;
using namespace dynamatic;
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

  std::map<unsigned,buffer::CFDFC> cfdfcs;
  llvm::SmallVector<LSQSizingResult> sizing_results; //TODO datatype?
  LSQSizingResult sizeLSQsForCFDFC(buffer::CFDFC cfdfc, unsigned II, TimingDatabase timingDB);

};
} // namespace


void HandshakeSizeLSQsPass::runDynamaticPass() {
  llvm::dbgs() << "\t [DBG] LSQ Sizing Pass Called!\n";

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

    // Read Attributes
    DenseMap<unsigned, SmallVector<unsigned>> cfdfc_attribute; // = funcOp.getCFDFCs();
    DenseMap<unsigned, float> troughput_attribute; // = funcOp.getThroughput();      
    
    // Extract Arch sets
    for(auto &entry: cfdfc_attribute) {
      SmallVector<experimental::ArchBB> arch_store;
      auto it = entry.second.begin();
      //TODO Implement more clean?
      int prev_bb_id = *it++;      
      for(; it != entry.second.end(); it++) {
        arch_store.push_back(experimental::ArchBB(prev_bb_id, *it, 0, false));
        prev_bb_id = *it;
      }

      buffer::ArchSet arch_set;
      for(auto &arch: arch_store) {
        arch_set.insert(&arch);
      }

      cfdfcs.insert_or_assign(entry.first, buffer::CFDFC(funcOp, arch_set, 0));
    }

    //TODO create adjacent list

    for(auto &cfdfc : cfdfcs) {
      unsigned II = troughput_attribute[cfdfc.first];
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

  // Add additional edges for Allocation preceding Memory access
  // Add additional nodes for backededge with -II latency
  // Build Adjacency Lists
  // 

  return DenseMap<unsigned, std::tuple<unsigned, unsigned>>();
}


std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::lsqsizing::createHandshakeSizeLSQs(StringRef timingModels) {
  return std::make_unique<HandshakeSizeLSQsPass>(timingModels);
}

