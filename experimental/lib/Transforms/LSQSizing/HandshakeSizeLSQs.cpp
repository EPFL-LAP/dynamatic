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

  void runDynamaticPass() override;

private:

  llvm::SmallDenseSet<LSQSizingResult> sizing_results; //TODO datatype?

};
} // namespace


void HandshakeSizeLSQsPass::runDynamaticPass() {
  llvm::dbgs() << "\t [DBG] LSQ Sizing Pass Called!\n";

  // 1. Read Attributes
  // 2. Reconstruct CFDFCs
  // 3. ???
  // 4. Profit

  mlir::ModuleOp mod = getOperation();
  for (handshake::FuncOp funcOp : mod.getOps<handshake::FuncOp>()) {
    llvm::dbgs() << "\t [DBG] Function: " << funcOp.getName() << "\n";

    // Read Attributes
    //DenseMap<unsigned, SmallVector<unsigned>> cfdfc_attribute = funcOp.getCFDFCs();
    //DenseMap<unsigned, float> troughput_attribute = funcOp.getThroughput();

    // TODO extract arch sets
    SmallVector<experimental::ArchBB> archs;
    buffer::ArchSet arch_set;

    for(auto &arch: archs) {
      buffer::CFDFC cfdfc = buffer::CFDFC(funcOp, arch_set, 0);
      unsigned II = 0; //TODO get II from attr
      sizing_results.insert(sizeLSQsForCFDFC(cfdfc, II));
    }
    
    DenseMap<unsigned, unsigned> max_store_size;
    DenseMap<unsigned, unsigned> max_load_size;
    for(auto &result: sizing_results) {
      for(auto &entry: result) {
        max_store_size[entry.first] = std::max(max_store_size[entry.first], std::get<1>(entry.second));
        max_load_size[entry.first] = std::max(max_load_size[entry.first], std::get<0>(entry.second));
      }
    }

    // Add Sizing to Attributes
  }
}

LSQSizingResult sizeLSQsForCFDFC(buffer::CFDFC cfdfc, unsigned II) {
  //TODO implement algo
  return DenseMap<unsigned, std::tuple<unsigned, unsigned>>();
}


std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::lsqsizing::createHandshakeSizeLSQs() {
  return std::make_unique<HandshakeSizeLSQsPass>();
}

