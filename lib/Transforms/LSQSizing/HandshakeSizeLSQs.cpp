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

#include "dynamatic/Transforms/LSQSizing/HandshakeSizeLSQs.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "handshake-size-lsqs"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace {

struct HandshakeSizeLSQsPass
    : public dynamatic::impl::HandshakeSizeLSQsBase<
          HandshakeSizeLSQsPass> {

  void runDynamaticPass() override;

private:

};
} // namespace

void HandshakeSizeLSQsPass::runDynamaticPass() {
      llvm::dbgs() << "\t [DBG] LSQ Sizing Pass Called!\n";
}


std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeSizeLSQs() {
  return std::make_unique<HandshakeSizeLSQsPass>();
}
