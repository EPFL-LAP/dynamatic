//===- HandshakeSpeculation.cpp - Speculative Dataflows ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Placement of Speculation components to enable speculative execution.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakeSpeculation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"

using namespace llvm::sys;
using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

HandshakeSpeculationPass::HandshakeSpeculationPass(
    StringRef srcOp, StringRef dstOp, bool dumpLogs) {
  this->srcOp = srcOp.str();
  this->dstOp = dstOp.str();
  this->dumpLogs = dumpLogs;
}

void HandshakeSpeculationPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Place the Speculator
  llvm::outs() << "Running speculation pass!\n";
  llvm::outs() << "Source op is <" << srcOp << "> and dest op is <" << dstOp << ">\n";
}

std::unique_ptr<dynamatic::DynamaticPass> 
dynamatic::buffer::createHandshakeSpeculation(
    StringRef srcOp, StringRef dstOp, bool dumpLogs) {
  return std::make_unique<HandshakeSpeculationPass>(
      srcOp, dstOp, dumpLogs);
}
