//===- ForceMemoryInterface.cpp - Force interface in Handshake --*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --force-memory-interface pass, internally setting/removing the
// `handshake::NoLSQAttr` from all memory operations to force placement of a
// specific type of memory interface.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/ForceMemoryInterface.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace circt;
using namespace dynamatic;

namespace {

/// Simple driver for memory interface forcing pass.
struct ForceMemoryInterfacePass
    : public dynamatic::impl::ForceMemoryInterfaceBase<
          ForceMemoryInterfacePass> {

  ForceMemoryInterfacePass(bool forceLSQ, bool forceMC) {
    this->forceLSQ = forceLSQ;
    this->forceMC = forceMC;
  }

  void runDynamaticPass() override {
    // Exactly one of the two pass options need to have been set
    if (forceLSQ && forceMC)
      llvm::errs() << "Both " << forceLSQ.ArgStr << " and " << forceMC.ArgStr
                   << " flags were provided. However, only one can be set at "
                      "the same time.";
    if (!forceLSQ && !forceMC)
      llvm::errs()
          << "Neither " << forceLSQ.ArgStr << " and " << forceMC.ArgStr
          << " flags were provided. However, exactly one needs to be set.";

    // Find all memory operations and set/remove the handshake::NoLSQAttr
    // attribute on/from them depending on the pass parameters
    MLIRContext *ctx = &getContext();
    getOperation()->walk([&](MemoryEffectOpInterface memEffectOp) {
      if (forceLSQ)
        memEffectOp->removeAttr(handshake::NoLSQAttr::getMnemonic());
      else
        memEffectOp->setAttr(handshake::NoLSQAttr::getMnemonic(),
                             handshake::NoLSQAttr::get(ctx));
      ;
    });
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createForceMemoryInterface(bool forceLSQ, bool forceMC) {
  return std::make_unique<ForceMemoryInterfacePass>(forceLSQ, forceMC);
}
