//===- ForceMemoryInterface.cpp - Force interface in Handshake --*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --force-memory-interface pass, internally adding/modifying the
// `handshake::MemInterfaceAttr` to/on all memory operations to force placement
// of a specific type of memory interface.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Support/Attribute.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

// [START Boiler-plate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_FORCEMEMORYINTERFACE
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boiler-plate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;

namespace {

/// Simple driver for memory interface forcing pass.
struct ForceMemoryInterfacePass
    : public dynamatic::impl::ForceMemoryInterfaceBase<
          ForceMemoryInterfacePass> {

  using ForceMemoryInterfaceBase::ForceMemoryInterfaceBase;

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

    MLIRContext *ctx = &getContext();
    DenseMap<Block *, unsigned> lsqGroups;
    unsigned nextGroupID = 0;

    // Find all memory operations and adds/modifies the
    // handshake::MemInterfaceAttr on them depending on the pass parameters
    getOperation()->walk([&](Operation *op) {
      // This only makes sense on load/store-like operations
      if (!isa<memref::LoadOp, memref::StoreOp, affine::AffineLoadOp,
               affine::AffineStoreOp>(op))
        return;

      if (forceMC) {
        setDialectAttr<handshake::MemInterfaceAttr>(op, ctx);
        return;
      }

      // Make every block its own LSQ group
      Block *block = op->getBlock();
      unsigned groupID;

      // Try to find the block's group ID. Failing that, assign a new group ID
      // to the block
      if (auto groupIt = lsqGroups.find(block); groupIt != lsqGroups.end())
        groupID = groupIt->second;
      else
        lsqGroups[block] = groupID = nextGroupID++;

      setDialectAttr<handshake::MemInterfaceAttr>(op, ctx, groupID);
    });
  }
};
} // namespace
