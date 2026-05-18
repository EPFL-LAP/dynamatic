//===- HandshakeInactivateDeps.cpp - Inactivate memory deps -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "handshake-inactivate-deps"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKEINACTIVATEDEPS
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]


namespace {

struct HandshakeInactivateDepsPass
    : public dynamatic::impl::HandshakeInactivateDepsBase<
          HandshakeInactivateDepsPass> {

  using HandshakeInactivateDepsBase::HandshakeInactivateDepsBase;

  void runDynamaticPass() override;
};

} // namespace

void HandshakeInactivateDepsPass::runDynamaticPass() {
}
