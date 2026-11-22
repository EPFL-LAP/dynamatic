//===- HandshakeUseBanbaLSQs.cpp --------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
// include tblgen base class definition
#define GEN_PASS_DEF_HANDSHAKEUSEBANBALSQS
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace {

struct HandshakeUseBanbaLSQsPass
    : public dynamatic::impl::HandshakeUseBanbaLSQsBase<
          HandshakeUseBanbaLSQsPass> {
  using HandshakeUseBanbaLSQsBase::HandshakeUseBanbaLSQsBase;

  void runOnOperation() override {
    getOperation().walk([&](LSQOp lsqOp) { lsqOp.setBanba(true); });
  }
};

} // namespace
