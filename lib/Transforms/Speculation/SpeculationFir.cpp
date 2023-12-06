//===- SpeculationFir.cpp - Hardcoded Speculative units placement for FIR
//-----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the --speculation-fir pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/Speculation/SpeculationFir.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace {

struct placeSpeculativeUnitsFirPass
    : public dynamatic::speculation::impl::SpeculationFirBase<
          placeSpeculativeUnitsFirPass> {

  void runDynamaticPass() override {

    // Place the Speculator
    llvm::outs() << "Call functions with hardcoded correct parameters for FIR "
                    "benchmark from here!\n";
  }
};

} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::speculation::placeSpeculativeUnitsFir() {
  return std::make_unique<placeSpeculativeUnitsFirPass>();
}