//===- HandshakeSetUnitImplAttributes.cpp------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/TimingModels.h"

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
// include tblgen base class definition
#define GEN_PASS_DEF_HANDSHAKESETUNITIMPLATTRIBUTES
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace {

std::optional<FPUImpl> symbolizeFPUImplOrEmitError(StringRef implStr) {
  if (auto parsed = symbolizeFPUImpl(implStr))
    return parsed;

  llvm::errs() << "Invalid FPU implementation: '" << implStr << "'\n";
  llvm::errs() << "Valid FPU Implementations:\n";
  for (int64_t i = 0; i <= getMaxEnumValForFPUImpl(); ++i) {
    if (auto e = symbolizeFPUImpl(i))
      llvm::errs() << "  '" << stringifyFPUImpl(*e) << "'\n";
  }
  return std::nullopt;
}

struct HandshakeSetUnitImplAttributesPass
    : public dynamatic::impl::HandshakeSetUnitImplAttributesBase<
          HandshakeSetUnitImplAttributesPass> {
  using HandshakeSetUnitImplAttributesBase::HandshakeSetUnitImplAttributesBase;

  void runOnOperation() override {

    TimingDatabase timingDB;
    if (failed(TimingDatabase::readFromJSON(timingModels, timingDB)))
      llvm::errs() << "=== TimindDB read failed ===\n";

    auto implOpt = symbolizeFPUImplOrEmitError(this->impl);
    if (!implOpt.has_value()) {
      signalPassFailure();
      return;
    }

    // FPUImpl local variable hides std::string pass option
    FPUImpl impl = implOpt.value();

    // iterate over each operation that implements FPUImplInterface
    getOperation().walk([&](FPUImplInterface fpuImplInterfaceOp) {
      // and set it based on the pass option
      // mark the FPU vendor
      fpuImplInterfaceOp.setFPUImpl(impl);

      double delay;
      // [START mark the internal delay of the FPU units]
      if (succeeded(timingDB.getInternalCombinationalDelay(
              fpuImplInterfaceOp, SignalType::DATA, delay, targetCP))) {
        std::string delayStr = std::to_string(delay);
        std::replace(delayStr.begin(), delayStr.end(), '.', '_');
        fpuImplInterfaceOp.setInternalDelay(delayStr);
      } else {
        fpuImplInterfaceOp->emitError(
            "Failed to get internal delay from timing model");
        return signalPassFailure();
      }
      // [END mark the internal delay of the FPU units]
    });

    getOperation().walk([&](LatencyInterface latencyInterfaceOp) {
      // [START mark the latency]
      double latency;
      if (!failed(timingDB.getLatency(latencyInterfaceOp, SignalType::DATA,
                                      latency, targetCP))) {

        int64_t latencyInt = static_cast<int64_t>(latency);
        latencyInterfaceOp.setLatency(latencyInt);
      } else {
        latencyInterfaceOp->emitError(
            "Failed to get latency from timing model");
        return signalPassFailure();
      }
      // [END mark the latency]
    });
  }
};

} // namespace
