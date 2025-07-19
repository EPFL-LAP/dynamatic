//===- FuncMaximizeSSA.cpp - Maximal SSA form within functions --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the SSA maximization pass as well as utilities
// for converting a function with standard control flow into maximal SSA form.
//
// This if largely inherited from CIRCT, with minor modifications.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeMarkFPUImpl.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace dynamatic {

// include tblgen base class definition
#define GEN_PASS_DEF_HANDSHAKEMARKFPUIMPL
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace dynamatic

namespace {

struct HandshakeMarkFPUImplPass
    : public dynamatic::impl::HandshakeMarkFPUImplBase<
          HandshakeMarkFPUImplPass> {
public:
  // use tblgen constructors from base class
  using HandshakeMarkFPUImplBase::HandshakeMarkFPUImplBase;

  // inherited TableGen Pass Options:
  // std::string impl

  void runDynamaticPass() override;
};

} // namespace

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

void HandshakeMarkFPUImplPass::runDynamaticPass() {
  // this->impl is the std::string pass option declared in tablegen
  auto implOpt = symbolizeFPUImplOrEmitError(this->impl);
  if (!implOpt.has_value()) {
    signalPassFailure();
    return;
  }

  // FPUImpl local variable hides std::string pass option
  FPUImpl impl = implOpt.value();

  // iterate over each operation that implements FPUImplInterface
  getOperation()->walk([&](FPUImplInterface fpuImplInterface) {
    // and set it based on the pass option
    fpuImplInterface.setFPUImpl(impl);
  });
}
