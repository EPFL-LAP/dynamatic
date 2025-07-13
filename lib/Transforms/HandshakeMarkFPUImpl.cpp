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

#define GEN_PASS_DEF_HANDSHAKEMARKFPUIMPL
#include "dynamatic/Transforms/Passes.h.inc"

} // namespace dynamatic


namespace {

struct HandshakeMarkFPUImplPass
    : public dynamatic::impl::HandshakeMarkFPUImplBase<HandshakeMarkFPUImplPass> {
public:
    using HandshakeMarkFPUImplBase::HandshakeMarkFPUImplBase;

    // inherited TableGen Pass Options:
    // FPUImpl imp

    void runDynamaticPass() override;
};

} // namespace

void HandshakeMarkFPUImplPass::runDynamaticPass() {
  auto implOpt = symbolizeFPUImpl(this->impl);

  if(!implOpt){
    llvm::errs()
      << "Invalid FPU implementation: expected one of FLOPOCO or VIVADO, but got '"
      this->impl << "'";
    signalPassFailure();
    return;
  }

  auto impl = *implOpt;

  getOperation()->walk([&](FPUImplInterface fpuImplInterface) {
    fpuImplInterface.setFPUImpl(impl);
  });
}
