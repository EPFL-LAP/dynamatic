//===- CfToHandshake.cpp - Convert func/cf to handshake dialect -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --lower-cf-to-handshake pass. See
// lib/Conversion/Utils/CfToHandshakeTranslator.h for the translation.
//
//===----------------------------------------------------------------------===//

#include "../Utils/CfToHandshakeTranslator.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Support/DynamaticPass.h"

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Conversion/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_CFTOHANDSHAKE
#include "dynamatic/Conversion/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace dynamatic;

namespace {

/// FPGA18's elastic pass. Translates every function (func::FuncOp) of the
/// module it is applied on into a Handshake function.
struct CfToHandshakePass
    : public dynamatic::impl::CfToHandshakeBase<CfToHandshakePass> {

  void runOnOperation() override {
    // Make sure all operation names are unique and haven't changed from what is
    // cached (if anything is cached)
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    if (!namer.isAnalysisValid())
      return signalPassFailure();

    if (failed(lowerCfToHandshake(getOperation(), namer)))
      return signalPassFailure();

    // Make sure all operation names are still unique and name operations that
    // do not currently have a name
    if (failed(namer.walk(NameAnalysis::UnnamedBehavior::NAME)))
      return signalPassFailure();
    markAnalysesPreserved<NameAnalysis>();
  }
};
} // namespace
