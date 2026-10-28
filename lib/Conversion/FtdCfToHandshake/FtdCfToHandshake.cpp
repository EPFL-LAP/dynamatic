//===- FtdCfToHandshake.cpp - FTD conversion cf -> handshake --*--- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --ftd-lower-cf-to-handshake pass. See
// lib/Conversion/Utils/FtdCfToHandshakeTranslator.h for the translation.
//
//===----------------------------------------------------------------------===//

#include "../Utils/FtdCfToHandshakeTranslator.h"
#include "dynamatic/Analysis/GSAAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Support/DynamaticPass.h"

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Conversion/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_FTDCFTOHANDSHAKE
#include "dynamatic/Conversion/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace dynamatic;

namespace {

struct FtdCfToHandshakePass
    : public dynamatic::impl::FtdCfToHandshakeBase<FtdCfToHandshakePass> {

  void runOnOperation() override {
    // Make sure all operation names are unique and haven't changed from what is
    // cached (if anything is cached)
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    if (!namer.isAnalysisValid())
      return signalPassFailure();

    if (failed(ftdLowerCfToHandshake(getOperation(), namer,
                                     getAnalysis<gsa::GSAAnalysis>())))
      return signalPassFailure();

    // Make sure all operation names are still unique and name operations that
    // do not currently have a name
    if (failed(namer.walk(NameAnalysis::UnnamedBehavior::NAME)))
      return signalPassFailure();
    markAnalysesPreserved<NameAnalysis>();
  }
};
} // namespace
