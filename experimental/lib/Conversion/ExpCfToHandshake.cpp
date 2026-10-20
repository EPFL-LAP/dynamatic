//===- ExpCfToHandshake.cpp - Exp. lowering of func/cf to handshake -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --exp-lower-cf-to-handshake pass. See
// Utils/CfToHandshakeTranslator.h for the translation.
//
//===----------------------------------------------------------------------===//

#include "Utils/CfToHandshakeTranslator.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"

// [START Boilerplate code for the MLIR pass]
#include "experimental/Conversion/Passes.h" // IWYU pragma: keep
namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_EXPCFTOHANDSHAKE
#include "experimental/Conversion/Passes.h.inc"
} // namespace experimental
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace dynamatic;

namespace {

struct ExpCfToHandshakePass
    : public dynamatic::experimental::impl::ExpCfToHandshakeBase<
          ExpCfToHandshakePass> {

  void runOnOperation() override {
    // Make sure all operation names are unique and haven't changed from what is
    // cached (if anything is cached)
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    if (!namer.isAnalysisValid())
      return signalPassFailure();

    if (failed(experimental::lowerCfToHandshake(getOperation(), namer)))
      return signalPassFailure();

    // Make sure all operation names are still unique and name operations that
    // do not currently have a name
    if (failed(namer.walk(NameAnalysis::UnnamedBehavior::NAME)))
      return signalPassFailure();
    markAnalysesPreserved<NameAnalysis>();
  }
};
} // namespace
