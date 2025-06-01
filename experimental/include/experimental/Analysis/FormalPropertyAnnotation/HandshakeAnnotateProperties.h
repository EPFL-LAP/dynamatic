//===- HandshakeAnnotateProperties.h - Property annotation ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --handshake-annotate-properties pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_ANALYSIS_ANNOTATE_PROPERTIES_PASS_H
#define DYNAMATIC_ANALYSIS_ANNOTATE_PROPERTIES_PASS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include <string>

namespace dynamatic {
namespace experimental {
namespace formalprop {

std::unique_ptr<dynamatic::DynamaticPass>
createAnnotateProperties(const std::string &jsonPath = "");

#define GEN_PASS_DECL_HANDSHAKEANNOTATEPROPERTIES
#define GEN_PASS_DEF_HANDSHAKEANNOTATEPROPERTIES
#include "experimental/Analysis/Passes.h.inc"

} // namespace formalprop
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_ANALYSIS_ANNOTATE_PROPERTIES_PASS_H