//===- HandshakeToxls.h - Lower handshake ops to XLS procs ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --lower-handshake-to-xls pass.
//
//===----------------------------------------------------------------------===//

#ifndef XLS_INTEROP_TRANSFORMS_HANDSHAKETOXLS_H
#define XLS_INTEROP_TRANSFORMS_HANDSHAKETOXLS_H

#include "dynamatic/Support/DynamaticPass.h"

#ifndef DYNAMATIC_ENABLE_XLS
#error "This file may not be included without XLS integration being enabled."
#endif // DYNAMATIC_ENABLE_XLS

/// Forward declare the XLS dialect which the pass depends on.
namespace mlir {
namespace xls {
class XlsDialect;
} // namespace xls
} // namespace mlir

namespace dynamatic {
namespace xls {

#define GEN_PASS_DECL_HANDSHAKETOXLS
#define GEN_PASS_DEF_HANDSHAKETOXLS
#include "xls-interop/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createHandshakeToXlsPass();

} // namespace xls
} // namespace dynamatic

#endif // XLS_INTEROP_TRANSFORMS_HANDSHAKETOXLS_H
