//===- Passes.h - XLS conversion passes registration ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the registration code for all XLS conversion
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef XLS_CONVERSION_PASSES_H
#define XLS_CONVERSION_PASSES_H

#include "xls-interop/Conversion/HandshakeToXls.h"

namespace dynamatic {
namespace xls {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "xls-interop/Conversion/Passes.h.inc"

} // namespace xls
} // namespace dynamatic

#endif // XLS_CONVERSION_PASSES_H
