//===- HandshakeToHW.h - Convert Handshake to HW ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --lower-handshake-to-hw conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_HANDSHAKETOHW_H
#define DYNAMATIC_CONVERSION_HANDSHAKETOHW_H

#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/DialectRegistry.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKETOHW
#define GEN_PASS_DEF_HANDSHAKETOHW
#include "dynamatic/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeToHWPass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_HANDSHAKETOHW_H
