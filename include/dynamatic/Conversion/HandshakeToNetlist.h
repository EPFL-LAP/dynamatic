//===- HandshakeToNetlist.h - Converts handshake to HW ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --lower-handshake-to-netlist conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_CONVERSION_HANDSHAKETONETLIST_H
#define DYNAMATIC_CONVERSION_HANDSHAKETONETLIST_H

#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/DialectRegistry.h"

namespace dynamatic {

#define GEN_PASS_DECL_HANDSHAKETONETLIST
#define GEN_PASS_DEF_HANDSHAKETONETLIST
#include "dynamatic/Conversion/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeToNetlistPass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_HANDSHAKETONETLIST_H
