//===- HandshakeToNetlist.h - Converts handshake to HW/ESI ------*- C++ -*-===//
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

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::hw;
using namespace circt::esi;

namespace dynamatic {

std::unique_ptr<dynamatic::DynamaticPass> createHandshakeToNetlistPass();

} // namespace dynamatic

#endif // DYNAMATIC_CONVERSION_HANDSHAKETONETLIST_H
