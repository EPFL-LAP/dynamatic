//===- Handshake.td - Handshake dialect definition ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TD
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Handshake_Dialect : Dialect {
  let name = "handshake";
  let cppNamespace = "::dynamatic::handshake";

  let summary = "Types and operations for the handshake dialect";
  let description = [{
    This dialect defined the `handshake` dialect, modeling dataflow circuits.
    Handshake/dataflow IR is describes independent, unsynchronized processes
    communicating data through First-in First-out (FIFO) communication channels.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 0;
}

include "dynamatic/Dialect/Handshake/HandshakeAttributes.td"
include "dynamatic/Dialect/Handshake/HandshakeTypeInterfaces.td"
include "dynamatic/Dialect/Handshake/HandshakeTypes.td"
include "dynamatic/Dialect/Handshake/HandshakeInterfaces.td"
include "dynamatic/Dialect/Handshake/HandshakeOps.td"
include "dynamatic/Dialect/Handshake/HandshakeArithOps.td"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TD
