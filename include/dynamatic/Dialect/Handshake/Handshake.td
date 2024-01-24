//===- Handshake.td - Handshake dialect definition ---------*- tablegen -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
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
include "mlir/IR/PatternBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "dynamatic/Dialect/Handshake/HandshakeInterfaces.td"

def Handshake_Dialect : Dialect {
  let name = "handshake";
  let cppNamespace = "::dynamatic::handshake";

  let summary = "Types and operations for the handshake dialect";
  let description = [{
    This dialect defined the `handshake` dialect, modeling dataflow circuits.
    Handshake/dataflow IR is describes independent, unsynchronized processes
    communicating data through First-in First-out (FIFO) communication channels.
  }];

  let useDefaultAttributePrinterParser = 1;

  // Opt-out of properties for now, must migrate by LLVM 19.  #5273.
  let usePropertiesForAttributes = 0;
}

// Base class for Handshake dialect opserations.
class Handshake_Op<string mnemonic, list<Trait> traits = []>
    : Op<Handshake_Dialect, mnemonic,
         traits #[HasParent<"handshake::FuncOp">,
         DeclareOpInterfaceMethods<NamedIOInterface>,
         DeclareOpInterfaceMethods<ControlInterface>]> {
}


include "dynamatic/Dialect/Handshake/HandshakeOps.td"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TD
