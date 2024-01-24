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
include "mlir/IR/AttrTypeBase.td"
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

/// Base class for Handshake dialect operations.
class Handshake_Op<string mnemonic, list<Trait> traits = []>
    : Op<Handshake_Dialect, mnemonic,
         traits #[HasParent<"handshake::FuncOp">,
         DeclareOpInterfaceMethods<NamedIOInterface>,
         DeclareOpInterfaceMethods<ControlInterface>]> {
}

/// Base class for Handshake dialect types.
class Handshake_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Handshake_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ChannelType : Handshake_Type<"Channel", "channel"> {
  let summary = "A dataflow channel with optional data.";
  let description = [{
    Represents a dataflow channel, which is made up of a 1-bit valid signal
    going downstream, a 1-bit ready signal going upsteam, and an arbitrary and
    optional data signal going downstream. 
  }];

  /// A channel is parameterized by the type of the data signal it carries. For
  /// control-only channels, this would be i0.  
  let parameters = (ins "::mlir::Type":$dataType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$type), [{
        return Base::get(type.getContext(), type);
      }]>
  ];

  let assemblyFormat = "`<` $dataType `>`";
}

include "dynamatic/Dialect/Handshake/HandshakeOps.td"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TD
