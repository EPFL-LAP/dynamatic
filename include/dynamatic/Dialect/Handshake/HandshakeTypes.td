//===- HandshakeTypes.td - Handshake types definition ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Handshake typesd in Tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPES_TD
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPES_TD

/// Base class for types in the Handshake dialect.
class Handshake_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Handshake_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ControlType : Handshake_Type<"Control", "control"> {
  let summary = "A control-only dataflow channel.";
  let description = [{
    Represents a control-only dataflow channel, which is made up of
    - a 1-bit valid signal going downstream (in the same direction as the
      natural SSA def-use relation's direction) and
    - a 1-bit ready signal going upsteam (in the opposite direction as the
      natural SSA def-use relation's direction).
  }];

  let parameters = (ins);
  let assemblyFormat = "";
}

def ChannelType : Handshake_Type<"Channel", "channel"> {
  let summary = "A dataflow channel.";
  let description = [{
    Represents a dataflow channel, which is made up of
    - a data signal of arbitrary width and type going downstream (in the same
      direction as the natural SSA def-use relation's direction),
    - a 1-bit valid signal going downstream (in the same direction as the
      natural SSA def-use relation's direction), and
    - a 1-bit ready signal going upsteam (in the opposite direction as the
      natural SSA def-use relation's direction).
  }];

  let parameters = (ins "::mlir::Type":$dataType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$dataType), [{
        return Base::get(dataType.getContext(), dataType);
      }]>
   ];

  let assemblyFormat = "`<` $dataType `>`";
}

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPES_TD
