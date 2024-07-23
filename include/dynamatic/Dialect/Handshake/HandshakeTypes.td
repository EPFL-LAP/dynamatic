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

include "mlir/IR/AttrTypeBase.td"
include "dynamatic/Dialect/Handshake/Handshake.td"

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
}

def ExtraSignals : TypeParameter<
  "::llvm::ArrayRef<::dynamatic::handshake::ExtraSignal>", 
  "An optional array of extra signals for a dataflow channel"> {
  let allocator = [{
    ::llvm::SmallVector<::dynamatic::handshake::ExtraSignal> tmpSignals;
    for (const ::dynamatic::handshake::ExtraSignal &signal : $_self) {
      ::dynamatic::handshake::ExtraSignal& tmp = tmpSignals.emplace_back(signal);
      tmp.name = $_allocator.copyInto(tmp.name);
    }
    }] # "$_dst = $_allocator.copyInto(" # cppType # [{ (tmpSignals));
  }];
  let cppStorageType = "::llvm::SmallVector<::dynamatic::handshake::ExtraSignal::Storage>";
  let convertFromStorage = [{convertExtraSignalsFromStorage($_self)}];
  let comparator = cppType # "($_lhs) == " # cppType # "($_rhs)";
  let defaultValue = cppType # "()";
}

def ChannelType : Handshake_Type<"Channel", "channel"> {
  let summary = "A dataflow channel with optional extra signals.";
  let description = [{
    Represents a dataflow channel, which is made up of
    - a data signal of arbitrary width and type going downstream (in the same
      direction as the natural SSA def-use relation's direction),
    - a 1-bit valid signal going downstream (in the same direction as the
      natural SSA def-use relation's direction),
    - a 1-bit ready signal going upsteam (in the opposite direction as the
      natural SSA def-use relation's direction), and
    - an optional list of named extra signals of arbitrary width and type which
      may go downstream or upstream.
  }];

  let parameters = (ins "::mlir::Type":$dataType, ExtraSignals:$extraSignals);

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$dataType), [{
        return ChannelType::get(dataType.getContext(), dataType, {});
      }]>
   ];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns the number of extra signals.
    unsigned getNumExtraSignals() const {
      return getExtraSignals().size();
    }

    /// Returns the number of downstream extra signals.
    unsigned getNumDownstreamExtraSignals() const;

    /// Returns the number of upstream extra signals.
    unsigned getNumUpstreamExtraSignals() const {
      return getNumExtraSignals() - getNumDownstreamExtraSignals();
    }

    /// Returns the data type's bitwidth.
    unsigned getDataBitWidth() const;
    
    /// Determines whether a type is supported as the data type or as the type
    /// of an extra signal.
    static bool isSupportedSignalType(::mlir::Type type) {
      return type.isIntOrIndexOrFloat();
    }
  }];
}

def SignalType : Type<
  CPred<"::dynamatic::handshake::ChannelType::isSupportedSignalType($_self)">,
  "must be an `IndexType`, `IntegerType` or `FloatType`"> {
}

def ChannelLikeType : Type<
  CPred<"::mlir::isa<::dynamatic::handshake::ControlType, ::dynamatic::handshake::ChannelType>($_self)">,
  "must be a `handshake::ControlType` or `handshake::ChannelType` type"> {
}

def IsChannel : CPred<"::mlir::isa<::dynamatic::handshake::ChannelType>($_self)">;

def BoolChannelType : Type<
  And<[
    IsChannel,
    CPred<"::mlir::cast<::dynamatic::handshake::ChannelType>($_self).getDataBitWidth() == 1">,
  ]>,
  "must be a `handshake::ChannelType` type whose data type's bitwidth is 1"> {
}

def IntChannelType : Type<
  And<[
    IsChannel,
    CPred<"::mlir::isa<IndexType, IntegerType>(::mlir::cast<::dynamatic::handshake::ChannelType>($_self).getDataType())">,
  ]>,
  "must be a `handshake::ChannelType` type whose data type is integer-like"> {
}

def FloatChannelType : Type<
  And<[
    IsChannel,
    CPred<"::mlir::isa<FloatType>(::mlir::cast<::dynamatic::handshake::ChannelType>($_self).getDataType())">,
  ]>,
  "must be a `handshake::ChannelType` type whose data type is float-like"> {
}

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPES_TD
