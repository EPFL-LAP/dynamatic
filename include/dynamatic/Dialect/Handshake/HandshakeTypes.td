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

def ControlType : Handshake_Type<"Control", "control"> {
  let summary = "a control-only dataflow channel";
  let description = [{
    Represents a control-only dataflow channel, which is made up of
    - a 1-bit valid signal going downstream (in the same direction as the
      natural SSA def-use relation's direction) and
    - a 1-bit ready signal going upsteam (in the opposite direction as the
      natural SSA def-use relation's direction).
    - an optional list of named extra signals of arbitrary width and type which
      may go downstream or upstream.
  }];

  let parameters = (ins ExtraSignals:$extraSignals);
  let builders = [
    TypeBuilder<(ins),
      [{
        return ControlType::get($_ctxt, {});
      }]>
  ];

  let hasCustomAssemblyFormat = 1;

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

    mlir::FailureOr<ExtraSignal> getExtraSignalByName(std::string name);
  }];
}

def ChannelType : Handshake_Type<"Channel", "channel"> {
  let summary = "a dataflow channel with optional extra signals";
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
    TypeBuilderWithInferredContext<(ins
      "::mlir::Type":$dataType,
      CArg<"::mlir::ArrayRef<::dynamatic::handshake::ExtraSignal>", "{}">:$extras),
      [{
        if (::mlir::isa<::mlir::IndexType>(dataType)) {
          // Convert IndexType's to equivalent signless integers automatically
          ::mlir::OpBuilder builder(dataType.getContext());
          dataType = builder.getIntegerType(32);
        }
        return ChannelType::get(dataType.getContext(), dataType, extras);
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

    /// Returns a channel type with identical extra signals but a potentially
    /// different data type.
    ChannelType withDataType(Type newDataType) {
      return ChannelType::get(newDataType, getExtraSignals());
    }

    /// Determines whether a type is supported as the data type or as the type
    /// of an extra signal.
    static bool isSupportedSignalType(::mlir::Type type) {
      return type.isIntOrFloat();
    }

    /// Returns a channel whose data type is the default one used to represent
    /// an address in an external memory.
    static ChannelType getAddrChannel(::mlir::MLIRContext* ctx);

    mlir::FailureOr<ExtraSignal> getExtraSignalByName(std::string name);
  }];
}

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

def IsChannel : CPred<"::mlir::isa<::dynamatic::handshake::ChannelType>($_self)">;

class ChannelHasDataType<string type> : CPred<
  "::mlir::isa<" # type #
  ">(::mlir::cast<::dynamatic::handshake::ChannelType>($_self).getDataType())"
>;

class ChannelHasDataWidth<int width> : CPred<
  "::mlir::cast<::dynamatic::handshake::ChannelType>" #
  "($_self).getDataBitWidth() == " # width
>;

class ControlHasNumExtras<int numExtras> : CPred<
  "::mlir::cast<::dynamatic::handshake::ControlType>" #
  "($_self).getNumExtraSignals() == " # numExtras
>;

class ChannelHasNumExtras<int numExtras> : CPred<
  "::mlir::cast<::dynamatic::handshake::ChannelType>" #
  "($_self).getNumExtraSignals() == " # numExtras
>;

def SignalType : Type<
  CPred<"::dynamatic::handshake::ChannelType::isSupportedSignalType($_self)">,
  "must be an `IntegerType` or `FloatType`"> {
}

def HandshakeType : Type<
  CPred<
    "::mlir::isa<::dynamatic::handshake::ControlType, " #
    "::dynamatic::handshake::ChannelType>($_self)"
  >,
  "must be a `handshake::ControlType` or `handshake::ChannelType` type"> {
}

class TypedChannel<string type> : TypeConstraint<
  ChannelHasDataType<type>,
  "must be a `handshake::ChannelType` type whose data is " # type,
  "::dynamatic::handshake::ChannelType"
>;

def IntChannelType : TypedChannel<"::mlir::IntegerType">;
def FloatChannelType : TypedChannel<"::mlir::FloatType">;

class TypedSizedChannel<string type, int width> : TypeConstraint<
  And<[ChannelHasDataType<type>, ChannelHasDataWidth<width>]>,
  "must be a `handshake::ChannelType` type whose data is " # type #
  " and whose bitwidth is " # width,
  "::dynamatic::handshake::ChannelType"
>;

class IntSizedChannel<int width> : TypedSizedChannel<
  "::mlir::IntegerType", width
>;

class FloatSizedChannel<int width> : TypedSizedChannel<
  "::mlir::FloatType", width
>;

def BoolChannel : IntSizedChannel<1>;

def SimpleControl : TypeConstraint<
  ControlHasNumExtras<0>,
  "must be a `handshake::ControlType` type with no extra signals",
  "::dynamatic::handshake::ControlType"
>;

def SimpleChannel : TypeConstraint<
  ChannelHasNumExtras<0>,
  "must be a `handshake::ChannelType` type with no extra signals",
  "::dynamatic::handshake::ChannelType"
>;

// Multi-entity constraint ensuring that all ChannelTypes have matching data types.
// Also applies to ControlTypes, as some operations treat them interchangeably 
// using HandshakeType.
class AllDataTypesMatch<list<string> names> : AllMatchSameOperatorTrait<
  names,
  [{
    ::mlir::isa<::dynamatic::handshake::ControlType>($_self.getType()) ?
    ::dynamatic::handshake::ControlType::get($_self.getContext(), {}) :
    ::mlir::cast<::dynamatic::handshake::ChannelType>($_self.getType()).getDataType()
  }],
  "data type"
>;

// Multi-entity constraints for spec tags

// Predicate to verify if the given type has a valid spec tag as an additional signal.
class HasSpecTag<string type> : CPred<
  [{
    ([&](FailureOr<ExtraSignal> signal) {
      if (failed(signal)) {
        // There is no extra signal called "spec"
        return false;
      }
      if (!(*signal).downstream) {
        // The signal is not a downstream signal
        return false;
      }
      if (!(*signal).type.isInteger(1)) {
        // The signal is not a 1-bit integer
        return false;
      }
      return true;
    })(
  }] #
  "    ::mlir::isa<::dynamatic::handshake::ChannelType>(" # type # ") ?\n" #
  "    ::mlir::cast<::dynamatic::handshake::ChannelType>(" # type # ").getExtraSignalByName(\"spec\") :\n" #
  "    ::mlir::cast<::dynamatic::handshake::ControlType>(" # type # ").getExtraSignalByName(\"spec\")\n" #
  "  )"
>;

// Multi-entity constraint ensuring that the type of `b` matches the type of `a` with a spec tag.
class AddingSpecTag<string a, string b> : PredOpTrait<
  b # " should be the type of " # a # " with spec tag",
  And<[
    AllDataTypesMatch<[a, b]>.predicate,
    Neg<HasSpecTag<"$" # a # ".getType()">>,
    HasSpecTag<"$" # b # ".getType()">
    // TODO: Add a constraint to ensure that no other extra bits are added to `b`.
  ]>
>;

// Multi-entity constraint ensuring that the type of `b` matches the type of `a` without a spec tag.
class RemovingSpecTag<string a, string b> : AddingSpecTag<b, a>;

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPES_TD
