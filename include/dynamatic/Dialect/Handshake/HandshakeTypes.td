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
include "dynamatic/Dialect/Handshake/HandshakeTypeInterfaces.td"

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
  let comparator = cppType # "($_lhs) == " # cppType # "($_rhs)";
  let defaultValue = cppType # "()";
}

def ControlType : Handshake_Type<"Control", "control", [
  DeclareTypeInterfaceMethods<ExtraSignalsTypeInterface, ["addExtraSignal"]>
]> {
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
    // If no parameters provided, build SimpleControl
    TypeBuilder<(ins),
      [{
        return ControlType::get($_ctxt, {});
      }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def ChannelType : Handshake_Type<"Channel", "channel", [
  DeclareTypeInterfaceMethods<ExtraSignalsTypeInterface, ["addExtraSignal"]>
]> {
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
    /// Returns the data type's bitwidth.
    unsigned getDataBitWidth() const;

    /// Returns a channel type with identical extra signals but a potentially
    /// different data type.
    ChannelType withDataType(Type newDataType) {
      return ChannelType::get(newDataType, getExtraSignals());
    }

    /// Returns a channel type with identical data type but a potentially
    /// different extra signals.
    ChannelType withExtraSignal(::mlir::ArrayRef<ExtraSignal> newExtraSignals) {
      return ChannelType::get(getDataType(), newExtraSignals);
    }

    /// Determines whether a type is supported as the data type or as the type
    /// of an extra signal.
    static bool isSupportedSignalType(::mlir::Type type) {
      return type.isIntOrFloat();
    }

    /// Returns a channel whose data type is the default one used to represent
    /// an address in an external memory.
    static ChannelType getAddrChannel(::mlir::MLIRContext* ctx);
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

class HasNumExtras<int numExtras> : CPred<
  "::mlir::cast<::dynamatic::handshake::ExtraSignalsTypeInterface>" #
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
  HasNumExtras<0>,
  "must be a `handshake::ControlType` type with no extra signals",
  "::dynamatic::handshake::ControlType"
>;

def SimpleChannel : TypeConstraint<
  HasNumExtras<0>,
  "must be a `handshake::ChannelType` type with no extra signals",
  "::dynamatic::handshake::ChannelType"
>;

// Temporary constraint for UnbundleOp.
def ChannelOrSimpleControl : TypeConstraint<
  CPred<[{
    ::mlir::isa<::dynamatic::handshake::ChannelType>($_self) ||
    (::mlir::isa<::dynamatic::handshake::ControlType>($_self) &&
     ::mlir::cast<::dynamatic::handshake::ControlType>($_self).getNumExtraSignals() == 0)
  }]>,
  "must be a `handshake::ControlType` with no extra signals or `handshake::ChannelType`"
>;

// Multi-entity constraint ensuring that all ChannelTypes have matching data types.
// Also applies to ControlTypes, as some operations treat them interchangeably
// using HandshakeType.
// In the ControlType case, the data type is regarded as a ControlType with no extra signals here.
class AllDataTypesMatch<list<string> names> : AllMatchSameOperatorTrait<
  names,
  [{
    ::mlir::isa<::dynamatic::handshake::ControlType>($_self.getType()) ?
    // ControlType without extra signals
    ::dynamatic::handshake::ControlType::get($_self.getContext(), {}) :
    ::mlir::cast<::dynamatic::handshake::ChannelType>($_self.getType()).getDataType()
  }],
  "data type"
>;

// A constraint ensuring that all types of a variadic operand are the same.
class AllTypesMatchWithinVariadic<string name> : PredOpTrait<
  "The operands inside the variadic " # name # " should have the same type",
  CPred<[{
    ([&](const mlir::OperandRange &operands) {
      // Operands are those specified as the variadic input.
      ValueTypeRange<OperandRange> types = operands.getTypes();
      size_t numTypes = types.size();

      // If there are less than 2 types, always return true
      if (numTypes < 2)
        return true;

      Type firstType = types.front();
      for (size_t i = 1; i < numTypes; i++) {
        if (types[i] != firstType)
          return false;
      }
      return true;
    })
  }] # "($" # name # ")">
>;

// Multi-entity constraint ensuring that all variadic/non-variadic operands have the same types.
// We assume that there is at least one element in nonvariadics.
// Otherwise, please use AllTypesMatchWithinVariadic.
class AllTypesMatchWithVariadics<list<string> variadics, list<string> nonvariadics> : PredOpTrait<
  "All variadic/non-variadic operands should have the same type",
  And<[
    // Non-variadic operands
    AllTypesMatch<nonvariadics>.predicate] #
    // Intra variadic operands
    !foreach(variadic, variadics, AllTypesMatchWithinVariadic<variadic>.predicate) #
    // Inter variadic operands (compare the head of each variadics and the head of non-variadics)
    // We assumed there is at least one element in nonvariadic
    !foreach(variadic, variadics, CPred<[{
      ([&](const mlir::OperandRange &operands, const mlir::OpResult &op) {
        auto types = operands.getTypes();
        if (types.empty())
          return true;
        return types.front() == op.getType();
      })
    }] # "($" # variadic # ", $" # !head(nonvariadics) # ")">)>
>;

// A constraint ensuring that all types of a variadic operand have the same data type.
// Also applies to ControlTypes, as some operations treat them interchangeably.
class AllDataTypesMatchWithinVariadic<string name> : PredOpTrait<
  "The operands inside the variadic " # name # " should have the same type",
  CPred<[{
    ([&](const mlir::OperandRange &operands) {
      // Operands are those specified as the variadic input.
      ValueTypeRange<OperandRange> types = operands.getTypes();
      size_t numTypes = types.size();

      // If there are no items, always return true
      if (numTypes < 1)
        return true;

      if (mlir::isa<ControlType>(types.front())) {
        // Just confirm that all types are ControlType
        for (size_t i = 1; i < numTypes; i++) {
          if (!mlir::isa<ControlType>(types[i]))
            return false;
        }
        return true;
      }

      // Confirm that dataTypes of all types are the same
      Type firstType = mlir::cast<ChannelType>(types.front()).getDataType();
      for (size_t i = 1; i < numTypes; i++) {
        if (mlir::cast<ChannelType>(types[i]).getDataType() != firstType)
          return false;
      }
      return true;
    })
  }] # "($" # name # ")">
>;

// Multi-entity constraint ensuring that all variadic/non-variadic operands have the same data types.
// Also applies to ControlTypes, as some operations treat them interchangeably.
// We assume that there is at least one element in nonvariadics.
// Otherwise, please use AllTypesMatchWithinVariadic.
class AllDataTypesMatchWithVariadics<list<string> variadics, list<string> nonvariadics> : PredOpTrait<
  "All variadic/non-variadic operands should have the same data types",
  And<[
    // Non-variadic operands
    AllDataTypesMatch<nonvariadics>.predicate] #
    // Intra variadic operands
    !foreach(variadic, variadics, AllDataTypesMatchWithinVariadic<variadic>.predicate) #
    // Inter variadic operands (compare the head of each variadics and the head of non-variadics)
    // We assumed there is at least one element in nonvariadic
    !foreach(variadic, variadics, CPred<[{
      ([&](const mlir::OperandRange &operands, const mlir::OpResult &op) {
        auto types = operands.getTypes();
        if (types.empty())
          return true;
        if (mlir::isa<ControlType>(types.front())) {
          // Just confirm that non-variadic type is also ControlType
          return mlir::isa<ControlType>(op.getType());
        }
        return mlir::cast<ChannelType>(types.front()).getDataType() == mlir::cast<ChannelType>(op.getType()).getDataType();
      })
    }] # "($" # variadic # ", $" # !head(nonvariadics) # ")">)>
>;

// Multi-entity constraints for spec tags

// Predicate to verify if the given type has a valid spec tag as an extra signal.
class HasSpecTag<string type> : CPred<
  [{
    ([&](std::optional<ExtraSignal> getExtraSignalResult) {
      if (!getExtraSignalResult.has_value()) {
        // There is no extra signal called "spec"
        return false;
      }

      ExtraSignal signal = getExtraSignalResult.value();

      // The signal should be a 1-bit integer downstream signal.
      if (!signal.downstream)
        return false;
      if (!signal.type.isInteger(1))
        return false;

      return true;
    })(::mlir::cast<::dynamatic::handshake::ExtraSignalsTypeInterface>(
  }] # type # ").getExtraSignal(\"spec\"))"
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

// Multi-entity constraint ensuring that if at least one of the `input` has a spec tag,
// then all of the `output` should have a spec tag.
// This verification is also done in the verifier of MergeLikeOpInterface
class PropagatingSpecTag<list<string> inputs, list<string> outputs> : PredOpTrait<
  "if at least one of the inputs has a spec tag, then all of the outputs should have a spec tag",
  Or<[
    And<[
      Or<!foreach(
        input, // variable name
        inputs,
        HasSpecTag<"$" # input # ".getType()">
      )>,
      And<!foreach(
        output, // variable name
        outputs,
        HasSpecTag<"$" # output # ".getType()">
      )>
    ]>,
    And<[
      And<!foreach(
        input, // variable name
        inputs,
        Neg<HasSpecTag<"$" # input # ".getType()">>
      )>,
      And<!foreach(
        output, // variable name
        outputs,
        Neg<HasSpecTag<"$" # output # ".getType()">>
      )>
    ]>
  ]>
>;

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPES_TD
