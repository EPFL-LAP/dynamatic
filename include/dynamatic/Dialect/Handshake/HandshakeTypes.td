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
  let cppStorageType = "::llvm::SmallVector<::dynamatic::handshake::ExtraSignal::Storage>";
  let comparator = cppType # "($_lhs) == " # cppType # "($_rhs)";
  let defaultValue = cppType # "()";
}

def ControlType : Handshake_Type<"Control", "control", [
  DeclareTypeInterfaceMethods<ExtraSignalsTypeInterface, ["addExtraSignal", "replaceExtraSignals"]>
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
  DeclareTypeInterfaceMethods<ExtraSignalsTypeInterface, ["addExtraSignal", "replaceExtraSignals"]>
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

def SimpleControl : Type<
  HasNumExtras<0>,
  "must be a `handshake::ControlType` type with no extra signals",
  "::dynamatic::handshake::ControlType"
>;

def SimpleChannel : Type<
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

/// Multi-entity constraint ensuring that all ChannelTypes have matching data types.
/// Also applies to ControlTypes, as some operations treat them interchangeably
/// using HandshakeType.
/// For the ControlType case, this predicate is satisfied only when all types are ControlType.
class AllDataTypesMatch<list<string> names> : AllMatchSameOperatorTrait<
  names,
  [{
    ::mlir::isa<::dynamatic::handshake::ControlType>($_self.getType()) ?
    // Compare using ControlType without extra signals
    ::dynamatic::handshake::ControlType::get($_self.getContext(), {}) :
    // Compare using data type
    ::mlir::cast<::dynamatic::handshake::ChannelType>($_self.getType()).getDataType()
  }],
  "data type"
>;

/// Multi-entity constraint ensuring that all HandshakeTypes have matching extra signals.
class AllExtraSignalsMatch<list<string> names> : AllMatchSameOperatorTrait<
  names,
  "::mlir::cast<::dynamatic::handshake::ExtraSignalsTypeInterface>($_self.getType()).getExtraSignals()",
  "extra signals"
>;

/// A constraint ensuring that all operands of the variadic have the same data type.
/// Also applies to ControlTypes, as some operations treat them interchangeably.
class AllDataTypesMatchWithinVariadic<string name> : PredOpTrait<
  "the operands inside the variadic " # name # " should have the same type",
  CPred<[{
    ([&](const mlir::OperandRange &operands) {
      // Retrive all types of operands in the variadic.
      ValueTypeRange<OperandRange> types = operands.getTypes();
      size_t numTypes = types.size();

      if (numTypes <= 1)
        return true;

      if (mlir::isa<ControlType>(types.front())) {
        // If the front operand is of ControlType, it is enough to
        // ensure that all other operands are also of ControlType
        for (size_t i = 1; i < numTypes; i++) {
          if (!mlir::isa<ControlType>(types[i]))
            return false;
        }
        return true;
      }

      // Here the front operand is of ChannelType.

      // Ensure that data types of all operands are the same
      Type frontDataType = mlir::cast<ChannelType>(types.front()).getDataType();
      for (size_t i = 1; i < numTypes; i++) {
        if (mlir::cast<ChannelType>(types[i]).getDataType() != frontDataType)
          return false;
      }
      return true;
    })
  }] # "($" # name # ")">
>;

/// Multi-entity constraint ensuring that the front operand of variadic and non-variadic
/// operand have the same data types.
/// Also applies to ControlTypes, as some operations treat them interchangeably.
class DataTypesMatchWithVariadicFront<string variadic, string nonvariadic> : PredOpTrait<
  "the front operand of variadic and non-variadic operand should have the same data type", CPred<[{
    ([&](const mlir::OperandRange &operands, const mlir::Value &op) {
      // Retrive all types of operands in the variadic.
      ValueTypeRange<OperandRange> types = operands.getTypes();

      // If variadic is empty (no front operand), returns true
      if (types.empty())
        return true;

      if (mlir::isa<ControlType>(types.front())) {
        // Pass if non-variadic type is also ControlType
        return mlir::isa<ControlType>(op.getType());
      }

      // Pass if non-variadic type has the same data type
      return mlir::cast<ChannelType>(types.front()).getDataType() == mlir::cast<ChannelType>(op.getType()).getDataType();
    })
  }] # "($" # variadic # ", $" # nonvariadic # ")">
>;

/// Multi-entity constraint ensuring that the all operands of variadic and non-variadic
/// operand have the same data types.
/// Also applies to ControlTypes, as some operations treat them interchangeably.
class AllDataTypesMatchWithVariadic<string variadic, string nonvariadic> : PredOpTrait<
  "the operands inside the variadic " # name # " and " # nonvariadic # " should all have the same type",
  And<[
    // Ensuring the data types of variadic[0], ..., variadic[n] is the same
    AllDataTypesMatchWithinVariadic<variadic>.predicate,
    // Ensuring the data types of variadic[0] and nonvariadic is the same
    DataTypesMatchWithVariadicFront<variadic, nonvariadic>.predicate
  ]>
>;

/// Constraint to ensure an input/output is of SimpleControl/Channel type.
class IsSimpleHandshake<string name> : PredOpTrait<
  name # " should be of SimpleControl or SimpleChannel type",
  CPred<"cast<::dynamatic::handshake::ExtraSignalsTypeInterface>($" # name #
    ".getType()).getNumExtraSignals() == 0">
>;

/// Merging Relationship of Extra Signals:
/// the output has the extra signal A if and only if
/// there exists an input which has the extra signal A.
/// In Merge-like Ops inputs are always provided as variadic.
class MergingExtraSignals<string inputs, string output> : PredOpTrait<
  "the extra signals of intraOperands and interOperands should satisfy \\\"Merging Relationship of Extra Signals\\\"",
  CPred<[{
    ([&](const mlir::OperandRange &inputs, Value output) {

      // Map to collect extra signals from inputs
      llvm::DenseMap<StringRef, const ExtraSignal *> unionOfExtraSignals;

      // Iterate over the type of each input operand
      for (auto inputOperandType : inputs.getTypes()) {
        for (const ExtraSignal &extraSignal :
            cast<ExtraSignalsTypeInterface>(inputOperandType).getExtraSignals()) {
          if (unionOfExtraSignals.contains(extraSignal.name)) {
            if (*unionOfExtraSignals.lookup(extraSignal.name) != extraSignal) {
              // Two different extra signals among inputs must have different names
              // to be merge-able.
              // e.g. [spec: i1] and [spec: i2] are prohibited.
              llvm::errs() << "inputs have different extra signals of the same name: "
                << extraSignal.name << "\n";
              return false;
            }
          } else {
            unionofExtraSignals.insert({extraSignal.name, &extraSignal});
          }
        }
      }

      auto outputOperandType =
        cast<ExtraSignalsTypeInterface>(output.getType());

      // The type of the output operand should have the exact unionOfExtraSignals
      // collected from input operands, though the order doesn't matter

      // (1 of 2) Check the number of extra signals
      // #(extraSignals of the output) = #(union(extraSignals of each input))
      if (unionOfExtraSignals.size() != outputOperandType.getExtraSignals().size()) {
        llvm::errs() << "the output doesn't have some extra signals inputs have\n";
        return false;
      }

      // (2 of 2) Check the inclusion relationship (only one direction is sufficient)
      // extraSignals of the output ⊆ union(extraSignals of each input)
      for (auto extraSignal : outputOperandType.getExtraSignals()) {
        if (unionOfExtraSignals.contains(extraSignal.name)) {
          if (*unionOfExtraSignals.lookup(extraSignal.name) != extraSignal) {
            llvm::errs() << "an input has an extra signal " << extraSignal.name
                << ", but it is different from the output\n";
            return false;
          }
        } else {
          llvm::errs() << "the output has an extra signal " << extraSignal.name
              << ", which is not from any input\n";
          return false;
        }
      }

      return true;
    })(
  }] # "$" # inputs # ", $" # output # ")">
>;

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPES_TD
