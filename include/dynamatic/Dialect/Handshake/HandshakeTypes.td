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

def ExtraSignals : ArrayRefOfSelfAllocationParameter<"ExtraSignal", [{
  This `TypeParameter` describes an array of `ExtraSignal` objects that a
  `ChannelType` or `ControlType` value carries.
  `ExtraSignal` contains data whose lifetime is not guaranteed, so this data must
  be allocated into the MLIR context when constructing a `ChannelType` or
  `ControlType`.
  This type aliases `ArrayRefOfSelfAllocationParameter` to automatically invoke
  `ExtraSignal::allocateInto` C++ function to perform that allocation process.
  `ArrayRefParameter` cannot be used here, as it is strictly for value types.
}]> {}

def ControlType : Handshake_Type<"Control", "control", [
  DeclareTypeInterfaceMethods<ExtraSignalsTypeInterface, ["copyWithExtraSignals"]>
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
    // If no parameters provided, build without extra signals
    TypeBuilder<(ins),
      [{
        return ControlType::get($_ctxt, {});
      }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def ChannelType : Handshake_Type<"Channel", "channel", [
  DeclareTypeInterfaceMethods<ExtraSignalsTypeInterface, ["copyWithExtraSignals"]>
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
// Tablegen helper functions
//===----------------------------------------------------------------------===//

class valuesToExtraSignalArrays<list<string> names> {
  string result = "{" # !interleave(!foreach(
    name, // variable name
    names,
    "cast<ExtraSignalsTypeInterface>($" # name #
    ".getType()).getExtraSignals()"
  ), ", ") # "}";
}

class valueToDataTypes<list<string> names> {
  string result = "{" # !interleave(!foreach(
    name, // variable name
    names,
    "isa<ControlType>($" # name # ".getType()) ? nullptr :"
    "$" # name # ".getType().cast<ChannelType>().getDataType()"
  ), ", ") # "}";
}

class variadicToExtraSignalArrays<string variadic> {
  string result = [{([](const mlir::OperandRange &variadic) {
    llvm::SmallVector<llvm::ArrayRef<ExtraSignal>> extraSignalArrays;
    // Retrieve all types of operands in the variadic.
    ValueTypeRange<OperandRange> variadicTypes = variadic.getTypes();
    for (Type type : variadicTypes) {
      extraSignalArrays.push_back(
        type.cast<ExtraSignalsTypeInterface>().getExtraSignals());
    }
    return extraSignalArrays;
  })}] # "($" # variadic # ")";
}

class variadicToDataTypes<string variadic> {
  string result = [{([](const mlir::OperandRange &variadic) {
    llvm::SmallVector<Type> dataTypes;
    // Retrieve all types of operands in the variadic.
    ValueTypeRange<OperandRange> variadicTypes = variadic.getTypes();
    for (Type type : variadicTypes) {
      if (isa<ControlType>(type))
        dataTypes.push_back(nullptr);
      else
        dataTypes.push_back(type.cast<ChannelType>().getDataType());
    }
    return dataTypes;
  })}] # "($" # variadic # ")";
}

class variadicToExtraSignalTypes<string variadic> {
  string result = [{([](const mlir::OperandRange &variadic) {
    llvm::SmallVector<ExtraSignalsTypeInterface> extraSignalTypes;
    // Retrieve all types of operands in the variadic.
    ValueTypeRange<OperandRange> variadicTypes = variadic.getTypes();
    for (Type type : variadicTypes) {
      extraSignalTypes.push_back(type.cast<ExtraSignalsTypeInterface>());
    }
    return extraSignalTypes;
  })}] # "($" # variadic # ")";
}

class concatArrays<string type, string a, string b> {
  string result = !subst("$T", type, [{[](llvm::ArrayRef<$T> a, llvm::ArrayRef<$T> b) {
    llvm::SmallVector<$T> result;
    result.insert(result.end(), a.begin(), a.end());
    result.insert(result.end(), b.begin(), b.end());
    return result;
  } }]) # "(" # a # ", " # b # ")";
}

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

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

class IsChannelPred<string name> : CPred<
  "::mlir::isa<::dynamatic::handshake::ChannelType>($" # name # ".getType())">;

class IsIntChannelPred<string name> : CPred<
  "::mlir::isa<::mlir::IntegerType>("
  "::mlir::cast<::dynamatic::handshake::ChannelType>($" # name # ".getType())"
  ".getDataType())">;

class IsFloatChannelPred<string name> : CPred<
  "::mlir::isa<::mlir::FloatType>("
  "::mlir::cast<::dynamatic::handshake::ChannelType>($" # name # ".getType())"
  ".getDataType())">;

class IsSizedChannelPred<int width, string name> : CPred<
  "::mlir::cast<::dynamatic::handshake::ChannelType>($" # name # ".getType())"
  ".getDataBitWidth() == " # width>;

/// Ensures an operand/result is of ChannelType carrying IntegerType data.
class IsIntChannel<string name> : PredOpTrait<
  name # " should be of ChannelType carrying IntegerType data",
  And<[
    IsChannelPred<name>,
    IsIntChannelPred<name>
  ]>
>;

/// Ensures an operand/result is of ChannelType carrying IntegerType data of the
/// specified width.
class IsIntSizedChannel<int width, string name> : PredOpTrait<
  name # " should be of ChannelType carrying IntegerType data of width " # width,
  And<[
    IsChannelPred<name>,
    IsIntChannelPred<name>,
    IsSizedChannelPred<width, name>
  ]>
>;

/// Ensures an operand/result is of ChannelType carrying FloatType data.
class IsFloatChannel<string name> : PredOpTrait<
  name # " should be of ChannelType carrying FloatType data",
  And<[
    IsChannelPred<name>,
    IsFloatChannelPred<name>
  ]>
>;

/// Constraint to ensure an operand/result shouldn't have any extra signals.
class IsSimpleHandshake<string name> : PredOpTrait<
  name # " shouldn't have any extra signals",
  CPred<"cast<::dynamatic::handshake::ExtraSignalsTypeInterface>($" # name #
    ".getType()).getNumExtraSignals() == 0">
>;

/// Constraint to ensure a variadic operand/result shouldn't have any extra signals.
class IsSimpleHandshakeVariadic<string name> : PredOpTrait<
  name # " shouldn't have any extra signals",
  CPred<[{
    ([](auto&& variadicTypes) {
      for (::mlir::Type variadicType : variadicTypes) {
        if (cast<ExtraSignalsTypeInterface>(variadicType).getNumExtraSignals() != 0)
          return false;
      }
      return true;
    })
  }] # "($" # name # ".getTypes())">
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
    ::mlir::isa<::dynamatic::handshake::ControlType>($_self.getType()) ? nullptr :
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

/// Multi-entity constraint ensuring that all HandshakeTypes have matching extra
/// signals, except for any signal with the given `except` name.
/// The core implementation of this predicate is in C++.
class AllExtraSignalsMatchExcept<string except, list<string> names> : PredOpTrait<
  "all of {" # !interleave(names, ", ") # "} should have the same extra " #
  "signals except for " # except,
  CPred<"::dynamatic::handshake::doesExtraSignalsMatchExcept(\"" # except #
  "\", " # valuesToExtraSignalArrays<names>.result  # ")">
>;

/// Multi-entity constraint ensuring that the all operands of variadic and non-variadic
/// operand have the same data types.
/// Also applies to ControlTypes, as some operations treat them interchangeably.
/// Returns true if the variadic is empty.
class AllDataTypesMatchWithVariadic<string variadic, list<string> nonvariadics> : PredOpTrait<
  "all of {" # variadic # ", " # !interleave(nonvariadics, ", ") # "} should "
  "have the same data type",
  CPred<[{
    [](llvm::ArrayRef<Type> dataTypes) {
      if (dataTypes.empty())
        return true;
      Type firstType = dataTypes.front();
      return llvm::all_of(dataTypes, [&](Type type) {
        return type == firstType;
      });
    }
  }] # "(" # concatArrays<"Type",
    variadicToDataTypes<variadic>.result,
    valueToDataTypes<nonvariadics>.result
  >.result # ")">
>;

class AllExtraSignalsMatchWithVariadicExcept<string except, string variadic, list<string> nonvariadics> : PredOpTrait<
  "all of {" # variadic # ", " # !interleave(nonvariadics, ", ") # "} should "
  "have the same extra signals except for " # except,
  CPred<"::dynamatic::handshake::doesExtraSignalsMatchExcept(\"" # except #
  "\", " # concatArrays<"llvm::ArrayRef<ExtraSignal>",
    variadicToExtraSignalArrays<variadic>.result,
    valuesToExtraSignalArrays<nonvariadics>.result
  >.result # ")">
>;

/// Constraint to ensure that the variadic has at least one element.
class VariadicHasElement<string variadic> : PredOpTrait<
  "the variadic " # variadic # " should have at least one element",
  CPred<"!$" # variadic # ".empty()">
>;

// Constraints for spec tags

class HasValidSpecTag<string operand> : PredOpTrait<
  "should have a valid spec tag as an extra signal",
  CPred<[{
    [](ExtraSignalsTypeInterface type) {
      auto getExtraSignalResult = type.getExtraSignal("spec");
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
    }
  }] # "($" # operand # ".getType().cast<ExtraSignalsTypeInterface>())">
>;

class LacksSpecTag<string operand> : PredOpTrait<
  "shouldn't have a spec tag as an extra signal",
  CPred<"!::mlir::cast<::dynamatic::handshake::ExtraSignalsTypeInterface>($" #
  operand # ".getType()).hasExtraSignal(\"spec\")">
>;

class HasSpecTagIfPresent<string inputVariadic, string output> : PredOpTrait<
  "the extra signals of inputs and output should satisfy \\\"Merging Relationship of Extra Signals\\\"",
  CPred<[{
    [](llvm::ArrayRef<ExtraSignalsTypeInterface> inputTypes,
        ExtraSignalsTypeInterface outputType) {
      bool inputHasSpecTag = llvm::any_of(inputTypes, [](ExtraSignalsTypeInterface type) {
        return type.hasExtraSignal("spec");
      });
      return outputType.hasExtraSignal("spec") == inputHasSpecTag;
    }
  }] # "(" # variadicToExtraSignalTypes<inputVariadic>.result # ", "
  "$" # output # ".getType().cast<ExtraSignalsTypeInterface>())">
>;

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPES_TD
