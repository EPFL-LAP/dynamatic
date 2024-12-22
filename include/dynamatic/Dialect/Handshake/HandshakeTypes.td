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
  DeclareTypeInterfaceMethods<ExtraSignalsTypeInterface, ["addExtraSignal", "replaceExtraSignal"]>
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
  DeclareTypeInterfaceMethods<ExtraSignalsTypeInterface, ["addExtraSignal", "replaceExtraSignal"]>
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
/// In the ControlType case, the data type is regarded as a ControlType with no extra signals here.
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

/// Multi-entity constraint ensuring that all HandshakeTypes have matching extra signals.
class AllExtraSignalsMatch<list<string> names> : AllMatchSameOperatorTrait<
  names,
  [{
    ::mlir::cast<::dynamatic::handshake::ExtraSignalsTypeInterface>($_self.getType()).getExtraSignals()
  }],
  "extra signals"
>;

/// A constraint ensuring that all types of a variadic operand have the same data type.
/// Also applies to ControlTypes, as some operations treat them interchangeably.
class AllDataTypesMatchWithinVariadic<string name> : PredOpTrait<
  "the operands inside the variadic " # name # " should have the same type",
  CPred<[{
    ([&](const mlir::OperandRange &operands) {
      // Operands are those specified as the variadic input.
      ValueTypeRange<OperandRange> types = operands.getTypes();
      size_t numTypes = types.size();

      if (numTypes <= 1)
        return true;

      // Here numTypes > 1 is guaranteed

      if (mlir::isa<ControlType>(types.front())) {
        // If the front operand is of ControlType, it is enough to
        // confirm that all other types are also ControlType
        for (size_t i = 1; i < numTypes; i++) {
          if (!mlir::isa<ControlType>(types[i]))
            return false;
        }
        return true;
      }

      // Here the front operand is of ChannelType.

      // Confirm that dataTypes of all types are the same
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
  "the front operand of variadic and non-variadic operand should have the same data types", CPred<[{
    ([&](const mlir::OperandRange &operands, const mlir::Value &op) {
      auto types = operands.getTypes();

      // If variadic is empty (no front operand), returns true
      if (types.empty())
        return true;

      if (mlir::isa<ControlType>(types.front())) {
        // Confirm that non-variadic type is also ControlType
        return mlir::isa<ControlType>(op.getType());
      }
      return mlir::cast<ChannelType>(types.front()).getDataType() == mlir::cast<ChannelType>(op.getType()).getDataType();
    })
  }] # "($" # variadic # ", $" # nonvariadic # ")">
>;

/// Multi-entity constraint ensuring that the operands of variadic and non-variadic
/// operand have the same data types.
/// Also applies to ControlTypes, as some operations treat them interchangeably.
class AllDataTypesMatchWithVariadic<string variadic, string nonvariadic> : PredOpTrait<
  "the operands inside the variadic " # name # " and " # nonvariadic # " should all have the same type",
  And<[
    AllDataTypesMatchWithinVariadic<variadic>.predicate,
    DataTypesMatchWithVariadicFront<variadic, nonvariadic>.predicate
  ]>
>;

/// Constraint to ensure an input/output is of SimpleControl/Channel type.
class IsSimpleHandshake<string name> : PredOpTrait<
  name # " should be of SimpleControl or SimpleChannel type",
  CPred<"cast<::dynamatic::handshake::ExtraSignalsTypeInterface>($" # name #
    ".getType()).getNumExtraSignals() == 0">
>;

/// Merging Relationship of Extra Signals: For all intraOperand and extra signal
/// A, intraOperand has the extra signal A if and only if there exists an interOperand
/// which has the extra signal A.
/// In Merge-like Ops interOperands are always provided as variadic.
class MergingExtraSignals<string variadicInterOperands, list<string> intraOperands> : PredOpTrait<
  "the extra signals of intraOperands and interOperands should satisfy \\\"Merging Relationship of Extra Signals\\\"",
  CPred<[{
    ([&](const mlir::OperandRange &interOperands, llvm::ArrayRef<Value> intraOperands) {
      llvm::DenseMap<StringRef, const ExtraSignal *> extraSignals;
      for (auto interOperandType : interOperands.getTypes()) {
        for (const ExtraSignal &extraSignal :
            cast<ExtraSignalsTypeInterface>(interOperandType).getExtraSignals()) {
          if (extraSignals.contains(extraSignal.name)) {
            if (*extraSignals.lookup(extraSignal.name) != extraSignal) {
              llvm::errs() << "interOperands have different extra signals of the same name: "
                << extraSignal.name << "\n";
              return false;
            }
          } else {
            extraSignals.insert({extraSignal.name, &extraSignal});
          }
        }
      }

      for (Value intraOperand : intraOperands) {
        auto intraOperandType =
          cast<ExtraSignalsTypeInterface>(intraOperand.getType());

        for (auto extraSignal : intraOperandType.getExtraSignals()) {
          if (extraSignals.contains(extraSignal.name)) {
            if (*extraSignals.lookup(extraSignal.name) != extraSignal) {
              llvm::errs() << "intraOperand has an extra signal " << extraSignal.name
                  << ", but it is different from the interOperands\n";
              return false;
            }
          } else {
            llvm::errs() << "intraOperand has an extra signal " << extraSignal.name
                << ", but none of the interOperands has it\n";
            return false;
          }
        }

        if (extraSignals.size() != intraOperandType.getExtraSignals().size()) {
          llvm::errs() << "intraOperand doesn't have some extra signals operands have\n";
          return false;
        }
      }

      return true;
    })(
  }] # "$" # variadicInterOperands # ", {" # !interleave(!foreach(
    /*var=*/intraOperand,
    intraOperands,
    "$" # intraOperand
  ), ", ") # "})">
>;

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPES_TD
