//===- HandshakeArithOps.td - Arithmetic ops in Handshake --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file define Handshake equivalent of operations from the `arith` dialect
// in tablegen. 
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_ARTIH_OPS_TD
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_ARTIH_OPS_TD

include "dynamatic/Dialect/Handshake/HandshakeOps.td"
include "dynamatic/Dialect/Handshake/HandshakeTypes.td"
include "dynamatic/Dialect/Handshake/HandshakeInterfaces.td"

class Handshake_Arith_Op<string mnemonic, list<Trait> traits = []> :
  Handshake_Op<mnemonic, traits # [Pure, ArithOpInterface]> {
}

class Handshake_Arith_BinaryOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_Op<mnemonic, traits # [
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>,
  ]> {
  let arguments = (ins ChannelType:$lhs, ChannelType:$rhs, OptionalAttr<StrAttr>:$internal_delay);
  let results = (outs ChannelType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned idx) {
      assert(idx < 2 && "index too high");
      return (idx == 0) ? "lhs" : "rhs";
    }

    std::string $cppClass::getResultName(unsigned idx) {
      assert(idx < 1 && "index too high");
      return "result";
    }
  }];
}

class Handshake_Arith_IntBinaryOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_BinaryOp<mnemonic, traits # [
    IsIntChannel<"lhs">,
    IsIntChannel<"rhs">,
    IsIntChannel<"result">
  ]> {}

class Handshake_Arith_FloatBinaryOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_BinaryOp<mnemonic, traits # [
    IsFloatChannel<"lhs">,
    IsFloatChannel<"rhs">,
    IsFloatChannel<"result">
  ]> {}

class Handshake_Arith_FloatUnaryOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_Op<mnemonic, traits # [SameOperandsAndResultType,
                                         IsFloatChannel<"operand">,
                                         IsFloatChannel<"result">]
> {
  let arguments = (ins ChannelType:$operand, OptionalAttr<StrAttr>:$internal_delay);
  let results = (outs ChannelType:$result);

  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

class Handshake_Arith_CompareOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_Op<mnemonic, traits # [
    AllTypesMatch<["lhs", "rhs"]>,
    AllExtraSignalsMatch<["lhs", "rhs", "result"]>,
    IsIntSizedChannel<1, "result">,
    DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
    DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>
  ]> {
  let results = (outs ChannelType:$result);

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";

  let extraClassDefinition = [{
    std::string $cppClass::getOperandName(unsigned idx) {
      assert(idx < getNumOperands() && "index too high");
      return (idx == 0) ? "lhs" : "rhs";
    }

    std::string $cppClass::getResultName(unsigned idx) {
      assert(idx < 1 && "index too high");
      return "result";
    }
  }];
}

class Handshake_Arith_IToICastOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_Op<mnemonic, traits # [AllExtraSignalsMatch<["in", "out"]>,
                                         IsIntChannel<"in">,
                                         IsIntChannel<"out">]
> {
  let arguments = (ins ChannelType:$in);
  let results = (outs ChannelType:$out);
  
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";

  let hasFolder = 1;
  let hasVerifier = 1;
}

class Handshake_Arith_FToFCastOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_Op<mnemonic, traits # [AllExtraSignalsMatch<["in", "out"]>,
                                         IsFloatChannel<"in">,
                                         IsFloatChannel<"out">]
> {
  let arguments = (ins ChannelType:$in);
  let results = (outs ChannelType:$out);
  
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";

  // TODO: add folding operation
}

class Handshake_Arith_IToFCastOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_Op<mnemonic, traits # [AllExtraSignalsMatch<["in", "out"]>,
                                         IsIntChannel<"in">,
                                         IsFloatChannel<"out">]
> {
  let arguments = (ins ChannelType:$in);
  let results = (outs ChannelType:$out);
  
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";

  // TODO: add folding operation
}

class Handshake_Arith_FToICastOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_Op<mnemonic, traits # [AllExtraSignalsMatch<["in", "out"]>,
                                         IsFloatChannel<"in">,
                                         IsIntChannel<"out">]
> {
  let arguments = (ins ChannelType:$in);
  let results = (outs ChannelType:$out);
  
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";

  // TODO: add folding operation
}

//===----------------------------------------------------------------------===//
// Operation definitions
//===----------------------------------------------------------------------===//

def Handshake_AddFOp : Handshake_Arith_FloatBinaryOp<"addf", [Commutative]> {
  let summary = "Floating-point addition.";
}

def Handshake_AddIOp : Handshake_Arith_IntBinaryOp<"addi", [Commutative]> {
  let summary = "Integer addition.";
}

def Handshake_AndIOp : Handshake_Arith_IntBinaryOp<"andi", [Commutative]> {
  let summary = "Bitwise conjunction.";
}

// Same as the arithmetic floating predicates
def Handshake_CmpFPredicateAttr : I64EnumAttr<
    "CmpFPredicate", "",
    [
      I64EnumAttrCase<"AlwaysFalse", 0, "false">,
      I64EnumAttrCase<"OEQ", 1, "oeq">,
      I64EnumAttrCase<"OGT", 2, "ogt">,
      I64EnumAttrCase<"OGE", 3, "oge">,
      I64EnumAttrCase<"OLT", 4, "olt">,
      I64EnumAttrCase<"OLE", 5, "ole">,
      I64EnumAttrCase<"ONE", 6, "one">,
      I64EnumAttrCase<"ORD", 7, "ord">,
      I64EnumAttrCase<"UEQ", 8, "ueq">,
      I64EnumAttrCase<"UGT", 9, "ugt">,
      I64EnumAttrCase<"UGE", 10, "uge">,
      I64EnumAttrCase<"ULT", 11, "ult">,
      I64EnumAttrCase<"ULE", 12, "ule">,
      I64EnumAttrCase<"UNE", 13, "une">,
      I64EnumAttrCase<"UNO", 14, "uno">,
      I64EnumAttrCase<"AlwaysTrue", 15, "true">,
    ]> {
  let cppNamespace = "::dynamatic::handshake";
}

def Handshake_CmpFOp : Handshake_Arith_CompareOp<"cmpf", [
  IsFloatChannel<"lhs">,
  IsFloatChannel<"rhs">
]> {
  let summary = "Floating-point comparison.";
 
  let arguments = (ins Handshake_CmpFPredicateAttr:$predicate, 
                       ChannelType:$lhs, ChannelType:$rhs);
}

// Same as the arithmetic integer predicates
def Handshake_CmpIPredicateAttr : I64EnumAttr<
    "CmpIPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"slt", 2>,
      I64EnumAttrCase<"sle", 3>,
      I64EnumAttrCase<"sgt", 4>,
      I64EnumAttrCase<"sge", 5>,
      I64EnumAttrCase<"ult", 6>,
      I64EnumAttrCase<"ule", 7>,
      I64EnumAttrCase<"ugt", 8>,
      I64EnumAttrCase<"uge", 9>,
    ]> {
  let cppNamespace = "::dynamatic::handshake";
}

def Handshake_CmpIOp : Handshake_Arith_CompareOp<"cmpi", [
  IsIntChannel<"lhs">,
  IsIntChannel<"rhs">
]> {
  let summary = "Integer comparison.";
 
  let arguments = (ins Handshake_CmpIPredicateAttr:$predicate, ChannelType:$lhs,
                       ChannelType:$rhs);
}

def Handshake_ConstantOp : Handshake_Arith_Op<"constant", [
  AllExtraSignalsMatch<["ctrl", "result"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let summary = "constant operation";
  let description = [{
    The const has a constant value. When triggered by its
    single `ctrl` input, it sends the constant value to its single
    successor.

    Example:

    ```mlir
    %0 = constant %ctrl {value = 42 : i32} : !handshake.channel<i32>
    ```
  }];

  // The ctrl operand may have extra signals.
  // See type verification documentation for details.
  let arguments = (ins TypedAttrInterface:$value, ControlType:$ctrl);
  let results = (outs ChannelType:$result);

  // The type of the control also needs to be specified in the IR.
  // It may have extra bits, which could affect the result's type and token.
  let assemblyFormat = "$ctrl attr-dict `:` type($ctrl) `,` type($result)";
  let hasVerifier = 1;
}

def Handshake_DivFOp : Handshake_Arith_FloatBinaryOp<"divf"> {
  let summary = "Floating-point division.";
}

def Handshake_DivSIOp : Handshake_Arith_IntBinaryOp<"divsi"> {
  let summary = "Signed integer division.";
}

def Handshake_DivUIOp : Handshake_Arith_IntBinaryOp<"divui"> {
  let summary = "Unsigned integer division.";
}

def Handshake_ExtSIOp : Handshake_Arith_IToICastOp<"extsi"> {
  let summary = "Integer unsigned width extension.";
  let hasCanonicalizer = 1;
}

def Handshake_ExtUIOp : Handshake_Arith_IToICastOp<"extui"> {
  let summary = "Integer signed width extension.";
}

def Handshake_MaximumFOp : Handshake_Arith_FloatBinaryOp<"maximumf", [
  Commutative
]> {
  let summary = "Floating-point maximum.";
}

def Handshake_MinimumFOp : Handshake_Arith_FloatBinaryOp< "minimumf", [
  Commutative
]> {
  let summary = "Floating-point minimum.";
}

def Handshake_MulFOp : Handshake_Arith_FloatBinaryOp<"mulf", [Commutative]> {
  let summary = "Floating-point multiplication.";
}

def Handshake_MulIOp : Handshake_Arith_IntBinaryOp<"muli", [Commutative]> {
  let summary = "Integer multiplication.";
}

def Handshake_NegFOp : Handshake_Arith_FloatUnaryOp<"negf"> {
  let summary = "Floating-point sign negation.";
}

def Handshake_OrIOp : Handshake_Arith_IntBinaryOp<"ori", [Commutative]> {
  let summary = "Bitwise union.";
}

def Handshake_SelectOp : Handshake_Arith_Op<"select", [
  AllTypesMatch<["trueValue", "falseValue", "result"]>,
  AllExtraSignalsMatch<["condition", "trueValue", "falseValue", "result"]>,
  IsIntSizedChannel<1, "condition">,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName", "getResultName"]>,
]> {
  let summary = "Select a value based on a 1-bit predicate.";
 
  let arguments = (ins ChannelType:$condition, ChannelType:$trueValue,
                       ChannelType:$falseValue);
  let results = (outs ChannelType:$result);

  let assemblyFormat = [{
    $condition `[` $trueValue `,` $falseValue `]` attr-dict
    `:` type($condition) `,` type($result)
  }];
}

def Handshake_ShLIOp : Handshake_Arith_IntBinaryOp<"shli"> {
  let summary = "Logical left shift.";
}

def Handshake_ShRSIOp : Handshake_Arith_IntBinaryOp<"shrsi"> {
  let summary = "Arithmetic right shift.";
}

def Handshake_ShRUIOp : Handshake_Arith_IntBinaryOp<"shrui"> {
  let summary = "Logical right shift.";
}

def Handshake_SubFOp : Handshake_Arith_FloatBinaryOp<"subf"> {
  let summary = "Floating-point subtraction.";
}

def Handshake_SubIOp : Handshake_Arith_IntBinaryOp<"subi"> {
  let summary = "Integer subtraction.";
}

def Handshake_TruncIOp : Handshake_Arith_IToICastOp<"trunci"> {
  let summary = "Integer truncation.";
  let hasCanonicalizer = 1;
}

def Handshake_TruncFOp : Handshake_Arith_FToFCastOp<"truncf"> {
  let summary = "Floating-point truncation.";
  // TODO: add canonicalizer
  let hasVerifier = 1;
}

def Handshake_XOrIOp : Handshake_Arith_IntBinaryOp<"xori", [Commutative]> {
  let summary = "Bitwise exclusive union.";
}

def Handshake_SIToFPOp : Handshake_Arith_IToFCastOp<"sitofp"> {
  let summary = "Converts a signed integer to float.";
}

def Handshake_FPToSIOp : Handshake_Arith_FToICastOp<"fptosi"> {
  let summary = "Converts a float to signed integer.";
}

def Handshake_ExtFOp : Handshake_Arith_FToFCastOp<"extf"> {
  let summary = "Floating point extension.";
  // TODO: add canonicalizer
  let hasVerifier = 1;
}

def Handshake_AbsFOp : Handshake_Arith_FToFCastOp<"absf"> {
  let summary = "floating point absolute-value operation";
  // TODO: add folder
}

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_ARTIH_OPS_TD