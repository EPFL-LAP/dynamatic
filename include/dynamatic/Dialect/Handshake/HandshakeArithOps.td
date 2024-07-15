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

class Handshake_Arith_Op<string mnemonic, list<Trait> traits = []> :
  Handshake_Op<mnemonic, traits # [Pure]> {
}

class Handshake_Arith_IntBinaryOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins IntChannelType:$lhs, IntChannelType:$rhs);
  let results = (outs IntChannelType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class Handshake_Arith_FloatBinaryOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins FloatChannelType:$lhs, FloatChannelType:$rhs);
  let results = (outs FloatChannelType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}


class Handshake_Arith_FloatUnaryOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins FloatChannelType:$operand);
  let results = (outs FloatChannelType:$result);

  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

class Handshake_Arith_CompareOp<string mnemonic, list<Trait> traits = []> :
  Handshake_Arith_Op<mnemonic, traits # [
    AllTypesMatch<["lhs", "rhs"]>,
    DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
  ]> {
  let results = (outs BoolChannelType:$result);

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
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

def Handshake_CmpFOp : Handshake_Arith_CompareOp<"cmpf"> {
  let summary = "Floating-point comparison.";
 
  let arguments = (ins Handshake_CmpFPredicateAttr:$predicate, 
                       FloatChannelType:$lhs, FloatChannelType:$rhs);
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

def Handshake_CmpIOp : Handshake_Arith_CompareOp<"cmpi"> {
  let summary = "Integer comparison.";
 
  let arguments = (ins Handshake_CmpIPredicateAttr:$predicate, IntChannelType:$lhs,
                       IntChannelType:$rhs);
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

def Handshake_ExtSIOp : Handshake_Arith_IntBinaryOp<"extsi"> {
  let summary = "Signed width extension.";
}

def Handshake_ExtUIOp : Handshake_Arith_IntBinaryOp<"extui"> {
  let summary = "Unsigned width extension.";
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

def Handshake_OrIOp : Handshake_Arith_IntBinaryOp<"muli", [Commutative]> {
  let summary = "Bitwise union.";
}

def Handshake_SelectOp : Handshake_Arith_Op<"select", [
  AllTypesMatch<["trueValue", "falseValue", "result"]>
]> {
  let summary = "Select a value based on a 1-bit predicate.";
 
  let arguments = (ins BoolChannelType:$select, ChannelType:$trueValue,
                       ChannelType:$falseValue);
  let results = (outs ChannelType:$result);

  let assemblyFormat = [{
    $select `[` $trueValue `,` $falseValue `]` attr-dict
    `:` type($select) `,` type($result)
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

def Handshake_SubIOp : Handshake_Arith_IntBinaryOp<"subi"> {
  let summary = "Integer subtraction.";
}

def Handshake_SubFOp : Handshake_Arith_FloatBinaryOp<"subf"> {
  let summary = "Floating-point subtraction.";
}

def Handshake_XOrIOp : Handshake_Arith_IntBinaryOp<"xori", [Commutative]> {
  let summary = "Bitwise exclusive union.";
}

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_ARTIH_OPS_TD
