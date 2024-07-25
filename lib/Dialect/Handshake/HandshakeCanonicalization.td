//===- HandshakeCanonicalize.td - Dialect patterns ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarative canonicalization patterns for Handhsake operations.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_CANONICALIZATION_TD
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_CANONICALIZATION_TD

include "mlir/IR/PatternBase.td"
include "dynamatic/Dialect/Handshake/HandshakeArithOps.td"

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

// extsi(extui(x iN : iM) : iL) -> extui(x : iL)
def ExtSIOfExtUI : Pat<
  (Handshake_ExtSIOp (Handshake_ExtUIOp $x)), (Handshake_ExtUIOp $x)
>;

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

def ValueWiderThan :
    Constraint<CPred<"getDataBitWidth($0) > getDataBitWidth($1)">>;

def TruncationMatchesShiftAmount :
  Constraint<And<[
    CPred<"succeeded(getIntOrSplatIntValue($2))">,
    CPred<"(getScalarOrElementWidth($0) - getScalarOrElementWidth($1)) == "
            "*getIntOrSplatIntValue($2)">]>>;

// trunci(extsi(x)) -> extsi(x), when only the sign-extension bits are truncated
def TruncIExtSIToExtSI : Pat<
  (Handshake_TruncIOp:$tr (Handshake_ExtSIOp:$ext $x)), (Handshake_ExtSIOp $x),
  [(ValueWiderThan $ext, $tr), (ValueWiderThan $tr, $x)]
>;

// trunci(extui(x)) -> extui(x), when only the zero-extension bits are truncated
def TruncIExtUIToExtUI : Pat<
  (Handshake_TruncIOp:$tr (Handshake_ExtUIOp:$ext $x)), (Handshake_ExtUIOp $x),
  [(ValueWiderThan $ext, $tr), (ValueWiderThan $tr, $x)]
>;

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_CANONICALIZATION_TD
