//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Synth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SYNTH_DIALECT_TD
#define DYNAMATIC_SYNTH_DIALECT_TD

include "mlir/IR/DialectBase.td"

def Synth_Dialect : Dialect {
  let name = "synth";
  let cppNamespace = "::dynamatic::synth";
  let summary = "Synthesis dialect for logic synthesis operations";
  let description = [{
    The Synth dialect provides operations and types for logic synthesis,
    including meta operations for synthesis decisions, logic representations
    like AIG and MIG, and synthesis pipeline infrastructure.
  }];

  let hasConstantMaterializer = 1;
  // Keep printing inherent attributes in `attr-dict` (e.g., the latch type of
  // synth.latch), as before properties became mandatory.
  // FIXME: This option is deprecated: bind the inherent attributes in the
  // assembly formats or use `prop-dict` instead.
  let useStrictPropertiesInAssemblyFormat = 0;
}

include "dynamatic/Dialect/Synth/SynthOps.td"

#endif // DYNAMATIC_SYNTH_DIALECT_TD
