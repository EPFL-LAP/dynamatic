//===- Passes.td - FTD transformation passes definition ---*- tablegen -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of all experimental transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef FTD_TRANSFORMS_PASSES_TD
#define FTD_TRANSFORMS_PASSES_TD

include "dynamatic/Support/Passes.td"
include "mlir/Pass/PassBase.td"

def HandshakeCombineSteeringLogic : DynamaticPass< "handshake-combine-steering-logic">  {
  let summary = "Combine common steering logic between different handshake operations.";
  let description = [{
      Eliminates redundant steering units that have the same inputs but
      different outputs by combining them into single units feeding multiple
      outputs. This is a simplification which comes immediately after the FTD
      logic.
  }];
  let constructor = "dynamatic::experimental::ftd::combineSteeringLogic()";
}

#endif // FTD_TRANSFORMS_PASSES_TD
