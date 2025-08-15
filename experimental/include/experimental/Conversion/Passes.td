//===-- Passes.td - Conversion passes definition ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for all dialect conversion and export passes.
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_CONVERSION_PASSES_TD
#define EXPERIMENTAL_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"
include "dynamatic/Support/Passes.td"

//===----------------------------------------------------------------------===//
// FtdCfToHandshake
//===----------------------------------------------------------------------===//

def FtdCfToHandshake : DynamaticPass<"ftd-lower-cf-to-handshake"> {
  let summary = "Lowers func and cf dialects to handshake with fast token delivery";
  let description = [{
    The fast token delivery (FTD) methodology was described by 
    Elakhras et al. in FPGA'22, and it represents an optimized way to obtain
    the final elastic circuit in terms of resources occupation and execution time.
    The study by Elakhras et al. in FPGA'23 is a natural continuation of the aforementioned
    work, improving the allocation of operations for LSQ.

    This pass aims at implementing this work, by substituting the current `CfToHandshake` pass.
    As the algorithm does not require any parameter, no input is necessary. This pass can 
    be used through the flag "--fast-token-delivery" at compile tile in dynamatic.
  }];
  let constructor = "dynamatic::experimental::ftd::createFtdCfToHandshake()";
}

#endif // EXPERIMENTAL_CONVERSION_PASSES_TD

