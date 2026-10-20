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

def FtdCfToHandshake : Pass<"ftd-lower-cf-to-handshake", "mlir::ModuleOp"> {
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
  let dependentDialects = ["dynamatic::handshake::HandshakeDialect"];
}

//===----------------------------------------------------------------------===//
// ExpCfToHandshake
//===----------------------------------------------------------------------===//

def ExpCfToHandshake : Pass<"exp-lower-cf-to-handshake", "mlir::ModuleOp"> {
  let summary = "Lowers func and cf dialects to handshake without the dialect "
                "conversion framework.";
  let description = [{
    Experimental version of --lower-cf-to-handshake that produces the same
    circuit. Instead of relying on the dialect conversion framework, each
    func-level function is translated into a new handshake-level function
    using an OpBuilder and a mapping between func-level and handshake-level
    values. Functions must be in maximal SSA form (see --func-maximize-ssa).
  }];
  let dependentDialects = ["dynamatic::handshake::HandshakeDialect"];
}

#endif // EXPERIMENTAL_CONVERSION_PASSES_TD

