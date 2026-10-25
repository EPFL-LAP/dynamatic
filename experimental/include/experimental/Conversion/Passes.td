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
// OperandRoutingCfToHandshake
//===----------------------------------------------------------------------===//

def OperandRoutingCfToHandshake
    : Pass<"operand-routing-lower-cf-to-handshake", "mlir::ModuleOp"> {
  let summary = "Lowers func and cf dialects to handshake, routing every "
                "consumer operand through its own control network";
  let description = [{
    Same lowering as `CfToHandshake`, except that every operand of a consumer
    that reads a block argument gets its own copy of the control network and
    its own branches and muxes from the producers to the consumer. Used
    through the "--operand-routing" flag of the compile command.
  }];
  let dependentDialects = ["dynamatic::handshake::HandshakeDialect"];
  let options = [
    Option<"maxCfgs", "max-cfgs", "unsigned", "0",
           "Maximum number of distinct folded CFGs, and so of control "
           "networks, the consumers are clustered into. 0 means no limit: "
           "only consumers with identical folded CFGs share one.">,
    Option<"mergeThreshold", "merge-threshold", "unsigned", "0",
           "Consumers share a folded CFG when every one of them has at most "
           "this many edges in the shared CFG that its own folded CFG does "
           "not have. Applied before max-cfgs. 0 means only identical "
           "folded CFGs are shared.">
  ];
}

//===----------------------------------------------------------------------===//
// OperandRouting2CfToHandshake
//===----------------------------------------------------------------------===//

def OperandRouting2CfToHandshake
    : Pass<"operand-routing-2-lower-cf-to-handshake", "mlir::ModuleOp"> {
  let summary = "Lowers func and cf dialects to handshake, routing every "
                "consumer on its own optimized CFG";
  let description = [{
    Same lowering as `CfToHandshake`, except that every consumer is routed
    with the control-network routing algorithm on a CFG optimized to that
    consumer. Used through the "--operand-routing-2" flag of the compile
    command.
  }];
  let dependentDialects = ["dynamatic::handshake::HandshakeDialect"];
}

#endif // EXPERIMENTAL_CONVERSION_PASSES_TD

