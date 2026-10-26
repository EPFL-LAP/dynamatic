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

#ifndef DYNAMATIC_CONVERSION_PASSES_TD
#define DYNAMATIC_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"
include "dynamatic/Support/Passes.td"

//===----------------------------------------------------------------------===//
// AffineToScf
//===----------------------------------------------------------------------===//

def AffineToScf : Pass<"lower-affine-to-scf", "mlir::ModuleOp"> {
  let summary =
      "Lower affine operations to a mix of SCF and standard operations";
  let description = [{
    Very close analog to the AffineToStandard pass from MLIR that preserves the
    attributes of AffineLoadOp's and AffineStoreOp's during lowering. Keeping
    these attributes is required for our memory analysis pass to pass down
    dependence information between memory accesses to lower IR levels, in
    particular to the std-to-handshake lowering pass. 
  }];
  let dependentDialects = ["mlir::memref::MemRefDialect", "mlir::scf::SCFDialect", "mlir::vector::VectorDialect", "dynamatic::handshake::HandshakeDialect"];
}

//===----------------------------------------------------------------------===//
// ScfToCf
//===----------------------------------------------------------------------===//

def ScfToCf : Pass<"lower-scf-to-cf", "mlir::ModuleOp"> {
  let summary = "Lower scf dialect to unstructured control flow (cf)";
  let description = [{
    Very close analog to the SCFToControlFlow pass from MLIR that replaces the
    structured for loop lowering pattern with an almost identical one that
    additionally attempts to insert an unsigned comparison (ult) in the IR
    instead of a signed one (lt) if the loop's iterator can be proven to be
    always positive.
  }];
  let dependentDialects = ["mlir::cf::ControlFlowDialect", "mlir::arith::ArithDialect", "dynamatic::handshake::HandshakeDialect"];
}

//===----------------------------------------------------------------------===//
// CfToHandshake
//===----------------------------------------------------------------------===//

def CfToHandshake : Pass<"lower-cf-to-handshake", "mlir::ModuleOp"> {
  let summary = "Lowers func and cf dialects to handshake.";
  let description = [{
    Lowers func-level functions whose body have unstructured control flow into
      their handshake counterparts,
      which represent dataflow circuits that can ultimately be converted to an
          RTL design.
    Functions must be in maximal SSA form (see --func-maximize-ssa).
  }];
  let dependentDialects = ["dynamatic::handshake::HandshakeDialect"];
}

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

//===----------------------------------------------------------------------===//
// HandshakeToHW
//===----------------------------------------------------------------------===//

def HandshakeToHW
    : Pass<"lower-handshake-to-hw", "mlir::ModuleOp"> {
  let summary = "Lowers Handshake to HW.";
  let description = [{
    Lowers Handshake IR into a "netlist-style" representation made up of HW
    instances of external modules interconnected by dataflow channels.
  }];
  let options = [Option<
      "instrumentII", "instrument-ii", "bool", "false",
      "Instrument the generated netlist with a simulation-only monitor per "
      "loop that reports the initiation of each of its iterations during "
      "simulation.">];
  let dependentDialects = ["dynamatic::hw::HWDialect", "dynamatic::handshake::HandshakeDialect"];
}

//===----------------------------------------------------------------------===//
// HandshakeToSynth
//===----------------------------------------------------------------------===//

def HandshakeToSynth
    : Pass<"lower-handshake-to-synth"> {
  let summary = "Lowers Handshake to Synth.";
  let description = [{
    Lowers Handshake IR into Synth IR, which is a hardware representation
    that focuses on logic synthesis constructs.
  }];
  let dependentDialects = ["dynamatic::hw::HWDialect", "dynamatic::synth::SynthDialect"];
}


#endif // DYNAMATIC_CONVERSION_PASSES_TD
