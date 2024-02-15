//===- Passes.td - Exp. transformation passes definition ---*- tablegen -*-===//
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

#ifndef EXPERIMENTAL_TRANSFORMS_PASSES_TD
#define EXPERIMENTAL_TRANSFORMS_PASSES_TD

include "dynamatic/Support/Passes.td"
include "mlir/Pass/PassBase.td"

def HandshakeFixArgNames : DynamaticPass<"handshake-fix-arg-names"> {
  let summary = "Replace function argument names with original ones from C";
  let description = [{
    Extremely hacky pass that text parses the original source code from which
    the IR was obtained and replaces the default argument names given by
    Polygeist to all function arguments with the ones used in the source code.
  }];
  let constructor = "dynamatic::experimental::createHandshakeFixArgNames()";
  let options =
    [Option<"source", "source", "std::string", "",
            "Path to source C file from which the IR was generated.">];
}

def HandshakeSpeculation : DynamaticPass<"handshake-speculation"> {
  let summary = "Place Speculation operations";
  let description = [{
      Speculative execution involves speculating on the output of a long 
      latency operation or chain of operations, allowing operations that 
      depend on it to begin execution early, without impacting correctness. 
      
      There are several situations that can benefit from early execution, 
      including reducing the initiation interval (II) of loops with conditions 
      that take several cycles to calculate, or loops with 
      loop-carried dependencies. 
      
      The pass needs to specify the position of the Speculation Units by 
      means of an input JSON file.
  }];
  let constructor = "dynamatic::experimental::speculation::createHandshakeSpeculation()";
  let options = [
    Option<"jsonPath", "json-path", "std::string", "",
    "Path to JSON-formatted file containing the positions for speculative "
    "operations.">];
}

//===----------------------------------------------------------------------===//
// Resource sharing passes
//===----------------------------------------------------------------------===//

def ResourceSharingFCCM22 : DynamaticPass<"sharing-FCCM22"> {
  let summary = "Merge sharable operations";
  let description = [{
    Performs two separate steps.
    Step 1: Detects which operations of the same type can be merged with no to little 
            performance overhead
    Step 2: Merges distinctive depictive operations to use one physical operation per 
            predefined clusters.
  }];
  let constructor = "dynamatic::experimental::sharing::createResourceSharingFCCM22Pass()";
  let options = [
    Option<"algorithm", "algorithm", "std::string", "\"fpga20\"",
    "Algorithm to use for buffer placement. Choices are: 'fpga20' (default), "
    "'fpga20-legacy' (same as fpga-20 but interprets the MILP's results as "
    "legacy Dynamatic would, placing at most a single buffer type on any given "
    "channel)">,
    Option<"frequencies", "frequencies", "std::string", "",
    "Path to CSV-formatted file containing estimated transition frequencies "
    "between basic blocks in the kernel.">,
    Option<"timingModels", "timing-models", "std::string", "",
    "Path to JSON-formatted file containing timing models for dataflow "
    "components.">,
    Option<"firstCFDFC", "first-cfdfc", "bool", "false",
    "If true, only extract the first CFDFC from the input file">,
    Option<"targetCP", "target-period", "double", "4.0",
    "Target clock period for the buffer placement CFDFC">,
    Option<"timeout", "timeout", "unsigned", "180",
    "Timeout (in seconds) for the buffer placement MILP (0 for no timeout)">,
    Option<"dumpLogs", "dump-logs", "bool", "false",
    "If true, dump debugging information (identified CFDFCs, MILP "
    "placement decisions, solution summaries) into a directory named "
    "'buffer-placement' at the binary's location.">];
}

#endif // EXPERIMENTAL_TRANSFORMS_PASSES_TD
