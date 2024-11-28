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

def HandshakeCombineSteeringLogic : DynamaticPass< "handshake-combine-steering-logic">  {
  let summary = "Combine common steering logic between different handshake operations.";
  let description = [{
      Eliminates redundant steering units that have the same inputs but
      different outputs by combining them into single units feeding multiple outputs.
      This is a simplification which comes immediately after the FTD logic.
  }];
  let constructor = "dynamatic::experimental::ftd::combineSteeringLogic()";
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
           "operations.">,
    Option<"automatic", "automatic", "bool", "true",
           "When the automatic mode is enabled, the speculative unit positions are "
           "automatically detected. The speculator position still needs to be "
           "specified in the JSON-formatted file.">];
}

def SpecAnnotatePaths : DynamaticPass<"spec-annotate-paths"> {
  let summary = "Adds an attribute to mark operations in speculative regions";
  let description = [{
    The operations within speculative regions need to be marked for
    VHDL exporting purposes. The pass adds the attribute "speculative" to
    operations that are found within the speculative region, which is defined from
    save units and the speculator, to commit units.
  }];
  let constructor = "dynamatic::experimental::speculation::createSpecAnnotatePaths()";
}

def HandshakePlaceBuffersCustom : DynamaticPass<"handshake-placebuffers-custom"> {
  let summary = "Place buffers on specific channels";
  let description = [{ Placing a single buffer on a specific output channel of
  a specfiic unit, this pass is useful for prototyping a custom placing pass
  externally, e.g., written in python. }];
  let constructor = "dynamatic::experimental::buffer::createHandshakePlaceBuffersCustom()";
  let options = [
  	Option<"pred", "pred", "std::string", "", "the predecessor unit of the channel">,
  	Option<"outid", "outid", "unsigned", "", "output id of the predecessor, range: from 0 to number of outputs - 1">,
  	Option<"slots", "slots", "unsigned", "", "num of slots of buffer, range: anything > 0">,
  	Option<"type", "type", "std::string", "", "type of buffer, can be either oehb or tehb">
  ];
}


def HandshakeAddSeqMem : DynamaticPass<"handshake-add-seq-mem"> {
  let summary = "Adds Sequentialization between memory operations";
  let description = [{
    This path addes forks after and joins before every memory operation, 
    so it can sequentialize the accesses which it needs.
  }];
  let constructor = "dynamatic::experimental::createHandshakeAddSeqMem()";
}

def CreditBasedSharing : DynamaticPass<"credit-based-sharing"> {
  let summary = "Credit-Based Sharing of Functional Units";
  let description = [{ Sharing functional units for better resource utilization.
  This pass currently target maintaining the performance while using as few
  functional units as possible.}];
  let constructor = "dynamatic::experimental::sharing::createCreditBasedSharing()";

  let options = [Option<"algorithm", "algorithm", "std::string", "\"fpga20\"",
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
