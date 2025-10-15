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
      different outputs by combining them into single units feeding multiple 
      outputs. This is a simplification which comes immediately after the FTD 
      logic.
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
  let options = [
    Option<"jsonPath", "json-path", "std::string", "",
           "Path to JSON-formatted file containing the positions for speculative "
           "operations.">,
    Option<"automatic", "automatic", "bool", "true",
           "When the automatic mode is enabled, the speculative unit positions are "
           "automatically detected. The speculator position still needs to be "
           "specified in the JSON-formatted file.">];
}

def HandshakeSpecPostBuffer : DynamaticPass<"handshake-spec-post-buffer"> {
  let summary = "Post-buffering speculation pass";
  let description = [{
    Speculation integration requires some steps to be performed after the
    buffer placement pass:
    - Finalize speculative units
  }];
  let options = [
  ];
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
  	Option<"type", "type", "std::string", "", "type of buffer">
  ];
}

def CreditBasedSharing : Pass<"credit-based-sharing"> {
  let summary = "Credit-based sharing of functional units.";
  let description = [{ Share functional units for better resource utilization.
  The behavior of this pass depends on whether the buffer placement pass was ran
  in the same pass pipeline before:
  - If yes, this pass share functional units as much as possible without
  penalizing the performance.
  - Otherwise, it simply shares all units of the same type.}];

  let options = [
      Option<"timingModels", "timing-models", "std::string", "",
      "Path to JSON-formatted file containing timing models for dataflow "
      "components.">,
      Option<"targetCP", "target-period", "double", "",
      "Target clock period for the buffer placement CFDFC">];

  let dependentDialects = ["handshake::HandshakeDialect"];
}

def HandshakeSizeLSQs : DynamaticPass<"handshake-size-lsqs"> {
  let summary = "Size the LSQs based on the buffer placement information.";
  let description = [{
    Calculates the necessary Load-Store-Queue depths, based on the buffer placement information, 
    to ensure that the memory accesses are not restricting the circuits troughput, while 
    trying to keep the area as low as possible.  
  }];
  let constructor = "dynamatic::experimental::lsqsizing::createHandshakeSizeLSQs()";
  
  let options = [Option<"timingModels", "timing-models", "std::string", "",
      "Path to JSON-formatted file containing timing models for dataflow "
      "components.">,
      Option<"collisions", "collisions", "std::string", "",
      "Three different cases for memory collsions: none/half/full">,
      Option<"targetCP", "target-period", "double", "4.0",
      "Target clock period for the LSQ pass">];
}

def HandshakeRigidification : DynamaticPass<"handshake-rigidification"> {
  let summary = "Simplifies handshake logic";
  let description = [{
    The pass uses the insights gathered from model checking stored in the property database
    to rigidify some channels and remove the redundant logic associated with them. To do so
    we place ReadyRemoverOps and ValidMergerOps, that manipulate valid and ready signals.
  }];
  let constructor = "dynamatic::experimental::rigidification::createRigidification()";
  let options = [ Option<"jsonPath", "json-path", "std::string", "",
    "Path to JSON-formatted file where the properties' information is stored.">];
}

#endif // EXPERIMENTAL_TRANSFORMS_PASSES_TD
