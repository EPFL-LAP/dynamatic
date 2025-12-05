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
           "specified in the JSON-formatted file.">,
    Option<"sparsePredictor", "sparse-predictor", "bool", "false", "custom predictor for sparse benchmark">,
  ];
}

def HandshakeSpeculationAdapted : DynamaticPass<"handshake-speculation-adapted"> {
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
  let summary = "post buffering spec pass";
  let description = [{
  }];
  let options = [
    Option<"defaultValue", "default-value", "unsigned", "1",
           "Default speculated value">,
    Option<"constant", "constant", "bool", "true",
           "Constant speculation">,
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

def PreSpecV2 : DynamaticPass<"handshake-pre-spec-v2"> {
  let summary = "Spec v2 preparation";
  let description = [{
  }];
  let options = [
    Option<"jsonPath", "json-path", "std::string", "",
           "Path to JSON-formatted file containing the kernel information related to speculation.">,
  ];
}

def PostSpecV2 : DynamaticPass<"handshake-post-spec-v2"> {
  let summary = "Spec v2 finalization";
  let description = [{
  }];
  let options = [
  ];
}

def HandshakeSpeculationV2 : DynamaticPass<"handshake-speculation-v2"> {
  let summary = "Place Speculation operations";
  let description = [{
  }];
  let options = [
    Option<"jsonPath", "json-path", "std::string", "",
           "Path to JSON-formatted file containing the kernel information related to speculation.">,
    Option<"n", "n", "unsigned", "", "the number of iterations to speculate">,
    Option<"variable", "variable", "bool", "false", "enable variable speculation">,
    Option<"bbMapping", "bb-mapping", "std::string", "",
           "Path to the output bb mapping file.">,
    Option<"disableInitialMotion", "disable-initial-motion", "bool", "false", "disable initial motion of suppressors when loop consists of multiple BBs">,
    Option<"exitEagerEval", "exit-eager-eval", "bool", "false", "enable eager evaluation of exit conditions">,
    Option<"resolver", "resolver", "bool", "false", "introduce spec resolver for better circuit construct">
  ];
}

def SpecV2PostBuffering : DynamaticPass<"handshake-spec-v2-post-buffering"> {
  let summary = "Spec v2 post buffering";
  let description = [{
  }];
  let options = [
  ];
}

def SpecV2CutCondDep : DynamaticPass<"handshake-spec-v2-cut-cond-dep"> {
  let summary = "Spec v2 cut condition dependency";
  let description = [{
  }];
  let options = [
    Option<"jsonPath", "json-path", "std::string", "",
           "Path to JSON-formatted file containing the kernel information related to speculation.">,
  ];
}

def PreSpecV2Gamma : DynamaticPass<"handshake-pre-spec-v2-gamma"> {
  let summary = "Place Speculation operations";
  let description = [{
  }];
  let options = [
    Option<"branchBB", "branch-bb", "unsigned", "false", "branching bb">,
    Option<"mergeBB", "merge-bb", "unsigned", "false", "merging bb">,
  ];
}

def SpecV2Gamma : DynamaticPass<"handshake-spec-v2-gamma"> {
  let summary = "Place Speculation operations";
  let description = [{
  }];
  let options = [
    Option<"branchBB", "branch-bb", "unsigned", "false", "branching bb">,
    Option<"mergeBB", "merge-bb", "unsigned", "false", "merging bb">,
    Option<"oneSided", "one-sided", "bool", "false", "one-sided spec">,
    Option<"prioritizedSide", "prioritized-side", "unsigned", "", "">,
    Option<"stepsUntil", "steps-until", "unsigned", "", "">,
    Option<"bbMapping", "bb-mapping", "std::string", "",
           "Path to the output bb mapping file.">,
    Option<"emulatePrediction", "emulate-prediction", "bool", "false",
           "Emulate prediction behavior (with oneSided option)">,
  ];
}

def SpecV2GammaPostBuffering : DynamaticPass<"handshake-spec-v2-gamma-post-buffering"> {
  let summary = "Place Speculation operations";
  let description = [{
  }];
  let options = [
    Option<"prioritizedSide", "prioritized-side", "unsigned", "", "">,
  ];
}

def HandshakeCopyBuffers : DynamaticPass<"handshake-copy-buffers"> {
  let summary = "Copy buffers";
  let description = [{
  }];
  let options = [
    Option<"preUnrollingPath", "pre-unrolling-path", "std::string", "", "">,
    Option<"preUnrollingBB", "pre-unrolling-bb", "unsigned", "", "">,
    Option<"postUnrollingBB", "post-unrolling-bb", "unsigned", "", "">,
  ];
}

def HandshakeCutBBsByBuffers : DynamaticPass<"handshake-cut-bbs-by-buffers"> {
  let summary = "Cut BBs by buffers";
  let description = [{
  }];
  let options = [
    Option<"preBB", "pre-bb", "unsigned", "", "">,
    Option<"postBB", "post-bb", "unsigned", "", "">,
  ];
}

def SpecV1Adaptor : DynamaticPass<"handshake-spec-v1-adaptor"> {
  let summary = "Place Speculation operations";
  let description = [{
  }];
  let options = [
    Option<"jsonPath", "json-path", "std::string", "",
           "Path to JSON-formatted file containing the kernel information related to speculation.">,
    Option<"bbMapping", "bb-mapping", "std::string", "",
           "Path to the output bb mapping file.">,
  ];
}

def SpecV1PostAdaptor : DynamaticPass<"handshake-spec-v1-post-adaptor"> {
  let summary = "Place Speculation operations";
  let description = [{
  }];
  let options = [
    Option<"disablePasserAtExits", "disable-passer-at-exits", "bool", "false", "">,
  ];
}

def CfGateBinarization : DynamaticPass<"cf-gate-binarization"> {
  let summary = "Limit entries to at most 2";
  let description = [{
  }];
  let options = [
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
