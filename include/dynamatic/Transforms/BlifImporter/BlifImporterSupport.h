//===- BlifImporterSupport.h - Support functions for BLIF importer -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of support functions for the BLIF importer
// pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/Synth/SynthOps.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

namespace synth {
/// Forward declare the Synth dialect which the pass depends on.
class SynthDialect;
} // namespace synth

namespace hw {
/// Forward declare the HW dialect which the pass depends on.
class HWDialect;
} // namespace hw

// Function to read a blif file and generate the synth circuit depending on the
// blif description
LogicalResult
generateSynthCircuitFromBlif(StringRef blifFilePath, Location loc,
                             DenseMap<StringAttr, Value> &synthInputs,
                             SmallVector<std::string> &synthOutputsNames,
                             SmallVector<Value> &synthOutputs,
                             OpBuilder &builder);

// Function to get the input mapping of a signal value
Value getInputMappingSynthSignal(Location loc,
                                 DenseMap<StringAttr, Value> &nodeValuesMap,
                                 DenseMap<StringAttr, Value> &tmpValuesMap,
                                 StringRef nodeName, OpBuilder &builder);

// Function to update the output signal mapping after creating a new synth
// operation
void updateOutputSynthSignalMapping(Value newResult, StringRef nodeName,
                                    DenseMap<StringAttr, Value> &nodeValuesMap,
                                    DenseMap<StringAttr, Value> &tmpValuesMap,
                                    OpBuilder &builder);

// Function to create a wire in function of synth logic gate operations based
// on .names definition
void createSynthWire(Location loc, std::vector<std::string> &ports,
                     std::string &function,
                     DenseMap<StringAttr, Value> &nodeValuesMap,
                     DenseMap<StringAttr, Value> &tmpValuesMap,
                     OpBuilder &builder);

// Function to create synth logic gate operations based on .names definition
void createSynthLogicGate(Location loc, std::vector<std::string> &ports,
                          std::string &function,
                          DenseMap<StringAttr, Value> &nodeValuesMap,
                          DenseMap<StringAttr, Value> &tmpValuesMap,
                          OpBuilder &builder);

// Function to get the module name and the input and output ports from the blif
// file
LogicalResult getBlifModuleHeader(StringRef blifFilePath,
                                  std::string &moduleNameBlif,
                                  SmallVector<std::string> &inputPorts,
                                  SmallVector<std::string> &outputPorts);

// Function to create a new synth circuit from a blif file from an empty IR
void importBlifCircuit(ModuleOp moduleOp, Location loc, StringRef blifFilePath);

} // namespace dynamatic
