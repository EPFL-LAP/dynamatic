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
#include "dynamatic/Support/BLIFIO.h"
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

class BlifImporter {

public:
  // Method to create BlifImporter object with the blif file path as argument
  BlifImporter(StringRef blifFilePath) : blifFilePath(blifFilePath.str()) {}

  // Function to read a blif file and generate the synth circuit depending on
  // the blif description
  LogicalResult generateSynthCircuitFromBlif(
      Location loc, llvm::StringMap<Value> &synthInputs,
      SmallVector<std::string> &synthOutputsNames,
      SmallVector<Value> &synthOutputs, OpBuilder &builder);

  // Function to get the input mapping of a signal value
  Value getInputMappingSynthSignal(Location loc, StringRef nodeName,
                                   OpBuilder &builder);

  // Function to update the output signal mapping after creating a new synth
  // operation
  void updateOutputSynthSignalMapping(Value newResult, StringRef nodeName,
                                      OpBuilder &builder);

  // Function to create a wire in function of synth logic gate operations based
  // on .names definition
  void createSynthWire(Location loc, std::vector<std::string> &ports,
                       std::string &function, OpBuilder &builder);

  // Function to create synth logic gate operations based on .names definition
  void createSynthLogicGate(Location loc, std::vector<std::string> &ports,
                            std::string &function, OpBuilder &builder);

  // Function to extract the module name and the input and output ports from the
  // blif file
  LogicalResult extractBlifModuleHeader();

  // Function to get the module name
  std::string getModuleName() {
    assert(!moduleName.empty() &&
           "Before calling getModuleName(), the module name must be set by "
           "extractBlifModuleHeader().");
    return moduleName;
  }

  // Function to get the input ports names
  SmallVector<std::string> getInputPortsNames() {
    assert(!inputPorts.empty() && "Before calling getInputPortsNames(), the "
                                  "input ports names must be set "
                                  "by extractBlifModuleHeader().");
    return inputPorts;
  }

  // Function to get the output ports names
  SmallVector<std::string> getOutputPortsNames() {
    assert(!outputPorts.empty() && "Before calling getOutputPortsNames(), the "
                                   "output ports names must be set "
                                   "by extractBlifModuleHeader().");
    return outputPorts;
  }

private:
  // String representing the blif file containing the circuit to import
  std::string blifFilePath;
  // String representing the module name in the blif file
  std::string moduleName = "";
  // Vector containing the input ports of the blif circuit
  SmallVector<std::string> inputPorts;
  // Vector containing the output ports of the blif circuit
  SmallVector<std::string> outputPorts;
  // Map to store the values of the nodes
  llvm::StringMap<Value> nodeValuesMap;
  // Map to store temporary values for nodes not yet defined
  llvm::StringMap<Value> tmpValuesMap;
};

// Function to create a new synth circuit from a blif file from an empty IR
void importBlifCircuit(ModuleOp moduleOp, Location loc, StringRef blifFilePath);

} // namespace dynamatic
