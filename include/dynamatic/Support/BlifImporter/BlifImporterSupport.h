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
  BlifImporter(StringRef blifFilePath, ModuleOp moduleOp)
      : blifFilePath(blifFilePath.str()), moduleOp(moduleOp) {}

  // Function to read a blif file and generate the synth circuit depending on
  // the blif description
  LogicalResult populateHWModuleShell();

  // Function to get the input mapping of a signal value
  Value getInputMappingSynthSignal(std::string nodeName);

  // Function to update the output signal mapping after creating a new synth
  // operation
  void updateOutputSynthSignalMapping(Value newResult, std::string nodeName);

  // Function to create hw module operation that represents the shell of the
  // synth circuit being generated from the blif file
  void createHWModuleShell();

  // Function to create a wire in function of synth logic gate operations
  // between an input and an output port based on the input and output bits in
  // the .names definition
  void createSynthWire(Value inputValue, std::string outputPortName,
                       unsigned inputBitFunc, unsigned outputBitFunc);

  // Function to create synth logic gate operations between two input ports and
  // an output port based on the input and output bits in the .names definition
  void createSynthLogicGate(SmallVector<Value> inputValues,
                            std::string outputNodeName, bool invertInput0,
                            bool invertInput1, unsigned outputBit);

  // Function to extract the module name and the input and output ports from the
  // blif file
  LogicalResult extractBlifModuleHeader();

  // Function to get the generated hw module shell containing the synth circuit
  // being generated from the blif file
  hw::HWModuleOp getHWModuleShell() { return hwModuleShell; }

private:
  // String representing the blif file containing the circuit to import
  std::string blifFilePath;
  // String representing the module name in the blif file
  std::string moduleName = "";
  // Vector containing the name of input ports of the blif circuit
  SmallVector<std::string> inputPorts;
  // Vector containing the name of output ports of the blif circuit
  SmallVector<std::string> outputPorts;
  // Map to store the values of the nodes
  llvm::StringMap<Value> nodeValuesMap;
  // Map to store temporary values for nodes not yet defined
  llvm::StringMap<Value> tmpValuesMap;
  // HW ModuleOp containing the synth circuit being generated from the blif file
  hw::HWModuleOp hwModuleShell;
  // Module op in which the hwModuleShell is created
  ModuleOp moduleOp;
};

// Function to generate a new hw module op containing a new synth circuit
// describing the functionality in the blif file
hw::HWModuleOp importBlifCircuit(ModuleOp moduleOp, StringRef blifFilePath);

} // namespace dynamatic
