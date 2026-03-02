
//===- BlifImporterSupport.cpp - Support functions for BLIF importer -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of support functions for the BLIF importer
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BlifImporter/BlifImporterSupport.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/HW/HW.h.inc"
#include "dynamatic/Dialect/HW/HWOpInterfaces.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/HW/HWTypes.h"
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Dialect/Synth/SynthDialect.h"
#include "dynamatic/Dialect/Synth/SynthOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <bitset>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>
#include <utility>

namespace dynamatic {

// Function to get the module name and the input and output ports from the blif
// file
LogicalResult BlifImporter::extractBlifModuleHeader() {

  std::ifstream file(blifFilePath);

  if (!file.is_open()) {
    llvm::errs() << "The blif file '" << blifFilePath
                 << "' has not been found or could not be opened." << "\n";
    return failure();
  }

  std::string line;
  // Loop over all the lines in .blif file.
  while (std::getline(file, line)) {
    // If comment or empty line, skip.
    if (line.empty() || line.find("#") == 0) {
      continue;
    }

    // If line ends with '\\', read the next line and append to the current
    // line.
    while (line.back() == '\\') {
      line.pop_back();
      std::string nextLine;
      std::getline(file, nextLine);
      line += nextLine;
    }

    std::istringstream iss(line);
    std::string type;
    iss >> type;

    // Model name
    if (type == modelNode) {
      std::string module;
      iss >> module;
      if (module.empty()) {
        llvm::errs()
            << "The .model line in the blif file does not contain a "
               "module name. Please provide a valid module name in the "
               ".model line of the blif file."
            << "\n";
        return failure();
      }
      moduleName = module;
    }

    // Input/Output nodes. These are also Dataflow graph channels.
    else if ((type == inputsNodes) || (type == outputsNodes)) {
      std::string nodeName;
      while (iss >> nodeName) {
        if (type == inputsNodes) {
          // Check if the input has already been added to the input ports list
          if (std::find(inputPorts.begin(), inputPorts.end(), nodeName) !=
              inputPorts.end()) {
            llvm::errs() << "Input node '" << nodeName
                         << "' has already been added to the input ports list."
                         << "\n";
            return failure();
          }
          // Push a copy of the node name to the input ports list
          inputPorts.push_back(nodeName);
        } else {
          // Check if the output has already been added to the output ports list
          if (std::find(outputPorts.begin(), outputPorts.end(), nodeName) !=
              outputPorts.end()) {
            llvm::errs() << "Output node '" << nodeName
                         << "' has already been added to the output ports list."
                         << "\n";
            return failure();
          }
          outputPorts.push_back(nodeName);
        }
      }
    }

    // If it is any other type (i.e., .latch, .names, .subckt), we can stop
    // reading the header since we have already collected the module name and
    // the input and output ports and we assume that they should be defined
    // before any other element in the blif file
    else if (type == latchNode || type == logicNode || type == subcktNode) {
      break;
    }

    // Ends the file.
    else if (line.find(endNode) == 0) {
      break;
    }
  }

  // Close the file
  file.close();

  return success();
}
// Function to read a blif file and generate the synth circuit depending on the
// blif description
LogicalResult BlifImporter::generateSynthCircuitFromBlif(
    Location loc, llvm::StringMap<Value> &synthInputs,
    SmallVector<std::string> &synthOutputsNames,
    SmallVector<Value> &synthOutputs, OpBuilder &builder) {
  // The following function reads the blif file specified by blifFilePath
  // and generates the synth circuit defined in it.

  // Iterate over the synth inputs and add the mapping to nodeValuesMap
  for (const auto &synthInput : synthInputs) {
    std::string inputName = synthInput.first().str();
    Value inputValue = synthInput.second;
    // Check the input name is not already in the map
    assert(!nodeValuesMap.count(inputName) &&
           "input name already exists in the node values map");
    nodeValuesMap[inputName] = inputValue;
  }

  std::ifstream file(blifFilePath);

  if (!file.is_open()) {
    llvm::errs() << "The blif file '" << blifFilePath
                 << "' has not been found or could not be opened." << "\n";
    return failure();
  }

  std::string line;
  // Loop over all the lines in .blif file.
  while (std::getline(file, line)) {
    // If comment or empty line, skip.
    if (line.empty() || line.find("#") == 0) {
      continue;
    }

    // If line ends with '\\', read the next line and append to the current
    // line.
    while (line.back() == '\\') {
      line.pop_back();
      std::string nextLine;
      std::getline(file, nextLine);
      line += nextLine;
    }

    std::istringstream iss(line);
    std::string type;
    iss >> type;

    // Latches.
    if (type == latchNode) {
      std::string regInput, regOutput;
      iss >> regInput >> regOutput;
      // Elaboration of optional fields for latches. The syntax for latches in
      // blif files is: .latch <input> <output> [type control] [init]
      std::string latchType, controlName;
      std::optional<int64_t> initVal;

      // Collect remaining tokens
      std::vector<std::string> remaining;
      std::string token;
      while (iss >> token) {
        remaining.push_back(token);
      }

      if (remaining.size() == 1) {
        // Only init value present
        initVal = std::stoll(remaining[0]);
      } else if (remaining.size() == 2) {
        // type and control present, no init value
        latchType = remaining[0];
        controlName = remaining[1];
      } else if (remaining.size() == 3) {
        // type, control, and init value present
        latchType = remaining[0];
        controlName = remaining[1];
        initVal = std::stoll(remaining[2]);
      } else if (remaining.size() > 3) {
        llvm::errs() << "Too many optional fields in .latch definition: "
                     << line << "\n";
        return failure();
      }
      // remaining.size() == 0: no optional fields present, which is also valid

      // Get the input signal for the register
      Value inputSignal = getInputMappingSynthSignal(loc, regInput, builder);

      // Get optional control signal
      Value controlSignal = nullptr;
      if (!controlName.empty()) {
        controlSignal = getInputMappingSynthSignal(loc, controlName, builder);
      }

      // Create synth register operation
      auto regOp = builder.create<synth::LatchOp>(
          loc, builder.getIntegerType(1), inputSignal,
          latchType.empty() ? nullptr : builder.getStringAttr(latchType),
          controlSignal,
          initVal.has_value() ? builder.getI64IntegerAttr(*initVal) : nullptr);

      // Map the output node to the register output
      updateOutputSynthSignalMapping(regOp.getResult(), regOutput, builder);

    }

    // .names stand for logic gates.
    else if (type == logicNode) {
      std::vector<std::string> nodeNames;
      std::string currentNode;

      // Read node names from current line (e.g., "a b c")
      while (iss >> currentNode) {
        nodeNames.push_back(currentNode);
      }

      // Read logic function from next line (e.g., "11 1")
      std::getline(file, line);
      std::string function = line;

      if (nodeNames.size() == 1) {
        // Convert string function to a single bit (e.g., "1" -> 1, "0" -> 0)
        function.erase(
            std::remove_if(function.begin(), function.end(), ::isspace),
            function.end());
        if (function != "0" && function != "1") {
          llvm::errs()
              << "Invalid function for a single node in .names definition: "
              << function << "\n";
          return failure();
        }
        // Create hw constant
        auto constOp = builder.create<hw::ConstantOp>(
            loc, builder.getIntegerType(1),
            builder.getIntegerAttr(builder.getIntegerType(1),
                                   function == "1" ? 1 : 0));

        // Map the output node to the constant output
        updateOutputSynthSignalMapping(constOp.getResult(), nodeNames[0],
                                       builder);
      } else if (nodeNames.size() == 2) {
        // Create wire as a function of other existing synth operations
        createSynthWire(loc, nodeNames, function, builder);
      } else {
        // Create logic gate as a function of other existing synth operations
        createSynthLogicGate(loc, nodeNames, function, builder);
      }
    }

    // Subcircuits. not used for now.
    else if (line.find(subcktNode) == 0) {
      llvm::errs() << "Subcircuits inside the blif file not supported " << "\n";
      continue;
    }

    // Ends the file.
    else if (line.find(endNode) == 0) {
      break;
    }
  }
  // Close the file
  file.close();

  // Collect the outputs of the synth circuit in a vector to pass back to the
  // caller
  // Map hw instance outputs to the corresponding synth signals
  for (const auto &outputNodeName : synthOutputsNames) {
    if (!nodeValuesMap.count(outputNodeName)) {
      llvm::errs() << "Output node '" << outputNodeName
                   << "' not found in synth circuit." << "\n";
      return failure();
    }
    Value outputSignal = nodeValuesMap[outputNodeName];
    synthOutputs.push_back(outputSignal);
  }

  return success();
}

// Function to get the input mapping of a signal value
Value BlifImporter::getInputMappingSynthSignal(Location loc, StringRef nodeName,
                                               OpBuilder &builder) {
  assert(builder.getInsertionBlock() && "Builder has no insertion block!");
  // Check if the signal is already in the map
  if (nodeValuesMap.count(nodeName.str())) {
    return nodeValuesMap[nodeName.str()];
  }
  // Check if the signal is in the tmp values map
  if (tmpValuesMap.count(nodeName.str())) {
    return tmpValuesMap[nodeName.str()];
  }
  // If not, create a temporary input signal (e.g., hw constant)
  auto tempInput = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerType(1),
      builder.getIntegerAttr(builder.getIntegerType(1), 0));

  // Add the temporary input signal to the map
  tmpValuesMap[nodeName.str()] = tempInput;
  return tempInput;
}

// Function to update the output signal mapping after creating a new synth
// operation
void BlifImporter::updateOutputSynthSignalMapping(Value newResult,
                                                  StringRef nodeName,
                                                  OpBuilder &builder) {
  // Check if the old result is in the tmp values map
  if (tmpValuesMap.count(nodeName.str())) {
    // Replace all uses of the temporary value with the new result value
    tmpValuesMap[nodeName.str()].replaceAllUsesWith(newResult);
    auto *constOp = tmpValuesMap[nodeName.str()].getDefiningOp();
    assert(constOp && "temporary value should have a defining operation");
    assert(constOp->use_empty() && "temporary value should have no more uses");
    // Erase the operation
    constOp->erase();
    // Remove the temporary value from the map
    tmpValuesMap.erase(nodeName.str());
  }
  // Add the new result value to the node values map
  nodeValuesMap[nodeName.str()] = newResult;
}

// Function to create a wire in function of synth logic gate operations based
// on .names definition
void BlifImporter::createSynthWire(Location loc,
                                   std::vector<std::string> &ports,
                                   std::string &function, OpBuilder &builder) {
  assert(builder.getInsertionBlock() && "Builder has no insertion block!");
  unsigned numInputs = ports.size() - 1;
  assert(numInputs == 1 && "wire should have only one input");
  SmallVector<Value> inputValues;
  // Get input values
  for (unsigned i = 0; i < numInputs; ++i) {
    StringRef inputNodeName = ports[i];
    Value inputValue = getInputMappingSynthSignal(loc, inputNodeName, builder);
    inputValues.push_back(inputValue);
  }
  StringRef outputNodeName = ports.back();
  // Elaborate function
  assert(function.size() == 3 &&
         "function string must be of size 3 for a wire");
  char inputValueFunc = function[0];
  char outputValueFunc = function[2];
  assert((inputValueFunc == '0' || inputValueFunc == '1') &&
         "input value must be 0 or 1");
  assert((outputValueFunc == '0' || outputValueFunc == '1') &&
         "output value must be 0 or 1");
  int inputBit = (inputValueFunc == '1') ? 1 : 0;
  int outputBit = (outputValueFunc == '1') ? 1 : 0;
  Value inputValue = inputValues[0];
  // If the input and output bits are identical, you can skip the creation
  // of any node since they are identical
  if (inputBit == outputBit) {
    // We just need to update the output map function so that the output
    // points to the same value as input
    updateOutputSynthSignalMapping(inputValue, outputNodeName, builder);
    return;
  }
  // If this is not the case, we have to create a new aig node which inverts
  // the value of the input and receives one as another input Create constant
  // node 1
  auto constOp = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerType(1),
      builder.getIntegerAttr(builder.getIntegerType(1), 1));

  // Create aig node
  auto aigOp = builder.create<synth::AndInverterOp>(
      loc, inputValue, constOp.getResult(),
      /*invertInput0=*/true, /*invertInput1=*/false);

  updateOutputSynthSignalMapping(aigOp.getResult(), outputNodeName, builder);
}

// Function to create synth logic gate operations based on .names definition
void BlifImporter::createSynthLogicGate(Location loc,
                                        std::vector<std::string> &ports,
                                        std::string &function,
                                        OpBuilder &builder) {
  assert(builder.getInsertionBlock() && "Builder has no insertion block!");
  unsigned numInputs = ports.size() - 1;
  assert(numInputs > 1 && "logic gate must have at least two inputs");
  assert(numInputs == 2 && "the following pass transforms the logic gates to "
                           "aig nodes only");
  SmallVector<Value> inputValues;
  // Get input values
  for (unsigned i = 0; i < numInputs; ++i) {
    StringRef inputNodeName = ports[i];
    Value inputValue = getInputMappingSynthSignal(loc, inputNodeName, builder);
    inputValues.push_back(inputValue);
  }
  StringRef outputNodeName = ports.back();

  // Elaborate function to understand inversion of inputs/outputs
  // Get the value of the first and second input and output
  assert(function.size() == 4 &&
         "function string must be of size 4 for 2-input logic gate");
  char firstInputValue = function[0];
  char secondInputValue = function[1];
  char outputValue = function[3];
  assert((firstInputValue == '0' || firstInputValue == '1') &&
         "first input value must be 0 or 1");
  assert((secondInputValue == '0' || secondInputValue == '1') &&
         "second input value must be 0 or 1");
  assert((outputValue == '0' || outputValue == '1') &&
         "output value must be 0 or 1");
  int firstInputBit = (firstInputValue == '1') ? 1 : 0;
  int secondInputBit = (secondInputValue == '1') ? 1 : 0;
  int outputBit = (outputValue == '1') ? 1 : 0;

  bool invertInput0 = (firstInputBit == 0) ? true : false;
  bool invertInput1 = (secondInputBit == 0) ? true : false;

  // Create aig node
  auto aigOp = builder.create<synth::AndInverterOp>(
      loc, inputValues[0], inputValues[1],
      /*invertInput0=*/invertInput0, /*invertInput1=*/invertInput1);

  Value outputValueAig = aigOp.getResult();

  // Invert output if output is 0 by creating a new aig node with one input as
  // the output of the previous aig node and the other input as constant 1 and
  // inverting the first input
  if (outputBit == 0) {
    // Create constant node 1
    auto constOp = builder.create<hw::ConstantOp>(
        loc, builder.getIntegerType(1),
        builder.getIntegerAttr(builder.getIntegerType(1), 1));

    // Create aig node
    auto aigOpInvertedOutput = builder.create<synth::AndInverterOp>(
        loc, aigOp.getResult(), constOp.getResult(),
        /*invertInput0=*/true, /*invertInput1=*/false);
    outputValueAig = aigOpInvertedOutput.getResult();
  }

  // Update output mapping
  updateOutputSynthSignalMapping(outputValueAig, outputNodeName, builder);
}

// Function to create a new synth circuit from a blif file from an empty IR
void importBlifCircuit(ModuleOp moduleOp, Location loc,
                       StringRef blifFilePath) {
  MLIRContext *ctx = moduleOp.getContext();
  // Create a new module operation
  OpBuilder builder(ctx);
  // Set as insertion point the beginning of the module body to create the synth
  // circuit inside the module body and before any other operation
  builder.setInsertionPointToStart(moduleOp.getBody());
  BlifImporter blifImporter(blifFilePath);
  // Read input and output ports from the blif file
  if (failed(blifImporter.extractBlifModuleHeader())) {
    llvm::errs() << "Failed to read the blif file header." << "\n";
    return;
  }
  std::string moduleName = blifImporter.getModuleName();
  SmallVector<std::string> inputPortsNames = blifImporter.getInputPortsNames();
  SmallVector<std::string> outputPortsNames =
      blifImporter.getOutputPortsNames();
  // Create array of port info to create the hw module operation
  SmallVector<hw::PortInfo> portInfos;
  for (auto [idx, inputPortName] : llvm::enumerate(inputPortsNames)) {
    portInfos.push_back(hw::PortInfo{hw::ModulePort{
        StringAttr::get(ctx, inputPortName), builder.getIntegerType(1),
        hw::ModulePort::Direction::Input}});
  }
  for (auto [idx, outputPortName] : llvm::enumerate(outputPortsNames)) {
    portInfos.push_back(hw::PortInfo{hw::ModulePort{
        StringAttr::get(ctx, outputPortName), builder.getIntegerType(1),
        hw::ModulePort::Direction::Output}});
  }
  // Module name
  StringAttr moduleNameAttr = builder.getStringAttr(moduleName);
  // Create a new hw module operation
  hw::HWModuleOp hwModule = builder.create<hw::HWModuleOp>(
      loc, moduleNameAttr, ArrayRef<hw::PortInfo>(portInfos));

  // Collect the inputs of the hw module in a map to pass to the function that
  // generates the synth circuit
  llvm::StringMap<Value> synthInputs;
  unsigned inputIndex = 0;
  for (auto port : hwModule.getPortList()) {
    if (!port.isInput()) {
      continue;
    }
    StringRef portName = port.name.getValue();
    // Get value of hw module input argument
    Value portValue = hwModule.getModuleBody().getArgument(inputIndex);
    inputIndex++;
    // Add the mapping between the input port name and its value to the map
    synthInputs[portName.str()] = portValue;
  }
  builder.setInsertionPointToStart(hwModule.getBodyBlock());

  // Get hwmodule location to pass it to the function that generates the synth
  // circuit so that the created synth operations have the same location as the
  // hw module
  Location hwModuleloc = hwModule.getLoc();
  // Collect the outputs of the synth circuit
  SmallVector<Value> synthOutputs;
  // Generate the synth circuit from the blif file
  if (failed(blifImporter.generateSynthCircuitFromBlif(
          hwModuleloc, synthInputs, outputPortsNames, synthOutputs, builder))) {
    llvm::errs() << "Failed to generate the synth circuit from the blif file."
                 << "\n";
    return;
  }
  // Erase the default hw module terminator and create a new one with the synth
  // outputs as operands
  hwModule.getBodyBlock()->getTerminator()->erase();
  // Set the insertion point at the end of the hw module body
  builder.setInsertionPointToEnd(hwModule.getBodyBlock());
  // Create hw module terminator operation with the synth outputs as operands
  builder.create<hw::OutputOp>(hwModuleloc, synthOutputs);
}

} // namespace dynamatic