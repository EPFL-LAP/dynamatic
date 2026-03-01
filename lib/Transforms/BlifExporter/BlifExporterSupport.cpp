//===- BlifExporterSupport.cpp - Support functions for BLIF exporter -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of support functions for the BLIF exporter
//
//===----------------------------------------------------------------------===//

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
#include "dynamatic/Transforms/BlifImporter/BlifImporterSupport.h"
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

// Function to generate the header of the blif file with the module name and the
// input and output ports
LogicalResult generateBlifHeader(hw::HWModuleOp hwModuleOp,
                                 llvm::raw_fd_ostream &outputFile,
                                 SmallVector<std::string> &iosNames) {
  // Get the name of the hwModuleOp to use it as the name of the blif file
  StringRef moduleName = hwModuleOp.getName();

  // Write the module name in the blif file
  outputFile << ".model " << moduleName.str() << "\n";

  // Write the input ports in the blif file
  outputFile << ".inputs";
  for (auto port : hwModuleOp.getPortList()) {
    if (port.isInput()) {
      outputFile << " " << port.getName().str();
      iosNames.push_back(port.getName().str());
    }
  }
  outputFile << "\n";

  // Write the output ports in the blif file
  outputFile << ".outputs";
  for (auto port : hwModuleOp.getPortList()) {
    if (port.isOutput()) {
      outputFile << " " << port.getName().str();
      iosNames.push_back(port.getName().str());
    }
  }
  outputFile << "\n";

  return success();
}

// Function to generate latches and logic gates in the blif file from the synth
// circuit inside the hw module
LogicalResult generateBlifCircuitFromSynth(hw::HWModuleOp hwModuleOp,
                                           llvm::raw_fd_ostream &outputFile,
                                           SmallVector<std::string> &iosNames) {
  // Build AsmState once for the whole module to get consistent value names
  mlir::AsmState asmState(hwModuleOp);

  // Keep a list of output names to check if there is any output value that is
  // just directly connected to an input port without any synth operation in
  // between, in which case we need to create a wire in the blif file to connect
  // the input and output ports
  SmallVector<std::string> synthOutputsNames;

  // Helper lambda to get the string ID of a value (e.g. "%4" or "%arg0")
  auto getValueName = [&](mlir::Value value) -> std::string {
    std::string name;
    // Check if the value is a block argument (input port)
    if (auto blockArg = dyn_cast<mlir::BlockArgument>(value)) {
      unsigned argIdx = blockArg.getArgNumber();
      if (argIdx < iosNames.size()) {
        return iosNames[argIdx];
      }
      llvm::errs() << "Block argument index " << argIdx
                   << " is out of bounds for iosNames.\n";
      return "invalid_arg";
    }
    // Check if the value is used as an input of the terminator of the hw module
    // (output port)
    for (auto &use : value.getUses()) {
      if (auto returnOp = dyn_cast<hw::OutputOp>(use.getOwner())) {
        unsigned operandIdx = use.getOperandNumber();
        unsigned numInputPorts = hwModuleOp.getNumInputPorts();
        // Output ports come after input ports in iosNames, so we need to add
        // the number of input ports to the operand index to get the correct
        // index in iosNames
        operandIdx += numInputPorts;
        if (operandIdx < iosNames.size()) {
          synthOutputsNames.push_back(iosNames[operandIdx]);
          return iosNames[operandIdx];
        }
        llvm::errs() << "Operand index " << operandIdx
                     << " is out of bounds for iosNames.\n";
        return "invalid_operand";
      }
    }
    llvm::raw_string_ostream os(name);
    value.printAsOperand(os, asmState);
    // If there is a % at the beginning of the name, remove it since blif does
    // not allow it in node names
    if (!name.empty() && name[0] == '%') {
      name.erase(0, 1);
    }
    // Add an `n` prefix to the name
    name = "n" + name;
    return name;
  };

  for (auto &op : hwModuleOp.getOps()) {
    if (isa<synth::LatchOp>(op)) {
      auto latchOp = dyn_cast<synth::LatchOp>(op);
      // .latch <input> <output> [<latch type> <control>] [<init value>]
      outputFile << ".latch";
      outputFile << " " << getValueName(latchOp.getOperand(0));
      outputFile << " " << getValueName(latchOp.getResult());
      // Check for optional fields: latch type, control and init value
      if (latchOp.getControl()) {
        std::string latchType = latchOp.getLatchType()
                                    ? latchOp.getLatchType()->str()
                                    : "re"; // default to rising edge
        outputFile << " " << latchType;
        outputFile << " " << getValueName(latchOp.getControl());
      }

      if (latchOp.getInitVal()) {
        outputFile << " " << latchOp.getInitVal().value();
      }
      outputFile << "\n";
    } else if (isa<synth::aig::AndInverterOp>(op)) {
      auto andOp = dyn_cast<synth::aig::AndInverterOp>(op);
      // .names <input1> <input2> <output>
      outputFile << ".names";
      outputFile << " " << getValueName(andOp.getOperand(0));
      outputFile << " " << getValueName(andOp.getOperand(1));
      outputFile << " " << getValueName(andOp.getResult());
      outputFile << "\n";
      // Build the truth table row: all inputs must be 1 (or 0 if inverted)
      // to produce output 1
      size_t numInputs = andOp.getNumOperands();
      for (size_t i = 0; i < numInputs; i++) {
        // Inverted input means the input must be 0 for the AND to be true
        outputFile << (andOp.isInverted(i) ? "0" : "1");
      }
      outputFile
          << " 1\n"; // output is 1 when all (possibly inverted) inputs satisfy
    } else if (isa<hw::ConstantOp>(op)) {
      auto constOp = dyn_cast<hw::ConstantOp>(op);
      // .names <output>
      APInt constValue = constOp.getValue();
      assert(constValue.getBitWidth() == 1 &&
             "only 1-bit constants supported in BLIF export");
      outputFile << ".names " << getValueName(constOp.getResult()) << "\n";
      outputFile << (constValue == 1 ? "1" : "0") << "\n";
    } else if (isa<hw::OutputOp>(op)) {
      // We will process the output ports at the end to check if there is any
      // output value that is just directly connected to an input port without
      // any synth operation in between, in which case we need to create a wire
      // in the blif file to connect the input and output ports
      continue;
    } else {
      // For now, we only support latches, AND gates and constants in the synth
      // circuit. We can extend this to other types of operations in the future.
      llvm::errs() << "Unsupported operation '" << op.getName()
                   << "' in synth circuit. Only latches, AND gates and "
                      "constants are supported for now.\n";
    }
  }

  // Check if there is any output value that is just directly connected to an
  // input port without any synth operation in between, in which case we need to
  // create a wire in the blif file to connect the input and output ports
  // Iterate through the inputs of the terminator of the hw module (output
  // ports) and check if any of them is directly connected to an input port
  auto terminator = hwModuleOp.getBody().front().getTerminator();
  for (int operandIdx = 0; operandIdx < terminator->getNumOperands();
       operandIdx++) {
    Value operand = terminator->getOperand(operandIdx);
    // Check if it is a block argument (input port)
    if (auto blockArg = dyn_cast<mlir::BlockArgument>(operand)) {
      unsigned argIdx = blockArg.getArgNumber();
      if (argIdx < iosNames.size()) {
        std::string inputPortName = iosNames[argIdx];
        // Assert it is not part of the synthOutputsNames variable
        assert(std::find(synthOutputsNames.begin(), synthOutputsNames.end(),
                         inputPortName) == synthOutputsNames.end() &&
               "The output port should have not been already processed in the "
               "synth circuit generation since it is directly connected to an "
               "input port without any synth operation in between");
        // Get the output port name corresponding to the operand
        std::string outputPortName =
            iosNames[operandIdx + hwModuleOp.getNumInputPorts()];
        // Write a .names statement to create a wire between the input and
        // output
        outputFile << ".names " << inputPortName << " " << outputPortName
                   << "\n";
        outputFile << "1 1\n"; // output is 1 when input is 1
        // Add the output port name to the synthOutputsNames variable to check
        // that all outputs are processed at the end
        synthOutputsNames.push_back(outputPortName);
      }
      assert(argIdx < iosNames.size() && "Block argument index out of bounds");
    }
  }
  // Assert that all output ports are processed
  for (auto &port : hwModuleOp.getPortList()) {
    if (port.isOutput()) {
      std::string outputPortName = port.getName().str();
      if (std::find(synthOutputsNames.begin(), synthOutputsNames.end(),
                    outputPortName) == synthOutputsNames.end()) {
        llvm::errs() << "Output port '" << outputPortName
                     << "' has not been generated." << "\n";
        return failure();
      }
    }
  }
  return success();
}

// Function to export the synth circuit inside an hwModuleOp to a blif file
LogicalResult exportBlifCircuit(hw::HWModuleOp hwModuleOp,
                                llvm::raw_fd_ostream &outputFile) {
  // Get the name of the hwModuleOp to use it as the name of the blif file
  StringRef moduleName = hwModuleOp.getName();

  SmallVector<std::string> iosNames;
  // Generate header with module name and input and output ports
  if (failed(generateBlifHeader(hwModuleOp, outputFile, iosNames))) {
    llvm::errs() << "Failed to generate the blif file header for module '"
                 << moduleName << "'.\n";
    return failure();
  }

  // Generate latches and logic gates of the blif file from the synth
  // circuit inside the hw module
  if (failed(generateBlifCircuitFromSynth(hwModuleOp, outputFile, iosNames))) {
    llvm::errs() << "Failed to generate the blif circuit for module '"
                 << moduleName << "'.\n";
    return failure();
  }
  // Write the end of the blif file
  outputFile << ".end\n";

  return success();
}

} // namespace dynamatic