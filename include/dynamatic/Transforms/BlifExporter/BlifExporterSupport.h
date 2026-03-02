//===- BlifExporterSupport.h - Support functions for BLIF exporter -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of support functions for the BLIF exporter
// pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/Synth/SynthDialect.h"
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

class BlifExporter {

public:
  // Method to create BlifExporter object with the hw module operation and the
  // output file stream as arguments
  BlifExporter(hw::HWModuleOp hwModuleOp, llvm::raw_fd_ostream &outputFile)
      : hwModuleOp(hwModuleOp), outputFile(outputFile) {}

  // Function to export the synth circuit inside an hwModuleOp to a blif file
  LogicalResult exportBlifCircuit();

  // Function to generate the header of the blif file with the module name and
  // the input and output ports
  LogicalResult generateBlifHeader();

  // Function to generate latches and logic gates in the blif file from the
  // synth circuit inside the hw module
  LogicalResult generateBlifCircuitFromSynth();

private:
  // HW module operation to be exported as a blif file
  hw::HWModuleOp hwModuleOp;
  // Output file stream to write the blif content
  llvm::raw_fd_ostream &outputFile;
  // Vector containing the input ports of the blif circuit
  SmallVector<std::string> inputPorts;
  // Vector containing the output ports of the blif circuit
  SmallVector<std::string> outputPorts;
};

} // namespace dynamatic