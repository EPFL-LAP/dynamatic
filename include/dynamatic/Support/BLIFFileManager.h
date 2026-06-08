//===- BLIFFileManager.h - Support functions for BLIF importer -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the function to retrieve the path of the blif file for a
// given handshake operation. The path is created by combining the base path
// contining all the blif files and the operation name and parameters. The file
// is expected to be named in the format <op_name>_<param1>_<param2>_..._.blif.
// If the file does not exist, an error is emitted.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <filesystem>
#include <regex>

namespace dynamatic {

class BLIFFileManager {
public:
  // Constructor for the BLIFFileManager class
  BLIFFileManager(std::string blifDirPath) : blifDirPath(blifDirPath) {}

  // Function to combine parameter values, module type and blif directory path
  // to create the blif file path
  std::string combineBlifFilePath(std::string moduleType,
                                  std::vector<std::string> paramValues,
                                  std::string extraSuffix = "");

  // Function to retrieve the path of the blif file for a given handshake
  // operation
  std::string getBlifFilePathForHandshakeOp(mlir::Operation *op);

private:
  // String containing the base path of the blif files
  std::string blifDirPath;
};

// Formats a bit-indexed port name: "sig[bit]".
// When width == 1 the name is returned unchanged (no "[0]" suffix).
// If baseName already ends with "[N]" the indices are linearised.
std::string legalizeDataPortName(mlir::StringRef baseName, unsigned bit,
                                 unsigned width);

// Inserts a suffix string before the first '[' in the input name, or appends
// it at the end if no '[' is found.
// Example: legalizeControlPortName("data[3]", "_valid") -> "data_valid[3]"
std::string legalizeControlPortName(const std::string &name,
                                    const std::string &suffix);

// Converts a (root, index) pair into the canonical "root[index]" form.
// If root already contains "[N]", the old index is linearised with
//  arrayWidth before adding index.
// Example: formatArrayName("data[2]", 3, 4) becomes "data[11]"  (2*4 + 3)
std::string formatArrayName(const std::string &root, unsigned index,
                            unsigned arrayWidth = 0);

// Legalizes a list of handshake port names by converting the "root_N" index
// pattern (used internally by NamedIOInterface) into the "root[N]" array
// notation expected by the BLIF importer
// Example: ["data_0", "data_1", "valid"] -> ["data[0]", "data[1]", "valid"]
void legalizeBlifPortNames(mlir::SmallVector<std::string> &names);

} // namespace dynamatic