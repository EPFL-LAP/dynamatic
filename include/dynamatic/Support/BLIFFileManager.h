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