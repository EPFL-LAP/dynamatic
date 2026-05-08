//===- BLIFFileManager.h - Support functions for BLIF importer -*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Retrieves the BLIF file path for a given Handshake operation by combining
// the base BLIF directory, the operation name, and its RTL parameter values.
// If the file does not exist and Yosys+ABC are configured, it is generated
// on demand via BackendGenerator.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_BLIFFILEMANAGER_H
#define DYNAMATIC_SUPPORT_BLIFFILEMANAGER_H

#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace dynamatic {

class BLIFFileManager {
public:
  BLIFFileManager(std::string blifDirPath, std::string dynamaticRootPath,
                  std::string rtlJsonFile)
      : blifDirPath(std::move(blifDirPath)),
        dynamaticRootPath(std::move(dynamaticRootPath)),
        rtlJsonFile(std::move(rtlJsonFile)) {}

  std::string combineBlifFilePath(const std::string &moduleType,
                                  const std::vector<std::string> &paramValues,
                                  const std::string &extraSuffix = "");

  std::string getBlifFilePathForHandshakeOp(mlir::Operation *op);

private:
  std::string blifDirPath;
  std::string dynamaticRootPath;
  std::string rtlJsonFile;
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
// arrayWidth before adding index.
// Example: formatArrayName("data[2]", 3, 4) becomes "data[11]"  (2*4 + 3)
std::string formatArrayName(const std::string &root, unsigned index,
                            unsigned arrayWidth = 0);

// Legalizes a list of handshake port names by converting the "root_N" index
// pattern (used internally by NamedIOInterface) into the "root[N]" array
// notation expected by the BLIF importer.
// Example: ["data_0", "data_1", "valid"] -> ["data[0]", "data[1]", "valid"]
void legalizeBlifPortNames(llvm::SmallVector<std::string> &names);

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_BLIFFILEMANAGER_H
