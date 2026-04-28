//===- BLIFGenerator.h - On-demand BLIF file generator ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates a BLIF file for a given Handshake operation by running Yosys
// (synthesis) followed by ABC (optimization).
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_BLIFGENERATOR_H
#define DYNAMATIC_SUPPORT_BLIFGENERATOR_H

#include "mlir/IR/Operation.h"
#include <string>
#include <vector>

namespace dynamatic {

class BLIFGenerator {
public:
  BLIFGenerator(const std::string &blifDirPath,
                const std::string &yosysExecutable,
                const std::string &abcExecutable, mlir::Operation *op,
                const std::string &expectedBlifPath);

  /// Generate the BLIF file for the operation supplied at construction.
  ///
  /// If the expected BLIF file already exists, this function does nothing and
  /// returns true. Otherwise, it runs the RTL generator, Yosys, and ABC in
  /// sequence to produce the BLIF file.
  ///
  /// Returns true when the BLIF file was successfully created.
  bool generate();

private:
  std::string blifDirPath;      /// Absolute path to the BLIF directory.
  std::string dynamaticRoot;    /// Dynamatic root, derived from blifDirPath.
  std::string yosysExecutable;  /// Path to the Yosys binary.
  std::string abcExecutable;    /// Path to the ABC binary.
  mlir::Operation *op;          /// The operation to synthesize.
  std::string expectedBlifPath; /// Expected output path of the final BLIF.

  /// Write run_yosys.sh inside outputDir and execute it.
  bool runYosys(const std::string &topModule, const std::string &outputDir,
                const std::vector<std::string> &verilogFiles,
                const std::string &outputName) const;

  /// Write run_abc.sh inside outputDir and execute it.
  bool runAbc(const std::string &inputFile, const std::string &outputDir,
              const std::string &outputName) const;
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_BLIFGENERATOR_H