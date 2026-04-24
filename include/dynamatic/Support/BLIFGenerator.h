//===- BLIFGenerator.h - On-demand BLIF file generator ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates a BLIF file for a given handshake component and parameter values
// by running Yosys (synthesis) followed by ABC (optimization), replicating the
// logic of tools/blif-generator/blif_generator.py in C++.
//
// The JSON configuration file (rtl-config-verilog.json) is read to
// determine the Verilog source files and the names of Yosys chparam parameters
// that correspond to the provided parameter values.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>
#include <vector>

namespace dynamatic {

class BLIFGenerator {
public:
  BLIFGenerator(const std::string &blifDirPath,
                const std::string &yosysExecutable,
                const std::string &abcExecutable);

  /// Generate the BLIF file for the given component and parameter values.
  ///
  /// The component is the full component name used in the BLIF directory tree,
  /// e.g. "addi", "fork_dataless", "fork_type", "oehb".  It matches the
  /// module-name field of the JSON configuration entry, or (if that field is
  /// absent) the operation name with the "handshake." prefix stripped.
  ///
  /// The paramValues contain the ordered parameter values that correspond to
  /// the generic (range) parameters declared in the JSON entry.
  ///
  /// Returns true when the BLIF file was successfully created.
  bool generate(const std::string &component,
                const std::vector<std::string> &paramValues);

private:
  std::string
      blifDirPath; /// Absolute path to the directory that holds all BLIF
                   /// files (e.g. "<repo>/data/blif").  The JSON config
                   /// and the Dynamatic root are derived from this path.
  std::string dynamaticRoot; /// Absolute path to the Dynamatic root directory,
                             /// derived from blifDirPath.
  std::string yosysExecutable; /// Path to the Yosys binary.
  std::string abcExecutable;   /// Path to the ABC binary.

  /// Expand the given path by replacing "$DYNAMATIC" with the Dynamatic root.
  std::string expandPath(const std::string &path) const;

  /// Write run_yosys.sh inside the output directory and execute it.
  bool runYosys(const std::string &topModule, const std::string &outputDir,
                const std::string &chparam,
                const std::vector<std::string> &verilogFiles,
                const std::string &outputName) const;

  /// Write run_abc.sh inside the output directory and execute it.
  bool runAbc(const std::string &inputFile, const std::string &outputDir,
              const std::string &outputName) const;
};

} // namespace dynamatic
