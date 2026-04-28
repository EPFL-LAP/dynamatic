//===- RTLGenerator.h - RTL code generator from JSON config -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates a Verilog file for a given Handshake operation by looking up its
// entry in a JSON RTL config, substituting RTL parameters, and running the
// configured generator command.
//
// Parameters are extracted from the operation via RTLAttrInterface
// (getRTLParameters()). Results (module name, output directory, Verilog file
// list) are available via getters after a successful generate() call.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_RTLGENERATOR_H
#define DYNAMATIC_SUPPORT_RTLGENERATOR_H

#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace dynamatic {

class RTLGenerator {
public:
  RTLGenerator(const std::string &rtlConfigPath,
               const std::string &dynamaticRoot,
               const std::string &outputBaseDir, mlir::Operation *op);

  /// Runs the RTL generator command found in the JSON config.
  ///
  /// Extracts parameters from the operation via RTLAttrInterface, locates the
  /// matching config entry, substitutes all variables into the generator
  /// command, and executes it. On success the generated and support Verilog
  /// files are collected and made available via getters.
  ///
  /// Returns true when RTL generation succeeded.
  bool generate();

  const std::string &getModuleName() const { return moduleName; }
  const std::string &getOutputDir() const { return outputDir; }
  const std::vector<std::string> &getVerilogFiles() const {
    return verilogFiles;
  }

private:
  std::string rtlConfigPath; /// Absolute path to the JSON RTL config file.
  std::string dynamaticRoot; /// Project root used to resolve $DYNAMATIC paths
                             /// in the config.
  std::string outputBaseDir; /// Base directory under which the per-module
                             /// output subdirectory will be created.
  mlir::Operation *op;       /// The Handshake operation to generate RTL for.

  std::string moduleName; /// The generated module name, derived from the
                          /// operation name and parameters.
  std::string outputDir;  /// The output directory for generated files, derived
                         /// from outputBaseDir, operation name, and parameters.
  std::vector<std::string>
      verilogFiles; /// List of generated and support Verilog files to be
                    /// synthesized, including the main generated file
                    /// (outputDir/moduleName.v) and any additional files
                    /// specified in the config with "generic" paths.
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_RTLGENERATOR_H