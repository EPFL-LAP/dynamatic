//===- BLIFGenerator.cpp - On-demand BLIF file generator --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Synthesizes a Verilog component to BLIF via Yosys, then optimizes the result
// with ABC. Verilog generation is delegated to RTLGenerator.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/BLIFGenerator.h"
#include "dynamatic/Support/RTLGenerator.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <fstream>

using namespace dynamatic;

BLIFGenerator::BLIFGenerator(const std::string &blifDirPath,
                             const std::string &dynamaticRootPath,
                             const std::string &RTLJSONFile,
                             const std::string &yosysExecutable,
                             const std::string &abcExecutable,
                             mlir::Operation *op,
                             const std::string &expectedBlifPath)
    : blifDirPath(blifDirPath), dynamaticRoot(dynamaticRootPath),
      RTLJSONFile(RTLJSONFile), yosysExecutable(yosysExecutable),
      abcExecutable(abcExecutable), op(op), expectedBlifPath(expectedBlifPath) {
}

bool BLIFGenerator::generate() {
  namespace fs = std::filesystem;

  std::string opName = op->getName().getStringRef().str();
  std::string opShortName = opName.substr(opName.find('.') + 1);

  RTLGenerator rtlGen(RTLJSONFile, dynamaticRoot, blifDirPath, op);
  if (!rtlGen.generate()) {
    llvm::errs() << "BLIFGenerator: RTL generation failed for `" << opName
                 << "`\n";
    return false;
  }

  std::string yosysOutName = rtlGen.getModuleName() + "_yosys.blif";
  if (!runYosys(rtlGen.getModuleName(), rtlGen.getOutputDir(),
                rtlGen.getVerilogFiles(), yosysOutName)) {
    llvm::errs() << "BLIFGenerator: Yosys failed for `" << opName << "`\n";
    return false;
  }

  std::string yosysOutPath =
      (fs::path(rtlGen.getOutputDir()) / yosysOutName).string();
  std::string abcOutName = opShortName + ".blif";
  if (!runAbc(yosysOutPath, rtlGen.getOutputDir(), abcOutName)) {
    llvm::errs() << "BLIFGenerator: ABC failed for `" << opName << "`\n";
    return false;
  }

  return fs::exists(fs::path(rtlGen.getOutputDir()) / abcOutName);
}

bool BLIFGenerator::runYosys(const std::string &topModule,
                             const std::string &outputDir,
                             const std::vector<std::string> &verilogFiles,
                             const std::string &outputName) const {
  std::string scriptPath = outputDir + "/run_yosys.sh";
  std::ofstream script(scriptPath);
  if (!script.is_open())
    return false;

  script << "#!/bin/bash\n";
  script << yosysExecutable << " -p \"";
  for (const auto &f : verilogFiles)
    script << "read_verilog -defer " << f << ";\n        ";
  script << "hierarchy -top " << topModule << ";\n";
  script << "        proc;\n";
  script << "        opt -nodffe -nosdff;\n";
  script << "        memory -nomap;\n";
  script << "        techmap;\n";
  script << "        flatten;\n";
  script << "        clean;\n";
  script << "        write_blif " << outputDir << "/" << outputName
         << "\" > /dev/null 2>&1\n";
  script.close();

  std::filesystem::permissions(scriptPath,
                               std::filesystem::perms::owner_exec |
                                   std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::add);

  return system(("bash " + scriptPath).c_str()) == 0;
}

bool BLIFGenerator::runAbc(const std::string &inputFile,
                           const std::string &outputDir,
                           const std::string &outputName) const {
  std::string scriptPath = outputDir + "/run_abc.sh";
  std::ofstream script(scriptPath);
  if (!script.is_open())
    return false;

  script << "#!/bin/bash\n";
  script << abcExecutable << " -c \"read_blif " << inputFile << ";\n";
  script << "        strash;\n";
  for (int i = 0; i < 6; ++i) {
    script << "        rewrite;\n";
    script << "        b;\n";
    script << "        refactor;\n";
    script << "        b;\n";
  }
  script << "        write_blif " << outputDir << "/" << outputName
         << "\" > /dev/null 2>&1\n";
  script.close();

  std::filesystem::permissions(scriptPath,
                               std::filesystem::perms::owner_exec |
                                   std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::add);

  return system(("bash " + scriptPath).c_str()) == 0;
}
