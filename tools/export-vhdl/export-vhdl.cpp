//===- export-vhdl.cpp - Export DOT to VHDL ---------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Legacy dot2vhdl tool, somewhat cleaned up.
//
//===----------------------------------------------------------------------===//

#include "src/DOTParser.h"
#include "src/LSQGenerator.h"
#include "src/VHDLWriter.h"
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  // Check that the corrent number of arguments were provided
  if (argc != 4) {
    std::cerr << "Expected three arguments: kernel name, path to DOT file to "
                 "convert, and path to output folder\n";
    return 1;
  }
  std::string kernelName(argv[1]);
  std::string dotPath(argv[2]);
  std::string outPath(argv[3]);

  // Parse the DOT
  parseDOT(dotPath);

  // Generate the VHDL
  writeVHDL(kernelName, outPath);

  // Generate LSQ configurations
  lsqGenerateConfiguration(outPath);
  return 0;
}
