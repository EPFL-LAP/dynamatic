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
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  // Check that the corrent number of arguments were provided
  if (argc != 4) {
    std::cerr << "Expected three arguments: kernel name, path to DOT file to "
                 "convert, and path to VHD file to write\n";
    return 1;
  }

  // Parse the DOT
  std::string dotPath = argv[2];
  std::cout << "Parsing " << dotPath << "\n";
  parseDOT(dotPath);

  // Generate the VHDL
  std::cout << "Generating VHDL\n";
  writeVHDL(argv[1], argv[3]);

  // Generate LSQ configurations
  lsqGenerateConfiguration(dotPath);
  lsqGenerate(dotPath);

  std::cout << "Done\n";
  return 0;
}
