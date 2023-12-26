//===- DOTReformat.cpp - Reformat DOT files before parsing ------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reformat DOT files.
//
//===----------------------------------------------------------------------===//

#include "DOTReformat.h"
#include "mlir/Support/LogicalResult.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace mlir;

LogicalResult reformatDot(const std::string &inputFile,
                          const std::string &outputFile) {
  std::ifstream inFile(inputFile);
  std::ofstream outFile(outputFile);
  std::string line;
  std::ostringstream accumulatedLine;
  bool isEdge = false;
  bool isPosLine = false;

  if (!inFile.is_open() || !outFile.is_open()) {
    std::cerr << "Error opening files!" << std::endl;
    return failure();
  }

  while (getline(inFile, line)) {
    // Check if we are within an edge definition
    if (line.find("->") != std::string::npos) {
      isEdge = true;
    }

    if (isEdge) {
      // Check if this line contains the beginning of a pos attribute
      if (line.find("pos=\"") != std::string::npos) {
        isPosLine = true;
        accumulatedLine.str("");
        accumulatedLine.clear();
        accumulatedLine << line;
        // If the line does not end with a backslash, it's a single-line pos
        // attribute
        if (line.back() != '\\') {
          isPosLine = false;
          outFile << accumulatedLine.str() << std::endl;
        }
        continue;
      }

      // If currently processing a pos attribute
      if (isPosLine) {
        // Remove trailing backslash and newline character
        if (accumulatedLine.str().back() == '\\') {
          accumulatedLine.seekp(-1, std::ios_base::end);
        }
        accumulatedLine << line;
        // Check if this is the end of the pos attribute
        if (line.back() != '\\') {
          isPosLine = false;
          outFile << accumulatedLine.str() << std::endl;
        }
      } else {
        // Normal line within an edge definition, write it to the output file
        outFile << line << std::endl;
      }

      // Check if the edge definition has ended
      if (line.find(";") != std::string::npos) {
        isEdge = false;
      }
    } else {
      // Outside of an edge definition, write the line as is
      outFile << line << std::endl;
    }
  }

  inFile.close();
  outFile.close();

  return success();
}
