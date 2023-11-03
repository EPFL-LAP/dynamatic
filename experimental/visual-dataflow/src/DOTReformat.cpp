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
#include <string>
#include <vector>

using namespace mlir;

LogicalResult reformatDot(const std::string &inputFileName,
                          const std::string &outputFileName) {
  std::ifstream inputFile(inputFileName);
  if (!inputFile.is_open())
    return failure();

  std::ofstream outputFile(outputFileName);
  if (!outputFile.is_open()) {
    inputFile.close();
    return failure();
  }

  std::vector<std::string> lines;
  std::string line;

  // Load entire file into memory (lines vector)
  while (std::getline(inputFile, line))
    lines.push_back(line);
  inputFile.close();

  // Single pass processing
  for (size_t i = 0; i < lines.size(); ++i) {
    // Process 1: putPosOnSameLine
    if (lines[i].find("pos=") != std::string::npos &&
        lines[i].find('e') != std::string::npos) {
      while (i + 1 < lines.size() &&
             lines[i + 1].find("];") == std::string::npos) {
        lines[i] += " " + lines[i + 1];
        lines.erase(lines.begin() + i + 1);
      }
    }

    // Process 2: insertNewlineBeforeStyle
    size_t stylePos = lines[i].find("style=");
    if (lines[i].find("pos=") != std::string::npos &&
        stylePos != std::string::npos) {
      lines[i].insert(stylePos, "\n");
    }

    // Process 3: removeBackslashWithSpaceFromPos
    size_t posPos = lines[i].find("pos=");
    size_t backslashPos = lines[i].find("\\ ");
    if (posPos != std::string::npos && backslashPos != std::string::npos &&
        posPos < backslashPos) {
      lines[i] =
          lines[i].substr(0, backslashPos) + lines[i].substr(backslashPos + 2);
    }

    // Process 4: removeEverythingAfterCommaInStyle
    size_t commaPos = lines[i].find(',');
    if (stylePos != std::string::npos && commaPos != std::string::npos &&
        stylePos < commaPos) {
      lines[i] = lines[i].substr(0, commaPos);
    }

    // Process 5: removeEverythingAfterApostropheComma
    size_t apostropheCommaPos = lines[i].find("\",");
    if (posPos != std::string::npos &&
        apostropheCommaPos != std::string::npos &&
        posPos < apostropheCommaPos) {
      lines[i] = lines[i].substr(0, apostropheCommaPos + 2);
    }
  }

  // Write to output file
  for (const auto &l : lines) {
    outputFile << l << '\n';
  }

  outputFile.close();

  return success();
}
