//===- CSVParser.cpp - Parses a csv transition line -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the methods needed to extract data from a transition csv
// file.
//
//===----------------------------------------------------------------------===//

#include "CSVParser.h"
#include "Graph.h"
#include "mlir/Support/LogicalResult.h"
#include <iostream>

using namespace mlir;
using namespace dynamatic::experimental::visual_dataflow;

/// This function transforms a state of type 'string' into its corresponding
/// state in type 'State'.
static LogicalResult findState(const std::string &stateString, State &state) {
  if (stateString == "undefined") {
    state = UNDEFINED;
    return success();
  }
  if (stateString == "accept") {
    state = ACCEPT;
    return success();
  }
  if (stateString == "idle") {
    state = IDLE;
    return success();
  }
  if (stateString == "stall") {
    state = STALL;
    return success();
  }
  if (stateString == "transfer") {
    state = TRANSFER;
    return success();
  }
  // Error unknown state for the edge.
  return failure();
}

static std::string trim(const std::string &str) {
  std::string s = str;

  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));

  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          s.end());

  return s;
}

LogicalResult dynamatic::experimental::visual_dataflow::processCSVLine(
    const std::string &line, size_t lineIndex, Graph &graph,
    CycleNb *currCycle) {

  // Jump the first line which contain column names
  if (lineIndex == 0 || line.empty())
    return success();

  std::string token;

  auto parseTokenString = [&](std::istringstream &iss,
                              std::string &value) -> ParseResult {
    std::getline(iss, token, ',');
    if (token.empty())
      return failure();
    value = trim(token);
    return success();
  };

  auto parseTokenInt = [&](std::istringstream &iss, int &value) -> ParseResult {
    std::getline(iss, token, ',');
    if (token.empty())
      return failure();
    value = std::stoi(token);
    return success();
  };

  std::istringstream iss(line);

  // Parse all 7 columns
  int cycleNb, inPort, outPort;
  std::string src, dst, stateString, data;

  if (parseTokenInt(iss, cycleNb) || parseTokenString(iss, src) ||
      parseTokenInt(iss, outPort) || parseTokenString(iss, dst) ||
      parseTokenInt(iss, inPort) || parseTokenString(iss, stateString) ||
      parseTokenString(iss, data)) {
    return failure();
  }

  if (cycleNb != *currCycle) {
    graph.dupilcateEdgeStates(*currCycle, cycleNb);
    *currCycle = cycleNb;
  }

  std::pair<NodeId, size_t> srcInfo = std::pair(src, ++outPort);

  std::pair<NodeId, size_t> dstInfo = std::pair(dst, ++inPort);

  std::pair<std::pair<NodeId, size_t>, std::pair<NodeId, size_t>> info =
      std::pair(srcInfo, dstInfo);

  EdgeId edgeId;
  State state;

  if (failed(graph.getEdgeId(info, edgeId)) ||
      failed(findState(stateString, state))) {
    return failure();
  }

  graph.addEdgeState(cycleNb, edgeId, state, data);

  return success();
}
