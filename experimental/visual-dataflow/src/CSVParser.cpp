//===- CSVParser.cpp - Parses a csv transition line ------------*- C++ -*-===//
//
// This file implements the methids needed to extract data from a transition csv
// file.
//===----------------------------------------------------------------------===//

#include "CSVParser.h"

using namespace mlir;
using namespace dynamatic::experimental::visual_dataflow;

/// This function transforms a state of type 'string' into its corresponding
/// state in type 'State'.
LogicalResult findState(const std::string &stateString, State &state) {
  if (stateString == "undefined") {
    state = UNDEFINED;
    return success();
  }
  if (stateString == "ready") {
    state = READY;
    return success();
  }
  if (stateString == "empty") {
    state = EMPTY;
    return success();
  }
  if (stateString == "valid") {
    state = VALID;
    return success();
  }
  if (stateString == "valid_ready") {
    state = VALID_READY;
    return success();
  }
  // Error unknown state for the edge.
  return failure();
}

/// This function finds an edge in the graph thanks to given information about
/// the edge.
LogicalResult findEdgeInGraph(Graph *graph,
                              const std::vector<std::string> &edgeInfo,
                              EdgeId &edgeId) {

  std::pair<NodeId, int> srcInfo =
      std::pair(edgeInfo[1], std::stoi(edgeInfo[2]));

  std::pair<NodeId, int> dstInfo =
      std::pair(edgeInfo[3], std::stoi(edgeInfo[4]));

  std::pair<std::pair<NodeId, int>, std::pair<NodeId, int>> info =
      std::pair(srcInfo, dstInfo);

  // Error : an edge defined in the csv file was not found in the graph.
  return graph->getEdgeId(info, edgeId);
}

std::string trim(const std::string &str) {
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

LogicalResult processLine(const std::string &line, Graph *graph,
                          size_t lineIndex) {

  // Jump the first 2 lines and empty line (Not needed)
  if (lineIndex == 0 || lineIndex == 1 || line.empty())
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

  // Parse all 6 columns
  int cycleNb, inPort, outPort;
  std::string src, dst, stateString;

  if (parseTokenInt(iss, cycleNb) || parseTokenString(iss, src) ||
      parseTokenInt(iss, outPort) || parseTokenString(iss, dst) ||
      parseTokenInt(iss, inPort) || parseTokenString(iss, stateString)) {
    return failure();
  }

  std::pair<NodeId, int> srcInfo = std::pair(src, outPort);

  std::pair<NodeId, int> dstInfo = std::pair(dst, inPort);

  std::pair<std::pair<NodeId, int>, std::pair<NodeId, int>> info =
      std::pair(srcInfo, dstInfo);

  EdgeId edgeId;

  if (failed(graph->getEdgeId(info, edgeId))) {
    return failure();
  }

  State state;

  if (failed(findState(stateString, state))) {
    return failure();
  }

  graph->addEdgeState(cycleNb, edgeId, state);

  return success();
}