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

/// This function parses a line from a csv file.
std::vector<std::string> parseOneLine(const std::string &line) {

  // index 0 : cycle number, index 1 : src_component, index 2 : src_port, index
  // 3 : dst_component, index 4 : dst_port, index 5 : status.
  std::vector<std::string> edgeInfo(6);

  size_t startIndex = 0;
  size_t commaIndex;

  for (int i = 0; i < 6; ++i) {

    commaIndex = line.find(',', startIndex);

    if (commaIndex == std::string::npos) {
      // No comma found : it's the last part of the string
      commaIndex = line.size();
    }

    edgeInfo[i] = line.substr(startIndex, commaIndex - startIndex);

    startIndex =
        commaIndex + 2; // +2 to skip the comma and the space after the comma
  }

  return edgeInfo;
}

void processLine(const std::string &line, Graph *graph, size_t lineIndex) {

  // Jump the first 2 lines and empty line (Not needed)
  if (lineIndex == 0 || lineIndex == 1 || line.empty())
    return;

  CycleNb cycleNb = line[0] - '0';

  std::vector<std::string> edgeInfo = parseOneLine(line);

  EdgeId edgeId;
  State state;

  if (failed(findEdgeInGraph(graph, edgeInfo, edgeId))) {
    return;
  }
  if (failed(findState(edgeInfo[5], state))) {
    return;
  }

  graph->addEdgeState(cycleNb, edgeId, state);
}