//
// Created by Alice Potter on 05.10.2023.
//

#include "graphComponents.h"
#include <iostream>

/**
 * @brief This function transforms a state of type string into its corresponding state in type State.
 *
 * @param state_string The state in type string.
 * @return The corresponding state of type State.
 */
State findState(const std::string &state_string) {
  if (state_string == "undefined") {
    return undefined;
  } else if (state_string == "ready") {
    return ready;
  } else if (state_string == "empty") {
    return empty;
  } else if (state_string == "valid") {
    return valid;
  } else if (state_string == "valid_ready") {
    return valid_ready;
  } else {
    throw std::runtime_error("Error unknown state for the edge.");
  }
}
/**
 * @brief This function finds an edge in the graph.
 *
 * @param graph The graph.
 * @param edge_info An array containing information about an edge of the graph.
 * index 0 : cycle number, index 1 : src_component, index 2 : src_port, index 3 : dst_component,
 * index 4 : dst_port, index 5 : status.
 * @return The corresponding edge id.
 */
EdgeId findEdgeInGraph(const Graph *graph, const std::vector<std::string> &edge_info) {

  for (Edge* edge : graph->edges) {
    if(edge_info[1] == edge->src->label && std::stoi(edge_info[2]) == edge->inPort
        && edge_info[3] == edge->dst->label && std::stoi(edge_info[4]) == edge->outPort) {
      return edge->id;
    }
  }

  throw std::runtime_error("Error : an edge defined in the csv file was not found in the graph.");
}
/**
 * @brief This function parses a line from a csv file.
 *
 * @param line The line.
 * @return An array of strings.
 * index 0 : cycle number, index 1 : src_component, index 2 : src_port, index 3 : dst_component,
 * index 4 : dst_port, index 5 : status.
 */
std::vector<std::string> parseOneLine(const std::string line) {

  std::vector<std::string> edge_info(6);

  size_t startIndex = 0;
  size_t commaIndex;

  for (int i = 0; i < 6; ++i) {

    commaIndex = line.find(',', startIndex);

    if (commaIndex == std::string::npos) {
      // No comma found : it's the last part of the string
      commaIndex = line.size();
    }

    edge_info[i] = line.substr(startIndex, commaIndex - startIndex);

    startIndex = commaIndex + 2; // +2 to skip the comma and the space after the comma
  }

  return edge_info;
}

void processLine(const std::string &line, Graph *graph, size_t lineIndex) {

  //Jump the first 2 lines and empty line (Not needed)
  if (lineIndex == 0 || lineIndex == 1 || line.empty()) return;

  CycleNb cycleNb = line[0] - '0';
  //Creates an empty map if the key cycleNB is not found in the cycleEdgeStates map
  std::map<EdgeId, State> mapEdgeState = (graph->cycleEdgeStates)[cycleNb];

  std::vector<std::string> edgeInfo = parseOneLine(line);
  EdgeId edgeID = findEdgeInGraph(graph, edgeInfo);
  mapEdgeState[edgeID] = findState(edgeInfo[5]);
  graph->cycleEdgeStates[cycleNb] = mapEdgeState;
}

std::string printState(State state) {
  if (state == undefined) {
    return "undefined";
  } else if (state == ready) {
    return "ready";
  } else if (state == empty) {
    return "empty";
  } else if (state == valid) {
    return "valid";
  } else if (state == valid_ready) {
    return "valid_ready";
  } else {
    throw std::runtime_error("Error : unknown state for the edge.");
  }
}

void printMapEdgeState(const std::map<EdgeId, State> &map_edge_state) {
  for(const auto& pair : map_edge_state) {
    std::cout << "EdgeID: " << pair.first << ", State: " << printState(pair.second) << std::endl;
  }
}

void printMapCycle(Graph *graph) {
  if (graph == NULL) return;
  for(const auto& pair : graph->cycleEdgeStates) {
    std::cout << '\n' << "**** Cycle number : " << pair.first << " ****" << '\n' << std::endl;
    printMapEdgeState(pair.second);
  }
}