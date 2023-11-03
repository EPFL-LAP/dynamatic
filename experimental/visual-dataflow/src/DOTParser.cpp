//===- DotParser.cpp - Parses a DOT file ------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DOT parsing.
//
//===----------------------------------------------------------------------===//

#include "DOTParser.h"
#include "DOTReformat.h"
#include "Graph.h"
#include "GraphEdge.h"
#include "GraphNode.h"
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace dynamatic::experimental::visual_dataflow;
using namespace mlir;

LogicalResult
dynamatic::experimental::visual_dataflow::processDOT(std::ifstream &file,
                                                     Graph &graph) {

  std::string line;

  EdgeId currentEdgeID = 0;
  GraphNode currentNode;
  bool insideNodeDefinition = false;
  GraphEdge currentEdge;
  bool insideEdgeDefinition = false;

  std::regex positionRegex("e,\\s*([0-9]+),([0-9]+(\\.[0-9]+)?)");
  std::smatch match;

  while (std::getline(file, line)) {
    if (line.find("->") == std::string::npos &&
        line.find('[') != std::string::npos &&
        line.find("node") == std::string::npos &&
        line.find("graph") == std::string::npos) {

      currentNode = GraphNode();
      insideNodeDefinition = true;

    } else if (line.find("->") != std::string::npos) {
      currentEdge = GraphEdge();
      insideEdgeDefinition = true;
      currentEdge.setId(currentEdgeID);

    } else if (insideNodeDefinition && line.find("in") != std::string::npos &&
               line.find("label") == std::string::npos &&
               line.find("mlir_op") == std::string::npos &&
               line.find('[') == std::string::npos &&
               line.find("fillcolor") == std::string::npos &&
               line.find("type") == std::string::npos) {
      size_t occurrences = std::count(line.begin(), line.end(), ' ');
      if (occurrences != std::string::npos) {
        for (size_t i = 1; i <= occurrences + 1; i++) {
          std::string portName = std::to_string(i);
          currentNode.addPort(portName, true);
        }
      }

    } else if (insideNodeDefinition &&
               line.find("label") != std::string::npos) {
      NodeId id =
          line.substr(line.find('=') + 1, line.find(',') - line.find('=') - 1);
      currentNode.setId(id);

    } else if (insideNodeDefinition && line.find("out") != std::string::npos &&
               line.find("label") == std::string::npos) {
      size_t occurrences = std::count(line.begin(), line.end(), ' ');
      if (occurrences != std::string::npos) {
        for (size_t i = 1; i <= occurrences + 1; i++) {
          std::string portName = std::to_string(i);
          currentNode.addPort(portName, false);
        }
      }

    } else if (insideNodeDefinition && line.find("pos") != std::string::npos) {
      std::istringstream iss(line.substr(
          line.find('\"') + 1, line.rfind('\"') - line.find('\"') - 1));
      float x, y;
      char comma;
      (iss >> x >> comma >> y);
      std::pair<float, float> position = std::make_pair(x, y);
      currentNode.setPosition(position);

    } else if (insideNodeDefinition && line.find(']') != std::string::npos) {
      graph.addNode(currentNode);
      insideNodeDefinition = false;
    }

    if (insideEdgeDefinition && line.find("pos") != std::string::npos) {
      size_t startPos = line.find('\"');
      size_t endPos = line.rfind('\"');
      if (endPos == startPos)
        endPos = line.find('\\');

      if (startPos != std::string::npos && endPos != std::string::npos) {
        size_t digitPos = std::string::npos;
        for (size_t i = startPos + 1; i < endPos; ++i) {
          if (std::isdigit(line[i])) {
            digitPos = i;
            break;
          }
        }

        if (digitPos != std::string::npos) {
          std::string positionString = line.substr(digitPos, endPos - digitPos);
          std::istringstream iss(positionString);
          std::string token;

          std::set<std::pair<float, float>>
              uniquePositions; // Use a set to store unique positions

          while ((std::getline(iss, token, ' '))) {
            if (token.empty())
              continue;

            size_t commaPos = token.find(',');

            if (commaPos != std::string::npos) {
              std::string xStr = token.substr(0, commaPos);
              std::string yStr = token.substr(commaPos + 1);
              float x = std::stof(xStr);
              float y = std::stof(yStr);
              std::pair<float, float> position = std::make_pair(x, y);

              // Check if the position is unique before adding it
              if (uniquePositions.insert(position).second) {
                currentEdge.addPosition(position);
              }
            }
          }
        }
      }

    } else if (insideEdgeDefinition && line.find("->") != std::string::npos) {
      size_t arrowPos = line.find("-> ");

      if (arrowPos != std::string::npos) {
        std::string leftPart = line.substr(0, arrowPos);
        leftPart.erase(0, leftPart.find_first_not_of(" \t\n\r\f\v"));
        leftPart.erase(leftPart.find_last_not_of(" \t\n\r\f\v") + 1);

        std::string rightPart = line.substr(arrowPos + 2);
        size_t firstSpacePos = rightPart.find('\t');
        if (firstSpacePos != std::string::npos) {
          rightPart = rightPart.substr(0, firstSpacePos);
        }
        rightPart.erase(0, rightPart.find_first_not_of(" \t\n\r\f\v"));
        rightPart.erase(rightPart.find_last_not_of(" \t\n\r\f\v") + 1);

        GraphNode src, dst;
        if (failed(graph.getNode(leftPart, src)) ||
            failed(graph.getNode(rightPart, dst)))
          return failure();

        currentEdge.setSrc(src);
        currentEdge.setDst(dst);
      }
    }

    if (insideEdgeDefinition && line.find("from") != std::string::npos) {
      int out = std::stoi(line.substr(line.find("out") + 3, line.find(',')));
      currentEdge.setOutPort(out);
    }

    if (insideEdgeDefinition && line.find("to") != std::string::npos &&
        line.find("->") == std::string::npos) {
      int in = std::stoi(line.substr(line.find('n') + 1, line.find(',')));
      currentEdge.setInPort(in);
    }

    if (insideEdgeDefinition && line.find(']') != std::string::npos) {
      insideEdgeDefinition = false;
      graph.addEdge(currentEdge);
      currentEdgeID += 1;
    }
  }

  return success();
}
