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
#include "llvm/ADT/StringRef.h"
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace dynamatic::visual;
using namespace mlir;

LogicalResult dynamatic::visual::processDOT(std::ifstream &file, Graph &graph) {

  std::string line;

  EdgeId currentEdgeID = 0;
  GraphNode currentNode;
  bool insideNodeDefinition = false;
  GraphEdge currentEdge;
  bool insideEdgeDefinition = false;
  BB currentBB;
  bool insideBBDefinition = false;

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

    } else if (insideNodeDefinition && line.find("in=") != std::string::npos) {
      size_t occurrences = std::count(line.begin(), line.end(), ' ');
      if (occurrences != std::string::npos) {
        for (size_t i = 1; i <= occurrences + 1; i++) {
          currentNode.addPort(i, true);
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
          currentNode.addPort(i, false);
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

    } else if (insideNodeDefinition &&
               line.find("fillcolor") != std::string::npos) {
      Color color =
          line.substr(line.find("=") + 1, line.find(',') - line.find('=') - 1);
      currentNode.setColor(color);

    } else if (insideNodeDefinition && line.find("]") != std::string::npos) {
      float width = std::stof(
          line.substr(line.find("=") + 1, line.find(']') - line.find('=') - 1));
      currentNode.setWidth(width);

      graph.addNode(currentNode);
      insideNodeDefinition = false;
    }

    if (insideNodeDefinition && line.find("shape") != std::string::npos) {
      std::size_t startPos = line.find('=') + 1;
      std::size_t endPos = line.find(',', startPos);
      Shape shape = line.substr(startPos, endPos - startPos);
      currentNode.setShape(shape);
    }

    if (insideNodeDefinition && line.find("style=") != std::string::npos &&
        line.find("dotted") != std::string::npos) {
      currentNode.setDashed(true);
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

          std::set<std::pair<float, float>> uniquePositions;

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
    if (insideEdgeDefinition && line.find("arrowhead") != std::string::npos) {
      size_t eqIDx = line.find('=') + 1;
      currentEdge.setArrowhead(line.substr(eqIDx, line.find(',') - eqIDx));
    }

    if (insideEdgeDefinition && line.find("start_0") != std::string::npos) {
      currentEdge.setDashed(true);
    }

    if (insideEdgeDefinition && line.find("style=") != std::string::npos &&
        line.find("dotted") != std::string::npos) {
      currentEdge.setDashed(true);
    }

    if (insideEdgeDefinition && line.find(']') != std::string::npos) {
      insideEdgeDefinition = false;
      graph.addEdge(currentEdge);
      currentEdgeID += 1;
    }

    if (!insideEdgeDefinition && !insideNodeDefinition &&
        line.find("subgraph") != std::string::npos) {
      BB newBB;
      currentBB = newBB;
      insideBBDefinition = true;
    }

    if (insideBBDefinition && line.find("bb") != std::string::npos) {
      std::size_t startPos = line.find('"') + 1;
      std::size_t endPos = line.find_last_of('"');
      std::string numbers = line.substr(startPos, endPos - startPos);

      std::stringstream ss(numbers);
      std::string item;
      while (std::getline(ss, item, ',')) {
        currentBB.boundries.push_back(std::stof(item));
      }
    }

    if (insideBBDefinition && line.find("label") != std::string::npos) {
      std::size_t startPos = line.find('=') + 1;
      std::size_t endPos = line.find(',', startPos);
      std::string label = line.substr(startPos, endPos - startPos);
      currentBB.label = label;
    }

    if (insideBBDefinition && line.find("lheight") != std::string::npos) {
      std::size_t startPos = line.find('=') + 1;
      std::size_t endPos = line.find(',', startPos);
      float height = std::stof(line.substr(startPos, endPos - startPos));
      currentBB.labelSize.first = height;
    }

    if (insideBBDefinition && line.find("lp") != std::string::npos) {
      std::size_t startPos = line.find('"') + 1;
      std::size_t endPos = line.find_last_of('"');
      std::string numbers = line.substr(startPos, endPos - startPos);

      std::stringstream ss(numbers);
      std::string item;
      std::getline(ss, item, ',');
      float x = std::stof(item);
      std::getline(ss, item, ',');
      float y = std::stof(item);

      currentBB.labelPosition.first = x;
      currentBB.labelPosition.second = y;
    }

    if (insideBBDefinition && line.find("lwidth") != std::string::npos) {
      std::size_t startPos = line.find('=') + 1;
      std::size_t endPos = line.find(',', startPos);
      float width = std::stof(line.substr(startPos, endPos - startPos));
      currentBB.labelSize.second = width;
    }

    if (insideBBDefinition && line.find("];") != std::string::npos) {
      graph.addBB(currentBB);
      insideBBDefinition = false;
    }
  }

  return success();
}
