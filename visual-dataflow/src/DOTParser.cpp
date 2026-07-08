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
    // Graphviz wraps long attribute values (notably spline `pos` coordinate
    // lists) across multiple physical lines using a trailing backslash. Splice
    // the continuation lines back into a single logical line before parsing.
    while (!line.empty() && line.back() == '\\') {
      line.pop_back();
      std::string continuation;
      if (!std::getline(file, continuation))
        break;
      line += continuation;
    }

    if (!insideNodeDefinition && !insideEdgeDefinition &&
        line.find("->") == std::string::npos &&
        line.find('[') != std::string::npos &&
        line.find("node") == std::string::npos &&
        line.find("graph") == std::string::npos) {

      currentNode = GraphNode();
      insideNodeDefinition = true;

      // The node's unique identifier is the token preceding the '['; edges
      // reference the node by this name.
      std::string name = line.substr(0, line.find('['));
      name.erase(0, name.find_first_not_of(" \t\n\r\f\v"));
      name.erase(name.find_last_not_of(" \t\n\r\f\v") + 1);
      currentNode.setId(name);

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
      std::string label =
          line.substr(line.find('=') + 1, line.find(',') - line.find('=') - 1);
      // Strip the surrounding double quotes Graphviz adds around labels that
      // are not bare identifiers (e.g. "LD (pixelR)").
      if (label.size() >= 2 && label.front() == '"' && label.back() == '"')
        label = label.substr(1, label.size() - 2);
      currentNode.setLabel(label);

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

    // The numeric source/destination port indices live in the `from_idx`/
    // `to_idx` attributes (the `from`/`to` attributes hold the port *names*).
    if (insideEdgeDefinition && line.find("from_idx=") != std::string::npos) {
      int out = std::stoi(line.substr(line.find("from_idx=") + 9));
      currentEdge.setOutPort(out);
    }

    if (insideEdgeDefinition && line.find("to_idx=") != std::string::npos) {
      int in = std::stoi(line.substr(line.find("to_idx=") + 7));
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
