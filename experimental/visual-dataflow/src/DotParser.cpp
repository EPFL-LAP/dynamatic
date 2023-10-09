//
// Created by Albert Fares on 05.10.2023.
//


#include <string>
#include "graphComponents.h"
#include "dotReformat.h"
#include "dotparse.h"
#include "errors.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <regex>
#include "dotparse.h"


static void processAll(std::ifstream &file, Graph *graph){

  std::string line;

  EdgeId currentEdgeID = 0;
  NodeId currentNodeID = 0;
  Node* currentNode = nullptr;
  bool insideNodeDefinition = false;
  Edge* currentEdge = nullptr;
  bool insideEdgeDefinition = false;

  std::regex positionRegex("e,\\s*([0-9]+),([0-9]+(\\.[0-9]+)?)");
  std::smatch match;


  while (std::getline(file, line)) {
    if (line.find("->") == std::string::npos
        && line.find('[') != std::string::npos
        && line.find("node") == std::string::npos
        && line.find("graph") == std::string::npos) {
      currentNode = new Node;
      insideNodeDefinition = true;
      currentNode->id = currentNodeID;
    }

    else if (line.find("->") != std::string::npos){
      currentEdge = new Edge;
      graph->edges.push_back(currentEdge);
      insideEdgeDefinition = true;
      currentEdge->id = currentEdgeID;
    }

    else if (insideNodeDefinition && line.find("in") != std::string::npos && line.find("label") == std::string::npos
             && line.find("mlir_op") == std::string::npos
             && line.find('[') == std::string::npos
             && line.find("fillcolor") == std::string::npos
             && line.find("type") == std::string::npos)
    {
      int occurrences = std::count(line.begin(), line.end(), ' ');
      if (occurrences != std::string::npos){
        currentNode->inPorts.resize(occurrences + 1);
        for (int i = 0; i <= occurrences; i++) {
          currentNode->inPorts[i] = std::to_string(i+1);
        }
      }
    }

    else if (insideNodeDefinition && line.find("label") != std::string::npos) {
      currentNode->label = line.substr(line.find('=') + 1, line.find(',') - line.find('=') - 1);
    }

    else if (insideNodeDefinition && line.find("out") != std::string::npos && line.find("label") == std::string::npos){
      int occurrences = std::count(line.begin(), line.end(), ' ');
      if (occurrences != std::string::npos) {
        currentNode->outPorts.resize(occurrences + 1);
        for (int i = 0; i <= occurrences; i++) {
          currentNode->outPorts[i] = std::to_string(i+1);
        }
      }
    }

    else if (insideNodeDefinition && line.find("pos") != std::string::npos){
      std::istringstream iss(line.substr(line.find('\"') + 1, line.rfind('\"') - line.find('\"') - 1));
      int x, y;
      char comma;
      (iss >> x >> comma >> y);
      currentNode->pos.x = x;
      currentNode->pos.y = y;

    }

    else if (insideNodeDefinition && line.find(']') != std::string::npos) {
      insideNodeDefinition = false;
      graph->nodes[currentNode->label] = currentNode;
      currentNodeID += 1;
    }

    if (insideEdgeDefinition && line.find("pos") != std::string::npos) {
      size_t startPos = line.find('\"');
      size_t endPos = line.rfind('\"');
      if (endPos == startPos) endPos = line.find('\\');

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

          while ((std::getline(iss, token, ' '))) {
            if (token.empty()) continue;

            size_t commaPos = token.find(',');

            if (commaPos != std::string::npos) {
              std::string xStr = token.substr(0, commaPos);
              std::string yStr = token.substr(commaPos + 1);
              float x = std::stof(xStr);
              float y = std::stof(yStr);
              Pos pos;
              pos.x = x;
              pos.y = y;
              currentEdge->pos.push_back(pos);
            }
          }
        }
      }
    }

    else if (insideEdgeDefinition && line.find("->") != std::string::npos) {
      // Find the position of "->"
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

        currentEdge->src = graph->nodes[leftPart];
        currentEdge->dst = graph->nodes[rightPart];

      }
    }

    if (insideEdgeDefinition && line.find("from") != std::string::npos){
      int out = std::stoi(line.substr(line.find("out")+3 , line.find(',')));
      currentEdge->outPort = out;

    }

    if (insideEdgeDefinition && line.find("to") != std::string::npos && line.find("->") == std::string::npos){
      int in = std::stoi(line.substr(line.find('n')+1 , line.find(',')));
      currentEdge->inPort = in;
    }


    if (insideEdgeDefinition && line.find(']') != std::string::npos) {
      insideEdgeDefinition = false;
      graph->edges[currentEdgeID] = currentEdge;
      currentEdgeID += 1;
    }
  }


}

//int main() {
//
//  processDotFile("gcd.dot","output.dot");
//  insertNewlineBeforeStyle("output.dot","output4.dot");
//  removeBackslashWithSpaceFromPos("output4.dot", "output5.dot");
//  removeEverythingAfterApostropheComma("output5.dot","output6.dot");
//  removeEverythingAfterCommaInStyle("output6.dot","outputFinal.dot");
//
//
//  std::ifstream f;
//  f.open("outputFinal.dot");
//
//  Graph* graph = new Graph;
//
//  std::string line;
//
//
//  processAll(f,graph);
//
//
//  // Iterate through the map and print each pair
//  for (const auto& pair : graph->nodes) {
//    std::string key = pair.first;
//    const Node* nodePtr = pair.second;
//
//    std::cout << "NodeId: " << key << " " << nodePtr->id << ", Node label: " << nodePtr->label <<", Node pos: " << nodePtr->pos.x << " "<< nodePtr->pos.y  << ", Node in/out:" << nodePtr->inPorts.size() << " "<< nodePtr->outPorts.size() << std::endl;
//  }
//
//  std::cout << " " << std::endl;
//
//  // Iterate through the map and print each pair
//  for (const auto& pair : graph->edges) {
//    const Edge* edgePtr = pair;
//
//    std::cout << "EdgeId: " << pair->id << ", src:"<< edgePtr->src->label << ", dst:" << edgePtr->dst->label << ", from:" << edgePtr->outPort << ", to:" << edgePtr->inPort << std::endl;
//
//    if (edgePtr) {
//      for (const Pos& pos : edgePtr->pos) {
//        std::cout << "  Pos(x=" << pos.x << ", y=" << pos.y << ")" << std::endl;
//      }
//    } else {
//      std::cout << "  Edge is nullptr" << std::endl;
//    }
//  }
//}




