#include "Graph.h"
#include "GraphEdge.h"
#include "GraphNode.h"
#include "dotReformat.h"
#include "DOTParser.h"
#include "mlir/Support/LogicalResult.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <regex>

using namespace dynamatic::experimental::visual_dataflow;

LogicalResult processDOT(std::ifstream &file, Graph *graph){

  std::string line;

  EdgeId currentEdgeID = 0;
  GraphNode* currentNode = nullptr;
  bool insideNodeDefinition = false;
  GraphEdge* currentEdge = nullptr;
  bool insideEdgeDefinition = false;

  std::regex positionRegex("e,\\s*([0-9]+),([0-9]+(\\.[0-9]+)?)");
  std::smatch match;

  


  while (std::getline(file, line)) {
    if (line.find("->") == std::string::npos
        && line.find('[') != std::string::npos
        && line.find("node") == std::string::npos
        && line.find("graph") == std::string::npos) {
      GraphNode currentNode;
      insideNodeDefinition = true;
    }

    else if (line.find("->") != std::string::npos){
      GraphEdge currentEdge;
      insideEdgeDefinition = true;
      currentEdge.setId(currentEdgeID);
    }

    else if (insideNodeDefinition && line.find("in") != std::string::npos && line.find("label") == std::string::npos
             && line.find("mlir_op") == std::string::npos
             && line.find('[') == std::string::npos
             && line.find("fillcolor") == std::string::npos
             && line.find("type") == std::string::npos)
    {
      int occurrences = std::count(line.begin(), line.end(), ' ');
      if (occurrences != std::string::npos){
        for (int i = 1; i <= occurrences + 1; i++) {
          std::string portName = std::to_string(i);
          currentNode->addPort(portName, true);
        }
      }
    }

    else if (insideNodeDefinition && line.find("label") != std::string::npos) {
      NodeId id = line.substr(line.find('=') + 1, line.find(',') - line.find('=') - 1);
      currentNode->setId(id);
    }

    else if (insideNodeDefinition && line.find("out") != std::string::npos && line.find("label") == std::string::npos){
      int occurrences = std::count(line.begin(), line.end(), ' ');
      if (occurrences != std::string::npos){
        for (int i = 1; i <= occurrences + 1; i++) {
          std::string portName = std::to_string(i);
          currentNode->addPort(portName, false);
        }
      }
    }

    else if (insideNodeDefinition && line.find("pos") != std::string::npos){
      std::istringstream iss(line.substr(line.find('\"') + 1, line.rfind('\"') - line.find('\"') - 1));
      float x, y;
      char comma;
      (iss >> x >> comma >> y);
      std::pair<float, float> position = std::make_pair(x,y);
      currentNode->setPosition(position);
    }

    else if (insideNodeDefinition && line.find(']') != std::string::npos) {
      graph->addNode(currentNode);
      insideNodeDefinition = false;
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
              std::pair<float, float> position = std::make_pair(x,y);
              currentEdge->addPosition(position);
            }
          }
        }
      }
    }

    else if (insideEdgeDefinition && line.find("->") != std::string::npos) {
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
        
        GraphNode* src = nullptr;
        LogicalResult tryGetSrc = graph->getNode(leftPart, src);
        GraphNode* dst = nullptr;
        LogicalResult tryGetDst = graph->getNode(rightPart, dst);

        currentEdge->setSrc(src);
        currentEdge->setDst(dst);

      }
    }

    if (insideEdgeDefinition && line.find("from") != std::string::npos){
      int out = std::stoi(line.substr(line.find("out")+3 , line.find(',')));
      currentEdge->setOutPort(out);

    }

    if (insideEdgeDefinition && line.find("to") != std::string::npos && line.find("->") == std::string::npos){
      int in = std::stoi(line.substr(line.find('n')+1 , line.find(',')));
      currentEdge->setInPort(in);
    }


    if (insideEdgeDefinition && line.find(']') != std::string::npos) {
      insideEdgeDefinition = false;
      graph->addEdge(currentEdge);
      currentEdgeID += 1;
    }

    
  }
    

  return success();
    
}






// !!!
// !!!
// !!!
// REMOVE BEFORE COMMIT

int main() {

 reformatDot("gcd.dot","outputFinal.dot");
 

 std::ifstream f;
 f.open("outputFinal.dot");

 Graph graph(1);

 LogicalResult result = dynamatic::experimental::visual_dataflow::processDOT(f, &graph);


 auto processNode = [](GraphNode* node) {
    // Perform some action on the node
    // For example, print the node's identifier
    std::cout << "Node ID: " << node->getNodeId() << std::endl;
};

// Use the iterateNodes function to process all nodes
graph.iterateNodes(processNode);


 // Iterate through the map and print each pair

//  for (const auto& pair : graph->nodes) {
//    std::string key = pair.first;
//    const Node* nodePtr = pair.second;

//    std::cout << "NodeId: " << key << " " << nodePtr->id << ", Node label: " << nodePtr->label <<", Node pos: " << nodePtr->pos.x << " "<< nodePtr->pos.y  << ", Node in/out:" << nodePtr->inPorts.size() << " "<< nodePtr->outPorts.size() << std::endl;
//  }

//  std::cout << " " << std::endl;

//  // Iterate through the map and print each pair
//  for (const auto& pair : graph->edges) {
//    const Edge* edgePtr = pair;

//    std::cout << "EdgeId: " << pair->id << ", src:"<< edgePtr->src->label << ", dst:" << edgePtr->dst->label << ", from:" << edgePtr->outPort << ", to:" << edgePtr->inPort << std::endl;

//    if (edgePtr) {
//      for (const Pos& pos : edgePtr->pos) {
//        std::cout << "  Pos(x=" << pos.x << ", y=" << pos.y << ")" << std::endl;
//      }
//    } else {
//      std::cout << "  Edge is nullptr" << std::endl;
//    }
//  }
}



  
