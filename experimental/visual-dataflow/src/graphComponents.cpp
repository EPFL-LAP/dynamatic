//
// Created by Albert Fares, Quentin Gross & Alice Potter on 05.10.2023.
//
#include "graphComponents.h"
#include <string>
#include <vector>
#include <map>
#include <fstream>

struct Pos{
    float x;
    float y;
};
struct Graph {
    std::vector<Edge*> edges;
    std::map<std::string, Node*> nodes;
    std::map<CycleNb, std::map<EdgeId, State> > cycleEdgeStates;
};
struct Edge {
    EdgeId id;
    Node* src;
    Node* dst;
    int inPort;
    int outPort;
    std::vector<Pos> pos;
};
struct Node {
    NodeId id;
    std::string label;
    std::vector<std::string> inPorts;
    std::vector<std::string> outPorts;
    Pos pos;
};
struct FileResult {
    std::ifstream file;
    int status;
};
