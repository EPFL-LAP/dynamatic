//
// Created by Albert Fares, Quentin Gross & Alice Potter on 05.10.2023.
//

#include <string>
#include <map>
#include <fstream>

using EdgeId = int;
using NodeId = int;
using CycleNb = int;

enum State {
    undefined,
    ready,
    empty,
    valid,
    valid_ready
};

struct Pos;
struct Graph;
struct Edge;
struct Node;
struct FileResult;

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

