//
// Created by Albert Fares, Quentin Gross & Alice Potter on 05.10.2023.
//
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
