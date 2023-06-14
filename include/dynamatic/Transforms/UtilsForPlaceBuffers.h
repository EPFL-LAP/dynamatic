//===- UtilsForPlaceBuffers.h - functions for placing buffer  ---*- C++ -*-===//
//
// This file declaresfunction supports for buffer placement.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_UTILSFORPLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_UTILSFORPLACEBUFFERS_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include <fstream>
#include <gurobi_c++.h>
#include <map>
// #include <optional>

namespace dynamatic {
namespace buffer {

using namespace circt;
using namespace circt::handshake;

/// An arch stores the basic information (execution frequency, isBackEdge) 
/// of an arch  between basic blocks.
struct arch {
  int srcBB, dstBB;
  unsigned freq;
  bool isBackEdge = false;
};

/// A channel is the entity of the connections between units; 
/// A channel is identified by its port and the connection units.  
struct channel : arch {
  channel () : arch() {};

  channel (Operation *opSrc, Operation *opDst, Value *valPort) : arch() {
    this->opSrc = opSrc;
    this->opDst = opDst;
    this->valPort = valPort;
  };

  Operation *opSrc, *opDst;
  Value *valPort;

  void print() {
    llvm::errs() << "opSrc: " << *(opSrc) << " ---> ";
    llvm::errs() << "opDst: " << *(opDst) << "\n";
  }
};

/// A port is the entity of the connections between units;
struct port {
  port () : opVal(nullptr) {};
  port (Value *opVal) : opVal(opVal) {};

  double portLatency = 0.0;
  Value *opVal;

  SmallVector<channel *> cntChannels;
};

/// A unit is the entity of the operations in the graph;
struct unit {
  unit () : op(nullptr) {};
  unit (Operation *op) : op(op) {};

  unsigned freq = 0;
  double latency = 0.0;
  double delay = 0.0;
  int ind = -1;
  Operation *op;
  SmallVector<port *> inPorts;
  SmallVector<port *> outPorts;
};

/// Identify whether an operation is a start point of the fucntion block.
bool isEntryOp(Operation *op,
               std::vector<Operation *> &visitedOp);

/// Ger the index of the basic block of an operation.
int getBBIndex(Operation *op);

/// Identify whether the connection between the source operation and
/// the destination operation is a back edge.
bool isBackEdge(Operation *opSrc, Operation *opDst);

/// Get the relative unit of an operation.
unit *getUnitWithOp(Operation *op, std::vector<unit *> &unitList);

/// Connect a unit with its input channels through the input ports.
void connectInChannel(unit *unitNode, channel *inChannel);

/// Deep first search the handshake file to get the units connection graph.
void dfsHandshakeGraph(Operation *opNode, std::vector<unit *> &unitList,
     std::vector<Operation *> &visited, channel *inChannel=nullptr);

struct dataFlowCircuit {

  std::map<std::string, int> compNameToIndex = {
      {"cmpi", 0},     {"addi", 1},
      {"subi", 2},      {"muli", 3},
      {"extsi", 4},    
      // {"load", 5}, {"store", 6},  ????
      {"d_load", 5}, {"d_store", 6},  
      {"LsqLoad", 7}, // ?
      {"LsqStore", 8}, // ?
      // {"merge", 9},    // ??
      {"Getelementptr", 10},
      {"Addf", 11},    {"Subf", 12},
      {"Mulf", 13},    {"divu", 14},
      {"Divs", 15},    {"Divf", 16},
      {"cmpf", 17},    
      // {"Phic", 18}, // ---> merge & control merge
      {"merge", 18}, {"control_merge", 18},
      {"zdl", 19},     {"fork", 20},
      {"Ret", 21}, // handshake.return ?
      {"br", 22},      // conditional branch ??
      {"end", 23},
      {"and", 24},     {"or", 25},
      {"xori", 26},     {"Shl", 27},
      {"Ashr", 28},    {"Lshr", 29},
      {"select", 30},  {"mux", 31}};

  std::map<int, int> bitWidthToIndex = {
      {1, 0}, {2, 1}, {4, 2}, {8, 3}, 
      {16, 4}, {32, 5}, {64, 6}};

  std::vector<std::vector<float>>
  readInfoFromFile(const std::string &filename);

  double targetCP, maxCP;
  std::vector<unit *> units;
  std::vector<channel *> channels;
  std::vector<int> selBBs;

  int execN = 0;
  std::vector<std::vector<float>> delayInfo;
  std::vector<std::vector<float>> latencyInfo;

  void printCircuits();
};

} // namespace handshake
} // namespace circt

#endif // DYNAMATIC_TRANSFORMS_UTILSFORPLACEBUFFERS_H