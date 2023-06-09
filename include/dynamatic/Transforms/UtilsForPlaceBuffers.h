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
#include <optional>

namespace dynamatic {
namespace buffer {

using namespace circt;
using namespace circt::handshake;

struct arch;
struct channel;

struct basicBlock {
  unsigned index = UINT_MAX;
  unsigned freq = UINT_MAX;
  bool selBB = false;
  bool isEntryBB = false;
  bool isExitBB = false;
  std::vector<arch *> inArchs;
  std::vector<arch *> outArchs;
  std::vector<channel *> inChannels;
  std::vector<channel *> outChannels;
};

struct arch {
  unsigned freq;
  basicBlock *bbSrc, *bbDst;

  bool selArc = false;
  bool isBackEdge = false;

  arch() {}
  arch(const unsigned freq, basicBlock *bbSrc, basicBlock *bbDst)
      : freq(freq), bbSrc(bbSrc), bbDst(bbDst) {}
};

struct channel : arch {
  // std::optional<Operation *> opSrc, opDst;
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

struct port {
  port () : opVal(nullptr) {};
  port (Value *opVal) : opVal(opVal) {};

  double portLatency = 0.0;
  Value *opVal;

  SmallVector<channel *> cntChannels;
};

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
  // SmallVector<channel *> inChannels;
  // SmallVector<channel *> outChannels;
};

basicBlock *findExistsBB(unsigned bbInd, std::vector<basicBlock *> &bbList);

arch *findExistsArch(basicBlock *bbSrc, basicBlock *bbDst,
                     std::vector<arch *> &archList);

// Graph build functions
bool isEntryOp(Operation *op,
               std::vector<Operation *> &visitedOp);

unsigned getBBIndex(Operation *op);

bool isConnected(basicBlock *bb, Operation *op);

bool isBackEdge(Operation *opSrc, Operation *opDst);

void linkBBViaChannel(Operation *opSrc, Operation *opDst, unsigned newbbInd,
                      basicBlock *curBB, std::vector<basicBlock *> &bbList);

unit *getUnitWithOp(Operation *op, std::vector<unit *> &unitList);

void connectInChannel(unit *unitNode, channel *inChannel);

// void createOutPort(unit *unitNode, channel *outChannel);

void dfsHandshakeGraph(Operation *opNode, std::vector<unit *> &unitList,
     std::vector<Operation *> &visited, channel *inChannel=nullptr);

void dfsBBGraphs(Operation *opNode, std::vector<Operation *> &visited,
                 basicBlock *curBB, std::vector<basicBlock *> &bbList);

void dfsBB(basicBlock *bb, std::vector<basicBlock *> &bbList,
           std::vector<unsigned> &bbIndexList,
           std::vector<Operation *> &visitedOpList);

void printBBConnectivity(std::vector<basicBlock *> &bbList);

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

  double targetCP, maxCP;
  std::vector<unit *> units;
  std::vector<channel *> channels;
  std::vector<basicBlock *> selBBs;
  std::string infoFielDefault = std::getenv("LEGACY_DYNAMATICPP");
  std::string delayFile = infoFielDefault + "/data/targets/default_delay.dat";
  std::string latencyFile = infoFielDefault + "data/targets/default_latency.dat";

  int execN = 0;
  std::vector<std::vector<float>> delayInfo;
  std::vector<std::vector<float>> latencyInfo;

  std::vector<std::vector<float>>
  readInfoFromFile(const std::string &filename);

  void insertSelBB(handshake::FuncOp funcOp, basicBlock *bb);

  void insertSelArc(arch *arc);

  void printCircuits();

  void insertChannel(channel *ch) {
    if (!hasChannel(ch))
      channels.push_back(ch);
  }

  int findChannelIndex(channel *ch) {
    for (int i = 0; i < channels.size(); i++)
      if (channels[i]->opSrc == ch->opSrc && channels[i]->opDst == ch->opDst)
        return i;

    return -1;
  }

  bool hasChannel(channel *ch) {
    for (auto channel : channels)
      if (channel->opSrc == ch->opSrc && channel->opSrc == ch->opSrc)
        return true;

    return false;
  }

  bool hasChannel(Operation *srcOp, Operation *dstOp) {
    for (auto channel : channels)
      if (channel->opSrc == srcOp && channel->opDst == dstOp)
        return true;

    return false;
  }

  bool hasUnit(Operation *op) {
    for (auto unit : units)
      if (unit->op == op)
        return true;

    return false;
  }

  int findUnitIndex(Operation *op) {
    for (int i = 0; i < units.size(); i++)
      if (units[i]->op == op)
        return i;

    return -1;
  }

  void initUnitTimeInfo();

  void insertBuffersInChannel(MLIRContext *ctx, channel *ch, bool fifo, int slots);

  LogicalResult initMLIPModelVars(GRBModel& milpModel,
                                  GRBVar &varThrpt,
                                  std::vector<std::map<std::string, GRBVar>> &channelVars,
                                  std::vector<std::map<std::string, GRBVar>> &unitVars);

  LogicalResult createMILPPathConstrs(GRBModel &milpModel,
                                    std::vector<std::map<std::string, GRBVar>> &channelVars,
                                    std::vector<std::map<std::string, GRBVar>> &unitVars);
  
  LogicalResult createMILPThroughputConstrs(GRBModel &milpModel,
                                    GRBVar &varThrpt,
                                    std::vector<std::map<std::string, GRBVar>> &channelVars,
                                    std::vector<std::map<std::string, GRBVar>> &unitVars);
  
  LogicalResult defineCostFunction(GRBModel &milpModel,
                                  GRBVar &varThrpt,
                                  std::vector<std::map<std::string, GRBVar>> &channelVars,
                                  std::vector<std::map<std::string, GRBVar>> &unitVars);
  LogicalResult instantiateBuffers(MLIRContext *ctx,
                                  GRBModel &milpModel,
                                  GRBVar &varThrpt,
                                  std::vector<std::map<std::string, GRBVar>> &channelVars,
                                  std::vector<std::map<std::string, GRBVar>> &unitVars);
};

} // namespace handshake
} // namespace circt

#endif // DYNAMATIC_TRANSFORMS_UTILSFORPLACEBUFFERS_H