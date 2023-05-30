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

using namespace circt;
using namespace circt::handshake;

namespace dynamatic {
namespace buffer {

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
  std::optional<Operation *> opSrc, opDst;

  // // If opDst and opSrc are not in the same basic blocks, and
  // // if opDst's users are in the same basic blocks as opDst, it is an
  // in-edge. bool isInEdge = false;

  // // If opDst and opSrc are not in the same basic blocks, and
  // // if opSrc's users are in the same basic blocks as opSrc, it is an
  // out-edge. bool isOutEdge = false;
  void print() {
    llvm::errs() << "opSrc: " << *(opSrc.value()) << " ---> ";
    llvm::errs() << "opDst: " << *(opDst.value()) << "\n";
  }
};

struct unit {
  unsigned freq = 0;
  double latency = 0.0;
  double delay = 0.0;
  int ind = -1;
  Operation *op;
  SmallVector<channel *> inChannels;
  SmallVector<channel *> outChannels;
};

struct dataFlowCircuit {

  std::map<std::string, int> compNameToIndex = {
      {"cmpi", 0},     {"add", 1},
      {"sub", 2},      {"muli", 3},
      {"extsi", 4},    {"load", 5},
      {"store", 6},    {"LsqLoad", 7}, // d_load ?
      {"LsqStore", 8},                 // d_store ?
      {"merge", 9},    {"Getelementptr", 10},
      {"Addf", 11},    {"Subf", 12},
      {"Mulf", 13},    {"divu", 14},
      {"Divs", 15},    {"Divf", 16},
      {"Cmpf", 17},    {"Phic", 18},
      {"zdl", 19},     {"Fork", 20},
      {"Ret", 21}, // handshake.return ?
      {"Br", 22},      {"end", 23},
      {"and", 24},     {"or", 25},
      {"xor", 26},     {"Shl", 27},
      {"Ashr", 28},    {"Lshr", 29},
      {"Select", 30},  {"Mux", 31}};

  double targetCP, maxCP;
  std::vector<unit *> units;
  std::vector<channel *> channels;
  std::vector<basicBlock *> selBBs;
  std::string delayFile = "/home/yuxuan/Projects/dynamatic/legacy-dynamatic/"
                          "dhls/etc/dynamatic/data/targets/default_delay.dat";
  int execN = 0;
  std::vector<std::vector<float>> delayInfo;

  std::vector<std::vector<float>>
  readDelayInfoFromFile(const std::string &filename);

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

  void initCombinationalUnitDelay();

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

basicBlock *findExistsBB(unsigned bbInd, std::vector<basicBlock *> &bbList);

arch *findExistsArch(basicBlock *bbSrc, basicBlock *bbDst,
                     std::vector<arch *> &archList);

// Graph build functions
Operation *foundEntryOp(handshake::FuncOp funcOp,
                        std::vector<Operation *> &visitedOp);

unsigned getBBIndex(Operation *op);

bool isConnected(basicBlock *bb, Operation *op);

bool isBackEdge(Operation *opSrc, Operation *opDst);

void linkBBViaChannel(Operation *opSrc, Operation *opDst, unsigned newbbInd,
                      basicBlock *curBB, std::vector<basicBlock *> &bbList);

void dfsBBGraphs(Operation *opNode, std::vector<Operation *> &visited,
                 basicBlock *curBB, std::vector<basicBlock *> &bbList);

void dfsBB(basicBlock *bb, std::vector<basicBlock *> &bbList,
           std::vector<unsigned> &bbIndexList,
           std::vector<Operation *> &visitedOpList);

void printBBConnectivity(std::vector<basicBlock *> &bbList);

// // MILP description functions
// arch *findArcWithVarName(std::string varName,
//                        std::vector<basicBlock *> &bbList);

// std::vector<std::string>
// findSameDstOpStrings(const std::string &inputString,
//                     const std::vector<std::string> &stringList,
//                     std::vector<basicBlock *> &bbList);

// std::vector<std::string>
// findSameSrcOpStrings(const std::string &inputString,
//                    const std::vector<std::string> &stringList,
//                    std::vector<basicBlock *> &bbList);

// void extractMarkedGraphBB(std::vector<basicBlock *> bbList);
} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_UTILSFORPLACEBUFFERS_H