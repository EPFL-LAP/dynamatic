//===- UtilsForPlaceBuffers.h - functions for placing buffer  ---*- C++ -*-===//
//
// This file declaresfunction supports for buffer placement.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_UTILSFORPLACEBUFFERS_H
#define DYNAMATIC_TRANSFORMS_UTILSFORPLACEBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
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

    // If opDst and opSrc are not in the same basic blocks, and
    // if opDst's users are in the same basic blocks as opDst, it is an in-edge.
    bool isInEdge = false;

    // If opDst and opSrc are not in the same basic blocks, and
    // if opSrc's users are in the same basic blocks as opSrc, it is an out-edge.
    bool isOutEdge = false;

    arch() {}
    arch(const unsigned freq, basicBlock *bbSrc, basicBlock *bbDst)
        : freq(freq), bbSrc(bbSrc), bbDst(bbDst) {}
  };

  struct channel : arch {
    std::optional<Operation *> opSrc, opDst;
    
  };

  basicBlock *findExistsBB(unsigned bbInd, std::vector<basicBlock *> &bbList);

  arch *findExistsArch(basicBlock *bbSrc, basicBlock *bbDst,
                              std::vector<arch *> &archList);

  struct dataFlowGraphBB {
    std::vector<arch> archList;
    std::vector<basicBlock> bbList;
    unsigned cstMaxN;
    unsigned valExecN;
  };


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