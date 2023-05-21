//===- UtilsForPlaceBuffers.cpp - functions for placing buffer  -*- C++ -*-===//
//
// This file implements function supports for buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/UtilsForPlaceBuffers.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include <optional>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

Operation *buffer::foundEntryOp(handshake::FuncOp funcOp,
                        std::vector<Operation *> &visitedOp) {
  for (auto &op : funcOp.getOps()) {
    if (op.getAttrs().data()->getName() == "bb") {
      if (op.getAttrOfType<IntegerAttr>("bb").getUInt() == 0 &&
          isa<MergeOp>(op)) {
        return &op;
      }
      visitedOp.push_back(&op);
    }
  }
  return nullptr;
}

unsigned buffer::getBBIndex(Operation *op) {
  if (op->getAttrs().data()->getName() == "bb")
    return op->getAttrOfType<IntegerAttr>("bb").getUInt();
  return UINT_MAX;
}

bool buffer::isConnected(basicBlock *bb, Operation *op) {
  for (auto channel : bb->outChannels) {
    if (auto connectOp = (channel->opDst).value(); connectOp == op) {
      return true;
    }
  }
  return false;
}

bool buffer::isBackEdge(Operation *opSrc, Operation *opDst) {
  unsigned bbSrcInd = getBBIndex(opSrc);
  unsigned bbDstInd = getBBIndex(opDst);
  if (bbSrcInd == UINT_MAX || bbDstInd == UINT_MAX ||
      isa<handshake::EndOp>(*opDst))
    return false;

  if (bbSrcInd > bbDstInd)
    return true;

  Operation *curOp = opDst->getNextNode();
  while (curOp != nullptr && !isa<handshake::EndOp>(*curOp)) {
    if (curOp == opSrc)
      return true;
    curOp = curOp->getNextNode();
  }
  return false;
}

arch *buffer::findExistsArch(basicBlock *bbSrc, basicBlock *bbDst,
                             std::vector<arch *> &archList) {
  for (auto arch : archList)
    if (arch->bbSrc == bbSrc && arch->bbDst == bbDst)
      return arch;
  return nullptr;
}

basicBlock *buffer::findExistsBB(unsigned bbInd,
                                    std::vector<basicBlock *> &bbList) {
  for (auto bb : bbList)
    if (bb->index == bbInd)
      return bb;
  return nullptr;
}

// link the basic blocks via channel between operations
void buffer::linkBBViaChannel(Operation *opSrc, Operation *opDst, unsigned newbbInd,
                basicBlock *curBB, std::vector<basicBlock *> &bbList) {
  auto *outChannel = new channel();
  outChannel->opSrc = opSrc;
  outChannel->opDst = opDst;
  outChannel->isOutEdge = true;
  outChannel->bbSrc = curBB;
  if (auto bbDst = findExistsBB(newbbInd, bbList); bbDst != nullptr) {
    outChannel->bbDst = bbDst;
    if (bbDst->index <= curBB->index)
      outChannel->isBackEdge = true;
  } else {
    basicBlock *bb = new basicBlock();
    bb->index = newbbInd;
    outChannel->bbDst = bb;
    bbList.push_back(bb);
  }
  curBB->outChannels.push_back(outChannel);
  return;
}

void buffer::dfsBBGraphs(Operation *opNode, std::vector<Operation *> &visited,
                 basicBlock *curBB, std::vector<basicBlock *> &bbList) {
  if (std::find(visited.begin(), visited.end(), opNode) != visited.end()) {
    return;
  }
  // marked as visited
  visited.push_back(opNode);

  if (isa<handshake::EndOp>(*opNode)) {
    curBB->isExitBB = true;
    return;
  }

  // vectors to store successor operation
  SmallVector<Operation *> sucOps;

  for (auto sucOp : opNode->getResults().getUsers()) {

    // get the index of the successor basic block
    unsigned bbInd = getBBIndex(sucOp);

    // not in a basic block
    if (bbInd == UINT_MAX)
      continue;

    if (bbInd != getBBIndex(opNode)) {
      // if not in the same basic block, link via out arc
      linkBBViaChannel(opNode, sucOp, bbInd, curBB, bbList);
      // stop tranversing nodes not in the same basic block
      continue;
    } else if (isBackEdge(opNode, sucOp)) {
      // need to determine whether is a back edge in a same block
      linkBBViaChannel(opNode, sucOp, bbInd, curBB, bbList);
    }

    dfsBBGraphs(sucOp, visited, curBB, bbList);
  }
}

// visit the basic block via channel between operations
void buffer::dfsBB(basicBlock *bb, std::vector<basicBlock *> &bbList,
                std::vector<unsigned> &bbIndexList,
                std::vector<Operation *> &visitedOpList) {
  // skip bb that marked as visited
  if (std::find(bbIndexList.begin(), bbIndexList.end(), bb->index) !=
      bbIndexList.end())
    return;

  bbIndexList.push_back(bb->index);

  basicBlock *nextBB;
  for (auto outChannel : bb->outChannels) {
    nextBB = outChannel->bbDst;
    nextBB->inChannels.push_back(outChannel);

    // if the identical arch does not exist, add it to the list
    if (findExistsArch(outChannel->bbSrc, outChannel->bbDst, nextBB->inArchs) ==
        nullptr) {
          // link the basic blocks via arch
          arch *newArch = new arch();
          newArch->bbSrc = outChannel->bbSrc;
          newArch->bbDst = outChannel->bbDst;
          newArch->freq = 1;
          if (isBackEdge((outChannel->opSrc).value(), (outChannel->opDst).value())) {
            newArch->isBackEdge = true;
            newArch->freq = 100;
          }
          nextBB->inArchs.push_back(newArch);
          bb->outArchs.push_back(newArch);
    }

    if (outChannel->opDst.has_value()) {
      if (Operation *op = outChannel->opDst.value();
          isa<handshake::ControlMergeOp, handshake::MergeOp>(*op))
        // DFS curBB from the entry port
        dfsBBGraphs(op, visitedOpList, nextBB, bbList);
    }
    dfsBB(nextBB, bbList, bbIndexList, visitedOpList);
  }
}

void buffer::printBBConnectivity(std::vector<basicBlock *> &bbList) {
  for (auto bb : bbList) {
    llvm::errs() << bb->index << "\n";
    llvm::errs() << "is entry BB? " << bb->isEntryBB << "\n";
    llvm::errs() << "is exit BB? " << bb->isExitBB << "\n";
    llvm::errs() << "inArcs: \n";
    for(auto arc : bb->inArchs) {
      // arch *selArch = bb->inArc;
      llvm::errs() << "--------From bb" << arc->bbSrc->index << " ";
      llvm::errs() << "To bb" << arc->bbDst->index << "\n";
      llvm::errs() << "--------isBackEdge: " << arc->isBackEdge << "\n";
      llvm::errs() << "--------frequency: " << arc->freq << "\n\n";
    }
    
    llvm::errs() << "outArcs: \n";
    for(auto arc : bb->outArchs) {
      llvm::errs() << "--------From bb" << arc->bbSrc->index << " ";
      llvm::errs() << "To bb" << arc->bbDst->index << "\n";
      llvm::errs() << "--------isBackEdge: " << arc->isBackEdge << "\n";
      llvm::errs() << "--------frequency: " << arc->freq << "\n\n";
    }
    llvm::errs() << "=========================================\n";
  }
}
