//===- HandshakePlaceBuffers.cpp - Place buffers in DFG ---------*- C++ -*-===//
//
// This file implements the --place-buffers pass for throughput optimization by
// inserting buffers in the data flow graphs.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakePlaceBuffers.h"
#include "dynamatic/Transforms/UtilsForMILPSolver.h"
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

Operation *foundEntryOp(handshake::FuncOp funcOp,
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

unsigned getBBIndex(Operation *op) {
  if (op->getAttrs().data()->getName() == "bb")
    return op->getAttrOfType<IntegerAttr>("bb").getUInt();
  return UINT_MAX;
}

bool isConnected(basicBlock *bb, Operation *op) {
  for (auto arch : bb->outArcs) {
    if (auto connectOp = (arch->opDst).value(); connectOp == op) {
      return true;
    }
  }
  return false;
}

bool isBackEdge(Operation *opSrc, Operation *opDst) {
  unsigned bbSrcInd = getBBIndex(opSrc);
  unsigned bbDstInd = getBBIndex(opDst);
  if (bbSrcInd == UINT_MAX || bbDstInd == UINT_MAX || isa<handshake::EndOp>(*opDst))
    return false;

  if (bbSrcInd > bbDstInd)
    return true;

  Operation* curOp = opDst->getNextNode();
  while (curOp != nullptr && !isa<handshake::EndOp>(*curOp)) {
    if (curOp == opSrc)
      return true;
    curOp = curOp->getNextNode();
  }
  return false;
  
}

basicBlock* findExistsBB(unsigned bbInd, std::vector<basicBlock *> &bbList) {
  for (auto bb : bbList)
    if (bb->index == bbInd)
      return bb;
  return nullptr;
}

void linkNextBB(Operation *opSrc, Operation *opDst, unsigned newbbInd, basicBlock *curBB,
                std::vector<basicBlock *> &bbList) {
    arch *outArc = new arch();
    outArc->freq = 1;
    outArc->opSrc = opSrc;
    outArc->opDst = opDst;
    outArc->isOutEdge = true;
    outArc->bbSrc = curBB;
    if (auto bbDst = findExistsBB(newbbInd, bbList); bbDst != nullptr) {
      outArc->bbDst = bbDst;
      if (bbDst->index <= curBB->index)
        outArc->isBackEdge = true;
    }
    else {
      basicBlock *bb = new basicBlock();
      bb->index = newbbInd;
      outArc->bbDst = bb;
      bbList.push_back(bb);
    }
    curBB->outArcs.push_back(outArc);
    return;
}

void dfsBBGraphs(Operation *opNode, std::vector<Operation *> &visited,
                 basicBlock *curBB, std::vector<basicBlock *> &bbList) {
  if (std::find(visited.begin(), visited.end(), opNode) != visited.end()) {
    return;
  }
  // marked as visited
  visited.push_back(opNode);

  if(isa<handshake::EndOp>(*opNode)) {
    curBB->isExitBB = true;
    return;
  }

  // vectors to store successor operation
  SmallVector<Operation *> sucOps;
          
  for (auto sucOp :  opNode->getResults().getUsers()) {
    
    // get the index of the successor basic block
    unsigned bbInd = getBBIndex(sucOp);

    // not in a basic block
    if (bbInd == UINT_MAX) 
      continue;
    
    if (bbInd != getBBIndex(opNode)) {
      // if not in the same basic block, link via out arc
      linkNextBB(opNode, sucOp, bbInd, curBB, bbList);
      // stop tranversing nodes not in the same basic block
      continue;
    } else if (isBackEdge(opNode, sucOp)){
      // need to determine whether is a back edge in a same block
      linkNextBB(opNode, sucOp, bbInd, curBB, bbList);
    }
  
    dfsBBGraphs(sucOp, visited, curBB, bbList);
    
  }
  
}

// visit the basic block via link
void dfsBB(basicBlock *bb, std::vector<basicBlock *> &bbList,
           std::vector<unsigned> &bbIndexList, std::vector<Operation *> &visitedOpList) {
  // skip bb that marked as visited
  if (std::find(bbIndexList.begin(), bbIndexList.end(), bb->index) !=
      bbIndexList.end()) 
    return;
  
  bbIndexList.push_back(bb->index);

  basicBlock *nextBB;
  for (auto outArch : bb->outArcs) {
    nextBB = outArch->bbDst;
    nextBB->inArcs.push_back(outArch);

    if (outArch->opDst.has_value()) {
      if(Operation *op =  outArch->opDst.value(); 
        isa<handshake::ControlMergeOp, handshake::MergeOp>(*op)) 
      // DFS curBB from the entry port 
      dfsBBGraphs(op, visitedOpList, nextBB, bbList);
      
    }
    dfsBB(nextBB, bbList, bbIndexList, visitedOpList);
  }
}

void printBBConnectivity(std::vector<basicBlock *> &bbList) {
    for (auto bb : bbList) {
    llvm::errs() << bb->index << "\n";
    llvm::errs() << "is entry BB? " << bb->isEntryBB << "\n";
    llvm::errs() << "is exit BB? " << bb->isExitBB << "\n";
    llvm::errs() << "inArcs: \n";
    for (auto arch : bb->inArcs){
      llvm::errs() << "--------From bb" << arch->bbSrc->index << "(port: "
      << *(arch->opSrc.value()) << ")\n";
      llvm::errs() << "--------To bb" << arch->bbDst->index << "(port: "
      << *(arch->opDst.value()) << ")\n";
      llvm::errs() << "--------isBackEdge: " << arch->isBackEdge << "\n";
      llvm::errs() << "--------frequency: " << arch->freq << "\n\n";

    }
    llvm::errs() << "outArcs: \n";
    for (auto arch : bb->outArcs){
      llvm::errs() << "--------From bb" << arch->bbSrc->index << "(port: "
      << *(arch->opSrc.value()) << ")\n";
      llvm::errs() << "--------To bb" << arch->bbDst->index << "(port: "
      << *(arch->opDst.value()) << ")\n";
      llvm::errs() << "--------isBackEdge: " << arch->isBackEdge << "\n";
      llvm::errs() << "--------frequency: " << arch->freq << "\n\n";
    }
    llvm::errs() << "=========================================\n";
  }
}



static LogicalResult insertBuffers(handshake::FuncOp funcOp, MLIRContext *ctx) {
  std::vector<Operation *> visitedOpList;
  std::vector<unsigned> bbIndexList;
  std::vector<basicBlock *> bbList;

  Operation *entryOp = foundEntryOp(funcOp, visitedOpList);

  if (entryOp == nullptr) {
    return failure();
  }

  unsigned maxBBInd = 0;
  for (auto &op : funcOp.getOps())
    if (unsigned bbInd = getBBIndex(&op); bbInd != UINT_MAX && bbInd > maxBBInd)
      maxBBInd = bbInd;

  // build the entry basic block
  basicBlock *entryBB = new basicBlock();
  entryBB->isEntryBB = true;
  entryBB->index = 0;
  // bbIndexList.push_back(entryBB->index);
  bbList.push_back(entryBB);
  dfsBBGraphs(entryOp, visitedOpList, entryBB, bbList);

  basicBlock *prevBB = entryBB;
  
  llvm::errs() << "---------------entry bb finished-------------\n";

  // DFS from the entryBB, the BB graph can be fully traversed 
  // as every basic blocks are connected. 
  dfsBB(entryBB, bbList, bbIndexList, visitedOpList);
  assert(bbIndexList.size() == maxBBInd + 1 && "Disconnected basic blocks exist!");
  printBBConnectivity(bbList);

  // extractCFDFC(bbList);


  return success();
}

namespace {

/// Simple driver for prepare for legacy pass.
struct PlaceBuffersPass : public PlaceBuffersBase<PlaceBuffersPass> {

  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if (failed(insertBuffers(funcOp, &getContext())))
        return signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakePlaceBuffersPass() {
  return std::make_unique<PlaceBuffersPass>();
}
