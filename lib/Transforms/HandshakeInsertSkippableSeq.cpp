//===- HandshakeInsertSkippableSeq.cpp - Out with LSQs ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeInsertSkippableSeq.h"
#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "experimental/Support/BooleanLogic/BDD.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/CFGAnnotation.h"
#include "experimental/Support/FtdImplementation.h"
#include "experimental/Support/FtdSupport.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <fstream>
#include <unordered_set>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::boolean;
using namespace dynamatic::experimental::ftd;

using MemAccesses = DenseMap<StringRef, Operation *>;
using SkipConditionForPair =
    DenseMap<StringRef, DenseMap<StringRef, SmallVector<Value>>>;
using WaitingSignalForSucc = DenseMap<StringRef, SmallVector<Value>>;
using IsWaitingSignalForSuccDirect = DenseMap<StringRef, SmallVector<bool>>;
using BlockControlDepsMap = ControlDependenceAnalysis::BlockControlDepsMap;
using BlockIndexing = ftd::BlockIndexing;

namespace {

std::ofstream outFile;
DenseMap<Operation *, std::vector<int>> consumerOpAndOperandIndexForFTD;
IsWaitingSignalForSuccDirect isWaitingSignalForSuccDirect;

struct HandshakeInsertSkippableSeqPass
    : public dynamatic::impl::HandshakeInsertSkippableSeqBase<
          HandshakeInsertSkippableSeqPass> {

  // HandshakeInsertSkippableSeqPass(const std::string NStr) {};

  void runDynamaticPass() override;

  void handleFuncOp(FuncOp funcOp, MLIRContext *ctx);
};
} // namespace

class FTDBoolExpressions {
  BoolExpression *supp, *regen, *prodCons;
  bool needsShanon;

public:
  FTDBoolExpressions(){};

  FTDBoolExpressions(BoolExpression *supp, BoolExpression *regen,
                     BoolExpression *prodCons, bool needsShanon) {
    this->supp = supp;
    this->regen = regen;
    this->prodCons = prodCons;
    this->needsShanon = needsShanon;
  }

  BoolExpression *getSupp() { return supp; }

  BoolExpression *getRegen() { return regen; }

  BoolExpression *getProdcons() { return prodCons; }

  bool DoesNeedShanon() { return needsShanon; }

  void SetNeedsShanon(bool needsShanon) { this->needsShanon = needsShanon; }

  SmallVector<BoolExpression *> getAsVector() {
    return {supp, regen, prodCons};
  }
};

using FTDBoolExpForPair =
    DenseMap<StringRef, DenseMap<StringRef, FTDBoolExpressions>>;

class FTDConditionValues {
  Value supp, regen, prodCons;

public:
  FTDConditionValues(SmallVector<Value> values) {
    supp = values[0];
    regen = values[1];
    prodCons = values[2];
  }

  Value getSupp() { return supp; }

  Value getRegen() { return regen; }

  Value getProdCons() { return prodCons; }

  SmallVector<Value> getAsVector() { return {supp, regen, prodCons}; }
};

class FuncOpInformation {

private:
  FuncOp funcOp;
  Region &region;
  mlir::DominanceInfo domInfo;
  BlockIndexing blockIndexing;
  mlir::CFGLoopInfo loopInfo;
  BlockControlDepsMap cdAnalysis;

public:
  FuncOpInformation(FuncOp funcOp)
      : region(funcOp.getBody()), domInfo(),
        loopInfo(domInfo.getDomTree(&region)), blockIndexing(region),
        cdAnalysis(ControlDependenceAnalysis(region).getAllBlockDeps()) {
    this->funcOp = funcOp;
  }

  Block *getStartBlock() { return &funcOp.getBody().front(); }

  Value getStartSignal() { return (Value)funcOp.getArguments().back(); }

  BlockControlDepsMap getCdAnalysis() { return cdAnalysis; }

  BlockIndexing getBlockIndexing() { return blockIndexing; }

  Region &getRegion() { return region; }

  mlir::CFGLoopInfo &getLoopInfo() { return loopInfo; }

  FuncOp getFuncOp() { return funcOp; }
};

class DependenceGraphEdge {
public:
  Operation *srcOp;
  Operation *dstOp;
  int comparatoreNum;
  DependenceGraphEdge(Operation *srcOp, Operation *dstOp, int comparatoreNum) {
    this->srcOp = srcOp;
    this->dstOp = dstOp;
    this->comparatoreNum = comparatoreNum;
  }
};

using DependenceGraph = SmallVector<DependenceGraphEdge, 2>;
DependenceGraph dependenceGraph;

/// These maps are used to distribute the start signal to all blocks in the CFG.
DenseMap<Block *, Value> startCopies;
std::unordered_set<Block *> visited;

/// This initializes the startCopies map with the start signal for the start
/// block.
void initializeStartCopies(FuncOpInformation &funcOpInformation) {
  Block *block = funcOpInformation.getStartBlock();
  Value startSignal = funcOpInformation.getStartSignal();
  startCopies[block] = startSignal;
}

/// This function traverses the function and finds all memory accesses.
MemAccesses findMemAccessesInFunc(FuncOp funcOp) {

  MemAccesses memAccesses;

  for (BlockArgument arg : funcOp.getArguments()) {
    // llvm::errs() << "[traversing arguments]" << arg << "\n";
    if (auto memref = dyn_cast<TypedValue<mlir::MemRefType>>(arg)) {
      auto memrefUsers = memref.getUsers();

      assert(std::distance(memrefUsers.begin(), memrefUsers.end()) <= 1 &&
             "expected at most one memref user");

      Operation *memOp = *memrefUsers.begin();

      handshake::LSQOp lsqOp;
      if (lsqOp = dyn_cast<handshake::LSQOp>(memOp); !lsqOp) {
        auto mcOp = cast<handshake::MemoryControllerOp>(memOp);

        MCPorts mcPorts = mcOp.getPorts();

        if (!mcPorts.connectsToLSQ()) {
          continue;
        }
        lsqOp = mcPorts.getLSQPort().getLSQOp();
      }

      LSQPorts lsqPorts = lsqOp.getPorts();
      for (LSQGroup &group : lsqPorts.getGroups()) {
        for (MemoryPort &port : group->accessPorts) {
          memAccesses[getUniqueName(port.portOp)] = port.portOp;
        }
      }
    }
  }

  return memAccesses;
}

/// This function finds the basic block number from the operation.
int getBBNumberFromOp(Operation *op) {
  std::string BB_STRING = "handshake.bb = ";

  std::string printed;
  llvm::raw_string_ostream os1(printed);
  os1 << *op;

  int start = printed.find(BB_STRING);

  std::string word = printed.substr(start + BB_STRING.length());
  int end = word.find(' ');
  std::string num_str = word.substr(0, end);

  return std::stoi(num_str);
}

/// This function gets the block object from the operation using the block
/// indexing.
Block *getBlockFromOp(Operation *op, BlockIndexing blockIndexing) {

  int opBBNum = getBBNumberFromOp(op);
  // llvm::errs() << "num of block: " << opBBNum << "\n";

  std::optional<Block *> blockOptional =
      blockIndexing.getBlockFromIndex(opBBNum);
  if (blockOptional)
    return blockOptional.value();
  else {
    llvm::errs() << "Error: Block not found for operation: " << *op << "\n";
    return nullptr;
  }
}

bool doesNeedShanon(Block *first, Block *second, DenseSet<Block *> firstDeps,
                    DenseSet<Block *> secondDeps) {
  return firstDeps.contains(second) || secondDeps.contains(first);
}

void handleConditionForLoop(Block *predecessorBlock, Block *successorBlock,
                            BoolExpression *&fSuppress,
                            FuncOpInformation &funcOpInformation,
                            ConversionPatternRewriter &rewriter) {
  BlockIndexing blockIndexing = funcOpInformation.getBlockIndexing();
  mlir::CFGLoopInfo &loopInfo = funcOpInformation.getLoopInfo();

  for (CFGLoop *loop = loopInfo.getLoopFor(predecessorBlock); loop;
       loop = loop->getParentLoop()) {

    if (!loop->contains(successorBlock)) {
      if (Block *loopExit = loop->getExitingBlock(); loopExit) {
        auto *exitCondition =
            getBlockLoopExitCondition(loopExit, loop, loopInfo, blockIndexing);

        fSuppress =
            BoolExpression::boolOr(fSuppress, exitCondition->boolNegate());

        llvm::errs() << "Added loop exit condition for suppress signal: "
                     << exitCondition->toString() << "to "
                     << fSuppress->toString() << "\n";
      }
    }
  }
}

void updateRegenConditionForLoop(Block *predecessorBlock, Block *successorBlock,
                                 BoolExpression *&fRegen,
                                 FuncOpInformation &funcOpInformation,
                                 ConversionPatternRewriter &rewriter) {
  BlockIndexing blockIndexing = funcOpInformation.getBlockIndexing();
  mlir::CFGLoopInfo &loopInfo = funcOpInformation.getLoopInfo();

  for (CFGLoop *loop = loopInfo.getLoopFor(successorBlock); loop;
       loop = loop->getParentLoop()) {

    if (!loop->contains(predecessorBlock)) {
      if (Block *loopExit = loop->getExitingBlock(); loopExit) {
        auto *exitCondition =

            getBlockLoopExitCondition(loopExit, loop, loopInfo, blockIndexing);

        fRegen = BoolExpression::boolOr(fRegen, exitCondition->boolNegate());

        llvm::errs() << "Added loop exit condition for suppress signal: "
                     << exitCondition->toString() << "to " << fRegen->toString()
                     << "\n";
      }
    }
  }
}

/// This function calculates the FTD conditions for a pair of memory accesses.
FTDBoolExpressions calculateFTDConditions(Block *predecessorBlock,
                                          Block *successorBlock,
                                          std::string kernelName,
                                          FuncOpInformation &funcOpInformation,
                                          ConversionPatternRewriter &rewriter) {

  BlockControlDepsMap cdAnalysis = funcOpInformation.getCdAnalysis();
  Block *startBlock = funcOpInformation.getStartBlock();
  BlockIndexing blockIndexing = funcOpInformation.getBlockIndexing();

  DenseSet<Block *> predControlDeps =
      cdAnalysis[predecessorBlock].forwardControlDeps;

  DenseSet<Block *> succControlDeps =
      cdAnalysis[successorBlock].forwardControlDeps;

  // Get rid of common entries in the two sets
  // eliminateCommonBlocks(predControlDeps, succControlDeps);

  BoolExpression *fProd1 = enumeratePaths(startBlock, predecessorBlock,
                                          blockIndexing, predControlDeps);
  BoolExpression *fCons1 = enumeratePaths(startBlock, successorBlock,
                                          blockIndexing, succControlDeps);

  BoolExpression *fProdAndCons = BoolExpression::boolAnd(fProd1, fCons1);

  BoolExpression *fProd2 = enumeratePaths(startBlock, predecessorBlock,
                                          blockIndexing, predControlDeps);
  BoolExpression *fCons2 = enumeratePaths(startBlock, successorBlock,
                                          blockIndexing, succControlDeps);

  BoolExpression *fSuppress =
      BoolExpression::boolAnd(fProd2, fCons2->boolNegate());

  BoolExpression *fProd3 = enumeratePaths(startBlock, predecessorBlock,
                                          blockIndexing, predControlDeps);
  BoolExpression *fCons3 = enumeratePaths(startBlock, successorBlock,
                                          blockIndexing, succControlDeps);

  BoolExpression *fRegen =
      BoolExpression::boolAnd(fProd3->boolNegate(), fCons3);

  BoolExpression *ajib = enumeratePaths(predecessorBlock, successorBlock,
                                        blockIndexing, succControlDeps);

  mlir::CFGLoopInfo &loopInfo = funcOpInformation.getLoopInfo();
  bool producingGtUsing =
      loopInfo.getLoopFor(predecessorBlock) &&
      !loopInfo.getLoopFor(predecessorBlock)->contains(successorBlock);

  if (producingGtUsing) {
    handleConditionForLoop(predecessorBlock, successorBlock, fSuppress,
                           funcOpInformation, rewriter);
  }

  bool usingGtProducing =
      loopInfo.getLoopFor(successorBlock) &&
      !loopInfo.getLoopFor(successorBlock)->contains(predecessorBlock);
  if (usingGtProducing) {
    updateRegenConditionForLoop(predecessorBlock, successorBlock, fRegen,
                                funcOpInformation, rewriter);
  }

  outFile << "----------------------------------------------\n";
  outFile << "Prod: " << fProd1->toString() << "\n";
  outFile << "Cons: " << fCons1->toString() << "\n";
  outFile << "Supp: " << fSuppress->toString() << "\n";
  outFile << "Reg:  " << fRegen->toString() << "\n";
  outFile << "Ajib: " << ajib->toString() << "\n";
  outFile << "Pred Block: " << getBBNumberFromOp(&predecessorBlock->front())
          << "\n";
  outFile << "Succ Block: " << getBBNumberFromOp(&successorBlock->front())
          << "\n";
  outFile << "----------------------------------------------\n";
  outFile.flush();

  // llvm::errs() << "Supp: " << fSuppress->toString() << "\n";
  // llvm::errs() << "Reg:  " << fRegen->toString() << "\n";
  // llvm::errs() << "Prod: " << fProd1->toString() << "\n";
  // llvm::errs() << "Cons: " << fCons1->toString() << "\n";

  return FTDBoolExpressions(fSuppress, fRegen, fProdAndCons,
                            doesNeedShanon(predecessorBlock, successorBlock,
                                           predControlDeps, succControlDeps));
}

/// This function calculates the FTD conditions for all pairs of memory
/// accesses that depend on each other. It uses the `calculateFTDConditions`
/// function to do so.
FTDBoolExpForPair calculateFtdConditionsForEachPair(
    DenseMap<StringRef, Operation *> &memAccesses, std::string kernelName,
    FuncOpInformation &funcOpInformation, ConversionPatternRewriter &rewriter) {

  FTDBoolExpForPair ftdConditionsForEachPair;

  for (auto [predecessorOpName, predecessorOpPointer] : memAccesses) {
    if (auto deps =
            getDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer)) {

      for (MemDependenceAttr dependency : deps.getDependencies()) {
        if (!dependency.getIsActive())
          continue;

        StringRef successorOpName = dependency.getDstAccess();
        Operation *successorOpPointer = memAccesses[successorOpName];

        Block *predecessorBlock = getBlockFromOp(
            predecessorOpPointer, funcOpInformation.getBlockIndexing());
        Block *successorBlock = getBlockFromOp(
            successorOpPointer, funcOpInformation.getBlockIndexing());

        // llvm::errs() << "------------\n";
        // llvm::errs() << getBBNumberFromOp(predecessorOpPointer) << "\n";
        // llvm::errs() << getBBNumberFromOp(successorOpPointer) << "\n";

        FTDBoolExpressions boolConditions =
            calculateFTDConditions(predecessorBlock, successorBlock, kernelName,
                                   funcOpInformation, rewriter);
        ftdConditionsForEachPair[predecessorOpName][successorOpName] =
            boolConditions;
      }
    }
  }
  return ftdConditionsForEachPair;
}

/// This function distributes the start signal to all blocks in the CFG.
Value distributStartSignalToDstBlock(Value lastStartCopy, Block *block,
                                     Block *dstBlock, bool firstExec,
                                     ConversionPatternRewriter &rewriter) {
  if (firstExec)
    visited = {};
  if (block == dstBlock)
    return lastStartCopy;

  visited.insert(block);
  if (startCopies.contains(block))
    lastStartCopy = startCopies[block];
  else {
    /// ManualBuff (trick)
    // auto bb =
    // block->front().getAttrOfType<mlir::IntegerAttr>(BB_ATTR_NAME);

    handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
        block->front().getLoc(), lastStartCopy, 1, BufferType::FIFO_BREAK_NONE);
    inheritBB(&block->front(), bufferOp);
    lastStartCopy = bufferOp.getResult();
  }

  for (Block *successor : block->getSuccessors()) {
    if (!visited.count(successor)) {
      Value returnVal = distributStartSignalToDstBlock(
          lastStartCopy, successor, dstBlock, false, rewriter);
      if (returnVal != nullptr)
        return returnVal;
    }
  }
  return nullptr;
}

/// This function checks if there is at least one active dependency in the
/// dependencies list.
bool hasAtLeastOneActiveDep(MemDependenceArrayAttr deps) {
  for (MemDependenceAttr dependency : deps.getDependencies()) {
    if (dependency.getIsActive())
      return true;
  }
  return false;
}

/// This function returns the delayed values (N values) for the given initial
/// value.
/// It does two type of delays: 0 to N-1 and 1 to N.
SmallVector<Value> getNDelayedValues(Value initialVal, Operation *BBOp,
                                     unsigned N,
                                     SmallVector<Operation *> &opList,
                                     ConversionPatternRewriter &rewriter) {

  rewriter.setInsertionPoint(BBOp);

  handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
      BBOp->getLoc(), initialVal, 1, BufferType::FIFO_BREAK_NONE);
  inheritBB(BBOp, bufferOp);
  opList.push_back(bufferOp);

  Value prevResult = bufferOp.getResult();
  SmallVector<Value> delayedVals = {prevResult};
  // SmallVector<Value> extraDelayedVals;

  SmallVector<Value, 2> values;

  unsigned effective_N = N;
  if (N == 0)
    effective_N = 1;
  for (unsigned i = 0; i < effective_N; i++) {
    handshake::InitOp initOp =
        rewriter.create<handshake::InitOp>(BBOp->getLoc(), prevResult);
    inheritBB(BBOp, initOp);
    opList.push_back(initOp);

    if (i != N - 1)
      delayedVals.push_back(initOp.getResult());
    // extraDelayedVals.push_back(initOp.getResult());

    prevResult = initOp.getResult();
  }

  return delayedVals;
}

Value addExtraSourceToDoneForRegen(Value doneSignal, Value regenCond,
                                   Operation *BBOp,
                                   SmallVector<Operation *> &opList,
                                   ConversionPatternRewriter &rewriter) {

  handshake::SourceOp sourceOp =
      rewriter.create<handshake::SourceOp>(BBOp->getLoc());
  inheritBB(BBOp, sourceOp);
  opList.push_back(sourceOp);

  handshake::ConditionalBranchOp conditionalBranchOp =
      rewriter.create<handshake::ConditionalBranchOp>(BBOp->getLoc(), regenCond,
                                                      sourceOp.getResult());
  inheritBB(BBOp, conditionalBranchOp);
  opList.push_back(conditionalBranchOp);

  // buffer for deadlock
  handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
      BBOp->getLoc(), conditionalBranchOp.getTrueResult(), 5,
      BufferType::FIFO_BREAK_NONE);
  inheritBB(BBOp, bufferOp);
  opList.push_back(bufferOp);

  handshake::MergeOp mergeOp = rewriter.create<handshake::MergeOp>(
      BBOp->getLoc(), ValueRange{
                          doneSignal,
                          bufferOp.getResult(),
                      });
  inheritBB(BBOp, mergeOp);
  opList.push_back(mergeOp);

  return mergeOp.getResult();
}

/// This function checks if the two operations are in the same basic block.
bool AreOpsinSameBB(Operation *first, Operation *second) {
  int firstBBNum = getBBNumberFromOp(first);
  int secondBBNum = getBBNumberFromOp(second);
  return firstBBNum == secondBBNum;
}

/// This function checks if the consumer operation is before the producer
bool isInitialConsWithoutProdInSameBB(Operation *prod, Operation *cons) {
  mlir::DominanceInfo domInfo;
  if (!AreOpsinSameBB(prod, cons)) {
    // llvm::errs() << getBBNumberFromOp(prod) << getBBNumberFromOp(cons)
    //              << "))))\n";
    llvm::errs() << domInfo.dominates(cons, prod) << "))))\n";
    return domInfo.dominates(cons, prod);
  }

  // llvm::errs() << getBBNumberFromOp(prod) << getBBNumberFromOp(cons)
  //              << "((((\n";
  return cons->isBeforeInBlock(prod);
}

// Value constructConditionNormally(BDD *bdd) {

//   boolVariabletoCircuit();

//   Value node = boolExpressionToCircuit(rewriter, bdd->boolVariable, block,
//   bi);

//   Value falseValue =
//   constructConditionNormally(bdd->successors.value().first); Value
//   TrueValue = constructConditionNormally(bdd->successors.value().second);
// }

/// This function creates the circuit for a specific FTD condition.
Value constructCircuitForCondition(BoolExpression *fBool,
                                   BlockIndexing blockIndexing, Block *block,
                                   Operation *opPointer,
                                   ConversionPatternRewriter &rewriter) {
  fBool = fBool->boolMinimize();

  if (fBool->type == experimental::boolean::ExpressionType::One) {
    handshake::SourceOp sourceOpConstOp =
        rewriter.create<handshake::SourceOp>(opPointer->getLoc());

    inheritBB(opPointer, sourceOpConstOp);

    handshake::ConstantOp falseConstOp = rewriter.create<handshake::ConstantOp>(
        opPointer->getLoc(), rewriter.getBoolAttr(true), sourceOpConstOp);
    inheritBB(opPointer, falseConstOp);
    return falseConstOp.getResult();
  }
  if (fBool->type != experimental::boolean::ExpressionType::Zero) {

    std::set<std::string> blocks = fBool->getVariables();

    std::vector<std::string> cofactorList(blocks.begin(), blocks.end());
    BDD *bdd = buildBDD(fBool, cofactorList);
    Value condValue = bddToCircuit(rewriter, bdd, block, blockIndexing);
    return condValue;
  }

  handshake::SourceOp sourceOpConstOp =
      rewriter.create<handshake::SourceOp>(opPointer->getLoc());
  inheritBB(opPointer, sourceOpConstOp);

  handshake::ConstantOp falseConstOp = rewriter.create<handshake::ConstantOp>(
      opPointer->getLoc(), rewriter.getBoolAttr(false), sourceOpConstOp);
  inheritBB(opPointer, falseConstOp);
  return falseConstOp.getResult();
}

/// This function creates all of the FTD conditions for a pair of memory
/// accesses.
FTDConditionValues constructCircuitForAllConditions(
    FTDBoolExpressions ftdBoolExpressions, BlockIndexing blockIndexing,
    Operation *opPointer, ConversionPatternRewriter &rewriter) {

  Block *block = getBlockFromOp(opPointer, blockIndexing);

  SmallVector<Value> results;
  for (BoolExpression *ftdCond : ftdBoolExpressions.getAsVector()) {
    Value value = constructCircuitForCondition(ftdCond, blockIndexing, block,
                                               opPointer, rewriter);

    results.push_back(value);
  }
  return FTDConditionValues(results);
}

/// This function creates the regeneration block for the given main values
SmallVector<Value> insertRegenBlock(SmallVector<Value> mainValues,
                                    Value regenCond,
                                    bool initialConsWithoutProd,
                                    Operation *BBOp,
                                    SmallVector<Operation *> &opList,
                                    ConversionPatternRewriter &rewriter) {

  if (initialConsWithoutProd) {
    regenCond = rewriter.create<handshake::InitOp>(BBOp->getLoc(), regenCond);
  }
  SmallVector<Value> results;
  for (Value mainValue : mainValues) {
    // The second operand will change later
    SmallVector<Value, 2> muxOpValues = {mainValue, mainValue};
    handshake::MuxOp muxOp = rewriter.create<handshake::MuxOp>(
        BBOp->getLoc(), mainValue.getType(), regenCond, muxOpValues);
    inheritBB(BBOp, muxOp);
    opList.push_back(muxOp);

    handshake::InitOp initOp =
        rewriter.create<handshake::InitOp>(BBOp->getLoc(), muxOp.getResult());
    inheritBB(BBOp, initOp);
    opList.push_back(initOp);

    handshake::ConditionalBranchOp conditionalBranchOp =
        rewriter.create<handshake::ConditionalBranchOp>(
            BBOp->getLoc(), regenCond, initOp.getResult());
    inheritBB(BBOp, conditionalBranchOp);
    opList.push_back(conditionalBranchOp);

    muxOp.setOperand(2, conditionalBranchOp.getTrueResult());
    results.push_back(muxOp.getResult());
  }

  return results;
}

/// This function gates the channel value by the control value.
/// It creates a `Join` operation to combine the channel value and the
/// control.
Value gateChannelValue(
    Value channelValue, Value gatingValue, Operation *BBOp,
    DenseMap<OpOperand *, SmallVector<Value>> &dependenciesMapForPhiNetwork,
    SmallVector<Operation *> &createdOperations,
    ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPoint(BBOp);
  handshake::GateOp gateOp = rewriter.create<handshake::GateOp>(
      BBOp->getLoc(), channelValue, gatingValue);
  inheritBB(BBOp, gateOp);
  createdOperations.push_back(gateOp);
  // llvm::errs() << "Gating value: " << *gatingValue.getDefiningOp() << " - "
  //              << gatingValue << "\n";
  dependenciesMapForPhiNetwork[&gateOp->getOpOperand(1)] = {gatingValue};
  consumerOpAndOperandIndexForFTD[gateOp].push_back(1);
  return gateOp.getResult();
}

/// This condition insets suppresses in front ot the main values based on the
/// given conditions.
/// The function is used twice: once for inserting the suppress block and once
/// for the `Conditional Sequentializer` component.
SmallVector<Value> insertBranches(
    SmallVector<Value> mainValues, SmallVector<Value> conds, Operation *BBOp,
    DenseMap<OpOperand *, SmallVector<Value>> &dependenciesMapForPhiNetwork,
    SmallVector<Operation *> &ops, ConversionPatternRewriter &rewriter) {
  SmallVector<Value> results;
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)) {
    /// ManualBuff
    // bool manualBuff_insertBranches = true;
    // handshake::BufferOp bufferOp;
    // handshake::ConditionalBranchOp conditionalBranchOp;
    // if (manualBuff_insertBranches) {
    //   bufferOp = rewriter.create<handshake::BufferOp>(
    //       BBOp->getLoc(), cond, 5, BufferType::FIFO_BREAK_NONE);
    //   inheritBB(BBOp, bufferOp);
    //   ops.push_back(bufferOp);
    //   conditionalBranchOp =
    //   rewriter.create<handshake::ConditionalBranchOp>(
    //       BBOp->getLoc(), bufferOp.getResult(), mainValue);
    // } else {
    //   conditionalBranchOp =
    //   rewriter.create<handshake::ConditionalBranchOp>(
    //       BBOp->getLoc(), cond, mainValue);
    // }

    handshake::ConditionalBranchOp conditionalBranchOp =
        rewriter.create<handshake::ConditionalBranchOp>(BBOp->getLoc(), cond,
                                                        mainValue);
    inheritBB(BBOp, conditionalBranchOp);
    ops.push_back(conditionalBranchOp);
    results.push_back(conditionalBranchOp.getResult(1));
    dependenciesMapForPhiNetwork[&conditionalBranchOp->getOpOperand(1)] = {
        mainValue};
    // llvm::errs() << "Branching value: " << *cond.getDefiningOp() << " - "
    //              << cond << "\n";
    // llvm::errs() << "Branching value: " << *mainValue.getDefiningOp() << " -
    // "
    //              << mainValue << "\n";
    consumerOpAndOperandIndexForFTD[conditionalBranchOp].push_back(1);
  }
  return results;
}

/// This function returns the start signal if the real value is a control
/// type. Otherwise, it creates a dummy constant value.
Value getDummyValue(Value realValue, Value startSignalInBB,
                    Operation *BBOpPointer,
                    ConversionPatternRewriter &rewriter) {
  if (isa<ControlType>(realValue.getType())) {
    return startSignalInBB;
  } else {
    handshake::ConstantOp dummyConstOp = rewriter.create<handshake::ConstantOp>(
        BBOpPointer->getLoc(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 1000), startSignalInBB);
    inheritBB(BBOpPointer, dummyConstOp);
    return dummyConstOp.getResult();
  }
}

void addDrawingAttrToList(ArrayRef<Operation *> operations, StringRef attr) {
  for (auto op : operations) {
    auto drawAttr = handshake::DrawingAttr::get(op->getContext(), attr);
    op->setAttr("drawing", drawAttr);
  }
}

/// This function creates the suppression block for the given main values.
SmallVector<Value> insertSuppressBlock(
    SmallVector<Value> mainValues, Value predecessorOpDoneSignal,
    Value suppressCond, Value regenCond, Operation *predecessorOpPointer,
    Operation *succcessorOpPointer, unsigned N,
    SmallVector<Operation *> &opList, ConversionPatternRewriter &rewriter) {

  unsigned effective_N_for_suppress = N - 1;
  if (isInitialConsWithoutProdInSameBB(predecessorOpPointer,
                                       succcessorOpPointer)) {
    effective_N_for_suppress = N;
  }

  Value prevResult = predecessorOpDoneSignal;
  for (unsigned i = 0; i < effective_N_for_suppress; i++) {
    handshake::InitOp initOp = rewriter.create<handshake::InitOp>(
        predecessorOpPointer->getLoc(), prevResult);
    inheritBB(predecessorOpPointer, initOp);
    opList.push_back(initOp);
    prevResult = initOp.getResult();
  }

  Value mergedWithExtra = prevResult;
  if (regenCond) {
    mergedWithExtra = addExtraSourceToDoneForRegen(
        prevResult, regenCond, predecessorOpPointer, opList, rewriter);
  }

  DenseMap<OpOperand *, SmallVector<Value>> dependenciesMapForPhiNetwork;
  Value gatedSuppressCond =
      gateChannelValue(suppressCond, mergedWithExtra, predecessorOpPointer,
                       dependenciesMapForPhiNetwork, opList, rewriter);

  handshake::ConditionalBranchOp conditionalBranchOpForGated =
      rewriter.create<handshake::ConditionalBranchOp>(
          predecessorOpPointer->getLoc(), suppressCond, gatedSuppressCond);
  inheritBB(predecessorOpPointer, conditionalBranchOpForGated);
  opList.push_back(conditionalBranchOpForGated);

  handshake::ConditionalBranchOp conditionalBranchOpForNormal =
      rewriter.create<handshake::ConditionalBranchOp>(
          predecessorOpPointer->getLoc(), suppressCond, suppressCond);
  inheritBB(predecessorOpPointer, conditionalBranchOpForNormal);
  opList.push_back(conditionalBranchOpForNormal);

  SmallVector<Value, 2> muxOpValues = {
      conditionalBranchOpForNormal.getFalseResult(),
      conditionalBranchOpForGated.getTrueResult()};
  handshake::MuxOp muxOp = rewriter.create<handshake::MuxOp>(
      predecessorOpPointer->getLoc(), suppressCond.getType(), suppressCond,
      muxOpValues);
  inheritBB(predecessorOpPointer, muxOp);
  opList.push_back(muxOp);

  SmallVector<Value> conds;

  unsigned effective_N = N;
  if (N == 0)
    effective_N = 1;

  // for (unsigned i = 0; i < effective_N; i++) {
  //   /// ManualBuff
  //   handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
  //       predecessorOpPointer->getLoc(), muxOp.getResult(), 3,
  //       BufferType::FIFO_BREAK_NONE);
  //   inheritBB(predecessorOpPointer, bufferOp);
  //   conds.push_back(bufferOp.getResult());
  // }

  for (unsigned i = 0; i < effective_N; i++)
    conds.push_back(muxOp.getResult());

  DenseMap<OpOperand *, SmallVector<Value>> emptyMap;
  auto result = insertBranches(mainValues, conds, predecessorOpPointer,
                               emptyMap, opList, rewriter);

  return result;
}

/// This function creates the skip condition for a pair of memory accesses.
SmallVector<Value> createSkipConditionForPair(
    Value predecessorOpDoneSignal, Operation *predecessorOpPointer,
    Operation *successorOpPointer, SmallVector<Value> delayedAddresses,
    FTDBoolExpressions ftdConditions, BlockIndexing blockIndexing, unsigned N,
    DenseMap<OpOperand *, SmallVector<Value>> &dependenciesMapForPhiNetwork,
    ConversionPatternRewriter &rewriter) {

  SmallVector<Operation *> skipConditionGeneratorOps;

  // FTDConditionValues ftdValues = constructCircuitForAllConditions(
  //     ftdConditions, blockIndexing, predecessorOpPointer, rewriter);

  // unsigned effective_N = N;
  // if (isInitialConsWithoutProdInSameBB(predecessorOpPointer,
  //                                      successorOpPointer)) {
  //   effective_N = N + 1;
  // }
  // paolo join
  // rewriter.setInsertionPoint(predecessorOpPointer);
  // Value prevResult = predecessorOpDoneSignal;
  // for (unsigned i = 0; i < N; i++) {
  //   handshake::InitOp initOp = rewriter.create<handshake::InitOp>(
  //       predecessorOpPointer->getLoc(), prevResult);
  //   inheritBB(predecessorOpPointer, initOp);
  //   skipConditionGeneratorOps.push_back(initOp);
  //   prevResult = initOp.getResult();
  // }

  // Value mergedWithExtra = prevResult;
  // if (ftdConditions.getRegen()->boolMinimize()->type !=
  //     experimental::boolean::ExpressionType::Zero) {
  //   mergedWithExtra = addExtraSourceToDoneForRegen(
  //       prevResult, ftdValues.getRegen(), predecessorOpPointer,
  //       skipConditionGeneratorOps, rewriter);
  // }

  // Value suppressedWithSupp = mergedWithExtra;
  // if (ftdConditions.getSupp()->boolMinimize()->type !=
  //     experimental::boolean::ExpressionType::Zero) {
  //   handshake::ConditionalBranchOp conditionalBranchOp =
  //       rewriter.create<handshake::ConditionalBranchOp>(
  //           successorOpPointer->getLoc(), ftdValues.getSupp(),
  //           mergedWithExtra);
  //   inheritBB(successorOpPointer, conditionalBranchOp);
  //   skipConditionGeneratorOps.push_back(conditionalBranchOp);
  //   suppressedWithSupp = conditionalBranchOp.getFalseResult();
  // }

  // Value gatedSuccessorOpaddr = gateChannelValue(
  //     successorOpPointer->getOperand(0), prevResult, successorOpPointer,
  //     dependenciesMapForPhiNetwork, skipConditionGeneratorOps, rewriter);

  // SmallVector<Value> delayedAddressesAfterSuppress = delayedAddresses;
  // if (ftdConditions.getSupp()->boolMinimize()->type !=
  //     experimental::boolean::ExpressionType::Zero) {
  //   SmallVector<Operation *> suppressOpList;
  //   delayedAddressesAfterSuppress = insertSuppressBlock(
  //       delayedAddresses, predecessorOpDoneSignal, ftdValues.getSupp(),
  //       ftdValues.getRegen(), predecessorOpPointer, successorOpPointer, N,
  //       suppressOpList, rewriter);
  //   addDrawingAttrToList(suppressOpList, "Suppress_Block_Cond");
  // }

  // SmallVector<Value> delayedAddressesAfterRegen =
  // delayedAddressesAfterSuppress; if
  // (ftdConditions.getRegen()->boolMinimize()->type !=
  //     experimental::boolean::ExpressionType::Zero) {
  //   llvm::errs() << "ey khoda\n";
  //   SmallVector<Operation *> regenOpList;
  //   bool initialConsWithoutProd = isInitialConsWithoutProdInSameBB(
  //       predecessorOpPointer, successorOpPointer);
  //   delayedAddressesAfterRegen = insertRegenBlock(
  //       delayedAddressesAfterSuppress, ftdValues.getRegen(),
  //       initialConsWithoutProd, predecessorOpPointer, regenOpList, rewriter);
  //   llvm::errs() << "----" << regenOpList.size() << "\n";
  //   addDrawingAttrToList(regenOpList, "Regen_Block_Cond");
  // }

  SmallVector<Value> skipConditions;
  rewriter.setInsertionPoint(successorOpPointer);
  for (Value delayedAddress : delayedAddresses) {
    handshake::CmpIOp cmpIOp = rewriter.create<handshake::CmpIOp>(
        successorOpPointer->getLoc(), CmpIPredicate::ne,
        successorOpPointer->getOperand(0), delayedAddress);
    inheritBB(successorOpPointer, cmpIOp);

    // llvm::errs() << "**********  " << cmpIOp << "\n";
    skipConditionGeneratorOps.push_back(cmpIOp);
    skipConditions.push_back(cmpIOp.getResult());
    // llvm::errs() << "Comparing value: " << *delayedAddress.getDefiningOp()
    //              << " - " << delayedAddress << "\n";
    // llvm::errs() << *successorOpPointer << " \n  ey khoda\n";
    dependenciesMapForPhiNetwork[&cmpIOp->getOpOperand(1)].push_back(
        delayedAddress);
    consumerOpAndOperandIndexForFTD[cmpIOp].push_back(1);
  }

  addDrawingAttrToList(skipConditionGeneratorOps, "Condition_Generator");
  return skipConditions;
}

/// This function gets the done signal from a memory operation.
/// If its a load, it returns the data output, which is a channel.
/// However, if its a store, it returns the done signal which is a control
/// signal. This difference needs to be taken care of when using the done
/// signal.
Value getDoneSignalFromMemoryOp(Operation *memOp,
                                ConversionPatternRewriter &rewriter) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(memOp)) {
    Value loadResult = loadOp->getResult(1);
    Location loc = loadOp->getLoc();
    handshake::UnbundleOp unbundleOp =
        rewriter.create<handshake::UnbundleOp>(loc, loadResult);
    inheritBB(loadOp, unbundleOp);

    // ValueRange *ab = new ValueRange();
    // handshake::ChannelType ch =
    //     handshake::ChannelType::get(unbundleOp.getResult(1).getType());
    // handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(
    //     loc, unbundleOp.getResult(0), unbundleOp.getResult(1), *ab, ch);
    // inheritBB(loadOp, bundleOp);

    return unbundleOp.getResult(0);
    // return loadOp->getResult(0);
  } else if (auto storeOp = dyn_cast<handshake::StoreOp>(memOp)) {
    return storeOp->getResult(2);
  } else {
    assert(false && "Unsupported memory operation");
    return nullptr;
  }
}

/// This function creates the skip conditions for all pairs of memory
/// accesses. This means that it creates the left side of the circuit for each
/// pair. This includes the delay generator which is shared for the same
/// predecessor. The regeneration/suppression block and the skip condition
/// generator are created specifically for each pair in
/// `createSkipConditionForPair`.
SkipConditionForPair createSkipConditionsForAllPairs(
    MemAccesses &memAccesses, FTDBoolExpForPair &ftdConditionsForEachPair,
    FuncOpInformation &funcOpInformation, std::vector<unsigned> Nvector,
    ConversionPatternRewriter &rewriter) {

  SkipConditionForPair skipConditionForEachPair;
  DenseMap<OpOperand *, SmallVector<Value>> dependenciesMapForPhiNetwork;

  // Value startSignal = funcOpInformation.getStartSignal();

  unsigned NvectorIndex = 0;
  unsigned N;

  for (auto [predecessorOpName, predecessorOpPointer] : memAccesses) {
    rewriter.setInsertionPointToStart(predecessorOpPointer->getBlock());
    Value predecessorOpDoneSignal =
        getDoneSignalFromMemoryOp(predecessorOpPointer, rewriter);
    Value predecessorOpAddr = predecessorOpPointer->getOperand(0);

    BlockIndexing blockIndexing = funcOpInformation.getBlockIndexing();
    // Block *predecessorBlock =
    //     getBlockFromOp(predecessorOpPointer, blockIndexing);

    // Block *startBlock = funcOpInformation.getStartBlock();

    if (auto deps =
            getDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer)) {

      if (hasAtLeastOneActiveDep(deps)) {

        // /// ManualBuff (store)
        // handshake::BufferOp bufferOp =
        // rewriter.create<handshake::BufferOp>(
        //     predecessorOpPointer->getLoc(), predecessorOpAddr,
        //     ::TimingInfo::tehb(), 3);
        // inheritBB(predecessorOpPointer, bufferOp);
        // predecessorOpPointer->setOperand(0, bufferOp.getResult());

        SmallVector<StringRef> handledSuccessors;
        for (MemDependenceAttr dependency : deps.getDependencies()) {
          if (!dependency.getIsActive())
            continue;

          if (std::find(handledSuccessors.begin(), handledSuccessors.end(),
                        dependency.getDstAccess()) != handledSuccessors.end()) {
            continue;
          }

          N = Nvector[NvectorIndex];
          NvectorIndex++;

          if (NvectorIndex == Nvector.size()) {
            NvectorIndex = 0;
          }

          if (N != 0) {
            SmallVector<Operation *> addressDelayGenerator;

            // auto bothDelayedAddresses =
            //     getNDelayedValues(predecessorOpAddr, predecessorOpPointer, N,
            //                       addressDelayGenerator, rewriter);
            // addDrawingAttrToList(addressDelayGenerator,
            // "Addr_Delay_Generator"); SmallVector<Value> delayedAddresses =
            //     std::get<0>(bothDelayedAddresses);
            // SmallVector<Value> extraDelayedAddresses =
            //     std::get<1>(bothDelayedAddresses);

            rewriter.setInsertionPoint(predecessorOpPointer);
            Value prevResult = predecessorOpDoneSignal;
            for (unsigned i = 0; i < N; i++) {
              handshake::InitOp initOp = rewriter.create<handshake::InitOp>(
                  predecessorOpPointer->getLoc(), prevResult);
              inheritBB(predecessorOpPointer, initOp);
              prevResult = initOp.getResult();
            }

            rewriter.setInsertionPoint(predecessorOpPointer);
            handshake::GateOp gateOp = rewriter.create<handshake::GateOp>(
                predecessorOpPointer->getLoc(), predecessorOpAddr, prevResult);
            inheritBB(predecessorOpPointer, gateOp);

            SmallVector<Value> delayedAddresses =
                getNDelayedValues(gateOp.getResult(), predecessorOpPointer, N,
                                  addressDelayGenerator, rewriter);

            StringRef successorOpName = dependency.getDstAccess();
            Operation *successorOpPointer = memAccesses[successorOpName];
            FTDBoolExpressions ftdConditions =
                ftdConditionsForEachPair[predecessorOpName][successorOpName];

            // If the first successor does not have a producer, we need to
            // use the extra delayed addresses.
            SmallVector<Value> effectiveDelayedAddresses = delayedAddresses;
            // if (isInitialConsWithoutProdInSameBB(predecessorOpPointer,
            //                                      successorOpPointer))
            //   effectiveDelayedAddresses = extraDelayedAddresses;

            // paolo join

            // rewriter.setInsertionPoint(predecessorOpPointer);
            // Value prevResult = predecessorOpDoneSignal;
            // for (unsigned i = 0; i < N - 1; i++) {
            //   handshake::InitOp initOp = rewriter.create<handshake::InitOp>(
            //       predecessorOpPointer->getLoc(), prevResult);
            //   inheritBB(predecessorOpPointer, initOp);
            //   prevResult = initOp.getResult();
            // }

            // rewriter.setInsertionPoint(predecessorOpPointer);
            // handshake::GateOp gateOp = rewriter.create<handshake::GateOp>(
            //     predecessorOpPointer->getLoc(), delayedAddresses.back(),
            //     prevResult);
            // inheritBB(predecessorOpPointer, gateOp);

            SmallVector<Value> skipConditions = createSkipConditionForPair(
                predecessorOpDoneSignal, predecessorOpPointer,
                successorOpPointer, effectiveDelayedAddresses, ftdConditions,
                blockIndexing, N, dependenciesMapForPhiNetwork, rewriter);
            skipConditionForEachPair[predecessorOpName][successorOpName] =
                skipConditions;
            handledSuccessors.push_back(successorOpName);
          }
        }
      }
    }
  }

  // llvm::errs() << "daram miram too\n";
  createPhiNetworkDeps(funcOpInformation.getFuncOp().getRegion(), rewriter,
                       dependenciesMapForPhiNetwork);

  llvm::errs() << "[SKIP][INFO] Created Skip Conditions\n";
  return skipConditionForEachPair;
}

/// This function creates the skip condtional skip component which is a mux
/// connected to a source operation.
Value createSkip(Value waitingToken, Value cond, Operation *predecessorOp,
                 SmallVector<Operation *> &opList,
                 ConversionPatternRewriter &rewriter) {

  SmallVector<Value, 2> muxOpValues;
  handshake::SourceOp sourceOp =
      rewriter.create<handshake::SourceOp>(predecessorOp->getLoc());
  inheritBB(predecessorOp, sourceOp);
  opList.push_back(sourceOp);

  if (isa<ControlType>(waitingToken.getType())) {
    muxOpValues = {waitingToken, sourceOp};
  } else {
    handshake::ConstantOp constOp = rewriter.create<handshake::ConstantOp>(
        predecessorOp->getLoc(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 1000), sourceOp);
    inheritBB(predecessorOp, constOp);
    opList.push_back(constOp);
    muxOpValues = {waitingToken, constOp.getResult()};
  }

  handshake::MuxOp muxOp = rewriter.create<handshake::MuxOp>(
      predecessorOp->getLoc(), waitingToken.getType(), cond, muxOpValues);
  inheritBB(predecessorOp, muxOp);
  opList.push_back(muxOp);

  return muxOp.getResult();
}

/// This function inserts the conditional skips using `createSkip`.
SmallVector<Value> insertConditionalSkips(SmallVector<Value> mainValues,
                                          SmallVector<Value> conds,
                                          Operation *predecessorOp,
                                          SmallVector<Operation *> &opList,
                                          ConversionPatternRewriter &rewriter) {
  SmallVector<Value> results;
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)) {
    Value skipValue =
        createSkip(mainValue, cond, predecessorOp, opList, rewriter);
    results.push_back(skipValue);
  }
  return results;
}

Value joinValues(SmallVector<Value> valuesToJoin, Operation *BBOp,
                 ConversionPatternRewriter &rewriter) {

  if (valuesToJoin.size() == 1)
    return valuesToJoin[0];

  if (isa<ControlType>(valuesToJoin[0].getType()) &&
      isa<ControlType>(valuesToJoin[1].getType())) {
    handshake::JoinOp joinOp =
        rewriter.create<handshake::JoinOp>(BBOp->getLoc(), valuesToJoin);
    inheritBB(BBOp, joinOp);
    return joinOp.getResult();
  } else if (isa<ControlType>(valuesToJoin[1].getType())) {
    // extract the values beside the first one
    SmallVector<Value> values;
    for (unsigned i = 1; i < valuesToJoin.size(); i++) {
      values.push_back(valuesToJoin[i]);
    }

    handshake::GateOp gateOp = rewriter.create<handshake::GateOp>(
        BBOp->getLoc(), valuesToJoin[0], values);
    inheritBB(BBOp, gateOp);
    return gateOp.getResult();
  } else {
    handshake::BlockerOp blockerOp = rewriter.create<handshake::BlockerOp>(
        BBOp->getLoc(), ValueRange{valuesToJoin});
    inheritBB(BBOp, blockerOp);
    return blockerOp.getResult();
  }
}

Value createWaitingSignalForPair(
    Value predecessorOpDoneSignal, SmallVector<Value> delayedDoneSignals,
    SmallVector<Value> conds, Operation *predecessorOp, Operation *successorOp,
    Value startSignal, FTDBoolExpressions ftdConditions,
    BlockIndexing blockIndexing, unsigned N,
    DenseMap<OpOperand *, SmallVector<Value>> &dependenciesMapForPhiNetwork,
    ConversionPatternRewriter &rewriter) {

  SmallVector<Operation *> conditionalSequentializerOps;

  // FTDConditionValues ftdValues = constructCircuitForAllConditions(
  //     ftdConditions, blockIndexing, predecessorOp, rewriter);

  // SmallVector<Value> delayedDoneSignalsAfterSuppress = delayedDoneSignals;
  // if (ftdConditions.getSupp()->boolMinimize()->type !=
  //     experimental::boolean::ExpressionType::Zero) {
  //   llvm::errs() << "[SKIP][INFO] Inserting Suppression\n";
  //   SmallVector<Operation *> suppressOpList;
  //   delayedDoneSignalsAfterSuppress = insertSuppressBlock(
  //       delayedDoneSignals, predecessorOpDoneSignal, ftdValues.getSupp(),
  //       ftdValues.getRegen(), predecessorOp, successorOp, N, suppressOpList,
  //       rewriter);
  //   addDrawingAttrToList(suppressOpList, "Suppress_Block_Done");

  // } else {
  //   llvm::errs() << "[SKIP][INFO] Skipping Suppression\n";
  // }

  // SmallVector<Value> delayedDoneSignalsAfterRegen =
  //     delayedDoneSignalsAfterSuppress;

  // if (ftdConditions.getRegen()->boolMinimize()->type !=
  //     experimental::boolean::ExpressionType::Zero) {
  //   llvm::errs() << "[SKIP][INFO] Inserting Regen\n";
  //   SmallVector<Operation *> regenOpList;
  //   bool initialConsWithoutProd =
  //       isInitialConsWithoutProdInSameBB(predecessorOp, successorOp);
  //   delayedDoneSignalsAfterRegen = insertRegenBlock(
  //       delayedDoneSignalsAfterSuppress, ftdValues.getRegen(),
  //       initialConsWithoutProd, predecessorOp, regenOpList, rewriter);
  //   addDrawingAttrToList(regenOpList, "Regen_Block_Done");
  // } else {
  //   llvm::errs() << "[SKIP][INFO] Skipping Regen\n";
  // }

  if (N == 0)
    return delayedDoneSignals[0];

  rewriter.setInsertionPoint(successorOp);
  SmallVector<Value> branchedDoneSignals = insertBranches(
      delayedDoneSignals, conds, successorOp, dependenciesMapForPhiNetwork,
      conditionalSequentializerOps, rewriter);
  SmallVector<Value> conditionallySkippedDoneSignals =
      insertConditionalSkips(branchedDoneSignals, conds, successorOp,
                             conditionalSequentializerOps, rewriter);

  handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(
      predecessorOp->getLoc(), conditionallySkippedDoneSignals);
  inheritBB(predecessorOp, joinOp);
  conditionalSequentializerOps.push_back(joinOp);

  addDrawingAttrToList(conditionalSequentializerOps,
                       "Conditional_Sequentializer");
  return joinOp.getResult();

  // return joinValues(conditionallySkippedDoneSignals, predecessorOp,
  // rewriter);
}

/// This function returns the inactivated version of a given dependency.
MemDependenceAttr getInactivatedDependency(MemDependenceAttr dependency) {
  MLIRContext *ctx = dependency.getContext();
  return MemDependenceAttr::get(ctx, dependency.getDstAccess(),
                                dependency.getLoopDepth(), false);
}

/// This function creates the waiting signals for all pairs of memory
/// accesses. This means that it creates the right side of the circuit for
/// each pair. This includes the delay generator which is shared for the same
/// predecessor. The regeneration/suppression block and the conditional
/// sequentializer are created specifically for each pair in
/// `createWaitingSignalForPair`.
WaitingSignalForSucc createWaitingSignals(
    MemAccesses &memAccesses, SkipConditionForPair &skipConditionForEachPair,
    MLIRContext *ctx, FuncOpInformation &funcOpInformation,
    FTDBoolExpForPair &ftdConditionsForEachPair, std::vector<unsigned> NVector,
    ConversionPatternRewriter &rewriter) {

  WaitingSignalForSucc waitingSignalsForEachSuccessor;
  BlockIndexing blockIndexing = funcOpInformation.getBlockIndexing();
  Value startSignal = funcOpInformation.getStartSignal();

  DenseMap<OpOperand *, SmallVector<Value>> dependenciesMapForPhiNetwork;
  // Block *startBlock = funcOpInformation.getStartBlock();

  unsigned NvectorIndex = 0;
  unsigned N;

  for (auto [predecessorOpName, predecessorOpPointer] : memAccesses) {
    rewriter.setInsertionPointToStart(predecessorOpPointer->getBlock());
    Value predecessorOpDoneSignal =
        getDoneSignalFromMemoryOp(predecessorOpPointer, rewriter);

    SmallVector<StringRef> handledSuccessors;
    SmallVector<MemDependenceAttr> newDeps;
    if (auto deps =
            getDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer)) {

      for (MemDependenceAttr dependency : deps.getDependencies()) {
        if (!dependency.getIsActive()) {
          newDeps.push_back(dependency);
          continue;
        }

        if (std::find(handledSuccessors.begin(), handledSuccessors.end(),
                      dependency.getDstAccess()) != handledSuccessors.end()) {
          newDeps.push_back(getInactivatedDependency(dependency));
          continue;
        }

        N = NVector[NvectorIndex];
        NvectorIndex++;

        if (NvectorIndex == NVector.size()) {
          NvectorIndex = 0;
        }

        // Block *predecessorBlock =
        //     getBlockFromOp(predecessorOpPointer, blockIndexing);

        SmallVector<Operation *> doneDelayGenerator;

        // auto bothDelayedDoneSignals =
        //     getNDelayedValues(predecessorOpDoneSignal, predecessorOpPointer,
        //     N,
        //                       doneDelayGenerator, rewriter);
        // addDrawingAttrToList(doneDelayGenerator, "Done_Delay_Generator");
        // SmallVector<Value> delayedDoneSignals =
        //     std::get<0>(bothDelayedDoneSignals);
        // SmallVector<Value> extraDelayedDoneSignals =
        //     std::get<1>(bothDelayedDoneSignals);

        SmallVector<Value> delayedDoneSignals =
            getNDelayedValues(predecessorOpDoneSignal, predecessorOpPointer, N,
                              doneDelayGenerator, rewriter);

        StringRef successorName = dependency.getDstAccess();
        Operation *successorOpPointer = memAccesses[successorName];

        dependenceGraph.push_back(
            DependenceGraphEdge(predecessorOpPointer, successorOpPointer, N));

        SmallVector<Value> conds =
            skipConditionForEachPair[predecessorOpName][successorName];

        FTDBoolExpressions ftdConditions =
            ftdConditionsForEachPair[predecessorOpName][successorName];

        SmallVector<Value> effectiveDelayedDoneSignals = delayedDoneSignals;
        // if (isInitialConsWithoutProdInSameBB(predecessorOpPointer,
        //                                      successorOpPointer))
        //   effectiveDelayedDoneSignals = extraDelayedDoneSignals;

        Value waitingSignal = createWaitingSignalForPair(
            predecessorOpDoneSignal, effectiveDelayedDoneSignals, conds,
            predecessorOpPointer, successorOpPointer, startSignal,
            ftdConditions, blockIndexing, N, dependenciesMapForPhiNetwork,
            rewriter);
        waitingSignalsForEachSuccessor[successorName].push_back(waitingSignal);
        isWaitingSignalForSuccDirect[successorName].push_back(N == 0);

        newDeps.push_back(getInactivatedDependency(dependency));
        handledSuccessors.push_back(successorName);
      }
      setDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer, ctx,
                                             newDeps);
    }
    setDialectAttr<MemInterfaceAttr>(predecessorOpPointer, ctx);
  }
  createPhiNetworkDeps(funcOpInformation.getFuncOp().getRegion(), rewriter,
                       dependenciesMapForPhiNetwork);
  llvm::errs() << "[SKIP][INFO] Created Waiting Signals\n";
  return waitingSignalsForEachSuccessor;
}

void gateAddress(
    Operation *op, SmallVector<Value> waitingValues,
    ConversionPatternRewriter &rewriter, Location loc,
    SmallVector<bool> isDirect,
    DenseMap<OpOperand *, SmallVector<Value>> &dependenciesMapForPhiNetwork) {
  Value address = op->getOperand(0);

  rewriter.setInsertionPoint(op);
  handshake::GateOp gateOp =
      rewriter.create<handshake::GateOp>(loc, address, waitingValues);
  inheritBB(op, gateOp);
  op->setOperand(0, gateOp.getResult());

  for (auto [idx, value, isDirect] : llvm::enumerate(waitingValues, isDirect)) {
    if (isDirect) {
      dependenciesMapForPhiNetwork[&gateOp->getOpOperand(idx + 1)].push_back(
          value);
      consumerOpAndOperandIndexForFTD[gateOp].push_back(idx + 1);
    }
  }
}

void gateAllSuccessorAccesses(
    MemAccesses &memAccesses,
    WaitingSignalForSucc &waitingSignalsForEachSuccessor,
    FuncOpInformation &funcOpInformation, ConversionPatternRewriter &rewriter) {

  DenseMap<OpOperand *, SmallVector<Value>> dependenciesMapForPhiNetwork;

  for (auto [dstAccess, waitingSignals] : waitingSignalsForEachSuccessor) {
    Operation *op = memAccesses[dstAccess];

    auto isDirect = isWaitingSignalForSuccDirect[dstAccess];
    gateAddress(op, waitingSignals, rewriter, op->getLoc(), isDirect,
                dependenciesMapForPhiNetwork);
  }

  createPhiNetworkDeps(funcOpInformation.getFuncOp().getRegion(), rewriter,
                       dependenciesMapForPhiNetwork);
  llvm::errs() << "[SKIP][INFO] Gated Successor Accesses\n";
}

/// This function casts the string `NStr` to std::vector<unsigned> N.
std::vector<unsigned> getNVector(const std::string &NStr) {
  std::vector<unsigned> NVector;
  std::istringstream iss(NStr);
  std::string token;
  while (std::getline(iss, token, ',')) {
    unsigned N = std::stoul(token);
    NVector.push_back(N);
  }
  return NVector;
}

/// This function is the main function. It is responsible to insert the
/// necessary components for skippable sequencializing in every funcOp.
void HandshakeInsertSkippableSeqPass::handleFuncOp(FuncOp funcOp,
                                                   MLIRContext *ctx) {
  ConversionPatternRewriter rewriter(ctx);
  FuncOpInformation funcOpInformation(funcOp);

  outFile.open("./skip/" + kernelName + ".txt");

  std::vector<unsigned> NVector = getNVector(NStr);

  MemAccesses memAccesses;
  SkipConditionForPair skipConditionForEachPair;
  WaitingSignalForSucc waitingSignalsForEachSuccessor;
  FTDBoolExpForPair ftdConditionsForEachPair;
  std::vector<Operation *> consumerOpListForFTD;

  initializeStartCopies(funcOpInformation);

  memAccesses = findMemAccessesInFunc(funcOp);

  // ftdConditionsForEachPair = calculateFtdConditionsForEachPair(
  //     memAccesses, kernelName, funcOpInformation, rewriter);

  skipConditionForEachPair =
      createSkipConditionsForAllPairs(memAccesses, ftdConditionsForEachPair,
                                      funcOpInformation, NVector, rewriter);

  waitingSignalsForEachSuccessor = createWaitingSignals(
      memAccesses, skipConditionForEachPair, ctx, funcOpInformation,
      ftdConditionsForEachPair, NVector, rewriter);

  gateAllSuccessorAccesses(memAccesses, waitingSignalsForEachSuccessor,
                           funcOpInformation, rewriter);
}

// write dep graph to a dot file
void writeDepGraphToDotFile(const std::string &filename) {
  std::ofstream file(filename);
  file << "digraph G {\n";
  for (const auto &edge : dependenceGraph) {
    std::string predName = getUniqueName(edge.srcOp).str();
    std::string succName = getUniqueName(edge.dstOp).str();
    file << "  \"" << predName << "\" -> \"" << succName
         << "\" [label=\"N=" << edge.comparatoreNum << "\"];\n";
  }
  file << "}\n";
  file.close();
  llvm::errs() << "Dependency graph written to " << filename << "\n";
}

void runFTDOnSpecificConsumerOps(
    FuncOp funcOp, PatternRewriter &rewriter,
    std::vector<Operation *> (*ftdFunc)(PatternRewriter &, FuncOp &,
                                        Operation *, Value)) {
  std::vector<std::vector<Operation *>> allNewUnits;
  for (auto const [consumerOp, indices] : consumerOpAndOperandIndexForFTD)
    for (auto index : indices) {
      // llvm::errs() << "Running FTD on consumer op: " << *consumerOp
      //              << " - operand index: " << index << "\n";
      std::vector<Operation *> newUnits =
          ftdFunc(rewriter, funcOp, consumerOp, consumerOp->getOperand(index));
      allNewUnits.push_back(newUnits);
    }

  for (auto &someNewUnits : allNewUnits) {
    for (auto *unit : someNewUnits) {
      int i = 0;
      for (auto operand : unit->getOperands()) {
        consumerOpAndOperandIndexForFTD[unit].push_back(i);
        i++;
      }
    }
  }
}

void HandshakeInsertSkippableSeqPass::runDynamaticPass() {

  mlir::ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  ConversionPatternRewriter rewriter(ctx);

  for (auto funcOp : modOp.getOps<handshake::FuncOp>()) {
    if (failed(cfg::restoreCfStructure(funcOp, rewriter)))
      signalPassFailure();

    // internally calls `createPhiNetworkDeps`
    handleFuncOp(funcOp, ctx);

    std::vector<Operation *> newUnits;
    replaceMergeToGSA(funcOp, rewriter, newUnits);

    for (auto *unit : newUnits) {
      int i = 0;
      for (auto operand : unit->getOperands()) {
        consumerOpAndOperandIndexForFTD[unit].push_back(i);
        i++;
      }
    }

    runFTDOnSpecificConsumerOps(funcOp, rewriter, addRegenOperandConsumer);
    runFTDOnSpecificConsumerOps(funcOp, rewriter, addSuppOperandConsumer);

    experimental::cfg::markBasicBlocks(funcOp, rewriter);

    // open a file to dump funcOp
    llvm::errs() << "Dumping funcOp after inserting skippable seq:\n";

    std::error_code EC;
    llvm::raw_fd_ostream file("funcOp_after_inserting_skippable_seq.mlir", EC);
    funcOp.print(file);
    file.close();
    // funcOp.print(llvm::errs());
    if (failed(cfg::flattenFunction(funcOp)))
      signalPassFailure();
  }

  std::string extra = "";
  if (kernelName.find("memory") != std::string::npos)
    extra = "memory/";

  std::string path = "./integration-test/" + extra + kernelName + "/out/comp/" +
                     kernelName + "_DEP_G.dot";

  writeDepGraphToDotFile(path);
  llvm::errs() << "done! \n";
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeInsertSkippableSeq(const std::string &NStr,
                                             const std::string &kernelName) {
  return std::make_unique<HandshakeInsertSkippableSeqPass>();
}
