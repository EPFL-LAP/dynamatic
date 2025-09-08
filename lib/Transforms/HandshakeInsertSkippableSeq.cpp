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
using BlockControlDepsMap = ControlDependenceAnalysis::BlockControlDepsMap;
using BlockIndexing = ftd::BlockIndexing;

namespace {

struct HandshakeInsertSkippableSeqPass
    : public dynamatic::impl::HandshakeInsertSkippableSeqBase<
          HandshakeInsertSkippableSeqPass> {

  // HandshakeInsertSkippableSeqPass(const std::string NStr) {};

  void runDynamaticPass() override;

  void handleFuncOp(FuncOp funcOp, MLIRContext *ctx);
};
} // namespace

class FTDBoolExpressions {
  BoolExpression *supp, *skip, *prodCons;

public:
  FTDBoolExpressions(){};

  FTDBoolExpressions(BoolExpression *supp, BoolExpression *skip,
                     BoolExpression *prodCons) {
    this->supp = supp;
    this->skip = skip;
    this->prodCons = prodCons;
  }

  BoolExpression *getSupp() { return supp; }

  BoolExpression *getSkip() { return skip; }

  BoolExpression *getProdcons() { return prodCons; }

  SmallVector<BoolExpression *> getAsVector() { return {supp, skip, prodCons}; }
};

using FTDBoolExpForPair =
    DenseMap<StringRef, DenseMap<StringRef, FTDBoolExpressions>>;

class FTDConditionValues {
  Value supp, skip, prodCons;

public:
  FTDConditionValues(SmallVector<Value> values) {
    supp = values[0];
    skip = values[1];
    prodCons = values[2];
  }

  Value getSupp() { return supp; }

  Value getSkip() { return skip; }

  Value getProdCons() { return prodCons; }

  SmallVector<Value> getAsVector() { return {supp, skip, prodCons}; }
};

class FuncOpInformation {

private:
  FuncOp funcOp;
  BlockIndexing blockIndexing;

public:
  FuncOpInformation(FuncOp funcOp) : blockIndexing(funcOp.getBody()) {
    this->funcOp = funcOp;
  }

  Block *getStartBlock() { return &funcOp.getBody().front(); }

  Value getStartSignal() { return (Value)funcOp.getArguments().back(); }

  BlockControlDepsMap getCdAnalysis() {
    BlockControlDepsMap cdAnalysis =
        ControlDependenceAnalysis(funcOp.getBody()).getAllBlockDeps();
    return cdAnalysis;
  }

  BlockIndexing getBlockIndexing() { return blockIndexing; }
};

/// These maps are used to distribute the start signal to all blocks in the CFG.
DenseMap<Block *, Value> startCopies;
std::unordered_set<Block *> visited;

/// This initializes the startCopies map with the start signal for the start
/// block.
void initializeStartCopies(FuncOpInformation funcOpInformation) {
  Block *block = funcOpInformation.getStartBlock();
  Value startSignal = funcOpInformation.getStartSignal();
  startCopies[block] = startSignal;
}

/// This function traverses the function and finds all memory accesses.
MemAccesses findMemAccessesInFunc(FuncOp funcOp) {

  MemAccesses memAccesses;

  for (BlockArgument arg : funcOp.getArguments()) {
    llvm::errs() << "[traversing arguments]" << arg << "\n";
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
  llvm::errs() << "num of block: " << opBBNum << "\n";

  std::optional<Block *> blockOptional =
      blockIndexing.getBlockFromIndex(opBBNum);
  if (blockOptional)
    return blockOptional.value();
  else {
    llvm::errs() << "Error: Block not found for operation: " << *op << "\n";
    return nullptr;
  }
}

/// This function calculates the FTD conditions for a pair of memory accesses.
FTDBoolExpressions calculateFTDConditions(Block *predecessorBlock,
                                          Block *successorBlock,
                                          std::string kernelName,
                                          FuncOpInformation funcOpInformation,
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

  std::ofstream outFile("./integration-test/" + kernelName + "/out/" +
                        kernelName + ".txt");

  outFile << "Supp: " << fSuppress->toString() << "\n";
  outFile << "Reg:  " << fRegen->toString() << "\n";
  outFile << "Prod: " << fProd1->toString() << "\n";
  outFile << "Cons: " << fCons1->toString() << "\n";

  return FTDBoolExpressions(fSuppress, fRegen, fProdAndCons);
}

/// This function calculates the FTD conditions for all pairs of memory
/// accesses that depend on each other. It uses the `calculateFTDConditions`
/// function to do so.
FTDBoolExpForPair calculateFtdConditionsForEachPair(
    DenseMap<StringRef, Operation *> &memAccesses, std::string kernelName,
    FuncOpInformation funcOpInformation, ConversionPatternRewriter &rewriter) {

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
    // auto bb = block->front().getAttrOfType<mlir::IntegerAttr>(BB_ATTR_NAME);

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
/// value which is delayed by a constant value.
/// It does two type of delays: 0 to N-1 and 1 to N.
std::tuple<SmallVector<Value>, SmallVector<Value>>
getNDelayedValues(Value initialVal, Value constVal, Operation *BBOp, unsigned N,
                  ConversionPatternRewriter &rewriter) {
  Value prevResult = initialVal;
  SmallVector<Value> delayedVals = {initialVal};
  SmallVector<Value> extraDelayedVals;

  SmallVector<Value, 2> values;

  unsigned effective_N = N;
  if (N == 0)
    effective_N = 1;
  for (unsigned i = 0; i < effective_N; i++) {
    // values = {prevResult, constVal};
    // handshake::MergeOp mergeOp =
    //     rewriter.create<handshake::MergeOp>(BBOp->getLoc(), values);
    // inheritBB(BBOp, mergeOp);

    // if (i != N - 1)
    //   delayedVals.push_back(mergeOp->getResult(0));
    // extraDelayedVals.push_back(mergeOp->getResult(0));

    // prevResult = mergeOp->getResult(0);

    handshake::InitOp initOp =
        rewriter.create<handshake::InitOp>(BBOp->getLoc(), prevResult);
    inheritBB(BBOp, initOp);
    // mlir::BoolAttr initTokenAttr = mlir::BoolAttr::get(ctx, true);
    // mlir::NamedAttrList attrList;
    // attrList.append("INIT_TOKEN", initTokenAttr);

    // mlir::DictionaryAttr paramsAttr =
    //     mlir::DictionaryAttr::get(context, attrList);

    // auto dictionaryAttr = initOp->getAttrDictionary();
    // dictionaryAttr = dictionaryAttr.set("params", paramsAttr);

    // setDialectAttr(initOp, ["INIT_TOKEN"] =); //

    // initOp
    //     ->setDialectAttr<INIT_TOKEN>()

    if (i != N - 1)
      delayedVals.push_back(initOp.getResult());
    extraDelayedVals.push_back(initOp.getResult());

    prevResult = initOp.getResult();
  }

  return std::tuple<SmallVector<Value>, SmallVector<Value>>(delayedVals,
                                                            extraDelayedVals);
}

/// This function checks if the two operations are in the same basic block.
bool AreOpsinSameBB(Operation *first, Operation *second) {
  int firstBBNum = getBBNumberFromOp(first);
  int secondBBNum = getBBNumberFromOp(second);
  return firstBBNum == secondBBNum;
}

/// This function checks if the consumer operation is before the producer
bool isInitialConsWithoutProd(Operation *prod, Operation *cons) {
  if (!AreOpsinSameBB(prod, cons))
    return false;
  return cons->isBeforeInBlock(prod);
}

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
    Operation *opPointer, bool addInitialTrueToSkip, Value startSignalInBB,
    ConversionPatternRewriter &rewriter) {

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
                                    Value regenCond, Operation *BBOp,
                                    ConversionPatternRewriter &rewriter) {
  SmallVector<Value> results;
  for (Value mainValue : mainValues) {
    SmallVector<Value, 2> muxOpValues = {mainValue, mainValue};
    handshake::MuxOp muxOp = rewriter.create<handshake::MuxOp>(
        BBOp->getLoc(), mainValue.getType(), regenCond, muxOpValues);
    inheritBB(BBOp, muxOp);

    handshake::ConditionalBranchOp conditionalBranchOp =
        rewriter.create<handshake::ConditionalBranchOp>(
            BBOp->getLoc(), regenCond, muxOp.getResult());
    inheritBB(BBOp, conditionalBranchOp);

    muxOp.setOperand(2, conditionalBranchOp.getResult(0));
    results.push_back(muxOp.getResult());
  }

  return results;
}

/// This function gates the channel value by the control value.
/// It creates a `Join` operation to combine the channel value and the control.
Value gateChannelValue(Value channelValue, Value gatingValue, Operation *BBOp,
                       ConversionPatternRewriter &rewriter) {
  // handshake::UnbundleOp unbundleOp =
  //     rewriter.create<handshake::UnbundleOp>(BBOp->getLoc(), channelValue);
  // inheritBB(BBOp, unbundleOp);

  // SmallVector<Value, 2> joinOpValues = {unbundleOp.getResult(0),
  // gatingValue}; handshake::JoinOp joinOp =
  //     rewriter.create<handshake::JoinOp>(BBOp->getLoc(), joinOpValues);
  // inheritBB(BBOp, joinOp);
  // ValueRange *ab = new ValueRange();
  // handshake::ChannelType ch =
  //     handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  // handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(
  //     BBOp->getLoc(), joinOp.getResult(), unbundleOp.getResult(1), *ab, ch);
  // inheritBB(BBOp, bundleOp);
  // return bundleOp.getResult(0);

  /// need to decide
  if (isa<ControlType>(gatingValue.getType())) {
    handshake::GateOp gateOp = rewriter.create<handshake::GateOp>(
        BBOp->getLoc(), channelValue, gatingValue);
    inheritBB(BBOp, gateOp);
    return gateOp.getResult();
  } else {
    handshake::BlockerOp blockerOp = rewriter.create<handshake::BlockerOp>(
        BBOp->getLoc(), ValueRange{channelValue, gatingValue});
    inheritBB(BBOp, blockerOp);
    return blockerOp.getResult();
  }
}

/// This condition insets suppresses in front ot the main values based on the
/// given conditions.
/// The function is used twice: once for inserting the suppress block and once
/// for the `Conditional Sequentializer` component.
SmallVector<Value> insertBranches(SmallVector<Value> mainValues,
                                  SmallVector<Value> conds, Operation *BBOp,
                                  ConversionPatternRewriter &rewriter) {
  SmallVector<Value> results;
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)) {
    /// ManualBuff
    bool manualBuff_insertBranches =
        true; // Best execution time with manual buffer present
    handshake::BufferOp bufferOp;
    handshake::ConditionalBranchOp conditionalBranchOp;
    if (manualBuff_insertBranches) {
      bufferOp = rewriter.create<handshake::BufferOp>(
          BBOp->getLoc(), cond, 5, BufferType::FIFO_BREAK_NONE);
      inheritBB(BBOp, bufferOp);
      conditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(
          BBOp->getLoc(), bufferOp.getResult(), mainValue);
    } else {
      conditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(
          BBOp->getLoc(), cond, mainValue);
    }
    inheritBB(BBOp, conditionalBranchOp);
    results.push_back(conditionalBranchOp.getResult(1));
  }
  return results;
}

/// This function returns the start signal if the real value is a control type.
/// Otherwise, it creates a dummy constant value.
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

/// This function creates the suppression block for the given main values.
SmallVector<Value> insertSuppressBlock(
    SmallVector<Value> mainValues, Value predecessorOpDoneSignal,
    Value suppressCond, Operation *predecessorOpPointer,
    Operation *succcessorOpPointer, Value startSignalInPredecessorBB,
    unsigned N, ConversionPatternRewriter &rewriter) {

  // SmallVector<Value> diffTokens = {predecessorOpDoneSignal};
  // Value dummyValue =
  //     getDummyValue(predecessorOpDoneSignal, startSignalInPredecessorBB,
  //                   predecessorOpPointer, rewriter);
  // unsigned effective_N_for_suppress = N - 1;
  // if (isInitialConsWithoutProd(predecessorOpPointer, succcessorOpPointer)) {
  //   effective_N_for_suppress = N;
  // }
  // if (effective_N_for_suppress > 0) {
  //   diffTokens.append(effective_N_for_suppress, dummyValue);
  // }

  // handshake::MergeOp mergeOp = rewriter.create<handshake::MergeOp>(
  //     predecessorOpPointer->getLoc(), diffTokens);
  // inheritBB(predecessorOpPointer, mergeOp);

  unsigned effective_N_for_suppress = N - 1;
  if (isInitialConsWithoutProd(predecessorOpPointer, succcessorOpPointer)) {
    effective_N_for_suppress = N;
  }

  Value prevResult = predecessorOpDoneSignal;
  for (unsigned i = 0; i < effective_N_for_suppress; i++) {
    handshake::InitOp initOp = rewriter.create<handshake::InitOp>(
        predecessorOpPointer->getLoc(), prevResult);
    inheritBB(predecessorOpPointer, initOp);
    prevResult = initOp.getResult();
  }

  /// ManualBuff (Init)
  // unsigned effective_N = N;
  // if (N == 0)
  //   effective_N = 1;

  // Value next_value = mergeOp.getResult();
  // if (effective_N > 1) {
  //   handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
  //       predecessorOpPointer->getLoc(), mergeOp.getResult(), effective_N - 1,
  //       BufferType::FIFO_BREAK_NONE);
  //   inheritBB(predecessorOpPointer, bufferOp);
  //   next_value = bufferOp.getResult();
  // }

  Value gatedSuppressCond = gateChannelValue(suppressCond, prevResult,
                                             predecessorOpPointer, rewriter);

  handshake::ConditionalBranchOp conditionalBranchOp =
      rewriter.create<handshake::ConditionalBranchOp>(
          predecessorOpPointer->getLoc(), suppressCond, gatedSuppressCond);
  inheritBB(predecessorOpPointer, conditionalBranchOp);

  SmallVector<Value, 2> muxOpValues = {suppressCond,
                                       conditionalBranchOp.getResult(0)};
  handshake::MuxOp muxOp = rewriter.create<handshake::MuxOp>(
      predecessorOpPointer->getLoc(), suppressCond.getType(), suppressCond,
      muxOpValues);
  inheritBB(predecessorOpPointer, muxOp);

  SmallVector<Value> conds;

  unsigned effective_N = N;
  if (N == 0)
    effective_N = 1;

  for (unsigned i = 0; i < effective_N; i++) {
    /// ManualBuff
    handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
        predecessorOpPointer->getLoc(), muxOp.getResult(), 3,
        BufferType::FIFO_BREAK_NONE);
    inheritBB(predecessorOpPointer, bufferOp);
    conds.push_back(bufferOp.getResult());
  }
  return insertBranches(mainValues, conds, predecessorOpPointer, rewriter);
}

/// This function creates the skip condition for a pair of memory accesses.
SmallVector<Value> createSkipConditionForPair(
    Value predecessorOpDoneSignal, Operation *predecessorOpPointer,
    Operation *successorOpPointer, SmallVector<Value> delayedAddresses,
    Value startSignalInBB, FTDBoolExpressions ftdConditions,
    BlockIndexing blockIndexing, unsigned N,
    ConversionPatternRewriter &rewriter) {

  FTDConditionValues ftdValues = constructCircuitForAllConditions(
      ftdConditions, blockIndexing, predecessorOpPointer,
      AreOpsinSameBB(predecessorOpPointer, successorOpPointer), startSignalInBB,
      rewriter);

  // SmallVector<Value> diffTokens = {predecessorOpDoneSignal};
  // Value dummyValue = getDummyValue(predecessorOpDoneSignal, startSignalInBB,
  //                                  predecessorOpPointer, rewriter);
  // // diffTokens.append(N, conditionalBranchOp2.getResult(0));
  // if (N > 0) {
  //   // mark
  //   unsigned effective_N = N;
  //   if (isInitialConsWithoutProd(predecessorOpPointer, successorOpPointer)) {
  //     effective_N = N + 1;
  //   }
  //   diffTokens.append(effective_N, dummyValue);
  // }

  // handshake::MergeOp mergeOp = rewriter.create<handshake::MergeOp>(
  //     predecessorOpPointer->getLoc(), diffTokens);
  // inheritBB(predecessorOpPointer, mergeOp);

  unsigned effective_N = N;
  if (isInitialConsWithoutProd(predecessorOpPointer, successorOpPointer)) {
    effective_N = N + 1;
  }
  Value prevResult = predecessorOpDoneSignal;
  for (unsigned i = 0; i < effective_N; i++) {
    handshake::InitOp initOp = rewriter.create<handshake::InitOp>(
        predecessorOpPointer->getLoc(), prevResult);
    inheritBB(predecessorOpPointer, initOp);
    prevResult = initOp.getResult();
  }

  /// ManualBuff (Init)
  // bool manualBuff_skip_cond =
  //     true; // Best execution time with manual buffer present
  // handshake::BufferOp bufferOp;
  // handshake::ConditionalBranchOp conditionalBranchOp;
  // if (manualBuff_skip_cond) {
  //   bufferOp = rewriter.create<handshake::BufferOp>(
  //       predecessorOpPointer->getLoc(), mergeOp.getResult(), N,
  //       BufferType::FIFO_BREAK_NONE);
  //   inheritBB(predecessorOpPointer, bufferOp);
  //   conditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(
  //       successorOpPointer->getLoc(), ftdValues.getSupp(),
  //       bufferOp.getResult());
  // } else {
  //   conditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(
  //       successorOpPointer->getLoc(), ftdValues.getSupp(),
  //       mergeOp.getResult());
  // }
  // inheritBB(successorOpPointer, conditionalBranchOp);

  handshake::ConditionalBranchOp conditionalBranchOp =
      rewriter.create<handshake::ConditionalBranchOp>(
          successorOpPointer->getLoc(), ftdValues.getSupp(), prevResult);
  inheritBB(successorOpPointer, conditionalBranchOp);

  // handshake::UnbundleOp unbundleOp = rewriter.create<handshake::UnbundleOp>(
  //     predecessorOpPointer->getLoc(), successorOpPointer->getOperand(0));
  // inheritBB(predecessorOpPointer, unbundleOp);

  // SmallVector<Value, 2> JoinOpValues = {conditionalBranchOp.getResult(1),
  //                                       unbundleOp.getResult(0)};
  // handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(
  //     predecessorOpPointer->getLoc(), JoinOpValues);
  // inheritBB(predecessorOpPointer, joinOp);

  // ValueRange *ab = new ValueRange();
  // handshake::ChannelType ch =
  //     handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  // handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(
  //     predecessorOpPointer->getLoc(), joinOp.getResult(),
  //     unbundleOp.getResult(1), *ab, ch);
  // inheritBB(predecessorOpPointer, bundleOp);

  // Value gatedSuccessorOpaddr = bundleOp.getResult(0);

  // need to decide

  // second comment
  // handshake::BlockerOp blockerOp = rewriter.create<handshake::BlockerOp>(
  //     predecessorOpPointer->getLoc(),
  //     ValueRange{successorOpPointer->getOperand(0),
  //                conditionalBranchOp.getResult(1)});
  // inheritBB(predecessorOpPointer, blockerOp);

  Value gatedSuccessorOpaddr = gateChannelValue(
      successorOpPointer->getOperand(0), conditionalBranchOp.getResult(1),
      predecessorOpPointer, rewriter);

  SmallVector<Value> delayedAddressesAfterRegen = delayedAddresses;
  if (ftdConditions.getSupp()->boolMinimize()->type !=
      experimental::boolean::ExpressionType::Zero) {
    delayedAddressesAfterRegen = insertRegenBlock(
        delayedAddresses, ftdValues.getSkip(), predecessorOpPointer, rewriter);
  }

  SmallVector<Value> delayedAddressesAfterSuppress = delayedAddressesAfterRegen;
  if (ftdConditions.getSupp()->boolMinimize()->type !=
      experimental::boolean::ExpressionType::Zero) {
    delayedAddressesAfterSuppress =
        insertSuppressBlock(delayedAddressesAfterRegen, predecessorOpDoneSignal,
                            ftdValues.getSupp(), predecessorOpPointer,
                            successorOpPointer, startSignalInBB, N, rewriter);
  }

  SmallVector<Value> skipConditions;
  for (Value delayedAddress : delayedAddressesAfterSuppress) {
    handshake::CmpIOp cmpIOp = rewriter.create<handshake::CmpIOp>(
        predecessorOpPointer->getLoc(), CmpIPredicate::ne, gatedSuccessorOpaddr,
        delayedAddress);
    inheritBB(predecessorOpPointer, cmpIOp);
    skipConditions.push_back(cmpIOp.getResult());
  }
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
    // Value loadResult = loadOp->getResult(0);
    // Location loc = loadOp->getLoc();
    // handshake::UnbundleOp unbundleOp =
    //     rewriter.create<handshake::UnbundleOp>(loc, loadResult);
    // inheritBB(loadOp, unbundleOp);
    // return unbundleOp.getResult(0);
    return loadOp->getResult(0);
  } else if (auto storeOp = dyn_cast<handshake::StoreOp>(memOp)) {
    return storeOp->getResult(2);
  } else {
    assert(false && "Unsupported memory operation");
    return nullptr;
  }
}

/// This function creates the skip conditions for all pairs of memory accesses.
/// This means that it creates the left side of the circuit for each pair.
/// This includes the delay generator which is shared for the same predecessor.
/// The regeneration/suppression block and the skip condition generator are
/// created specifically for each pair in `createSkipConditionForPair`.
SkipConditionForPair createSkipConditionsForAllPairs(
    MemAccesses &memAccesses, FTDBoolExpForPair &ftdConditionsForEachPair,
    FuncOpInformation funcOpInformation, std::vector<unsigned> Nvector,
    ConversionPatternRewriter &rewriter) {

  SkipConditionForPair skipConditionForEachPair;
  Value startSignal = funcOpInformation.getStartSignal();

  unsigned NvectorIndex = 0;
  unsigned N;

  for (auto [predecessorOpName, predecessorOpPointer] : memAccesses) {
    rewriter.setInsertionPointToStart(predecessorOpPointer->getBlock());
    Value predecessorOpDoneSignal =
        getDoneSignalFromMemoryOp(predecessorOpPointer, rewriter);
    Value predecessorOpAddr = predecessorOpPointer->getOperand(0);

    BlockIndexing blockIndexing = funcOpInformation.getBlockIndexing();
    Block *predecessorBlock =
        getBlockFromOp(predecessorOpPointer, blockIndexing);

    Block *startBlock = funcOpInformation.getStartBlock();

    Value startSignalInBB = distributStartSignalToDstBlock(
        startSignal, startBlock, predecessorBlock, true, rewriter);

    // This dummy constant is the dummy value for addresses used to create delay
    handshake::ConstantOp dummyConstOp = rewriter.create<handshake::ConstantOp>(
        predecessorOpPointer->getLoc(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 1000), startSignalInBB);
    inheritBB(predecessorOpPointer, dummyConstOp);

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

        for (MemDependenceAttr dependency : deps.getDependencies()) {
          if (!dependency.getIsActive())
            continue;

          N = Nvector[NvectorIndex];
          NvectorIndex++;

          if (N != 0) {

            llvm::errs() << "[SKIP][INFO] predecessorOpName: "
                         << predecessorOpName << " N: " << N << "\n";

            auto bothDelayedAddresses =
                getNDelayedValues(predecessorOpAddr, dummyConstOp,
                                  predecessorOpPointer, N, rewriter);
            SmallVector<Value> delayedAddresses =
                std::get<0>(bothDelayedAddresses);
            SmallVector<Value> extraDelayedAddresses =
                std::get<1>(bothDelayedAddresses);

            StringRef successorOpName = dependency.getDstAccess();
            Operation *successorOpPointer = memAccesses[successorOpName];
            FTDBoolExpressions ftdConditions =
                ftdConditionsForEachPair[predecessorOpName][successorOpName];

            // If the first successor does not have a producer, we need to
            // use the extra delayed addresses.
            SmallVector<Value> effectiveDelayedAddresses = delayedAddresses;
            if (isInitialConsWithoutProd(predecessorOpPointer,
                                         successorOpPointer))
              effectiveDelayedAddresses = extraDelayedAddresses;

            SmallVector<Value> skipConditions = createSkipConditionForPair(
                predecessorOpDoneSignal, predecessorOpPointer,
                successorOpPointer, effectiveDelayedAddresses, startSignalInBB,
                ftdConditions, blockIndexing, N, rewriter);
            skipConditionForEachPair[predecessorOpName][successorOpName] =
                skipConditions;
          }
        }
      }
    }
  }
  llvm::errs() << "[SKIP][INFO] Created Skip Conditions\n";
  return skipConditionForEachPair;
}

/// This function creates the skip condtional skip component which is a mux
/// connected to a source operation.
Value createSkip(Value waitingToken, Value cond, Value startSignal,
                 Operation *predecessorOp,
                 ConversionPatternRewriter &rewriter) {
  handshake::SourceOp sourceOp =
      rewriter.create<handshake::SourceOp>(predecessorOp->getLoc());

  inheritBB(predecessorOp, sourceOp);
  SmallVector<Value, 2> muxOpValues = {waitingToken, sourceOp.getResult()};
  handshake::MuxOp muxOp = rewriter.create<handshake::MuxOp>(
      predecessorOp->getLoc(), waitingToken.getType(), cond, muxOpValues);
  inheritBB(predecessorOp, muxOp);

  return muxOp.getResult();
}

/// This function inserts the conditional skips using `createSkip`.
SmallVector<Value> insertConditionalSkips(SmallVector<Value> mainValues,
                                          SmallVector<Value> conds,
                                          Operation *predecessorOp,
                                          Value startSignal,
                                          ConversionPatternRewriter &rewriter) {
  SmallVector<Value> results;
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)) {
    Value skipValue =
        createSkip(mainValue, cond, startSignal, predecessorOp, rewriter);
    results.push_back(skipValue);
  }
  return results;
}

Value createWaitingSignalForPair(
    Value predecessorOpDoneSignal, SmallVector<Value> delayedDoneSignals,
    SmallVector<Value> conds, Operation *predecessorOp, Operation *successorOp,
    Value startSignal, Value startSignalInBB, FTDBoolExpressions ftdConditions,
    BlockIndexing blockIndexing, unsigned N,
    ConversionPatternRewriter &rewriter) {

  FTDConditionValues ftdValues = constructCircuitForAllConditions(
      ftdConditions, blockIndexing, predecessorOp,
      AreOpsinSameBB(predecessorOp, successorOp), startSignalInBB, rewriter);

  SmallVector<Value> delayedDoneSignalsAfterRegen = delayedDoneSignals;

  if (ftdConditions.getSkip()->boolMinimize()->type !=
      experimental::boolean::ExpressionType::Zero) {
    delayedDoneSignalsAfterRegen = insertRegenBlock(
        delayedDoneSignals, ftdValues.getSkip(), predecessorOp, rewriter);
  } else {
    llvm::errs() << "[SKIP][INFO] Skipping Regen\n";
  }

  SmallVector<Value> delayedDoneSignalsAfterSuppress =
      delayedDoneSignalsAfterRegen;
  if (ftdConditions.getSupp()->boolMinimize()->type !=
      experimental::boolean::ExpressionType::Zero) {
    delayedDoneSignalsAfterSuppress = insertSuppressBlock(
        delayedDoneSignalsAfterRegen, predecessorOpDoneSignal,
        ftdValues.getSupp(), predecessorOp, successorOp, startSignalInBB, N,
        rewriter);
  } else {
    llvm::errs() << "[SKIP][INFO] Skipping Suppression\n";
  }

  if (N == 0)
    return delayedDoneSignalsAfterSuppress[0];

  SmallVector<Value> branchedDoneSignals = insertBranches(
      delayedDoneSignalsAfterSuppress, conds, predecessorOp, rewriter);
  SmallVector<Value> conditionallySkippedDoneSignals = insertConditionalSkips(
      branchedDoneSignals, conds, successorOp, startSignal, rewriter);

  llvm::errs() << "lanat" << conds.size() << "  "
               << delayedDoneSignalsAfterSuppress.size() << "  "
               << branchedDoneSignals.size() << "  "
               << conditionallySkippedDoneSignals.size() << "\n";

  handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(
      predecessorOp->getLoc(), conditionallySkippedDoneSignals);
  inheritBB(predecessorOp, joinOp);

  return joinOp.getResult();
}

/// This function returns the inactivated version of a given dependency.
MemDependenceAttr getInactivatedDependency(MemDependenceAttr dependency) {
  MLIRContext *ctx = dependency.getContext();
  return MemDependenceAttr::get(ctx, dependency.getDstAccess(),
                                dependency.getLoopDepth(), false);
}

/// This function creates the waiting signals for all pairs of memory accesses.
/// This means that it creates the right side of the circuit for each pair.
/// This includes the delay generator which is shared for the same predecessor.
/// The regeneration/suppression block and the conditional sequentializer
/// are created specifically for each pair in `createWaitingSignalForPair`.
WaitingSignalForSucc createWaitingSignals(
    MemAccesses &memAccesses, SkipConditionForPair &skipConditionForEachPair,
    MLIRContext *ctx, FuncOpInformation funcOpInformation,
    FTDBoolExpForPair &ftdConditionsForEachPair, std::vector<unsigned> NVector,
    ConversionPatternRewriter &rewriter) {

  WaitingSignalForSucc waitingSignalsForEachSuccessor;
  BlockIndexing blockIndexing = funcOpInformation.getBlockIndexing();
  Value startSignal = funcOpInformation.getStartSignal();
  Block *startBlock = funcOpInformation.getStartBlock();

  unsigned NvectorIndex = 0;
  unsigned N;

  for (auto [predecessorOpName, predecessorOpPointer] : memAccesses) {
    rewriter.setInsertionPointToStart(predecessorOpPointer->getBlock());
    Value predecessorOpDoneSignal =
        getDoneSignalFromMemoryOp(predecessorOpPointer, rewriter);

    SmallVector<MemDependenceAttr> newDeps;
    if (auto deps =
            getDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer)) {

      for (MemDependenceAttr dependency : deps.getDependencies()) {
        if (!dependency.getIsActive()) {
          newDeps.push_back(dependency);
          continue;
        }

        N = NVector[NvectorIndex];
        NvectorIndex++;

        Block *predecessorBlock =
            getBlockFromOp(predecessorOpPointer, blockIndexing);

        Value startSignalInBB = distributStartSignalToDstBlock(
            startSignal, startBlock, predecessorBlock, true, rewriter);

        Value dummyValue =
            getDummyValue(predecessorOpDoneSignal, startSignalInBB,
                          predecessorOpPointer, rewriter);
        auto bothDelayedDoneSignals =
            getNDelayedValues(predecessorOpDoneSignal, dummyValue,
                              predecessorOpPointer, N, rewriter);
        SmallVector<Value> delayedDoneSignals =
            std::get<0>(bothDelayedDoneSignals);
        SmallVector<Value> extraDelayedDoneSignals =
            std::get<1>(bothDelayedDoneSignals);

        StringRef successorName = dependency.getDstAccess();
        Operation *successorOpPointer = memAccesses[successorName];

        SmallVector<Value> conds =
            skipConditionForEachPair[predecessorOpName][successorName];

        FTDBoolExpressions ftdConditions =
            ftdConditionsForEachPair[predecessorOpName][successorName];

        SmallVector<Value> effectiveDelayedDoneSignals = delayedDoneSignals;
        if (isInitialConsWithoutProd(predecessorOpPointer, successorOpPointer))
          effectiveDelayedDoneSignals = extraDelayedDoneSignals;

        Value waitingSignal = createWaitingSignalForPair(
            predecessorOpDoneSignal, effectiveDelayedDoneSignals, conds,
            predecessorOpPointer, successorOpPointer, startSignal,
            startSignalInBB, ftdConditions, blockIndexing, N, rewriter);
        waitingSignalsForEachSuccessor[successorName].push_back(waitingSignal);

        newDeps.push_back(getInactivatedDependency(dependency));
      }
      setDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer, ctx,
                                             newDeps);
    }
    setDialectAttr<MemInterfaceAttr>(predecessorOpPointer, ctx);
  }
  llvm::errs() << "[SKIP][INFO] Created Waiting Signals\n";
  return waitingSignalsForEachSuccessor;
}

void gateAddress(Operation *op, SmallVector<Value> waitingValues,
                 ConversionPatternRewriter &rewriter, Location loc) {
  Value address = op->getOperand(0);
  // handshake::UnbundleOp unbundleOp =
  //     rewriter.create<handshake::UnbundleOp>(loc, address);
  // inheritBB(op, unbundleOp);
  // waitingValues.push_back(unbundleOp.getResult(0));
  // handshake::JoinOp joinOp =
  //     rewriter.create<handshake::JoinOp>(loc, waitingValues);
  // inheritBB(op, joinOp);
  // ValueRange *ab = new ValueRange();
  // handshake::ChannelType ch =
  //     handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  // handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(
  //     loc, joinOp.getResult(), unbundleOp.getResult(1), *ab, ch);
  // inheritBB(op, bundleOp);

  if (isa<ControlType>(waitingValues[0].getType())) {
    handshake::GateOp gateOp = rewriter.create<handshake::GateOp>(
        loc, address, ValueRange(waitingValues));
    inheritBB(op, gateOp);
    op->setOperand(0, gateOp.getResult());
  } else {
    waitingValues.insert(waitingValues.begin(), address);
    handshake::BlockerOp blockerOp =
        rewriter.create<handshake::BlockerOp>(loc, ValueRange(waitingValues));
    inheritBB(op, blockerOp);
    op->setOperand(0, blockerOp.getResult());
  }
}

void gateAllSuccessorAccesses(
    MemAccesses &memAccesses,
    WaitingSignalForSucc &waitingSignalsForEachSuccessor,
    ConversionPatternRewriter &rewriter) {
  for (auto [dstAccess, waitingSignals] : waitingSignalsForEachSuccessor) {
    Operation *op = memAccesses[dstAccess];
    gateAddress(op, waitingSignals, rewriter, op->getLoc());
  }
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

  std::vector<unsigned> NVector = getNVector(NStr);

  MemAccesses memAccesses;
  SkipConditionForPair skipConditionForEachPair;
  WaitingSignalForSucc waitingSignalsForEachSuccessor;
  FTDBoolExpForPair ftdConditionsForEachPair;

  initializeStartCopies(funcOpInformation);

  memAccesses = findMemAccessesInFunc(funcOp);

  ftdConditionsForEachPair = calculateFtdConditionsForEachPair(
      memAccesses, kernelName, funcOpInformation, rewriter);

  skipConditionForEachPair =
      createSkipConditionsForAllPairs(memAccesses, ftdConditionsForEachPair,
                                      funcOpInformation, NVector, rewriter);

  waitingSignalsForEachSuccessor = createWaitingSignals(
      memAccesses, skipConditionForEachPair, ctx, funcOpInformation,
      ftdConditionsForEachPair, NVector, rewriter);

  gateAllSuccessorAccesses(memAccesses, waitingSignalsForEachSuccessor,
                           rewriter);
}

void HandshakeInsertSkippableSeqPass::runDynamaticPass() {

  mlir::ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  ConversionPatternRewriter rewriter(ctx);

  for (auto funcOp : modOp.getOps<handshake::FuncOp>()) {
    if (failed(cfg::restoreCfStructure(funcOp, rewriter)))
      signalPassFailure();

    handleFuncOp(funcOp, ctx);

    // funcOp.print(llvm::errs());
    if (failed(cfg::flattenFunction(funcOp)))
      signalPassFailure();
  }

  llvm::errs() << "done! \n";
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeInsertSkippableSeq(const std::string &NStr,
                                             const std::string &kernelName) {
  return std::make_unique<HandshakeInsertSkippableSeqPass>();
}
