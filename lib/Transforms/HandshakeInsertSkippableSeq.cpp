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
  FTDBoolExpressions() {};

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
          llvm::errs() << "no LSQ so continue";
          continue;
        }
        lsqOp = mcPorts.getLSQPort().getLSQOp();
      }

      LSQPorts lsqPorts = lsqOp.getPorts();
      llvm::errs() << "lsqPorts" << "--\n";
      for (LSQGroup &group : lsqPorts.getGroups()) {
        llvm::errs() << "group" << "--\n";
        for (MemoryPort &port : group->accessPorts) {
          llvm::errs() << "oomad \n";
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

  std::optional<Block *> blockOptional =
      blockIndexing.getBlockFromIndex(opBBNum);
  if (blockOptional)
    return blockOptional.value();
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
        if (!dependency.getIsActive().getValue())
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
        ftdConditionsForEachPair[predecessorOpName][successorOpName] = boolConditions;
      }
    }
  }
  return ftdConditionsForEachPair;
}

/// This function distributes the start signal to all blocks in the CFG.
Value distributStartSignalToDstBlock(Value lastStartCopy, Block *block,
                                     Block *dstBlock, bool firstExec,
                                     ConversionPatternRewriter &rewriter) {

  llvm::errs() << "last " << lastStartCopy << "\n";

  if (firstExec)
    visited = {};
  if (block == dstBlock)
    return lastStartCopy;

  visited.insert(block);
  if (startCopies.contains(block))
    lastStartCopy = startCopies[block];
  else {
    /// ManualBuff (trick)
    if (auto bb =
            block->front().getAttrOfType<mlir::IntegerAttr>(BB_ATTR_NAME)) {
      llvm::errs() << "mikham" << bb << "\n";
    } else {
      llvm::errs() << "nadasht\n";
    }

    handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
        block->front().getLoc(), lastStartCopy, handshake::TimingInfo::tehb(),
        1);
    inheritBB(&block->front(), bufferOp);
    llvm::errs() << "trick \n" << bufferOp << "\n";
    lastStartCopy = bufferOp.getResult();
  }

  for (Block *successor : block->getSuccessors()) {
    if (!visited.count(successor)) {
      Value returnVal = distributStartSignalToDstBlock(
          lastStartCopy, successor, dstBlock, false, rewriter);
      llvm::errs() << "uuu \n" << returnVal << "\n";
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
    if (dependency.getIsActive().getValue())
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
  auto hi = constVal.getType();
  llvm::errs() << hi << initialVal.getType() << "rrrrrrrrrrrrrrrrrrr\n";

  Value prevResult = initialVal;
  SmallVector<Value> delayedVals = {initialVal};
  SmallVector<Value> extraDelayedVals;

  SmallVector<Value, 2> values;

  unsigned effective_N = N;
  if (N == 0)
    effective_N = 1;
  for (unsigned i = 0; i < effective_N; i++) {
    /// ManualBuff
    handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
        BBOp->getLoc(), prevResult, handshake::TimingInfo::tehb(), 1);
    inheritBB(BBOp, bufferOp);

    llvm::errs() << "N delay \n" << bufferOp << "\n";

    values = {bufferOp.getResult(), constVal};
    handshake::MergeOp mergeOp =
        rewriter.create<handshake::MergeOp>(BBOp->getLoc(), values);
    inheritBB(BBOp, mergeOp);

    if (i != N - 1)
      delayedVals.push_back(mergeOp->getResult(0));
    extraDelayedVals.push_back(mergeOp->getResult(0));

    prevResult = mergeOp->getResult(0);
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
    llvm::errs() << "[4]  " << sourceOpConstOp << "\n";

    handshake::ConstantOp falseConstOp = rewriter.create<handshake::ConstantOp>(
        opPointer->getLoc(), rewriter.getBoolAttr(true), sourceOpConstOp);
    inheritBB(opPointer, falseConstOp);
    llvm::errs() << " return\n";
    return falseConstOp.getResult();
  }
  if (fBool->type != experimental::boolean::ExpressionType::Zero) {

    std::set<std::string> blocks = fBool->getVariables();

    std::vector<std::string> cofactorList(blocks.begin(), blocks.end());
    BDD *bdd = buildBDD(fBool, cofactorList);
    Value condValue = bddToCircuit(rewriter, bdd, block, blockIndexing);
    llvm::errs() << "hebheb" << condValue << "\n";
    return condValue;
  }

  llvm::errs() << " Zero\n";

  handshake::SourceOp sourceOpConstOp =
      rewriter.create<handshake::SourceOp>(opPointer->getLoc());

  inheritBB(opPointer, sourceOpConstOp);
  llvm::errs() << "[2]  " << sourceOpConstOp << "\n";

  handshake::ConstantOp falseConstOp = rewriter.create<handshake::ConstantOp>(
      opPointer->getLoc(), rewriter.getBoolAttr(false), sourceOpConstOp);
  inheritBB(opPointer, falseConstOp);
  llvm::errs() << " return\n";
  return falseConstOp.getResult();
}

/// This function creates all of the FTD conditions for a pair of memory accesses.
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
        BBOp->getLoc(), regenCond, muxOpValues);
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
Value gateChannelValuebyControlValue(Value channelValue, Value controlValue,
                                     Operation *BBOp,
                                     ConversionPatternRewriter &rewriter) {
  handshake::UnbundleOp unbundleOp =
      rewriter.create<handshake::UnbundleOp>(BBOp->getLoc(), channelValue);
  inheritBB(BBOp, unbundleOp);

  SmallVector<Value, 2> joinOpValues = {unbundleOp.getResult(0), controlValue};
  handshake::JoinOp joinOp =
      rewriter.create<handshake::JoinOp>(BBOp->getLoc(), joinOpValues);
  inheritBB(BBOp, joinOp);
  ValueRange *ab = new ValueRange();
  handshake::ChannelType ch =
      handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(
      BBOp->getLoc(), joinOp.getResult(), unbundleOp.getResult(1), *ab, ch);
  inheritBB(BBOp, bundleOp);
  return bundleOp.getResult(0);
}

/// This condition insets suppresses in front ot the main values based on the
/// given conditions.
/// The function is used twice: once for inserting the suppress block and once for
/// the `Conditional Sequentializer` component.
SmallVector<Value> insertBranches(SmallVector<Value> mainValues,
                                  SmallVector<Value> conds, Operation *BBOp,
                                  ConversionPatternRewriter &rewriter) {
  SmallVector<Value> results;
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)) {
    /// ManualBuff
    handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
        BBOp->getLoc(), cond, handshake::TimingInfo::tehb(), 5);
    inheritBB(BBOp, bufferOp);

    handshake::ConditionalBranchOp conditionalBranchOp =
        rewriter.create<handshake::ConditionalBranchOp>(
            BBOp->getLoc(), bufferOp.getResult(), mainValue);
    inheritBB(BBOp, conditionalBranchOp);
    results.push_back(conditionalBranchOp.getResult(1));
  }
  return results;
}

/// This function creates the suppression block for the given main values.
SmallVector<Value> insertSuppressBlock(SmallVector<Value> mainValues,
                                       Value predecessorOpDoneSignal,
                                       Value suppressCond,
                                       Operation *predecessorOpPointer,
                                       Value startSignalInPredecessorBB,
                                       unsigned N,
                                       ConversionPatternRewriter &rewriter) {

  SmallVector<Value> diffTokens = {predecessorOpDoneSignal};
  if (N > 1) {
    diffTokens.append(N - 1, startSignalInPredecessorBB);
  }

  handshake::MergeOp mergeOp = rewriter.create<handshake::MergeOp>(
      predecessorOpPointer->getLoc(), diffTokens);
  inheritBB(predecessorOpPointer, mergeOp);
  /// ManualBuff (Init)
  unsigned effective_N = N;
  if (N == 0)
    effective_N = 1;

  Value next_value = mergeOp.getResult();
  if (effective_N > 1) {
    handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
        predecessorOpPointer->getLoc(), mergeOp.getResult(),
        handshake::TimingInfo::tehb(), effective_N - 1);
    inheritBB(predecessorOpPointer, bufferOp);
    next_value = bufferOp.getResult();
  }

  Value gatedSuppressCond = gateChannelValuebyControlValue(
      suppressCond, next_value, predecessorOpPointer, rewriter);

  handshake::ConditionalBranchOp conditionalBranchOp =
      rewriter.create<handshake::ConditionalBranchOp>(
          predecessorOpPointer->getLoc(), suppressCond, gatedSuppressCond);
  inheritBB(predecessorOpPointer, conditionalBranchOp);

  SmallVector<Value, 2> muxOpValues = {suppressCond,
                                       conditionalBranchOp.getResult(0)};
  handshake::MuxOp muxOp = rewriter.create<handshake::MuxOp>(
      predecessorOpPointer->getLoc(), suppressCond, muxOpValues);
  inheritBB(predecessorOpPointer, muxOp);

  SmallVector<Value> conds;

  for (unsigned i = 0; i < effective_N; i++) {
    /// ManualBuff
    handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
        predecessorOpPointer->getLoc(), muxOp.getResult(),
        handshake::TimingInfo::tehb(), 3);
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
    BlockIndexing blockIndexing, unsigned N, std::string optimizeZero,
    ConversionPatternRewriter &rewriter) {

  FTDConditionValues ftdValues = constructCircuitForAllConditions(
      ftdConditions, blockIndexing, predecessorOpPointer,
      AreOpsinSameBB(predecessorOpPointer, successorOpPointer), startSignalInBB,
      rewriter);

  SmallVector<Value> diffTokens = {predecessorOpDoneSignal};
  // diffTokens.append(N, conditionalBranchOp2.getResult(0));
  if (N > 0) {
    diffTokens.append(N, startSignalInBB);
  }

  handshake::MergeOp mergeOp = rewriter.create<handshake::MergeOp>(
      predecessorOpPointer->getLoc(), diffTokens);
  inheritBB(predecessorOpPointer, mergeOp);
  /// ManualBuff (Init)
  handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
      predecessorOpPointer->getLoc(), mergeOp.getResult(),
      handshake::TimingInfo::tehb(), N);
  inheritBB(predecessorOpPointer, bufferOp);

  handshake::ConditionalBranchOp conditionalBranchOp =
      rewriter.create<handshake::ConditionalBranchOp>(
          successorOpPointer->getLoc(), ftdValues.getSupp(),
          bufferOp.getResult());
  inheritBB(successorOpPointer, conditionalBranchOp);
  // Not sure which one (0 or 1)
  handshake::UnbundleOp unbundleOp = rewriter.create<handshake::UnbundleOp>(
      predecessorOpPointer->getLoc(), successorOpPointer->getOperand(0));
  inheritBB(predecessorOpPointer, unbundleOp);

  SmallVector<Value, 2> JoinOpValues = {conditionalBranchOp.getResult(1),
                                        unbundleOp.getResult(0)};
  handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(
      predecessorOpPointer->getLoc(), JoinOpValues);
  inheritBB(predecessorOpPointer, joinOp);

  ValueRange *ab = new ValueRange();
  handshake::ChannelType ch =
      handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(
      predecessorOpPointer->getLoc(), joinOp.getResult(),
      unbundleOp.getResult(1), *ab, ch);
  inheritBB(predecessorOpPointer, bundleOp);

  Value gatedSuccessorOpaddr = bundleOp.getResult(0);

  SmallVector<Value> delayedAddressesAfterRegen = delayedAddresses;
  if (optimizeZero != "1" || ftdConditions.getSupp()->boolMinimize()->type !=
                                 experimental::boolean::ExpressionType::Zero) {
    delayedAddressesAfterRegen = insertRegenBlock(
        delayedAddresses, ftdValues.getSkip(), predecessorOpPointer, rewriter);
  }

  SmallVector<Value> delayedAddressesAfterSuppress = delayedAddressesAfterRegen;
  if (optimizeZero != "1" || ftdConditions.getSupp()->boolMinimize()->type !=
                                 experimental::boolean::ExpressionType::Zero) {
    delayedAddressesAfterSuppress =
        insertSuppressBlock(delayedAddressesAfterRegen, predecessorOpDoneSignal,
                            ftdValues.getSupp(), predecessorOpPointer,
                            startSignalInBB, N, rewriter);
  }

  SmallVector<Value> skipConditions;
  for (Value delayedAddress : delayedAddressesAfterSuppress) {
    /// ManualBuff (Comparator)
    handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
        predecessorOpPointer->getLoc(), delayedAddress,
        handshake::TimingInfo::tehb(), 1);
    inheritBB(predecessorOpPointer, bufferOp);
    llvm::errs() << "comp \n" << bufferOp << "\n";
    handshake::CmpIOp cmpIOp = rewriter.create<handshake::CmpIOp>(
        predecessorOpPointer->getLoc(), CmpIPredicate::ne, gatedSuccessorOpaddr,
        bufferOp.getResult());
    inheritBB(predecessorOpPointer, cmpIOp);
    skipConditions.push_back(cmpIOp.getResult());
  }
  return skipConditions;
}

/// This function creates the skip conditions for all pairs of memory accesses.
/// This means that it creates the left side of the circuit for each pair.
/// This includes the delay generator which is shared for the same predecessor.
/// The regeneration/suppression block and the skip condition generator are
/// created specifically for each pair in `createSkipConditionForPair`.
SkipConditionForPair createSkipConditionsForAllPairs(
    MemAccesses &memAccesses, FTDBoolExpForPair &ftdConditionsForEachPair,
    FuncOpInformation funcOpInformation, unsigned N, std::string optimizeZero,
    ConversionPatternRewriter &rewriter) {

  SkipConditionForPair skipConditionForEachPair;
  Value startSignal = funcOpInformation.getStartSignal();

  for (auto [predecessorOpName, predecessorOpPointer] : memAccesses) {
    rewriter.setInsertionPointToStart(predecessorOpPointer->getBlock());
    Value predecessorOpDoneSignal = predecessorOpPointer->getResult(2);
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

        auto bothDelayedAddresses = getNDelayedValues(
            predecessorOpAddr, dummyConstOp, predecessorOpPointer, N, rewriter);
        SmallVector<Value> delayedAddresses = std::get<0>(bothDelayedAddresses);
        SmallVector<Value> extraDelayedAddresses =
            std::get<1>(bothDelayedAddresses);

        /// ManualBuff (store)
        handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
            predecessorOpPointer->getLoc(), predecessorOpAddr,
            ::TimingInfo::tehb(), 20);
        inheritBB(predecessorOpPointer, bufferOp);
        predecessorOpPointer->setOperand(0, bufferOp.getResult());

        for (MemDependenceAttr dependency : deps.getDependencies()) {
          if (!dependency.getIsActive().getValue())
            continue;

          StringRef successorOpName = dependency.getDstAccess();
          Operation *successorOpPointer = memAccesses[successorOpName];
          FTDBoolExpressions ftdConditions =
              ftdConditionsForEachPair[predecessorOpName][successorOpName];

          // If the first predecessor does not have a producer, we need to
          // use the extra delayed addresses.
          SmallVector<Value> effectiveDelayedAddresses = delayedAddresses;
          if (isInitialConsWithoutProd(predecessorOpPointer,
                                       successorOpPointer))
            effectiveDelayedAddresses = extraDelayedAddresses;

          SmallVector<Value> skipConditions = createSkipConditionForPair(
              predecessorOpDoneSignal, predecessorOpPointer, successorOpPointer,
              effectiveDelayedAddresses, startSignalInBB, ftdConditions,
              blockIndexing, N, optimizeZero, rewriter);
          skipConditionForEachPair[predecessorOpName][successorOpName] =
              skipConditions;
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
      predecessorOp->getLoc(), cond, muxOpValues);
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
    BlockIndexing blockIndexing, unsigned N, std::string optimizeZero,
    ConversionPatternRewriter &rewriter) {

  FTDConditionValues ftdValues = constructCircuitForAllConditions(
      ftdConditions, blockIndexing, predecessorOp,
      AreOpsinSameBB(predecessorOp, successorOp), startSignalInBB, rewriter);

  SmallVector<Value> delayedDoneSignalsAfterRegen = delayedDoneSignals;

  if (optimizeZero != "1" || ftdConditions.getSkip()->boolMinimize()->type !=
                                 experimental::boolean::ExpressionType::Zero) {
    delayedDoneSignalsAfterRegen = insertRegenBlock(
        delayedDoneSignals, ftdValues.getSkip(), predecessorOp, rewriter);
  } else {
    llvm::errs() << "skipping regen\n";
  }

  SmallVector<Value> delayedDoneSignalsAfterSuppress =
      delayedDoneSignalsAfterRegen;
  if (optimizeZero != "1" || ftdConditions.getSupp()->boolMinimize()->type !=
                                 experimental::boolean::ExpressionType::Zero) {
    delayedDoneSignalsAfterSuppress = insertSuppressBlock(
        delayedDoneSignalsAfterRegen, predecessorOpDoneSignal,
        ftdValues.getSupp(), predecessorOp, startSignalInBB, N, rewriter);
  } else {
    llvm::errs() << "skipping supp\n";
  }


  if (N == 0)
    return delayedDoneSignalsAfterSuppress[0];

  SmallVector<Value> branchedDoneSignals = insertBranches(
      delayedDoneSignalsAfterSuppress, conds, predecessorOp, rewriter);
  SmallVector<Value> conditionallySkippedDoneSignals = insertConditionalSkips(
      branchedDoneSignals, conds, successorOp, startSignal, rewriter);
  handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(
      predecessorOp->getLoc(), conditionallySkippedDoneSignals);
  inheritBB(predecessorOp, joinOp);

  return joinOp.getResult();
}

/// This function returns the inactivated version of a given dependency.
MemDependenceAttr getInactivatedDependency(MemDependenceAttr dependency) {
  MLIRContext *ctx = dependency.getContext();
  return MemDependenceAttr::get(
      ctx, dependency.getDstAccess(), dependency.getLoopDepth(),
      dependency.getComponents(), BoolAttr::get(ctx, false));
}

/// This function creates the waiting signals for all pairs of memory accesses.
/// This means that it creates the right side of the circuit for each pair.
/// This includes the delay generator which is shared for the same predecessor.
/// The regeneration/suppression block and the conditional sequentializer
/// are created specifically for each pair in `createWaitingSignalForPair`.
WaitingSignalForSucc createWaitingSignals(
    MemAccesses &memAccesses, SkipConditionForPair &skipConditionForEachPair,
    MLIRContext *ctx, FuncOpInformation funcOpInformation,
    FTDBoolExpForPair &ftdConditionsForEachPair, unsigned N,
    std::string optimizeZero, ConversionPatternRewriter &rewriter) {

  WaitingSignalForSucc waitingSignalsForEachSuccessor;
  BlockIndexing blockIndexing = funcOpInformation.getBlockIndexing();
  Value startSignal = funcOpInformation.getStartSignal();
  Block *startBlock = funcOpInformation.getStartBlock();

  for (auto [predecessorOpName, predecessorOpPointer] : memAccesses) {
    rewriter.setInsertionPointToStart(predecessorOpPointer->getBlock());
    Value predecessorOpDoneSignal = predecessorOpPointer->getResult(2);

    SmallVector<MemDependenceAttr> newDeps;
    if (auto deps =
            getDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer)) {

      Block *predecessorBlock =
          getBlockFromOp(predecessorOpPointer, blockIndexing);

      Value startSignalInBB = distributStartSignalToDstBlock(
          startSignal, startBlock, predecessorBlock, true, rewriter);

      auto bothDelayedDoneSignals =
          getNDelayedValues(predecessorOpDoneSignal, startSignalInBB,
                            predecessorOpPointer, N, rewriter);
      SmallVector<Value> delayedDoneSignals =
          std::get<0>(bothDelayedDoneSignals);
      SmallVector<Value> extraDelayedDoneSignals =
          std::get<1>(bothDelayedDoneSignals);

      for (MemDependenceAttr dependency : deps.getDependencies()) {
        if (!dependency.getIsActive().getValue()) {
          newDeps.push_back(dependency);
          continue;
        }

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
            startSignalInBB, ftdConditions, blockIndexing, N, optimizeZero,
            rewriter);
        waitingSignalsForEachSuccessor[successorName].push_back(waitingSignal);

        newDeps.push_back(getInactivatedDependency(dependency));
      }
      setDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer, ctx,
                                             newDeps);
      setDialectAttr<MemInterfaceAttr>(predecessorOpPointer, ctx);
    }
  }
  llvm::errs() << "[] Created Waiting Signals\n";
  return waitingSignalsForEachSuccessor;
}

void gateAddress(Operation *op, SmallVector<Value> waitingValues,
                 ConversionPatternRewriter &rewriter, Location loc) {
  Value address = op->getOperand(0);
  handshake::UnbundleOp unbundleOp =
      rewriter.create<handshake::UnbundleOp>(loc, address);
  inheritBB(op, unbundleOp);
  waitingValues.push_back(unbundleOp.getResult(0));
  handshake::JoinOp joinOp =
      rewriter.create<handshake::JoinOp>(loc, waitingValues);
  inheritBB(op, joinOp);
  ValueRange *ab = new ValueRange();
  handshake::ChannelType ch =
      handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(
      loc, joinOp.getResult(), unbundleOp.getResult(1), *ab, ch);
  inheritBB(op, bundleOp);
  op->setOperand(0, bundleOp.getResult(0));
}

void gateAllSuccessorAccesses(
    MemAccesses &memAccesses,
    WaitingSignalForSucc &waitingSignalsForEachSuccessor,
    ConversionPatternRewriter &rewriter) {
  for (auto [dstAccess, waitingSignals] : waitingSignalsForEachSuccessor) {
    Operation *op = memAccesses[dstAccess];
    gateAddress(op, waitingSignals, rewriter, op->getLoc());
  }
  llvm::errs() << "[] Gated Dst Accesses\n";
}

/// This function is the main function. It is responsible to insert the 
/// necessary components for skippable sequencializing in every funcOp.
void HandshakeInsertSkippableSeqPass::handleFuncOp(FuncOp funcOp,
                                                   MLIRContext *ctx) {
  ConversionPatternRewriter rewriter(ctx);
  FuncOpInformation funcOpInformation(funcOp);

  unsigned N = stoul(NStr);

  llvm::errs() << "!! " << kernelName << "\n";

  MemAccesses memAccesses;
  SkipConditionForPair skipConditionForEachPair;
  WaitingSignalForSucc waitingSignalsForEachSuccessor;
  FTDBoolExpForPair ftdConditionsForEachPair;

  initializeStartCopies(funcOpInformation);

  memAccesses = findMemAccessesInFunc(funcOp);

  ftdConditionsForEachPair = calculateFtdConditionsForEachPair(
      memAccesses, kernelName, funcOpInformation, rewriter);

  if (N != 0) {
    skipConditionForEachPair = createSkipConditionsForAllPairs(
        memAccesses, ftdConditionsForEachPair, funcOpInformation, N,
        optimizeZero, rewriter);
  }

  waitingSignalsForEachSuccessor = createWaitingSignals(
      memAccesses, skipConditionForEachPair, ctx, funcOpInformation,
      ftdConditionsForEachPair, N, optimizeZero, rewriter);

  gateAllSuccessorAccesses(memAccesses, waitingSignalsForEachSuccessor,
                           rewriter);
}

void HandshakeInsertSkippableSeqPass::runDynamaticPass() {

  llvm::errs() << "opt " << optimizeZero << "\n";

  mlir::ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  ConversionPatternRewriter rewriter(ctx);

  for (auto funcOp : modOp.getOps<handshake::FuncOp>()) {
    if (failed(experimental::cfg::restoreCfStructure(funcOp, rewriter)))
      signalPassFailure();

    handleFuncOp(funcOp, ctx);

    funcOp.print(llvm::errs());
    if (failed(cfg::flattenFunction(funcOp)))
      signalPassFailure();
  }

  llvm::errs() << "done! \n";
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeInsertSkippableSeq(const std::string &NStr,
                                             const std::string &kernelName,
                                             const std::string &optimizeZero) {
  return std::make_unique<HandshakeInsertSkippableSeqPass>();
}
