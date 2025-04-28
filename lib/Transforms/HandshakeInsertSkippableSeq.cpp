//===- HandshakeInsertSkippableSeq.cpp - LSQ flow analysis ------*- C++ -*-===//
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

using BoolExpForPair =
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

DenseMap<Block *, Value> startCopies;
std::unordered_set<Block *> visited;

void initializeStartCopies(FuncOpInformation funcOpInformation) {
  Block *block = funcOpInformation.getStartBlock();
  Value startSignal = funcOpInformation.getStartSignal();
  startCopies[block] = startSignal;
}

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

bool hasAtLeastOneActiveDep(MemDependenceArrayAttr deps) {
  for (MemDependenceAttr dependency : deps.getDependencies()) {
    if (dependency.getIsActive().getValue())
      return true;
  }
  return false;
}

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

  llvm::errs() << delayedVals.size() << "*********** "
               << extraDelayedVals.size() << "\n";

  return std::tuple<SmallVector<Value>, SmallVector<Value>>(delayedVals,
                                                            extraDelayedVals);
}

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

Value boolToSelect(Value cond, Value startSignal, Operation *predecessorOp,
                   ConversionPatternRewriter &rewriter) {
  llvm::errs() << "yahoo\n";
  handshake::SourceOp sourceOp =
      rewriter.create<handshake::SourceOp>(predecessorOp->getLoc());
  inheritBB(predecessorOp, sourceOp);
  handshake::ConstantOp zeroConstOp = rewriter.create<handshake::ConstantOp>(
      predecessorOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0),
      sourceOp);
  handshake::ConstantOp oneConstOp = rewriter.create<handshake::ConstantOp>(
      predecessorOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1),
      sourceOp);
  inheritBB(predecessorOp, zeroConstOp);
  inheritBB(predecessorOp, oneConstOp);

  handshake::ConditionalBranchOp zeroConditionalBranchOp =
      rewriter.create<handshake::ConditionalBranchOp>(predecessorOp->getLoc(),
                                                      cond, zeroConstOp);
  handshake::ConditionalBranchOp oneConditionalBranchOp =
      rewriter.create<handshake::ConditionalBranchOp>(predecessorOp->getLoc(),
                                                      cond, oneConstOp);
  inheritBB(predecessorOp, zeroConditionalBranchOp);
  inheritBB(predecessorOp, oneConditionalBranchOp);

  SmallVector<Value, 2> values = {zeroConditionalBranchOp.getResult(0),
                                  oneConditionalBranchOp.getResult(1)};
  handshake::MergeOp mergeOp =
      rewriter.create<handshake::MergeOp>(predecessorOp->getLoc(), values);
  inheritBB(predecessorOp, mergeOp);

  llvm::errs() << "yahoo" << mergeOp.getResult().getType() << "\n";
  llvm::errs() << "yahoo" << cond.getType() << "\n";

  return mergeOp.getResult();
}

Value createSkip(Value waitingToken, Value cond, Value startSignal,
                 Operation *predecessorOp,
                 ConversionPatternRewriter &rewriter) {
  // handshake::UnbundleOp unbundleOp =
  // rewriter.create<handshake::UnbundleOp>(predecessorOp->getLoc(), cond);
  // inheritBB(predecessorOp, unbundleOp);
  // handshake::ConditionalBranchOp conditionalBranchOp =
  // rewriter.create<handshake::ConditionalBranchOp>(predecessorOp->getLoc(),
  // cond, unbundleOp.getResult(0)); inheritBB(predecessorOp,
  // conditionalBranchOp);
  handshake::SourceOp sourceOp =
      rewriter.create<handshake::SourceOp>(predecessorOp->getLoc());

  inheritBB(predecessorOp, sourceOp);
  llvm::errs() << "[1]  " << sourceOp << "\n";
  SmallVector<Value, 2> muxOpValues = {waitingToken, sourceOp.getResult()};
  // Value select = boolToSelect(cond, startSignal, predecessorOp, rewriter);
  handshake::MuxOp muxOp = rewriter.create<handshake::MuxOp>(
      predecessorOp->getLoc(), cond, muxOpValues);
  inheritBB(predecessorOp, muxOp);

  // ValueRange *ab = new ValueRange();
  // handshake::ChannelType ch =
  // handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  // handshake::BundleOp bundleOp =
  // rewriter.create<handshake::BundleOp>(predecessorOp->getLoc(),
  // unbundleOp.getResult(0), unbundleOp.getResult(1), *ab, ch);
  // inheritBB(predecessorOp, bundleOp);

  // rewriter.create<handshake::SinkOp>(predecessorOp->getLoc(),
  // bundleOp.getResult(0));
  return muxOp.getResult();
}

SmallVector<Value> insertConditionalSkips(SmallVector<Value> mainValues,
                                          SmallVector<Value> conds,
                                          Operation *predecessorOp,
                                          Value startSignal,
                                          ConversionPatternRewriter &rewriter) {
  SmallVector<Value> results;
  llvm::errs() << "function \n";
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)) {
    Value skipValue =
        createSkip(mainValue, cond, startSignal, predecessorOp, rewriter);
    llvm::errs() << "skip: ";
    llvm::errs() << skipValue << "\n";
    results.push_back(skipValue);
  }
  return results;
}

DenseMap<int, Operation *> getControlMergeOps(FuncOp funcOp) {
  DenseMap<int, Operation *> results;
  for (ControlMergeOp controlMergeOp :
       funcOp.getOps<handshake::ControlMergeOp>()) {
    int BB = getBBNumberFromOp(controlMergeOp);
    results[BB] = controlMergeOp;
  }
  return results;
}

bool AreOpsinSameBB(Operation *first, Operation *second) {
  int firstBBNum = getBBNumberFromOp(first);
  int secondBBNum = getBBNumberFromOp(second);
  return firstBBNum == secondBBNum;
}

bool isInitialConsWithoutProd(Operation *prod, Operation *cons) {
  if (!AreOpsinSameBB(prod, cons))
    return false;
  return cons->isBeforeInBlock(prod);
}

Block *getBlockFromOp(Operation *op, BlockIndexing blockIndexing) {

  int opBBNum = getBBNumberFromOp(op);

  std::optional<Block *> blockOptional =
      blockIndexing.getBlockFromIndex(opBBNum);
  llvm::errs() << "hi " << blockOptional << "\n";
  llvm::errs() << "hi " << blockOptional.value() << "\n";
  if (blockOptional)
    return blockOptional.value();
}

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

Value calculateCFGCond(
    Operation *predecessorOp, Operation *successorOp,
    DenseMap<int, Operation *> &controlMerges, BlockIndexing blockIndexing,
    Block *startBlock,
    ControlDependenceAnalysis::BlockControlDepsMap cdAnalysis,
    ConversionPatternRewriter &rewriter) {

  llvm::errs() << "be to che --------------\n";

  Block *predecessorBlock = getBlockFromOp(predecessorOp, blockIndexing);
  Block *successorBlock = getBlockFromOp(successorOp, blockIndexing);

  DenseSet<Block *> predControlDeps =
      cdAnalysis[predecessorBlock].forwardControlDeps;

  DenseSet<Block *> succControlDeps =
      cdAnalysis[successorBlock].forwardControlDeps;

  // Get rid of common entries in the two sets
  eliminateCommonBlocks(predControlDeps, succControlDeps);

  BoolExpression *fProd = enumeratePaths(startBlock, predecessorBlock,
                                         blockIndexing, predControlDeps);
  BoolExpression *fCons = enumeratePaths(startBlock, successorBlock,
                                         blockIndexing, succControlDeps);

  /// f_supp = f_prod and not f_cons
  BoolExpression *fSup = BoolExpression::boolAnd(fProd, fCons->boolNegate());
  fSup = fSup->boolMinimize();

  // If the activation function is not zero, then a suppress block is to be
  // inserted
  if (fSup->type != experimental::boolean::ExpressionType::Zero) {
    llvm::errs() << "not zero :( \n";

    std::set<std::string> blocks = fSup->getVariables();

    std::vector<std::string> cofactorList(blocks.begin(), blocks.end());
    BDD *bdd = buildBDD(fSup, cofactorList);
    Value branchCond =
        bddToCircuit(rewriter, bdd, successorBlock, blockIndexing);

    llvm::errs() << "Finally: " << branchCond << "\n";
  }

  else
    llvm::errs() << " zero :) \n";

  llvm::errs() << "be to che --------------\n";

  llvm::errs() << "jaleb: " << fProd->sopToString() << "\n";
  llvm::errs() << "jaleb: " << fCons->sopToString() << "\n";

  ControlMergeOp controlMergeOp =
      dyn_cast<handshake::ControlMergeOp>(controlMerges[1]);

  handshake::SourceOp sourceOp =
      rewriter.create<handshake::SourceOp>(successorOp->getLoc());

  inheritBB(successorOp, sourceOp);
  llvm::errs() << "[3]  " << sourceOp << "\n";
  handshake::ConstantOp constOp = rewriter.create<handshake::ConstantOp>(
      successorOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0),
      sourceOp);
  inheritBB(successorOp, constOp);
  llvm::errs() << controlMergeOp->getResult(1).getType() << "tof %% \n";
  llvm::errs() << constOp << "tof %% \n";
  llvm::errs() << constOp.getValue().getType() << "tof %% \n";
  handshake::CmpIOp cmpIOp = rewriter.create<handshake::CmpIOp>(
      successorOp->getLoc(), CmpIPredicate::eq, constOp.getResult(),
      controlMergeOp->getResult(1));
  inheritBB(successorOp, cmpIOp);
  // needed

  return cmpIOp.getResult();
}

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

SmallVector<Value> insertSuppressBlock(SmallVector<Value> mainValues,
                                       Value predecessorOpDoneSignal,
                                       Value suppressCond,
                                       Operation *predecessorOpPointer,
                                       Value startSignalInPredecessorBB,
                                       unsigned N,
                                       ConversionPatternRewriter &rewriter) {

  llvm::errs() << "$@#@@!#$@#$#@%@!%#@$%#$%#@%#@%#@%#@%#@%#$\n";
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

  // /// ManualBuff
  // handshake::BufferOp suppressCondBufferOp =
  //     rewriter.create<handshake::BufferOp>(predecessorOpPointer->getLoc(),
  //                                          suppressCond,
  //                                          handshake::TimingInfo::tehb(),
  //                                          10);
  // inheritBB(predecessorOpPointer, suppressCondBufferOp);

  // handshake::ConditionalBranchOp conditionalBranchOpForSupp =
  //     rewriter.create<handshake::ConditionalBranchOp>(
  //         predecessorOpPointer->getLoc(), suppressCondBufferOp.getResult(),
  //         next_value);
  // inheritBB(predecessorOpPointer, conditionalBranchOpForSupp);

  // SmallVector<Value> results;
  // for (Value mainValue : mainValues) {

  //   handshake::ConditionalBranchOp conditionalBranchOp =
  //       rewriter.create<handshake::ConditionalBranchOp>(
  //           predecessorOpPointer->getLoc(), suppressCond, mainValue);
  //   inheritBB(predecessorOpPointer, conditionalBranchOp);

  //   Value gatedMainValue;
  //   if (isa<handshake::ChannelType>(mainValue.getType())) {
  //     llvm::errs() << "channel &&&&&&&&\n" << conditionalBranchOp << "\n";

  //     gatedMainValue = gateChannelValuebyControlValue(
  //         conditionalBranchOp.getResult(0),
  //         conditionalBranchOpForSupp.getResult(0), predecessorOpPointer,
  //         rewriter);
  //   } else {
  //     llvm::errs() << "sade &&&&&&&&\n";
  //     handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(
  //         predecessorOpPointer->getLoc(),
  //         SmallVector<Value>{conditionalBranchOp.getResult(0),
  //                            conditionalBranchOpForSupp.getResult(0)});
  //     inheritBB(predecessorOpPointer, joinOp);

  //     gatedMainValue = joinOp.getResult();
  //   }

  //   results.push_back(conditionalBranchOp.getResult(1));
  // }
  // return results;

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

  for (int i = 0; i < effective_N; i++) {
    /// ManualBuff
    handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
        predecessorOpPointer->getLoc(), muxOp.getResult(),
        handshake::TimingInfo::tehb(), 3);
    inheritBB(predecessorOpPointer, bufferOp);
    conds.push_back(bufferOp.getResult());
  }
  return insertBranches(mainValues, conds, predecessorOpPointer, rewriter);

  /// OLD
  // Value gatedSuppressCond = gateChannelValuebyControlValue(
  //     suppressCond, next_value, predecessorOpPointer, rewriter);

  // SmallVector<Value> conds;
  // conds.append(effective_N, gatedSuppressCond);
  // return insertBranches(mainValues, conds, predecessorOpPointer, rewriter);
  /// OLD
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
  llvm::errs() << (optimizeZero != "1") << "\n"
               << (ftdConditions.getSkip()->boolMinimize()->type !=
                   experimental::boolean::ExpressionType::Zero)
               << "mah\n";
  llvm::errs() << "mm " << delayedDoneSignals.size() << "\n";
  if (optimizeZero != "1" || ftdConditions.getSkip()->boolMinimize()->type !=
                                 experimental::boolean::ExpressionType::Zero) {
    delayedDoneSignalsAfterRegen = insertRegenBlock(
        delayedDoneSignals, ftdValues.getSkip(), predecessorOp, rewriter);
  } else {
    llvm::errs() << "skipping regen\n";
  }
  llvm::errs() << "mm " << delayedDoneSignalsAfterRegen.size() << "\n";
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

  llvm::errs() << "mm " << delayedDoneSignalsAfterSuppress.size() << "\n";

  if (N == 0)
    return delayedDoneSignalsAfterSuppress[0];

  SmallVector<Value> branchedDoneSignals = insertBranches(
      delayedDoneSignalsAfterSuppress, conds, predecessorOp, rewriter);
  SmallVector<Value> conditionallySkippedDoneSignals = insertConditionalSkips(
      branchedDoneSignals, conds, successorOp, startSignal, rewriter);
  handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(
      predecessorOp->getLoc(), conditionallySkippedDoneSignals);
  inheritBB(predecessorOp, joinOp);
  // return createSkip(joinOp.getResult(), ftdValues.getSkip(), startSignal,
  //                   successorOp, rewriter);
  return joinOp.getResult();
}

MemDependenceAttr getInactivatedDependency(MemDependenceAttr dependency) {
  MLIRContext *ctx = dependency.getContext();
  return MemDependenceAttr::get(
      ctx, dependency.getDstAccess(), dependency.getLoopDepth(),
      dependency.getComponents(), BoolAttr::get(ctx, false));
}

WaitingSignalForSucc createWaitingSignals(
    MemAccesses &memAccesses, SkipConditionForPair &skipConditionForEachPair,
    MLIRContext *ctx, FuncOpInformation funcOpInformation,
    BoolExpForPair &ftdConditionsForEachPair, unsigned N,
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

SkipConditionForPair createSkipConditionGenerator(
    MemAccesses &memAccesses, BoolExpForPair &ftdConditionsForEachPair,
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

          StringRef dstAccess = dependency.getDstAccess();
          Operation *successorOpPointer = memAccesses[dstAccess];
          FTDBoolExpressions ftdConditions =
              ftdConditionsForEachPair[predecessorOpName][dstAccess];

          SmallVector<Value> effectiveDelayedAddresses = delayedAddresses;
          if (isInitialConsWithoutProd(predecessorOpPointer,
                                       successorOpPointer))
            effectiveDelayedAddresses = extraDelayedAddresses;

          SmallVector<Value> skipConditions = createSkipConditionForPair(
              predecessorOpDoneSignal, predecessorOpPointer, successorOpPointer,
              effectiveDelayedAddresses, startSignalInBB, ftdConditions,
              blockIndexing, N, optimizeZero, rewriter);
          skipConditionForEachPair[predecessorOpName][dstAccess] =
              skipConditions;
        }
      }
    }
  }
  return skipConditionForEachPair;
  llvm::errs() << "[S][INFO] Created Skip Conditions\n";
}

BoolExpForPair calculateAllFtdConditions(
    DenseMap<StringRef, Operation *> &memAccesses, std::string kernelName,
    FuncOpInformation funcOpInformation, ConversionPatternRewriter &rewriter) {

  BoolExpForPair ftdConditionsForEachPair;

  for (auto [predecessorOpName, predecessorOpPointer] : memAccesses) {
    if (auto deps =
            getDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer)) {

      for (MemDependenceAttr dependency : deps.getDependencies()) {
        if (!dependency.getIsActive().getValue())
          continue;

        StringRef dstAccess = dependency.getDstAccess();
        Operation *successorOpPointer = memAccesses[dstAccess];

        llvm::errs() << "calculating for " << predecessorOpName << dstAccess
                     << "\n";
        Block *predecessorBlock = getBlockFromOp(
            predecessorOpPointer, funcOpInformation.getBlockIndexing());
        Block *successorBlock = getBlockFromOp(
            successorOpPointer, funcOpInformation.getBlockIndexing());

        FTDBoolExpressions boolConditions =
            calculateFTDConditions(predecessorBlock, successorBlock, kernelName,
                                   funcOpInformation, rewriter);
        ftdConditionsForEachPair[predecessorOpName][dstAccess] = boolConditions;
      }
    }
  }
  return ftdConditionsForEachPair;
}

void HandshakeInsertSkippableSeqPass::handleFuncOp(FuncOp funcOp,
                                                   MLIRContext *ctx) {
  ConversionPatternRewriter rewriter(ctx);
  FuncOpInformation funcOpInformation(funcOp);

  unsigned N = stoul(NStr);

  llvm::errs() << "!! " << kernelName << "\n";

  MemAccesses memAccesses;
  SkipConditionForPair skipConditionForEachPair;
  WaitingSignalForSucc waitingSignalsForEachSuccessor;
  BoolExpForPair ftdConditionsForEachPair;

  initializeStartCopies(funcOpInformation);

  memAccesses = findMemAccessesInFunc(funcOp);

  ftdConditionsForEachPair = calculateAllFtdConditions(
      memAccesses, kernelName, funcOpInformation, rewriter);

  if (N != 0) {
    skipConditionForEachPair = createSkipConditionGenerator(
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

    // funcOp.print(llvm::errs());
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
