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
using delayedDict = DenseMap<Operation *, SmallVector<Value>>;

namespace {

struct HandshakeInsertSkippableSeqPass
    : public dynamatic::impl::HandshakeInsertSkippableSeqBase<
          HandshakeInsertSkippableSeqPass> {

  void runDynamaticPass() override;

  void handleFuncOp(FuncOp funcOp, MLIRContext *ctx);
};
} // namespace

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
DenseMap<Operation *, std::vector<int>> consumerOpAndOperandIndexForFTD;
IsWaitingSignalForSuccDirect isWaitingSignalForSuccDirect;
delayedDict delayedAddressesForEachPred;
delayedDict delayedDoneSignalsForEachPred;

/// This function traverses the function and finds all memory accesses.
MemAccesses findMemAccessesInFunc(FuncOp funcOp) {
  MemAccesses memAccesses;

  for (BlockArgument arg : funcOp.getArguments()) {
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
/// value. If the delay generator is not created yet, it creates one.
SmallVector<Value> createDelayGenerator(Value initialVal,
                                        Operation *predecessorOp, unsigned N,
                                        bool isAddress,
                                        SmallVector<Operation *> &opList,
                                        ConversionPatternRewriter &rewriter) {

  delayedDict &delayedValuesForEachPred =
      isAddress ? delayedAddressesForEachPred : delayedDoneSignalsForEachPred;

  // first check whether the dicionary conatains the predecessor op
  if (delayedValuesForEachPred.find(predecessorOp) ==
      delayedValuesForEachPred.end()) {

    rewriter.setInsertionPoint(predecessorOp);

    // ** IMPORTANT **
    // This buffer is necessary, becuase otherwise the first non-delayed signal
    // will not go through FTD
    handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(
        predecessorOp->getLoc(), initialVal, 1, BufferType::FIFO_BREAK_NONE);
    inheritBB(predecessorOp, bufferOp);
    opList.push_back(bufferOp);

    Value prevResult = bufferOp.getResult();
    SmallVector<Value> delayedVals = {prevResult};

    SmallVector<Value, 2> values;

    for (unsigned i = 0; i < N - 1; i++) {
      handshake::InitOp initOp = rewriter.create<handshake::InitOp>(
          predecessorOp->getLoc(), prevResult);
      inheritBB(predecessorOp, initOp);
      opList.push_back(initOp);

      delayedVals.push_back(initOp.getResult());
      prevResult = initOp.getResult();
    }
    delayedValuesForEachPred[predecessorOp] = delayedVals;
    return delayedVals;
  }

  // If the delay generator for the given initial value is already created
  // and the len is longer than N, return the existing delayed values.
  if (delayedValuesForEachPred[predecessorOp].size() >= N) {
    SmallVector<Value> existingDelayedValues;
    for (unsigned i = 0; i < N; i++) {
      existingDelayedValues.push_back(
          delayedValuesForEachPred[predecessorOp][i]);
    }
    return existingDelayedValues;
  }

  // Otherwise, extend the existing delay generator to have N delayed values.
  SmallVector<Value> extendedDelayedValues =
      delayedValuesForEachPred[predecessorOp];
  Value prevValue = extendedDelayedValues.back();
  unsigned currentLen = extendedDelayedValues.size();
  rewriter.setInsertionPoint(predecessorOp);
  for (unsigned i = currentLen; i < N; i++) {
    handshake::InitOp initOp =
        rewriter.create<handshake::InitOp>(predecessorOp->getLoc(), prevValue);
    inheritBB(predecessorOp, initOp);
    opList.push_back(initOp);

    extendedDelayedValues.push_back(initOp.getResult());
    prevValue = initOp.getResult();
  }
  return extendedDelayedValues;
}

/// This condition insets suppresses in front ot the main values based on the
/// given conditions.
/// The function is used for inserting before the conditional skips in
/// `Conditional Sequentializer` component.
SmallVector<Value> insertBranches(
    SmallVector<Value> mainValues, SmallVector<Value> conds, Operation *BBOp,
    DenseMap<OpOperand *, SmallVector<Value>> &dependenciesMapForPhiNetwork,
    SmallVector<Operation *> &ops, ConversionPatternRewriter &rewriter) {
  SmallVector<Value> results;
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)) {
    handshake::ConditionalBranchOp conditionalBranchOp =
        rewriter.create<handshake::ConditionalBranchOp>(BBOp->getLoc(), cond,
                                                        mainValue);
    inheritBB(BBOp, conditionalBranchOp);
    ops.push_back(conditionalBranchOp);
    results.push_back(conditionalBranchOp.getResult(1));
    dependenciesMapForPhiNetwork[&conditionalBranchOp->getOpOperand(1)] = {
        mainValue};
    consumerOpAndOperandIndexForFTD[conditionalBranchOp].push_back(1);
  }
  return results;
}

void addDrawingAttrToList(ArrayRef<Operation *> operations, StringRef attr) {
  for (auto op : operations) {
    auto drawAttr = handshake::DrawingAttr::get(op->getContext(), attr);
    op->setAttr("drawing", drawAttr);
  }
}

/// This function creates the skip condition for a pair of memory accesses.
SmallVector<Value> createSkipConditionForPair(
    Value predecessorOpDoneSignal, Operation *predecessorOpPointer,
    Operation *successorOpPointer, SmallVector<Value> delayedAddresses,
    unsigned N,
    DenseMap<OpOperand *, SmallVector<Value>> &dependenciesMapForPhiNetwork,
    ConversionPatternRewriter &rewriter) {

  SmallVector<Operation *> skipConditionGeneratorOps;

  SmallVector<Value> extraDelayedPredDoneSignals =
      createDelayGenerator(predecessorOpDoneSignal, predecessorOpPointer, N + 1,
                           false, skipConditionGeneratorOps, rewriter);
  Value extraDelayedPredDoneSignal = extraDelayedPredDoneSignals[N];

  // synchronizing join to limit the advance of the window
  rewriter.setInsertionPoint(successorOpPointer);
  handshake::GateOp gateOp = rewriter.create<handshake::GateOp>(
      successorOpPointer->getLoc(), successorOpPointer->getOperand(0),
      extraDelayedPredDoneSignal);
  inheritBB(successorOpPointer, gateOp);

  skipConditionGeneratorOps.push_back(gateOp);
  dependenciesMapForPhiNetwork[&gateOp->getOpOperand(1)].push_back(
      extraDelayedPredDoneSignal);
  consumerOpAndOperandIndexForFTD[gateOp].push_back(1);

  SmallVector<Value> skipConditions;

  for (Value delayedAddress : delayedAddresses) {
    handshake::CmpIOp cmpIOp = rewriter.create<handshake::CmpIOp>(
        successorOpPointer->getLoc(), CmpIPredicate::ne, gateOp.getResult(),
        delayedAddress);
    inheritBB(successorOpPointer, cmpIOp);

    skipConditionGeneratorOps.push_back(cmpIOp);
    skipConditions.push_back(cmpIOp.getResult());
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
SkipConditionForPair
createSkipConditionsForAllPairs(MemAccesses &memAccesses, FuncOp funcOp,
                                std::vector<unsigned> Nvector,
                                ConversionPatternRewriter &rewriter) {

  SkipConditionForPair skipConditionForEachPair;
  DenseMap<OpOperand *, SmallVector<Value>> dependenciesMapForPhiNetwork;

  unsigned NvectorIndex = 0;
  unsigned N;

  for (auto [predecessorOpName, predecessorOpPointer] : memAccesses) {
    rewriter.setInsertionPointToStart(predecessorOpPointer->getBlock());
    Value predecessorOpDoneSignal =
        getDoneSignalFromMemoryOp(predecessorOpPointer, rewriter);
    Value predecessorOpAddr = predecessorOpPointer->getOperand(0);

    if (auto deps =
            getDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer)) {

      if (hasAtLeastOneActiveDep(deps)) {
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
          NvectorIndex = NvectorIndex % Nvector.size();

          if (N != 0) {
            SmallVector<Operation *> addressDelayGenerator;

            // rewriter.setInsertionPoint(predecessorOpPointer);
            // Value prevResult = predecessorOpDoneSignal;
            // for (unsigned i = 0; i < N; i++) {
            //   handshake::InitOp initOp = rewriter.create<handshake::InitOp>(
            //       predecessorOpPointer->getLoc(), prevResult);
            //   inheritBB(predecessorOpPointer, initOp);
            //   prevResult = initOp.getResult();
            // }

            // mark
            // rewriter.setInsertionPoint(predecessorOpPointer);
            // handshake::GateOp gateOp = rewriter.create<handshake::GateOp>(
            //     predecessorOpPointer->getLoc(), predecessorOpAddr,
            //     prevResult);
            // inheritBB(predecessorOpPointer, gateOp);

            SmallVector<Value> delayedAddresses =
                createDelayGenerator(predecessorOpAddr, predecessorOpPointer, N,
                                     true, addressDelayGenerator, rewriter);

            StringRef successorOpName = dependency.getDstAccess();
            Operation *successorOpPointer = memAccesses[successorOpName];

            SmallVector<Value> skipConditions = createSkipConditionForPair(
                predecessorOpDoneSignal, predecessorOpPointer,
                successorOpPointer, delayedAddresses, N,
                dependenciesMapForPhiNetwork, rewriter);
            skipConditionForEachPair[predecessorOpName][successorOpName] =
                skipConditions;
            handledSuccessors.push_back(successorOpName);
          }
        }
      }
    }
  }

  if (failed(createPhiNetworkDeps(funcOp.getRegion(), rewriter,
                                  dependenciesMapForPhiNetwork)))
    llvm::errs() << "Failed to create phi network dependencies\n";

  llvm::errs() << "[INFO][SKIP] Created Skip Conditions\n";
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
    unsigned N,
    DenseMap<OpOperand *, SmallVector<Value>> &dependenciesMapForPhiNetwork,
    ConversionPatternRewriter &rewriter) {

  SmallVector<Operation *> conditionalSequentializerOps;

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
WaitingSignalForSucc createWaitingSignalsForAllPairs(
    MemAccesses &memAccesses, SkipConditionForPair &skipConditionForEachPair,
    MLIRContext *ctx, FuncOp funcOp, std::vector<unsigned> NVector,
    ConversionPatternRewriter &rewriter) {

  WaitingSignalForSucc waitingSignalsForEachSuccessor;

  DenseMap<OpOperand *, SmallVector<Value>> dependenciesMapForPhiNetwork;

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

        SmallVector<Operation *> doneDelayGenerator;

        unsigned effective_N = N == 0 ? 1 : N;
        SmallVector<Value> delayedDoneSignals = createDelayGenerator(
            predecessorOpDoneSignal, predecessorOpPointer, effective_N, false,
            doneDelayGenerator, rewriter);

        StringRef successorName = dependency.getDstAccess();
        Operation *successorOpPointer = memAccesses[successorName];

        dependenceGraph.push_back(
            DependenceGraphEdge(predecessorOpPointer, successorOpPointer, N));

        SmallVector<Value> conds =
            skipConditionForEachPair[predecessorOpName][successorName];

        Value waitingSignal = createWaitingSignalForPair(
            predecessorOpDoneSignal, delayedDoneSignals, conds,
            predecessorOpPointer, successorOpPointer, N,
            dependenciesMapForPhiNetwork, rewriter);
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
  if (failed(createPhiNetworkDeps(funcOp.getRegion(), rewriter,
                                  dependenciesMapForPhiNetwork)))
    llvm::errs() << "Failed to create phi network dependencies\n";

  llvm::errs() << "[INFO][SKIP] Created Waiting Signals\n";
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

  // Checking is direct is necessary, because if it was direct this means that
  // it hasn't gone through FTD
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
    WaitingSignalForSucc &waitingSignalsForEachSuccessor, FuncOp &funcOp,
    ConversionPatternRewriter &rewriter) {

  DenseMap<OpOperand *, SmallVector<Value>> dependenciesMapForPhiNetwork;

  for (auto [dstAccess, waitingSignals] : waitingSignalsForEachSuccessor) {
    Operation *op = memAccesses[dstAccess];

    auto isDirect = isWaitingSignalForSuccDirect[dstAccess];
    gateAddress(op, waitingSignals, rewriter, op->getLoc(), isDirect,
                dependenciesMapForPhiNetwork);
  }

  if (failed(createPhiNetworkDeps(funcOp.getRegion(), rewriter,
                                  dependenciesMapForPhiNetwork)))
    llvm::errs() << "Failed to create phi network dependencies\n";
  llvm::errs() << "[INFO][SKIP] Gated Successor Accesses\n";
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

  std::vector<unsigned> NVector = getNVector(NStr);

  MemAccesses memAccesses;
  SkipConditionForPair skipConditionForEachPair;
  WaitingSignalForSucc waitingSignalsForEachSuccessor;
  std::vector<Operation *> consumerOpListForFTD;

  memAccesses = findMemAccessesInFunc(funcOp);

  skipConditionForEachPair =
      createSkipConditionsForAllPairs(memAccesses, funcOp, NVector, rewriter);

  waitingSignalsForEachSuccessor = createWaitingSignalsForAllPairs(
      memAccesses, skipConditionForEachPair, ctx, funcOp, NVector, rewriter);

  gateAllSuccessorAccesses(memAccesses, waitingSignalsForEachSuccessor, funcOp,
                           rewriter);
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
  llvm::errs() << "[INFO][SKIP] Dependency graph written to " << filename
               << "\n";
}

void runFTDOnSpecificConsumerOps(
    FuncOp funcOp, PatternRewriter &rewriter,
    std::vector<Operation *> (*ftdFunc)(PatternRewriter &, FuncOp &,
                                        Operation *, Value)) {
  std::vector<std::vector<Operation *>> allNewUnits;
  for (auto const [consumerOp, indices] : consumerOpAndOperandIndexForFTD)
    for (auto index : indices) {
      std::vector<Operation *> newUnits =
          ftdFunc(rewriter, funcOp, consumerOp, consumerOp->getOperand(index));
      allNewUnits.push_back(newUnits);
    }

  for (auto &someNewUnits : allNewUnits) {
    for (auto *unit : someNewUnits) {
      int i = 0;
      for (auto _ : unit->getOperands()) {
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
    if (failed(replaceMergeToGSA(funcOp, rewriter, newUnits)))
      signalPassFailure();

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

    if (failed(cfg::flattenFunction(funcOp)))
      signalPassFailure();
  }

  std::string path = compDir + "/" + kernelName + "_DEP_G.dot";

  writeDepGraphToDotFile(path);
  llvm::errs()
      << "[INFO][SKIP] Inserted Out with LSQs circuit successfully! \n";
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeInsertSkippableSeq(const std::string &NStr,
                                             const std::string &kernelName,
                                             const std::string &compDir) {
  return std::make_unique<HandshakeInsertSkippableSeqPass>();
}
