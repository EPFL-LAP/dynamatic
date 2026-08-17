//===- HandshakeCreateOutWithLSQsCircuit.cpp - Out with LSQs ----------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements the HandshakeCreateOutWithLSQsCircuit pass which inserts
// the `Out with LSQs` circuit based on https://doi.org/10.1145/3748173.3779204.
// This circuit replaces the LSQ circuit with elastic components. The circuit is
// inserted for each pair of memory accesses that have a dependency. For better
// comprehension, the reader is advised to refer to figure 8 in the paper.
//
//===----------------------------------------------------------------------===//

// #include "dynamatic/Transforms/HandshakeCreateOutWithLSQsCircuit.h"
#include "dynamatic/Analysis/ControlDependenceAnalysis.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/CfToHandshake.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DOT.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "experimental/Support/BooleanLogic/BoolExpression.h"
#include "experimental/Support/CFGAnnotation.h"
#include "experimental/Support/FtdImplementation.h"
#include "experimental/Support/FtdSupport.h"
#include "experimental/Transforms/HandshakeStraightToQueue.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include <unordered_set>

#define DEBUG_TYPE "out-with-lsqs"

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKECREATEOUTWITHLSQSCIRCUIT
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate]

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
using DoneSignalForMemoryOp = DenseMap<Operation *, Value>;

constexpr llvm::StringLiteral SKIP_COND_GEN("Skip.Condition_Generator");
constexpr llvm::StringLiteral SKIP_COND_SEQ("Skip.Conditional_Sequentializer");
const int DEFAULT_COMPARATOR_NUM = 3;

namespace {

struct HandshakeCreateOutWithLSQsCircuitPass
    : public dynamatic::impl::HandshakeCreateOutWithLSQsCircuitBase<
          HandshakeCreateOutWithLSQsCircuitPass> {

  using HandshakeCreateOutWithLSQsCircuitBase::
      HandshakeCreateOutWithLSQsCircuitBase;

  void runDynamaticPass() override;

  LogicalResult createBaseOutWithLSQsCircuit(FuncOp funcOp, MLIRContext *ctx);
};
} // namespace

DenseMap<Operation *, std::vector<int>> consumerOpAndOperandIndexForFTD;
IsWaitingSignalForSuccDirect isWaitingSignalForSuccDirect;
delayedDict delayedAddressesForEachPred;
delayedDict delayedDoneSignalsForEachPred;
std::unique_ptr<DOTGraph> comparatorGraph;

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
/// Returns the comparator count for a dependence. With a DOT graph, an edge
/// attribute overrides the default count of three; otherwise CLI counts cycle.
unsigned getNumComparatorsForDependence(StringRef srcAccess,
                                        StringRef dstAccess,
                                        ArrayRef<unsigned> counts,
                                        unsigned &countIndex) {
  if (!comparatorGraph) {
    unsigned count = counts[countIndex];
    countIndex = (countIndex + 1) % counts.size();
    return count;
  }
  const DOTGraph::Node *srcNode = comparatorGraph->getNode(srcAccess);
  if (!srcNode)
    return DEFAULT_COMPARATOR_NUM;
  for (const DOTGraph::Edge *edge : comparatorGraph->getSuccessors(*srcNode)) {
    if (edge->dstNode->id != dstAccess)
      continue;
    for (StringRef attrName : {"num-comparator", "num-comparators",
                               "num_comparator", "num_comparators"}) {
      auto attr = edge->attrs.find(attrName);
      if (attr == edge->attrs.end())
        continue;
      unsigned count;
      if (!StringRef(attr->getValue()).getAsInteger(10, count))
        return count;
      llvm::errs() << "[WARN] Invalid comparator count for " << srcAccess
                   << " -> " << dstAccess << "; using "
                   << DEFAULT_COMPARATOR_NUM << "\n";
      return DEFAULT_COMPARATOR_NUM;
    }
  }
  return DEFAULT_COMPARATOR_NUM;
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
  delayedValuesForEachPred[predecessorOp] = extendedDelayedValues;
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

// void addDrawingAttrToList(ArrayRef<Operation *> operations, StringRef attr) {
//   for (auto op : operations) {
//     auto drawAttr = handshake::DrawingAttr::get(op->getContext(), attr);
//     op->setAttr("drawing", drawAttr);
//   }
// }

void addAttrToList(ArrayRef<Operation *> operations, StringRef attrName,
                   Attribute attr) {
  for (auto op : operations) {
    op->setAttr(attrName, attr);
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

  addAttrToList(skipConditionGeneratorOps, SKIP_COND_GEN,
                rewriter.getUnitAttr());
  return skipConditions;
}

/// This function gets the done signal from a memory operation.
/// If its a load, it returns the data output, which is a channel.
/// However, if its a store, it returns the done signal which is a control
/// signal. This difference needs to be taken care of when using the done
/// signal.
Value getDoneSignalFromMemoryOp(Operation *memOp,
                                DoneSignalForMemoryOp &doneSignals,
                                ConversionPatternRewriter &rewriter) {
  if (auto loadOp = dyn_cast<handshake::LoadOp>(memOp)) {
    if (auto it = doneSignals.find(memOp); it != doneSignals.end())
      return it->second;

    Value loadResult = loadOp->getResult(1);
    Location loc = loadOp->getLoc();

    handshake::CtrlExtractorOp ctrlExtractorOp =
        rewriter.create<handshake::CtrlExtractorOp>(loc, loadResult);
    inheritBB(loadOp, ctrlExtractorOp);

    Value doneSignal = ctrlExtractorOp.getResult();
    doneSignals[memOp] = doneSignal;
    return doneSignal;
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
/// predecessor.
SkipConditionForPair
createSkipConditionsForAllPairs(MemAccesses &memAccesses, FuncOp funcOp,
                                std::vector<unsigned> Nvector,
                                DoneSignalForMemoryOp &doneSignals,
                                ConversionPatternRewriter &rewriter) {

  SkipConditionForPair skipConditionForEachPair;
  DenseMap<OpOperand *, SmallVector<Value>> dependenciesMapForPhiNetwork;

  unsigned NvectorIndex = 0;
  unsigned N;

  for (auto [predecessorOpName, predecessorOpPointer] : memAccesses) {
    rewriter.setInsertionPointToStart(predecessorOpPointer->getBlock());

    if (auto deps =
            getDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer)) {

      if (hasAtLeastOneActiveDep(deps)) {
        SmallVector<StringRef> handledSuccessors;
        Value predecessorOpDoneSignal =
            getDoneSignalFromMemoryOp(predecessorOpPointer, doneSignals,
                                      rewriter);
        Value predecessorOpAddr = predecessorOpPointer->getOperand(0);

        for (MemDependenceAttr dependency : deps.getDependencies()) {
          if (!dependency.getIsActive())
            continue;

          if (std::find(handledSuccessors.begin(), handledSuccessors.end(),
                        dependency.getDstAccess()) != handledSuccessors.end()) {
            continue;
          }

          N = getNumComparatorsForDependence(predecessorOpName,
                                             dependency.getDstAccess(), Nvector,
                                             NvectorIndex);
          if (N != 0) {
            SmallVector<Operation *> addressDelayGenerator;

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

  LLVM_DEBUG(llvm::errs() << "[INFO][SKIP] Created Skip Conditions\n";);
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

  addAttrToList(conditionalSequentializerOps, SKIP_COND_SEQ,
                rewriter.getUnitAttr());
  return joinOp.getResult();
}

/// This function returns the deactivated version of a given dependency.
MemDependenceAttr getDeactivatedDependency(MemDependenceAttr dependency) {
  MLIRContext *ctx = dependency.getContext();
  return MemDependenceAttr::get(ctx, dependency.getDstAccess(),
                                dependency.getLoopDepth(),
                                dependency.getDistance(), false);
}

/// This function creates the waiting signals for all pairs of memory
/// accesses. This means that it creates the right side of the circuit for
/// each pair. This includes the delay generator which is shared for the same
/// predecessor.
WaitingSignalForSucc createWaitingSignalsForAllPairs(
    MemAccesses &memAccesses, SkipConditionForPair &skipConditionForEachPair,
    MLIRContext *ctx, FuncOp funcOp, std::vector<unsigned> NVector,
    DoneSignalForMemoryOp &doneSignals,
    ConversionPatternRewriter &rewriter) {

  WaitingSignalForSucc waitingSignalsForEachSuccessor;

  DenseMap<OpOperand *, SmallVector<Value>> dependenciesMapForPhiNetwork;

  unsigned NvectorIndex = 0;
  unsigned N;

  for (auto [predecessorOpName, predecessorOpPointer] : memAccesses) {
    rewriter.setInsertionPointToStart(predecessorOpPointer->getBlock());

    if (auto deps =
            getDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer)) {

      if (hasAtLeastOneActiveDep(deps)) {
        Value predecessorOpDoneSignal =
            getDoneSignalFromMemoryOp(predecessorOpPointer, doneSignals,
                                      rewriter);

        SmallVector<StringRef> handledSuccessors;
        SmallVector<MemDependenceAttr> newDeps;

        for (MemDependenceAttr dependency : deps.getDependencies()) {
          if (!dependency.getIsActive()) {
            newDeps.push_back(dependency);
            continue;
          }
          if (std::find(handledSuccessors.begin(), handledSuccessors.end(),
                        dependency.getDstAccess()) != handledSuccessors.end()) {
            newDeps.push_back(getDeactivatedDependency(dependency));
            continue;
          }

          N = getNumComparatorsForDependence(predecessorOpName,
                                             dependency.getDstAccess(), NVector,
                                             NvectorIndex);

          SmallVector<Operation *> doneDelayGenerator;
          unsigned effective_N = N == 0 ? 1 : N;
          SmallVector<Value> delayedDoneSignals = createDelayGenerator(
              predecessorOpDoneSignal, predecessorOpPointer, effective_N, false,
              doneDelayGenerator, rewriter);

          StringRef successorName = dependency.getDstAccess();
          Operation *successorOpPointer = memAccesses[successorName];

          SmallVector<Value> conds =
              skipConditionForEachPair[predecessorOpName][successorName];

          Value waitingSignal = createWaitingSignalForPair(
              predecessorOpDoneSignal, delayedDoneSignals, conds,
              predecessorOpPointer, successorOpPointer, N,
              dependenciesMapForPhiNetwork, rewriter);
          waitingSignalsForEachSuccessor[successorName].push_back(
              waitingSignal);
          isWaitingSignalForSuccDirect[successorName].push_back(N == 0);

          newDeps.push_back(getDeactivatedDependency(dependency));
          handledSuccessors.push_back(successorName);
        }
        setDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer, ctx,
                                               newDeps);
      }
    }
    setDialectAttr<MemInterfaceAttr>(predecessorOpPointer, ctx);
  }
  if (failed(createPhiNetworkDeps(funcOp.getRegion(), rewriter,
                                  dependenciesMapForPhiNetwork)))
    llvm::errs() << "Failed to create phi network dependencies\n";

  LLVM_DEBUG(llvm::errs() << "[INFO][SKIP] Created Waiting Signals\n");
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

// This function gates all successor accesses with the waiting signals created
// before. This means that it creates the very last join in the figure.
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
  LLVM_DEBUG(llvm::errs() << "[INFO][SKIP] Gated Successor Accesses\n");
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
/// necessary components to replace the LSQ circuit with elastic components.
LogicalResult
HandshakeCreateOutWithLSQsCircuitPass::createBaseOutWithLSQsCircuit(
    FuncOp funcOp, MLIRContext *ctx) {

  if (numComparators.empty()) {
    funcOp.emitError("--num-of-comparators must not be empty");
    return failure();
  }

  comparatorGraph.reset();
  if (!depGraphFile.empty()) {
    comparatorGraph = std::make_unique<DOTGraph>();
    if (failed(comparatorGraph->getBuilder().parseFromFile(dynamaticDir + "/" +
                                                           depGraphFile)))
      llvm::errs() << "[INFO] failed to read the dot file.";
  }

  ConversionPatternRewriter rewriter(ctx);

  std::vector<unsigned> NVector = getNVector(numComparators);
  if (NVector.empty()) {
    funcOp.emitError("--num-of-comparators must contain a comparator count");
    return failure();
  }

  MemAccesses memAccesses;
  SkipConditionForPair skipConditionForEachPair;
  WaitingSignalForSucc waitingSignalsForEachSuccessor;
  DoneSignalForMemoryOp doneSignals;
  std::vector<Operation *> consumerOpListForFTD;

  memAccesses = findMemAccessesInFunc(funcOp);

  // Inserting the components is done in three main steps. First, the skip
  // conditions are created for each pair of memory accesses (i.e. the
  // comparators' part). Then, the waiting signals are created for each
  // successor (i.e. the join of all control signals it needs to wait for).
  // Finally, the successor accesses are gated with the waiting signals.

  skipConditionForEachPair =
      createSkipConditionsForAllPairs(memAccesses, funcOp, NVector,
                                      doneSignals, rewriter);

  waitingSignalsForEachSuccessor = createWaitingSignalsForAllPairs(
      memAccesses, skipConditionForEachPair, ctx, funcOp, NVector, doneSignals,
      rewriter);

  gateAllSuccessorAccesses(memAccesses, waitingSignalsForEachSuccessor, funcOp,
                           rewriter);
  // funcOp.print(llvm::errs());
  return success();
}

void runFTDOnSpecificConsumerOps(
    FuncOp funcOp, mlir::OpBuilder &builder,
    std::vector<Operation *> (*ftdFunc)(mlir::OpBuilder &, FuncOp &,
                                        Operation *, Value, ftd::ShadowCFG &),
    ftd::ShadowCFG shadowCFG) {
  std::vector<std::vector<Operation *>> allNewUnits;
  for (auto [consumerOp, indices] : consumerOpAndOperandIndexForFTD)
    for (auto index : indices) {
      std::vector<Operation *> newUnits =
          ftdFunc(builder, funcOp, consumerOp, consumerOp->getOperand(index),
                  shadowCFG);
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

ftd::ShadowCFG getShadow(FuncOp funcOp, MLIRContext *ctx) {
  unsigned capturedNumBlocks = 0;
  SmallVector<CapturedEdgeInfo> capturedEdges;
  DenseMap<unsigned, Value> capturedConditions;
  captureCFGTopology(funcOp, capturedNumBlocks, capturedEdges,
                     capturedConditions);

  OpBuilder builder(ctx);
  ftd::ShadowCFG shadow = buildShadowFromCapturedTopology(
      builder, funcOp, capturedNumBlocks, capturedEdges, capturedConditions);
  return shadow;
}

void HandshakeCreateOutWithLSQsCircuitPass::runDynamaticPass() {

  mlir::ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();

  ConversionPatternRewriter rewriter(ctx);
  OpBuilder builder(ctx);

  for (auto funcOp : modOp.getOps<handshake::FuncOp>()) {
    if (failed(cfg::restoreCfStructure(funcOp, rewriter)))
      signalPassFailure();

    // This is the main function which inserts the base circuit without FTD. It
    // internally calls `createPhiNetworkDeps`
    if (failed(createBaseOutWithLSQsCircuit(funcOp, ctx)))
      return signalPassFailure();

    LLVM_DEBUG(llvm::errs()
               << "[INFO][SKIP] Inserted skippable sequentializer circuit "
                  "successfully! \n");

    std::vector<Operation *> newUnits;
    if (failed(replaceMergeToGSA(funcOp, rewriter, newUnits)))
      signalPassFailure();

    LLVM_DEBUG(llvm::errs()
                   << "[INFO][SKIP] Replaced Merge to GSA successfully! \n";);

    for (auto *unit : newUnits) {
      int i = 0;
      for (auto operand : unit->getOperands()) {
        consumerOpAndOperandIndexForFTD[unit].push_back(i);
        i++;
      }
    }

    ftd::ShadowCFG shadowCFG = getShadow(funcOp, ctx);

    runFTDOnSpecificConsumerOps(funcOp, builder, addRegenOperandConsumer,
                                shadowCFG);
    runFTDOnSpecificConsumerOps(funcOp, builder, addSuppOperandConsumer,
                                shadowCFG);

    LLVM_DEBUG(llvm::errs() << "[INFO][SKIP] Added FTD successfully! \n");

    experimental::cfg::markBasicBlocks(funcOp, rewriter);

    if (failed(cfg::flattenFunction(funcOp)))
      signalPassFailure();

    ftd::resolveCondPlaceholders(funcOp, builder, shadowCFG);
    ftd::finalizeCondPlaceholders(funcOp);

    shadowCFG.destroy();
  }

  LLVM_DEBUG(llvm::errs()
             << "[INFO][SKIP] Inserted Out with LSQs circuit successfully! \n");
}
