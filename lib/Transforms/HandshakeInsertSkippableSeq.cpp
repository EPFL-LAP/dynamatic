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
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"


using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

unsigned N = 3;

namespace {

struct HandshakeInsertSkippableSeqPass
    : public dynamatic::impl::HandshakeInsertSkippableSeqBase<
          HandshakeInsertSkippableSeqPass> {

  void runDynamaticPass() override;


};


void findMemAccessesInFunc(FuncOp funcOp, DenseMap<StringRef, Operation*> &memAccesses){

      for (BlockArgument arg : funcOp.getArguments()) {
        llvm::errs() << "[traversing arguments]" << arg << "\n";
        if (auto memref = dyn_cast<TypedValue<mlir::MemRefType>>(arg)){
            auto memrefUsers = memref.getUsers();

            assert(std::distance(memrefUsers.begin(), memrefUsers.end()) <= 1 && "expected at most one memref user");

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
}

SmallVector<Value> getNDelayedValues(Value initialVal, Value constVal, OpBuilder& builder, Location loc){
  Value prevResult = initialVal;
  SmallVector<Value> delayedVals= {initialVal};
  for (unsigned i = 0; i < N-1; i++){
      SmallVector<Value, 2> values = {prevResult, constVal};
      handshake::MergeOp mergOp = builder.create<handshake::MergeOp>(loc, values);
      delayedVals.push_back(mergOp->getResult(0));
  }
  return delayedVals;
}

SmallVector<Value> insertBranches(SmallVector<Value> mainValues, SmallVector<Value> conds, OpBuilder& builder, Location loc){
  SmallVector<Value> results;
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)){
    handshake::ConditionalBranchOp conditionalBranchOp = builder.create<handshake::ConditionalBranchOp>(loc, cond, mainValue);
    results.push_back(conditionalBranchOp.getResult(0));
  }
  return results;
}

Value createSkip(Value waitingToken, Value cond, OpBuilder& builder, Location loc){
  handshake::UnbundleOp unbundleOp = builder.create<handshake::UnbundleOp>(loc, cond);
  handshake::ConditionalBranchOp conditionalBranchOp = builder.create<handshake::ConditionalBranchOp>(loc, cond, unbundleOp.getResult(0));
  SmallVector<Value, 2> muxOpValues = {conditionalBranchOp.getResult(1), waitingToken};
  handshake::MuxOp muxOp = builder.create<handshake::MuxOp>(loc, cond, muxOpValues);
  return muxOp.getResult();
}

SmallVector<Value> insertMuxes(SmallVector<Value> mainValues, SmallVector<Value> conds, OpBuilder& builder, Location loc){
  SmallVector<Value> results;
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)){
    results.push_back(createSkip(mainValue, cond, builder, loc));
  }
  return results;
}

Value calculateCFGCond(Operation* sourceOp, Operation* dstOp, Value startSignal, OpBuilder& builder, Location loc){
  return builder.create<handshake::ConstantOp>(loc, builder.getBoolAttr(true), startSignal);
}



Value createWaitingSignalForPair(Value sourceOpDoneSignal, SmallVector<Value> conds, Value CFGCond, Value startSignal, OpBuilder& builder, Location loc){
  SmallVector<Value> delayedDoneSignals = getNDelayedValues(sourceOpDoneSignal, startSignal, builder, loc);
  SmallVector<Value> branchedDoneSignals = insertBranches(delayedDoneSignals, conds, builder, loc);
  SmallVector<Value> muxedDoneSignals = insertMuxes(branchedDoneSignals, conds, builder, loc);
  handshake::JoinOp joinOp = builder.create<handshake::JoinOp>(loc, muxedDoneSignals);
  return createSkip(joinOp.getResult(), CFGCond, builder, loc);
}

void createWaitingSignals(FuncOp funcOp, DenseMap<StringRef, Operation*> &memAccesses, DenseMap<StringRef, SmallVector<Value>> &waitingSignalsForEachDst, DenseMap<StringRef, DenseMap<StringRef, SmallVector<Value>>> &skipConditionForEachPair, OpBuilder& builder){
    Value startSignal = (Value)funcOp.getArguments().back();
    
    for (auto [sourceOpName, sourceOpPointer] : memAccesses){
      builder.setInsertionPointToStart(sourceOpPointer->getBlock());
      Value sourceOpDoneSignal = sourceOpPointer->getResult(2);
      Location loc = sourceOpPointer->getLoc();
      
      if (auto deps = getDialectAttr<MemDependenceArrayAttr>(sourceOpPointer)){
        for (MemDependenceAttr dependency : deps.getDependencies()) {
          StringRef dstAccess = dependency.getDstAccess();
          Operation* dstOpPointer = memAccesses[dstAccess];

          SmallVector<Value> conds = skipConditionForEachPair[sourceOpName][dstAccess];
          Value CFGCond = calculateCFGCond(sourceOpPointer, dstOpPointer, startSignal, builder, loc);

          Value waitingSignal = createWaitingSignalForPair(sourceOpDoneSignal, conds, CFGCond, startSignal, builder, loc);
          waitingSignalsForEachDst[dstAccess].push_back(waitingSignal);
        }
      }
    }
}

void gateAddress(Operation* op, SmallVector<Value> waitingValues, OpBuilder& builder, Location loc){
  Value address = op->getOperand(0);
  waitingValues.push_back(address);
  handshake::JoinOp joinOp = builder.create<handshake::JoinOp>(loc, waitingValues);
  op->setOperand(0, joinOp.getResult());
}

void gateAllDstAccesses(DenseMap<StringRef, Operation*> &memAccesses, DenseMap<StringRef, SmallVector<Value>> &waitingSignalsForEachDst, OpBuilder& builder){
  for (auto [dstAccess, waitingSignals]: waitingSignalsForEachDst){
    Operation* op = memAccesses[dstAccess];
    gateAddress(op, waitingSignals, builder, op->getLoc());
  }
}

void createSkipConditions(FuncOp funcOp, DenseMap<StringRef, Operation*> &memAccesses, DenseMap<StringRef, DenseMap<StringRef, SmallVector<Value>>> &skipConditionForEachPair, OpBuilder &builder){
    Value startSignal = (Value)funcOp.getArguments().back();

    for (auto [sourceOpName, sourceOpPointer] : memAccesses){

      builder.setInsertionPointToStart(sourceOpPointer->getBlock());
      Value sourceOpDoneSignal = sourceOpPointer->getResult(2);
      Value sourceOpAddr = sourceOpPointer->getOperand(0);
      Location loc = sourceOpPointer->getLoc();


      handshake::ConstantOp constOp = builder.create<handshake::ConstantOp>(loc, builder.getI32IntegerAttr(1000), startSignal);
      SmallVector<Value> delayedAddresses = getNDelayedValues(sourceOpAddr, constOp.getResult(), builder, loc);
           

      if (auto deps = getDialectAttr<MemDependenceArrayAttr>(sourceOpPointer)){
        for (MemDependenceAttr dependency : deps.getDependencies()) {
          StringRef dstAccess = dependency.getDstAccess();
          Operation* dstOpPointer = memAccesses[dstAccess];

          SmallVector<Value> diffTokens = {sourceOpDoneSignal};
          diffTokens.append(N, startSignal);
          handshake::MergeOp mergeOp2 = builder.create<handshake::MergeOp>(loc, diffTokens);

          Value CFGCond = calculateCFGCond(sourceOpPointer, dstOpPointer, startSignal, builder, loc);
          handshake::ConditionalBranchOp conditionalBranchOp = builder.create<handshake::ConditionalBranchOp>(loc, CFGCond, dstOpPointer->getOperand(0));

          SmallVector<Value, 2> JoinOpValues = {mergeOp2->getResult(0), conditionalBranchOp.getResult(0)};
          handshake::JoinOp JoinOp = builder.create<handshake::JoinOp>(loc, JoinOpValues);
          Value gatedDstOpaddr = JoinOp.getResult();

          for (Value delayedAddress : delayedAddresses){
              handshake::CmpIOp cmpIOp = builder.create<handshake::CmpIOp>(loc, CmpIPredicate::ne, gatedDstOpaddr, delayedAddress);
              skipConditionForEachPair[sourceOpName][dstAccess].push_back(cmpIOp.getResult());
          }          
        }
      }
    }
    
}

void HandshakeInsertSkippableSeqPass::runDynamaticPass() {
  mlir::ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);


  for (auto funcOp : modOp.getOps<handshake::FuncOp>()){
      DenseMap<StringRef, Operation*> memAccesses;
      DenseMap<StringRef, DenseMap<StringRef, SmallVector<Value>>> skipConditionForEachPair;
      DenseMap<StringRef, SmallVector<Value>> waitingSignalsForEachDst;

      findMemAccessesInFunc(funcOp, memAccesses);

      createSkipConditions(funcOp, memAccesses, skipConditionForEachPair, builder);

      createWaitingSignals(funcOp, memAccesses, waitingSignalsForEachDst, skipConditionForEachPair, builder);

      gateAllDstAccesses(memAccesses, waitingSignalsForEachDst, builder);
  }
}

} // namespace


std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeInsertSkippableSeq() {
  return std::make_unique<HandshakeInsertSkippableSeqPass>();
}

