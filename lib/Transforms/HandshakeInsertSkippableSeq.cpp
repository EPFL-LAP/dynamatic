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
} // namespace

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

Value boolToSelect (Value cond, Value startSignal, OpBuilder& builder, Location loc){
  handshake::ConstantOp zeroConstOp = builder.create<handshake::ConstantOp>(loc, builder.getI32IntegerAttr(0), startSignal);
  handshake::ConstantOp oneConstOp = builder.create<handshake::ConstantOp>(loc, builder.getI32IntegerAttr(1), startSignal);
  handshake::NotOp notOp = builder.create<handshake::NotOp>(loc, cond);
  handshake::ConditionalBranchOp zeroConditionalBranchOp = builder.create<handshake::ConditionalBranchOp>(loc, cond, zeroConstOp);
  handshake::ConditionalBranchOp oneConditionalBranchOp = builder.create<handshake::ConditionalBranchOp>(loc, notOp.getResult(), oneConstOp);

  SmallVector<Value, 2> values = {zeroConditionalBranchOp.getResult(0), oneConditionalBranchOp.getResult(0)};
  handshake::MergeOp mergeOp = builder.create<handshake::MergeOp>(loc, values);

  return mergeOp.getResult();
}


Value createSkip(Value waitingToken, Value cond, Value startSignal, OpBuilder& builder, Location loc){
  handshake::UnbundleOp unbundleOp = builder.create<handshake::UnbundleOp>(loc, cond);
  handshake::ConditionalBranchOp conditionalBranchOp = builder.create<handshake::ConditionalBranchOp>(loc, cond, unbundleOp.getResult(0));
  SmallVector<Value, 2> muxOpValues = {conditionalBranchOp.getResult(1), waitingToken};
  Value select = boolToSelect(cond, startSignal, builder, loc);
  handshake::MuxOp muxOp = builder.create<handshake::MuxOp>(loc, select, muxOpValues);

  ValueRange *ab = new ValueRange();
  handshake::ChannelType ch =  handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  handshake::BundleOp bundleOp = builder.create<handshake::BundleOp>(loc, unbundleOp.getResult(0), unbundleOp.getResult(1), *ab, ch);

  builder.create<handshake::SinkOp>(loc, bundleOp.getResult(0));
  return muxOp.getResult();
}

SmallVector<Value> insertMuxes(SmallVector<Value> mainValues, SmallVector<Value> conds, Value startSignal, OpBuilder& builder, Location loc){
  SmallVector<Value> results;
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)){
    results.push_back(createSkip(mainValue, cond, startSignal, builder, loc));
  }
  return results;
}

Value calculateCFGCond(Operation* sourceOp, Operation* dstOp, Value startSignal, OpBuilder& builder, Location loc){
  return builder.create<handshake::ConstantOp>(loc, builder.getBoolAttr(true), startSignal);
}



Value createWaitingSignalForPair(Value sourceOpDoneSignal, SmallVector<Value> conds, Value CFGCond, Value startSignal, OpBuilder& builder, Location loc){
  SmallVector<Value> delayedDoneSignals = getNDelayedValues(sourceOpDoneSignal, startSignal, builder, loc);
  SmallVector<Value> branchedDoneSignals = insertBranches(delayedDoneSignals, conds, builder, loc);
  SmallVector<Value> muxedDoneSignals = insertMuxes(branchedDoneSignals, conds, startSignal, builder, loc);
  handshake::JoinOp joinOp = builder.create<handshake::JoinOp>(loc, muxedDoneSignals);
  return createSkip(joinOp.getResult(), CFGCond, startSignal, builder, loc);
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
    llvm::errs() << "[] Created Waiting Signals\n";
}

void gateAddress(Operation* op, SmallVector<Value> waitingValues, OpBuilder& builder, Location loc){
  Value address = op->getOperand(0);
  handshake::UnbundleOp unbundleOp = builder.create<handshake::UnbundleOp>(loc, address);
  waitingValues.push_back(unbundleOp.getResult(0));
  llvm::errs() << "&&&" << unbundleOp.getResult(0) << "\n";
  handshake::JoinOp joinOp = builder.create<handshake::JoinOp>(loc, waitingValues);
  ValueRange *ab = new ValueRange();
  handshake::ChannelType ch =  handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  handshake::BundleOp bundleOp = builder.create<handshake::BundleOp>(loc, joinOp.getResult(), unbundleOp.getResult(1), *ab, ch);
  op->setOperand(0, bundleOp.getResult(0));
}

void gateAllDstAccesses(DenseMap<StringRef, Operation*> &memAccesses, DenseMap<StringRef, SmallVector<Value>> &waitingSignalsForEachDst, OpBuilder& builder){
  for (auto [dstAccess, waitingSignals]: waitingSignalsForEachDst){
    Operation* op = memAccesses[dstAccess];
    gateAddress(op, waitingSignals, builder, op->getLoc());
  }
  llvm::errs() << "[] Gated Dst Accesses\n";
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
          handshake::MergeOp mergeOp = builder.create<handshake::MergeOp>(loc, diffTokens);

          Value CFGCond = calculateCFGCond(sourceOpPointer, dstOpPointer, startSignal, builder, loc);
          handshake::ConditionalBranchOp conditionalBranchOp = builder.create<handshake::ConditionalBranchOp>(loc, CFGCond, dstOpPointer->getOperand(0));

          // Type addrType = conditionalBranchOp.getResult(0).getType();
          // // TypedAttr typedattr = TypedAttr(addrType);
          // handshake::ConstantOp constOp = builder.create<handshake::ConstantOp>(loc, builder.getI32IntegerAttr(1000), startSignal);
          // ValueRange *ab = new ValueRange();
          // llvm::errs()<< addrType<<"**\n";
          
          // handshake::ChannelType ch =  dyn_cast<handshake::ChannelType>(addrType);
          // llvm::errs()<< ch <<"**\n";
          // handshake::UnbundleOp unbundleOp = builder.create<handshake::UnbundleOp>(loc, constOp);

          // llvm::errs() << unbundleOp.getResult(1) << "8\n";


          // handshake::BundleOp bundleOp = builder.create<handshake::BundleOp>(loc, mergeOp2.getResult(), unbundleOp.getResult(1), *ab, ch);


          // llvm::errs()<< "^^^^^" << bundleOp.getResult(0).getType() << "\n";

          // llvm::errs()<< "^^^^^" << conditionalBranchOp.getResult(0).getType() << "\n";

          handshake::UnbundleOp unbundleOp = builder.create<handshake::UnbundleOp>(loc, conditionalBranchOp.getResult(0));

          SmallVector<Value, 2> JoinOpValues = {mergeOp.getResult(), unbundleOp.getResult(0)};
          handshake::JoinOp joinOp = builder.create<handshake::JoinOp>(loc, JoinOpValues);

          ValueRange *ab = new ValueRange();
          handshake::ChannelType ch =  handshake::ChannelType::get(unbundleOp.getResult(1).getType());
          handshake::BundleOp bundleOp = builder.create<handshake::BundleOp>(loc, joinOp.getResult(), unbundleOp.getResult(1), *ab, ch);

          Value gatedDstOpaddr = bundleOp.getResult(0);

          for (Value delayedAddress : delayedAddresses){
              handshake::CmpIOp cmpIOp = builder.create<handshake::CmpIOp>(loc, CmpIPredicate::ne, gatedDstOpaddr, delayedAddress);
              skipConditionForEachPair[sourceOpName][dstAccess].push_back(cmpIOp.getResult());
          }          
        }
      }
    }
    llvm::errs() << "[] Created Skip Conditions\n";
}

void HandshakeInsertSkippableSeqPass::runDynamaticPass() {
  llvm::errs() << "@@@@@@@@@@@@@@@@@@@@@@@@@\n";
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




std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeInsertSkippableSeq() {
  return std::make_unique<HandshakeInsertSkippableSeqPass>();
}

