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
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "mlir/Transforms/DialectConversion.h"


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

int getBBNumberFromOp(Operation * op){
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

SmallVector<Value> getNDelayedValues(Value initialVal, Value constVal, Operation* BBOp, ConversionPatternRewriter& rewriter){
  Value prevResult = initialVal;
  SmallVector<Value> delayedVals= {initialVal};
  for (unsigned i = 0; i < N-1; i++){
      SmallVector<Value, 2> values = {prevResult, constVal};
      handshake::MergeOp mergOp = rewriter.create<handshake::MergeOp>(BBOp->getLoc(), values);
      inheritBB(BBOp, mergOp);
      delayedVals.push_back(mergOp->getResult(0));
  }
  return delayedVals;
}

SmallVector<Value> insertBranches(SmallVector<Value> mainValues, SmallVector<Value> conds, Operation* BBOp, ConversionPatternRewriter& rewriter){
  SmallVector<Value> results;
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)){
    handshake::ConditionalBranchOp conditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(BBOp->getLoc(), cond, mainValue);
    inheritBB(BBOp, conditionalBranchOp);
    results.push_back(conditionalBranchOp.getResult(0));
  }
  return results;
}

Value boolToSelect (Value cond, Value startSignal, Operation* predecessorOp, ConversionPatternRewriter& rewriter){
  handshake::ConstantOp zeroConstOp = rewriter.create<handshake::ConstantOp>(predecessorOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0), startSignal);
  handshake::ConstantOp oneConstOp = rewriter.create<handshake::ConstantOp>(predecessorOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1), startSignal);
  inheritBB(predecessorOp, zeroConstOp);
  inheritBB(predecessorOp, oneConstOp);

  handshake::NotOp notOp = rewriter.create<handshake::NotOp>(predecessorOp->getLoc(), cond);
  inheritBB(predecessorOp, notOp);
  handshake::ConditionalBranchOp zeroConditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(predecessorOp->getLoc(), cond, zeroConstOp);
  handshake::ConditionalBranchOp oneConditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(predecessorOp->getLoc(), notOp.getResult(), oneConstOp);
  inheritBB(predecessorOp, zeroConditionalBranchOp);
  inheritBB(predecessorOp, oneConditionalBranchOp);

  SmallVector<Value, 2> values = {zeroConditionalBranchOp.getResult(0), oneConditionalBranchOp.getResult(0)};
  handshake::MergeOp mergeOp = rewriter.create<handshake::MergeOp>(predecessorOp->getLoc(), values);
  inheritBB(predecessorOp, mergeOp); 

  return mergeOp.getResult();
}


Value createSkip(Value waitingToken, Value cond, Value startSignal, Operation* predecessorOp, ConversionPatternRewriter& rewriter){
  handshake::UnbundleOp unbundleOp = rewriter.create<handshake::UnbundleOp>(predecessorOp->getLoc(), cond);
  inheritBB(predecessorOp, unbundleOp);
  handshake::ConditionalBranchOp conditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(predecessorOp->getLoc(), cond, unbundleOp.getResult(0));
  inheritBB(predecessorOp, conditionalBranchOp);
  SmallVector<Value, 2> muxOpValues = {conditionalBranchOp.getResult(1), waitingToken};
  Value select = boolToSelect(cond, startSignal, predecessorOp, rewriter);
  handshake::MuxOp muxOp = rewriter.create<handshake::MuxOp>(predecessorOp->getLoc(), select, muxOpValues);
  inheritBB(predecessorOp, muxOp);

  ValueRange *ab = new ValueRange();
  handshake::ChannelType ch =  handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(predecessorOp->getLoc(), unbundleOp.getResult(0), unbundleOp.getResult(1), *ab, ch);
  inheritBB(predecessorOp, bundleOp);

  rewriter.create<handshake::SinkOp>(predecessorOp->getLoc(), bundleOp.getResult(0));
  return muxOp.getResult();
}

SmallVector<Value> insertConditionalSkips(SmallVector<Value> mainValues, SmallVector<Value> conds, Operation* predecessorOp, Value startSignal, ConversionPatternRewriter& rewriter){
  SmallVector<Value> results;
  llvm::errs() << "function \n";
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)){
    Value skipValue = createSkip(mainValue, cond, startSignal, predecessorOp, rewriter);
    llvm::errs() << "skip: ";
    llvm::errs() << skipValue << "\n";
    results.push_back(skipValue);
  }
  return results;
}

DenseMap<int, Operation*> getControlMergeOps(FuncOp funcOp){
  DenseMap<int, Operation*> results;
  for (ControlMergeOp controlMergeOp : funcOp.getOps<handshake::ControlMergeOp>()){
    int BB = getBBNumberFromOp(controlMergeOp);
    results[BB] = controlMergeOp;
  }
  return results;
}


Value calculateCFGCond(Operation* sourceOp, Operation* dstOp, DenseMap<int, Operation*> &controlMerges, Value startSignal, ConversionPatternRewriter& rewriter){
  int sourceBB = getBBNumberFromOp(sourceOp);
  int dstBB = getBBNumberFromOp(dstOp);
  ControlMergeOp controlMergeOp = dyn_cast<handshake::ControlMergeOp>(controlMerges[dstBB]);

  handshake::ConstantOp constOp = rewriter.create<handshake::ConstantOp>(dstOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0), startSignal);
  inheritBB(dstOp, constOp);
  llvm::errs() << controlMergeOp->getResult(1).getType() << "tof %% \n";
  llvm::errs() << constOp << "tof %% \n";
  llvm::errs() << constOp.getValue().getType() << "tof %% \n";
  handshake::CmpIOp cmpIOp = rewriter.create<handshake::CmpIOp>(dstOp->getLoc(), CmpIPredicate::ugt, constOp.getResult(), controlMergeOp->getResult(1));
  inheritBB(dstOp, cmpIOp);

  return cmpIOp.getResult();
}


Value createWaitingSignalForPair(Value sourceOpDoneSignal, SmallVector<Value> conds, Value CFGCond, Operation* predecessorOp, Operation* successorOp, Value startSignal, ConversionPatternRewriter& rewriter){
  handshake::ConstantOp constOp = rewriter.create<handshake::ConstantOp>(predecessorOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0), startSignal);
  inheritBB(predecessorOp, constOp);
  handshake::UnbundleOp unbundleOp = rewriter.create<handshake::UnbundleOp>(predecessorOp->getLoc(), constOp.getResult());
  inheritBB(predecessorOp, unbundleOp);
  ValueRange *ab = new ValueRange();
  handshake::ChannelType ch =  handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(predecessorOp->getLoc(), unbundleOp.getResult(0), unbundleOp.getResult(1), *ab, ch);
  inheritBB(predecessorOp, bundleOp);
  rewriter.create<handshake::SinkOp>(predecessorOp->getLoc(), bundleOp.getResult(0));
  

  SmallVector<Value> delayedDoneSignals = getNDelayedValues(sourceOpDoneSignal, unbundleOp.getResult(0), predecessorOp, rewriter);
  SmallVector<Value> branchedDoneSignals = insertBranches(delayedDoneSignals, conds, predecessorOp, rewriter);
  SmallVector<Value> conditionallySkippedDoneSignals = insertConditionalSkips(branchedDoneSignals, conds, successorOp, startSignal, rewriter);
  handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(predecessorOp->getLoc(), conditionallySkippedDoneSignals);
  inheritBB(predecessorOp, joinOp);
  return createSkip(joinOp.getResult(), CFGCond, startSignal, successorOp, rewriter);
}

MemDependenceAttr getInactivatedDependency(MemDependenceAttr dependency) {
  MLIRContext *ctx = dependency.getContext();
  return MemDependenceAttr::get(
      ctx, dependency.getDstAccess(), dependency.getLoopDepth(),
      dependency.getComponents(), BoolAttr::get(ctx, false));
}

void createWaitingSignals(FuncOp funcOp, DenseMap<StringRef, Operation*> &memAccesses, DenseMap<StringRef, SmallVector<Value>> &waitingSignalsForEachDst, DenseMap<StringRef, DenseMap<StringRef, SmallVector<Value>>> &skipConditionForEachPair, DenseMap<int, Operation*> &controlMerges, MLIRContext* ctx, ConversionPatternRewriter& rewriter){
    Value startSignal = (Value)funcOp.getArguments().back();
    
    for (auto [sourceOpName, sourceOpPointer] : memAccesses){
      rewriter.setInsertionPointToStart(sourceOpPointer->getBlock());
      Value sourceOpDoneSignal = sourceOpPointer->getResult(2);
      
      SmallVector<MemDependenceAttr> newDeps;
      if (auto deps = getDialectAttr<MemDependenceArrayAttr>(sourceOpPointer)){
        for (MemDependenceAttr dependency : deps.getDependencies()) {
          if (!dependency.getIsActive().getValue()){
            newDeps.push_back(dependency);
            continue;
          }

          StringRef dstAccess = dependency.getDstAccess();
          Operation* dstOpPointer = memAccesses[dstAccess];

          SmallVector<Value> conds = skipConditionForEachPair[sourceOpName][dstAccess];
          Value CFGCond = calculateCFGCond(sourceOpPointer, dstOpPointer, controlMerges, startSignal, rewriter);

          Value waitingSignal = createWaitingSignalForPair(sourceOpDoneSignal, conds, CFGCond, sourceOpPointer, dstOpPointer, startSignal, rewriter);
          waitingSignalsForEachDst[dstAccess].push_back(waitingSignal);

          newDeps.push_back(getInactivatedDependency(dependency));
        }
        setDialectAttr<MemDependenceArrayAttr>(sourceOpPointer, ctx, newDeps);
        setDialectAttr<MemInterfaceAttr>(sourceOpPointer, ctx);
      }
    }
    llvm::errs() << "[] Created Waiting Signals\n";
}

void gateAddress(Operation* op, SmallVector<Value> waitingValues, ConversionPatternRewriter& rewriter, Location loc){
  Value address = op->getOperand(0);
  handshake::UnbundleOp unbundleOp = rewriter.create<handshake::UnbundleOp>(loc, address);
  inheritBB(op, unbundleOp);
  waitingValues.push_back(unbundleOp.getResult(0));
  handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(loc, waitingValues);
  inheritBB(op, joinOp);
  ValueRange *ab = new ValueRange();
  handshake::ChannelType ch =  handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(loc, joinOp.getResult(), unbundleOp.getResult(1), *ab, ch);
  inheritBB(op, bundleOp);
  llvm::errs() << "&&&" << bundleOp.getResult(0);
  op->setOperand(0, bundleOp.getResult(0));
}

void gateAllDstAccesses(DenseMap<StringRef, Operation*> &memAccesses, DenseMap<StringRef, SmallVector<Value>> &waitingSignalsForEachDst, ConversionPatternRewriter& rewriter){
  for (auto [dstAccess, waitingSignals]: waitingSignalsForEachDst){
    Operation* op = memAccesses[dstAccess];
    gateAddress(op, waitingSignals, rewriter, op->getLoc());
  }
  llvm::errs() << "[] Gated Dst Accesses\n";
}

SmallVector<Value> createSkipConditionForPair(Value sourceOpDoneSignal, Value startSignal, Operation* sourceOpPointer, Operation* dstOpPointer, 
                    SmallVector<Value> delayedAddresses, DenseMap<int, Operation*> &controlMerges, 
                    ConversionPatternRewriter& rewriter){

  handshake::ConstantOp constOp = rewriter.create<handshake::ConstantOp>(sourceOpPointer->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0), startSignal);
  inheritBB(sourceOpPointer, constOp);
  handshake::UnbundleOp dummyUnbundleOp = rewriter.create<handshake::UnbundleOp>(sourceOpPointer->getLoc(), constOp.getResult());
  inheritBB(sourceOpPointer, dummyUnbundleOp);

  ValueRange *ab2 = new ValueRange();
  handshake::ChannelType ch2 =  handshake::ChannelType::get(dummyUnbundleOp.getResult(1).getType());
  handshake::BundleOp dummyBundleOp = rewriter.create<handshake::BundleOp>(sourceOpPointer->getLoc(), dummyUnbundleOp.getResult(0), dummyUnbundleOp.getResult(1), *ab2, ch2);
  inheritBB(sourceOpPointer, dummyBundleOp);

  rewriter.create<handshake::SinkOp>(sourceOpPointer->getLoc(), dummyBundleOp.getResult(0));

  SmallVector<Value> diffTokens = {sourceOpDoneSignal};
  diffTokens.append(N, dummyUnbundleOp.getResult(0));
  handshake::MergeOp mergeOp = rewriter.create<handshake::MergeOp>(sourceOpPointer->getLoc(), diffTokens);
  inheritBB(sourceOpPointer, mergeOp);

  Value CFGCond = calculateCFGCond(sourceOpPointer, dstOpPointer, controlMerges, startSignal, rewriter);
  handshake::ConditionalBranchOp conditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(dstOpPointer->getLoc(), CFGCond, dstOpPointer->getOperand(0));
  inheritBB(dstOpPointer, conditionalBranchOp);

  // Not sure which one (0 or 1)
  handshake::UnbundleOp unbundleOp = rewriter.create<handshake::UnbundleOp>(sourceOpPointer->getLoc(), conditionalBranchOp.getResult(1));
  inheritBB(sourceOpPointer, unbundleOp);

  SmallVector<Value, 2> JoinOpValues = {mergeOp.getResult(), unbundleOp.getResult(0)};
  handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(sourceOpPointer->getLoc(), JoinOpValues);
  inheritBB(sourceOpPointer, joinOp);

  ValueRange *ab = new ValueRange();
  handshake::ChannelType ch =  handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(sourceOpPointer->getLoc(), joinOp.getResult(), unbundleOp.getResult(1), *ab, ch);
  inheritBB(sourceOpPointer, bundleOp);

  Value gatedDstOpaddr = bundleOp.getResult(0);

  SmallVector<Value> skipConditions;
  for (Value delayedAddress : delayedAddresses){
      handshake::CmpIOp cmpIOp = rewriter.create<handshake::CmpIOp>(sourceOpPointer->getLoc(), CmpIPredicate::ne, gatedDstOpaddr, delayedAddress);
      inheritBB(sourceOpPointer, cmpIOp);
      skipConditions.push_back(cmpIOp.getResult());
  }          
  return skipConditions;
}

void createSkipConditions(FuncOp funcOp, DenseMap<StringRef, Operation*> &memAccesses, DenseMap<StringRef, DenseMap<StringRef, SmallVector<Value>>> &skipConditionForEachPair, DenseMap<int, Operation*> &controlMerges, ConversionPatternRewriter &rewriter){
    Value startSignal = (Value)funcOp.getArguments().back();

    for (auto [sourceOpName, sourceOpPointer] : memAccesses){
      rewriter.setInsertionPointToStart(sourceOpPointer->getBlock());
      Value sourceOpDoneSignal = sourceOpPointer->getResult(2);
      Value sourceOpAddr = sourceOpPointer->getOperand(0);

      // The dummy address
      handshake::ConstantOp constOp = rewriter.create<handshake::ConstantOp>(sourceOpPointer->getLoc(), rewriter.getIntegerAttr(rewriter.getI32Type(), 1000), startSignal);
      inheritBB(sourceOpPointer, constOp);
      SmallVector<Value> delayedAddresses = getNDelayedValues(sourceOpAddr, constOp.getResult(), sourceOpPointer, rewriter);

      if (auto deps = getDialectAttr<MemDependenceArrayAttr>(sourceOpPointer)){
        for (MemDependenceAttr dependency : deps.getDependencies()) {
          if (!dependency.getIsActive().getValue())
            continue;
          StringRef dstAccess = dependency.getDstAccess();
          Operation* dstOpPointer = memAccesses[dstAccess];

          SmallVector<Value> skipConditions = createSkipConditionForPair(sourceOpDoneSignal, startSignal, sourceOpPointer, dstOpPointer,delayedAddresses,controlMerges, rewriter);
          skipConditionForEachPair[sourceOpName][dstAccess] = skipConditions;
        }
      }
    }
    llvm::errs() << "[] Created Skip Conditions\n";
}

void HandshakeInsertSkippableSeqPass::runDynamaticPass() {
  mlir::ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  ConversionPatternRewriter rewriter(ctx);

  for (auto funcOp : modOp.getOps<handshake::FuncOp>()){
      DenseMap<StringRef, Operation*> memAccesses;
      DenseMap<StringRef, DenseMap<StringRef, SmallVector<Value>>> skipConditionForEachPair;
      DenseMap<StringRef, SmallVector<Value>> waitingSignalsForEachDst;
      DenseMap<int, Operation*> controlMerges =  getControlMergeOps(funcOp);

      findMemAccessesInFunc(funcOp, memAccesses);

      createSkipConditions(funcOp, memAccesses, skipConditionForEachPair, controlMerges, rewriter);
      createWaitingSignals(funcOp, memAccesses, waitingSignalsForEachDst, skipConditionForEachPair, controlMerges, ctx, rewriter);
      gateAllDstAccesses(memAccesses, waitingSignalsForEachDst, rewriter);   

  }


  llvm::errs() << "done! \n";
  
}



std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeInsertSkippableSeq() {
  return std::make_unique<HandshakeInsertSkippableSeqPass>();
}

