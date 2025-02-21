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
  auto hi = constVal.getType();
  llvm::errs() << hi << "rrrrrrrrrrrrrrrrrrr\n";

  Value prevResult = initialVal;
  SmallVector<Value> delayedVals= {initialVal};
  SmallVector<Value, 2> values;
  
  for (unsigned i = 0; i < N-1; i++){   
    values = {prevResult, constVal};
    handshake::MergeOp mergeOp = rewriter.create<handshake::MergeOp>(BBOp->getLoc(), values);
    inheritBB(BBOp, mergeOp);
    delayedVals.push_back(mergeOp->getResult(0));
    prevResult = mergeOp->getResult(0);
  }
  return delayedVals;
}

SmallVector<Value> insertBranches(SmallVector<Value> mainValues, SmallVector<Value> conds, Operation* BBOp, ConversionPatternRewriter& rewriter){
  SmallVector<Value> results;
  for (auto [mainValue, cond] : llvm::zip(mainValues, conds)){
    handshake::ConditionalBranchOp conditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(BBOp->getLoc(), cond, mainValue);
    inheritBB(BBOp, conditionalBranchOp);
    results.push_back(conditionalBranchOp.getResult(1));
  }
  return results;
}

Value boolToSelect (Value cond, Value startSignal, Operation* predecessorOp, ConversionPatternRewriter& rewriter){
  llvm::errs() << "yahoo\n";
  handshake::SourceOp sourceOp = rewriter.create<handshake::SourceOp>(predecessorOp->getLoc());
  inheritBB(predecessorOp, sourceOp);
  handshake::ConstantOp zeroConstOp = rewriter.create<handshake::ConstantOp>(predecessorOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0), sourceOp);
  handshake::ConstantOp oneConstOp = rewriter.create<handshake::ConstantOp>(predecessorOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1), sourceOp);
  inheritBB(predecessorOp, zeroConstOp);
  inheritBB(predecessorOp, oneConstOp);

  handshake::ConditionalBranchOp zeroConditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(predecessorOp->getLoc(), cond, zeroConstOp);
  handshake::ConditionalBranchOp oneConditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(predecessorOp->getLoc(), cond, oneConstOp);
  inheritBB(predecessorOp, zeroConditionalBranchOp);
  inheritBB(predecessorOp, oneConditionalBranchOp);

  SmallVector<Value, 2> values = {zeroConditionalBranchOp.getResult(0), oneConditionalBranchOp.getResult(1)};
  handshake::MergeOp mergeOp = rewriter.create<handshake::MergeOp>(predecessorOp->getLoc(), values);
  inheritBB(predecessorOp, mergeOp); 

  llvm::errs() << "yahoo" << mergeOp.getResult().getType() << "\n";
  llvm::errs() << "yahoo" << cond.getType() << "\n";

  return mergeOp.getResult();


}


Value createSkip(Value waitingToken, Value cond, Value startSignal, Operation* predecessorOp, ConversionPatternRewriter& rewriter){
  // handshake::UnbundleOp unbundleOp = rewriter.create<handshake::UnbundleOp>(predecessorOp->getLoc(), cond);
  // inheritBB(predecessorOp, unbundleOp);
  // handshake::ConditionalBranchOp conditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(predecessorOp->getLoc(), cond, unbundleOp.getResult(0));
  // inheritBB(predecessorOp, conditionalBranchOp);
  handshake::SourceOp sourceOp = rewriter.create<handshake::SourceOp>(predecessorOp->getLoc());
  inheritBB(predecessorOp, sourceOp);
  SmallVector<Value, 2> muxOpValues = {waitingToken, sourceOp.getResult()};
  // Value select = boolToSelect(cond, startSignal, predecessorOp, rewriter);
  handshake::MuxOp muxOp = rewriter.create<handshake::MuxOp>(predecessorOp->getLoc(), cond, muxOpValues);
  inheritBB(predecessorOp, muxOp);

  // ValueRange *ab = new ValueRange();
  // handshake::ChannelType ch =  handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  // handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(predecessorOp->getLoc(), unbundleOp.getResult(0), unbundleOp.getResult(1), *ab, ch);
  // inheritBB(predecessorOp, bundleOp);

  // rewriter.create<handshake::SinkOp>(predecessorOp->getLoc(), bundleOp.getResult(0));
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


Value calculateCFGCond(Operation* predecessorOp, Operation* successorOp, DenseMap<int, Operation*> &controlMerges, Value startSignal, ConversionPatternRewriter& rewriter){
  int predecessorBB = getBBNumberFromOp(predecessorOp);
  int successorBB = getBBNumberFromOp(successorOp);
  ControlMergeOp controlMergeOp = dyn_cast<handshake::ControlMergeOp>(controlMerges[1]);

  handshake::SourceOp sourceOp = rewriter.create<handshake::SourceOp>(successorOp->getLoc());
  inheritBB(successorOp, sourceOp);
  handshake::ConstantOp constOp = rewriter.create<handshake::ConstantOp>(successorOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0), sourceOp);
  inheritBB(successorOp, constOp);
  llvm::errs() << controlMergeOp->getResult(1).getType() << "tof %% \n";
  llvm::errs() << constOp << "tof %% \n";
  llvm::errs() << constOp.getValue().getType() << "tof %% \n";
  handshake::CmpIOp cmpIOp = rewriter.create<handshake::CmpIOp>(successorOp->getLoc(), CmpIPredicate::eq, constOp.getResult(), controlMergeOp->getResult(1));
  inheritBB(successorOp, cmpIOp);

  return cmpIOp.getResult();
}


Value createWaitingSignalForPair(Value predecessorOpDoneSignal, SmallVector<Value> conds, Value CFGCond, Operation* predecessorOp, Operation* successorOp, Value startSignal, DenseMap<int, Operation*> &controlMerges, ConversionPatternRewriter& rewriter){
  handshake::SourceOp sourceOp = rewriter.create<handshake::SourceOp>(predecessorOp->getLoc());
  inheritBB(predecessorOp, sourceOp);

  ControlMergeOp controlMergeOp = dyn_cast<handshake::ControlMergeOp>(controlMerges[1]);
  handshake::ConstantOp constOp = rewriter.create<handshake::ConstantOp>(predecessorOp->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0), sourceOp);
  inheritBB(predecessorOp, constOp);
  handshake::CmpIOp cmpIOp = rewriter.create<handshake::CmpIOp>(predecessorOp->getLoc(), CmpIPredicate::eq, constOp.getResult(), controlMergeOp->getResult(1));
  inheritBB(predecessorOp, cmpIOp);

  handshake::ConditionalBranchOp conditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(predecessorOp->getLoc(), cmpIOp, sourceOp);
  inheritBB(predecessorOp, conditionalBranchOp);


  llvm::errs() << "mmmmmmm \n";
  SmallVector<Value> delayedDoneSignals = getNDelayedValues(predecessorOpDoneSignal, conditionalBranchOp.getResult(0), predecessorOp, rewriter);
  SmallVector<Value> branchedDoneSignals = insertBranches(delayedDoneSignals, conds, predecessorOp, rewriter);
  SmallVector<Value> conditionallySkippedDoneSignals = insertConditionalSkips(branchedDoneSignals, conds, successorOp, startSignal, rewriter);
  handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(predecessorOp->getLoc(), conditionallySkippedDoneSignals);
  inheritBB(predecessorOp, joinOp);
  // handshake::NotOp notOp = rewriter.create<handshake::NotOp>(predecessorOp->getLoc(), CFGCond);
  // inheritBB(predecessorOp, notOp);
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
    
    for (auto [predecessorOpName, predecessorOpPointer] : memAccesses){
      rewriter.setInsertionPointToStart(predecessorOpPointer->getBlock());
      Value predecessorOpDoneSignal = predecessorOpPointer->getResult(2);
      
      SmallVector<MemDependenceAttr> newDeps;
      if (auto deps = getDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer)){
        for (MemDependenceAttr dependency : deps.getDependencies()) {
          if (!dependency.getIsActive().getValue()){
            newDeps.push_back(dependency);
            continue;
          }

          StringRef successorName = dependency.getDstAccess();
          Operation* SuccessorOpPointer = memAccesses[successorName];

          SmallVector<Value> conds = skipConditionForEachPair[predecessorOpName][successorName];
          Value CFGCond = calculateCFGCond(predecessorOpPointer, SuccessorOpPointer, controlMerges, startSignal, rewriter);

          Value waitingSignal = createWaitingSignalForPair(predecessorOpDoneSignal, conds, CFGCond, predecessorOpPointer, SuccessorOpPointer, startSignal, controlMerges, rewriter);
          waitingSignalsForEachDst[successorName].push_back(waitingSignal);

          newDeps.push_back(getInactivatedDependency(dependency));
        }
        setDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer, ctx, newDeps);
        setDialectAttr<MemInterfaceAttr>(predecessorOpPointer, ctx);
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

void gateAllSuccessorAccesses(DenseMap<StringRef, Operation*> &memAccesses, DenseMap<StringRef, SmallVector<Value>> &waitingSignalsForEachDst, ConversionPatternRewriter& rewriter){
  for (auto [dstAccess, waitingSignals]: waitingSignalsForEachDst){
    Operation* op = memAccesses[dstAccess];
    gateAddress(op, waitingSignals, rewriter, op->getLoc());
  }
  llvm::errs() << "[] Gated Dst Accesses\n";
}

SmallVector<Value> createSkipConditionForPair(Value predecessorOpDoneSignal, Value startSignal, Operation* predecessorOpPointer, Operation* successorOpPointer, 
                    SmallVector<Value> delayedAddresses, DenseMap<int, Operation*> &controlMerges, 
                    ConversionPatternRewriter& rewriter){
  handshake::SourceOp sourceOp = rewriter.create<handshake::SourceOp>(predecessorOpPointer->getLoc());
  inheritBB(successorOpPointer, sourceOp);
  // handshake::ConstantOp constOp = rewriter.create<handshake::ConstantOp>(predecessorOpPointer->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0), sourceOp);
  // inheritBB(predecessorOpPointer, constOp);
  // handshake::UnbundleOp dummyUnbundleOp = rewriter.create<handshake::UnbundleOp>(predecessorOpPointer->getLoc(), constOp.getResult());
  // inheritBB(predecessorOpPointer, dummyUnbundleOp);

  // ValueRange *ab2 = new ValueRange();
  // handshake::ChannelType ch2 =  handshake::ChannelType::get(dummyUnbundleOp.getResult(1).getType());
  // handshake::BundleOp dummyBundleOp = rewriter.create<handshake::BundleOp>(predecessorOpPointer->getLoc(), dummyUnbundleOp.getResult(0), dummyUnbundleOp.getResult(1), *ab2, ch2);
  // inheritBB(predecessorOpPointer, dummyBundleOp);

  // rewriter.create<handshake::SinkOp>(predecessorOpPointer->getLoc(), dummyBundleOp.getResult(0));


  ControlMergeOp controlMergeOp = dyn_cast<handshake::ControlMergeOp>(controlMerges[1]);

  handshake::SourceOp sourceOp2 = rewriter.create<handshake::SourceOp>(predecessorOpPointer->getLoc());
  inheritBB(predecessorOpPointer, sourceOp2);
  handshake::ConstantOp constOp = rewriter.create<handshake::ConstantOp>(predecessorOpPointer->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0), sourceOp2);
  inheritBB(predecessorOpPointer, constOp);
  handshake::CmpIOp cmpIOp = rewriter.create<handshake::CmpIOp>(predecessorOpPointer->getLoc(), CmpIPredicate::eq, constOp.getResult(), controlMergeOp->getResult(1));
  inheritBB(predecessorOpPointer, cmpIOp);

  handshake::ConditionalBranchOp conditionalBranchOp2 = rewriter.create<handshake::ConditionalBranchOp>(predecessorOpPointer->getLoc(), cmpIOp, sourceOp);
  inheritBB(predecessorOpPointer, conditionalBranchOp2);




  SmallVector<Value> diffTokens = {predecessorOpDoneSignal};
  diffTokens.append(N, conditionalBranchOp2.getResult(0));
  handshake::MergeOp mergeOp = rewriter.create<handshake::MergeOp>(predecessorOpPointer->getLoc(), diffTokens);
  inheritBB(predecessorOpPointer, mergeOp);
  handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(predecessorOpPointer->getLoc(), mergeOp.getResult(), handshake::TimingInfo::tehb(), N);
  inheritBB(predecessorOpPointer, bufferOp);

  //here
  Value CFGCond = calculateCFGCond(predecessorOpPointer, successorOpPointer, controlMerges, startSignal, rewriter);

  handshake::SourceOp sourceOpConstOp = rewriter.create<handshake::SourceOp>(predecessorOpPointer->getLoc());
  inheritBB(successorOpPointer, sourceOpConstOp);

  handshake::ConstantOp falseConstOp = rewriter.create<handshake::ConstantOp>(predecessorOpPointer->getLoc(), rewriter.getBoolAttr(false), sourceOpConstOp);
  inheritBB(predecessorOpPointer, falseConstOp);

  handshake::ConditionalBranchOp conditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(successorOpPointer->getLoc(), falseConstOp.getResult(), bufferOp.getResult());
  inheritBB(successorOpPointer, conditionalBranchOp);


  handshake::ConditionalBranchOp SuccessorAddrConditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(successorOpPointer->getLoc(), CFGCond, successorOpPointer->getOperand(0));
  inheritBB(successorOpPointer, SuccessorAddrConditionalBranchOp);
  // Not sure which one (0 or 1)
  handshake::UnbundleOp unbundleOp = rewriter.create<handshake::UnbundleOp>(predecessorOpPointer->getLoc(), SuccessorAddrConditionalBranchOp.getResult(1));
  inheritBB(predecessorOpPointer, unbundleOp);


  SmallVector<Value, 2> JoinOpValues = {conditionalBranchOp.getResult(1), unbundleOp.getResult(0)};
  handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(predecessorOpPointer->getLoc(), JoinOpValues);
  inheritBB(predecessorOpPointer, joinOp);

  ValueRange *ab = new ValueRange();
  handshake::ChannelType ch =  handshake::ChannelType::get(unbundleOp.getResult(1).getType());
  handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(predecessorOpPointer->getLoc(), joinOp.getResult(), unbundleOp.getResult(1), *ab, ch);
  inheritBB(predecessorOpPointer, bundleOp);

  Value gatedSuccessorOpaddr = bundleOp.getResult(0);

  //lazy fork maybe
  SmallVector<Value> skipConditions;
  for (Value delayedAddress : delayedAddresses){
      handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(predecessorOpPointer->getLoc(), delayedAddress, handshake::TimingInfo::tehb(), 5);
      inheritBB(predecessorOpPointer, bufferOp);
      handshake::CmpIOp cmpIOp = rewriter.create<handshake::CmpIOp>(predecessorOpPointer->getLoc(), CmpIPredicate::ne, gatedSuccessorOpaddr, bufferOp.getResult());
      inheritBB(predecessorOpPointer, cmpIOp);
      skipConditions.push_back(cmpIOp.getResult());
  }          
  return skipConditions;
}

void createSkipConditionGenerator(FuncOp funcOp, DenseMap<StringRef, Operation*> &memAccesses, DenseMap<StringRef, DenseMap<StringRef, SmallVector<Value>>> &skipConditionForEachPair, DenseMap<int, Operation*> &controlMerges, ConversionPatternRewriter &rewriter){
    Value startSignal = (Value)funcOp.getArguments().back();

    for (auto [predecessorOpName, predecessorOpPointer] : memAccesses){
      rewriter.setInsertionPointToStart(predecessorOpPointer->getBlock());
      Value predecessorOpDoneSignal = predecessorOpPointer->getResult(2);
      Value predecessorOpAddr = predecessorOpPointer->getOperand(0);

      // The dummy address
      handshake::SourceOp sourceOp = rewriter.create<handshake::SourceOp>(predecessorOpPointer->getLoc());
      inheritBB(predecessorOpPointer, sourceOp);


      ControlMergeOp controlMergeOp = dyn_cast<handshake::ControlMergeOp>(controlMerges[1]);
      handshake::ConstantOp constOp = rewriter.create<handshake::ConstantOp>(predecessorOpPointer->getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 0), sourceOp);
      inheritBB(predecessorOpPointer, constOp);
      handshake::CmpIOp cmpIOp = rewriter.create<handshake::CmpIOp>(predecessorOpPointer->getLoc(), CmpIPredicate::eq, constOp.getResult(), controlMergeOp->getResult(1));
      inheritBB(predecessorOpPointer, cmpIOp);

      handshake::ConditionalBranchOp conditionalBranchOp = rewriter.create<handshake::ConditionalBranchOp>(predecessorOpPointer->getLoc(), cmpIOp, sourceOp);
      inheritBB(predecessorOpPointer, conditionalBranchOp);
    
      handshake::ConstantOp dummyConstOp = rewriter.create<handshake::ConstantOp>(predecessorOpPointer->getLoc(), rewriter.getIntegerAttr(rewriter.getI32Type(), 1000), conditionalBranchOp.getResult(0));
      inheritBB(predecessorOpPointer, dummyConstOp);

      

      if (auto deps = getDialectAttr<MemDependenceArrayAttr>(predecessorOpPointer)){
        
        for (MemDependenceAttr dependency : deps.getDependencies()) {
          if (!dependency.getIsActive().getValue())
            continue;
          /// This needs improvement!
          // rouzbeh
          // should be done once
          


          SmallVector<Value> delayedAddresses = getNDelayedValues(predecessorOpAddr, dummyConstOp, predecessorOpPointer, rewriter);

          
          handshake::BufferOp bufferOp = rewriter.create<handshake::BufferOp>(predecessorOpPointer->getLoc(), predecessorOpAddr, handshake::TimingInfo::tehb(), 20);
          inheritBB(predecessorOpPointer, bufferOp);
          predecessorOpPointer->setOperand(0, bufferOp.getResult());


          StringRef dstAccess = dependency.getDstAccess();
          Operation* successorOpPointer = memAccesses[dstAccess];

          
          SmallVector<Value> skipConditions = createSkipConditionForPair(predecessorOpDoneSignal, startSignal, predecessorOpPointer, successorOpPointer,delayedAddresses,controlMerges, rewriter);
          skipConditionForEachPair[predecessorOpName][dstAccess] = skipConditions;
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

      createSkipConditionGenerator(funcOp, memAccesses, skipConditionForEachPair, controlMerges, rewriter);
      createWaitingSignals(funcOp, memAccesses, waitingSignalsForEachDst, skipConditionForEachPair, controlMerges, ctx, rewriter);
      gateAllSuccessorAccesses(memAccesses, waitingSignalsForEachDst, rewriter);   

  }


  llvm::errs() << "done! \n";
  
}



std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeInsertSkippableSeq() {
  return std::make_unique<HandshakeInsertSkippableSeqPass>();
}

