//===- HandshakeAddSeqMem.cpp - LSQ flow analysis ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-add-seq-mem pass,
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/HandshakeAddSeqMem.h"
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
#include "mlir/Transforms/DialectConversion.h"
#include "experimental/Support/FtdSupport.h"
#include "experimental/Support/CFGAnnotation.h"


#define DEBUG_TYPE "handshake-add-seq-mem"


using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;




namespace {
/// Simple pass driver for the Seq Mem pass. This pass adds the forks and joins needed
/// to sequentialize the memory access.
struct HandshakeAddSeqMemPass
    : public dynamatic::experimental::impl::HandshakeAddSeqMemBase<
          HandshakeAddSeqMemPass> {


  void runDynamaticPass() override;

  void traverseMemRef(ModuleOp modOp, DenseMap<StringRef, Operation *> &forkOpDict, void (*func)(Operation*, DenseMap<StringRef, Operation *>&, ConversionPatternRewriter&, NameAnalysis&), ConversionPatternRewriter &rewriter);

};

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


void HandshakeAddSeqMemPass::traverseMemRef(ModuleOp modOp, DenseMap<StringRef, Operation *> &forkOpDict, void (*func)(Operation*, DenseMap<StringRef, Operation *>&, ConversionPatternRewriter&, NameAnalysis&),ConversionPatternRewriter &rewriter){
   for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()){
      for (BlockArgument arg : funcOp.getArguments()) {
        llvm::errs() << "[traversing arguments]" << arg << "\n";
        if (auto memref = dyn_cast<TypedValue<mlir::MemRefType>>(arg)){
            // MLIRContext *ctx = &getContext();
            // OpBuilder builder(ctx);
            NameAnalysis &namer = getAnalysis<NameAnalysis>();

            auto memrefUsers = memref.getUsers();
            Operation *memOp = *memrefUsers.begin();
            handshake::LSQOp lsqOp = dyn_cast<handshake::LSQOp>(memOp);
            
            llvm::errs() << "lsq op \n";
            if (lsqOp) {
                LSQPorts lsqPorts = lsqOp.getPorts();
                llvm::errs() << "lsqPorts" << "--\n";
                for (LSQGroup &group : lsqPorts.getGroups()) {
                  llvm::errs() << "group" << "--\n";
                  for (MemoryPort &port : group->accessPorts) {
                      llvm::errs() << "oomad \n";
                      func (port.portOp, forkOpDict, rewriter, namer);
                    }
                }
            }
            llvm::errs() << "not lsq op \n";
        }
    }

  }
}
    


void insertForkForOp(Operation* op, DenseMap<StringRef, Operation *> &forkOpDict, ConversionPatternRewriter& rewriter, NameAnalysis& namer){

    llvm::errs() << "heyyyy\n";
    if (LoadOp memOp = dyn_cast<LoadOp>(op); memOp){

    rewriter.setInsertionPointToStart(memOp->getBlock());



    handshake::UnbundleOp unbundleOp = rewriter.create<handshake::UnbundleOp>(memOp.getLoc(), memOp.getResult(1));
    unbundleOp->setAttr(BB_ATTR_NAME, rewriter.getUI32IntegerAttr(getBBNumberFromOp(memOp)));


    handshake::ForkOp forkOp = rewriter.create<handshake::ForkOp>(memOp.getLoc(), unbundleOp->getResult(0), 13);
    forkOp->setAttr(BB_ATTR_NAME, rewriter.getUI32IntegerAttr(getBBNumberFromOp(memOp)));



    ValueRange *ab = new ValueRange();
    handshake::ChannelType ch =  handshake::ChannelType::get(unbundleOp.getResult(1).getType());
    handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(memOp.getLoc(), forkOp->getResult(0), unbundleOp->getResult(1), *ab, ch);
    bundleOp->setAttr(BB_ATTR_NAME, rewriter.getUI32IntegerAttr(getBBNumberFromOp(memOp)));

    
    rewriter.create<handshake::SinkOp>(memOp.getLoc(), bundleOp->getResult(0));


    forkOpDict[getUniqueName(memOp)] = forkOp;
    llvm::errs() << "[insert fork] for " << getUniqueName(memOp) << "\n inserted fork " << forkOp << "\n now the size of dictionary is " << forkOpDict.size() <<  "\n";

    }
    
}



void insertJoinForOp(Operation* op, DenseMap<StringRef, Operation *> &forkOpDict, ConversionPatternRewriter &rewriter, NameAnalysis& namer){

    SmallVector<Value> joinValues;

    if (LoadOp memOp = dyn_cast<LoadOp>(op); memOp){
      llvm::errs() << "[insert join] reached the load operation: " << getUniqueName(memOp) << "\n";

    if (auto deps = getDialectAttr<MemDependenceArrayAttr>(memOp)) {
      
      for (MemDependenceAttr dependency : deps.getDependencies()) {
          auto dstAccess = dependency.getDstAccess();

          llvm::errs() << "the name of dest access is " << dstAccess << "\n";
          // Operation *dstOp = namer.getOp(dstAccess);


          if (Operation* op = forkOpDict[dstAccess]; op){
            llvm::errs() << "[fork Op dict] found" <<  op << "\n";
            llvm::errs() << "[fork Op dict] found" <<  *op << "\n";

          llvm::errs() << "fork Op " << op->getResult(0) << "\n";
          joinValues.push_back(op->getResult(0));


          } else {
            llvm::errs() << "[fork Op dict] didn't found anything \n";
          }

        }
        
      } 
    }else if (StoreOp memOp = dyn_cast<StoreOp>(op); memOp){
      llvm::errs() << "[insert join] reached the store operation: " << getUniqueName(memOp) << "\n";

      if (auto deps = getDialectAttr<MemDependenceArrayAttr>(memOp)) {
        for (MemDependenceAttr dependency : deps.getDependencies()) {
            auto dstAccess = dependency.getDstAccess();
            
            llvm::errs() << "[insert join] the name of dest access is " << dstAccess << "\n";


            if (Operation* op = forkOpDict[dstAccess]; op){
              llvm::errs() << "[fork Op dict] found" <<  op << "\n";
              llvm::errs() << "[fork Op dict] found" <<  *op << "\n";

            llvm::errs() << "fork Op " << op->getResult(0) << "\n";
            joinValues.push_back(op->getResult(0));


            } else {
              llvm::errs() << "[fork Op dict] didn't found anything \n";
            }
          }
          
      }

      
      llvm::errs() << "The join values are: ";
      for (auto & joinValue : joinValues){
          llvm::errs() << joinValue << "-----\n";
      }
      llvm::errs() << "-------------------------------\n";



      rewriter.setInsertionPointToStart(memOp->getBlock());

      int bb_num = getBBNumberFromOp(memOp);

      handshake::UnbundleOp unbundleOp = rewriter.create<handshake::UnbundleOp>(memOp.getLoc(), memOp.getOperand(0));
      unbundleOp->setAttr(BB_ATTR_NAME, rewriter.getUI32IntegerAttr(bb_num));

      joinValues.push_back(unbundleOp.getResult(0));

      handshake::JoinOp joinOp = rewriter.create<handshake::JoinOp>(memOp.getLoc(), joinValues);
      joinOp->setAttr(BB_ATTR_NAME, rewriter.getUI32IntegerAttr(bb_num));


      ValueRange *ab = new ValueRange();
      handshake::ChannelType ch = handshake::ChannelType::get(unbundleOp.getResult(1).getType());

      handshake::BundleOp bundleOp = rewriter.create<handshake::BundleOp>(memOp.getLoc(), joinOp.getResult(), unbundleOp.getResult(1), *ab, ch);
      bundleOp->setAttr(BB_ATTR_NAME, rewriter.getUI32IntegerAttr(bb_num));

      llvm::errs() << "fetne2" << bundleOp <<  "\n";
      llvm::errs() << joinOp.getResult() << unbundleOp.getResult(1) <<  "\n";
       

      memOp.setOperand(0, bundleOp.getResult(0));

      llvm::errs() << "[insert join] created the join operation\n" ;

    }


}


void HandshakeAddSeqMemPass::runDynamaticPass() {
  mlir::ModuleOp modOp = getOperation();

  // Check that memory access ports are named
  NameAnalysis &namer = getAnalysis<NameAnalysis>();
  WalkResult res = modOp.walk([&](Operation *op) {
    if (!isa<handshake::LoadOp, handshake::StoreOp>(op))
      return WalkResult::advance();
    if (!namer.hasName(op)) {
      op->emitError() << "Memory access port must be named.";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return signalPassFailure();

  // Check that all eligible operations within Handshake function belon to a
  // basic block
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : funcOp.getOps()) {
      if (!cannotBelongToCFG(&op) && !getLogicBB(&op)) {
        op.emitError() << "Operation should have basic block attribute.";
        return signalPassFailure();
      }
    }
  }

      MLIRContext *ctx = &getContext();
  
  ConversionPatternRewriter rewriter(ctx);




  
  llvm::errs() << "Finished Traversing \n";
  for (auto funcOp : modOp.getOps<handshake::FuncOp>()){
      // Restore the cf structure to work on a structured IR
      if (failed(experimental::cfg::restoreCfStructure(funcOp, rewriter)))
        signalPassFailure();
  }

  DenseMap<StringRef, Operation *> forkOpDict;

  traverseMemRef(modOp, forkOpDict, &insertForkForOp, rewriter);
  traverseMemRef(modOp, forkOpDict, &insertJoinForOp, rewriter);

for (auto funcOp : modOp.getOps<handshake::FuncOp>()){
      funcOp.print(llvm::dbgs());
      llvm::errs() << "funcOp";

      experimental::ftd::addRegen(funcOp, rewriter);
      experimental::ftd::addSupp(funcOp, rewriter);
      experimental::cfg::markBasicBlocks(funcOp, rewriter);

      // Remove the blocks and terminators
      if (failed(cfg::flattenFunction(funcOp)))
        signalPassFailure();
  }
  


  
}
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::createHandshakeAddSeqMem() {
  return std::make_unique<HandshakeAddSeqMemPass>();
}
