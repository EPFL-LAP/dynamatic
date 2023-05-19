//===- BitsOptimize.cpp -  Optimize bits width ------------------*- C++ -*-===//
//
// This file contains the implementation of the bits optimization pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/OptimizeBits.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "dynamatic/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

static LogicalResult rewriteBitsWidths(handshake::FuncOp funcOp,
                                       MLIRContext *ctx) {
  OpBuilder builder(ctx);
  SmallVector<Operation *> vecOp;

  using forward_func =
      std::function<unsigned(mlir::Operation::operand_range vecOperands)>;
  using backward_func =
      std::function<unsigned(mlir::Operation::result_range vecResults)>;

  SmallVector<Operation *> containerOps;

  bool changed = true;
  int savedBits = 0;

  // Construct the functions w.r.t. the operation name for forward and backward
  DenseMap<StringRef, forward_func> forMapOpNameWidth;
  bitwidth::constructForwardFuncMap(forMapOpNameWidth);

  DenseMap<StringRef, backward_func> backMapOpNameWidth;
  bitwidth::constructBackwardFuncMap(backMapOpNameWidth);

  while (changed) {
    // init
    changed = false;
    containerOps.clear();

    for (auto &op : funcOp.getOps())
      containerOps.push_back(&op);

    // Forward process
    for (auto &op : containerOps) {

      if (isa<handshake::ConstantOp>(*op) ||
          bitwidth::propType(op))
        continue;

      if (isa<mlir::arith::ExtSIOp>(op) || isa<mlir::arith::ExtUIOp>(op)) {
        bitwidth::replaceWithPredecessor(op);
        // op->erase();
        continue;
      }

      const auto opName = op->getName().getStringRef();
      unsigned int newWidth = 0, resInd = 0;
      if (forMapOpNameWidth.find(opName) != forMapOpNameWidth.end())
        newWidth = forMapOpNameWidth[opName](op->getOperands());
      else {continue;}

      if (isa<handshake::ControlMergeOp>(op))
        resInd = 1; // the second result is the one that needs to be updated

      // if the new type can be optimized, update the type
      if (Type newOpResultType =
              getNewType(op->getResult(resInd), newWidth, true);
          newWidth <
          op->getResult(resInd).getType().getIntOrFloatBitWidth()) {
        changed = true;
        savedBits += op->getResult(resInd).getType().getIntOrFloatBitWidth() -
                      newWidth;
        op->getResult(resInd).setType(newOpResultType);
      }
    }

    // Backward Process
    for (auto op : llvm::reverse(containerOps)) {
      if (isa<handshake::ConstantOp>(*op))
        continue;

      if (isa<mlir::arith::TruncIOp>(*op)) {
        bitwidth::replaceWithPredecessor(op, op->getResult(0).getType());
        // op->erase();
        continue;
      }

      const auto opName = op->getName().getStringRef();

      unsigned int newWidth = 0;
      // get the new bit width of the result operator
      if (backMapOpNameWidth.find(opName) != backMapOpNameWidth.end())
        newWidth = backMapOpNameWidth[opName](op->getResults());
      else {continue;}

      // if the new type can be optimized, update the type
      if (Type newOpResultType =
              getNewType(op->getOperand(0), newWidth, true);
          newWidth < op->getOperand(0).getType().getIntOrFloatBitWidth()) {
        changed = true;
        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
          unsigned int origWidth = 
            op->getOperand(i).getType().getIntOrFloatBitWidth();
          if (newWidth < origWidth) {
            savedBits += origWidth - newWidth;
            op->getOperand(i).setType(newOpResultType);
          }
        }
      }
    }
  }

  // Store new inserted truncation or extension operation during validation
  SmallVector<Operation *> OpTruncExt;
  for (auto &op : llvm::make_early_inc_range(funcOp.getOps()))
    bitwidth::validateOp(&op, ctx, OpTruncExt);
  
  // Validate the new inserted operation
  for (auto op : OpTruncExt) 
    bitwidth::revertTruncOrExt(op, ctx);

  // LLVM_DEBUG(llvm::errs() << "Forward-Backward saved bits " << savedBits << "\n");

  return success();
}

struct HandshakeOptimizeBitsPass
    : public HandshakeOptimizeBitsBase<HandshakeOptimizeBitsPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    ModuleOp m = getOperation();

    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if (failed(rewriteBitsWidths(funcOp, ctx)))
        return signalPassFailure();
  };
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createOptimizeBitsPass() {
  return std::make_unique<HandshakeOptimizeBitsPass>();
}