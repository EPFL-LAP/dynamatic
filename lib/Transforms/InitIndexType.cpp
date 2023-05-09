//===- InitIndexType.cpp - Change IndexType to IntegerType ------*- C++ -*-===//
//
// This file contains the implementation of the init-indextype optimization
// pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/InitIndexType.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

unsigned indexWidth = IndexType::kInternalStorageBitWidth;

// Adapt the Index type to the Integer type
static LogicalResult initIndexType(handshake::FuncOp funcOp, MLIRContext *ctx) {
  OpBuilder builder(ctx);
  SmallVector<Operation *> indexCastOps;

  for (auto &op : funcOp.getOps()) {
    // insert trunc|extsi operation for index_cast operation
    if (isa<mlir::arith::IndexCastOp>(op)) {
      indexCastOps.push_back(&op);
      auto indexCastOp = dyn_cast<mlir::arith::IndexCastOp>(op);
      bool isOpIndType = isa<IndexType>(indexCastOp.getOperand().getType());
      bool isResIndType = isa<IndexType>(indexCastOp.getResult().getType());

      // if cast index to integer type
      if (!isResIndType) {
        if (indexCastOp.getResult().getType().getIntOrFloatBitWidth() ==
            indexCastOp.getOperand().getType().getIntOrFloatBitWidth())
          indexCastOp.getResult().replaceAllUsesWith(indexCastOp.getOperand());
        else {
          auto newOp =
              insertWidthMatchOp(&op, 0, op.getResult(0).getType(), ctx);
          if (newOp.has_value())
            op.replaceAllUsesWith(newOp.value());
        }
      }

      // if cast integer to index type
      if (isResIndType && !isOpIndType) {
        if (indexCastOp.getOperand().getType().getIntOrFloatBitWidth() ==
            indexWidth)
          indexCastOp.getResult().replaceAllUsesWith(indexCastOp.getOperand());
        else {
          builder.setInsertionPoint(&op);
          auto extOp = builder.create<mlir::arith::ExtSIOp>(
              op.getLoc(), IntegerType::get(ctx, indexWidth),
              indexCastOp.getOperand());
          op.replaceAllUsesWith(extOp);
        }
      }
    }

    // set type for other operations
    else {
      for (unsigned int i = 0; i < op.getNumResults(); ++i)
        if (OpResult result = op.getResult(i);
            isa<IndexType>(result.getType())) {
          result.setType(IntegerType::get(ctx, indexWidth));

          // For constant operation, change the value attribute to match the new
          // type
          if (isa<handshake::ConstantOp>(op)) {
            handshake::ConstantOp cstOp = dyn_cast<handshake::ConstantOp>(op);
            cstOp.setValueAttr(IntegerAttr::get(
                IntegerType::get(ctx, indexWidth),
                cstOp.getValue().cast<IntegerAttr>().getInt()));
          }
        }
    }
  }

  for (auto op : indexCastOps)
    op->erase();

  return success();
}

namespace {

struct HandshakeInitIndTypePass
    : public HandshakeInitIndTypeBase<HandshakeInitIndTypePass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    ModuleOp m = getOperation();
    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if (failed(initIndexType(funcOp, ctx)))
        return signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createInitIndTypePass() {
  return std::make_unique<HandshakeInitIndTypePass>();
}