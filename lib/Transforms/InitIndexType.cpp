//===- InitIndexType.cpp -  Transform the Index Type to IntegerType with system bit width -------*- C++ -*-===//
//
// This file contains the implementation of the init-indextype optimization pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/InitIndexType.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Dialect.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/IndentedOstream.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

unsigned indexWidth = IndexType::kInternalStorageBitWidth;

// Adapt the Index type to the Integer type
static LogicalResult initIndexType(handshake::FuncOp funcOp, MLIRContext *ctx){
  OpBuilder builder(ctx);

  for (Operation &op : funcOp.getOps()){
    if (isa<handshake::ControlMergeOp>(op)) 
      continue;
      
    for (int i=0; i<op.getNumResults(); ++i)
      if (OpResult result = op.getResult(i); isa<IndexType>(result.getType())){
        result.setType(IntegerType::get(ctx, indexWidth));

        // For constant operation, change the value attribute to match the new type
        if (isa<handshake::ConstantOp>(op)){
          handshake::ConstantOp cstOp = dyn_cast<handshake::ConstantOp>(op);
          cstOp.setValueAttr(IntegerAttr::get(IntegerType::get(ctx, indexWidth), 
                      cstOp.getValue().cast<IntegerAttr>().getInt()));
        }
      }
  }
  return success();
}

namespace{

struct HandshakeInitIndexTypePass
    : public HandshakeInitIndexTypeBase<HandshakeInitIndexTypePass> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    ModuleOp m = getOperation();

    for (auto funcOp : m.getOps<handshake::FuncOp>())
      if(failed(initIndexType(funcOp, ctx)))
        return signalPassFailure();
  };

};
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createInitIndexTypePass() {
  return std::make_unique<HandshakeInitIndexTypePass>();
}