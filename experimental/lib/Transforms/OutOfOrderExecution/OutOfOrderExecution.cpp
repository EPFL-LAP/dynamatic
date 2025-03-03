//===- FtdCfToHandshake.cpp - FTD conversion cf -> handshake --*--- C++ -*-===//
//
// Implements the out-of-order execution methodology
// https://dl.acm.org/doi/10.1145/3626202.3637556
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/OutOfOrderExecution/OutOfOrderExecution.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental;

namespace {

static LogicalResult applyOutOfOrderExecution(handshake::FuncOp funcOp,
                                              MLIRContext *ctx) {
  return success();
}

struct OutOfOrderExecutionPass
    : public dynamatic::experimental::impl::OutOfOrderExecutionBase<
          OutOfOrderExecutionPass> {

  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();
    mlir::ModuleOp module = getOperation();

    for (auto funcOp : module.getOps<handshake::FuncOp>())
      if (failed(applyOutOfOrderExecution(funcOp, ctx)))
        signalPassFailure();
  };
};
}; // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::createOutOfOrderExecution() {
  return std::make_unique<OutOfOrderExecutionPass>();
}