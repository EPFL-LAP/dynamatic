//===- FtdCfToHandshake.cpp - FTD conversion cf -> handshake --*--- C++ -*-===//
//
// Implements the out-of-order execution methodology
// https://dl.acm.org/doi/10.1145/3626202.3637556
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/OutOfOrderExecution/OutOfOrderExecution.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cmath>
#include <memory>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::outoforder;

int numTags = 4;
int numTagsOuter = 8;

namespace {
struct OutOfOrderExecutionPass
    : public dynamatic::experimental::outoforder::impl::OutOfOrderExecutionBase<
          OutOfOrderExecutionPass> {

  void runDynamaticPass() override;

private:
  // Step 1: Add the FIFO, Taggerand Untagger Operations
  LogicalResult createOutOfExecutionGraph(handshake::FuncOp funcOp,
                                          MLIRContext *ctx);

  // Step 2: Add the tag signals to the channels in the tagged region
  LogicalResult addTagSignals(handshake::FuncOp funcOp, MLIRContext *ctx);
};
} // namespace

LogicalResult
OutOfOrderExecutionPass::createOutOfExecutionGraph(handshake::FuncOp funcOp,
                                                   MLIRContext *ctx) {

  OpBuilder builder(ctx);
  for (auto loadOp : funcOp.getOps<handshake::LoadOp>()) {
    auto startValue = (Value)funcOp.getArguments().back();
    Value addrInput = loadOp.getAddressInput();
    builder.setInsertionPoint(loadOp);

    auto tagType = builder.getIntegerType(ceil(log2(numTags)));

    FreeTagsFifoOp fifo = builder.create<handshake::FreeTagsFifoOp>(
        loadOp.getLoc(), handshake::ChannelType::get(tagType), startValue);

    // Tag the address input of the load of
    handshake::TaggerOp taggerOp = builder.create<handshake::TaggerOp>(
        loadOp.getLoc(), addrInput.getType(), addrInput, fifo.getTagOut());

    // Connect the tagger to the load
    loadOp.getOperation()->replaceUsesOfWith(addrInput,
                                             taggerOp.getDataOut());

    // Create the untagegr and connect it to teh load
    UntaggerOp untaggerOp = builder.create<handshake::UntaggerOp>(
        loadOp.getLoc(), loadOp.getDataResult().getType(),
        fifo.getTagOut().getType(), loadOp.getDataResult());

    // Replaces all the connections load->consumer to untagger->consumer
    Value loadOutput = loadOp.getDataOutput();
    for (Operation *user : loadOutput.getUsers()) {
      if (!isa<handshake::UntaggerOp>(user))
        user->replaceUsesOfWith(loadOutput, untaggerOp.getDataOut());
    }

    // Connet the free tag from the untagger to the fifo
    fifo.getOperation()->replaceUsesOfWith(startValue, untaggerOp.getTagOut());
  }
  return success();
}

const std::string EXTRA_TAG = "tag0";

static LogicalResult addTagToValue(Value value) {
  OpBuilder builder(value.getContext());

  // The value type must implement ExtraSignalsTypeInterface (e.g.,
  // ChannelType or ControlType).
  if (auto valueType =
          value.getType().dyn_cast<handshake::ExtraSignalsTypeInterface>()) {
    // Skip if the spec tag was already added during the algorithm.
    if (!valueType.hasExtraSignal(EXTRA_TAG)) {
      llvm::SmallVector<ExtraSignal> newExtraSignals(
          valueType.getExtraSignals());
      newExtraSignals.emplace_back(EXTRA_TAG,
                                   builder.getIntegerType(ceil(log2(numTags))));
      value.setType(valueType.copyWithExtraSignals(newExtraSignals));
    }
    return success();
  }
  value.getDefiningOp()->emitError("Unexpected type");
  return failure();
}

static LogicalResult
addTagSignalsRecursive(MLIRContext &ctx, OpOperand &opOperand,
                       bool isDownstream,
                       llvm::DenseSet<Operation *> &visited) {
  // Add the tag to the current operand
  if (failed(addTagToValue(opOperand.get())))
    return failure();

  Operation *op;

  // Traversal may be either upstream or downstream
  if (isDownstream) {
    // Owner is the consumer of the operand
    op = opOperand.getOwner();
  } else {
    // DefiningOp is the producer of the operand
    op = opOperand.get().getDefiningOp();
  }

  if (!op)
    // As long as the algorithm traverses inside the tagged region,
    // all operands should have an owner and defining operation.
    return failure();

  if (visited.contains(op))
    return success();
  visited.insert(op);

  // Exceptional cases
  // UntaggerOp
  if (isa<handshake::UntaggerOp>(op)) {
    if (isDownstream) {
      // Stop the traversal at the untagger
      return success();
    }

    // The upstream stream shouldn't reach the untagger unit,
    // as that would indicate it originated outside the tagged region.
    op->emitError("UntaggerOp should not be reached from "
                  "outside the tagged region");
    return failure();
  }
  // MemPortOp (Load and Store)
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    if (isDownstream) {
      // Continue traversal to dataOut, skipping ports connected to the memory
      // controller.
      for (auto &operand : loadOp->getOpResult(1).getUses()) {
        if (failed(addTagSignalsRecursive(ctx, operand, true, visited)))
          return failure();
      }
    } else {
      // Continue traversal to addrIn, skipping ports connected to the memory
      // controller.
      auto &operand = loadOp->getOpOperand(0);
      if (failed(addTagSignalsRecursive(ctx, operand, false, visited)))
        return failure();
    }

    return success();
  }

  // General case

  // Upstream traversal
  /*
  for (auto &operand : op->getOpOperands()) {
    // Skip the operand that is the same as the current operand
    if (isDownstream && &operand == &opOperand)
      continue;
    if (failed(addTagSignalsRecursive(ctx, operand, false, visited))) {
      llvm::errs() << "Failed upstream\n";
      return failure();
    }
  }*/

  // Downstream traversal
  for (auto result : op->getResults()) {
    for (auto &operand : result.getUses()) {
      // Skip the operand that is the same as the current operand
      if (!isDownstream && &operand == &opOperand)
        continue;
      if (failed(addTagSignalsRecursive(ctx, operand, true, visited)))
        return failure();
    }
  }

  return success();
}

LogicalResult OutOfOrderExecutionPass::addTagSignals(handshake::FuncOp funcOp,
                                                     MLIRContext *ctx) {
  // The TaggerOp marks the ebginning of teh tagged region, so we use it as a
  // starting point for tagging
  for (auto taggerOp : funcOp.getOps<handshake::TaggerOp>()) {
    llvm::DenseSet<Operation *> visited;
    visited.insert(taggerOp);

    // For the speculator, perform downstream traversal to only dataOut,
    // skipping control signals. The upstream dataIn will be handled by the
    // recursive traversal.
 

      Value taggerResult = taggerOp.getDataOut();
      for (OpOperand &opOperand : taggerResult.getUses()) {
        if (failed(addTagSignalsRecursive(*ctx, opOperand, true, visited)))
          return failure();
      }
    
  }

  return success();
}

void OutOfOrderExecutionPass::runDynamaticPass() {
  MLIRContext *ctx = &getContext();
  mlir::ModuleOp module = getOperation();

  for (auto funcOp : module.getOps<handshake::FuncOp>()) {

    if (failed(createOutOfExecutionGraph(funcOp, ctx)))
      signalPassFailure();

    if (failed(addTagSignals(funcOp, ctx)))
      signalPassFailure();
  }
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::outoforder::createOutOfOrderExecution() {
  return std::make_unique<OutOfOrderExecutionPass>();
}