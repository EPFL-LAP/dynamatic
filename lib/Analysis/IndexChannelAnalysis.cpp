#include "dynamatic/Analysis/IndexChannelAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace dynamatic;
using namespace dynamatic::handshake;

namespace dynamatic {
void IndexChannelAnalysis::annotateIndexChannels(mlir::Value index,
                                                 size_t numValues) {
  map.insert({index, numValues});

  Operation *op = index.getDefiningOp();
  if (!op)
    return;

  // Arithmetic ops can turn non-index into index, so stop following
  if (isa<ArithOpInterface>(op))
    return;

  if (auto bufOp = dyn_cast<BufferOp>(op)) {
    annotateIndexChannels(bufOp.getOperand(), numValues);
    return;
  }

  if (auto forkOp = dyn_cast<ForkOp>(op)) {
    annotateIndexChannels(forkOp.getOperand(), numValues);
    return;
  }

  if (auto muxOp = dyn_cast<MuxOp>(op)) {
    for (auto prevOp : muxOp.getDataOperands()) {
      annotateIndexChannels(prevOp, numValues);
    }
    return;
  }

  if (auto mergeOp = dyn_cast<MergeOp>(op)) {
    for (auto prevOp : mergeOp.getOperands()) {
      annotateIndexChannels(prevOp, numValues);
    }
    return;
  }
}
IndexChannelAnalysis::IndexChannelAnalysis(mlir::Operation *modOpPtr) {
  if (modOpPtr == nullptr)
    return;
  ModuleOp modOp = dyn_cast<ModuleOp>(modOpPtr);
  if (!modOp)
    return;
  map = llvm::DenseMap<mlir::Value, size_t>();

  for (auto funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (auto &op : funcOp.getOps()) {
      if (auto muxOp = dyn_cast<MuxOp>(op)) {
        mlir::Value index = muxOp.getSelectOperand();
        size_t numValues = muxOp.getDataOperands().size();
        annotateIndexChannels(index, numValues);
      } else if (auto branchOp = dyn_cast<ConditionalBranchOp>(op)) {
        mlir::Value index = branchOp.getConditionOperand();
        size_t numValues = 2;
        annotateIndexChannels(index, numValues);
      }
    }
  }
}

std::optional<size_t>
IndexChannelAnalysis::getIndexChannelValues(mlir::Value channel) const {
  for (auto &[key, value] : map) {
    if (key == channel)
      return value;
  }
  std::optional<size_t> none;
  return none;
}
} // namespace dynamatic
