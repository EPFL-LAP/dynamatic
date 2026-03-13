#pragma once
#include "mlir/IR/Value.h"

namespace dynamatic {
class IndexChannelAnalysis {
public:
  IndexChannelAnalysis(mlir::Operation *modOp);
  std::optional<size_t> getIndexChannelValues(mlir::Value channel) const;
  llvm::DenseMap<mlir::Value, size_t> map;

private:
  void annotateIndexChannels(mlir::Value index, size_t numValues);
};
} // namespace dynamatic
