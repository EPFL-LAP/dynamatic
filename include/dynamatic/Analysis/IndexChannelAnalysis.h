#pragma once
#include "mlir/IR/Value.h"

namespace dynamatic {
class IndexChannelAnalysis {
public:
  IndexChannelAnalysis(mlir::Operation *modOp);
  std::optional<size_t> getIndexChannelValues(mlir::Value channel) const;

  // A map from channel to the number of possible index values.
  // For example, if the channel is used as the select input of a mux with 3
  // data inputs, the corresponding value within the map will be 3. The indices
  // are always assumed to range from (0 .. numValues - 1), so all possible
  // values within this channel would be {0, 1, 2} in the example.
  // Note that this is slightly more information than simply the bit width of a
  // channel: In the example, the bit width of the channel is 2 bits, but one of
  // the 4 resulting options is invalid.
  llvm::DenseMap<mlir::Value, size_t> map;

private:
  void annotateIndexChannels(mlir::Value index, size_t numValues);
};
} // namespace dynamatic
