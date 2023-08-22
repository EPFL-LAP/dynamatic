//===- OptimizeMILP.h - optimize MILP model over CFDFC  ---------*- C++ -*-===//
//
// This file declares functions of MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H

#include "dynamatic/Support/BufferingStrategy.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/ExtractCFDFC.h"
#include "dynamatic/Transforms/BufferPlacement/ParseCircuitJson.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace dynamatic {
namespace buffer {

using namespace circt;
using namespace handshake;

/// Data structure to store the results of buffer placement, including the
/// property and the total slots of the channel
struct Result {
  bool opaque;
  bool transparent;
  unsigned numSlots;

  Result operator+(const Result &other) const {
    Result result;
    result.opaque = this->opaque || other.opaque;
    result.numSlots = this->numSlots + other.numSlots;
    return result;
  }
};

/// Build and solve the MILP model for buffer placement, the funcOp and
/// allChannels stores all the units and channels relate to the circuits. The
/// results are solved and store to res w.r.t to each channel.
LogicalResult placeBufferInCFDFCircuit(
    DenseMap<Value, Result> &res, handshake::FuncOp &funcOp,
    std::vector<Value> &allChannels, std::vector<CFDFC> &cfdfcList,
    std::vector<unsigned> &cfdfcInds, double targetCP, int timeLimit,
    bool setCustom, std::map<std::string, UnitInfo> &unitInfo,
    DenseMap<Value, ChannelBufProps> &channelBufProps);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H
