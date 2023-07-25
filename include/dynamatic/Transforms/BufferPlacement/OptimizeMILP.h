//===- OptimizeMILP.h - optimize MILP model over CFDFC  ---------*- C++ -*-===//
//
// This file declaresfunction the functions of MILP solver for buffer placement.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H

#include "dynamatic/Support/BufferingStrategy.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/ExtractMG.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

namespace dynamatic {
namespace buffer {

using namespace circt;
using namespace handshake;

/// Get user of a value, which should be a single user as the value indicating a
/// channel should be connected to only one unit.
inline Operation *getUserOp(Value val) {
  auto dstOp = val.getUsers().begin();
  unsigned numUsers = 0;
  for (auto c : val.getUsers()) {
    numUsers++;
  }

  assert(numUsers <= 1 && "There are multiple users!");
  return *dstOp;
}

/// Data structure to store the variables w.r.t to a unit(operation), including
/// whether it belongs to a CFDFC, and its retime variables.
struct UnitVar {
public:
  bool select;
  GRBVar retIn, retOut;
};

/// Data structure to store the variables w.r.t to a channel(value), including
/// whether it belongs to a CFDFC, and its time, throughput, and buffer
/// placement decision.
struct ChannelVar {
public:
  bool select;
  GRBVar tDataIn, tDataOut, tElasIn, tElasOut;
  GRBVar tValidIn, tValidOut, tReadyIn, tReadyOut;
  GRBVar thrptTok, bufIsOp, bufNSlots, hasBuf;
  GRBVar valbufIsOp, rdybufIsTr;
};

/// Data structure to store the results of buffer placement, including the
/// property and the total slots of the channel
struct Result {
  bool opaque;
  bool transparent;
  unsigned numSlots;

  Result operator+(const Result &other) const {
    Result result;
    result.opaque = this->opaque + other.opaque;
    result.numSlots = this->numSlots + other.numSlots;
    return result;
  }
};

/// Build and solve the MILP model for buffer placement, the funcOp and
/// allChannels stores all the units and channels relate to the circuits. The
/// results are solved and store to res w.r.t to each channel.
LogicalResult placeBufferInCFDFCircuit(handshake::FuncOp funcOp,
                                       std::vector<Value> &allChannels,
                                       CFDFC &CFDFCircuit,
                                       std::map<Value *, Result> &res,
                                       double targetCP);

/// Get the port index of a unit
unsigned getPortInd(Operation *op, Value val);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H