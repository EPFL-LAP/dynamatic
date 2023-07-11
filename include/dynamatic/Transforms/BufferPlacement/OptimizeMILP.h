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

inline Operation *getUserOp(Value val) {
  auto dstOp = val.getUsers().begin();
  // llvm::errs() << "first user: " << *dstOp << "\n";
  unsigned numUsers = 0;
  for (auto c : val.getUsers()) {
    numUsers++;
  }

  assert(numUsers <= 1 && "There are multiple users!");
  return *dstOp;
}

struct UnitVar {
public:
  bool select;
  GRBVar retIn, retOut;
};

struct ChannelVar {
public:
  bool select;
  GRBVar tDataIn, tDataOut, tElasIn, tElasOut;
  GRBVar tValidIn, tValidOut, tReadyIn, tReadyOut;
  GRBVar thrptTok, bufIsOp, bufNSlots, hasBuf;
  GRBVar valbufIsOp, readybufIsOp;
};

struct Result {
  bool opaque;
  bool transparent;
  unsigned numSlots;

  Result operator+(const Result &other) const {
    Result result;
    result.opaque = this->opaque + other.opaque;
    // result.transparent = this->transparent + other.transparent;
    result.numSlots = this->numSlots + other.numSlots;
    return result;
  }
};

LogicalResult placeBufferInCFDFCircuit(handshake::FuncOp funcOp,
                                       std::vector<Value> allChannels,
                                       CFDFC &CFDFCircuit,
                                       std::map<Value *, Result> &res,
                                       double targetCP);

unsigned getPortInd(Operation *op, Value val);

} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H