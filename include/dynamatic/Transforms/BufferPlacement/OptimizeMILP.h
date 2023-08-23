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

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
namespace dynamatic {
namespace buffer {

using namespace circt;
using namespace handshake;

/// Data structure to store the results of buffer placement, including the
/// property and the total slots of the channel
struct PlacementResult {
  bool opaque;
  bool transparent;
  unsigned numSlots;

  PlacementResult operator+(const PlacementResult &other) const {
    PlacementResult result;
    result.opaque = this->opaque || other.opaque;
    result.numSlots = this->numSlots + other.numSlots;
    return result;
  }
};

/// Data structure to store the variables w.r.t to a unit(operation), including
/// whether it belongs to a CFDFC, and its retime variables.
struct UnitVar {
  bool select;
  GRBVar retIn, retOut;
};

/// Data structure to store the variables w.r.t to a channel(value), including
/// whether it belongs to a CFDFC, and its time, throughput, and buffer
/// placement decision.
struct ChannelVar {
  bool select;
  GRBVar tDataIn, tDataOut, tElasIn, tElasOut;
  GRBVar bufIsOp, bufNSlots, hasBuf;
};

/// Helper data structure to hold mappings between operations/values and their
/// corresponding Gurobi variables.
struct MILPVars {
  DenseMap<CFDFC *, DenseMap<Operation *, UnitVar>> units;
  DenseMap<Value, ChannelVar> channels;
  DenseMap<CFDFC *, DenseMap<Value, GRBVar>> channelThroughput;
  DenseMap<CFDFC *, GRBVar> circuitThroughput;
};

class BufferPlacementMILP {
public:
  /// Handshake function being optimized.
  handshake::FuncOp funcOp;
  /// Maps each CFDFC in the function to a boolean indicating whether it should
  /// be optimized.
  llvm::MapVector<CFDFC *, bool> &cfdfcs;
  /// Units characterization.
  std::map<std::string, UnitInfo> &unitInfo;
  /// Target clock period.
  double targetPeriod;

  BufferPlacementMILP(handshake::FuncOp funcOp,
                      llvm::MapVector<CFDFC *, bool> &cfdfcs,
                      std::map<std::string, UnitInfo> &unitInfo,
                      double targetPeriod, GRBEnv &env, double timeLimit);

  LogicalResult optimize(DenseMap<Value, PlacementResult> &placement);

protected:
  /// After construction, contains all channels in the function, mapped to any
  /// specific channel buffering properties they may have.
  DenseMap<Value, ChannelBufProps> channels;
  /// Number of units in the function.
  unsigned numUnits;
  // Gurobi model for creating/solving the MILP.
  GRBModel model;
  /// MILP variables.
  MILPVars vars;
  // Whether the MILP is impossible to satisfy.
  bool unsatisfiable = false;

  void setupMILP();

  void initializeVars();

  void addCustomChannelConstraints();

  void addPathConstraints();

  void addElasticityConstraints();

  void addThroughputConstraints();

  void addObjective();

  LogicalResult hardcodeBufProps(Value channel, ChannelBufProps &props);
};

} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_OPTIMIZEMILP_H
