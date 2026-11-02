//===- FPGA20FPL22Buffers.h - FPGA'20/FPL'22 buffer placement ---*- C -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// 
// This is an attempt at merging two differen talgorithms that were preivously 
// used for bfufers placement: 
// FPGA20 (https://dl.acm.org/doi/full/10.1145/3477053) and FPL22
// (https://ieeexplore.ieee.org/abstract/document/10035122).
//
// The two algorithms differed in two main ways:
// 1. The timing model (`SignalModel`): FPGA20 only models the data signal,
//    while FPL22 models data, valid, and ready signals as well as 
//    combinational paths inside units.
// 2. The number of MILPs (`MILPMode`): either we treat the whole circuit as a
//    single MILP (like it was done in FPGA20), or we decompose it into one MILP 
//    per disjoint CFDFC union plus one MILP for everything outside of all CFDFCs. 
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_MIXEDBUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_MIXEDBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/Utils/CFDFC.h"
#include "llvm/ADT/MapVector.h"

namespace dynamatic {
namespace buffer {
namespace mixedbuffers {

enum class SignalModel {
  /// FPGA20: only the data signal is modelled. Ready paths are broken
  /// heuristically after the MILP is solved.
  DATA_ONLY,
  /// FPL22: data, valid, and ready signals are modelled, including
  /// paths inside units.
  DATA_VALID_READY
};


/// Which channels and units of the function a single MILP covers.
enum class MILPMode {
  /// All in one MILP
  ONE_MILP,
  /// Split by disjoint CFDFCs unions
  CFDFC_UNION,
  // We need an additional mode for "out of cycle buffers" from FPL22
  OUT_OF_CYCLE
};

/// MILP implementing buffer placement. To buffer a dataflow
/// circuit, it creates, for the channels and units in its scope:
/// 1. custom channel constraints derived from channel-specific buffering
///    properties
/// 2. path constraints for every modelled timing domain (plus mixed-domain
///    constraints inside units for `SignalModel::DATA_VALID_READY`)
/// 3. elasticity constraints
/// 4. throughput constraints for all CFDFCs in scope that were marked for
///    optimization (none for `MILPMode::OUT_OF_CYCLE`)
/// 5. a maximization objective that rewards high CFDFC throughputs (except for
///    `MILPMode::OUT_OF_CYCLE`) and penalizes the placement of many large
///    buffers in the circuit
class MixedBuffers : public BufferPlacementMILP {
public:
  /// Setups the entire MILP that buffers the part of the input dataflow
  /// circuit given by `scope` for the target clock period, after which (absent
  /// errors) it is ready for optimization. `cfUnion` must be non-null if and
  /// only if `scope` is `MILPMode::CFDFC_UNION`. If a channel's buffering
  /// properties are provably unsatisfiable, the MILP will not be marked ready
  /// for optimization, ensuring that further calls to `optimize` fail.
  MixedBuffers(CPSolver::SolverKind solverKind, int timeout,
              FuncInfo &funcInfo, const TimingDatabase &timingDB,
              double targetPeriod, SignalModel signalModel,
              MILPMode mode, CFDFCUnion *cfUnion = nullptr,
              StringRef writeTo = "", bool ftd = false);

protected:
  /// Interprets the MILP solution to derive buffer placement decisions. Since
  /// the MILP cannot encode the placement of both opaque and transparent slots
  /// on a single channel, some "interpretation" of the results is necessary to
  /// derive "mixed" placements where some buffer slots are opaque and some are
  /// transparent.
  void extractResult(BufferPlacement &placement) override;

private:
  /// Timing model of the MILP.
  const SignalModel signalModel;
  /// Part of the function the MILP covers.
  const MILPMode mode;
  /// The CFDFC union over which the MILP is described. Only non-null for
  /// `MILPMode::CFDFC_UNION`.
  CFDFCUnion *cfUnion;
  /// Union of all CFDFCs in the function. Only set for
  /// `MILPMode::OUT_OF_CYCLE`, where it makes "is this channel/unit part of
  /// any CFDFC?" queries fast(er).
  std::optional<CFDFCUnion> allCFDFCs;
  /// Buffering groups for the signal model, ordered from the group whose
  /// buffer sits first on a channel to the group whose buffer sits last.
  SmallVector<BufferingGroup> bufGroups;
  /// All signals modelled by the MILP (the concatenation of all buffering
  /// groups' signals).
  SmallVector<SignalType> signalTypes;
  /// Returns true for channels that are in the MILP's scope; used to filter
  /// unit ports when creating unit constraints.
  std::function<bool(Value)> channelFilter;

  /// Returns the signal model's buffering groups.
  static SmallVector<BufferingGroup> getBufferingGroups(SignalModel model);

  /// Returns the base class' algorithm identifier for a signal model.
  static Algorithm getAlgorithm(SignalModel model);

  /// Whether the channel is adjacent to a memory interface.
  static bool isMemoryAdjacent(Value channel);

  /// Whether the channel is covered by the MILP.
  bool isChannelInScope(Value channel) const;

  /// Whether path and elasticity constraints are created for a channel covered
  /// by the MILP.
  bool hasTimingConstraints(Value channel) const;

  /// Whether the unit is covered by the MILP.
  bool isUnitInScope(Operation *unit) const;

  /// Adds channel-specific buffering constraints that were parsed from IR
  /// annotations to the model. Dispatches on the signal model.
  void addCustomChannelConstraints(Value channel);
  /// FPGA20 encoding of `addCustomChannelConstraints`.
  void addDataOnlyCustomChannelConstraints(Value channel);
  /// FPL'22 encoding of `addCustomChannelConstraints`.
  void addDataValidReadyCustomChannelConstraints(Value channel);

  /// Adds path constraints between pins of the unit's input and output ports in
  /// different timing domains. At the moment the set of mixed-domain
  /// constraints is determined by the method itself, which contains
  /// case-by-case logic to derive them based on the operation's type. Only
  /// meaningful for `SignalModel::DATA_VALID_READY`.
  ///
  /// A `filter` can be provided to filter out constraints involving input or
  /// output ports connected to channels for which the filter returns false. It
  /// is only valid to call this method after having added channel variables to
  /// the model for all channels adjacent to the unit, unless these channels are
  /// filtered out by the `filter` function.
  void addUnitMixedPathConstraints(Operation *unit,
                                  ChannelFilter filter = nullFilter);

  /// Adds the channel variables and all channel-level constraints (custom,
  /// path, and elasticity) for the channel.
  void addChannelConstraints(Value channel);

  /// Adds all unit-level constraints for the unit.
  void addUnitConstraints(Operation *unit);

  /// Returns the CFDFCs in scope that are marked for optimization.
  SmallVector<CFDFC *> getCFDFCsToOptimize() const;

  /// Sets the objective that only penalizes buffer placement, used when the
  /// MILP has no CFDFC to optimize the throughput of.
  void addMinBuffersObjective(ArrayRef<Value> channels);

  /// Setups the entire MILP, creating all variables, constraints, and setting
  /// the system's objective. Called by the constructor in the absence of prior
  /// failures, after which the MILP is ready to be optimized.
  void setup();
};

/// Sets the `handshake.tp` attribute of the function from the throughputs
/// stored in its CFDFCs. MUST be called once, after all MILPs covering the
/// function have been solved. Keys are the CFDFCs' indices in
/// `funcInfo.cfdfcs`, matching the `handshake.cfdfc` attribute.
void setCFDFCThroughputAttr(FuncInfo &funcInfo);

} // namespace mixedbuffers
} // namespace buffer
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPGA20FPL22BUFFERS_H
