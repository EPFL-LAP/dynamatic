//===- CostAwareBuffers.h - Cost-aware buffer placement ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cost-aware smart buffer placement, as presented in
// https://dl.acm.org/doi/full/10.1145/3477053
//
// This mainly declares the `CostAwareBuffers` class, which inherits the
// abstract `BufferPlacementMILP` class to setup and solve a real MILP from 
// which buffering decisions can be made. Every public member declared in 
// this file is under the `dynamatic::buffer::costaware` namespace, as to not 
// create name conflicts for common structs with other implementors of
// `BufferPlacementMILP`.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_COSTAWAREBUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_COSTAWAREBUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace dynamatic {
namespace buffer {
namespace costaware {

/// The type of a buffer in a handshake channel: OEHB chain, TEHB chain, 
/// Full-transparent FIFO, SingleEnable, or DVR.
enum class BufferType { OB, TB, FT, SE, DR };

struct ChannelVars {
  /// For specific signals on the channel, arrival time at channel's endpoints
  /// (real, real) and buffer presence (binary).
  std::map<SignalType, ChannelSignalVars> signalVars;
  /// Elastic arrival time at channel's endpoints (real).
  // TimeVars elastic;
  /// Presence of any buffer on the channel (binary).
  GRBVar bufPresent;
  /// Number of each type of buffer's slots on the channel (integer).
  std::map<BufferType, GRBVar> bufNumSlots;
  /// Presence of a DVSE on the channel (binary).
  GRBVar sePresent;
};

struct MILPVars {
  /// Mapping between CFDFCs and their related variables.
  llvm::MapVector<CFDFC *, CFDFCVars> cfVars;
  /// Mapping between channels and their related variables.
  llvm::MapVector<Value, ChannelVars> channelVars;
};


class CostAwareBuffers : public BufferPlacementMILP {
public:

  /// Setups the entire MILP that buffers the input dataflow circuit for the
  /// target clock period, after which (absent errors) it is ready for
  /// optimization. If a channel's buffering properties are provably unsatisfiable,
  /// the MILP will not be marked ready for optimization, ensuring that further
  /// calls to `optimize` fail.
  CostAwareBuffers(GRBEnv &env, FuncInfo &funcInfo,
                   const TimingDatabase &timingDB, double targetPeriod);

  CostAwareBuffers(GRBEnv &env, FuncInfo &funcInfo,
                   const TimingDatabase &timingDB, double targetPeriod,
                   Logger &logger, StringRef milpName);

protected:

  MILPVars vars;

  // Matrix to store the relationship between SignalType and BufferType
  // Certain type of buffers cut certain signals
  std::map<SignalType, std::map<BufferType, bool>> signalBufferMatrix = {
    { SignalType::DATA, {
        { BufferType::OB, true },
        { BufferType::TB, false },
        { BufferType::FT, false },
        { BufferType::SE, true },
        { BufferType::DR, true }
    }},
    { SignalType::VALID, {
        { BufferType::OB, true },
        { BufferType::TB, false },
        { BufferType::FT, false },
        { BufferType::SE, true },
        { BufferType::DR, true }
    }},
    { SignalType::READY, {
        { BufferType::OB, false },
        { BufferType::TB, true },
        { BufferType::FT, false },
        { BufferType::SE, false },
        { BufferType::DR, true }
    }}
  };
  
  void addCFDFCVars(CFDFC &cfdfc);

  void addChannelVars(Value channel, ArrayRef<BufferType> buffers,
                                          ArrayRef<SignalType> signals);

  void addChannelPathConstraints(Value channel, SignalType signal,
                                 const TimingModel *bufModel);

  void addUnitPathConstraints(Operation *unit, SignalType type,
                              ChannelFilter filter = nullFilter);  

  void addChannelCustomConstraints(Value channel);

  void addChannelThroughputConstraints(CFDFC &cfdfc);

  void addUnitThroughputConstraints(CFDFC &cfdfc);

  void addObjective(ValueRange channels, ArrayRef<BufferType> buffers, ArrayRef<CFDFC *> cfdfcs);

  void extractResult(BufferPlacement &placement) override;

private:

  /// Setups the entire MILP, creating all variables, constraints, and setting
  /// the system's objective. Called by the constructor in the absence of prior
  /// failures, after which the MILP is ready to be optimized.
  void setup();
};

} // namespace costaware
} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_COSTAWAREBUFFERS_H