//===- HandshakeInterfaces.h - Handshake interfaces -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces of the handshake dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES_H
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES_H

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOpInternalStateNamer.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallSet.h"

namespace dynamatic {
namespace handshake {

class NamedIOInterface;
class LatencyInterface;
class FuncOp;

/// Provides an opaque interface for generating the port names of an operation;
/// handshake operations generate names by the `handshake::NamedIOInterface`;
/// other operations, such as arithmetic ones, are assigned default names.
class PortNamer {
public:
  /// Does nothing; no port name will be generated.
  PortNamer() = default;

  /// Derives port names for the operation on object creation.
  PortNamer(Operation *op);

  /// Returs the port name of the input at the specified index.
  StringRef getInputName(unsigned idx) const { return inputs[idx]; }

  /// Returs the port name of the output at the specified index.
  StringRef getOutputName(unsigned idx) const { return outputs[idx]; }

private:
  /// Maps the index of an input or output to its port name.
  using IdxToStrF = const std::function<std::string(unsigned)> &;

  /// Infers port names for the operation using the provided callbacks.
  void infer(Operation *op, IdxToStrF &inF, IdxToStrF &outF);

  /// Infers default port names when nothing better can be achieved.
  void inferDefault(Operation *op);

  /// Infers port names for an operation implementing the
  /// `handshake::NamedIOInterface` interface.
  void inferFromNamedOpInterface(NamedIOInterface namedIO);

  /// Infers port names for a Handshake function.
  void inferFromFuncOp(FuncOp funcOp);

  /// List of input port names.
  SmallVector<std::string> inputs;
  /// List of output port names.
  SmallVector<std::string> outputs;
};

class ControlType;

} // end namespace handshake

namespace buffer {

/// One independent retiming path declared by an op implementing
/// handshake::RetimingPathsOpInterface. The path groups a set of operand
/// indices and a set of result indices that share a single
/// fluid-retiming token-conservation accounting in the buffer placement
/// MILP, plus the cycle latency for that path.
struct RetimingPath {
  llvm::SmallSet<unsigned, 4> operands;
  llvm::SmallSet<unsigned, 4> results;

  RetimingPath() = default;

  /// Builds a single-path descriptor that spans every operand and every
  /// result of `unit`.
  RetimingPath(Operation *unit) {
    for (unsigned i = 0, e = unit->getNumOperands(); i < e; ++i)
      operands.insert(i);
    for (unsigned i = 0, e = unit->getNumResults(); i < e; ++i)
      results.insert(i);
  }
};

/// Returns the path partition for `unit`: the interface's declaration if the
/// op implements RetimingPathsOpInterface, otherwise a single path spanning
/// every operand and result. Free function (not a member of RetimingPath)
/// because `SmallVector<RetimingPath>` cannot be used inside RetimingPath's
/// own class definition (incomplete type).
SmallVector<RetimingPath> getRetimingPaths(Operation *unit);

} // end namespace buffer
} // end namespace dynamatic

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h.inc"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES_H
