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

class LatencyInterface;
class NamedIOInterface;
class FuncOp;

/// Returns the NamedIOInterface for `op`, emitting a fatal error if the op
/// does not implement it.
NamedIOInterface getNamedIO(Operation *op);

class ControlType;

namespace detail {

inline std::string simpleOperandName(unsigned idx, unsigned numOperands) {
  assert(idx < numOperands && "index too high");

  // TODO: Remove 2D I/O packing
  // but for now this is needed
  if (numOperands == 1) {
    return "ins";
  }

  return "ins_" + std::to_string(idx);
}

inline std::string simpleResultName(unsigned idx, unsigned numResults) {
  assert(idx < numResults && "index too high");

  // TODO: Remove 2D I/O packing
  // but for now this is needed
  if (numResults == 1) {
    return "outs";
  }

  return "outs_" + std::to_string(idx);
}

} // end namespace detail
} // end namespace handshake
} // end namespace dynamatic

namespace mlir {
namespace OpTrait {

template <typename ConcreteOp>
class SimpleNamedIO : public TraitBase<ConcreteOp, SimpleNamedIO> {
public:
  std::string getOperandName(unsigned idx) {
    return ::dynamatic::handshake::detail::simpleOperandName(
        idx, this->getOperation()->getNumOperands());
  }
  std::string getResultName(unsigned idx) {
    return ::dynamatic::handshake::detail::simpleResultName(
        idx, this->getOperation()->getNumResults());
  }
};

template <typename ConcreteOp>
class BinaryArithNamedIO : public TraitBase<ConcreteOp, BinaryArithNamedIO> {
public:
  std::string getOperandName(unsigned idx) {
    assert(idx < 2 && "index too high");
    return (idx == 0) ? "lhs" : "rhs";
  }
  std::string getResultName(unsigned idx) {
    assert(idx < 1 && "index too high");
    return "result";
  }
};

template <typename ConcreteOp>
class ValidateIO : public TraitBase<ConcreteOp, ValidateIO> {
public:
  void validateOperandIdx(unsigned idx) {
    if (idx >= this->getOperation()->getNumOperands())
      llvm::report_fatal_error("operand index too high");
  }
  void validateResultIdx(unsigned idx) {
    if (idx >= this->getOperation()->getNumResults())
      llvm::report_fatal_error("result index too high");
  }
};

} // namespace OpTrait
} // namespace mlir

namespace dynamatic {
namespace handshake {

namespace buffer {

/// One independent retiming path through a unit.
/// Each operand and result must belong to a single path through the unit
///
/// Channels which connect to an operand or result use the retiming variable
/// of the path the operand or result belongs to.
///
/// This allows the MILP solver to correctly distribute token occupancy
/// accross channels to achieve maximum performance.
struct RetimingPath {
  llvm::SmallSet<unsigned, 4> operands;
  llvm::SmallSet<unsigned, 4> results;

  RetimingPath() = default;

  /// Default constructor for units which do not specify otherwise:
  /// a single retiming path through the unit
  /// which all operands and all results belong to
  explicit RetimingPath(Operation *unit) {
    for (unsigned i = 0, e = unit->getNumOperands(); i < e; ++i)
      operands.insert(i);
    for (unsigned i = 0, e = unit->getNumResults(); i < e; ++i)
      results.insert(i);
  }
};

/// Returns the retiming paths through a unit
SmallVector<RetimingPath> getRetimingPaths(Operation *unit);

} // end namespace buffer
} // end namespace handshake
} // end namespace dynamatic

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h.inc"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES_H
