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

class ControlType;

} // end namespace handshake
} // end namespace dynamatic

namespace dynamatic {
namespace handshake {

// Trait with useful functionality for use
// in getting string representations of operand and result names
template <typename ConcreteOp>
class NamedIOUtilities
    : public mlir::OpTrait::TraitBase<ConcreteOp, NamedIOUtilities> {
public:
  static void validateOperandIdx(unsigned idx, unsigned count) {
    assert(idx < count && "operand index too high");
  }

  void validateOperandIdx(unsigned idx) {
    validateOperandIdx(idx, this->getOperation()->getNumOperands());
  }

  static void validateResultIdx(unsigned idx, unsigned count) {
    assert(idx < count && "result index too high");
  }

  void validateResultIdx(unsigned idx) {
    validateResultIdx(idx, this->getOperation()->getNumResults());
  }

  static std::string simpleOperandName(unsigned idx, unsigned count) {
    validateOperandIdx(idx, count);

    // TODO: remove 2D io packing
    // but the underscore marking is needed currently
    // as the netlist printer uses it to identify 2D signals
    if (count == 1)
      return "ins";
    return "ins_" + std::to_string(idx);
  }

  std::string simpleOperandName(unsigned idx) {
    return simpleOperandName(idx, this->getOperation()->getNumOperands());
  }

  static std::string simpleResultName(unsigned idx, unsigned count) {
    validateResultIdx(idx, count);

    // TODO: remove 2D io packing
    // but the underscore marking is needed currently
    // as the netlist printer uses it to identify 2D signals
    if (count == 1)
      return "outs";
    return "outs_" + std::to_string(idx);
  }

  std::string simpleResultName(unsigned idx) {
    return simpleResultName(idx, this->getOperation()->getNumResults());
  }
};

// Trait that calls simpleOperandName and simpleResultName
// for the NamedIOInterface functions
// so ops with this behaviour can just declare the trait
template <typename ConcreteOp>
class SimpleNamedIO
    : public mlir::OpTrait::TraitBase<ConcreteOp, SimpleNamedIO> {
public:
  std::string getOperandName(unsigned idx) {
    return static_cast<ConcreteOp *>(this)->simpleOperandName(idx);
  }
  std::string getResultName(unsigned idx) {
    return static_cast<ConcreteOp *>(this)->simpleResultName(idx);
  }
};

// Trait that returns "lhs"/"rhs" and "result"
// for the NamedIOInterface functions
// so ops with this behaviour can just declare the trait
template <typename ConcreteOp>
class BinaryArithNamedIO
    : public mlir::OpTrait::TraitBase<ConcreteOp, BinaryArithNamedIO> {
public:
  static std::string getOperandName(unsigned idx) {
    ConcreteOp::validateOperandIdx(idx, 2);
    return (idx == 0) ? "lhs" : "rhs";
  }
  static std::string getResultName(unsigned idx) {
    ConcreteOp::validateResultIdx(idx, 1);
    return "result";
  }
};

} // namespace handshake
} // namespace dynamatic

namespace dynamatic {

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
} // end namespace dynamatic

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h.inc"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES_H
