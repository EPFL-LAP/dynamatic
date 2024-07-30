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

#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/Any.h"

namespace dynamatic {
namespace handshake {

namespace detail {

/// `PreservesExtraSignals` trait's verification function (defined as a free
/// function to avoid instantiating an implementation for every concrete
/// operation type).
LogicalResult verifyPreservesExtraSignals(Operation *op);
} // namespace detail

class ControlType;

} // end namespace handshake
} // end namespace dynamatic

namespace mlir {
namespace OpTrait {
/// Operation trait guranteeing that all the operands and results of an
/// operation that have the `handshake::ChannelType` have the exact same list of
/// extra signals.
template <typename ConcreteType>
class PreservesExtraSignals
    : public TraitBase<ConcreteType, PreservesExtraSignals> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return dynamatic::handshake::detail::verifyPreservesExtraSignals(op);
  }
};

} // namespace OpTrait
} // namespace mlir

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h.inc"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES_H
