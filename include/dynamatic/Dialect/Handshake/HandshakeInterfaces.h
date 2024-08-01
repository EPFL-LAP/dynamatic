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

class ChannelType;

namespace detail {
/// `SameExtraSignalsInterface` default `getChannelsWithSameExtraSignals`'s
/// function (defined as a free function to avoid instantiating an
/// implementation for every concrete operation type).
SmallVector<mlir::TypedValue<handshake::ChannelType>>
getChannelsWithSameExtraSignals(Operation *op);

/// `SameExtraSignalsInterface`'s verification function (defined as a free
/// function to avoid instantiating an implementation for every concrete
/// operation type).
LogicalResult verifySameExtraSignalsInterface(
    Operation *op, ArrayRef<mlir::TypedValue<ChannelType>> channels);
} // namespace detail

class ControlType;

} // end namespace handshake
} // end namespace dynamatic

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h.inc"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES_H
