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
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace dynamatic {
namespace handshake {

class NamedIOInterface;
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
} // end namespace dynamatic

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h.inc"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES_H
