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

#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
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

class FuncOp;

/// Returns the name of an operand
/// based on which interface the operation implements
std::string getOperandName(Operation *op, size_t oprdIdx);


/// Returns the name of a result
/// based on which interface the operation implements
std::string getResultName(Operation *op, size_t resIdx);

unsigned getNumInputPorts(Operation *op);
std::string getInputPortName(Operation *op, size_t portIdx);

unsigned getNumOutputPorts(Operation *op);
std::string getOutputPortName(Operation *op, size_t portIdx);

/// Provides an opaque interface for generating the port names of an operation;

// class PortNamer {
// public:
//   /// Does nothing; no port name will be generated.
//   PortNamer() = default;

//   /// Derives port names for the operation on object creation.
//   PortNamer(Operation *op);

//   /// Returs the port name of the input at the specified index.
//   StringRef getInputName(unsigned idx) const { return inputs[idx]; }

//   /// Returs the port name of the output at the specified index.
//   StringRef getOutputName(unsigned idx) const { return outputs[idx]; }

// private:

//   /// List of input port names.
//   SmallVector<std::string> inputs;
//   /// List of output port names.
//   SmallVector<std::string> outputs;
// };

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

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h.inc"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_INTERFACES_H
