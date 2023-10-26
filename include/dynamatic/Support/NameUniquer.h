//===- NameUniquer.h - Unique Handshake ops and values ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a way to give a deterministically computed unique name to each
// operation and operand inside all Handshake functions in a module. This is
// very useful when exporting Handshake-level IR to a representation which
// requires that all operations and/or operands have unique names serializable
// to a text file.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_NAMEUNIQUER_H
#define DYNAMATIC_SUPPORT_NAMEUNIQUER_H

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"

namespace dynamatic {

/// Creates a deterministic, queryable, bi-directional mapping between unique
/// string names and individual operations/operands inside Handshake functions.
/// The mapping is deterministic in the sense that it will always be the same
/// when created on the exact same MLIR module or Handshake function. The
/// mapping never changes after object creation, meaning that, for example,
/// operations inserted after object creation will not be given unique names.
class NameUniquer {
public:
  /// Creates the mapping between all operations/operands inside all Handshake
  /// functions in the module.
  NameUniquer(mlir::ModuleOp modOp);

  /// Creates the mapping between all operations/operands inside the Handshake
  /// function.
  NameUniquer(circt::handshake::FuncOp funcOp);

  /// Returns the unique name associated to the operation. Asserts if the
  /// operation did not exist as part of a Handshake function at object creation
  /// time.
  StringRef getName(Operation &op);

  /// Returns the unique name associated to the operand. Asserts if the operand
  /// did not exist as part of a Handshake function at object creation time.
  StringRef getName(OpOperand &opr);

  /// Returns the operation that was given the unique name. Asserts if the name
  /// does not map to an operation that existed as part of a Handsake function
  /// at object creation time.
  Operation &getOp(StringRef opName);

  /// Returns the operand that was given the unique name. Asserts if the name
  /// does not map to an operand that existed as part of a Handsake function at
  /// object creation time.
  OpOperand &getOperand(StringRef oprdName);

private:
  /// Maps all operations inside Handshake functions to a unique name.
  mlir::DenseMap<Operation *, std::string> opToName;
  /// Reverse mapping between names and operations inside Handshake functions.
  llvm::StringMap<Operation *> nameToOp;
  /// Maps all operands inside Handshake functions to a unique name.
  mlir::DenseMap<OpOperand *, std::string> oprdToName;
  /// Reverse mapping between names and operands inside Handshake functions.
  llvm::StringMap<OpOperand *> nameToOprd;

  /// Reisters all operations and operands inside the Handshake function into
  /// the various maps, using operation counters to determine unique names for
  /// operations.
  void registerFunc(circt::handshake::FuncOp funcOp,
                    DenseMap<mlir::OperationName, unsigned> &opCounters);
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_NAMEUNIQUER_H
