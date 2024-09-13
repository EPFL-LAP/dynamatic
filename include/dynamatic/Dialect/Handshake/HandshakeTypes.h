//===- HandshakeTypes.h - Handshake types -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares backing data-structures and API to express and manipulate custom
// Handshake types.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPES_H
#define DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPES_H

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

namespace dynamatic {
namespace handshake {

/// Returns the width of a type supported in Handshake-level IR. This is 0 for
/// `handshake::ControlType`, the data type's width for
/// `handshake::ChannelType`, or the regular width for the standard
/// `IntegerType` and `FloatType`.
unsigned getHandshakeTypeBitWidth(mlir::Type type);

/// A dataflow channel's extra signal. The signal has a unique (within a
/// channel's context) name, specific MLIR type, and a direction (downstream or
/// upstream).
struct ExtraSignal {

  /// Used when creating `handshake::ChannelType` instances. Owns its name
  /// instead of referencing it.
  struct Storage {
    std::string name;
    mlir::Type type = nullptr;
    bool downstream = true;

    Storage() = default;
    Storage(llvm::StringRef name, mlir::Type type, bool downstream = true);
  };

  /// The signal's name.
  llvm::StringRef name;
  /// The signal's MLIR type.
  mlir::Type type;
  /// Whether the signal is going downstream or upstream.
  bool downstream;

  /// Simple member-by-member constructor.
  ExtraSignal(llvm::StringRef name, mlir::Type type, bool downstream = true);

  /// Constructs from the storage type (should not be used by client code).
  ExtraSignal(const Storage &storage);

  /// Returns the signal type's bitwidth.
  unsigned getBitWidth() const;
};

bool operator==(const ExtraSignal &lhs, const ExtraSignal &rhs);
inline bool operator!=(const ExtraSignal &lhs, const ExtraSignal &rhs) {
  return !(lhs == rhs);
}

// NOLINTNEXTLINE(readability-identifier-naming)
llvm::hash_code hash_value(const ExtraSignal &signal);

namespace detail {
/// Parses a handshake::ControlType or handshake::ChannelType and returns it as
/// an opaque Type. Returns nullptr on a parsing failure.
mlir::Type jointHandshakeTypeParser(mlir::AsmParser &parser);
} // namespace detail

} // namespace handshake
} // namespace dynamatic

namespace mlir {
class IndexType;
class IntegerType;
class FloatType;
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h.inc"

#endif // DYNAMATIC_DIALECT_HANDSHAKE_HANDSHAKE_TYPES_H
