//===- Utils.h - Common dependency-less entities ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header is meant to contain entities and utilities used in multiple in
// Dynamatic and that do not depend on anything from LLVM/MLIR, yielding a small
// object file that is inexpensive to link against.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_UTILS_UTILS_H
#define DYNAMATIC_SUPPORT_UTILS_UTILS_H

namespace dynamatic {
/// The type of a signal in a handshake channel: DATA, VALID, or READY.
enum class SignalType { DATA, VALID, READY };
/// The type of a port: IN or OUT.
enum class PortType { IN, OUT };
} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_UTILS_UTILS_H