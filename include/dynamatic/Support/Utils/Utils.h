//===- Utils.h - Common dependency-less entities ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header is meant to contain entities and utilities used in multiple
// places in Dynamatic and that do not depend on anything from LLVM/MLIR,
// yielding a small object file that is inexpensive to link against.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_UTILS_UTILS_H
#define DYNAMATIC_SUPPORT_UTILS_UTILS_H

#include <array>

namespace dynamatic {

/// Name of the `StringAttr` under which the backend expects the name of the
/// associated RTL component to be stored.
static constexpr const char *RTL_NAME_ATTR_NAME = "hw.name";

/// Name of the `DictionaryAttr` under which the backend expects the parameters
/// of the associated RTL component to be stored.
static constexpr const char *RTL_PARAMETERS_ATTR_NAME = "hw.parameters";

/// The type of a signal in a handshake channel: DATA, VALID, or READY.
enum class SignalType { DATA, VALID, READY };

/// Returns all the possible signal types in a dataflow channel.
std::array<SignalType, 3> getSignalTypes();

/// The type of a port: IN or OUT.
enum class PortType { IN, OUT };

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_UTILS_UTILS_H
