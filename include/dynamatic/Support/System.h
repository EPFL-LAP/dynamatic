//===- System.h - System helpers --------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper for interacting with the host system.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_SYSTEM_H
#define DYNAMATIC_SUPPORT_SYSTEM_H

#include "dynamatic/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace dynamatic {

constexpr int COMMAND_EMPTY = -1000;

/// Issues a command to the system with the provided tokens separated by spaces.
/// Returns the command return code or the `COMMAND_EMPTY` constant if there are
/// no tokens.
int exec(std::initializer_list<StringRef> tokens);

/// Issues a command to the system with the provided tokens separated by spaces.
/// Returns the command return code or the `COMMAND_EMPTY` constant if there are
/// no tokens.
template <typename... Tokens>
int exec(Tokens... tokens) {
  return exec({tokens...});
}

} // namespace dynamatic

namespace llvm {
namespace sys {
namespace path {
/// Removes any trailing separators from the path.
StringRef removeTrailingSeparators(StringRef path);
} // namespace path
} // namespace sys
} // namespace llvm

#endif // DYNAMATIC_SUPPORT_SYSTEM_H