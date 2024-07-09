//===- System.cpp - System helpers ------------------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of system helpers.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/System.h"
#include "llvm/Support/Path.h"
#include <sstream>

using namespace llvm;
using namespace dynamatic;

int dynamatic::exec(std::initializer_list<StringRef> tokens) {
  if (tokens.size() == 0)
    return COMMAND_EMPTY;

  // Append all arguments into the command, with a space in between each
  std::stringstream cmd;
  auto *it = tokens.begin();
  StringRef arg = *it;
  while (++it != tokens.end()) {
    cmd << arg.str() + " ";
    arg = *it;
  }
  cmd << arg.str();

  return std::system(cmd.str().c_str());
}

StringRef llvm::sys::path::removeTrailingSeparators(StringRef path) {
  StringRef sep = sys::path::get_separator();
  while (path.ends_with(sep))
    path = path.substr(0, path.size() - sep.size());
  return path;
}
