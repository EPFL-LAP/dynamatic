//===- Attribute.cpp - Support for Dynamatic attributes ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements helpers to work with attributes.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/Attribute.h"

using namespace mlir;
using namespace dynamatic;

size_t dynamatic::detail::toIdx(const mlir::NamedAttribute &attr) {
  std::string str = attr.getName().str();
  bool validNumber = std::all_of(str.begin(), str.end(),
                                 [](char c) { return std::isdigit(c); });
  assert(validNumber && "invalid index");
  return stoi(str);
}