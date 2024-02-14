//===- InnerSymbolNamespace.h - Inner Symbol Table Namespace ----*- C++ -*-===//
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
// This file declares the InnerSymbolNamespace, which tracks the names
// used by inner symbols within an InnerSymbolTable.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HW_INNER_SYMBOL_NAMESPACE_H
#define DYNAMATIC_DIALECT_HW_INNER_SYMBOL_NAMESPACE_H

#include "dynamatic/Dialect/HW/InnerSymbolTable.h"
#include "dynamatic/Support/Namespace.h"

namespace dynamatic {
namespace hw {

struct InnerSymbolNamespace : Namespace {
  InnerSymbolNamespace() = default;
  InnerSymbolNamespace(Operation *module) { add(module); }

  /// Populate the namespace from a module-like operation. This namespace will
  /// be composed of the `inner_sym`s of the module's ports and declarations.
  void add(Operation *module) {
    hw::InnerSymbolTable::walkSymbols(
        module, [&](StringAttr name, const InnerSymTarget &target) {
          nextIndex.insert({name.getValue(), 0});
        });
  }
};

struct InnerSymbolNamespaceCollection {

  InnerSymbolNamespace &get(Operation *op) {
    return collection.try_emplace(op, op).first->second;
  }

  InnerSymbolNamespace &operator[](Operation *op) { return get(op); }

private:
  DenseMap<Operation *, InnerSymbolNamespace> collection;
};

} // namespace hw
} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HW_INNER_SYMBOL_NAMESPACE_H
