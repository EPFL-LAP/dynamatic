//===- HandshakeRemoveUnusedMemRef.cpp - Remove unused MemRefs --*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --handshake-remove-unused-memrefs pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/Passes.h"

namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKEREMOVEUNUSEDMEMREFS
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic

using namespace dynamatic;

namespace {

struct HandshakeRemoveUnusedMemRefsPass
    : dynamatic::impl::HandshakeRemoveUnusedMemRefsBase<
          HandshakeRemoveUnusedMemRefsPass> {

  void runOnOperation() override {
    handshake::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    // Iterate through the arguments backwards, this ensures that the block
    // argument indices remain stable after deletion.
    auto argNames = to_vector(func.getArgNames().getValue());
    for (BlockArgument arg : llvm::reverse(func.getArguments())) {
      if (!arg.use_empty() || !isa<MemRefType>(arg.getType()))
        continue;

      func.eraseArgument(arg.getArgNumber());
      argNames.erase(argNames.begin() + arg.getArgNumber());
    }
    // Update the argument names as well.
    func->setAttr("argNames", ArrayAttr::get(&getContext(), argNames));
  }
};
} // namespace
