//===- HandshakeMarkBLIFImpl.cpp - Mark BLIF Implementation of Handshake Ops
//--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that marks each handshake operation with the path
// of the BLIF file where its AIG implementation is located. This is used in the
// conversion from Handshake to Synth to convert each handshake operation into
// an hw module with the correct BLIF file path attribute.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Support/BLIFFileManager.h"
#include "llvm/ADT/TypeSwitch.h"
#include <filesystem>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Transforms/Passes.h" // IWYU pragma: keep
namespace dynamatic {
// include tblgen base class definition
#define GEN_PASS_DEF_HANDSHAKEMARKBLIFIMPL
#include "dynamatic/Transforms/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

namespace {

struct HandshakeMarkBLIFImplPass
    : public dynamatic::impl::HandshakeMarkBLIFImplBase<
          HandshakeMarkBLIFImplPass> {
  // use tblgen constructors from base class
  using HandshakeMarkBLIFImplBase::HandshakeMarkBLIFImplBase;

  void runDynamaticPass() override {
    // Check blifDirPath is not empty
    if (blifDirPath.empty()) {
      llvm::errs() << "BLIF directory path is empty\n";
      return signalPassFailure();
    }
    // Generate blif manager with the provided directory path
    BLIFFileManager blifFileManager(blifDirPath);
    // Get the module op
    auto moduleOp = getOperation();
    // Walk through all handshake operations in the function
    moduleOp.walk([&](Operation *op) {
      // Skip function operations
      if (isa<handshake::FuncOp>(op))
        return;
      // Create the blif path by combining the directory path and the op name
      // depending on the operation type
      std::string blifFilePath =
          blifFileManager.getBlifFilePathForHandshakeOp(op);
      // Create a string attribute for the blif path
      mlir::StringAttr blifPathAttr =
          mlir::StringAttr::get(op->getContext(), blifFilePath);
      // Set the blif path attribute on the operation
      BLIFImplInterface blifImplInterface = dyn_cast<BLIFImplInterface>(op);
      assert(blifImplInterface &&
             "Operation does not implement BLIFImplInterface");
      blifImplInterface.setBLIFImpl(blifPathAttr);
    });
  }
};

} // namespace
