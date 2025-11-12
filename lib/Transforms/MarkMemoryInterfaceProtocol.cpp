//===- MarkMemoryInterfaceProtocol.cpp - Mark Memory Interfaces with the
// selected protocol //--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --hw-mark-li-memory-interface pass, which marks
// handshake memory interfaces protocol.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/MarkMemoryInterfaceProtocol.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace dynamatic;

namespace {
class MarkMemoryInterfaceProtocolPass
    : public dynamatic::impl::MarkMemoryInterfaceProtocolBase<
          MarkMemoryInterfaceProtocolPass> {
public:
  MarkMemoryInterfaceProtocolPass(std::string protocol) {
    this->protocol = protocol;
  }
  void runDynamaticPass() override {
    MLIRContext *ctx = &getContext();

    // Validate protocol option
    if (protocol != "synchronous" && protocol != "fifo") {
      llvm::errs() << "Invalid protocol option: " << protocol << "\n";
      llvm::errs() << "Valid options are 'synchronous' and 'fifo'.\n";
      llvm::report_fatal_error("Aborting due to invalid protocol option.");
    }

    // Walk all memory interface operations and set the protocol attribute
    getOperation()->walk([&](Operation *op) {
      handshake::MemoryControllerOp memControlOp =
          dyn_cast<dynamatic::handshake::MemoryControllerOp>(op);
      if (!memControlOp)
        return;
      if (protocol == "synchronous") {
        memControlOp.setMemProtocolSync();
      } else if (protocol == "fifo") {
        memControlOp.setMemProtocolFIFO();
      }
    });
  }
};
} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createMarkMemoryInterfaceProtocolPass(std::string protocol) {
  return std::make_unique<MarkMemoryInterfaceProtocolPass>(protocol);
}
