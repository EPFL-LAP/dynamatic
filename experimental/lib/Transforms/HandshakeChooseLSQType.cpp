//===- HandshakeChooseLSQType.cpp - Choose LSQ Type ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// 
// Pass on the handshake dialect 
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/HandshakeChooseLSQType.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeEnums.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"


using namespace llvm;
using namespace mlir;
using namespace dynamatic;

namespace dynamatic {
namespace experimental {
#define GEN_PASS_DEF_HANDSHAKECHOOSELSQTYPE
#include "experimental/Transforms/Passes.h.inc"
}; // namespace experimental
}; // namespace dynamatic

struct HandshakeChooseLSQTypePass
    : public dynamatic::experimental::impl::HandshakeChooseLSQTypeBase<
          HandshakeChooseLSQTypePass> {
  using HandshakeChooseLSQTypeBase<HandshakeChooseLSQTypePass>::HandshakeChooseLSQTypeBase;
  void runDynamaticPass() override;
};

void HandshakeChooseLSQTypePass::runDynamaticPass() {
  ModuleOp module = getOperation();

  llvm::errs() << "[ChooseLSQType] Requested lsq-type: " << lsqType << "\n";

  auto enumType = handshake::symbolizeLSQType(lsqType);
  if (!enumType) {
    module.emitError() << "invalid lsq-type: " << lsqType;
    return signalPassFailure();
  }

  llvm::errs() << "[ChooseLSQType] Resolved enum value: "
               << static_cast<int64_t>(*enumType) << "\n";

  unsigned count = 0;
  module.walk([&](handshake::LSQOp lsq) {
    llvm::errs() << "  - Updating LSQOp @" << lsq.getLoc() << "\n";
    lsq.setLsqType(*enumType);
    ++count;
  });

  llvm::errs() << "[ChooseLSQType] Updated " << count << " LSQOps\n";
}