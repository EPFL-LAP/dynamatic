//===- HandshakeAnnotateProperties.cpp - Property annotation ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --handshake-annotate-properties pass.
//
//===----------------------------------------------------------------------===//

#include "experimental/Analysis/FormalPropertyAnnotation/HandshakeAnnotateProperties.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "experimental/Support/FormalProperty.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include <fstream>
#include <ostream>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::formalprop;

namespace {

struct HandshakeAnnotatePropertiesPass
    : public dynamatic::experimental::formalprop::impl::
          HandshakeAnnotatePropertiesBase<HandshakeAnnotatePropertiesPass> {

  HandshakeAnnotatePropertiesPass(const std::string &jsonPath = "") {
    this->jsonPath = jsonPath;
    this->uid = 0;
  }

  void runDynamaticPass() override;

private:
  unsigned int uid;
  json::Array propertyTable;

  LogicalResult annotateAbsenceOfBackpressure(ModuleOp modOp);
  LogicalResult annotateValidEquivalence(ModuleOp modOp);
  LogicalResult annotateValidEquivalenceBetweenOps(Operation &op1,
                                                   Operation &op2);
};
} // namespace

LogicalResult
HandshakeAnnotatePropertiesPass::annotateValidEquivalenceBetweenOps(
    Operation &op1, Operation &op2) {
  for (auto res1 : op1.getResults())
    for (auto res2 : op2.getResults()) {
      if (res1 == res2)
        continue;

      ValidEquivalence p(uid, FormalProperty::TAG::OPT, res1, res2);

      propertyTable.push_back(p.toJSON());
      uid++;
    }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateValidEquivalence(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (auto [i, op_i] : llvm::enumerate(funcOp.getOps())) {
      for (auto [j, op_j] : llvm::enumerate(funcOp.getOps())) {
        // equivalence is symmetrical so it needs to be checked only once for
        // each pair of signals (therefore operations)
        if (i <= j && getLogicBB(&op_i) == getLogicBB(&op_i)) {
          if (failed(annotateValidEquivalenceBetweenOps(op_i, op_j))) {
            return failure();
          }
        }
      }
    }
  }
  return success();
}

LogicalResult
HandshakeAnnotatePropertiesPass::annotateAbsenceOfBackpressure(ModuleOp modOp) {
  for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
    for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
      for (auto [resIndex, res] : llvm::enumerate(op.getResults()))
        if (res.getType()
                .isa<handshake::ChannelType, handshake::ControlType>()) {
          if (res.getUsers().empty()) {
            continue;
          }
          auto *userOp = *res.getUsers().begin();

          // skip connections to the output
          if (isa<handshake::EndOp>(userOp))
            continue;

          AbsenceOfBackpressure p(uid, FormalProperty::TAG::OPT, res);

          propertyTable.push_back(p.toJSON());
          uid++;
        }
    }
  }
  return success();
}

void HandshakeAnnotatePropertiesPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  if (failed(annotateAbsenceOfBackpressure(modOp)))
    return signalPassFailure();
  if (failed(annotateValidEquivalence(modOp)))
    return signalPassFailure();

  llvm::json::Value jsonVal(std::move(propertyTable));

  std::error_code EC;
  llvm::raw_fd_ostream jsonOut(jsonPath, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return;

  jsonOut << formatv("{0:2}", jsonVal);
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::formalprop::createAnnotateProperties(
    const std::string &jsonPath) {
  return std::make_unique<HandshakeAnnotatePropertiesPass>(jsonPath);
}