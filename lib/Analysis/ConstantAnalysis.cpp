//===- ConstantAnalysis.cpp - Constant analyis utilities --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of Handshake constant analysis infrastructure.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/ConstantAnalysis.h"
#include "dynamatic/Support/LogicBB.h"

using namespace circt;
using namespace dynamatic;

/// Determines whether the control value of two constants can be considered
/// equivalent.
static bool areCstCtrlEquivalent(Value ctrl, Value otherCtrl) {
  if (ctrl == otherCtrl)
    return true;

  // Both controls are equivalent if they originate from sources in the same
  // block
  Operation *defOp = ctrl.getDefiningOp();
  if (!defOp || !isa<handshake::SourceOp>(defOp))
    return false;
  Operation *otherDefOp = otherCtrl.getDefiningOp();
  if (!otherDefOp || !isa<handshake::SourceOp>(otherDefOp))
    return false;
  std::optional<unsigned> block = getLogicBB(defOp);
  std::optional<unsigned> otherBlock = getLogicBB(otherDefOp);
  return block.has_value() && otherBlock.has_value() &&
         block.value() == otherBlock.value();
}

handshake::ConstantOp
dynamatic::findEquivalentCst(handshake::ConstantOp cstOp) {
  auto cstAttr = cstOp.getValue();
  auto funcOp = cstOp->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "constant should have parent function");

  for (auto otherCstOp : funcOp.getOps<handshake::ConstantOp>()) {
    // Don't match ourself
    if (cstOp == otherCstOp)
      continue;

    // The constant operation needs to have the same value attribute and the
    // same control
    auto otherCstAttr = otherCstOp.getValue();
    if (otherCstAttr == cstAttr &&
        areCstCtrlEquivalent(cstOp.getCtrl(), otherCstOp.getCtrl()))
      return otherCstOp;
  }

  return nullptr;
}

/// Determines whether a user of a constant makes the constant un-sourcable.
/// NOTE: (lucas) I doubt this works in half-degenerate cases, but this is the
/// logic that legacy Dynamatic follows.
static bool cstUserIsSourcable(Operation *cstUser) {
  return !isa<handshake::BranchOp, handshake::ConditionalBranchOp,
              handshake::ReturnOp, handshake::LoadOpInterface,
              handshake::StoreOpInterface>(cstUser);
}

bool dynamatic::isCstSourcable(mlir::arith::ConstantOp cstOp) {
  return llvm::all_of(cstOp->getUsers(), cstUserIsSourcable);
}
