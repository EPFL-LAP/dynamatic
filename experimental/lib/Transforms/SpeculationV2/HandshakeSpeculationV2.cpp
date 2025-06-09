//===- HandshakeSpeculationV2.cpp - Speculative Dataflows -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Placement of Speculation components to enable speculative execution.
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/SpeculationV2/HandshakeSpeculationV2.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculationv2;

namespace {

struct HandshakeSpeculationV2Pass
    : public dynamatic::experimental::speculationv2::impl::
          HandshakeSpeculationV2Base<HandshakeSpeculationV2Pass> {
  HandshakeSpeculationV2Pass() {}

  std::optional<Value> loopSuppressCtrl;
  std::optional<Value> exitSuppressCtrl;
  std::optional<Value> storeSuppressCtrl;

  void placeSpeculator(FuncOp &funcOp, unsigned preBB, unsigned specBB,
                       unsigned postBB);

  void runDynamaticPass() override;
};
} // namespace

void HandshakeSpeculationV2Pass::placeSpeculator(FuncOp &funcOp, unsigned preBB,
                                                 unsigned specBB,
                                                 unsigned postBB) {

  ConditionalBranchOp condBrOp = nullptr;
  for (auto condBrCandidate : funcOp.getOps<ConditionalBranchOp>()) {
    auto condBB = getLogicBB(condBrCandidate);
    if (condBB && *condBB == specBB) {
      // Found the condBr in the specBB
      condBrOp = condBrCandidate;
      break;
    }
  }
  assert(condBrOp && "Could not find any ConditionalBranchOp");

  OpBuilder builder(funcOp->getContext());
  builder.setInsertionPoint(condBrOp);

  Value actualCondition = condBrOp.getConditionOperand();
  Location specLoc = actualCondition.getLoc();
  Type conditionType = actualCondition.getType();

  // Append SuppressControl
  BackedgeBuilder backedgeBuilder(builder, specLoc);
  Backedge generatedConditionBackedge = backedgeBuilder.get(conditionType);

  SpecV2SuppressControlOp specSuppressCtrlOp =
      builder.create<SpecV2SuppressControlOp>(specLoc, actualCondition,
                                              generatedConditionBackedge);

  SuppressOp loopConditionSuppressor = builder.create<SuppressOp>(
      specLoc, actualCondition, specSuppressCtrlOp.getCtrl());

  NotOp notCondition =
      builder.create<NotOp>(specLoc, loopConditionSuppressor.getResult());

  SourceOp conditionGenerator = builder.create<SourceOp>(specLoc);
  ConstantOp conditionConstant =
      builder.create<ConstantOp>(specLoc, IntegerAttr::get(conditionType, 0),
                                 conditionGenerator.getResult());

  MergeOp merge = builder.create<MergeOp>(
      specLoc, llvm::ArrayRef<Value>{conditionConstant.getResult(),
                                     notCondition.getResult()});
  generatedConditionBackedge.setValue(merge.getResult());

  OrIOp orCondition = builder.create<OrIOp>(specLoc, actualCondition,
                                            specSuppressCtrlOp.getCtrl());

  loopSuppressCtrl = merge.getResult();
  exitSuppressCtrl = orCondition.getResult();
  storeSuppressCtrl = specSuppressCtrlOp.getCtrl();
}

void HandshakeSpeculationV2Pass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  // NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();

  placeSpeculator(funcOp, 0, 1, 2);

  // if (failed(eraseUnusedControlNetwork(funcOp, 1)))
  //   return signalPassFailure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::experimental::speculationv2::createHandshakeSpeculationV2() {
  return std::make_unique<HandshakeSpeculationV2Pass>();
}
