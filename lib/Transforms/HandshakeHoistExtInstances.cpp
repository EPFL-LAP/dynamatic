//===- HandshakeHoistExtInstances.cpp - Instances to IO ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass uses an `OpBuilder` to hoist external functions instances out of
// each internal Handshake function, resulting in the latter's gaining extra
// arguments (mapping to the instance's results) and results (mapping to the
// instance's arguments). External Handhsake functions with no uses after this
// process are removed from the IR.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/HandshakeHoistExtInstances.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <iterator>

using namespace mlir;
using namespace dynamatic;

static bool isIntrinsic(handshake::FuncOp funcOp) {
  return funcOp.getNameAttr().strref().starts_with("__");
}

namespace {

/// Simple pass driver for the external instance hoisitng pass.
struct HandshakeHoistExtInstancesPass
    : public dynamatic::impl::HandshakeHoistExtInstancesBase<
          HandshakeHoistExtInstancesPass> {

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();

    // Make sure all operations are named
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    namer.nameAllUnnamedOps();

    // Hoist external instances from each internal Handshake function
    SymbolTable symbols(modOp);
    auto funcOps = modOp.getOps<handshake::FuncOp>();
    for (auto funcOp : llvm::make_early_inc_range(funcOps)) {
      if (!funcOp.isExternal())
        if (failed(hoistInstances(funcOp, symbols)))
          return signalPassFailure();
    }

    // Remove unused external functions
    funcOps = modOp.getOps<handshake::FuncOp>();
    for (auto funcOp : llvm::make_early_inc_range(funcOps)) {
      if (funcOp.isExternal())
        eraseIfUnused(funcOp);
    }
  }

private:
  /// If the function contains any instance of an external Handshake function,
  /// hoist it "outside" the function and add arguments/results to the
  /// function's signature to represent the removed instance's
  /// results/arguments, respectively.
  LogicalResult hoistInstances(handshake::FuncOp funcOp, SymbolTable &symbols);

  /// Erases the external function if is never referenced elsewhere in the IR.
  void eraseIfUnused(handshake::FuncOp funcOp);
};
} // namespace

LogicalResult
HandshakeHoistExtInstancesPass::hoistInstances(handshake::FuncOp funcOp,
                                               SymbolTable &symbols) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // First function arguments will stay the same
  SmallVector<Type, 16> argTypes(funcOp.getArgumentTypes());
  SmallVector<Attribute> argNames(funcOp.getArgNames().getValue());

  // First function results will stay the same
  SmallVector<Type, 16> resTypes(funcOp.getResultTypes());
  SmallVector<Attribute> resNames(funcOp.getResNames().getValue());

  // First end operands will stay the same
  auto endOp = cast<handshake::EndOp>(funcOp.getBodyBlock()->getTerminator());
  SmallVector<Value> endOperands(endOp->getOperands());

  Block *bodyBlock = funcOp.getBodyBlock();

  // Verify that each external function is instantiated a single time
  llvm::SmallSet<handshake::FuncOp, 4> calledFunctions;
  // Collect all instances inside the function that reference an external
  // Handshake functions
  bool anyInstance = false;
  auto instOps = funcOp.getOps<handshake::InstanceOp>();
  for (auto instOp : llvm::make_early_inc_range(instOps)) {
    auto instFuncOp = symbols.lookup<handshake::FuncOp>(instOp.getModule());
    // Only replace instances of non-intrinsic external functions
    if (!instFuncOp.isExternal() || isIntrinsic(instFuncOp))
      continue;

    anyInstance = true;
    StringRef instFuncName = instFuncOp.getNameAttr().strref();

    if (auto [_, newFunc] = calledFunctions.insert(instFuncOp); !newFunc) {
      return instFuncOp.emitError() << "External function is instantiated "
                                    << "multiple times, but we only support "
                                       "a single instantation in any "
                                    << "given kernel";
    }

    // Iterate over the instance's arguments and add them to the function's
    // results
    auto namedArguments =
        llvm::zip_equal(instFuncOp.getArgNames(), instOp.getOperandTypes());
    for (auto [argNameAttr, argType] : namedArguments) {
      StringRef argName = argNameAttr.cast<StringAttr>().strref();
      resTypes.push_back(argType);
      resNames.push_back(StringAttr::get(ctx, instFuncName + "_" + argName));
    }

    // Iterate over the instance's results and add them to the function's
    // arguments
    auto namedResults =
        llvm::zip_equal(instFuncOp.getResNames(), instOp.getResultTypes());
    for (auto [argNameAttr, resType] : namedResults) {
      StringRef argName = argNameAttr.cast<StringAttr>().strref();
      argTypes.push_back(resType);
      argNames.push_back(StringAttr::get(ctx, instFuncName + "_" + argName));
    }

    // Instance arguments will exit the function through the end terminator
    llvm::copy(instOp.getOperands(), std::back_inserter(endOperands));

    // Instance results will come from the function's arguments
    size_t numResults = instOp.getNumResults();
    SmallVector<Location> locs(numResults, instOp.getLoc());
    bodyBlock->addArguments(instOp->getResultTypes(), locs);
    instOp->replaceAllUsesWith(bodyBlock->getArguments().take_back(numResults));
    instOp->erase();
  }

  if (!anyInstance)
    return success();

  // Change the function's signature
  funcOp.setFunctionType(builder.getFunctionType(argTypes, resTypes));
  funcOp->setAttr("argNames", ArrayAttr::get(ctx, argNames));
  funcOp->setAttr("resNames", ArrayAttr::get(ctx, resNames));

  // Replace the terminator's operands
  endOp->setOperands(endOperands);
  return success();
}

void HandshakeHoistExtInstancesPass::eraseIfUnused(handshake::FuncOp funcOp) {
  if (funcOp.getSymbolUses(funcOp->getParentOfType<mlir::ModuleOp>())
          ->empty()) {
    funcOp->erase();
  }
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeHoistExtInstances() {
  return std::make_unique<HandshakeHoistExtInstancesPass>();
}
