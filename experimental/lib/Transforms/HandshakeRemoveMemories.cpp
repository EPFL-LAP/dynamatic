//===- HandshakeFixArgNames.cpp - Match argument names with C ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the --handshake-remove-memories pass.
//
//===----------------------------------------------------------------------===//

#include "experimental/Transforms/HandshakeRemoveMemories.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <fstream>
#include <iterator>

using namespace mlir;
using namespace circt;
using namespace dynamatic;

namespace {

struct HandshakeRemoveMemoriesPass
    : public dynamatic::experimental::impl::HandshakeRemoveMemoriesBase<
          HandshakeRemoveMemoriesPass> {

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    for (auto funcOp : modOp.getOps<handshake::FuncOp>())
      if (failed(removeMemories(funcOp)))
        return signalPassFailure();
  };

  LogicalResult removeMemories(handshake::FuncOp funcOp);
};

} // namespace

LogicalResult
HandshakeRemoveMemoriesPass::removeMemories(handshake::FuncOp funcOp) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  Block *body = &funcOp.getBody().front();
  BlockArgument startArg = body->getArguments().back();

  SmallVector<Type> argTypes, resTypes;
  SmallVector<Attribute> argNames, resNames;

  // For arguments, take all of them but the start and memories
  for (auto [idx, argType] :
       llvm::enumerate(funcOp.getFunctionType().getInputs())) {
    if (isa<MemRefType>(argType) || idx == funcOp.getNumArguments() - 1)
      continue;
    argTypes.push_back(argType);
    argNames.push_back(funcOp.getArgName(idx));
  }

  // For results, take all of them
  for (auto [idx, resType] :
       llvm::enumerate(funcOp.getFunctionType().getResults())) {
    resTypes.push_back(resType);
    resNames.push_back(funcOp.getResName(idx));
  }

  SmallVector<std::pair<Value, Value>> replacements;

  // Return operands are the same as before + extra
  SmallVector<Value> retOperands;
  auto retOp = *funcOp.getOps<handshake::DynamaticReturnOp>().begin();
  llvm::copy(retOp.getOperands(), std::back_inserter(retOperands));

  // Iterate over all memory operations and record modifications to make
  unsigned memDoneIdx = 0, memCtrlIdx = 0, loadAddrIdx = 0, loadDataIdx = 0,
           storeAddrIdx = 0, storeDataIdx = 0;
  unsigned numIface = 0;
  for (Operation &op : funcOp.getOps()) {
    llvm::TypeSwitch<Operation *, void>(&op)
        .Case<handshake::MemoryOpInterface>([&](handshake::MemoryOpInterface
                                                    ifaceOp) {
          // Count the number of memory interfaces
          numIface++;

          // Create a function argument for the done control
          body->addArgument(builder.getNoneType(), ifaceOp.getLoc());
          replacements.push_back(
              {ifaceOp->getResults().back(), body->getArguments().back()});
          argTypes.push_back(builder.getNoneType());
          argNames.push_back(
              StringAttr::get(ctx, "memDone" + std::to_string(memDoneIdx++)));

          // Create function results for control signals
          FuncMemoryPorts ports = getMemoryPorts(ifaceOp);
          for (GroupMemoryPorts &group : ports.groups) {
            if (group.hasControl()) {
              Value groupCtrl =
                  ifaceOp.getMemOperands()[group.ctrlPort->getCtrlInputIndex()];
              resTypes.push_back(groupCtrl.getType());
              resNames.push_back(StringAttr::get(
                  ctx, "memCtrl" + std::to_string(memCtrlIdx++)));
              retOperands.push_back(groupCtrl);
            }
          }
        })
        .Case<handshake::LoadOpInterface>(
            [&](handshake::LoadOpInterface loadOp) {
              // Create a function result for the addr in from circuit
              Value addrIn = loadOp.getAddressInput();
              resTypes.push_back(addrIn.getType());
              retOperands.push_back(addrIn);
              resNames.push_back(StringAttr::get(
                  ctx, "loadAddrOut" + std::to_string(loadAddrIdx++)));

              // Create a function argument for the data in from MC
              Value dataOut = loadOp.getDataOutput();
              body->addArgument(dataOut.getType(), loadOp.getLoc());
              replacements.push_back({dataOut, body->getArguments().back()});
              argTypes.push_back(dataOut.getType());
              argNames.push_back(StringAttr::get(
                  ctx, "loadDataIn" + std::to_string(loadDataIdx++)));

              // Clear load operands
              loadOp->setOperands({});
            })
        .Case<handshake::StoreOpInterface>(
            [&](handshake::StoreOpInterface storeOp) {
              Value addrIn = storeOp.getAddressInput();
              // Create a function result for the addr in from circuit
              resTypes.push_back(addrIn.getType());
              resNames.push_back(StringAttr::get(
                  ctx, "storeAddrOut" + std::to_string(storeAddrIdx++)));
              retOperands.push_back(addrIn);

              // Create a function result for the data in from circuit
              Value dataIn = storeOp.getDataInput();
              resTypes.push_back(dataIn.getType());
              resNames.push_back(StringAttr::get(
                  ctx, "storeDataOut" + std::to_string(storeDataIdx++)));
              retOperands.push_back(dataIn);

              // Clear store operands
              storeOp->setOperands({});
            });
  }

  // Add last start argument
  argTypes.push_back(builder.getNoneType());
  argNames.push_back(StringAttr::get(ctx, "start"));

  assert(argTypes.size() == argNames.size() && "arg mismatch");
  assert(resTypes.size() == resNames.size() && "res mismatch");
  assert(argTypes.size() != body->getNumArguments() && "block mismatch");

  // llvm::errs() << "Function has " << body->getNumArguments()
  //              << " block arguments\n";

  // llvm::errs() << "Argument names are\n";
  // for (auto name : argNames)
  //   llvm::errs() << "\t" << cast<StringAttr>(name).str() << "\n";
  // llvm::errs() << "Result names are\n";
  // for (auto name : resNames)
  //   llvm::errs() << "\t" << cast<StringAttr>(name).str() << "\n";

  // Replace function type and arguments/results names
  funcOp.setFunctionType(builder.getFunctionType(argTypes, resTypes));
  funcOp->setAttr("argNames", ArrayAttr::get(ctx, argNames));
  funcOp->setAttr("resNames", ArrayAttr::get(ctx, resNames));

  // Create a new ret to replace the old one
  builder.setInsertionPoint(retOp);
  auto newRetOp =
      builder.create<handshake::DynamaticReturnOp>(retOp.getLoc(), retOperands);
  inheritBB(retOp, newRetOp);

  // Replace the end arguments with the new one
  SmallVector<Value> endOperands;
  auto endOp = cast<handshake::EndOp>(funcOp.getBody().front().getTerminator());
  llvm::copy(newRetOp->getResults(), std::back_inserter(endOperands));
  llvm::copy(endOp->getOperands().take_back(numIface),
             std::back_inserter(endOperands));
  endOp->setOperands(endOperands);

  // Delete the original ret
  retOp->erase();

  // Perform all value replacements in the IR
  for (auto [toReplace, replaceWith] : replacements)
    toReplace.replaceAllUsesWith(replaceWith);

  // Now delete all memory operations which should have no uses
  for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
    if (isa<handshake::MemoryOpInterface, handshake::LoadOpInterface,
            handshake::StoreOpInterface>(op))
      op.erase();
  }

  // Add a new start argument at the end of the argument list, to match the
  // function's argument list
  body->addArgument(builder.getNoneType(), funcOp->getLoc());
  startArg.replaceAllUsesWith(body->getArguments().back());

  // Erase memref block arguments and the original start
  body->eraseArguments([&](BlockArgument arg) {
    return isa<MemRefType>(arg.getType()) || arg == startArg;
  });

  // Should be done?
  return success();
}

std::unique_ptr<DynamaticPass>
dynamatic::experimental::createHandshakeRemoveMemories() {
  return std::make_unique<HandshakeRemoveMemoriesPass>();
}
