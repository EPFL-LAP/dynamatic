//===- VisualizerSupport.cpp - Utilities for vizualizer csv------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to generate csv files used by the visualizer.
//
//===----------------------------------------------------------------------===//

#include "VisualizerSupport.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

SignalInfo::SignalInfo(Value val, StringRef signalName)
    : signalName(signalName) {
  // Derive the source component's name and ID
  if (auto res = dyn_cast<OpResult>(val)) {
    Operation *producerOp = res.getOwner();
    srcPortID = res.getResultNumber();
    srcComponent = getUniqueName(producerOp);
  } else {
    auto arg = cast<BlockArgument>(val);
    auto funcOp = cast<handshake::FuncOp>(arg.getParentBlock()->getParentOp());
    srcPortID = arg.getArgNumber();
    srcComponent = funcOp.getArgName(arg.getArgNumber()).str();
  }

  // Derive the destination component's name and ID
  OpOperand &oprd = *val.getUses().begin();
  Operation *consumerOp = oprd.getOwner();
  if (!isa<MemRefType>(oprd.get().getType()))
    dstPortID = oprd.getOperandNumber();
  if (isa<handshake::EndOp>(consumerOp)) {
    auto funcOp = cast<handshake::FuncOp>(val.getParentBlock()->getParentOp());
    dstComponent = funcOp.getResName(oprd.getOperandNumber()).str();
  } else {
    dstComponent = getUniqueName(consumerOp);
  }
}

LogicalResult mapSignalsToValues(mlir::ModuleOp modOp,
                                 llvm::StringMap<Value> &ports,
                                 bool mapOperands) {
  // Extract the non-external Handshake function from the module
  handshake::FuncOp funcOp = nullptr;
  for (auto op : modOp.getOps<handshake::FuncOp>()) {
    if (op.isExternal())
      continue;
    if (funcOp) {
      return modOp.emitError() << "we currently only support one non-external "
                                  "handshake function per module";
    }
    funcOp = op;
  }
  if (!funcOp)
    return modOp.emitError() << "No Handshake function in input module";

  // Make sure all operations inside the function have names
  NameAnalysis namer(funcOp);
  if (!namer.areAllOpsNamed()) {
    return funcOp.emitError() << "Not all operations in the function have "
                                 "names, this is a requirement";
  }

  // First associate names to all function arguments
  handshake::PortNamer argNameGen(funcOp);
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments()))
    ports.insert({argNameGen.getInputName(idx), arg});

  // Then associate names to each operation's results
  for (Operation &op : funcOp.getOps()) {
    handshake::PortNamer opNameGen(&op);
    if (mapOperands) {
      // Only needed for SMV
      for (auto [idx, oprd] : llvm::enumerate(op.getOperands())) {
        std::string signalName =
            getUniqueName(&op).str() + "_" + opNameGen.getInputName(idx).str();
        ports.insert({signalName, oprd});
      }
    }
    for (auto [idx, res] : llvm::enumerate(op.getResults())) {
      std::string signalName =
          getUniqueName(&op).str() + "_" + opNameGen.getOutputName(idx).str();
      ports.insert({signalName, res});
    }
  }

  return success();
}

void writeCSVHeader(llvm::raw_ostream &os) {
  os << "cycle, src_component, src_port, dst_component, dst_port, "
        "state, data\n";
}

void writeChannelStateChanges(
    llvm::raw_ostream &os, size_t cycle, const mlir::DenseSet<Value> &toUpdate,
    const mlir::DenseMap<Value, ChannelState> &state,
    const mlir::DenseMap<Value, SignalInfo> &valueToSignalInfo) {
  for (Value val : toUpdate) {
    const ChannelState &channelState = state.at(val);
    WireState valid = channelState.valid;
    WireState ready = channelState.ready;

    StringRef dataflowState;
    if (valid != WireState::LOGIC_1 && ready == WireState::LOGIC_1)
      dataflowState = ACCEPT;
    else if (valid == WireState::LOGIC_1)
      dataflowState = ready != WireState::LOGIC_1 ? STALL : TRANSFER;
    else if (valid == WireState::LOGIC_0 && ready == WireState::LOGIC_0)
      dataflowState = IDLE;
    else
      dataflowState = UNDEFINED;

    const SignalInfo &info = valueToSignalInfo.at(val);
    llvm::outs() << cycle << ", " << info.srcComponent << ", " << info.srcPortID
                 << ", " << info.dstComponent << ", " << info.dstPortID << ", "
                 << dataflowState.str() << ", " << channelState.decodeData()
                 << "\n";
  }
}
