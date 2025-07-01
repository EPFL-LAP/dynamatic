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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

ChannelInfo::ChannelInfo(Value val, StringRef signalName)
    : channelName(signalName) {

  // Derive the source op's name and result index
  if (auto res = dyn_cast<OpResult>(val)) {
    Operation *producerOp = res.getOwner();
    srcChannelIdx = res.getResultNumber();
    srcOpName = getUniqueName(producerOp);
  } else {
    auto arg = cast<BlockArgument>(val);
    auto funcOp = cast<handshake::FuncOp>(arg.getParentBlock()->getParentOp());
    srcChannelIdx = arg.getArgNumber();
    srcOpName = funcOp.getArgName(arg.getArgNumber()).str();
  }

  // Derive the destination op's name and operand index
  auto oprdIt = val.getUses();
  if (oprdIt.empty() || std::next(oprdIt.begin()) != oprdIt.end()) {
    llvm::report_fatal_error(
        "ChannelInfo can only be constructed from a single use value");
  }

  OpOperand &oprd = *oprdIt.begin();
  Operation *consumerOp = oprd.getOwner();
  if (isa<MemRefType>(oprd.get().getType())) {
    dstChannelIdx = 0; // TODO: Why???
  } else {
    dstChannelIdx = oprd.getOperandNumber();
  }
  if (isa<handshake::EndOp>(consumerOp)) {
    auto funcOp = cast<handshake::FuncOp>(val.getParentBlock()->getParentOp());
    dstOpName = funcOp.getResName(oprd.getOperandNumber()).str();
  } else {
    dstOpName = getUniqueName(consumerOp);
  }
}

LogicalResult CSVBuilder::initialize(mlir::ModuleOp modOp,
                                     bool registerOperands) {
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
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (arg.getType().isa<handshake::ControlType, handshake::ChannelType>())
      channelNameToValue.insert({argNameGen.getInputName(idx), arg});
  }

  // Then associate names to each operation's results
  for (Operation &op : funcOp.getOps()) {
    handshake::PortNamer opNameGen(&op);
    if (registerOperands) {
      // Only needed for SMV
      for (auto [idx, oprd] : llvm::enumerate(op.getOperands())) {
        if (oprd.getType()
                .isa<handshake::ControlType, handshake::ChannelType>()) {
          std::string signalName = getUniqueName(&op).str() + "_" +
                                   opNameGen.getInputName(idx).str();
          channelNameToValue.insert({signalName, oprd});
        }
      }
    }
    for (auto [idx, res] : llvm::enumerate(op.getResults())) {
      if (res.getType().isa<handshake::ControlType, handshake::ChannelType>()) {
        std::string signalName =
            getUniqueName(&op).str() + "_" + opNameGen.getOutputName(idx).str();
        channelNameToValue.insert({signalName, res});
      }
    }
  }

  for (auto &[signalName, val] : channelNameToValue) {
    channelInfos.try_emplace(val, val, signalName);
    states.insert({val, ChannelState::fromValueType(val.getType())});
  }

  return success();
}

void CSVBuilder::writeCSVHeader() const {
  os << "cycle, src_component, src_port, dst_component, dst_port, "
        "state, data\n";
}

void CSVBuilder::commitChannelStateChanges() {
  // Write the channel state changes to the output stream
  for (Value val : toUpdate) {
    const ChannelState &channelState = states.at(val);
    WireState valid = channelState.valid;
    WireState ready = channelState.ready;

    StringRef dataflowState;
    if (valid != WireState::Logic1 && ready == WireState::Logic1)
      dataflowState = "accept";
    else if (valid == WireState::Logic1)
      dataflowState = ready != WireState::Logic1 ? "stall" : "transfer";
    else if (valid == WireState::Logic0 && ready == WireState::Logic0)
      dataflowState = "idle";
    else
      dataflowState = "undefined";

    const ChannelInfo &info = channelInfos.at(val);
    os << cycle << ", " << info.srcOpName << ", " << info.srcChannelIdx << ", "
       << info.dstOpName << ", " << info.dstChannelIdx << ", "
       << dataflowState.str() << ", " << channelState.decodeData() << "\n";
  }

  // Clear the toUpdate set for the next cycle
  toUpdate.clear();
}

void CSVBuilder::updateCycle(size_t newCycle) { cycle = newCycle; }

std::optional<ChannelState> CSVBuilder::getChannelState(Value val) const {
  auto it = states.find(val);
  if (it == states.end())
    return std::nullopt;
  return it->second;
}

void CSVBuilder::updateChannelState(Value val, const ChannelState &newState) {
  auto it = states.find(val);
  if (it == states.end()) {
    states.insert({val, newState});
  } else {
    it->second = newState;
  }
  toUpdate.insert(val);
}

std::optional<Value> CSVBuilder::getChannel(StringRef channelName) const {
  auto it = channelNameToValue.find(channelName);
  if (it == channelNameToValue.end())
    return std::nullopt;
  return it->second;
}
