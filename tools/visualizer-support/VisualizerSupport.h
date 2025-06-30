//===- VisualizerSupport.h - Utilities for vizualizer csv--------*- C++ -*-===//
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

#ifndef VISUALIZER_SUPPORT_H
#define VISUALIZER_SUPPORT_H

#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

enum class WireState { UNDEFINED, LOGIC_0, LOGIC_1 };

struct SignalInfo {
  std::string signalName;
  std::string srcComponent;
  unsigned srcPortID;
  std::string dstComponent;
  unsigned dstPortID;

  SignalInfo(Value val, StringRef signalName);
};

struct ChannelState {
  Type type;

  WireState valid = WireState::UNDEFINED;
  WireState ready = WireState::UNDEFINED;
  std::vector<WireState> data;

  ChannelState(Type type)
      : type(type), data(std::vector{handshake::getHandshakeTypeBitWidth(type),
                                     WireState::UNDEFINED}) {}

  std::string decodeData() const {
    if (data.empty())
      return "";

    // Build a string from the vector of bits
    std::string dataString;
    bool partlyUndef = false;
    for (WireState bit : llvm::reverse(data)) {
      switch (bit) {
      case WireState::UNDEFINED:
        dataString.push_back('u');
        partlyUndef = true;
        break;
      case WireState::LOGIC_0:
        dataString.push_back('0');
        break;
      case WireState::LOGIC_1:
        dataString.push_back('1');
        break;
      }
    }
    if (partlyUndef)
      return dataString;

    auto channelType = cast<handshake::ChannelType>(type);
    auto dataType = channelType.getDataType();
    if (auto intType = dyn_cast<IntegerType>(dataType)) {
      APInt intVal(data.size(), dataString, 2);
      if (intType.isSigned())
        return std::to_string(intVal.getSExtValue());
      return std::to_string(intVal.getZExtValue());
    }
    if (auto floatType = dyn_cast<FloatType>(dataType)) {
      APFloat floatVal(floatType.getFloatSemantics(), dataString);
      return std::to_string(floatVal.convertToDouble());
    }
    return "undef";
  }
};

struct WireReference {
  static constexpr StringLiteral VALID_SUFFIX = "_valid",
                                 READY_SUFFIX = "_ready";

  Value value;
  SignalType signalType;
  std::optional<size_t> idx;
};

LogicalResult mapSignalsToValues(mlir::ModuleOp modOp,
                                 llvm::StringMap<Value> &ports,
                                 bool mapOperands = false);

static constexpr StringLiteral ACCEPT("accept"), STALL("stall"),
    TRANSFER("transfer"), IDLE("idle"), UNDEFINED("undefined");

void writeCSVHeader(llvm::raw_ostream &os);
void writeChannelStateChanges(
    llvm::raw_ostream &os, size_t cycle, const mlir::DenseSet<Value> &toUpdate,
    const mlir::DenseMap<Value, ChannelState> &state,
    const mlir::DenseMap<Value, SignalInfo> &valueToSignalInfo);

#endif // VISUALIZER_SUPPORT_H
