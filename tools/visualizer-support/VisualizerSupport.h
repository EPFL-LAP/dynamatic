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
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <optional>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

enum class WireState { Undefined, Logic0, Logic1 };

struct ChannelInfo {
  std::string channelName;
  std::string srcOpName;
  unsigned srcChannelIdx;
  std::string dstOpName;
  unsigned dstChannelIdx;

  ChannelInfo(Value val, StringRef channelName);
};

enum class VisualizerChannelType { SignedInt, UnsignedInt, Float, Control };

struct ChannelState {
  WireState valid = WireState::Undefined;
  WireState ready = WireState::Undefined;

  VisualizerChannelType type;
  unsigned bitWidth;
  std::vector<WireState> data;

  ChannelState(VisualizerChannelType type, unsigned bitWidth)
      : type(type), bitWidth(bitWidth),
        data(std::vector{bitWidth, WireState::Undefined}) {}

  static ChannelState fromValueType(Type valueType) {
    if (auto channelType = dyn_cast<handshake::ChannelType>(valueType)) {
      Type dataType = channelType.getDataType();
      unsigned bitWidth = dataType.getIntOrFloatBitWidth();
      if (auto intDataType = dyn_cast<IntegerType>(dataType)) {
        if (intDataType.isSigned()) {
          return ChannelState(VisualizerChannelType::SignedInt, bitWidth);
        }
        // Treat sign-less integers (e.g., i32) also as unsigned integers
        return ChannelState(VisualizerChannelType::UnsignedInt, bitWidth);
      }
      if (dataType.isa<FloatType>()) {
        return ChannelState(VisualizerChannelType::Float, bitWidth);
      }
      dataType.dump();
      llvm_unreachable("Unsupported data type for ChannelState");
    }
    if (auto controlType = dyn_cast<handshake::ControlType>(valueType)) {
      return ChannelState(VisualizerChannelType::Control, 1);
    }
    valueType.dump();
    llvm_unreachable("Unsupported value type for ChannelState");
  }

  std::string decodeData() const {
    if (type == VisualizerChannelType::Control) {
      return "";
    }

    // Build a string from the vector of bits
    std::string dataString;
    for (WireState bit : llvm::reverse(data)) {
      switch (bit) {
      case WireState::Undefined:
        return "u";
      case WireState::Logic0:
        dataString.push_back('0');
        break;
      case WireState::Logic1:
        dataString.push_back('1');
        break;
      }
    }
    APInt intVal(bitWidth, dataString, 2);

    if (type == VisualizerChannelType::SignedInt) {
      return std::to_string(intVal.getSExtValue());
    }
    if (type == VisualizerChannelType::UnsignedInt) {
      APInt intVal(data.size(), dataString, 2);
      return std::to_string(intVal.getZExtValue());
    }
    assert(type == VisualizerChannelType::Float &&
           "Unsupported channel type for data decoding");
    if (bitWidth == 32) {
      // Convert to float
      APFloat floatVal(APFloat::IEEEsingle(), intVal);
      return std::to_string(floatVal.convertToFloat());
    }
    if (bitWidth == 64) {
      // Convert to double
      APFloat doubleVal(APFloat::IEEEdouble(), intVal);
      return std::to_string(doubleVal.convertToDouble());
    }
    llvm::errs() << "Unsupported bit width for float conversion: " << bitWidth
                 << "\n";
    llvm_unreachable("Unsupported bit width for float conversion");
  }
};

struct WireReference {
  Value value;
  SignalType signalType;
  /// In ModelSim, each bit of the data vector is reported independently.
  /// idx indicates the index of the bit in the data vector.
  std::optional<size_t> idx;
};

struct CSVBuilder {
  CSVBuilder(llvm::raw_ostream &os)
      : os(os), channelNameToValue({}), toUpdate({}), states({}),
        channelInfos({}) {}

  LogicalResult initialize(mlir::ModuleOp modOp, bool registerOperands = false);
  void writeCSVHeader() const;
  void commitChannelStateChanges();
  void updateCycle(size_t newCycle);
  std::optional<ChannelState> getChannelState(Value val) const;
  void updateChannelState(Value val, const ChannelState &newState);
  std::optional<Value> getChannel(StringRef channelName) const;

private:
  llvm::raw_ostream &os;
  size_t cycle = 0;
  llvm::StringMap<Value> channelNameToValue;
  mlir::DenseSet<Value> toUpdate;
  mlir::DenseMap<Value, ChannelState> states;
  mlir::DenseMap<Value, ChannelInfo> channelInfos;
};

#endif // VISUALIZER_SUPPORT_H
