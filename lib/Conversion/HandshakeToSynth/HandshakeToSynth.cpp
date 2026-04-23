//===- HandshakeToSynth.cpp - Convert Handshake to Synth --------------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --handshake-to-synth conversion pass in three steps:
//
//   Step 1 - Unbundle: convert every Handshake op into an hw::HWModuleOp
//            whose ports are flat integer signals (data/valid/ready split). The
//            ready is inverted compared to the handshake port. The data signals
//            are split into individual bits. The clock and reset signals are
//            added. In the body of the hw::HWModuleOp, we insert a
//            synth::SubcktOp as a placeholder.
//
//   Step 2 - Populate: replace each hw::HWModuleOp's placeholder body with
//            the actual gate-level netlist imported from the BLIF file whose
//            path was recorded during Step 1.
//
//===----------------------------------------------------------------------===//

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Conversion/Passes.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/Synth/SynthOps.h"
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKETOSYNTH
#include "dynamatic/Conversion/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/HW/HWTypes.h"
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/BLIFFileManager.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/BlifImporter/BlifImporterSupport.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/RTL/RTL.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
#include <string>
#include <utility>

// ===----------------------------------------------------------------------===//
// Class, struct and function definitions for the Handshake to Synth conversion
// ===----------------------------------------------------------------------===//

namespace dynamatic {

//===----------------------------------------------------------------------===//
// Pass-wide shared state
//===----------------------------------------------------------------------===//

/// Maps each hw::HWModuleOp (by Operation*) to the BLIF file path it should
/// be populated from.  An empty string means the module needs no replacement.
/// Written during Step 1 (unbundling) and consumed during Step 2 (populate).
static mlir::DenseMap<mlir::Operation *, std::string> opToBlifPathMap;

//===----------------------------------------------------------------------===//
// Step 1: Unbundle handshake channels into individual bits and create
// placeholder hw modules
//===----------------------------------------------------------------------===//

class HandshakeUnitPortInfo {
public:
  enum class PortType { DATA, VALID, READY };

  HandshakeUnitPortInfo(std::string name, hw::ModulePort::Direction direction,
                        Value handshakeSignal, PortType portType)
      : name(std::move(name)), direction(direction),
        handshakeSignal(handshakeSignal), portType(portType) {}

  virtual ~HandshakeUnitPortInfo() = default;

  const std::string &getName() const { return name; }
  hw::ModulePort::Direction getDirection() const { return direction; }
  Value getHandshakeSignal() const { return handshakeSignal; }
  PortType getPortType() const { return portType; }

  bool isData() const { return portType == PortType::DATA; }
  bool isValid() const { return portType == PortType::VALID; }
  bool isReady() const { return portType == PortType::READY; }

private:
  std::string name;
  hw::ModulePort::Direction direction;
  Value handshakeSignal;
  PortType portType;
};

// Represents a single unbundled bit of a data signal
class DataPortInfo : public HandshakeUnitPortInfo {
public:
  DataPortInfo(std::string name, hw::ModulePort::Direction direction,
               Value handshakeSignal, unsigned bitIndex, unsigned totalBits)
      : HandshakeUnitPortInfo(std::move(name), direction, handshakeSignal,
                              PortType::DATA),
        bitIndex(bitIndex), totalBits(totalBits) {}

  unsigned getBitIndex() const { return bitIndex; }
  unsigned getTotalBits() const { return totalBits; }

private:
  unsigned bitIndex;
  unsigned totalBits;
};

// Represents a valid signal
class ValidPortInfo : public HandshakeUnitPortInfo {
public:
  ValidPortInfo(std::string name, hw::ModulePort::Direction direction,
                Value handshakeSignal)
      : HandshakeUnitPortInfo(std::move(name), direction, handshakeSignal,
                              PortType::VALID) {}
};

// Represents a ready signal
class ReadyPortInfo : public HandshakeUnitPortInfo {
public:
  ReadyPortInfo(std::string name, hw::ModulePort::Direction direction,
                Value handshakeSignal)
      : HandshakeUnitPortInfo(std::move(name), direction, handshakeSignal,
                              PortType::READY) {}
};

using HandshakeUnitPortList =
    SmallVector<std::unique_ptr<HandshakeUnitPortInfo>>;

// Struct to hold the unbundled values for a handshake channel
struct UnbundledHandshakeChannel {
  SmallVector<Value> dataBits;
  Value valid;
  Value ready;

  bool empty() const { return dataBits.empty() && !valid && !ready; }

  // Function to set the values in the tuple based on the port kind
  void setValues(const HandshakeUnitPortInfo *portInfo,
                 SmallVector<Value> newValues) {
    switch (portInfo->getPortType()) {
    case HandshakeUnitPortInfo::PortType::DATA:
      dataBits = std::move(newValues);
      break;
    case HandshakeUnitPortInfo::PortType::VALID:
      valid = newValues.empty() ? Value() : newValues[0];
      break;
    case HandshakeUnitPortInfo::PortType::READY:
      ready = newValues.empty() ? Value() : newValues[0];
      break;
    }
  }

  // Function to get the values from the tuple based on the port kind
  SmallVector<Value> getValues(const HandshakeUnitPortInfo *portInfo) const {
    switch (portInfo->getPortType()) {
    case HandshakeUnitPortInfo::PortType::DATA:
      return dataBits;
    case HandshakeUnitPortInfo::PortType::VALID:
      return valid ? SmallVector<Value>{valid} : SmallVector<Value>{};
    case HandshakeUnitPortInfo::PortType::READY:
      return ready ? SmallVector<Value>{ready} : SmallVector<Value>{};
    }
    return {};
  }

  void clearValues(const HandshakeUnitPortInfo *portInfo) {
    setValues(portInfo, {});
  }
};

// Class that controls the unbundling of handshake channel types into integer
// types
class HandshakeUnbundler {
public:
  HandshakeUnbundler(ModuleOp modOp)
      : modOp(modOp), symTable(modOp), builder(modOp) {}

  // Top-level entry point for unbundling funcOp and all ops inside it
  mlir::LogicalResult unbundleHandshakeChannels();

private:
  // Converts one Handshake op inside the funcop into an hw instance inside the
  // top function
  mlir::LogicalResult convertHandshakeOp(Operation *op);

  // Converts one Handshake function into an hw module
  mlir::LogicalResult convertHandshakeFunc();

  // Function to create hw module from a handshake operation
  hw::HWModuleOp createHWModuleHandshakeOp(StringRef moduleName,
                                           Operation *handshakeOp,
                                           MLIRContext *ctx);

  // Function that returns unbundled values from a channel value. If the channel
  // value has not been unbundled yet, creates backedge placeholders for each
  // bit and saves them.
  llvm::SmallVector<Value>
  findOrCreateUnbundledValues(unsigned totalBits,
                              HandshakeUnitPortInfo *portKind, Location loc);

  // Function that saves the mapping from a channel value to its unbundled bit
  // values. If there were placeholders for the channel value, replaces them
  // with the real bit values.
  void resolveUnbundledValues(HandshakeUnitPortInfo *portKind,
                              llvm::SmallVector<Value> unbundledValues);

  // Module op
  ModuleOp modOp;
  handshake::FuncOp topFunction;
  // Symbol table for looking up functions and modules
  SymbolTable symTable;
  // Builder for creating new operations
  OpBuilder builder;
  // Top HW Module
  hw::HWModuleOp topHWModule;

  // Maps handshake channel values to their unbundled bit values. The tuple is
  // data bits, valid bit, ready bit
  llvm::DenseMap<Value, UnbundledHandshakeChannel> unbundledValuesMap;
  // Maps handshake channel values to any placeholder backedges created for
  // their unbundled bits. The tuple is data bits, valid bit, ready bit
  llvm::DenseMap<Value, UnbundledHandshakeChannel> pendingValuesMap;

  // clk and rst values from the top module
  Value clk;
  Value rst;

  // Backedge builder for creating placeholders for unbundled bits
  std::unique_ptr<BackedgeBuilder> backedgeBuilder;
};

} // namespace dynamatic

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace dynamatic {

//===----------------------------------------------------------------------===//
// Step 1: Unbundle handshake channels into individual bits and create
// placeholder hw modules
//===----------------------------------------------------------------------===//

// Function to unbundle a handshake port into its constituent signals (ready,
// valid, data)
HandshakeUnitPortList
unbundleChannel(const std::string &handshakePortName,
                hw::ModulePort::Direction handshakePortDir,
                Value handshakePort) {

  HandshakeUnitPortList unbundledPorts;
  Type type = handshakePort.getType();

  // Flipped direction: input becomes output and vice versa.
  auto flip = [](hw::ModulePort::Direction d) {
    return d == hw::ModulePort::Direction::Input
               ? hw::ModulePort::Direction::Output
               : hw::ModulePort::Direction::Input;
  };

  // Depending on the type, the unbundling will be different
  if (auto channel = dyn_cast<handshake::ChannelType>(type)) {
    // For a channel type, we unbundle into ready, valid, and data ports
    unsigned dataWidth = channel.getDataType().cast<IntegerType>().getWidth();
    for (unsigned bit = 0; bit < dataWidth; ++bit) {
      unbundledPorts.push_back(std::make_unique<DataPortInfo>(
          legalizeDataPortName(handshakePortName, bit, dataWidth),
          handshakePortDir, handshakePort, bit, dataWidth));
    }
    // valid: keep direction, always i1
    unbundledPorts.push_back(std::make_unique<ValidPortInfo>(
        legalizeControlPortName(handshakePortName, "_valid"), handshakePortDir,
        handshakePort));
    // ready: flip direction, always i1
    unbundledPorts.push_back(std::make_unique<ReadyPortInfo>(
        legalizeControlPortName(handshakePortName, "_ready"),
        flip(handshakePortDir), handshakePort));
  } else if (isa<handshake::ControlType>(type)) {
    // valid: keep direction, always i1
    unbundledPorts.push_back(std::make_unique<ValidPortInfo>(
        legalizeControlPortName(handshakePortName, "_valid"), handshakePortDir,
        handshakePort));
    // ready: flip direction, always i1
    unbundledPorts.push_back(std::make_unique<ReadyPortInfo>(
        legalizeControlPortName(handshakePortName, "_ready"),
        flip(handshakePortDir), handshakePort));
  } else if (auto mem = dyn_cast<MemRefType>(type)) {
    unsigned dataWidth = mem.getElementType().cast<IntegerType>().getWidth();
    for (unsigned bit = 0; bit < dataWidth; ++bit) {
      unbundledPorts.push_back(std::make_unique<DataPortInfo>(
          legalizeDataPortName(handshakePortName, bit, dataWidth),
          handshakePortDir, handshakePort, bit, dataWidth));
    }
    // valid: keep direction, always i1
    unbundledPorts.push_back(std::make_unique<ValidPortInfo>(
        legalizeControlPortName(handshakePortName, "_valid"), handshakePortDir,
        handshakePort));
    // ready: flip direction, always i1
    unbundledPorts.push_back(std::make_unique<ReadyPortInfo>(
        legalizeControlPortName(handshakePortName, "_ready"),
        flip(handshakePortDir), handshakePort));
  } else {
    // Pass-through: split multi-bit integer types into i1 ports.
    if (auto intTy = dyn_cast<IntegerType>(type)) {
      unsigned width = intTy.getWidth();
      for (unsigned b = 0; b < width; ++b)
        unbundledPorts.push_back(std::make_unique<DataPortInfo>(
            legalizeDataPortName(handshakePortName, b, width), handshakePortDir,
            handshakePort, b, width));
    } else {
      unbundledPorts.push_back(std::make_unique<ValidPortInfo>(
          legalizeControlPortName(handshakePortName, "_valid"),
          handshakePortDir, handshakePort));
    }
  }
  return unbundledPorts;
}

// Function to unbundle handshake channels of an handshake op
static HandshakeUnitPortList unbundlePorts(Operation *handshakeOp,
                                           MLIRContext *ctx) {

  HandshakeUnitPortList unbundledPorts;

  // First, we have to derive the base names from NamedIOInterface
  SmallVector<std::string> handshakeInPortNames, handshakeOutPortNames;
  // Check if it is a funcOp
  if (auto funcOp = dyn_cast<handshake::FuncOp>(handshakeOp)) {
    // Extract the port names from the function argument
    for (auto [i, argNameAttr] : llvm::enumerate(funcOp.getArgNames()))
      handshakeInPortNames.push_back(argNameAttr.dyn_cast<StringAttr>().data());
    // Extract the port names from the function results
    for (auto [i, resNameAttr] : llvm::enumerate(funcOp.getResNames()))
      handshakeOutPortNames.push_back(
          resNameAttr.dyn_cast<StringAttr>().data());
  } else {
    // For other ops we use the op name as base for the port names
    // We have to fix the names of the ports so that the root_N pattern is
    // converted to root[N] to be compatible with the BLIF importer
    auto namedIO = dyn_cast<handshake::NamedIOInterface>(handshakeOp);
    assert(namedIO && "op must implement NamedIOInterface");

    // Iterate over the operand and result names
    for (auto [i, _] : llvm::enumerate(handshakeOp->getOperands()))
      handshakeInPortNames.push_back(namedIO.getOperandName(i));
    for (auto [i, _] : llvm::enumerate(handshakeOp->getResults()))
      handshakeOutPortNames.push_back(namedIO.getResultName(i));
    // Legalize the port names following BLIF expected format
    legalizeBlifPortNames(handshakeInPortNames);
    legalizeBlifPortNames(handshakeOutPortNames);
  }

  // Second, we use the base names to unbundle the ports and build the
  // hw::ModulePortInfo
  // Check if the op is a funcOp
  if (auto funcOp = dyn_cast<handshake::FuncOp>(handshakeOp)) {
    // Unbundle the input ports
    for (auto [i, val] : llvm::enumerate(funcOp.getArguments())) {
      auto unbundled = unbundleChannel(handshakeInPortNames[i],
                                       hw::ModulePort::Direction::Input, val);
      for (auto &port : unbundled)
        unbundledPorts.push_back(std::move(port));
    }
    // Unbundle the output ports
    for (auto [i, val] : llvm::enumerate(funcOp.getResultTypes())) {
      // Collect the terminator operands corresponding to the function result
      handshake::EndOp terminator =
          dyn_cast<handshake::EndOp>(funcOp.getBodyBlock()->getTerminator());
      auto unbundled = unbundleChannel(handshakeOutPortNames[i],
                                       hw::ModulePort::Direction::Output,
                                       terminator.getOperand(i));
      for (auto &port : unbundled)
        unbundledPorts.push_back(std::move(port));
    }
  } else {
    // Unbundle the input ports
    for (auto [i, val] : llvm::enumerate(handshakeOp->getOperands())) {
      auto unbundled = unbundleChannel(handshakeInPortNames[i],
                                       hw::ModulePort::Direction::Input, val);
      for (auto &port : unbundled)
        unbundledPorts.push_back(std::move(port));
    }
    // Unbundle the output ports
    for (auto [i, val] : llvm::enumerate(handshakeOp->getResults())) {
      auto unbundled = unbundleChannel(handshakeOutPortNames[i],
                                       hw::ModulePort::Direction::Output, val);
      for (auto &port : unbundled)
        unbundledPorts.push_back(std::move(port));
    }
  }
  return unbundledPorts;
}

// Function to build the hw::ModulePortInfo from the unbundled ports
static hw::ModulePortInfo
buildPortInfoFromHandshakeUnitPorts(const HandshakeUnitPortList &unbundledPorts,
                                    MLIRContext *ctx) {

  // Build the hw::ModulePortInfo from the unbundled ports
  SmallVector<hw::PortInfo> hwInputs, hwOutputs;
  Type i1 = IntegerType::get(ctx, 1);
  unsigned inIdx = 0, outIdx = 0;
  for (auto &port : unbundledPorts) {
    unsigned &counter = port->getDirection() == hw::ModulePort::Direction::Input
                            ? inIdx
                            : outIdx;
    hw::PortInfo pi{
        {StringAttr::get(ctx, port->getName()), i1, port->getDirection()},
        counter++};
    (port->getDirection() == hw::ModulePort::Direction::Input ? hwInputs
                                                              : hwOutputs)
        .push_back(pi);
  }
  // Add clk and rst ports
  unsigned clkIdx = inIdx++;
  unsigned rstIdx = inIdx++;
  hwInputs.emplace_back(hw::PortInfo{
      {StringAttr::get(ctx, CLK_PORT), i1, hw::ModulePort::Direction::Input},
      clkIdx});
  hwInputs.emplace_back(hw::PortInfo{
      {StringAttr::get(ctx, RST_PORT), i1, hw::ModulePort::Direction::Input},
      rstIdx});

  return hw::ModulePortInfo(hwInputs, hwOutputs);
}

//===----------------------------------------------------------------------===//
// unbundleHandshakeChannels: entry point for step 1
//===----------------------------------------------------------------------===//

// Function to unbundle all handshake channels in the module and create the
// corresponding hw modules with placeholder bodies
LogicalResult HandshakeUnbundler::unbundleHandshakeChannels() {
  MLIRContext *ctx = modOp.getContext();
  // Get top function
  topFunction = nullptr;
  for (auto op : modOp.getOps<handshake::FuncOp>()) {
    if (op.isExternal())
      continue;
    if (topFunction) {
      modOp->emitOpError() << "we currently only support one non-external "
                              "handshake function per module";
      return failure();
    }
    topFunction = op;
  }
  if (!topFunction) {
    modOp->emitOpError()
        << "expected to find a non-external handshake function";
    return failure();
  }
  // Get the port info for the top function and build the corresponding hw
  // module
  HandshakeUnitPortList handshakeUnbundledPortsTop =
      unbundlePorts(topFunction, ctx);
  hw::ModulePortInfo topHWPortInfo =
      buildPortInfoFromHandshakeUnitPorts(handshakeUnbundledPortsTop, ctx);
  builder.setInsertionPointToStart(modOp.getBody());
  StringAttr topName = StringAttr::get(ctx, getUniqueName(topFunction));
  topHWModule = builder.create<hw::HWModuleOp>(topFunction.getLoc(), topName,
                                               topHWPortInfo);

  // Initialize the backedge builder for creating placeholders for unbundled
  // bits
  backedgeBuilder =
      std::make_unique<BackedgeBuilder>(builder, topHWModule.getLoc());

  // Record the clk and rst from the top module block args which are the last
  // two ports in the port list
  Block *topBlock = topHWModule.getBodyBlock();
  unsigned numArgs = topBlock->getNumArguments();
  rst = topBlock->getArgument(numArgs - 1);
  clk = topBlock->getArgument(numArgs - 2);

  // Save the mapping between the old handshake channels of the func and the new
  // hw module ports
  SmallVector<Value> visitedDataArgs;
  unsigned argIdx = 0;
  for (auto &unbundledPort : handshakeUnbundledPortsTop) {
    if (unbundledPort->getDirection() == hw::ModulePort::Direction::Input) {
      // If port type is DATA (check the kind)
      if (unbundledPort->isData()) {
        auto *dataPortInfo = static_cast<DataPortInfo *>(unbundledPort.get());
        // If the channel has been visited already, skip it
        if (llvm::is_contained(visitedDataArgs,
                               unbundledPort->getHandshakeSignal()))
          continue;
        SmallVector<Value> unbundledValues;
        unsigned totalBits = dataPortInfo->getTotalBits();
        // Collect all the bits for this channel
        for (unsigned bit = 0; bit < totalBits; ++bit) {
          unbundledValues.push_back(topBlock->getArgument(argIdx++));
        }
        resolveUnbundledValues(unbundledPort.get(), unbundledValues);
        visitedDataArgs.push_back(unbundledPort->getHandshakeSignal());
      } else {
        auto newArg = topBlock->getArgument(argIdx++);
        resolveUnbundledValues(unbundledPort.get(), {newArg});
      }
    }
  }

  // Now, we unbundle the channels of each inner handshake op
  for (Operation &op : *topFunction.getBodyBlock()) {
    if (isa<handshake::EndOp>(op))
      continue; // handled as part of convertFuncOp
    if (failed(convertHandshakeOp(&op)))
      return failure();
  }

  // Connect the top module's output to the terminator
  return convertHandshakeFunc();
}

// Function to save the unbundled values for a given handshake
// signal, which will be used as operands for the new hw instance
void HandshakeUnbundler::resolveUnbundledValues(
    HandshakeUnitPortInfo *portKind, SmallVector<Value> unbundledValues) {

  Value handshakeSignal = portKind->getHandshakeSignal();
  UnbundledHandshakeChannel valTuple;
  // Check if a tuple already exists for this signal
  if (auto it = unbundledValuesMap.find(handshakeSignal);
      it != unbundledValuesMap.end()) {
    valTuple = it->second;
  }
  // Update the tuple with the new values
  valTuple.setValues(portKind, unbundledValues);
  unbundledValuesMap[handshakeSignal] = valTuple;

  // Remove the placeholder if it exists, since we now have the real unbundled
  // values
  if (pendingValuesMap.contains(handshakeSignal)) {
    // Check if the placeholder has the same bit type as the new unbundled
    // values
    SmallVector<Value> placeholderValues =
        pendingValuesMap[handshakeSignal].getValues(portKind);
    if (!placeholderValues.empty()) {
      // Assert the size of the placeholder values is the same as the new
      // unbundled values
      assert(placeholderValues.size() == unbundledValues.size() &&
             "placeholder values size should be the same as the new unbundled "
             "values");
      // Replace all its uses
      for (unsigned i = 0; i < placeholderValues.size(); ++i) {
        placeholderValues[i].replaceAllUsesWith(unbundledValues[i]);
      }
      // Update the map by removing the placeholder entry for that bit type
      UnbundledHandshakeChannel placeholderTuple =
          pendingValuesMap[handshakeSignal];
      placeholderTuple.setValues(portKind, {});
      // If the tuple is empty, we can erase the entry from
      // the map
      if (placeholderTuple.empty()) {
        pendingValuesMap.erase(handshakeSignal);
      } else {
        // If not empty, update the tuple in the map
        pendingValuesMap[handshakeSignal] = placeholderTuple;
      }
    }
  }
}

// Function to get the unbundled values for a given handshake signal, which
// will be used as operands for the new hw instance
SmallVector<Value> HandshakeUnbundler::findOrCreateUnbundledValues(
    unsigned totalBits, HandshakeUnitPortInfo *portKind, Location loc) {

  Value handshakeSignal = portKind->getHandshakeSignal();
  // Check if the unbundled values for this signal have already been
  // computed
  if (auto it = unbundledValuesMap.find(handshakeSignal);
      it != unbundledValuesMap.end()) {
    // Check per bit type
    SmallVector<Value> extractedValues = it->second.getValues(portKind);
    if (!extractedValues.empty()) {
      // Assert the size of the extracted values is the same as the total bits
      // expected
      assert(extractedValues.size() == totalBits &&
             "extracted values size should be the same as the total bits "
             "expected");
      return extractedValues;
    }
  }
  UnbundledHandshakeChannel valTuple;
  // Check if a placeholder exists for this signal
  if (auto it = pendingValuesMap.find(handshakeSignal);
      it != pendingValuesMap.end()) {
    valTuple = it->second;
    // Check per bit type
    SmallVector<Value> extractedValues = valTuple.getValues(portKind);
    if (!extractedValues.empty()) {
      // Assert the size of the extracted values is the same as the total bits
      // expected
      assert(extractedValues.size() == totalBits &&
             "extracted values size should be the same as the total bits"
             "expected");
      return extractedValues;
    }
  }
  // If not, create a new placeholder and return it. We use backedges as
  // placeholders
  SmallVector<Value> placeholders;
  builder.setInsertionPoint(topHWModule.getBodyBlock()->getTerminator());
  for (unsigned i = 0; i < totalBits; ++i) {
    auto backedge = backedgeBuilder->get(builder.getIntegerType(1));
    placeholders.push_back(backedge);
  }
  pendingValuesMap[handshakeSignal].setValues(portKind, placeholders);
  return placeholders;
}

// Function to create hw module
hw::HWModuleOp HandshakeUnbundler::createHWModuleHandshakeOp(
    StringRef moduleName, Operation *handshakeOp, MLIRContext *ctx) {

  hw::HWModuleOp hwModRes;
  HandshakeUnitPortList unbundledPorts = unbundlePorts(handshakeOp, ctx);
  hw::ModulePortInfo portInfo =
      buildPortInfoFromHandshakeUnitPorts(unbundledPorts, ctx);

  // Insert submodule definitions before the top module.
  builder.setInsertionPointToEnd(modOp.getBody());
  hwModRes = builder.create<hw::HWModuleOp>(
      handshakeOp->getLoc(), StringAttr::get(ctx, moduleName), portInfo);

  // Create subckt placeholder body for the new module
  OpBuilder bodyBuilder(hwModRes.getBodyBlock(),
                        hwModRes.getBodyBlock()->begin());
  SmallVector<Value> bodyInputs(hwModRes.getBodyBlock()->getArguments());
  SmallVector<Type> outputTypes;
  for (auto &port : hwModRes.getPortList())
    if (port.isOutput())
      outputTypes.push_back(port.type);
  Operation *term = hwModRes.getBodyBlock()->getTerminator();
  auto subckt = bodyBuilder.create<synth::SubcktOp>(
      term->getLoc(), TypeRange(outputTypes), bodyInputs, "synth_subckt");
  term->setOperands(subckt.getResults());

  // Record BLIF path for this handshake op
  auto blifIface = dyn_cast<BLIFImplInterface>(handshakeOp);
  StringAttr blifAttr = blifIface ? blifIface.getBLIFImpl() : StringAttr{};
  opToBlifPathMap[hwModRes.getOperation()] =
      (blifAttr && !blifAttr.getValue().empty()) ? blifAttr.getValue().str()
                                                 : "";
  return hwModRes;
}

// Function to convert a handshake operation by unbundling its channels and
// replacing it with the corresponding hw instance.
mlir::LogicalResult HandshakeUnbundler::convertHandshakeOp(Operation *op) {
  MLIRContext *ctx = modOp.getContext();
  Location loc = topHWModule.getBodyBlock()->getTerminator()->getLoc();
  StringRef opName = getUniqueName(op);
  StringAttr moduleName = StringAttr::get(ctx, opName);

  // Check if an hw module for this operation already exists (e.g. because of
  // multiple instances of the same op)
  hw::HWModuleOp hwModDef = symTable.lookup<hw::HWModuleOp>(moduleName);
  if (!hwModDef) {
    // If it does not exist, create it
    hwModDef = createHWModuleHandshakeOp(opName.str(), op, ctx);
  }
  if (!hwModDef) {
    llvm::errs() << "failed to create or find hw module for op: " << *op
                 << "\n";
    return failure();
  }

  // Build the instance inside the top hw module
  builder.setInsertionPoint(topHWModule.getBodyBlock()->getTerminator());
  // Collect the unbundled operands
  SmallVector<Value> hwInstOperands;
  HandshakeUnitPortList unbundledPorts = unbundlePorts(op, ctx);
  hw::ModulePortInfo portInfo =
      buildPortInfoFromHandshakeUnitPorts(unbundledPorts, ctx);
  for (auto &port : unbundledPorts) {
    if (port->getDirection() != hw::ModulePort::Direction::Input)
      continue;
    // Check if the port is clk or rst
    if (port->getName() == CLK_PORT || port->getName() == RST_PORT) {
      // We will connect them later, for now we can skip them
      continue;
    }
    unsigned totalBits = 1, bitIndex = 0;
    if (port->isData()) {
      auto *dataPortInfo = static_cast<DataPortInfo *>(port.get());
      totalBits = dataPortInfo->getTotalBits();
      bitIndex = dataPortInfo->getBitIndex();
    }
    auto unbundledValues =
        findOrCreateUnbundledValues(totalBits, port.get(), loc);
    if (bitIndex >= unbundledValues.size()) {
      llvm::errs() << "Error: bit index " << bitIndex
                   << " is out of bounds for unbundled values of signal "
                   << port->getHandshakeSignal() << "\n";
      llvm::errs() << "Unbundled values size: " << unbundledValues.size()
                   << "\n";
      return failure();
    }
    hwInstOperands.push_back(unbundledValues[bitIndex]);
  }
  // Add clk and rst signals as operands
  hwInstOperands.push_back(clk);
  hwInstOperands.push_back(rst);

  // Assert the number of input operands match the number of inputs of the hw
  // mod def
  unsigned numInputs = 0;
  for (auto &port : hwModDef.getPortList())
    if (port.isInput())
      numInputs++;
  if (hwInstOperands.size() != numInputs) {
    llvm::errs() << "Error: number of instance operands ("
                 << hwInstOperands.size()
                 << ") does not match the number of module inputs ("
                 << numInputs << ") for op: " << *op << "\n";
    return failure();
  }
  builder.setInsertionPoint(topHWModule.getBodyBlock()->getTerminator());

  // Create the instance
  // The input of the instance should be the unbundled operands we collected
  // The output of the instance are specified by the hw module definition we
  // created or looked up (hwModDef)
  auto instOp = builder.create<hw::InstanceOp>(
      loc, hwModDef, StringAttr::get(ctx, opName.str() + "_inst"),
      hwInstOperands);

  // Save the mapping between the old handshake channels for the op output and
  // the new hw module ports output
  // Create a map from handshake result to the list of (instance result index,
  // bit index) pairs that correspond to it. We need to group by handshake
  // result first because multiple hw module outputs. This is done only for
  // data signals. For the other signals we can directly save the mapping
  // since they are single bit
  DenseMap<Value, SmallVector<std::pair<unsigned, unsigned>>> outputPortsBitMap;
  unsigned outIdx = 0;
  for (auto &port : unbundledPorts) {
    if (port->getDirection() != hw::ModulePort::Direction::Output)
      continue;
    if (port->isData()) {
      auto *dataPortInfo = static_cast<DataPortInfo *>(port.get());
      unsigned bitIndex = dataPortInfo->getBitIndex();
      outputPortsBitMap[port->getHandshakeSignal()].push_back(
          {outIdx++, bitIndex});
    } else {
      resolveUnbundledValues(port.get(), {instOp->getResult(outIdx++)});
    }
  }
  for (auto &[handshakeValue, bitIdxPairs] : outputPortsBitMap) {
    // Sort by bitIndex to reconstruct the correct bit order.
    llvm::sort(bitIdxPairs,
               [](auto &a, auto &b) { return a.second < b.second; });
    SmallVector<Value> unbundledValues;
    for (auto [instIdx, _] : bitIdxPairs)
      unbundledValues.push_back(instOp->getResult(instIdx));
    DataPortInfo portInfo{legalizeDataPortName("", 0, unbundledValues.size()),
                          hw::ModulePort::Direction::Output, handshakeValue, 0,
                          (unsigned)unbundledValues.size()};
    resolveUnbundledValues(&portInfo, unbundledValues);
  }

  return success();
}

LogicalResult HandshakeUnbundler::convertHandshakeFunc() {
  if (!topHWModule) {
    llvm::errs() << "Top hw module must exist before convertHandshakeFunc\n";
    return failure();
  }
  // Get the terminator handshake::EndOp
  handshake::EndOp endOp =
      *topFunction.getBody().getOps<handshake::EndOp>().begin();
  if (!endOp) {
    topFunction.emitOpError()
        << "handshake::FuncOp must contain a handshake::EndOp terminator";
    return failure();
  }

  // Collect the unbundled operands for the terminator.
  // Order must match the output port order declared by unbundlePorts for a
  // funcOp: first the ready signals for every input argument, then the
  // data+valid signals for every result (EndOp operand).
  SmallVector<Value> hwTermOperands;

  // 1. Ready signals for every function input argument.
  for (auto [i, arg] : llvm::enumerate(topFunction.getArguments())) {
    ReadyPortInfo portInfo{"", hw::ModulePort::Direction::Input, arg};
    auto unbundledValues =
        findOrCreateUnbundledValues(1, &portInfo, endOp.getLoc());
    hwTermOperands.append(unbundledValues.begin(), unbundledValues.end());
  }

  // 2. Data + valid signals for every EndOp operand (function results).
  for (Value operand : endOp.getOperands()) {
    unsigned dataBitwidth = 0;
    if (auto channelType = operand.getType().dyn_cast<handshake::ChannelType>())
      dataBitwidth = channelType.getDataType().cast<IntegerType>().getWidth();
    else if (auto memType = operand.getType().dyn_cast<MemRefType>())
      dataBitwidth = memType.getElementType().cast<IntegerType>().getWidth();
    else if (auto intType = operand.getType().dyn_cast<IntegerType>())
      dataBitwidth = intType.getWidth();
    else
      dataBitwidth = 0;
    SmallVector<Value> unbundledValues;
    if (dataBitwidth > 0) {
      DataPortInfo portInfo{"", hw::ModulePort::Direction::Output, operand, 0,
                            dataBitwidth};
      unbundledValues =
          findOrCreateUnbundledValues(dataBitwidth, &portInfo, endOp.getLoc());
      hwTermOperands.append(unbundledValues.begin(), unbundledValues.end());
    }
    ValidPortInfo validPortInfo{"", hw::ModulePort::Direction::Output, operand};
    unbundledValues =
        findOrCreateUnbundledValues(1, &validPortInfo, endOp.getLoc());
    hwTermOperands.append(unbundledValues.begin(), unbundledValues.end());
  }
  topHWModule.getBodyBlock()->getTerminator()->setOperands(hwTermOperands);

  // Check there are no more placeholders left, which would indicate some
  // signals were not properly connected
  if (!pendingValuesMap.empty()) {
    llvm::errs() << "Error: there are still placeholders left after converting "
                    "the handshake function. This indicates some signals were "
                    "not properly connected.\n";
    return failure();
  }

  // Erase the original terminator and the handshake function
  endOp.erase();
  topFunction.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Step 2: Populate the hw modules with BLIF content
//===----------------------------------------------------------------------===//

// Function to populate a single hw module by importing the corresponding BLIF
// netlist and swapping the module body.
mlir::LogicalResult
populateHWModule(mlir::ModuleOp modOp, hw::InstanceOp inst,
                 llvm::DenseSet<StringRef> &populatedModules) {
  mlir::SymbolTable symTable(modOp);
  // Look up the hw module definition referenced by this instance.
  hw::HWModuleOp hwModule =
      symTable.lookup<hw::HWModuleOp>(inst.getModuleName());
  if (!hwModule) {
    llvm::errs() << "could not find hw module for instance: " << inst << "\n";
    return mlir::failure();
  }

  // Skip modules that have already been populated.
  mlir::StringAttr moduleNameAttr = hwModule.getNameAttr();
  if (populatedModules.count(moduleNameAttr.getValue().str()))
    return mlir::success();

  // Every hw module created in Step 1 must have an entry in opToBlifPathMap.
  if (!opToBlifPathMap.contains(hwModule.getOperation())) {
    llvm::errs() << "hw module is missing its BLIF path: " << hwModule
                 << "\nCheck Step 1 for correct path recording.\n";
    return mlir::failure();
  }
  llvm::StringRef blifPath = opToBlifPathMap[hwModule.getOperation()];

  // An empty path means this module should be left with subkct op inside
  if (blifPath.empty()) {
    populatedModules.insert(moduleNameAttr.getValue().str());
    return mlir::success();
  }

  // Collect the existing port names in declaration order so that the BLIF
  // importer can match them to the netlist's pin names.
  mlir::SmallVector<std::string> inputPins, outputPins;
  for (auto &port : hwModule.getPortList())
    (port.isInput() ? inputPins : outputPins)
        .push_back(port.name.getValue().str());

  // Import the BLIF netlist as a new hw::HWModuleOp.
  hw::HWModuleOp imported =
      importBlifCircuit(modOp, blifPath, {inputPins, outputPins});
  if (!imported) {
    llvm::errs() << "failed to import BLIF circuit for instance: " << inst
                 << "\n";
    return mlir::failure();
  }

  // Swap the placeholder module out and the imported one in, keeping the
  // original name so all existing hw::InstanceOp references stay valid.
  std::string origName = hwModule.getName().str();
  imported.setName(origName);
  symTable.erase(hwModule);
  symTable.insert(imported);

  populatedModules.insert(origName);
  return mlir::success();
}

// ---------------------------------------------------------------------------
// populateAllHWModules: entry point for step 2
// ---------------------------------------------------------------------------

// Function to populate all hw modules in the module by walking the instance
// hierarchy starting from the top-level module.
mlir::LogicalResult populateAllHWModules(mlir::ModuleOp modOp,
                                         llvm::StringRef topModuleName) {
  // Locate the top-level hw module.
  mlir::SymbolTable symTable(modOp);
  hw::HWModuleOp topMod = symTable.lookup<hw::HWModuleOp>(topModuleName);
  if (!topMod) {
    llvm::errs() << "could not find top hw module: " << topModuleName << "\n";
    return mlir::failure();
  }

  // Collect all instances
  mlir::SmallVector<hw::InstanceOp> instances;
  topMod.walk([&](hw::InstanceOp inst) { instances.push_back(inst); });

  // Collect all already populated modules to avoid duplicate work
  llvm::DenseSet<StringRef> populatedModules;
  for (hw::InstanceOp inst : instances)
    if (mlir::failed(populateHWModule(modOp, inst, populatedModules)))
      return mlir::failure();

  return mlir::success();
}

} // namespace dynamatic

// ------------------------------------------------------------------
// Main pass definition
// ------------------------------------------------------------------

namespace {
// The following pass converts handshake operations into synth operations
// It executes in multiple steps:
// 0) Check each handshake operation is marked with the path of the blif file
//    where its definition is located. This is done in a separate pass
//    (--mark-handshake-blif-impl).
// 1) Unbundle all handshake types used in the handshake function and the
//    inner handshake operations into their constituent signals (ready, valid,
//    data) and create a new hw module for the top-level handshake function with
//    ports corresponding to the unbundled signals.
// 2) Populate the hw module operations with the correspoding synth operations.
//    The description of the implementation is defined in the path specified
//    by the attribute blifPathAttrStr on each hw module
class HandshakeToSynthPass
    : public dynamatic::impl::HandshakeToSynthBase<HandshakeToSynthPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp modOp = cast<mlir::ModuleOp>(getOperation());
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);
    // We only support one function per module
    handshake::FuncOp funcOp = nullptr;
    // Check this is the case
    for (auto op : modOp.getOps<handshake::FuncOp>()) {
      if (op.isExternal())
        continue;
      if (funcOp) {
        modOp->emitOpError() << "we currently only support one non-external "
                                "handshake function per module";
        return signalPassFailure();
      }
      funcOp = op;
    }
    // If there is no function, nothing to do
    if (!funcOp)
      return;
    // Save the name of the top module function
    StringRef topModuleName = funcOp.getName();

    // Step 0: Check each handshake operation is marked with the path of the
    // blif file
    funcOp->walk([&](Operation *op) {
      if (isa<handshake::FuncOp>(op))
        return;
      BLIFImplInterface blifImplInterface = dyn_cast<BLIFImplInterface>(op);
      if (!blifImplInterface) {
        llvm::errs() << "Handshake operation " << getUniqueName(op)
                     << " does not implement the BLIFImplInterface\n";
        llvm::errs()
            << "Make sure to run the pass that marks handshake operations "
               "with BLIF file paths (--mark-handshake-blif-impl) before "
               "this pass\n";
        return signalPassFailure();
      }
      std::string blifFilePath = blifImplInterface.getBLIFImpl().str();
      if (blifFilePath.empty()) {
        // If the operation is an EndOp, it is expected to not have a BLIF file
        // path since it will be converted to the terminator of the top module,
        // so we can skip the warning in this case
        if (isa<handshake::EndOp>(op))
          return;
        // Write out warning
        llvm ::errs() << "Warning: Handshake operation " << getUniqueName(op)
                      << " has an empty BLIF file path\n";
      }
    });

    // Step 1: unbundle all handshake types in the handshake operations
    HandshakeUnbundler unbundler(modOp);
    if (failed(unbundler.unbundleHandshakeChannels()))
      return signalPassFailure();

    // Step 2: Populate the hw module operations with the correspoding synth
    // operations. The description of the implementation is defined in the
    // path specified by the attribute blifPathAttrStr on each hw module.
    if (failed(populateAllHWModules(modOp, topModuleName)))
      return signalPassFailure();
  }
};

} // namespace