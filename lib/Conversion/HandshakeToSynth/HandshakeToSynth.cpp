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

#include "dynamatic/Conversion/HandshakeToSynth.h"

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Conversion/Passes.h"
#include "dynamatic/Dialect/HW/HWOps.h"
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKETOSYNTH
#include "dynamatic/Conversion/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

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
SmallVector<HandshakeUnitPort>
unbundleChannel(const std::string &handshakePortName,
                hw::ModulePort::Direction handshakePortDir,
                Value handshakePort) {

  SmallVector<HandshakeUnitPort> unbundledPorts;
  Type type = handshakePort.getType();

  // Flipped direction: input becomes output and vice versa.
  auto flip = [](hw::ModulePort::Direction d) {
    return d == hw::ModulePort::Direction::Input
               ? hw::ModulePort::Direction::Output
               : hw::ModulePort::Direction::Input;
  };

  auto addPort = [&](StringRef name, hw::ModulePort::Direction dir, Value val,
                     PortKind kind) {
    unbundledPorts.emplace_back(name.str(), dir, val, kind);
  };

  // Depending on the type, the unbundling will be different
  if (auto channel = dyn_cast<handshake::ChannelType>(type)) {
    // For a channel type, we unbundle into ready, valid, and data ports
    unsigned dataWidth = channel.getDataType().cast<IntegerType>().getWidth();
    for (unsigned bit = 0; bit < dataWidth; ++bit) {
      addPort(legalizeDataPortName(handshakePortName, bit, dataWidth),
              handshakePortDir, handshakePort, DataPortInfo{bit, dataWidth});
    }
    // valid: keep direction, always i1
    addPort(legalizeControlPortName(handshakePortName, "_valid"),
            handshakePortDir, handshakePort, ValidPortInfo{});
    // ready: flip direction, always i1
    addPort(legalizeControlPortName(handshakePortName, "_ready"),
            flip(handshakePortDir), handshakePort, ReadyPortInfo{});
  } else if (isa<handshake::ControlType>(type)) {
    // valid: keep direction, always i1
    addPort(legalizeControlPortName(handshakePortName, "_valid"),
            handshakePortDir, handshakePort, ValidPortInfo{});
    // ready: flip direction, always i1
    addPort(legalizeControlPortName(handshakePortName, "_ready"),
            flip(handshakePortDir), handshakePort, ReadyPortInfo{});
  } else if (auto mem = dyn_cast<MemRefType>(type)) {
    unsigned dataWidth = mem.getElementType().cast<IntegerType>().getWidth();
    for (unsigned bit = 0; bit < dataWidth; ++bit) {
      addPort(legalizeDataPortName(handshakePortName, bit, dataWidth),
              handshakePortDir, handshakePort, DataPortInfo{bit, dataWidth});
    }
    // valid: keep direction, always i1
    addPort(legalizeControlPortName(handshakePortName, "_valid"),
            handshakePortDir, handshakePort, ValidPortInfo{});
    // ready: flip direction, always i1
    addPort(legalizeControlPortName(handshakePortName, "_ready"),
            flip(handshakePortDir), handshakePort, ReadyPortInfo{});
  } else {
    // Pass-through: split multi-bit integer types into i1 ports.
    if (auto intTy = dyn_cast<IntegerType>(type)) {
      unsigned width = intTy.getWidth();
      for (unsigned b = 0; b < width; ++b)
        addPort(legalizeDataPortName(handshakePortName, b, width),
                handshakePortDir, handshakePort, DataPortInfo{b, width});
    } else {
      addPort(handshakePortName, handshakePortDir, handshakePort,
              ValidPortInfo{});
    }
  }
  return unbundledPorts;
}

// Function to unbundle handshake channels of an handshake op
static SmallVector<HandshakeUnitPort> unbundlePorts(Operation *handshakeOp,
                                                    MLIRContext *ctx) {

  SmallVector<HandshakeUnitPort> unbundledPorts;

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
      unbundledPorts.append(unbundled.begin(), unbundled.end());
    }
    // Unbundle the output ports
    for (auto [i, val] : llvm::enumerate(funcOp.getResultTypes())) {
      // Collect the terminator operands corresponding to the function result
      handshake::EndOp terminator =
          dyn_cast<handshake::EndOp>(funcOp.getBodyBlock()->getTerminator());
      auto unbundled = unbundleChannel(handshakeOutPortNames[i],
                                       hw::ModulePort::Direction::Output,
                                       terminator.getOperand(i));
      unbundledPorts.append(unbundled.begin(), unbundled.end());
    }
  } else {
    // Unbundle the input ports
    for (auto [i, val] : llvm::enumerate(handshakeOp->getOperands())) {
      auto unbundled = unbundleChannel(handshakeInPortNames[i],
                                       hw::ModulePort::Direction::Input, val);
      unbundledPorts.append(unbundled.begin(), unbundled.end());
    }
    // Unbundle the output ports
    for (auto [i, val] : llvm::enumerate(handshakeOp->getResults())) {
      auto unbundled = unbundleChannel(handshakeOutPortNames[i],
                                       hw::ModulePort::Direction::Output, val);
      unbundledPorts.append(unbundled.begin(), unbundled.end());
    }
  }
  return unbundledPorts;
}

// Function to build the hw::ModulePortInfo from the unbundled ports
static hw::ModulePortInfo buildPortInfoFromHandshakeUnitPorts(
    SmallVector<HandshakeUnitPort> unbundledPorts, MLIRContext *ctx) {

  // Build the hw::ModulePortInfo from the unbundled ports
  SmallVector<hw::PortInfo> hwInputs, hwOutputs;
  Type i1 = IntegerType::get(ctx, 1);
  unsigned inIdx = 0, outIdx = 0;
  for (auto &port : unbundledPorts) {
    unsigned &counter =
        port.direction == hw::ModulePort::Direction::Input ? inIdx : outIdx;
    hw::PortInfo pi{{StringAttr::get(ctx, port.name), i1, port.direction},
                    counter++};
    (port.direction == hw::ModulePort::Direction::Input ? hwInputs : hwOutputs)
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
  SmallVector<HandshakeUnitPort> handshakeUnbundledPortsTop =
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
    if (unbundledPort.direction == hw::ModulePort::Direction::Input) {
      // If port type is DATA (check the kind)
      auto *dataPortInfo = std::get_if<DataPortInfo>(&unbundledPort.kind);
      if (dataPortInfo) {
        // If the channel has been visited already, skip it
        if (llvm::is_contained(visitedDataArgs, unbundledPort.handshakeSignal))
          continue;
        SmallVector<Value> unbundledValues;
        unsigned totalBits = dataPortInfo->totalBits;
        // Collect all the bits for this channel
        for (unsigned bit = 0; bit < totalBits; ++bit) {
          unbundledValues.push_back(topBlock->getArgument(argIdx++));
        }
        saveUnbundledValues(unbundledPort.handshakeSignal, unbundledPort.kind,
                            unbundledValues);
        visitedDataArgs.push_back(unbundledPort.handshakeSignal);
      } else {
        auto newArg = topBlock->getArgument(argIdx++);
        saveUnbundledValues(unbundledPort.handshakeSignal, unbundledPort.kind,
                            {newArg});
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
void HandshakeUnbundler::saveUnbundledValues(
    Value handshakeSignal, PortKind portKind,
    SmallVector<Value> unbundledValues) {

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
SmallVector<Value> HandshakeUnbundler::getUnbundledValues(Value handshakeSignal,
                                                          unsigned totalBits,
                                                          PortKind portKind,
                                                          Location loc) {

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
  SmallVector<HandshakeUnitPort> unbundledPorts =
      unbundlePorts(handshakeOp, ctx);
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
  SmallVector<HandshakeUnitPort> unbundledPorts = unbundlePorts(op, ctx);
  hw::ModulePortInfo portInfo =
      buildPortInfoFromHandshakeUnitPorts(unbundledPorts, ctx);
  for (auto &port : unbundledPorts) {
    if (port.direction != hw::ModulePort::Direction::Input)
      continue;
    // Check if the port is clk or rst
    if (port.name == CLK_PORT || port.name == RST_PORT) {
      // We will connect them later, for now we can skip them
      continue;
    }
    unsigned totalBits = 1, bitIndex = 0;
    if (auto *dataPortInfo = std::get_if<DataPortInfo>(&port.kind)) {
      totalBits = dataPortInfo->totalBits;
      bitIndex = dataPortInfo->bitIndex;
    }
    auto unbundledValues =
        getUnbundledValues(port.handshakeSignal, totalBits, port.kind, loc);
    if (bitIndex >= unbundledValues.size()) {
      llvm::errs() << "Error: bit index " << bitIndex
                   << " is out of bounds for unbundled values of signal "
                   << port.handshakeSignal << "\n";
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
    if (port.direction != hw::ModulePort::Direction::Output)
      continue;
    if (auto *dataPortInfo = std::get_if<DataPortInfo>(&port.kind)) {
      unsigned bitIndex = dataPortInfo->bitIndex;
      outputPortsBitMap[port.handshakeSignal].push_back({outIdx++, bitIndex});
    } else {
      saveUnbundledValues(port.handshakeSignal, port.kind,
                          {instOp->getResult(outIdx++)});
    }
  }
  for (auto &[handshakeValue, bitIdxPairs] : outputPortsBitMap) {
    // Sort by bitIndex to reconstruct the correct bit order.
    llvm::sort(bitIdxPairs,
               [](auto &a, auto &b) { return a.second < b.second; });
    SmallVector<Value> unbundledValues;
    for (auto [instIdx, _] : bitIdxPairs)
      unbundledValues.push_back(instOp->getResult(instIdx));
    saveUnbundledValues(handshakeValue, DataPortInfo{}, unbundledValues);
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

  // Collect the unbundled operands for the terminator
  SmallVector<Value> hwTermOperands;
  for (Value operand : endOp.getOperands()) {
    // Get the data bitwidth of the operand
    unsigned dataBitwidth = 0;
    if (auto channelType = operand.getType().dyn_cast<handshake::ChannelType>())
      dataBitwidth = channelType.getDataType().cast<IntegerType>().getWidth();
    else if (auto memType = operand.getType().dyn_cast<MemRefType>())
      dataBitwidth = memType.getElementType().cast<IntegerType>().getWidth();
    else if (auto intType = operand.getType().dyn_cast<IntegerType>())
      dataBitwidth = intType.getWidth();
    else
      dataBitwidth =
          0; // For control types and other types, we treat them as 0 bits
    SmallVector<Value> unbundledValues;
    if (dataBitwidth > 0) {
      // Add data signals
      unbundledValues = getUnbundledValues(operand, dataBitwidth,
                                           DataPortInfo{}, endOp.getLoc());
      hwTermOperands.append(unbundledValues.begin(), unbundledValues.end());
    }
    // Add valid signals
    unbundledValues =
        getUnbundledValues(operand, 1, ValidPortInfo{}, endOp.getLoc());
    hwTermOperands.append(unbundledValues.begin(), unbundledValues.end());
  }
  // Check also the inputs of the topFuncOp whose ready signals are connected
  // to the terminator
  for (auto [i, arg] : llvm::enumerate(topFunction.getArguments())) {
    auto unbundledValues =
        getUnbundledValues(arg, 1, ReadyPortInfo{}, endOp.getLoc());
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
        // Write out warning
        llvm::errs() << "Warning: Handshake operation " << getUniqueName(op)
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