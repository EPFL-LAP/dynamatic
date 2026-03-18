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
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/Passes.h" // IWYU pragma: keep
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
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

// Inserts a suffix string before the first '[' in the input name, or appends
// it at the end if no '[' is found.
// Example: insertSuffix("data[3]", "_valid") -> "data_valid[3]"
std::string insertSuffix(const std::string &name, const std::string &suffix) {
  std::size_t pos = name.find('[');
  if (pos != std::string::npos) {
    std::string result = name;
    result.insert(pos, suffix);
    return result;
  }
  return name + suffix;
}

// Converts a (root, index) pair into the canonical "root[index]" form.
// If root already contains "[N]", the old index is linearised with
//  arrayWidth before adding index.
// Example: formatArrayName("data[2]", 3, 4) becomes "data[11]"  (2*4 + 3)
std::string formatArrayName(const std::string &root, unsigned index,
                            unsigned arrayWidth = 0) {
  static const std::regex arrayPattern(R"((\w+)\[(\d+)\])");
  std::smatch m;
  if (std::regex_match(root, m, arrayPattern)) {
    assert(arrayWidth != 0 && "arrayWidth required for already-indexed names");
    unsigned linearised = std::stoi(m[2].str()) * arrayWidth + index;
    return m[1].str() + "[" + std::to_string(linearised) + "]";
  }
  return root + "[" + std::to_string(index) + "]";
}

// Formats a bit-indexed port name: "sig[bit]".
// When width == 1 the name is returned unchanged (no "[0]" suffix).
// If baseName already ends with "[N]" the indices are linearised.
static std::string bitPortName(StringRef baseName, unsigned bit,
                               unsigned width) {
  if (width == 1)
    return baseName.str();
  // Reuse the formatArrayName() helper from Step 1 / Layer 2.
  return formatArrayName(baseName.str(), bit, width);
}

//===----------------------------------------------------------------------===//
// unbundleHandshakePort: entry point for step 1
//===----------------------------------------------------------------------===//

// Function to unbundle a handshake port into its constituent signals (ready,
// valid, data)
SmallVector<UnbundledPort>
unbundleHandshakePort(Type type, std::string handshakePortName,
                      hw::ModulePort::Direction handshakePortDir,
                      Value handshakePort) {

  SmallVector<UnbundledPort> unbundledPorts;

  // Flipped direction: input becomes output and vice versa.
  auto flip = [](hw::ModulePort::Direction d) {
    return d == hw::ModulePort::Direction::Input
               ? hw::ModulePort::Direction::Output
               : hw::ModulePort::Direction::Input;
  };

  auto addPort = [&](StringRef name, hw::ModulePort::Direction dir, Value val,
                     PortBitType bitType, unsigned bitIdx = 0,
                     unsigned nBits = 1) {
    unbundledPorts.push_back({name.str(), dir, val, bitIdx, nBits, bitType});
  };

  // Depending on the type, the unbundling will be different
  if (auto channel = dyn_cast<handshake::ChannelType>(type)) {
    // For a channel type, we unbundle into ready, valid, and data ports
    unsigned dataWidth = channel.getDataType().cast<IntegerType>().getWidth();
    for (unsigned bit = 0; bit < dataWidth; ++bit) {
      addPort(bitPortName(handshakePortName, bit, dataWidth), handshakePortDir,
              handshakePort, PortBitType::DATA, bit, dataWidth);
    }
    // valid: keep direction, always i1
    addPort(insertSuffix(handshakePortName, "_valid"), handshakePortDir,
            handshakePort, PortBitType::VALID);
    // ready: flip direction, always i1
    addPort(insertSuffix(handshakePortName, "_ready"), flip(handshakePortDir),
            handshakePort, PortBitType::READY);
  } else if (isa<handshake::ControlType>(type)) {
    // valid: keep direction, always i1
    addPort(insertSuffix(handshakePortName, "_valid"), handshakePortDir,
            handshakePort, PortBitType::VALID);
    // ready: flip direction, always i1
    addPort(insertSuffix(handshakePortName, "_ready"), flip(handshakePortDir),
            handshakePort, PortBitType::READY);
  } else if (auto mem = dyn_cast<MemRefType>(type)) {
    unsigned dataWidth = mem.getElementType().cast<IntegerType>().getWidth();
    for (unsigned bit = 0; bit < dataWidth; ++bit) {
      addPort(bitPortName(handshakePortName, bit, dataWidth), handshakePortDir,
              handshakePort, PortBitType::DATA, bit, dataWidth);
    }
    // valid: keep direction, always i1
    addPort(insertSuffix(handshakePortName, "_valid"), handshakePortDir,
            handshakePort, PortBitType::VALID);
    // ready: flip direction, always i1
    addPort(insertSuffix(handshakePortName, "_ready"), flip(handshakePortDir),
            handshakePort, PortBitType::READY);
  } else {
    // Pass-through: split multi-bit integer types into i1 ports.
    if (auto intTy = dyn_cast<IntegerType>(type)) {
      unsigned width = intTy.getWidth();
      for (unsigned b = 0; b < width; ++b)
        addPort(bitPortName(handshakePortName, b, width), handshakePortDir,
                handshakePort, PortBitType::DATA, b, width);
    } else {
      addPort(handshakePortName, handshakePortDir, handshakePort,
              PortBitType::DATA);
    }
  }
  return unbundledPorts;
}

// Function to build the hw::ModulePortInfo for a given handshake operation by
// unbundling its ports
std::pair<hw::ModulePortInfo, SmallVector<UnbundledPort>>
buildPortInfo(Operation *handshakeOp, MLIRContext *ctx) {

  SmallVector<UnbundledPort> unbundledPorts;

  // First, we have to derive the base names from NamedIOInterface
  SmallVector<std::string> handshakeInPortNames, handshakeOutPortNames;
  // Check if it is a funcOp
  if (auto funcOp = dyn_cast<handshake::FuncOp>(handshakeOp)) {
    // In this case the naming convention is generic
    for (auto [i, _] : llvm::enumerate(funcOp.getArguments()))
      handshakeInPortNames.push_back("in" + std::to_string(i));
    for (auto [i, _] : llvm::enumerate(funcOp.getResultTypes()))
      handshakeOutPortNames.push_back("out" + std::to_string(i));
  } else {
    // For other ops we use the op name as base for the port names
    // We have to fix the names of the ports so that the root_N pattern is
    // converted to root[N] to be compatible with the BLIF importer
    auto namedIO = dyn_cast<handshake::NamedIOInterface>(handshakeOp);
    assert(namedIO && "op must implement NamedIOInterface");

    // Pattern "root_N": convert to "root[N]"
    static const std::regex indexPat(R"((\w+)_(\d+))");
    auto elaborateName = [&](std::string operandName,
                             SmallVector<std::string> &operandNames) {
      std::smatch m;
      if (!std::regex_match(operandName, m, indexPat)) {
        operandNames.push_back(operandName);
        return;
      }
      // If the name matches the pattern, elaborate it into an array name.
      std::string root = m[1].str();
      unsigned idx = std::stoi(m[2].str());
      if (idx == 0) {
        operandNames.push_back(root);
      } else {
        if (idx == 1) {
          // The first time we see this pattern, we need to back-patch the root
          // name to have the "[0]" suffix, since the original name was used for
          // the first element.
          auto it = llvm::find(operandNames, root);
          assert(it != operandNames.end() &&
                 "index-0 port not found for back-patch");
          *it = formatArrayName(root, 0);
        }
        operandNames.push_back(formatArrayName(root, idx));
      }
    };
    // Iterate over the operand and result names and fix the ones that match the
    // pattern
    for (auto [i, _] : llvm::enumerate(handshakeOp->getOperands()))
      elaborateName(namedIO.getOperandName(i), handshakeInPortNames);
    for (auto [i, _] : llvm::enumerate(handshakeOp->getResults()))
      elaborateName(namedIO.getResultName(i), handshakeOutPortNames);
  }

  // Second, we use the base names to unbundle the ports and build the
  // hw::ModulePortInfo
  // Check if the op is a funcOp
  if (auto funcOp = dyn_cast<handshake::FuncOp>(handshakeOp)) {
    // Unbundle the input ports
    for (auto [i, val] : llvm::enumerate(funcOp.getArguments())) {
      auto unbundled =
          unbundleHandshakePort(val.getType(), handshakeInPortNames[i],
                                hw::ModulePort::Direction::Input, val);
      unbundledPorts.append(unbundled.begin(), unbundled.end());
    }
    // Unbundle the output ports
    for (auto [i, val] : llvm::enumerate(funcOp.getResultTypes())) {
      auto unbundled = unbundleHandshakePort(
          val, handshakeOutPortNames[i], hw::ModulePort::Direction::Output, {});
      unbundledPorts.append(unbundled.begin(), unbundled.end());
    }
  } else {
    // Unbundle the input ports
    for (auto [i, val] : llvm::enumerate(handshakeOp->getOperands())) {
      auto unbundled =
          unbundleHandshakePort(val.getType(), handshakeInPortNames[i],
                                hw::ModulePort::Direction::Input, val);
      unbundledPorts.append(unbundled.begin(), unbundled.end());
    }
    // Unbundle the output ports
    for (auto [i, val] : llvm::enumerate(handshakeOp->getResults())) {
      auto unbundled =
          unbundleHandshakePort(val.getType(), handshakeOutPortNames[i],
                                hw::ModulePort::Direction::Output, val);
      unbundledPorts.append(unbundled.begin(), unbundled.end());
    }
  }
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
  hwInputs.push_back({{StringAttr::get(ctx, clockSignal), i1,
                       hw::ModulePort::Direction::Input},
                      clkIdx});
  hwInputs.push_back({{StringAttr::get(ctx, resetSignal), i1,
                       hw::ModulePort::Direction::Input},
                      rstIdx});

  return {hw::ModulePortInfo(hwInputs, hwOutputs), unbundledPorts};
}

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
  assert(topFunction && "expected to find a non-external handshake function");
  // Get the port info for the top function and build the corresponding hw
  // module
  auto [topHWPortInfo, unbundledPortsTop] = buildPortInfo(topFunction, ctx);
  builder.setInsertionPointToStart(modOp.getBody());
  StringAttr topName = StringAttr::get(ctx, getUniqueName(topFunction));
  topHWModule = builder.create<hw::HWModuleOp>(topFunction.getLoc(), topName,
                                               topHWPortInfo);

  // Record the clk and rst from the top module block args which are the last
  // two ports in the port list
  Block *topBlock = topHWModule.getBodyBlock();
  unsigned numArgs = topBlock->getNumArguments();
  rst = topBlock->getArgument(numArgs - 1);
  clk = topBlock->getArgument(numArgs - 2);

  // Save the mapping between the old handshake channels of the func and the new
  // hw module ports
  unsigned argIdx = 0;
  for (auto [i, arg] : llvm::enumerate(topFunction.getArguments())) {
    // Collect all unbundled ports that correspond to this argument only for
    // data and valid bits
    // For the ready signals, coming from the output channels, we will collect
    // them later when we convert the terminator
    for (auto portType : {PortBitType::DATA, PortBitType::VALID}) {
      SmallVector<Value> unbundledValues;
      for (auto &unbundledPort : unbundledPortsTop) {
        if (unbundledPort.handshakeSignal == arg &&
            unbundledPort.direction == hw::ModulePort::Direction::Input &&
            unbundledPort.bitType == portType) {
          unbundledValues.push_back(topBlock->getArgument(argIdx++));
        }
      }
      if (!unbundledValues.empty())
        saveUnbundledValues(arg, portType, unbundledValues);
    }
  }

  // Save the mapping for the input ready signals that belong to the output of
  // the top func
  // These values correspond to the inputs of the terminator
  handshake::EndOp endOp =
      *topFunction.getBody().getOps<handshake::EndOp>().begin();
  unsigned endOpIdx = 0;
  // For each operand, extract the ready signal and save it
  for (auto &unbundledPort : unbundledPortsTop) {
    if (unbundledPort.direction == hw::ModulePort::Direction::Input &&
        unbundledPort.bitType == PortBitType::READY) {
      assert(endOpIdx < endOp.getNumOperands() &&
             "not enough operands in the terminator for the ready signals");
      Value operand = endOp.getOperand(endOpIdx++);
      saveUnbundledValues(operand, PortBitType::READY,
                          {topBlock->getArgument(argIdx++)});
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

// Helper function to format tuple from map
UnbundledValuesTuple
HandshakeUnbundler::updateTuple(UnbundledValuesTuple oldTuple,
                                SmallVector<Value> newValues,
                                PortBitType bitType) {
  SmallVector<Value> dataValues = std::get<0>(oldTuple);
  Value validValue = std::get<1>(oldTuple);
  Value readyValue = std::get<2>(oldTuple);
  if (bitType == PortBitType::DATA)
    dataValues = newValues;
  else if (bitType == PortBitType::VALID) {
    if (!newValues.empty())
      validValue = newValues[0];
    else
      validValue = Value();
  } else if (bitType == PortBitType::READY) {
    if (!newValues.empty())
      readyValue = newValues[0];
    else
      readyValue = Value();
  }
  return {dataValues, validValue, readyValue};
}

// Helper function to get newValues from a tuple
SmallVector<Value>
HandshakeUnbundler::getValuesFromTuple(UnbundledValuesTuple valTuple,
                                       PortBitType bitType) {
  if (bitType == PortBitType::DATA)
    return std::get<0>(valTuple);
  if (bitType == PortBitType::VALID) {
    Value v = std::get<1>(valTuple);
    return v ? SmallVector<Value>{v} : SmallVector<Value>{};
  }
  if (bitType == PortBitType::READY) {
    Value v = std::get<2>(valTuple);
    return v ? SmallVector<Value>{v} : SmallVector<Value>{};
  }
  return {};
}

// Function to save the unbundled values for a given handshake
// signal, which will be used as operands for the new hw instance
void HandshakeUnbundler::saveUnbundledValues(
    Value handshakeSignal, PortBitType bitType,
    SmallVector<Value> unbundledValues) {

  UnbundledValuesTuple valTuple;
  // Check if a tuple already exists for this signal
  if (auto it = unbundledValuesMap.find(handshakeSignal);
      it != unbundledValuesMap.end()) {
    valTuple = it->second;
  }
  // Update the tuple with the new values
  valTuple = updateTuple(valTuple, unbundledValues, bitType);
  unbundledValuesMap[handshakeSignal] = valTuple;

  // Remove the placeholder if it exists, since we now have the real unbundled
  // values
  if (placeholderMap.contains(handshakeSignal)) {
    // Check if the placeholder has the same bit type as the new unbundled
    // values
    SmallVector<Value> placeholderValues =
        getValuesFromTuple(placeholderMap[handshakeSignal], bitType);
    if (!placeholderValues.empty()) {
      // Assert the size of the placeholder values is the same as the new
      // unbundled values
      assert(placeholderValues.size() == unbundledValues.size() &&
             "placeholder values size should be the same as the new unbundled "
             "values");
      // Replace all its uses
      for (unsigned i = 0; i < placeholderValues.size(); ++i) {
        placeholderValues[i].replaceAllUsesWith(unbundledValues[i]);
        // Erase the placeholder
        if (auto defOp = placeholderValues[i].getDefiningOp())
          defOp->erase();
      }
      // Update the map by removing the placeholder entry for that bit type
      UnbundledValuesTuple placeholderTuple =
          updateTuple(placeholderMap[handshakeSignal], {}, bitType);
      // If all the values in the tuple are empty, we can erase the entry from
      // the map
      if (std::get<0>(placeholderTuple).empty() &&
          std::get<1>(placeholderTuple) == Value() &&
          std::get<2>(placeholderTuple) == Value()) {
        placeholderMap.erase(handshakeSignal);
      } else {
        // If not empty, update the tuple in the map
        placeholderMap[handshakeSignal] = placeholderTuple;
      }
    }
  }
}

// Function to get the unbundled values for a given handshake signal, which
// will be used as operands for the new hw instance
SmallVector<Value> HandshakeUnbundler::getUnbundledValues(Value handshakeSignal,
                                                          unsigned totalBits,
                                                          PortBitType bitType,
                                                          Location loc) {

  // Check if the unbundled values for this signal have already been
  // computed
  if (auto it = unbundledValuesMap.find(handshakeSignal);
      it != unbundledValuesMap.end()) {
    // Check per bit type
    SmallVector<Value> extractedValues =
        getValuesFromTuple(it->second, bitType);
    if (!extractedValues.empty())
      return extractedValues;
  }
  std::tuple<SmallVector<Value>, Value, Value> valTuple;
  // Check if a placeholder exists for this signal
  if (auto it = placeholderMap.find(handshakeSignal);
      it != placeholderMap.end()) {
    valTuple = it->second;
    // Check per bit type
    SmallVector<Value> extractedValues =
        getValuesFromTuple(it->second, bitType);
    if (!extractedValues.empty())
      return extractedValues;
  }
  // If not, create a new placeholder and return it
  SmallVector<Value> placeholders;
  builder.setInsertionPoint(topHWModule.getBodyBlock()->getTerminator());
  for (unsigned i = 0; i < totalBits; ++i) {
    auto c = builder.create<hw::ConstantOp>(
        loc, builder.getIntegerType(1),
        builder.getIntegerAttr(builder.getIntegerType(1), 0));
    placeholders.push_back(c.getResult());
  }
  placeholderMap[handshakeSignal] =
      updateTuple(valTuple, placeholders, bitType);
  return placeholders;
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
    auto [portInfo, unbundledPorts] = buildPortInfo(op, ctx);

    // Insert submodule definitions before the top module.
    builder.setInsertionPointToEnd(modOp.getBody());
    hwModDef =
        builder.create<hw::HWModuleOp>(op->getLoc(), moduleName, portInfo);

    // Create subckt placeholder body for the new module
    OpBuilder bodyBuilder(hwModDef.getBodyBlock(),
                          hwModDef.getBodyBlock()->begin());
    SmallVector<Value> bodyInputs(hwModDef.getBodyBlock()->getArguments());
    SmallVector<Type> outputTypes;
    for (auto &port : hwModDef.getPortList())
      if (port.isOutput())
        outputTypes.push_back(port.type);
    Operation *term = hwModDef.getBodyBlock()->getTerminator();
    auto subckt = bodyBuilder.create<synth::SubcktOp>(
        term->getLoc(), TypeRange(outputTypes), bodyInputs, "synth_subckt");
    term->setOperands(subckt.getResults());

    // Record BLIF path for this handshake op
    auto blifIface = dyn_cast<BLIFImplInterface>(op);
    StringAttr blifAttr = blifIface ? blifIface.getBLIFImpl() : StringAttr{};
    opToBlifPathMap[hwModDef.getOperation()] =
        (blifAttr && !blifAttr.getValue().empty()) ? blifAttr.getValue().str()
                                                   : "";
  }
  assert(hwModDef && "failed to create or find hw module for handshake op");

  // Build the instance inside the top hw module
  builder.setInsertionPoint(topHWModule.getBodyBlock()->getTerminator());
  // Collect the unbundled operands
  SmallVector<Value> hwInstOperands;
  auto [portInfo, unbundledPorts] = buildPortInfo(op, ctx);
  for (auto &port : unbundledPorts) {
    if (port.direction != hw::ModulePort::Direction::Input)
      continue;
    // Check if the port is clk or rst
    if (port.name == clockSignal || port.name == resetSignal) {
      // We will connect them later, for now we can skip them
      continue;
    }
    auto unbundledValues = getUnbundledValues(
        port.handshakeSignal, port.totalBits, port.bitType, loc);
    assert(port.bitIndex < unbundledValues.size() &&
           "Unbundled values size should be the same at the bit index of the "
           "original port");
    hwInstOperands.push_back(unbundledValues[port.bitIndex]);
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
  assert(
      hwInstOperands.size() == numInputs &&
      "number of instance operands should match the number of module inputs");
  builder.setInsertionPoint(topHWModule.getBodyBlock()->getTerminator());

  // Create the instance
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
    if (port.bitType == PortBitType::DATA) {
      outputPortsBitMap[port.handshakeSignal].push_back(
          {outIdx++, port.bitIndex});
    } else {
      saveUnbundledValues(port.handshakeSignal, port.bitType,
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
    saveUnbundledValues(handshakeValue, PortBitType::DATA, unbundledValues);
  }

  return success();
}

LogicalResult HandshakeUnbundler::convertHandshakeFunc() {
  assert(topHWModule && "Top hw module must exist before convertHandshakeFunc");
  // Get the terminator handshake::EndOp
  handshake::EndOp endOp;
  for (Operation &op : *topFunction.getBodyBlock()) {
    if ((endOp = dyn_cast<handshake::EndOp>(op)))
      break;
  }
  assert(endOp && "handshake::FuncOp must contain an EndOp");

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
                                           PortBitType::DATA, endOp.getLoc());
      hwTermOperands.append(unbundledValues.begin(), unbundledValues.end());
    }
    // Add valid signals
    unbundledValues =
        getUnbundledValues(operand, 1, PortBitType::VALID, endOp.getLoc());
    hwTermOperands.append(unbundledValues.begin(), unbundledValues.end());
  }
  // Check also the inputs of the topFuncOp whose ready signals are connected
  // to the terminator
  for (auto [i, arg] : llvm::enumerate(topFunction.getArguments())) {
    auto unbundledValues =
        getUnbundledValues(arg, 1, PortBitType::READY, endOp.getLoc());
    hwTermOperands.append(unbundledValues.begin(), unbundledValues.end());
  }
  topHWModule.getBodyBlock()->getTerminator()->setOperands(hwTermOperands);

  // Check there are no more placeholders left, which would indicate some
  // signals were not properly connected
  if (!placeholderMap.empty()) {
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
                 llvm::DenseSet<std::string> &populatedModules) {
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
  assert(opToBlifPathMap.contains(hwModule.getOperation()) &&
         "hw module is missing its BLIF path. Check Step 1 for correct path "
         "recording.");
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

  // Collect all instances up front to avoid iterator invalidation when
  // populate() swaps module bodies in the symbol table.
  mlir::SmallVector<hw::InstanceOp> instances;
  topMod.walk([&](hw::InstanceOp inst) { instances.push_back(inst); });

  // A single BlifPopulator owns the SymbolTable and dedup set for the whole
  // walk; populate() can be called once per instance without any
  // "already done" bookkeeping at the call site.
  llvm::DenseSet<std::string> populatedModules;
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
  HandshakeToSynthPass(std::string blifDirPath);
  using HandshakeToSynthBase::HandshakeToSynthBase;

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
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
        assert(false && "Check the pass that marks handshake operations with "
                        "BLIF file paths is executed correctly");
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