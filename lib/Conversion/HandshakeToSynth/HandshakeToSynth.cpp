//===- HandshakeToSynth.cpp - Convert Handshake to Synth --------------*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts Handshake constructs into equivalent Synth constructs.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/HW/HWOpInterfaces.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/HW/HWTypes.h"
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Dialect/Synth/SynthDialect.h"
#include "dynamatic/Dialect/Synth/SynthOps.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "dynamatic/Transforms/BlifImporter/BlifImporterSupport.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <bitset>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>
#include <utility>

// [START Boilerplate code for the MLIR pass]
#include "dynamatic/Conversion/Passes.h" // IWYU pragma: keep
namespace dynamatic {
#define GEN_PASS_DEF_HANDSHAKETOSYNTH
#include "dynamatic/Conversion/Passes.h.inc"
} // namespace dynamatic
// [END Boilerplate code for the MLIR pass]

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

// Keywords for data and control signals when unbundling Handshake types using
// enums.
enum SignalKind {
  DATA_SIGNAL = 0,
  VALID_SIGNAL = 1,
  READY_SIGNAL = 2,
};

// Strings representing the name of the clock and reset signals
static const std::string clockSignal = "clk";
static const std::string resetSignal = "rst";

// ------------------------------------------------------------------
// Utilities for unbundling Handshake types and operations into flat hw module
// ports and operations
// ------------------------------------------------------------------

// Function to unbundle a single handshake type into its subcomponents
SmallVector<std::pair<SignalKind, Type>> unbundleType(Type type) {
  // The reason for unbundling is that handshake channels are bundled
  // types that contain data, valid and ready signals, while hw module
  // ports are flat, so we need to unbundle the channel into its components
  return TypeSwitch<Type, SmallVector<std::pair<SignalKind, Type>>>(type)
      // Case for channel type where we unbundle into data, valid, ready
      .Case<handshake::ChannelType>([](handshake::ChannelType chanType) {
        return SmallVector<std::pair<SignalKind, Type>>{
            std::make_pair(DATA_SIGNAL, chanType.getDataType()),
            std::make_pair(VALID_SIGNAL,
                           IntegerType::get(chanType.getContext(), 1)),
            std::make_pair(READY_SIGNAL,
                           IntegerType::get(chanType.getContext(), 1))};
      })
      // Case for control type where we unbundle into valid, ready
      .Case<handshake::ControlType>([](handshake::ControlType ctrlType) {
        return SmallVector<std::pair<SignalKind, Type>>{
            std::make_pair(VALID_SIGNAL,
                           IntegerType::get(ctrlType.getContext(), 1)),
            std::make_pair(READY_SIGNAL,
                           IntegerType::get(ctrlType.getContext(), 1))};
      })
      // Case for memref type where we extract the element type as data,
      // valid, ready
      .Case<MemRefType>([](MemRefType memType) {
        return SmallVector<std::pair<SignalKind, Type>>{
            std::make_pair(DATA_SIGNAL, memType.getElementType()),
            std::make_pair(VALID_SIGNAL,
                           IntegerType::get(memType.getContext(), 1)),
            std::make_pair(READY_SIGNAL,
                           IntegerType::get(memType.getContext(), 1))};
      })
      .Default([&](Type t) {
        return SmallVector<std::pair<SignalKind, Type>>{
            std::make_pair(DATA_SIGNAL, t)};
      });
}
// Function to unbundle the ports of an handshake operation into hw module
// Unbundling handshake op ports into hw module ports since handshake ops
// have bundled channel types (composed of data, valid, ready) while hw
// modules have flat ports
void unbundleOpPorts(
    Operation *op,
    SmallVector<SmallVector<std::pair<SignalKind, Type>>> &inputPorts,
    SmallVector<SmallVector<std::pair<SignalKind, Type>>> &outputPorts) {
  // If the operation is a handshake function, the ports are extracted
  // differently
  if (isa<handshake::FuncOp>(op)) {
    handshake::FuncOp funcOp = cast<handshake::FuncOp>(op);
    // Extract ports from the function type
    for (auto arg : funcOp.getArguments()) {
      SmallVector<std::pair<SignalKind, Type>> inputUnbundled =
          unbundleType(arg.getType());
      inputPorts.push_back(inputUnbundled);
    }
    for (auto resultType : funcOp.getResultTypes()) {
      SmallVector<std::pair<SignalKind, Type>> outputUnbundled =
          unbundleType(resultType);
      outputPorts.push_back(outputUnbundled);
    }
    return;
  }
  for (auto input : op->getOperands()) {
    // Unbundle the input type depending on its actual type
    SmallVector<std::pair<SignalKind, Type>> inputUnbundled =
        unbundleType(input.getType());
    inputPorts.push_back(inputUnbundled);
  }
  for (auto result : op->getResults()) {
    // Unbundle the output type depending on its actual type
    SmallVector<std::pair<SignalKind, Type>> outputUnbundled =
        unbundleType(result.getType());
    outputPorts.push_back(outputUnbundled);
  }
}

// Type converter to unbundle Handshake types
class ChannelUnbundlingTypeConverter : public TypeConverter {
public:
  ChannelUnbundlingTypeConverter() {
    addConversion([](Type type,
                     SmallVectorImpl<Type> &results) -> LogicalResult {
      // Unbundle the type into its components
      SmallVector<Type> unbundledTypes;
      SmallVector<std::pair<SignalKind, Type>> unbundled = unbundleType(type);
      for (auto &pair : unbundled) {
        unbundledTypes.push_back(pair.second);
      }
      results.append(unbundledTypes.begin(), unbundledTypes.end());
      return success();
    });

    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;
      return inputs[0];
    });

    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;
      return inputs[0];
    });
  }
};

// Utility function to insert a string before the first occurrence of '['
std::string insertBeforeBracket(const std::string &s,
                                const std::string &toInsert) {
  std::string result = s;
  std::size_t pos = result.find('[');
  // Find first occurrence of '[' and insert the string before it
  if (pos != std::string::npos) {
    result.insert(pos, toInsert);
  } else {
    // If no '[', just append at the end
    result += toInsert;
  }
  return result;
}

// Utility function to get the string formatted in the desired portname[idx]
// format
// If the root name already contains an index, the old index is linearized and
// added to the new one using the width input
std::string formatPortName(const std::string &rootName,
                           const std::optional<unsigned> &index,
                           unsigned width = 0) {
  if (index.has_value()) {
    unsigned newIndex = index.value();
    std::string newRootName = rootName;
    // Check if there is already present an index
    std::regex pattern(R"((\w+)\[(\d+)\])");
    // We use regex to identify this pattern
    std::smatch matches;
    if (std::regex_match(rootName, matches, pattern)) {
      // If the pattern matches, assert the width is not 0
      assert(width != 0 &&
             "the width of the signal cannot be 0 if it is an array");
      // Linearize old index
      std::string oldIndex = matches[2].str();
      unsigned oldIndexLinearized = std::stoi(oldIndex) * width;
      // Add it to new one
      newIndex += oldIndexLinearized;
      newRootName = matches[1].str();
    }
    return newRootName + "[" + std::to_string(newIndex) + "]";
  }
  return rootName;
}

// Function to get the list of names of hardware ports using the format:
// portname[idx] for each port part of an array or portname for single ports
std::pair<SmallVector<std::string>, SmallVector<std::string>>
getHWModulePortNames(Operation *op) {
  SmallVector<std::string> inputPortNames;
  SmallVector<std::string> outputPortNames;
  // Check if the operation implements NamedIOInterface to get port names
  auto namedIO = dyn_cast<handshake::NamedIOInterface>(op);
  if (!namedIO) {
    // If the operation is the handshake function, just name it in a default
    // way
    if (auto funcOp = dyn_cast<handshake::FuncOp>(op)) {
      for (auto [idx, _] : llvm::enumerate(funcOp.getArguments())) {
        inputPortNames.push_back("in" + std::to_string(idx));
      }
      for (auto [idx, _] : llvm::enumerate(funcOp.getResultTypes())) {
        outputPortNames.push_back("out" + std::to_string(idx));
      }
      return {inputPortNames, outputPortNames};
    }
    // It is essential to retrieve the port names to correctly connect signals
    // during unbundling. If the operation does not implement this interface,
    // we cannot proceed.
    llvm::errs() << op->getName() << " does not implement NamedIOInterface\n";
    assert(false && "Operation does not implement NamedIOInterface");
  }
  std::regex pattern(R"((\w+)_(\d+))");
  // Elaborate the portname in case it contains the following format
  // "portname_index" where index is an integer representing the index-th
  // input. However, in order to respect the format of port names in the blif
  // files, we have to change this format into "portname[index]".
  // However, the pattern "portname_index" might indicate an index with size
  // 0. For this reason we need to keep track of the size of each port.
  // Iterate over input operands to get their names
  for (auto [idx, _] : llvm::enumerate(op->getOperands())) {
    std::string portNameStr = namedIO.getOperandName(idx).c_str();
    // We use regex to identify this pattern
    std::string elaboratedPortName;
    std::smatch matches;
    if (std::regex_match(portNameStr, matches, pattern)) {
      // If the pattern matches, construct the new port name
      std::string rootName = matches[1].str();
      unsigned index = std::stoi(matches[2].str());
      if (index == 0) {
        // If index is 0, keep the original port name in case there is only
        // one port
        elaboratedPortName = rootName;
      } else {
        elaboratedPortName = formatPortName(rootName, index);
      }
      // If index == 1, fix the name of index 0 to follow the same format
      if (index == 1) {
        std::string firstPortName = rootName;
        firstPortName = formatPortName(firstPortName, 0);
        // Replace the port named only rootName with the new one
        int idxToReplace = -1;
        for (auto [i, name] : llvm::enumerate(inputPortNames)) {
          if (name == rootName) {
            idxToReplace = i;
            break;
          }
        }
        assert(idxToReplace != -1 && "Could not find the first port to rename");
        inputPortNames[idxToReplace] = firstPortName;
      }
    } else {
      // If the pattern does not match, keep the original port name
      elaboratedPortName = portNameStr;
    }
    inputPortNames.push_back(elaboratedPortName);
  }
  // Iterate over outputs to replicate the same logic
  for (auto [idx, _] : llvm::enumerate(op->getResults())) {
    std::string portNameStr = namedIO.getResultName(idx).c_str();
    std::string elaboratedPortName;
    std::smatch matches;
    if (std::regex_match(portNameStr, matches, pattern)) {
      std::string rootName = matches[1].str();
      unsigned index = std::stoi(matches[2].str());
      if (index == 0) {
        elaboratedPortName = rootName;
      } else {
        elaboratedPortName = formatPortName(rootName, index);
      }
      if (index == 1) {
        std::string firstPortName = rootName;
        firstPortName = formatPortName(firstPortName, 0);
        int idxToReplace = -1;
        for (auto [i, name] : llvm::enumerate(outputPortNames)) {
          if (name == rootName) {
            idxToReplace = i;
            break;
          }
        }
        assert(idxToReplace != -1 && "Could not find the first port to rename");
        outputPortNames[idxToReplace] = firstPortName;
      }
    } else {
      elaboratedPortName = portNameStr;
    }
    outputPortNames.push_back(elaboratedPortName);
  }

  return {inputPortNames, outputPortNames};
}

// Function to get the HW Module input and output port info from an handshake
// operation
void getHWModulePortInfo(Operation *op, SmallVector<hw::PortInfo> &hwInputPorts,
                         SmallVector<hw::PortInfo> &hwOutputPorts) {
  MLIRContext *ctx = op->getContext();
  // Get the port names of the operation
  auto [inputPortNames, outputPortNames] = getHWModulePortNames(op);
  // Unbundle the operation ports
  SmallVector<SmallVector<std::pair<SignalKind, Type>>> unbundledInputPorts;
  SmallVector<SmallVector<std::pair<SignalKind, Type>>> unbundledOutputPorts;
  unbundleOpPorts(op, unbundledInputPorts, unbundledOutputPorts);
  // Iterate over each set of unbundled input ports and create hw port info
  // from them
  for (auto [idx, inputPortSet] : llvm::enumerate(unbundledInputPorts)) {
    std::string portName = inputPortNames[idx];
    // Fill in the hw port info for inputs
    for (auto port : inputPortSet) {
      std::string name;
      Type type;
      // Unpack the port info as name and type
      name = port.first == DATA_SIGNAL ? portName
             : port.first == VALID_SIGNAL
                 ? insertBeforeBracket(portName, "_valid")
                 : insertBeforeBracket(portName, "_ready");
      type = port.second;
      // Create the hw port info and add it to the list
      hwInputPorts.push_back(
          hw::PortInfo{hw::ModulePort{StringAttr::get(ctx, name), type,
                                      hw::ModulePort::Direction::Input},
                       idx});
    }
  }
  // Iterate over each set of unbundled output ports and create hw port info
  // from them
  for (auto [idx, outputPortSet] : llvm::enumerate(unbundledOutputPorts)) {
    std::string portName = outputPortNames[idx];
    // Fill in the hw port info for outputs
    for (auto port : outputPortSet) {
      std::string name;
      Type type;
      // Unpack the port info as name and type
      name = port.first == DATA_SIGNAL ? portName
             : port.first == VALID_SIGNAL
                 ? insertBeforeBracket(portName, "_valid")
                 : insertBeforeBracket(portName, "_ready");
      type = port.second;
      // Create the hw port info and add it to the list
      hwOutputPorts.push_back(
          hw::PortInfo{hw::ModulePort{StringAttr::get(ctx, name), type,
                                      hw::ModulePort::Direction::Output},
                       idx});
    }
  }
}

// Function to create a cast operation between a bundled type to an unbundled
// type
UnrealizedConversionCastOp
createCastBundledToUnbundled(Value input, Location loc,
                             PatternRewriter &rewriter) {
  SmallVector<std::pair<SignalKind, Type>> unbundledTypes =
      unbundleType(input.getType());
  SmallVector<Type> unbundledTypeList;
  for (auto &pair : unbundledTypes) {
    unbundledTypeList.push_back(pair.second);
  }
  TypeRange unbundledTypeRange(unbundledTypeList);
  auto cast = rewriter.create<UnrealizedConversionCastOp>(
      loc, unbundledTypeRange, input);
  return cast;
}

// Function to create a cast operation for each result of an unbundled
// operation when converting an original bundled operation to an unbundled
// one. This cast is essential to connect the unbundled operation back to the
// rest of the circuit which is potentially still bundled
SmallVector<UnrealizedConversionCastOp>
createCastResultsUnbundledOp(TypeRange originalOpResultType,
                             SmallVector<Value> unbundledValues, Location loc,
                             PatternRewriter &rewriter) {
  // Assert the number of results of unbundled op is larger or equal to the
  // number of results in original operation
  assert(unbundledValues.size() >= originalOpResultType.size() &&
         "the number of results of the unbundled operation must be larger or "
         "equal to the original operation");
  SmallVector<UnrealizedConversionCastOp> castsOps;
  unsigned resultIdx = 0;
  // Iterate over each result of the original operation
  for (unsigned i = 0; i < originalOpResultType.size(); ++i) {
    unsigned numUnbundledTypes = 0;
    // Identify how many unbundled types correspond to this result
    SmallVector<std::pair<SignalKind, Type>> unbundledTypes =
        unbundleType(originalOpResultType[i]);
    numUnbundledTypes = unbundledTypes.size();
    // Get a slice of the unbundled values corresponding to this result from
    // the unbundled operation
    SmallVector<Value> unbundledValuesSlice(unbundledValues.begin() + resultIdx,
                                            unbundledValues.begin() +
                                                resultIdx + numUnbundledTypes);
    // Create the cast from unbundled values to the original bundled type
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{originalOpResultType[i]}, unbundledValuesSlice);
    // Collect the created cast
    castsOps.push_back(cast);
    resultIdx += numUnbundledTypes;
  }
  return castsOps;
}

// Function to convert a handshake function operation into an hw module
// operation
// It is divided into three parts: first, create the hw module definition if
// it does not already exist, then move the function body of the handshake
// function into the hw module, and finally create the output operation for
// the hw module
hw::HWModuleOp convertFuncOpToHWModule(handshake::FuncOp funcOp,
                                       ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = funcOp.getContext();
  // Obtain the hw module port info by unbundling the function ports
  SmallVector<hw::PortInfo> hwInputPorts;
  SmallVector<hw::PortInfo> hwOutputPorts;
  getHWModulePortInfo(funcOp, hwInputPorts, hwOutputPorts);
  hw::ModulePortInfo portInfo(hwInputPorts, hwOutputPorts);
  // Create the hw module operation if it does not already exist
  auto modOp = funcOp->getParentOfType<mlir::ModuleOp>();
  SymbolTable symbolTable(modOp);
  StringRef uniqueOpName = getUniqueName(funcOp);
  StringAttr moduleName = StringAttr::get(ctx, uniqueOpName);
  hw::HWModuleOp hwModule = symbolTable.lookup<hw::HWModuleOp>(moduleName);

  // If it does not exist, create it
  if (!hwModule) {
    // IMPORTANT: modules must be created at module scope
    rewriter.setInsertionPointToStart(modOp.getBody());

    hwModule =
        rewriter.create<hw::HWModuleOp>(funcOp.getLoc(), moduleName, portInfo);

    // Inline the function body of the handshake function into the hw module
    // body
    Block *funcBlock = funcOp.getBodyBlock();
    Block *modBlock = hwModule.getBodyBlock();
    Operation *termOp = modBlock->getTerminator();
    ValueRange modBlockArgs = modBlock->getArguments();
    // Set insertion point inside the module body
    rewriter.setInsertionPointToStart(modBlock);
    // There must be a match between the arguments of the hw module and the
    // handshake function to be able to inline the function body of the
    // handshake function in the hw module. However, the arguments of the
    // handshake function are bundled and the ones of the hw module are
    // unbundled. For this reason, create a cast for each argument of the hw
    // module. The output of the casts will be used to inline the function
    // body First create casts for each argument of the hw module
    SmallVector<Value> castedArgs;
    TypeRange funcArgTypes = funcOp.getArgumentTypes();
    SmallVector<UnrealizedConversionCastOp> castsArgs =
        createCastResultsUnbundledOp(funcArgTypes, modBlockArgs,
                                     funcOp.getLoc(), rewriter);
    // Collect the results of the casts
    for (auto castOp : castsArgs) {
      // Assert that each cast has only one result
      assert(castOp.getNumResults() == 1 &&
             "each cast operation must have only one result");
      castedArgs.push_back(castOp.getResult(0));
    }
    // Inline the function body before the terminator of the hw module using
    // the casted arguments
    rewriter.inlineBlockBefore(funcBlock, termOp, castedArgs);

    // Finally, create the output operation for the hw module
    // Find the handshake.end operation in the inlined body which is the
    // terminator for handshake functions
    handshake::EndOp endOp;
    for (Operation &op : *hwModule.getBodyBlock()) {
      if (auto e = dyn_cast<handshake::EndOp>(op)) {
        endOp = e;
        break;
      }
    }

    assert(endOp && "Expected handshake.end after inlining");

    // Create a cast for each operand of the end operation to unbundle the
    // types which were previously handshake bundled types
    SmallVector<Value> hwOutputs;
    for (Value operand : endOp.getOperands()) {
      // Unbundle the operand type depending by creating a cast from the
      // original type to the unbundled types
      auto cast =
          createCastBundledToUnbundled(operand, endOp->getLoc(), rewriter);
      hwOutputs.append(cast.getResults().begin(), cast.getResults().end());
    }
    // Create the output operation for the hw module
    rewriter.setInsertionPointToEnd(endOp->getBlock());
    rewriter.replaceOpWithNewOp<hw::OutputOp>(endOp, hwOutputs);
    // Remove any old output operation if present (should not be the case
    // after replacing the end op)
    Operation *oldOutput = nullptr;
    for (Operation &op : *hwModule.getBodyBlock()) {
      if (auto out = dyn_cast<hw::OutputOp>(op)) {
        if (out.getOperands().empty()) {
          oldOutput = out;
          break;
        }
      }
    }
    if (oldOutput)
      rewriter.eraseOp(oldOutput);
    // Remove the original handshake function operation
    rewriter.eraseOp(funcOp);
  }
  return hwModule;
}

// Function to instantiate synth or hw operations inside an hw module
// describing the behavior of the original handshake operation
LogicalResult
instantiateSynthOpInHWModule(hw::HWModuleOp hwModule,
                             ConversionPatternRewriter &rewriter) {
  // Set insertion point to the start of the hw module body
  rewriter.setInsertionPointToStart(hwModule.getBodyBlock());
  // Collect inputs values of the hw module
  SmallVector<Value> synthInputs;
  for (auto arg : hwModule.getBodyBlock()->getArguments()) {
    synthInputs.push_back(arg);
  }
  // Collect the types of the hardware modules outputs
  SmallVector<Type> synthOutputTypes;
  for (auto &outputPort : hwModule.getPortList()) {
    if (outputPort.isOutput())
      synthOutputTypes.push_back(outputPort.type);
  }
  TypeRange synthOutputsTypeRange(synthOutputTypes);
  // Connect the outputs of the synth operation to the outputs of the hw
  // module
  Operation *synthTerminator = hwModule.getBodyBlock()->getTerminator();
  // Create the synth subcircuit operation inside the hw module
  synth::SubcktOp synthInstOp = rewriter.create<synth::SubcktOp>(
      synthTerminator->getLoc(), synthOutputsTypeRange, synthInputs,
      "synth_subckt");
  synthTerminator->setOperands(synthInstOp.getResults());
  return success();
}

// Function to convert an handshake operation into an hw module operation
// It is divided into two parts: first, create the hw module definition if it
// does not already exist, then instantiate the hw instance of the module and
// replace the original operation with it
hw::HWModuleOp convertOpToHWModule(Operation *op,
                                   ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = op->getContext();
  // Obtain the hw module port info by unbundling the operation ports
  SmallVector<hw::PortInfo> hwInputPorts;
  SmallVector<hw::PortInfo> hwOutputPorts;
  getHWModulePortInfo(op, hwInputPorts, hwOutputPorts);
  hw::ModulePortInfo portInfo(hwInputPorts, hwOutputPorts);
  // Create the hw module operation if it does not already exist
  auto modOp = op->getParentOfType<mlir::ModuleOp>();
  SymbolTable symbolTable(modOp);
  StringRef uniqueOpName = getUniqueName(op);
  StringAttr moduleName = StringAttr::get(ctx, uniqueOpName);
  hw::HWModuleOp hwModule = symbolTable.lookup<hw::HWModuleOp>(moduleName);

  // If it does not exist, create it
  if (!hwModule) {
    // IMPORTANT: modules must be created at module scope
    rewriter.setInsertionPointToStart(modOp.getBody());

    hwModule =
        rewriter.create<hw::HWModuleOp>(op->getLoc(), moduleName, portInfo);
    // Instantiate the corresponding synth operation inside the hw module
    // definition
    if (failed(instantiateSynthOpInHWModule(hwModule, rewriter)))
      return nullptr;
    // Copy attribute for blif data path from the original operation to the hw
    // module
    BLIFImplInterface blifInterfaceOp = dyn_cast<BLIFImplInterface>(op);
    BLIFImplInterface blifInterfaceModule =
        dyn_cast<BLIFImplInterface>(hwModule);
    if (auto blifAttr = blifInterfaceOp.getBLIFImpl()) {
      blifInterfaceModule.setBLIFImpl(blifAttr);
    }
  }

  // Create the hw instance and replace the original operation with it
  rewriter.setInsertionPointAfter(op);
  // First, collect the operands for the hw instance
  SmallVector<Value> operandValues;
  for (auto operand : op->getOperands()) {
    auto definingOp = operand.getDefiningOp();
    // Check if the operation generating the operand is a handshake operation
    // or a block argument
    bool isBlockArg = isa<BlockArgument>(operand);
    if ((definingOp &&
         definingOp->getDialect()->getNamespace() ==
             handshake::HandshakeDialect::getDialectNamespace()) ||
        isBlockArg) {
      // If so, create an unrealized conversion cast to unbundle the operand
      // into its components to connect it to the input ports of the hw
      // instance
      auto cast = createCastBundledToUnbundled(operand, op->getLoc(), rewriter);
      // Append all cast results to the operand values
      operandValues.append(cast.getResults().begin(), cast.getResults().end());
      continue;
    }
    // If the operand is already unbundled, just use it directly
    operandValues.push_back(operand);
  }
  // Then, create the hw instance operation
  hw::InstanceOp hwInstOp = rewriter.create<hw::InstanceOp>(
      op->getLoc(), hwModule,
      StringAttr::get(ctx, uniqueOpName.str() + "_inst"), operandValues);
  // Create a cast for each group of unbundled results
  SmallVector<UnrealizedConversionCastOp> castsOpResults =
      createCastResultsUnbundledOp(op->getResultTypes(), hwInstOp->getResults(),
                                   op->getLoc(), rewriter);
  // Collect the results of the casts
  SmallVector<Value> castsResults;
  for (auto castOp : castsOpResults) {
    // Assert that each cast has only one result
    assert(castOp.getNumResults() == 1 &&
           "each cast operation must have only one result");
    castsResults.push_back(castOp.getResult(0));
  }
  // Assert that the number of cast results matches the number of original
  // operation results
  assert(castsResults.size() == op->getNumResults());

  //  Replace all uses of the original operation with the new hw module
  rewriter.replaceOp(op, castsResults);

  return hwModule;
}

// Conversion pattern to convert a generic handshake operation into an hw
// module operation
// This is used only for operation inside the handshake function, since the
// function itself is converted separately
namespace {
template <typename T>
class ConvertToHWMod : public OpConversionPattern<T> {
public:
  using OpConversionPattern<T>::OpConversionPattern;
  using OpAdaptor = typename T::Adaptor;
  LogicalResult
  matchAndRewrite(T op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

template <typename T>
LogicalResult
ConvertToHWMod<T>::matchAndRewrite(T op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {

  // Convert the operation into an hw module operation
  hw::HWModuleOp newOp = convertOpToHWModule(op, rewriter);
  if (!newOp)
    return failure();
  return success();
}

// Conversion pattern to convert handshake function operation into an hw
// module operation
namespace {
class ConvertFuncToHWMod : public OpConversionPattern<handshake::FuncOp> {
public:
  using OpConversionPattern<handshake::FuncOp>::OpConversionPattern;
  using OpAdaptor = typename handshake::FuncOp::Adaptor;
  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
ConvertFuncToHWMod::matchAndRewrite(handshake::FuncOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {

  // Convert the function operation into an hw module operation
  hw::HWModuleOp newOp = convertFuncOpToHWModule(op, rewriter);
  if (!newOp)
    return failure();
  return success();
}

// Function to remove unrealized conversion casts after conversion
LogicalResult removeUnrealizedConversionCasts(mlir::ModuleOp modOp) {
  SmallVector<UnrealizedConversionCastOp> castsToErase;
  // Walk through all unrealized conversion casts in the module
  modOp.walk([&](UnrealizedConversionCastOp castOp1) {
    // Consider only the following pattern op/arg -> cast (1) -> cast (2)
    // -> op where there could be multiple cast2
    // Check if the input of the cast is another unrealized
    // conversion cast
    auto inputCastOp =
        castOp1.getOperand(0).getDefiningOp<UnrealizedConversionCastOp>();
    if (inputCastOp) {
      // Skip this cast since it will be handled when processing the
      // input cast but first assert that it is not followed by any other
      // cast. If this is the case, there is a issue in the code since there
      // are 3 casts in chain and this break the assumption of the code
      assert(llvm::none_of(castOp1->getUsers(),
                           [](Operation *user) {
                             return isa<UnrealizedConversionCastOp>(user);
                           }) &&
             "unrealized conversion cast removal failed due to chained casts");
      return;
    }
    // Check that the output/s of the cast are used by only unrealized
    // conversion casts
    bool allUsersAreCasts = true;
    SmallVector<UnrealizedConversionCastOp> vecCastOps2;
    for (auto result : castOp1.getResults()) {
      for (auto &use : result.getUses()) {
        if (!isa<UnrealizedConversionCastOp>(use.getOwner())) {
          allUsersAreCasts = false;
        } else {
          vecCastOps2.push_back(
              cast<UnrealizedConversionCastOp>(use.getOwner()));
        }
      }
      if (!allUsersAreCasts)
        break;
    }
    if (!allUsersAreCasts) {
      // This breaks assumption that casts are chained in pairs
      castOp1.emitError()
          << "unrealized conversion cast removal failed due to complex "
             "usage pattern";
      return;
    }
    // Replace all uses of the user cast op results with the original
    // cast op inputs
    // Assert that the number of inputs and outputs match
    for (auto castOp2 : vecCastOps2) {
      if (castOp1->getNumOperands() != castOp2->getNumResults()) {
        castOp1.emitError()
            << "unrealized conversion cast removal failed due to "
               "mismatched number of operands and results";
        return;
      }
      for (auto [idx, result] : llvm::enumerate(castOp2.getResults())) {
        result.replaceAllUsesWith(castOp1->getOperand(idx));
      }
      // Add the cast to the list of casts to remove
      castsToErase.push_back(castOp2);
    }
    // Add the cast to the list of casts to remove
    castsToErase.push_back(castOp1);
  });
  // Erase all collected casts
  for (auto castOp : castsToErase) {
    castOp.erase();
  }

  // Check that there are no more unrealized conversion casts
  bool hasCasts = false;
  modOp.walk([&](UnrealizedConversionCastOp castOp) {
    hasCasts = true;
    llvm::errs() << "Remaining unrealized conversion cast: " << castOp << "\n";
  });

  if (hasCasts) {
    modOp.emitError()
        << "unrealized conversion cast removal failed due to remaining "
           "casts";
    return failure();
  }
  return success();
}

// Function to unbundle all handshake type in a handshake function operation
LogicalResult unbundleAllHandshakeTypes(ModuleOp modOp, MLIRContext *ctx) {

  // This function executes the unbundling conversion in three steps:
  // 1) Convert all handshake operations (except for function and end ops)
  //    into hw module operations connecting them with unrealized conversion
  //    casts
  // 2) Convert the handshake function operation into an hw module operation
  // 3) Remove all unrealized conversion casts by connecting directly the
  //    inputs and outputs of the hw module instances

  // Step 1: Apply conversion patterns to convert each handshake operation
  // into an hw module operation
  RewritePatternSet patterns(ctx);
  ChannelUnbundlingTypeConverter typeConverter;
  ConversionTarget target(*ctx);
  target.addLegalDialect<synth::SynthDialect>();
  target.addLegalDialect<hw::HWDialect>();
  // Add casting as legal
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addIllegalDialect<handshake::HandshakeDialect>();
  // In the first step, we convert all handshake operations into hw module
  // operations, except for the function and end operations which are kept
  // to convert in the next step
  target.addLegalOp<handshake::FuncOp>();
  target.addLegalOp<handshake::EndOp>();
  patterns.insert<
      ConvertToHWMod<handshake::BufferOp>, ConvertToHWMod<handshake::NDWireOp>,
      ConvertToHWMod<handshake::ConditionalBranchOp>,
      ConvertToHWMod<handshake::BranchOp>, ConvertToHWMod<handshake::MergeOp>,
      ConvertToHWMod<handshake::ControlMergeOp>,
      ConvertToHWMod<handshake::MuxOp>, ConvertToHWMod<handshake::JoinOp>,
      ConvertToHWMod<handshake::BlockerOp>, ConvertToHWMod<handshake::SourceOp>,
      ConvertToHWMod<handshake::ConstantOp>, ConvertToHWMod<handshake::SinkOp>,
      ConvertToHWMod<handshake::ForkOp>, ConvertToHWMod<handshake::LazyForkOp>,
      ConvertToHWMod<handshake::LoadOp>, ConvertToHWMod<handshake::StoreOp>,
      ConvertToHWMod<handshake::ReadyRemoverOp>,
      ConvertToHWMod<handshake::ValidMergerOp>,
      ConvertToHWMod<handshake::SharingWrapperOp>,
      ConvertToHWMod<handshake::MemoryControllerOp>,

      // Arith operations
      ConvertToHWMod<handshake::AddFOp>, ConvertToHWMod<handshake::AddIOp>,
      ConvertToHWMod<handshake::AndIOp>, ConvertToHWMod<handshake::CmpFOp>,
      ConvertToHWMod<handshake::CmpIOp>, ConvertToHWMod<handshake::DivFOp>,
      ConvertToHWMod<handshake::DivSIOp>, ConvertToHWMod<handshake::DivUIOp>,
      ConvertToHWMod<handshake::RemSIOp>, ConvertToHWMod<handshake::ExtSIOp>,
      ConvertToHWMod<handshake::ExtUIOp>, ConvertToHWMod<handshake::MulFOp>,
      ConvertToHWMod<handshake::MulIOp>, ConvertToHWMod<handshake::NegFOp>,
      ConvertToHWMod<handshake::OrIOp>, ConvertToHWMod<handshake::SelectOp>,
      ConvertToHWMod<handshake::ShLIOp>, ConvertToHWMod<handshake::ShRSIOp>,
      ConvertToHWMod<handshake::ShRUIOp>, ConvertToHWMod<handshake::SubFOp>,
      ConvertToHWMod<handshake::SubIOp>, ConvertToHWMod<handshake::TruncIOp>,
      ConvertToHWMod<handshake::TruncFOp>, ConvertToHWMod<handshake::XOrIOp>,
      ConvertToHWMod<handshake::SIToFPOp>, ConvertToHWMod<handshake::UIToFPOp>,
      ConvertToHWMod<handshake::FPToSIOp>, ConvertToHWMod<handshake::ExtFOp>,
      ConvertToHWMod<handshake::AbsFOp>, ConvertToHWMod<handshake::MaxSIOp>,
      ConvertToHWMod<handshake::MaxUIOp>, ConvertToHWMod<handshake::MinSIOp>,
      ConvertToHWMod<handshake::MinUIOp>,

      // Speculative operations
      ConvertToHWMod<handshake::SpecCommitOp>,
      ConvertToHWMod<handshake::SpecSaveOp>,
      ConvertToHWMod<handshake::SpecSaveCommitOp>,
      ConvertToHWMod<handshake::SpeculatorOp>,
      ConvertToHWMod<handshake::SpeculatingBranchOp>,
      ConvertToHWMod<handshake::NonSpecOp>>(typeConverter, ctx);
  if (failed(applyPartialConversion(modOp, target, std::move(patterns))))
    return failure();

  // Step 2: Convert the handshake function operation into
  // an hw module operation and the corresponding terminator into an hw
  // terminator
  RewritePatternSet funcPatterns(ctx);
  ConversionTarget funcTarget(*ctx);
  funcTarget.addLegalDialect<synth::SynthDialect>();
  funcTarget.addLegalDialect<hw::HWDialect>();
  // Add casting as legal
  funcTarget.addLegalOp<UnrealizedConversionCastOp>();
  funcTarget.addIllegalDialect<handshake::HandshakeDialect>();
  funcPatterns.insert<ConvertFuncToHWMod>(typeConverter, ctx);
  if (failed(
          applyPartialConversion(modOp, funcTarget, std::move(funcPatterns))))
    return failure();

  // Step 3: remove all unrealized conversion casts
  // Execute without conversion function but walking the module
  if (failed(removeUnrealizedConversionCasts(modOp)))
    return failure();
  return success();
}

// ------------------------------------------------------------------
// Instantiation of the synth operations in the hw modules and conversion of hw
// instances to synth operations
// ------------------------------------------------------------------

// Function to read a blif file and generate the synth circuit depending on the
// blif description. It uses the blif importer class to read the blif file and
// generate the synth circuit. The core functionality is checking the
// consistency of the blif file with the hw module and hw instance.
LogicalResult generateSynthCircuitFromBlif(StringRef blifFilePath, Location loc,
                                           hw::InstanceOp hwInst,
                                           hw::HWModuleOp hwModule,
                                           OpBuilder &builder) {
  builder.setInsertionPoint(hwInst);
  // The following function reads the blif file specified by blifFilePath
  // and generates the synth circuit defined in it.

  // Generate blif importer
  BlifImporter blifImporter(blifFilePath);
  // Read input and output ports from the blif file
  if (failed(blifImporter.extractBlifModuleHeader())) {
    llvm::errs() << "Failed to read the blif file header." << "\n";
    return failure();
  }

  std::string moduleName = blifImporter.getModuleName();
  SmallVector<std::string> inputPortsNames = blifImporter.getInputPortsNames();
  SmallVector<std::string> outputPortsNames =
      blifImporter.getOutputPortsNames();

  // Collect the inputs of the hw instance in a map to pass to the function that
  // generates the synth circuit
  llvm::StringMap<Value> hwInstInputs;

  // Collect the outputs of the hw instance in a vector to check that they match
  // the output ports in the blif file
  SmallVector<StringRef> hwInstOutputs;

  // Collect hw instance inputs
  for (auto &port : hwModule.getPortList()) {
    if (port.isInput()) {
      // Check the port is not already in the map
      assert(!hwInstInputs.count(builder.getStringAttr(port.name.getValue())) &&
             "port name already exists in the hw instance inputs map");
      // Check bounds of argNum
      unsigned argNum = port.argNum;
      assert(argNum < hwInst.getNumOperands() &&
             "Instance operand index out of bounds");
      Value hwInstInput = hwInst.getOperand(argNum);
      assert(hwInstInput && "hw instance input value should not be null");
      hwInstInputs[port.name] = hwInstInput;
    }
  }

  // Collect hw instance outputs
  for (auto &port : hwModule.getPortList()) {
    if (port.isOutput()) {
      hwInstOutputs.push_back(port.name.getValue());
    }
  }

  // Check that the instance is fully connected
  assert((hwInst.getOperands().size() == hwModule.getNumInputPorts() &&
          hwInst.getResults().size() == hwModule.getNumOutputPorts()) &&
         "Cannot use positional operands on partially-connected instance");

  // Check the number of inputs in the blif file matches the number of input
  // ports of the hw module
  if (inputPortsNames.size() != hwModule.getNumInputPorts() ||
      inputPortsNames.size() != hwInstInputs.size()) {
    llvm::errs()
        << "The number of input ports in the blif file ("
        << inputPortsNames.size()
        << ") does not match the number of input ports in the hw module ("
        << hwModule.getNumInputPorts() << ")." << "\n";
    return failure();
  }

  // Check the number of outputs in the blif file matches the number of output
  // ports of the hw module
  if (outputPortsNames.size() != hwModule.getNumOutputPorts() ||
      outputPortsNames.size() != hwInstOutputs.size()) {
    llvm::errs()
        << "The number of output ports in the blif file ("
        << outputPortsNames.size()
        << ") does not match the number of output ports in the hw module ("
        << hwModule.getNumOutputPorts() << ")." << "\n";
    return failure();
  }

  Location hwInstanceloc = hwInst.getLoc();
  // Collect the outputs of the synth circuit
  SmallVector<Value> synthOutputs;
  // Generate the synth circuit from the blif file
  if (failed(blifImporter.generateSynthCircuitFromBlif(
          hwInstanceloc, hwInstInputs, outputPortsNames, synthOutputs,
          builder))) {
    llvm::errs() << "Failed to generate the synth circuit from the blif file."
                 << "\n";
    return failure();
  }

  assert(synthOutputs.size() == outputPortsNames.size() &&
         "The number of outputs of the synth circuit does not match the number "
         "of output ports in the blif file");

  // Replace hw instance outputs with the corresponding synth signals
  for (unsigned i = 0; i < hwInst.getNumResults(); ++i) {
    hwInst.getResult(i).replaceAllUsesWith(synthOutputs[i]);
  }
  // Erase the hw instance operation
  hwInst.erase();
  return success();
}

// Function to replace an instance with synth operations
// It manages the replacement of a single hw instance operation
LogicalResult convertHWInstanceToSynthOps(hw::InstanceOp op,
                                          OpBuilder &builder) {

  builder.setInsertionPoint(op);
  // Ensure that the blif path is specified in the hw module of the hw
  // instance operation
  SymbolTable symTable = SymbolTable(op->getParentOfType<ModuleOp>());
  hw::HWModuleOp hwModule = symTable.lookup<hw::HWModuleOp>(op.getModuleName());
  if (!hwModule) {
    llvm::errs() << "could not find hw module for instance: " << op << "\n";
    return failure();
  }
  BLIFImplInterface blifInterfaceModule = dyn_cast<BLIFImplInterface>(hwModule);
  if (!blifInterfaceModule || !blifInterfaceModule.getBLIFImpl()) {
    llvm::errs() << "hw module for instance does not have blif path attribute: "
                 << hwModule << "\n";
    return failure();
  }
  // Get the blif path attribute
  mlir::StringAttr blifPathAttr = blifInterfaceModule.getBLIFImpl();
  StringRef blifFilePath = blifPathAttr.getValue();

  // Check if blifFilePath is empty
  if (blifFilePath == "") {
    // If this is the case, there is no blif for this specific module and it
    // should be replaced with a subckt operation
    // Create the synth subcircuit operation inside the hw module
    synth::SubcktOp synthInstOp = builder.create<synth::SubcktOp>(
        op->getLoc(), op.getResultTypes(), op.getOperands(), "synth_subckt");
    //  Collect results of the subckt operation
    SmallVector<Value> synthResults = synthInstOp.getResults();
    // For each output of the hw instance, replace with the corresponding
    // output of the synth subckt operation
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      op.getResult(i).replaceAllUsesWith(synthResults[i]);
    }
    // Erase the hw instance operation
    op.erase();
    return success();
  }

  // Read the blif path and extract the synth circuit
  if (failed(generateSynthCircuitFromBlif(blifFilePath, op.getLoc(), op,
                                          hwModule, builder))) {
    return failure();
  }

  return success();
}

// Function to convert hw instances into synth operations
LogicalResult convertHWInstancesToSynthOps(mlir::ModuleOp modOp,
                                           StringRef topModuleName,
                                           MLIRContext *ctx) {
  // The following function iterates through all the hw instances in the top
  // module and convert them into synth operations like registers,
  // combinational logic, etc. if possible. The description of the
  // implementation is defined in the path specified by the attribute
  // blifPathAttrStr on each hw module

  // Get hw module corresponding to the top module
  SymbolTable symTable(modOp);
  hw::HWModuleOp topHWModule = symTable.lookup<hw::HWModuleOp>(topModuleName);
  if (!topHWModule) {
    llvm::errs() << "could not find hw module for top module: " << topModuleName
                 << "\n";
    return failure();
  }

  // Collect all hw instances in the top module
  SmallVector<hw::InstanceOp> hwInstances;
  topHWModule.walk([&](hw::InstanceOp op) { hwInstances.push_back(op); });
  OpBuilder builder(ctx);
  // Convert each hw instance into synth operations
  for (hw::InstanceOp hwInst : hwInstances) {
    if (failed(convertHWInstanceToSynthOps(hwInst, builder))) {
      llvm::errs() << "Failed to convert hw instance to synth ops: " << hwInst
                   << "\n";
      return failure();
    }
  }
  return success();
}

// ------------------------------------------------------------------
// Main pass definition
// ------------------------------------------------------------------

namespace {
// The following pass converts handshake operations into synth operations
// It executes in multiple steps:
// 0) Check each handshake operation is marked with the path of the blif file
//    where its definition is located. This is done in a separate pass
//    (--mark-handshake-blif-impl).
// 1) Unbundle all handshake types used in the handshake function
// 2) Invert the direction of ready signals in all hw modules and hw instances
//    to follow the standard handshake protocol where ready signals go in the
//    opposite direction with respect to data and valid signals. Additionally,
//    data signals are unbundled into single-bit signals.
// 3) Convert hw instances into other synth operations like registers,
//    combinational logic, etc. if possible. The description of the
//    implementation is defined in the path specified by the attribute
//    blifPathAttrStr on each hw module
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
        llvm::errs() << "Handshake operation " << getUniqueName(op)
                     << " is not marked with a BLIF file path\n";
        return signalPassFailure();
      }
    });

    // Step 1: unbundle all handshake types in the handshake operations
    if (failed(unbundleAllHandshakeTypes(modOp, ctx)))
      return signalPassFailure();

    /*
    // Step 2: invert the direction of all ready signals in the hw modules
    // created from handshake operations. Additionally, unbundle data signals
    // into single-bit signals.
    //
    // Create on object of the SignalRewriter class to manage the
    // inversion
    SignalRewriter signalRewriter;
    // Set the name of the top function
    signalRewriter.setTopFunctionName(topModuleName);
    if (failed(signalRewriter.rewriteAllSignals(modOp)))
      return signalPassFailure();
    */

    // Step 3: Convert hw instances into other synth operations like
    // registers, combinational logic, etc. if possible. The description of
    // the implementation is defined in the path specified by the attribute
    // blifPathAttrStr on each hw module
    if (failed(convertHWInstancesToSynthOps(modOp, topModuleName, ctx)))
      return signalPassFailure();
    // Remove all hw modules that are different from the top function hw
    // module
    SmallVector<hw::HWModuleOp> hwModuleToErase;
    modOp.walk([&](hw::HWModuleOp op) {
      if (op.getName() != topModuleName) {
        hwModuleToErase.push_back(op);
      }
    });
    for (Operation *hwMod : hwModuleToErase) {
      hwMod->erase();
    }
  }
};

} // namespace