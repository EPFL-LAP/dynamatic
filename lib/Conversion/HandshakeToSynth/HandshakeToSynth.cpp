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

#include "dynamatic/Conversion/HandshakeToSynth.h"
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

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

#define DEBUG_TYPE "handshake-to-synth"

class SignalRewriter {
public:
  /// Rewrite a HW module interface and body to apply signal-direction and
  /// bit-level rewrites (e.g., ready inversion and bit unbundling).
  void rewriteHWModule(hw::HWModuleOp oldMod, ModuleOp parent,
                       SymbolTable &symTable,
                       DenseMap<StringRef, hw::HWModuleOp> &newHWModules,
                       DenseMap<StringRef, hw::HWModuleOp> &oldHWModules);

  /// Rewrite a HW instance to use a rewritten module and updated operands.
  void rewriteHWInstance(hw::InstanceOp oldInst, ModuleOp parent,
                         SymbolTable &symTable,
                         DenseMap<StringRef, hw::HWModuleOp> &newHWModules,
                         DenseMap<StringRef, hw::HWModuleOp> &oldHWModules);

  /// Get the name of the rewritten HW module (e.g., by appending a suffix).
  mlir::StringAttr getRewrittenModuleName(hw::HWModuleOp oldMod,
                                          mlir::MLIRContext *ctx);

  /// Get the mapping of an old input signal to its rewritten signals.
  SmallVector<Value> getInputSignalMapping(Value oldInputSignal,
                                           OpBuilder builder, Location loc);

  /// Update the mapping for an output signal group after a new instance is
  /// created.
  void updateOutputSignalMapping(Value oldResult, StringRef outputName,
                                 int oldOutputIdx, hw::HWModuleOp oldMod,
                                 hw::HWModuleOp newMod, hw::InstanceOp newInst);

  /// Rewrite all HW modules in the given MLIR module to apply signal
  /// restructuring (direction changes, bit unbundling, etc.).
  LogicalResult rewriteAllSignals(mlir::ModuleOp modOp);

  // Function to set the name of the top function
  void setTopFunctionName(StringRef topFunctionName) {
    assert(topFunctionName != "" &&
           "top function name cannot be set to an empty string");
    topFunction = topFunctionName;
  }

  // Function to get the name of the top function
  StringRef getTopFunctionName() {
    assert(
        topFunction != "" &&
        "top function name should be set before being able to get its value");
    return topFunction;
  }

private:
  // IMPORTANT: A fundamental assumption for these maps to work is that each
  // handshake channel connects uniquely one handshake unit to another one. If
  // this assumption is broken, the maps will not work correctly since one key
  // could correspond to multiple values.
  //
  // Maps to keep track of signal connections during instance rewriting.
  //
  // "Old" refers to the original HW instance, with potentially incorrect
  // signal directions and HW types that may have bitwidths larger than 1.
  // "New" refers to the rewritten HW instance, with corrected signal
  // directions and HW types normalized to 1-bit signals for each component.
  DenseMap<Value, SmallVector<Value>> oldModuleSignalToNewModuleSignalsMap;

  // Since the HW instances are rewritten in a recursive manner, it might be
  // possible that we need results of an instance that has not been created
  // yet. To handle this case, we create temporary HW constant operations
  // to hold places for these values. This map keeps track of these temporary
  // values grouped per original signal.
  DenseMap<Value, SmallVector<Value>> oldModuleSignalToTempValuesMap;

  // Map to connect the idx of the old output signal to the idxs of the new
  // output signals after rewriting.
  DenseMap<StringAttr, SmallVector<std::pair<unsigned, SmallVector<unsigned>>>>
      oldOutputIdxToNewOutputIdxMap;

  // Name of the top function
  StringRef topFunction = "";
  // Signal of the top function that refers to the clock signal of the top func
  Value clkSignalTop;
  // Signal of the top function that refers to the reset signal of the top func
  Value rstSignalTop;
};

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
    // Copy attribute for blif data path
    if (auto blifAttr = op->getAttrOfType<StringAttr>(blifPathAttrStr)) {
      hwModule->setAttr(blifPathAttrStr, blifAttr);
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
      ConvertToHWMod<handshake::NotOp>,
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
// Definition of functions of SignalRewriter class
// ------------------------------------------------------------------
// Function to get a new module name for the rewritten hw module
mlir::StringAttr
SignalRewriter::getRewrittenModuleName(hw::HWModuleOp oldMod,
                                       mlir::MLIRContext *ctx) {
  return mlir::StringAttr::get(ctx, oldMod.getName() + "_rewritten");
}

// Function to get the mapping of an input signal from the old module to the
// new module
SmallVector<Value> SignalRewriter::getInputSignalMapping(Value oldInputSignal,
                                                         OpBuilder builder,
                                                         Location loc) {
  auto it = oldModuleSignalToNewModuleSignalsMap.find(oldInputSignal);
  if (it != oldModuleSignalToNewModuleSignalsMap.end()) {
    return it->second;
  }
  // If there is no mapping, it means that the value is not yet
  // available since the instance that produces it has not been
  // rewritten yet. In this case, check if we already created a
  // temporary value for this ready signal
  auto tempIt = oldModuleSignalToTempValuesMap.find(oldInputSignal);
  if (tempIt != oldModuleSignalToTempValuesMap.end()) {
    return tempIt->second;
  }
  // Else, create a new temporary hw constant to hold the connection for each
  // bit
  SmallVector<Value> tempValues;
  Type signalType = oldInputSignal.getType();
  assert(signalType.isa<IntegerType>() &&
         "only integer types are supported for signals for now");
  unsigned bitWidth = signalType.cast<IntegerType>().getWidth();
  assert(bitWidth > 0 && "signal must have positive bit width");
  for (unsigned i = 0; i < bitWidth; ++i) {
    auto tempConst = builder.create<hw::ConstantOp>(
        loc, oldInputSignal.getType(),
        builder.getIntegerAttr(oldInputSignal.getType(), 0));
    tempValues.push_back(tempConst.getResult());
  }
  // Store the temporary value in the map
  oldModuleSignalToTempValuesMap[oldInputSignal] = tempValues;
  // Use the result of the constant as the operand
  return tempValues;
}

// Function to update the mapping between old module signals and new module
// signals after getting a new result value
void SignalRewriter::updateOutputSignalMapping(
    Value oldResult, StringRef outputName, int oldOutputIdx,
    hw::HWModuleOp oldMod, hw::HWModuleOp newMod, hw::InstanceOp newInst) {
  // Check if you can find the output idx of the oldResult in the list
  // oldOutputIdxToNewOutputIdxMap
  OpBuilder builder(oldMod);
  StringAttr oldModName = builder.getStringAttr(oldMod.getName());
  SmallVector<std::pair<unsigned, SmallVector<unsigned>>>
      oldOutIdxToNewOutIdxs = oldOutputIdxToNewOutputIdxMap[oldModName];
  SmallVector<unsigned> outputIdxsNewInst = {};
  for (auto &pair : oldOutIdxToNewOutIdxs) {
    if (oldOutputIdx != -1 &&
        pair.first == static_cast<unsigned>(oldOutputIdx)) {
      outputIdxsNewInst = pair.second;
      break;
    }
  }
  // If you cannot find it, use the output name to find it
  if (outputIdxsNewInst.empty()) {
    // The only case in which this is possible is when it is a ready signal
    assert(outputName.contains("_ready") &&
           "could not find output index mapping for non-ready signal");
    // Find the corresponding output index of the output in the new module
    for (auto &p : newMod.getPortList()) {
      StringRef portName = p.name.getValue();
      if (portName == outputName) {
        outputIdxsNewInst.push_back(p.argNum);
      }
    }
  }
  assert(!outputIdxsNewInst.empty() &&
         "could not find output port in new module");
  SmallVector<Value> newResults;
  for (unsigned idx : outputIdxsNewInst) {
    newResults.push_back(newInst->getResult(idx));
  }
  // Add mapping between old non-ready signal and new non-ready signal
  oldModuleSignalToNewModuleSignalsMap[oldResult] = newResults;
  // Check if any temporary value has been created for the new result
  // value by seeing if the old output value is in the map of temporary
  // values
  auto tempIt = oldModuleSignalToTempValuesMap.find(oldResult);
  if (tempIt != oldModuleSignalToTempValuesMap.end()) {
    assert(newResults.size() == tempIt->second.size() &&
           "mismatched number of bits between temporary value and new result");
    for (size_t i = 0; i < newResults.size(); ++i) {
      // Replace all uses of the temporary value with the new result value
      tempIt->second[i].replaceAllUsesWith(newResults[i]);
    }
    for (auto tempVal : tempIt->second) {
      // Remove the operation creating the temporary value
      tempVal.getDefiningOp()->erase();
    }
    // Remove the temporary value from the map
    oldModuleSignalToTempValuesMap.erase(tempIt);
  }
}

/// Rewrite an HW instance to use the rewritten module interface and operands.
/// This updates operand connections, reconstructs result groups, and updates
/// internal mapping structures used by the module-level rewriting.
void SignalRewriter::rewriteHWInstance(
    hw::InstanceOp oldInst, ModuleOp parent, SymbolTable &symTable,
    DenseMap<StringRef, hw::HWModuleOp> &newHWmodules,
    DenseMap<StringRef, hw::HWModuleOp> &oldHWmodules) {

  // This function executes the following steps:
  // 1. Check if the instance's module has already been rewritten. If not,
  //    rewrite it.
  // 2. Create a new instance operation with the new module and updated
  //    operands.
  // 3. Update the mapping between old signals and new signals for the outputs
  //    of the new hw instance.

  // Step 1: Check if the instance's module has already been rewritten
  OpBuilder builder(parent);
  StringRef moduleName = oldInst.getModuleName();
  // If it has not been processed yet, rewrite it
  if (!newHWmodules.count(moduleName)) {
    hw::HWModuleOp oldMod = symTable.lookup<hw::HWModuleOp>(moduleName);
    if (oldMod) {
      rewriteHWModule(oldMod, parent, symTable, newHWmodules, oldHWmodules);
    }
  }

  // Step 2: Create a new instance operation with the new module and updated
  // operands.
  // Get the new hw module after rewriting to be used for the new instance
  hw::HWModuleOp newMod = newHWmodules[moduleName];
  assert(newMod && "could not find new hw module for instance");

  // Get the old hw module to identify the ready signals to change
  hw::HWModuleOp oldMod = oldHWmodules[moduleName];
  assert(oldMod && "could not find old hw module for instance");

  // Get the top-level module of the old instance
  hw::HWModuleOp oldInstTopModule = oldInst->getParentOfType<hw::HWModuleOp>();
  StringRef oldInstTopModuleName = oldInstTopModule.getName();
  // Find the corresponding new module in the new hw modules map where to
  // insert the new operations
  hw::HWModuleOp newTopMod = newHWmodules[oldInstTopModuleName];
  assert(newTopMod && "could not find new top module for instance");
  // Create the new operations within the new hw module
  builder.setInsertionPoint(newTopMod.getBodyBlock()->getTerminator());
  Location locNewOps = newTopMod.getBodyBlock()->getTerminator()->getLoc();

  // Save the list of outputs that are not ready signals to replace uses later
  SmallVector<std::pair<StringRef, unsigned>> nonReadyOutputs;
  // Save the list of the old ready inputs which will be mapped to the new
  // ready output values
  SmallVector<std::pair<StringRef, Value>> oldReadyInputs;
  // Save the new operands for the new instance
  SmallVector<Value> newOperands;
  // Iterate through the name of the ports to identify the mapping between
  // signals of the old hw module and new hw module
  for (auto &port : oldMod.getPortList()) {
    bool isReady = port.name.getValue().contains("ready");
    if (port.isInput() && isReady) {
      // This is an input of the old module and should become an output of the
      // new module. Save the old input value to map it later to the
      // corresponding new output value
      oldReadyInputs.push_back(std::make_pair(port.name.getValue(),
                                              oldInst.getOperand(port.argNum)));
    } else if (port.isOutput() && isReady) {
      // This is an output of the old module and should become an input of the
      // new module. Check if there is a mapping for this ready signal
      Value oldReadyOutput = oldInst->getResult(port.argNum);
      SmallVector<Value> newReadyInputs =
          getInputSignalMapping(oldReadyOutput, builder, locNewOps);
      assert(newReadyInputs.size() == 1 && "ready signal should be 1 bit wide");
      Value newReadyInput = newReadyInputs[0];
      newOperands.push_back(newReadyInput);
    } else if (port.isInput()) {
      // Check if there is a mapping for this non-ready signal
      Value oldNonReadyInput = oldInst.getOperand(port.argNum);
      SmallVector<Value> newNonReadyInputs =
          getInputSignalMapping(oldNonReadyInput, builder, locNewOps);
      // Append each bit separately of the non-ready signal to the new
      // operands
      for (auto newNonReadyInput : newNonReadyInputs)
        newOperands.push_back(newNonReadyInput);
    } else {
      // Collect non-ready outputs to replace uses later
      nonReadyOutputs.push_back(
          std::make_pair(port.name.getValue(), port.argNum));
    }
  }

  // Add new inputs for clk and rst from the top function
  newOperands.push_back(oldModuleSignalToNewModuleSignalsMap[clkSignalTop][0]);
  newOperands.push_back(oldModuleSignalToNewModuleSignalsMap[rstSignalTop][0]);

  // Create the new instance operation within the new hw module
  auto newInst = builder.create<hw::InstanceOp>(
      locNewOps, newMod, oldInst.getInstanceNameAttr(), newOperands);

  // Step 3: Update the mapping between old signals and new signals after
  // getting the new result values
  for (auto [outputName, outputIdxOldInst] : nonReadyOutputs) {
    // Find the output port in the old module
    Value oldResult = oldInst.getResult(outputIdxOldInst);
    updateOutputSignalMapping(oldResult, outputName, outputIdxOldInst, oldMod,
                              newMod, newInst);
  }
  // Update the mapping between old ready inputs and new ready outputs after
  // getting the new result values
  for (auto [inputName, oldReadyInput] : oldReadyInputs) {
    updateOutputSignalMapping(oldReadyInput, inputName, -1, oldMod, newMod,
                              newInst);
  }
}

/// Rewrite a HW module to normalize signal directions (e.g., ready going
/// opposite to data/valid) and unbundle multi-bit signals into single-bit
/// signals according to the chosen convention.
void SignalRewriter::rewriteHWModule(
    hw::HWModuleOp oldMod, ModuleOp parent, SymbolTable &symTable,
    DenseMap<StringRef, hw::HWModuleOp> &newHWmodules,
    DenseMap<StringRef, hw::HWModuleOp> &oldHWmodules) {

  // This function executes the following steps:
  // 1. Check if the module has already been rewritten. If so, return.
  // 2. Create a new hw module with inverted ready signal directions.
  // 3. Iterate through the body operations of the old module:
  //    a. If the operation is an hw instance, rewrite it to fix ready signal
  //       directions.
  //    b. If the operation is not an hw instance, check if it is only
  //       output operations and synth subckt operations. If so, create a
  //       synth subckt operation to connect the module ports. If not, raise
  //       an error.
  // 4. Finally, connect the hw instances to the terminator operands of the
  // new
  //    module.

  // Step 1: Check if the module has already been rewritten
  if (newHWmodules.count(oldMod.getName())) {
    return;
  }

  // Step 2: Create a new hw module with inverted ready signal directions
  MLIRContext *ctx = parent.getContext();
  OpBuilder builder(parent);

  SmallVector<hw::PortInfo> newInputs;
  SmallVector<hw::PortInfo> newOutputs;

  // Collect the new inputs and outputs with inverted ready signal directions
  unsigned inputIdx = 0;
  unsigned outputIdx = 0;
  // Store mapping from new output idx start and end to old signal
  SmallVector<std::pair<std::pair<unsigned, unsigned>, Value>>
      newModuleOutputIdxToOldSignal;
  // Store mapping from new input idx start and end to old signal
  SmallVector<std::pair<std::pair<unsigned, unsigned>, Value>>
      newModuleInputIdxToOldSignal;
  // Store the mapping between old output signal idx and new output signal idxs.
  // This should be applied only to non-ready signals.
  SmallVector<std::pair<unsigned, SmallVector<unsigned>>>
      oldOutputIdxToNewOutputIdxs;
  // Iterate over the ports of the old hw module
  for (auto &p : oldMod.getPortList()) {
    bool isReady = p.name.getValue().contains("ready");

    unsigned startOutputIdx = outputIdx;
    unsigned startInputIdx = inputIdx;
    if ((p.isInput() && isReady) || (p.isOutput() && !isReady)) {
      // If the port is input and ready, it becomes output and ready in the
      // new module. If a port is output and not ready, it stays as such.
      Type signalType = p.type;
      assert(signalType.isa<IntegerType>() &&
             "only integer types are supported for signals for now");
      unsigned bitWidth = signalType.cast<IntegerType>().getWidth();
      for (unsigned i = 0; i < bitWidth; ++i) {
        // Update portname indexing the specific bit unless it is 1 bit wide
        std::string portName =
            formatPortName(p.name.getValue().str(), i, bitWidth);
        if (bitWidth == 1) {
          portName = p.name.getValue().str();
        }
        newOutputs.push_back({hw::ModulePort{StringAttr::get(ctx, portName),
                                             builder.getIntegerType(1),
                                             hw::ModulePort::Direction::Output},
                              outputIdx});
        outputIdx++;
      }
      Value oldSignal;
      if (isReady) {
        // Record mapping from new ready output to old ready input
        oldSignal = oldMod.getBodyBlock()->getArgument(p.argNum);
      } else {
        // Record mapping from new non-ready output to old non-ready output
        oldSignal =
            oldMod.getBodyBlock()->getTerminator()->getOperand(p.argNum);
      }
      unsigned endOutputIdx = outputIdx - 1;
      newModuleOutputIdxToOldSignal.push_back(std::make_pair(
          std::make_pair(startOutputIdx, endOutputIdx), oldSignal));
      if (!isReady) {
        // Store mapping from old output idx to new output idxs for
        // non-ready outputs
        SmallVector<unsigned> newOutputIdxs;
        for (unsigned i = startOutputIdx; i <= endOutputIdx; ++i) {
          newOutputIdxs.push_back(i);
        }
        oldOutputIdxToNewOutputIdxs.push_back(
            std::make_pair(p.argNum, newOutputIdxs));
      }
    } else if ((p.isOutput() && isReady) || (p.isInput() && !isReady)) {
      // If the port is output and ready, it becomes input and ready in the
      // new module. If a port is input and not ready, it stays as such.
      Type signalType = p.type;
      assert(signalType.isa<IntegerType>() &&
             "only integer types are supported for signals for now");
      unsigned bitWidth = signalType.cast<IntegerType>().getWidth();
      for (unsigned i = 0; i < bitWidth; ++i) {
        // Update portname indexing the specific bit unless it is 1 bit wide
        std::string portName =
            formatPortName(p.name.getValue().str(), i, bitWidth);
        if (bitWidth == 1) {
          portName = p.name.getValue().str();
        }
        newInputs.push_back(
            {hw::ModulePort{StringAttr::get(ctx, StringRef{portName}),
                            builder.getIntegerType(1),
                            hw::ModulePort::Direction::Input},
             inputIdx});
        inputIdx++;
      }
      Value oldSignal;
      if (isReady) {
        // Record mapping from new ready input to old ready output
        oldSignal =
            oldMod.getBodyBlock()->getTerminator()->getOperand(p.argNum);
      } else {
        // Record mapping from new non-ready input to old non-ready input
        oldSignal = oldMod.getBodyBlock()->getArgument(p.argNum);
      }
      unsigned endInputIdx = inputIdx - 1;
      newModuleInputIdxToOldSignal.push_back(std::make_pair(
          std::make_pair(startInputIdx, endInputIdx), oldSignal));
    } else {
      assert(false && "port is neither input nor output");
    }
  }

  // Record the old output idx to new output idxs mapping for non-ready outputs
  oldOutputIdxToNewOutputIdxMap[builder.getStringAttr(oldMod.getName())] =
      oldOutputIdxToNewOutputIdxs;

  // Add clk and rst for all new inputs of hw modules
  newInputs.push_back({hw::ModulePort{mlir::StringAttr::get(ctx, clockSignal),
                                      builder.getIntegerType(1),
                                      hw::ModulePort::Direction::Input},
                       inputIdx});
  unsigned clkInputIdx = inputIdx;
  inputIdx++;
  newInputs.push_back({hw::ModulePort{mlir::StringAttr::get(ctx, resetSignal),
                                      builder.getIntegerType(1),
                                      hw::ModulePort::Direction::Input},
                       inputIdx});
  unsigned rstInputIdx = inputIdx;
  inputIdx++;

  hw::ModulePortInfo newPortInfo(newInputs, newOutputs);

  // Create new hw module
  builder.setInsertionPointAfter(oldMod);
  mlir::StringAttr newModuleName = getRewrittenModuleName(oldMod, ctx);
  auto newMod = builder.create<hw::HWModuleOp>(oldMod.getLoc(), newModuleName,
                                               newPortInfo);
  // Save it in the list of new module using the same name as the old module
  // as key
  newHWmodules[oldMod.getName()] = newMod;

  // Copy the blif path attribute
  if (oldMod->hasAttr(blifPathAttrStr)) {
    newMod->setAttr(blifPathAttrStr, oldMod->getAttr(blifPathAttrStr));
  }

  // Add mapping between new inputs to old signals
  for (auto [newModuleInputIdxs, oldSignal] : newModuleInputIdxToOldSignal) {
    unsigned newModuleInputIdxStart = newModuleInputIdxs.first;
    unsigned newModuleInputIdxEnd = newModuleInputIdxs.second;
    // For each bit of the input signal, get the corresponding argument
    SmallVector<Value> newReadyInput;
    for (unsigned idx = newModuleInputIdxStart; idx <= newModuleInputIdxEnd;
         ++idx) {
      newReadyInput.push_back(newMod.getBodyBlock()->getArgument(idx));
    }
    assert(newReadyInput.size() && "could not find mapping for input signal");
    oldModuleSignalToNewModuleSignalsMap[oldSignal] = newReadyInput;
  }

  // If the hw module represent the top function
  if (oldMod.getName() == getTopFunctionName()) {
    // Save the clock and reset signals
    assert((clkSignalTop == nullptr && rstSignalTop == nullptr) &&
           "reset and clock signals should be set only once for the top "
           "function");
    clkSignalTop = newMod.getBodyBlock()->getArgument(clkInputIdx);
    rstSignalTop = newMod.getBodyBlock()->getArgument(rstInputIdx);
    oldModuleSignalToNewModuleSignalsMap[clkSignalTop] = {clkSignalTop};
    oldModuleSignalToNewModuleSignalsMap[rstSignalTop] = {rstSignalTop};
  }

  // Step 3: Iterate through the body operations of the old module
  bool hasHwInstances = true;
  // Iterate through old body operations and invert ready signals in instances
  for (auto &op : oldMod.getBody().getOps()) {
    // Check if the operation is an hw instance
    if (auto instOp = dyn_cast<hw::InstanceOp>(op)) {
      // Step 3a: If the operation is an hw instance, rewrite it to fix ready
      // signal directions.
      rewriteHWInstance(instOp, parent, symTable, newHWmodules, oldHWmodules);
    } else if (!isa<hw::OutputOp>(op)) {
      hasHwInstances = false;
    }
  }

  if (!hasHwInstances) {
    // Step 3b: If the operation is not an hw instance, check if it is only
    // output operations and synth subckt operations. If so, create a
    // synth subckt operation to connect the module ports. If not, raise an
    // error.
    for (auto &op : oldMod.getBody().getOps()) {
      if (!isa<hw::OutputOp>(op) && !isa<synth::SubcktOp>(op)) {
        llvm::errs() << "Found non-instance operation in hw module: " << op
                     << "\n";
        assert(false && "found non-instance operation in hw module");
      }
    }
    ConversionPatternRewriter rewriter(ctx);
    if (failed(instantiateSynthOpInHWModule(newMod, rewriter))) {
      assert(false && "synth instantiation in hw module failed");
    }
    return;
  }

  // Step 4: Finally, connect the hw instances to the terminator operands of
  // the new module.
  builder.setInsertionPointToEnd(newMod.getBodyBlock());
  SmallVector<Value> newTerminatorOperands;
  for (auto [newOutputIdxs, oldSignal] : newModuleOutputIdxToOldSignal) {
    SmallVector<Value> newModuleOutputs =
        oldModuleSignalToNewModuleSignalsMap[oldSignal];
    assert(newModuleOutputs.size() &&
           "could not find mapping for output signal");
    newTerminatorOperands.append(newModuleOutputs.begin(),
                                 newModuleOutputs.end());
  }

  // Add operands to existing terminator
  Operation *newTerminator = newMod.getBodyBlock()->getTerminator();
  newTerminator->setOperands(newTerminatorOperands);
}

/// Function to apply signal-direction and bit-level rewrites to all HW
/// modules in the MLIR module.
LogicalResult SignalRewriter::rewriteAllSignals(mlir::ModuleOp modOp) {

  // The following function iterates through all hw modules in the module
  // and rewrites them to invert the direction of ready signals to follow
  // the standard handshake protocol where ready signals go in the opposite
  // direction with respect to data and valid signals. It applies recursively
  // to all hw modules instantiated within other hw modules.
  // It does so by creating new hw modules with the rewritten ready signal
  // directions and then connecting them following the same graph structure of
  // the old modules. Finally, it removes the old hw modules and renames the
  // new hw modules to the original names. Additionally, during the ready
  // signal inversion, it also unbundles multi-bit signals into single-bit
  // signals.

  // Maps to keep track of old and new hw modules
  // The old hw modules have the wrong ready signal directions where ready
  // signals follows the same direction as data and valid signals
  // The new hw modules will contain the rewritten modules with correct ready
  // signal directions
  DenseMap<StringRef, hw::HWModuleOp> oldHWModules;
  DenseMap<StringRef, hw::HWModuleOp> newHWModules;
  // Get symbol table for hw modules
  SymbolTable symTable(modOp);
  // Collect all hw modules in the module
  modOp.walk([&](hw::HWModuleOp m) { oldHWModules.insert({m.getName(), m}); });
  // Iterate through all hw modules and rewrite them
  for (auto [name, hwMod] : oldHWModules)
    rewriteHWModule(hwMod, modOp, symTable, newHWModules, oldHWModules);
  // Erase old hw modules
  for (auto [modName, oldMod] : oldHWModules) {
    oldMod.erase();
  }
  // Iterate through all new hw modules and rename them to the original names
  for (auto [originalName, newMod] : newHWModules) {
    mlir::StringAttr originalNameAttr =
        mlir::StringAttr::get(modOp.getContext(), originalName);
    StringRef currentModName = newMod.getName();
    // Change the module name of the instances of this module
    modOp.walk([&](hw::InstanceOp instOp) {
      if (instOp.getModuleName() == currentModName) {
        instOp.setModuleName(originalNameAttr);
      }
    });
    // Rename the new module to the original name
    newMod.setName(originalName);
  }
  return success();
}

// ------------------------------------------------------------------
// Functions to mark handshake operations with blif path
// ------------------------------------------------------------------

// Function to combine parameter values, module type and blif directory path
// to create the blif file path
std::string combineBlifFilePath(std::string blifDirPath, std::string moduleType,
                                std::vector<std::string> paramValues,
                                std::string extraSuffix = "") {
  std::string blifFilePath = blifDirPath + "/" + moduleType + extraSuffix;
  for (const auto &paramValue : paramValues) {
    blifFilePath += "/" + paramValue;
  }
  blifFilePath += "/" + moduleType + extraSuffix + ".blif";
  return blifFilePath;
}

// Function to retrieve the path of the blif file for a given handshake
// operation
std::string getBlifFilePathForHandshakeOp(Operation *op,
                                          std::string blifDirPath) {
  // Depending on the operation type, there is a different number and type of
  // parameters. For now, we use a switch hard-coded on the operation type. In
  // the future, this should be improved by allowing extraction of parameter
  // information from the operation itself
  std::string moduleType = op->getName().getStringRef().str();
  // Erase the dialect name from the moduleType
  moduleType = moduleType.substr(moduleType.find('.') + 1);
  std::string blifFileName;
  llvm::TypeSwitch<Operation *>(op)
      .Case<handshake::AddIOp, handshake::AndIOp, handshake::CmpIOp,
            handshake::OrIOp, handshake::ShLIOp, handshake::ShRSIOp,
            handshake::ShRUIOp, handshake::SubIOp, handshake::XOrIOp,
            handshake::MulIOp, handshake::DivSIOp, handshake::DivUIOp,
            handshake::SelectOp, handshake::SIToFPOp, handshake::FPToSIOp,
            handshake::ExtFOp, handshake::TruncFOp>([&](auto) {
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        blifFileName = combineBlifFilePath(blifDirPath, moduleType,
                                           {std::to_string(dataWidth)});
      })
      .Case<handshake::ConstantOp>([&](handshake::ConstantOp constOp) {
        handshake::ChannelType cstType = constOp.getResult().getType();
        // Get the data width of the constant operation
        unsigned dataWidth = cstType.getDataBitWidth();
        blifFileName = combineBlifFilePath(blifDirPath, moduleType,
                                           {std::to_string(dataWidth)});
      })
      .Case<handshake::BranchOp, handshake::SinkOp>([&](auto) {
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        if (dataWidth == 0) {
          blifFileName =
              combineBlifFilePath(blifDirPath, moduleType, {}, "_dataless");
        } else {
          blifFileName = combineBlifFilePath(blifDirPath, moduleType,
                                             {std::to_string(dataWidth)});
        }
      })
      .Case<handshake::BufferOp>([&](handshake::BufferOp op) {
        std::string bufferType =
            handshake::stringifyEnum(op.getBufferType()).str();
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        if (dataWidth == 0) {
          blifFileName =
              combineBlifFilePath(blifDirPath, bufferType, {}, "_dataless");
        } else {
          blifFileName = combineBlifFilePath(blifDirPath, bufferType,
                                             {std::to_string(dataWidth)});
        }
      })
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp cbrOp) {
            unsigned dataWidth = handshake::getHandshakeTypeBitWidth(
                cbrOp.getDataOperand().getType());
            if (dataWidth == 0) {
              blifFileName =
                  combineBlifFilePath(blifDirPath, moduleType, {}, "_dataless");
            } else {
              blifFileName = combineBlifFilePath(blifDirPath, moduleType,
                                                 {std::to_string(dataWidth)});
            }
          })
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp cmergeOp) {
        unsigned size = cmergeOp.getDataOperands().size();
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(cmergeOp.getResult().getType());
        unsigned indexType =
            handshake::getHandshakeTypeBitWidth(cmergeOp.getIndex().getType());
        if (dataWidth == 0) {
          blifFileName = combineBlifFilePath(
              blifDirPath, moduleType,
              {std::to_string(size), std::to_string(indexType)}, "_dataless");
        } else {
          assert(false && "ControlMerge with data not supported yet");
        }
      })
      .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::ExtFOp,
            handshake::TruncIOp, handshake::TruncFOp>([&](auto op) {
        unsigned inputWidth =
            handshake::getHandshakeTypeBitWidth(op.getOperand().getType());
        unsigned outputWidth =
            handshake::getHandshakeTypeBitWidth(op.getResult().getType());
        blifFileName = combineBlifFilePath(
            blifDirPath, moduleType,
            {std::to_string(inputWidth), std::to_string(outputWidth)});
      })
      .Case<handshake::ForkOp, handshake::LazyForkOp>([&](auto) {
        unsigned size = op->getNumResults();
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        if (dataWidth == 0) {
          blifFileName = combineBlifFilePath(
              blifDirPath, moduleType, {std::to_string(size)}, "_dataless");
        } else {
          blifFileName = combineBlifFilePath(
              blifDirPath, moduleType,
              {std::to_string(size), std::to_string(dataWidth)}, "_type");
        }
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        unsigned size = muxOp.getDataOperands().size();
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(muxOp.getResult().getType());
        unsigned selectType = handshake::getHandshakeTypeBitWidth(
            muxOp.getSelectOperand().getType());
        blifFileName = combineBlifFilePath(blifDirPath, moduleType,
                                           {std::to_string(size),
                                            std::to_string(dataWidth),
                                            std::to_string(selectType)});
      })
      .Case<handshake::MergeOp>([&](handshake::MergeOp mergeOp) {
        unsigned size = mergeOp.getDataOperands().size();
        unsigned dataWidth = handshake::getHandshakeTypeBitWidth(
            mergeOp.getDataOperands()[0].getType());
        blifFileName = combineBlifFilePath(
            blifDirPath, moduleType,
            {std::to_string(size), std::to_string(dataWidth)});
      })
      .Case<handshake::LoadOp, handshake::StoreOp>([&](auto op) {
        unsigned dataWidth =
            handshake::getHandshakeTypeBitWidth(op.getDataInput().getType());
        unsigned addrType =
            handshake::getHandshakeTypeBitWidth(op.getAddressInput().getType());
        blifFileName = combineBlifFilePath(
            blifDirPath, moduleType,
            {std::to_string(dataWidth), std::to_string(addrType)});
      })
      .Case<handshake::SourceOp>([&](auto) {
        blifFileName = combineBlifFilePath(blifDirPath, moduleType, {});
      })
      .Default([&](auto) { blifFileName = ""; });
  // Check blifFileName path exists
  if (blifFileName != "") {
    // Check if the file exists
    if (!std::filesystem::exists(blifFileName)) {
      llvm::errs() << "BLIF file for operation " << getUniqueName(op)
                   << "does not exist: " << blifFileName << "\n";
      assert(false && "Check the file path specified for this operation is "
                      "built correctly");
    }
  }
  return blifFileName;
}

// Function to add blif path attribute to each handshake operation
LogicalResult markHandshakeOpsWithBlifPath(handshake::FuncOp funcOp,
                                           std::string blifDirPath,
                                           OpBuilder &builder) {
  // Check blifDirPath is not empty
  if (blifDirPath.empty()) {
    llvm::errs() << "BLIF directory path is empty\n";
    return failure();
  }
  // Walk through all handshake operations in the function
  funcOp.walk([&](Operation *op) {
    // Skip function operations
    if (isa<handshake::FuncOp>(op))
      return;
    // Create the blif path by combining the directory path and the op name
    // depending on the operation type
    std::string blifFilePath = getBlifFilePathForHandshakeOp(op, blifDirPath);
    // Create a string attribute for the blif path
    mlir::StringAttr blifPathAttr =
        mlir::StringAttr::get(builder.getContext(), blifFilePath);
    // Set the blif path attribute on the operation
    op->setAttr(blifPathAttrStr, blifPathAttr);
  });
  return success();
}

// ------------------------------------------------------------------
// Functions to convert hw instances to synth operations
// ------------------------------------------------------------------

// Function to get the input mapping of a signal value
Value getInputMappingSynthSignal(Location loc,
                                 DenseMap<StringAttr, Value> &nodeValuesMap,
                                 DenseMap<StringAttr, Value> &tmpValuesMap,
                                 StringRef nodeName, OpBuilder &builder) {
  assert(builder.getInsertionBlock() && "Builder has no insertion block!");
  // Check if the signal is already in the map
  if (nodeValuesMap.count(builder.getStringAttr(nodeName))) {
    return nodeValuesMap[builder.getStringAttr(nodeName)];
  }
  // Check if the signal is in the tmp values map
  if (tmpValuesMap.count(builder.getStringAttr(nodeName))) {
    return tmpValuesMap[builder.getStringAttr(nodeName)];
  }
  // If not, create a temporary input signal (e.g., hw constant)
  auto tempInput = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerType(1),
      builder.getIntegerAttr(builder.getIntegerType(1), 0));

  // Add the temporary input signal to the map
  tmpValuesMap[builder.getStringAttr(nodeName)] = tempInput;
  return tempInput;
}

// Function to update the output signal mapping after creating a new synth
// operation
void updateOutputSynthSignalMapping(Value newResult, StringRef nodeName,
                                    DenseMap<StringAttr, Value> &nodeValuesMap,
                                    DenseMap<StringAttr, Value> &tmpValuesMap,
                                    OpBuilder &builder) {
  // Check if the old result is in the tmp values map
  if (tmpValuesMap.count(builder.getStringAttr(nodeName))) {
    // Replace all uses of the temporary value with the new result value
    tmpValuesMap[builder.getStringAttr(nodeName)].replaceAllUsesWith(newResult);
    auto *constOp =
        tmpValuesMap[builder.getStringAttr(nodeName)].getDefiningOp();
    assert(constOp && "temporary value should have a defining operation");
    assert(constOp->use_empty() && "temporary value should have no more uses");
    // Erase the operation
    constOp->erase();
    // Remove the temporary value from the map
    tmpValuesMap.erase(builder.getStringAttr(nodeName));
  }
  // Add the new result value to the node values map
  nodeValuesMap[builder.getStringAttr(nodeName)] = newResult;
}

// Function to create a wire in function of synth logic gate operations based
// on .names definition
void createSynthWire(Location loc, std::vector<std::string> &ports,
                     std::string &function,
                     DenseMap<StringAttr, Value> &nodeValuesMap,
                     DenseMap<StringAttr, Value> &tmpValuesMap,
                     OpBuilder &builder) {
  assert(builder.getInsertionBlock() && "Builder has no insertion block!");
  unsigned numInputs = ports.size() - 1;
  assert(numInputs == 1 && "wire should have only one input");
  SmallVector<Value> inputValues;
  // Get input values
  for (unsigned i = 0; i < numInputs; ++i) {
    StringRef inputNodeName = ports[i];
    Value inputValue = getInputMappingSynthSignal(
        loc, nodeValuesMap, tmpValuesMap, inputNodeName, builder);
    inputValues.push_back(inputValue);
  }
  StringRef outputNodeName = ports.back();
  // Elaborate function
  assert(function.size() == 3 &&
         "function string must be of size 3 for a wire");
  char inputValueFunc = function[0];
  char outputValueFunc = function[2];
  assert((inputValueFunc == '0' || inputValueFunc == '1') &&
         "input value must be 0 or 1");
  assert((outputValueFunc == '0' || outputValueFunc == '1') &&
         "output value must be 0 or 1");
  int inputBit = (inputValueFunc == '1') ? 1 : 0;
  int outputBit = (outputValueFunc == '1') ? 1 : 0;
  Value inputValue = inputValues[0];
  // If the input and output bits are identical, you can skip the creation
  // of any node since they are identical
  if (inputBit == outputBit) {
    // We just need to update the output map function so that the output
    // points to the same value as input
    updateOutputSynthSignalMapping(inputValue, outputNodeName, nodeValuesMap,
                                   tmpValuesMap, builder);
    return;
  }
  // If this is not the case, we have to create a new aig node which inverts
  // the value of the input and receives one as another input Create constant
  // node 1
  auto constOp = builder.create<hw::ConstantOp>(
      loc, builder.getIntegerType(1),
      builder.getIntegerAttr(builder.getIntegerType(1), 1));

  // Create aig node
  auto aigOp = builder.create<synth::aig::AndInverterOp>(
      loc, inputValue, constOp.getResult(),
      /*invertInput0=*/true, /*invertInput1=*/false);

  updateOutputSynthSignalMapping(aigOp.getResult(), outputNodeName,
                                 nodeValuesMap, tmpValuesMap, builder);
}

// Function to create synth logic gate operations based on .names definition
void createSynthLogicGate(Location loc, std::vector<std::string> &ports,
                          std::string &function,
                          DenseMap<StringAttr, Value> &nodeValuesMap,
                          DenseMap<StringAttr, Value> &tmpValuesMap,
                          OpBuilder &builder) {
  assert(builder.getInsertionBlock() && "Builder has no insertion block!");
  unsigned numInputs = ports.size() - 1;
  assert(numInputs > 1 && "logic gate must have at least two inputs");
  assert(numInputs == 2 && "the following pass transforms the logic gates to "
                           "aig nodes only");
  SmallVector<Value> inputValues;
  // Get input values
  for (unsigned i = 0; i < numInputs; ++i) {
    StringRef inputNodeName = ports[i];
    Value inputValue = getInputMappingSynthSignal(
        loc, nodeValuesMap, tmpValuesMap, inputNodeName, builder);
    inputValues.push_back(inputValue);
  }
  StringRef outputNodeName = ports.back();

  // Elaborate function to understand inversion of inputs/outputs
  // Get the value of the first and second input and output
  assert(function.size() == 4 &&
         "function string must be of size 4 for 2-input logic gate");
  char firstInputValue = function[0];
  char secondInputValue = function[1];
  char outputValue = function[3];
  assert((firstInputValue == '0' || firstInputValue == '1') &&
         "first input value must be 0 or 1");
  assert((secondInputValue == '0' || secondInputValue == '1') &&
         "second input value must be 0 or 1");
  assert((outputValue == '0' || outputValue == '1') &&
         "output value must be 0 or 1");
  int firstInputBit = (firstInputValue == '1') ? 1 : 0;
  int secondInputBit = (secondInputValue == '1') ? 1 : 0;
  int outputBit = (outputValue == '1') ? 1 : 0;
  // Invert inputs/outputs if output is 0
  if (outputBit == 0) {
    firstInputBit = 1 - firstInputBit;
    secondInputBit = 1 - secondInputBit;
  }
  bool invertInput0 = (firstInputBit == 0) ? true : false;
  bool invertInput1 = (secondInputBit == 0) ? true : false;

  // Create aig node
  auto aigOp = builder.create<synth::aig::AndInverterOp>(
      loc, inputValues[0], inputValues[1],
      /*invertInput0=*/invertInput0, /*invertInput1=*/invertInput1);

  // Update output mapping
  updateOutputSynthSignalMapping(aigOp.getResult(), outputNodeName,
                                 nodeValuesMap, tmpValuesMap, builder);
}

// Function to read a blif file and generate the synth circuit depending on the
// blif description
LogicalResult generateSynthCircuitFromBlif(StringRef blifFilePath, Location loc,
                                           hw::InstanceOp hwInst,
                                           hw::HWModuleOp hwModule,
                                           OpBuilder &builder) {
  builder.setInsertionPoint(hwInst);
  // The following function reads the blif file specified by blifFilePath
  // and generates the synth circuit defined in it.

  // Map to store the values of the nodes
  DenseMap<StringAttr, Value> nodeValuesMap;
  // Map to store temporary values for nodes not yet defined
  DenseMap<StringAttr, Value> tmpValuesMap;
  // Collect hw instance inputs and the corresponding naming in the hw module
  DenseMap<StringAttr, Value> hwInstInputs;

  // Check that the instance is fully connected
  assert(hwInst.getOperands().size() == hwModule.getNumInputPorts() &&
         "Cannot use positional operands on partially-connected instance");

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
      StringAttr portNameAttr = builder.getStringAttr(port.name.getValue());
      hwInstInputs[portNameAttr] = hwInstInput;
      nodeValuesMap[portNameAttr] = hwInstInput;
    }
  }

  // Collect hw instance outputs
  SmallVector<StringRef> hwInstOutputs;
  for (auto &port : hwModule.getPortList()) {
    if (port.isOutput()) {
      hwInstOutputs.push_back(port.name.getValue());
    }
  }

  std::ifstream file(blifFilePath.str());

  if (!file.is_open()) {
    llvm::errs() << "The blif file '" << blifFilePath
                 << "' has not been found or could not be opened." << "\n";
    return failure();
  }

  std::string line;
  // Loop over all the lines in .blif file.
  while (std::getline(file, line)) {
    // If comment or empty line, skip.
    if (line.empty() || line.find("#") == 0) {
      continue;
    }

    // If line ends with '\\', read the next line and append to the current
    // line.
    while (line.back() == '\\') {
      line.pop_back();
      std::string nextLine;
      std::getline(file, nextLine);
      line += nextLine;
    }

    std::istringstream iss(line);
    std::string type;
    iss >> type;

    // Model name
    if (type == ".model") {
      std::string moduleName;
      iss >> moduleName;
    }

    // Input/Output nodes. These are also Dataflow graph channels.
    else if ((type == ".inputs") || (type == ".outputs")) {
      std::string nodeName;
      while (iss >> nodeName) {
        if (type == ".inputs") {
          // Check if the input node is in the hw instance inputs map
          if (!hwInstInputs.count(builder.getStringAttr(nodeName))) {
            llvm::errs() << "Input node '" << nodeName
                         << "' not found in hw instance inputs." << "\n";
            return failure();
          }
        } else {
          // Check if the output node is in the hw instance outputs
          if (std::find(hwInstOutputs.begin(), hwInstOutputs.end(), nodeName) ==
              hwInstOutputs.end()) {
            llvm::errs() << "Output node '" << nodeName
                         << "' not found in hw instance outputs." << "\n";
            return failure();
          }
        }
      }
    }

    // Latches.
    else if (type == ".latch") {
      std::string regInput, regOutput;
      iss >> regInput >> regOutput;
      // Get the input signal for the register
      Value inputSignal = getInputMappingSynthSignal(
          loc, nodeValuesMap, tmpValuesMap, regInput, builder);
      // Create synth register operation
      auto regOp = builder.create<synth::LatchOp>(
          loc, builder.getIntegerType(1), inputSignal);

      // Map the output node to the register output
      updateOutputSynthSignalMapping(regOp.getResult(), regOutput,
                                     nodeValuesMap, tmpValuesMap, builder);

    }

    // .names stand for logic gates.
    else if (type == ".names") {
      std::vector<std::string> nodeNames;
      std::string currentNode;

      // Read node names from current line (e.g., "a b c")
      while (iss >> currentNode) {
        nodeNames.push_back(currentNode);
      }

      // Read logic function from next line (e.g., "11 1")
      std::getline(file, line);
      std::string function = line;

      if (nodeNames.size() == 1) {
        // Create hw constant
        auto constOp = builder.create<hw::ConstantOp>(
            loc, builder.getIntegerType(1),
            builder.getIntegerAttr(builder.getIntegerType(1),
                                   function == "1" ? 1 : 0));

        // Map the output node to the constant output
        updateOutputSynthSignalMapping(constOp.getResult(), nodeNames[0],
                                       nodeValuesMap, tmpValuesMap, builder);
      } else if (nodeNames.size() == 2) {
        // Create wire as a function of other existing synth operations
        createSynthWire(loc, nodeNames, function, nodeValuesMap, tmpValuesMap,
                        builder);
      } else {
        // Create logic gate as a function of other existing synth operations
        createSynthLogicGate(loc, nodeNames, function, nodeValuesMap,
                             tmpValuesMap, builder);
      }
    }

    // Subcircuits. not used for now.
    else if (line.find(".subckt") == 0) {
      llvm::errs() << "Subcircuits inside the blif file not supported " << "\n";
      continue;
    }

    // Ends the file.
    else if (line.find(".end") == 0) {
      break;
    }
  }

  SmallVector<Value> newOutputs;
  // Map hw instance outputs to the corresponding synth signals
  for (const auto &outputNodeName : hwInstOutputs) {
    if (!nodeValuesMap.count(builder.getStringAttr(outputNodeName))) {
      llvm::errs() << "Output node '" << outputNodeName
                   << "' not found in synth circuit." << "\n";
      return failure();
    }
    Value outputSignal = nodeValuesMap[builder.getStringAttr(outputNodeName)];
    newOutputs.push_back(outputSignal);
  }
  // Close the file
  file.close();
  // Replace hw instance outputs with the corresponding synth signals
  for (unsigned i = 0; i < hwInst.getNumResults(); ++i) {
    hwInst.getResult(i).replaceAllUsesWith(newOutputs[i]);
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
  if (!hwModule->hasAttr(blifPathAttrStr)) {
    llvm::errs() << "hw module for instance does not have blif path attribute: "
                 << hwModule << "\n";
    return failure();
  }
  // Get the blif path attribute
  mlir::StringAttr blifPathAttr =
      hwModule->getAttrOfType<mlir::StringAttr>(blifPathAttrStr);
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
// 0) Mark each handshake operation with the path of the blif file where its
//    definition is located
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

    // Step 0: Mark each handshake operation with the path of the blif file
    // where its definition is located
    if (failed(markHandshakeOpsWithBlifPath(funcOp, blifDirPath, builder)))
      return signalPassFailure();

    // Step 1: unbundle all handshake types in the handshake function
    if (failed(unbundleAllHandshakeTypes(modOp, ctx)))
      return signalPassFailure();

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

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeToSynthPass() {
  return std::make_unique<HandshakeToSynthPass>();
}
