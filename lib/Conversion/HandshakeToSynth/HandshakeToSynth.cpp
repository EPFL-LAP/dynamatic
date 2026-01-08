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
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <bitset>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

#define DEBUG_TYPE "handshake-to-synth"

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
void unbundleOpPorts(Operation *op,
                     SmallVector<std::pair<SignalKind, Type>> &inputPorts,
                     SmallVector<std::pair<SignalKind, Type>> &outputPorts) {
  for (auto input : op->getOperands()) {
    // Unbundle the input type depending on its actual type
    SmallVector<std::pair<SignalKind, Type>> inputUnbundled =
        unbundleType(input.getType());
    inputPorts.append(inputUnbundled.begin(), inputUnbundled.end());
  }
  for (auto result : op->getResults()) {
    // Unbundle the output type depending on its actual type
    SmallVector<std::pair<SignalKind, Type>> outputUnbundled =
        unbundleType(result.getType());
    outputPorts.append(outputUnbundled.begin(), outputUnbundled.end());
  }
  // If the operation is a handshake function, the ports are extracted
  // differently
  if (isa<handshake::FuncOp>(op)) {
    handshake::FuncOp funcOp = cast<handshake::FuncOp>(op);
    // Clear existing ports
    inputPorts.clear();
    outputPorts.clear();
    // Extract ports from the function type
    for (auto arg : funcOp.getArguments()) {
      SmallVector<std::pair<SignalKind, Type>> inputUnbundled =
          unbundleType(arg.getType());
      inputPorts.append(inputUnbundled.begin(), inputUnbundled.end());
    }
    for (auto resultType : funcOp.getResultTypes()) {
      SmallVector<std::pair<SignalKind, Type>> outputUnbundled =
          unbundleType(resultType);
      outputPorts.append(outputUnbundled.begin(), outputUnbundled.end());
    }
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

// Function to get the HW Module input and output port info from an handshake
// operation
void getHWModulePortInfo(Operation *op, SmallVector<hw::PortInfo> &hwInputPorts,
                         SmallVector<hw::PortInfo> &hwOutputPorts) {
  MLIRContext *ctx = op->getContext();
  // Unbundle the operation ports
  SmallVector<std::pair<SignalKind, Type>> unbundledInputPorts;
  SmallVector<std::pair<SignalKind, Type>> unbundledOutputPorts;
  unbundleOpPorts(op, unbundledInputPorts, unbundledOutputPorts);
  // Fill in the hw port info for inputs
  for (auto [idx, port] : llvm::enumerate(unbundledInputPorts)) {
    std::string name;
    Type type;
    // Unpack the port info as name and type
    name = port.first == DATA_SIGNAL    ? "in_data_" + std::to_string(idx)
           : port.first == VALID_SIGNAL ? "in_valid_" + std::to_string(idx)
                                        : "in_ready_" + std::to_string(idx);
    type = port.second;
    // Create the hw port info and add it to the list
    hwInputPorts.push_back(
        hw::PortInfo{hw::ModulePort{StringAttr::get(ctx, name), type,
                                    hw::ModulePort::Direction::Input},
                     idx});
  }
  // Fill in the hw port info for outputs
  for (auto [idx, port] : llvm::enumerate(unbundledOutputPorts)) {
    std::string name;
    Type type;
    // Unpack the port info as name and type
    name = port.first == DATA_SIGNAL    ? "out_data_" + std::to_string(idx)
           : port.first == VALID_SIGNAL ? "out_valid_" + std::to_string(idx)
                                        : "out_ready_" + std::to_string(idx);
    type = port.second;
    // Create the hw port info and add it to the list
    hwOutputPorts.push_back(
        hw::PortInfo{hw::ModulePort{StringAttr::get(ctx, name), type,
                                    hw::ModulePort::Direction::Output},
                     idx});
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

// Function to create a cast operation for each result of an unbundled operation
// when converting an original bundled operation to an unbundled one. This cast
// is essential to connect the unbundled operation back to the rest of the
// circuit which is potentially still bundled
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
    // Get a slice of the unbundled values corresponding to this result from the
    // unbundled operation
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
// It is divided into three parts: first, create the hw module definition if it
// does not already exist, then move the function body of the handshake function
// into the hw module, and finally create the output operation for the hw module
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
    // module. The output of the casts will be used to inline the function body
    // First create casts for each argument of the hw module
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
    // Inline the function body before the terminator of the hw module using the
    // casted arguments
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

    // Create a cast for each operand of the end operation to unbundle the types
    // which were previously handshake bundled types
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
    // Remove any old output operation if present (should not be the case after
    // replacing the end op)
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

// Function to instantiate synth or hw operations inside an hw module describing
// the behavior of the original handshake operation
LogicalResult
instantiateSynthOpInHWModule(Operation *op, hw::HWModuleOp hwModule,
                             SmallVector<hw::PortInfo> &hwInputPorts,
                             SmallVector<hw::PortInfo> &hwOutputPorts,
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
  for (auto &outputPort : hwOutputPorts) {
    synthOutputTypes.push_back(outputPort.type);
  }
  TypeRange synthOutputsTypeRange(synthOutputTypes);
  // Create the synth subcircuit operation inside the hw module
  synth::SubcktOp synthInstOp = rewriter.create<synth::SubcktOp>(
      op->getLoc(), synthOutputsTypeRange, synthInputs, "synth_subckt");
  // Connect the outputs of the synth operation to the outputs of the hw module
  Operation *synthTerminator = hwModule.getBodyBlock()->getTerminator();
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
    if (failed(instantiateSynthOpInHWModule(op, hwModule, hwInputPorts,
                                            hwOutputPorts, rewriter)))
      return nullptr;
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
      // into its components to connect it to the input ports of the hw instance
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
      // input cast but first assert that it is not followed by any other cast.
      // If this is the case, there is a issue in the code since there are 3
      // casts in chain and this break the assumption of the code
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

namespace {

// The following pass converts handshake operations into synth operations
// It executes in multiple steps:
// 1) Convert all handshake operations (except for function and end ops)
//    into hw module operations connecting them with unrealized conversion
//    casts
// 2) Convert the handshake function operation into an hw module operation
// 3) Remove all unrealized conversion casts by connecting directly the
//    inputs and outputs
class HandshakeToSynthPass
    : public dynamatic::impl::HandshakeToSynthBase<HandshakeToSynthPass> {
public:
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
    // operations, except for the function and end operations which are kept to
    // convert in the next step
    target.addLegalOp<handshake::FuncOp>();
    target.addLegalOp<handshake::EndOp>();
    patterns.insert<
        ConvertToHWMod<handshake::BufferOp>,
        ConvertToHWMod<handshake::NDWireOp>,
        ConvertToHWMod<handshake::ConditionalBranchOp>,
        ConvertToHWMod<handshake::BranchOp>, ConvertToHWMod<handshake::MergeOp>,
        ConvertToHWMod<handshake::ControlMergeOp>,
        ConvertToHWMod<handshake::MuxOp>, ConvertToHWMod<handshake::JoinOp>,
        ConvertToHWMod<handshake::BlockerOp>,
        ConvertToHWMod<handshake::SourceOp>,
        ConvertToHWMod<handshake::ConstantOp>,
        ConvertToHWMod<handshake::SinkOp>, ConvertToHWMod<handshake::ForkOp>,
        ConvertToHWMod<handshake::LazyForkOp>,
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
        ConvertToHWMod<handshake::SIToFPOp>,
        ConvertToHWMod<handshake::UIToFPOp>,
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
      return signalPassFailure();

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
      return signalPassFailure();

    // Step 3: remove all unrealized conversion casts
    // Execute without conversion function but walking the module
    if (failed(removeUnrealizedConversionCasts(modOp)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeToSynthPass() {
  return std::make_unique<HandshakeToSynthPass>();
}
