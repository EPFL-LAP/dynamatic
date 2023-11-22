//===- HandshakeToNetlist.cpp - Converts handshake to HW/ESI ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the handshake to netlist pass.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/HandshakeToNetlist.h"
#include "circt/Conversion/HandshakeToHW.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Conversion/PassDetails.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace circt::hw;
using namespace dynamatic;

//===----------------------------------------------------------------------===//
// Internal data-structures
//===----------------------------------------------------------------------===//

namespace {

/// Function that returns a unique name for each distinct operation it is passed
/// as input.
using NameUniquer = std::function<std::string(Operation *)>;

/// Shared state used during lowering. Captured in a struct to reduce the number
/// of arguments we have to pass around.
struct HandshakeLoweringState {
  /// Module containing the handshake-level function to lower.
  ModuleOp parentModule;
  /// Producer of unique names for external modules.
  NameUniquer nameUniquer;
  /// Builder for (external) modules.
  OpBuilder extModBuilder;

  /// Creates the lowering state, producing an OpBuilder from the parent
  /// module's context and setting its insertion point in the module's body.
  HandshakeLoweringState(ModuleOp mod, NameUniquer nameUniquer)
      : parentModule(mod), nameUniquer(nameUniquer),
        extModBuilder(mod->getContext()) {
    extModBuilder.setInsertionPointToStart(mod.getBody());
  }
};

/// Holds information about a function's ports. This is used primarily to keep
/// track of ports that connect an internal memory interface with an external
/// memory component.
struct FuncModulePortInfo {
  /// Number of memory input ports that are created for each input memref to the
  /// handshake-level function.
  static const size_t NUM_MEM_INPUTS = 1;
  /// Number of memory output ports that are created for each input memref to
  /// the handshake-level function.
  static const size_t NUM_MEM_OUTPUTS = 5;

  /// Function's ports.
  ModulePortInfo ports;

  /// Index mapping between input ports and output ports that correspond to
  /// memory IOs. Each entry in the vector represent a different memory
  /// interface, with NUM_MEM_INPUTS input ports starting at the index contained
  /// in the pair's first value and with NUM_MEM_OUTPUTS output ports starting
  /// at the index contained in the pair's second value.
  std::vector<std::pair<size_t, size_t>> memIO;

  /// Initializes the struct with empty ports.
  FuncModulePortInfo() : ports({}, {}){};

  /// Adds the IO (input and output ports) corresponding to a memory interface
  /// to the function. The name argument is used to prefix the name of all ports
  /// associated with the memory interface; it must therefore be unique across
  /// calls to the method.
  void addMemIO(MemRefType memref, std::string name, MLIRContext *ctx);

  /// Computes the number of output ports that are associated with internal
  /// memory interfaces.
  size_t getNumMemOutputs() { return NUM_MEM_OUTPUTS * memIO.size(); }
};

} // namespace

void FuncModulePortInfo::addMemIO(MemRefType memref, std::string name,
                                  MLIRContext *ctx) {
  // Types used by memory IO
  Type i1Type = IntegerType::get(ctx, 1);
  Type addrType = IntegerType::get(ctx, 32); // todo: hardcoded to 32 for now
  Type dataType = memref.getElementType();

  // Remember ports which correspond to memory
  auto inputIdx = ports.inputs.size();
  auto outputIdx = ports.outputs.size();
  memIO.push_back(std::make_pair(inputIdx, outputIdx));

  // Load data input
  ports.inputs.push_back({StringAttr::get(ctx, name + "_load_data"),
                          hw::PortDirection::INPUT, dataType, inputIdx,
                          hw::InnerSymAttr{}});
  // Load enable output
  ports.outputs.push_back({StringAttr::get(ctx, name + "_load_en"),
                           hw::PortDirection::OUTPUT, i1Type, outputIdx++,
                           hw::InnerSymAttr{}});
  // Load address output
  ports.outputs.push_back({StringAttr::get(ctx, name + "_load_addr"),
                           hw::PortDirection::OUTPUT, addrType, outputIdx++,
                           hw::InnerSymAttr{}});
  // Store enable output
  ports.outputs.push_back({StringAttr::get(ctx, name + "_store_en"),
                           hw::PortDirection::OUTPUT, i1Type, outputIdx++,
                           hw::InnerSymAttr{}});
  // Store address output
  ports.outputs.push_back({StringAttr::get(ctx, name + "_store_addr"),
                           hw::PortDirection::OUTPUT, addrType, outputIdx++,
                           hw::InnerSymAttr{}});
  // Store data output
  ports.outputs.push_back({StringAttr::get(ctx, name + "_store_data"),
                           hw::PortDirection::OUTPUT, dataType, outputIdx,
                           hw::InnerSymAttr{}});
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Returns the generic external module name corresponding to an operation,
/// without discriminating type information.
static std::string getBareExtModuleName(Operation *oldOp) {
  std::string extModuleName = oldOp->getName().getStringRef().str();
  std::replace(extModuleName.begin(), extModuleName.end(), '.', '_');
  return extModuleName;
}

/// Extracts the data-carrying type of a value. If the value is an ESI
/// channel, extracts the data-carrying type, else assumes that the value's type
/// itself is the data-carrying type.
static Type getOperandDataType(Value val) {
  auto valType = val.getType();
  if (auto channelType = valType.dyn_cast<esi::ChannelType>())
    return channelType.getInner();
  return valType;
}

/// Returns a set of types which may uniquely identify the provided operation.
/// The first element of the returned pair represents the discriminating input
/// types, while the second element represents the discriminating output types.
using DiscriminatingTypes = std::pair<SmallVector<Type>, SmallVector<Type>>;
static DiscriminatingTypes getDiscriminatingParameters(Operation *op) {
  SmallVector<Type> inTypes, outTypes;
  llvm::transform(op->getOperands(), std::back_inserter(inTypes),
                  getOperandDataType);
  llvm::transform(op->getResults(), std::back_inserter(outTypes),
                  getOperandDataType);
  return DiscriminatingTypes{inTypes, outTypes};
}

/// Returns the string representation of a type. Emits an error at the given
/// location if the type isn't supported.
static std::string getTypeName(Type type, Location loc) {
  // Integer-like types
  if (type.isIntOrIndex()) {
    if (auto indexType = type.dyn_cast<IndexType>())
      return std::to_string(indexType.kInternalStorageBitWidth);
    if (type.isSignedInteger())
      return std::to_string(type.getIntOrFloatBitWidth());
    return std::to_string(type.getIntOrFloatBitWidth());
  }

  // Float type
  if (isa<FloatType>(type))
    return std::to_string(type.getIntOrFloatBitWidth());

  // Tuple type
  if (auto tupleType = type.dyn_cast<TupleType>()) {
    std::string tupleName = "_tuple";
    for (auto elementType : tupleType.getTypes())
      tupleName += getTypeName(elementType, loc);
    return tupleName;
  }

  // NoneType (dataless channel)
  if (isa<NoneType>(type))
    return "0";

  emitError(loc) << "data type \"" << type << "\" not supported";
  return "";
}

/// Constructs an external module name corresponding to an operation. The
/// returned name is unique with respect to the operation's discriminating
/// parameters.
static std::string getExtModuleName(Operation *oldOp) {
  std::string extModName = getBareExtModuleName(oldOp);
  extModName += "_node.";
  auto types = getDiscriminatingParameters(oldOp);
  mlir::Location loc = oldOp->getLoc();
  SmallVector<Type> &inTypes = types.first;
  SmallVector<Type> &outTypes = types.second;

  llvm::TypeSwitch<Operation *>(oldOp)
      .Case<handshake::BufferOp>([&](handshake::BufferOp bufOp) {
        // buffer type
        extModName +=
            bufOp.getBufferType() == BufferTypeEnum::seq ? "seq" : "fifo";
        // bitwidth
        extModName += "_" + getTypeName(outTypes[0], loc);
      })
      .Case<handshake::ForkOp, handshake::LazyForkOp>([&](auto) {
        // number of outputs
        extModName += std::to_string(outTypes.size());
        // bitwidth
        extModName += "_" + getTypeName(outTypes[0], loc);
      })
      .Case<handshake::MuxOp>([&](auto) {
        // number of inputs (without select param)
        extModName += std::to_string(inTypes.size() - 1);
        // bitwidth
        extModName += "_" + getTypeName(inTypes[1], loc);
        // select bitwidth
        extModName += "_" + getTypeName(inTypes[0], loc);
      })
      .Case<handshake::ControlMergeOp>([&](auto) {
        // number of inputs
        extModName += std::to_string(inTypes.size());
        // bitwidth
        extModName += "_" + getTypeName(inTypes[0], loc);
        // index result bitwidth
        extModName += "_" + getTypeName(outTypes[outTypes.size() - 1], loc);
      })
      .Case<handshake::MergeOp>([&](auto) {
        // number of inputs
        extModName += std::to_string(inTypes.size());
        // bitwidth
        extModName += "_" + getTypeName(inTypes[0], loc);
      })
      .Case<handshake::ConditionalBranchOp>([&](auto) {
        // bitwidth
        extModName += getTypeName(inTypes[1], loc);
      })
      .Case<handshake::BranchOp, handshake::SinkOp, handshake::SourceOp>(
          [&](auto) {
            // bitwidth
            if (!inTypes.empty())
              extModName += getTypeName(inTypes[0], loc);
            else
              extModName += getTypeName(outTypes[0], loc);
          })
      .Case<handshake::LoadOpInterface, handshake::StoreOpInterface>([&](auto) {
        // data bitwidth
        extModName += getTypeName(inTypes[0], loc);
        // address bitwidth
        extModName += "_" + getTypeName(inTypes[1], loc);
      })
      .Case<handshake::ConstantOp>([&](auto) {
        // constant value
        if (auto constOp = dyn_cast<handshake::ConstantOp>(oldOp)) {
          if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
            APInt val = intAttr.getValue();
            extModName += std::to_string(val.getZExtValue());
          } else if (auto floatAttr = constOp.getValue().dyn_cast<FloatAttr>())
            extModName += std::to_string(floatAttr.getValue().convertToFloat());
          else
            llvm_unreachable("unsupported constant type");
        }
        // bitwidth
        extModName += "_" + getTypeName(outTypes[0], loc);
      })
      .Case<handshake::JoinOp, handshake::SyncOp>([&](auto) {
        // array of bitwidths
        for (auto inType : inTypes)
          extModName += getTypeName(inType, loc) + "_";
        extModName = extModName.substr(0, extModName.size() - 1);
      })
      .Case<handshake::EndOp>([&](auto) {
        // mem_inputs
        extModName += std::to_string(inTypes.size() - 1);
        // bitwidth
        extModName += "_" + getTypeName(inTypes[0], loc);
      })
      .Case<handshake::DynamaticReturnOp>([&](auto) {
        // bitwidth
        extModName += getTypeName(inTypes[0], loc);
      })
      .Case<handshake::MemoryControllerOp>(
          [&](handshake::MemoryControllerOp op) {
            FuncMemoryPorts ports = op.getPorts();
            // Data bitwidth
            extModName += std::to_string(ports.dataWidth);
            // Address bitwidth
            extModName += '_' + std::to_string(ports.addrWidth);
            // Port counts
            extModName += '_' + std::to_string(ports.getNumPorts<LoadPort>()) +
                          '_' + std::to_string(ports.getNumPorts<StorePort>()) +
                          '_' +
                          std::to_string(ports.getNumPorts<ControlPort>());
          })
      .Case<arith::AddFOp, arith::AddIOp, arith::AndIOp, arith::BitcastOp,
            arith::CeilDivSIOp, arith::CeilDivUIOp, arith::DivFOp,
            arith::DivSIOp, arith::DivUIOp, arith::FloorDivSIOp, arith::MaxFOp,
            arith::MaxSIOp, arith::MaxUIOp, arith::MinFOp, arith::MinSIOp,
            arith::MinUIOp, arith::MulFOp, arith::MulIOp, arith::NegFOp,
            arith::OrIOp, arith::RemFOp, arith::RemSIOp, arith::RemUIOp,
            arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp, arith::SubFOp,
            arith::SubIOp, arith::XOrIOp>([&](auto) {
        // bitwidth
        extModName += getTypeName(inTypes[0], loc);
      })
      .Case<arith::SelectOp>([&](auto) {
        // bitwidth
        extModName += getTypeName(outTypes[0], loc);
      })
      .Case<arith::CmpFOp, arith::CmpIOp>([&](auto) {
        // predicate
        if (auto cmpOp = dyn_cast<mlir::arith::CmpIOp>(oldOp))
          extModName += stringifyEnum(cmpOp.getPredicate()).str();
        else if (auto cmpOp = dyn_cast<mlir::arith::CmpFOp>(oldOp))
          extModName += stringifyEnum(cmpOp.getPredicate()).str();
        // bitwidth
        extModName += "_" + getTypeName(inTypes[0], loc);
      })
      .Case<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::FPToSIOp,
            arith::FPToUIOp, arith::SIToFPOp, arith::TruncFOp, arith::TruncIOp,
            arith::UIToFPOp>([&](auto) {
        // input bitwidth
        extModName += getTypeName(inTypes[0], loc);
        // output bitwidth
        extModName += "_" + getTypeName(outTypes[0], loc);
      })
      .Default([&](auto) {
        oldOp->emitError() << "No matching component for operation";
      });

  return extModName;
}

/// Checks whether a module with the same name has been created elsewhere in the
/// top level module. Returns the matched module operation if true, otherwise
/// returns nullptr.
static hw::HWModuleLike findModule(mlir::ModuleOp parentModule,
                                   StringRef modName) {
  if (auto mod = parentModule.lookupSymbol<HWModuleOp>(modName))
    return mod;
  if (auto mod = parentModule.lookupSymbol<HWModuleExternOp>(modName))
    return mod;
  return nullptr;
}

/// Returns a module-like operation that matches the provided name. The
/// functions first attempts to find an existing module with the same name,
/// which it returns if it exists. Failing the latter, the function creates an
/// external module at the provided location and with the provided ports.
static hw::HWModuleLike getModule(HandshakeLoweringState &ls,
                                  std::string modName, ModulePortInfo &ports,
                                  Location modLoc) {
  hw::HWModuleLike mod = findModule(ls.parentModule, modName);
  if (!mod)
    return ls.extModBuilder.create<hw::HWModuleExternOp>(
        modLoc, ls.extModBuilder.getStringAttr(modName), ports);
  return mod;
}

/// Derives port information for an operation so that it can be converted to a
/// hardware module.
static ModulePortInfo getPortInfo(Operation *op) {
  return getPortInfoForOpTypes(op, op->getOperandTypes(), op->getResultTypes());
}

/// Name of port representing the clock signal.
static const std::string CLK_PORT = "clock";
/// Name of port representing the reset signal.
static const std::string RST_PORT = "reset";

/// Adds clock and reset ports to a module's inputs ports.
static void addClkAndRstPorts(ModulePortInfo &ports, MLIRContext *ctx) {
  Type i1Type = IntegerType::get(ctx, 1);
  auto idx = ports.inputs.size();
  ports.inputs.push_back({StringAttr::get(ctx, CLK_PORT),
                          hw::PortDirection::INPUT, i1Type, idx,
                          hw::InnerSymAttr{}});
  ports.inputs.push_back({StringAttr::get(ctx, RST_PORT),
                          hw::PortDirection::INPUT, i1Type, idx + 1,
                          hw::InnerSymAttr{}});
}

/// Adds the clock and reset arguments of a module to the list of operands of an
/// operation within the module.
static void addClkAndRstOperands(SmallVector<Value> &operands,
                                 hw::HWModuleOp mod) {
  auto numArguments = mod.getNumArguments();
  assert(numArguments >= 2 &&
         "module should have at least a clock and reset arguments");
  auto ports = mod.getPorts();
  assert(ports.inputs[numArguments - 2].getName() == CLK_PORT &&
         "second to last module port should be clock");
  assert(ports.inputs[numArguments - 1].getName() == RST_PORT &&
         "last module port should be clock");
  operands.push_back(mod.getArgument(numArguments - 2));
  operands.push_back(mod.getArgument(numArguments - 1));
}

/// Derives port information for a handshake function so that it can be
/// converted to a hardware module.
static FuncModulePortInfo getFuncPortInfo(handshake::FuncOp funcOp) {

  FuncModulePortInfo info;
  auto *ctx = funcOp.getContext();

  // Add all outputs of function
  TypeRange outputs = funcOp.getResultTypes();
  for (auto [idx, res] : llvm::enumerate(outputs))
    info.ports.outputs.push_back({funcOp.getResName(idx),
                                  hw::PortDirection::OUTPUT, esiWrapper(res),
                                  idx, hw::InnerSymAttr{}});

  // Add all inputs of function, replacing memrefs with appropriate ports
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments()))
    if (isa<MemRefType>(arg.getType()))
      info.addMemIO(dyn_cast<MemRefType>(arg.getType()),
                    funcOp.getArgName(idx).str(), ctx);
    else
      info.ports.inputs.push_back(
          {funcOp.getArgName(idx), hw::PortDirection::INPUT,
           esiWrapper(arg.getType()), idx, hw::InnerSymAttr{}});

  addClkAndRstPorts(info.ports, ctx);
  return info;
}

/// Derives port information for a handshake memory controller so that it can be
/// converted to a hardware module. The function needs information from the
/// enclosing function's ports as well as the index of the memory interface
/// within that data-structure to derive the future module's ports.
static ModulePortInfo getMemPortInfo(handshake::MemoryControllerOp memOp,
                                     FuncModulePortInfo &info, size_t memIdx) {

  ModulePortInfo ports({}, {});
  auto *ctx = memOp.getContext();

  auto &[inFuncPortIdx, outFuncPortIdx] = info.memIO[memIdx];

  // Add input ports coming from outside the containing module
  size_t inPortIdx = 0;
  for (size_t i = inFuncPortIdx,
              e = inFuncPortIdx + FuncModulePortInfo::NUM_MEM_INPUTS;
       i < e; i++) {
    auto funcInput = info.ports.inputs[i];
    ports.inputs.push_back({funcInput.name, hw::PortDirection::INPUT,
                            funcInput.type, inPortIdx++, hw::InnerSymAttr{}});
  }

  // Add input ports corresponding to memory interface operands
  for (auto [idx, arg] : llvm::enumerate(memOp.getMemInputs()))
    ports.inputs.push_back({StringAttr::get(ctx, memOp.getOperandName(idx + 1)),
                            hw::PortDirection::INPUT, esiWrapper(arg.getType()),
                            inPortIdx++, hw::InnerSymAttr{}});

  // Add output ports corresponding to memory interface operands
  size_t outPortIdx = 0;
  for (auto [idx, arg] : llvm::enumerate(memOp.getResults()))
    ports.outputs.push_back({StringAttr::get(ctx, memOp.getResultName(idx)),
                             hw::PortDirection::OUTPUT,
                             esiWrapper(arg.getType()), outPortIdx++,
                             hw::InnerSymAttr{}});

  // Add output ports going outside the containing module
  for (size_t i = outFuncPortIdx,
              e = outFuncPortIdx + FuncModulePortInfo::NUM_MEM_OUTPUTS;
       i < e; i++) {
    auto funcOutput = info.ports.outputs[i];
    ports.outputs.push_back({funcOutput.name, hw::PortDirection::OUTPUT,
                             funcOutput.type, outPortIdx++,
                             hw::InnerSymAttr{}});
  }

  addClkAndRstPorts(ports, ctx);
  return ports;
}

/// Derives port information for the end operation of a handshake function so
/// that it can be converted to a hardware module. The function needs
/// information from the enclosing function's ports to determine the number of
/// return values in the original function that the future module should output.
static ModulePortInfo getEndPortInfo(handshake::EndOp endOp,
                                     FuncModulePortInfo &info) {

  ModulePortInfo ports({}, {});
  auto *ctx = endOp.getContext();

  // Add input ports corresponding to end operands
  size_t inPortIdx = 0;
  for (auto [idx, arg] : llvm::enumerate(endOp.getOperands()))
    ports.inputs.push_back({StringAttr::get(ctx, "in" + std::to_string(idx)),
                            hw::PortDirection::INPUT, esiWrapper(arg.getType()),
                            inPortIdx++, hw::InnerSymAttr{}});

  // Add output ports corresponding to function return values
  auto numReturnValues = info.ports.outputs.size() - info.getNumMemOutputs();
  auto returnValOperands = endOp.getOperands().take_front(numReturnValues);
  for (auto [idx, arg] : llvm::enumerate(returnValOperands))
    ports.outputs.push_back({StringAttr::get(ctx, "out" + std::to_string(idx)),
                             hw::PortDirection::OUTPUT,
                             esiWrapper(arg.getType()), idx,
                             hw::InnerSymAttr{}});

  addClkAndRstPorts(ports, ctx);
  return ports;
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//
namespace {

/// A type converter is needed to perform the in-flight materialization of "raw"
/// (non-ESI channel) types to their ESI channel correspondents.
class ESITypeConverter : public TypeConverter {
public:
  ESITypeConverter() {
    addConversion([](Type type) -> Type { return esiWrapper(type); });

    addTargetMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;
          return inputs[0];
        });

    addSourceMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;
          return inputs[0];
        });
  }
};

/// Converts handshake functions to hardware modules. The pattern creates a
/// hw::HWModuleOp or hw::HWModuleExternOp with IO corresponding to the original
/// handshake function. In the case where the matched function is not external,
/// the pattern additionally (1) buffers the function's inputs, (2) converts
/// internal memory interfaces to hw::HWInstanceOp's and connects them to the
/// containing module IO, (3) converts the function's end operation to a
/// hw::HWInstanceOp that also outputs the function's return values, and (4)
/// combines all the module outputs in the hw::HwOutputOp operation at the end
/// of the module.
class FuncOpConversionPattern : public OpConversionPattern<handshake::FuncOp> {
public:
  FuncOpConversionPattern(MLIRContext *ctx, HandshakeLoweringState &ls)
      : OpConversionPattern<handshake::FuncOp>(ctx, 1), ls(ls) {}

  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    FuncModulePortInfo info = getFuncPortInfo(op);

    if (op.isExternal())
      ls.extModBuilder.create<hw::HWModuleExternOp>(
          op.getLoc(), ls.extModBuilder.getStringAttr(op.getName()),
          info.ports);
    else {
      // Create module for the function
      auto mod = ls.extModBuilder.create<hw::HWModuleOp>(
          op.getLoc(), ls.extModBuilder.getStringAttr(op.getName()),
          info.ports);

      // Replace uses of function arguments with module inputs
      for (auto it : llvm::zip(op.getArguments(), mod.getArguments()))
        std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

      // Move all operations to the new module
      auto &moduleBlockOps = mod.getBodyBlock()->getOperations();
      moduleBlockOps.splice(moduleBlockOps.begin(),
                            op.getBody().front().getOperations());

      // Insert a start buffer for each channel-typed input
      bufferInputs(mod, rewriter);

      // Convert memory interfaces (and connect them with function IO)
      auto memInstances = convertMemories(mod, info, rewriter);

      // Convert end operation (add results to represent function return values)
      auto endInstance = convertEnd(mod, info, rewriter);

      // Set operands of output operation to match module outputs
      setModuleOutputs(mod, memInstances, endInstance);
    }

    // Original function can safely be deleted before returning
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Lowering state to help in the creation of new hardware modules/instances.
  HandshakeLoweringState &ls;

  /// Inserts a "start module" that acts as a buffer for all module inputs that
  /// are of type esi::ChannelType. This is done to match legacy Dynamatic's
  /// implementation of circuits.
  void bufferInputs(HWModuleOp mod, ConversionPatternRewriter &rewriter) const;

  /// Converts memory interfaces within the module into hardware instances with
  /// added IO to handle interactions with external memory through the module
  /// IO. The function returns the list of newly created hardware instances.
  SmallVector<hw::InstanceOp>
  convertMemories(HWModuleOp mod, FuncModulePortInfo &info,
                  ConversionPatternRewriter &rewriter) const;

  /// Converts the end operation of a handshake function into a corresponding
  /// hardware instance with added outputs to hold the function return values.
  /// The function returns the created hardware instance.
  hw::InstanceOp convertEnd(HWModuleOp mod, FuncModulePortInfo &info,
                            ConversionPatternRewriter &rewriter) const;

  /// Modifies the operands of the hw::OutputOp operation within the newly
  /// created module to match the latter's outputs.
  void setModuleOutputs(HWModuleOp mod,
                        SmallVector<hw::InstanceOp> memInstances,
                        hw::InstanceOp endInstance) const;
};

/// Converts an operation (of type indicated by the template argument) into an
/// equivalent hardware instance. The method creates an external module to
/// instantiate the new component from if a module with matching IO one does not
/// already exist. Valid/Ready semantics are made explicit thanks to the type
/// converter which converts implicit handshaked types into ESI channels with a
/// corresponding data-type.
template <typename T>
class ExtModuleConversionPattern : public OpConversionPattern<T> {
public:
  ExtModuleConversionPattern(ESITypeConverter &typeConverter, MLIRContext *ctx,
                             HandshakeLoweringState &ls)
      : OpConversionPattern<T>::OpConversionPattern(typeConverter, ctx),
        ls(ls) {}
  using OpAdaptor = typename T::Adaptor;

  LogicalResult
  matchAndRewrite(T op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto modName = getExtModuleName(op);
    auto ports = getPortInfo(op);
    hw::HWModuleLike extModule = getModule(ls, modName, ports, op.getLoc());

    // Replace operation with corresponding hardware module instance
    SmallVector<Value> operands = adaptor.getOperands();
    if (op.template hasTrait<mlir::OpTrait::HasClock>())
      addClkAndRstOperands(operands, cast<hw::HWModuleOp>(op->getParentOp()));

    auto instanceOp = rewriter.replaceOpWithNewOp<hw::InstanceOp>(
        op, extModule, rewriter.getStringAttr(ls.nameUniquer(op)), operands);

    // Replace operation results with new instance results
    for (auto it : llvm::zip(op->getResults(), instanceOp->getResults()))
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

    return success();
  }

private:
  /// Lowering state to help in the creation of new hardware modules/instances.
  HandshakeLoweringState &ls;
};
} // namespace

void FuncOpConversionPattern::bufferInputs(
    HWModuleOp mod, ConversionPatternRewriter &rewriter) const {

  unsigned argIdx = 0;
  rewriter.setInsertionPointToStart(mod.getBodyBlock());

  for (auto arg : mod.getArguments()) {
    auto argType = arg.getType();
    if (!isa<esi::ChannelType>(argType))
      continue;

    // Create ports for input buffer
    ModulePortInfo ports({}, {});
    auto *ctx = mod.getContext();

    // Function argument input
    ports.inputs.push_back({StringAttr::get(ctx, "in0"),
                            hw::PortDirection::INPUT, argType, 0,
                            hw::InnerSymAttr{}});
    addClkAndRstPorts(ports, ctx);

    // Module output matches function argument type
    ports.outputs.push_back({StringAttr::get(ctx, "out0"),
                             hw::PortDirection::OUTPUT, argType, 0,
                             hw::InnerSymAttr{}});

    // Check whether we need to create a new external module
    auto argLoc = arg.getLoc();
    std::string modName = "handshake_start_node";
    if (auto channelType = argType.dyn_cast<esi::ChannelType>())
      modName += "." + getTypeName(channelType.getInner(), argLoc);

    hw::HWModuleLike extModule = getModule(ls, modName, ports, argLoc);

    // Replace operation with corresponding hardware module instance
    llvm::SmallVector<Value> operands;
    operands.push_back(arg);
    addClkAndRstOperands(operands, mod);
    auto instanceOp = rewriter.create<hw::InstanceOp>(
        argLoc, extModule,
        rewriter.getStringAttr("handshake_start_node" +
                               std::to_string(argIdx++)),
        operands);

    rewriter.replaceAllUsesExcept(arg, instanceOp.getResult(0), instanceOp);
  }
}

SmallVector<hw::InstanceOp> FuncOpConversionPattern::convertMemories(
    HWModuleOp mod, FuncModulePortInfo &info,
    ConversionPatternRewriter &rewriter) const {

  SmallVector<hw::InstanceOp> instances;

  for (auto [memIdx, portIndices] : llvm::enumerate(info.memIO)) {
    // Identify the memory controller refering to the memref
    auto user = *mod.getArgument(portIndices.first).getUsers().begin();
    assert(user && "old memref value should have a user");
    auto memOp = dyn_cast<handshake::MemoryControllerOp>(user);
    assert(memOp && "user of old memref value should be memory interface");

    std::string memName = getExtModuleName(memOp);
    auto ports = getMemPortInfo(memOp, info, memIdx);

    // Create an external module definition if necessary
    hw::HWModuleLike extModule = getModule(ls, memName, ports, memOp->getLoc());

    // Combine memory inputs from the function and internal memory inputs into
    // the new instance operands
    SmallVector<Value> operands;
    for (auto i = portIndices.first,
              e = portIndices.first + FuncModulePortInfo::NUM_MEM_INPUTS;
         i < e; i++)
      operands.push_back(mod.getArgument(i));
    operands.insert(operands.end(), memOp.getMemInputs().begin(),
                    memOp.getMemInputs().end());
    addClkAndRstOperands(operands, mod);

    // Create instance of memory interface
    rewriter.setInsertionPoint(memOp);
    auto instance = rewriter.create<hw::InstanceOp>(
        memOp.getLoc(), extModule,
        rewriter.getStringAttr(ls.nameUniquer(memOp)), operands);

    // Replace uses of memory interface results with new instance results
    for (auto it :
         llvm::zip(memOp->getResults(),
                   instance->getResults().take_front(memOp->getNumResults())))
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

    rewriter.eraseOp(memOp);
    instances.push_back(instance);
  }

  return instances;
}

hw::InstanceOp
FuncOpConversionPattern::convertEnd(HWModuleOp mod, FuncModulePortInfo &info,
                                    ConversionPatternRewriter &rewriter) const {
  // End operation is guaranteed to exist and be unique
  auto endOp = *mod.getBodyBlock()->getOps<handshake::EndOp>().begin();

  // Create external module
  auto extModule = ls.extModBuilder.create<hw::HWModuleExternOp>(
      endOp->getLoc(), ls.extModBuilder.getStringAttr(getExtModuleName(endOp)),
      getEndPortInfo(endOp, info));

  // Create instance of end operation
  SmallVector<Value> operands(endOp.getOperands());
  addClkAndRstOperands(operands, mod);
  rewriter.setInsertionPoint(endOp);
  auto instance = rewriter.create<hw::InstanceOp>(
      endOp.getLoc(), extModule, rewriter.getStringAttr(ls.nameUniquer(endOp)),
      operands);

  rewriter.eraseOp(endOp);
  return instance;
}

void FuncOpConversionPattern::setModuleOutputs(
    HWModuleOp mod, SmallVector<hw::InstanceOp> memInstances,
    hw::InstanceOp endInstance) const {
  // Output operation is guaranteed to exist and be unique
  auto outputOp = *mod.getBodyBlock()->getOps<hw::OutputOp>().begin();

  // Derive new operands
  SmallVector<Value> newOperands;
  newOperands.insert(newOperands.end(), endInstance.getResults().begin(),
                     endInstance.getResults().end());
  for (auto &mem : memInstances) {
    auto memOutputs =
        mem.getResults().take_back(FuncModulePortInfo::NUM_MEM_OUTPUTS);
    newOperands.insert(newOperands.end(), memOutputs.begin(), memOutputs.end());
  }

  // Switch operands
  outputOp->setOperands(newOperands);
}

/// Verifies that all the operations inside the function, which may be more
/// general than what we can turn into an RTL design, will be successfully
/// exportable to an RTL design. Fails if at least one operation inside the
/// function is not exportable to RTL.
static LogicalResult verifyExportToRTL(handshake::FuncOp funcOp) {
  for (Operation &op : funcOp.getOps()) {
    LogicalResult res =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<handshake::ConstantOp>(
                [&](handshake::ConstantOp cstOp) -> LogicalResult {
                  if (!cstOp.getValue().isa<IntegerAttr, FloatAttr>())
                    return cstOp->emitError()
                           << "Incompatible attribute type, our VHDL component "
                              "only supports integer and floating-point types";
                  return success();
                })
            .Case<handshake::ReturnOp>(
                [&](handshake::ReturnOp retOp) -> LogicalResult {
                  if (retOp->getNumOperands() != 1)
                    return retOp.emitError()
                           << "Incompatible number of return values, our VHDL "
                              "component only supports a single return value";
                  return success();
                })
            .Case<handshake::EndOp>(
                [&](handshake::EndOp endOp) -> LogicalResult {
                  if (endOp.getReturnValues().size() != 1)
                    return endOp.emitError()
                           << "Incompatible number of return values, our VHDL "
                              "component only supports a single return value";
                  return success();
                })
            .Default([](auto) { return success(); });
    if (failed(res))
      return failure();
  }
  return success();
}

namespace {

/// Unpack buffers with more that one slot into an equivalent sequence of
/// multiple one-slot buffers. Our RTL buffer components cannot be parameterized
/// with a number of slots (they are all 1-slot) so we have to do this unpacking
/// prior to running the conversion pass.
struct UnpackBufferSlots : public OpRewritePattern<handshake::BufferOp> {
  using OpRewritePattern<handshake::BufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::BufferOp bufOp,
                                PatternRewriter &rewriter) const override {
    // Only operate on buffers with strictly more than one slots
    unsigned numSlots = bufOp.getSlots();
    if (numSlots == 1)
      return failure();

    rewriter.setInsertionPoint(bufOp);

    // Create a sequence of one-slot buffers
    BufferTypeEnum bufType = bufOp.getBufferType();
    Value bufVal = bufOp.getOperand();
    for (size_t idx = 0; idx < numSlots; ++idx)
      bufVal =
          rewriter
              .create<handshake::BufferOp>(bufOp.getLoc(), bufVal, 1, bufType)
              .getResult();

    // Replace the original multi-slots buffer with the output of the last
    // buffer in the sequence
    rewriter.replaceOp(bufOp, bufVal);
    return success();
  }
};

/// Handshake to netlist conversion pass. The conversion only works on modules
/// containing a single handshake function (handshake::FuncOp) at the moment.
/// The function and all the operations it contains are converted to operations
/// from the HW dialect. Dataflow semantics are made explicit with ESI channels.
class HandshakeToNetListPass
    : public HandshakeToNetlistBase<HandshakeToNetListPass> {
public:
  void runDynamaticPass() override {
    mlir::ModuleOp mod = getOperation();
    MLIRContext &ctx = getContext();

    // We only support one function per module
    auto functions = mod.getOps<handshake::FuncOp>();
    if (++functions.begin() != functions.end()) {
      mod->emitOpError()
          << "we currently only support one handshake function per module";
      return signalPassFailure();
    }
    handshake::FuncOp funcOp = *functions.begin();

    // Check that some preconditions are met before doing anything
    if (failed(verifyAllValuesHasOneUse(funcOp))) {
      funcOp.emitOpError()
          << "Lowering to netlist requires that all values in the IR are used "
             "exactly once. Run the --handshake-materialize-forks-sinks pass "
             "before to insert forks and sinks in the IR and make every "
             "value used exactly once.";
      return signalPassFailure();
    }
    if (failed(verifyAllIndexConcretized(funcOp))) {
      funcOp.emitOpError()
          << "Lowering to netlist requires that all index types in the IR have "
             "been concretized."
          << ERR_RUN_CONCRETIZATION;
      return signalPassFailure();
    }
    if (failed(verifyExportToRTL(funcOp)))
      return signalPassFailure();

    // Run a pre-processing pass on the IR
    if (failed(preprocessMod())) {
      mod->emitError()
          << "Failed to pre-process IR to make it valid for netlist conversion";
      return signalPassFailure();
    }

    // Create helper struct for lowering
    std::map<std::string, unsigned> instanceNameCntr;
    NameUniquer instanceUniquer = [&](Operation *op) {
      std::string instName = getBareExtModuleName(op);
      return instName + std::to_string(instanceNameCntr[instName]++);
    };
    HandshakeLoweringState ls(mod, instanceUniquer);

    // Create pattern set
    ESITypeConverter typeConverter;
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    patterns.insert<FuncOpConversionPattern>(&ctx, ls);
    patterns.insert<
        // Handshake operations
        ExtModuleConversionPattern<handshake::BufferOp>,
        ExtModuleConversionPattern<handshake::ConditionalBranchOp>,
        ExtModuleConversionPattern<handshake::BranchOp>,
        ExtModuleConversionPattern<handshake::MergeOp>,
        ExtModuleConversionPattern<handshake::ControlMergeOp>,
        ExtModuleConversionPattern<handshake::MuxOp>,
        ExtModuleConversionPattern<handshake::SourceOp>,
        ExtModuleConversionPattern<handshake::ConstantOp>,
        ExtModuleConversionPattern<handshake::SinkOp>,
        ExtModuleConversionPattern<handshake::ForkOp>,
        ExtModuleConversionPattern<handshake::DynamaticReturnOp>,
        ExtModuleConversionPattern<handshake::MCLoadOp>,
        ExtModuleConversionPattern<handshake::LSQLoadOp>,
        ExtModuleConversionPattern<handshake::MCStoreOp>,
        ExtModuleConversionPattern<handshake::LSQStoreOp>,
        // Arith operations
        ExtModuleConversionPattern<arith::AddFOp>,
        ExtModuleConversionPattern<arith::AddIOp>,
        ExtModuleConversionPattern<arith::AndIOp>,
        ExtModuleConversionPattern<arith::BitcastOp>,
        ExtModuleConversionPattern<arith::CeilDivSIOp>,
        ExtModuleConversionPattern<arith::CeilDivUIOp>,
        ExtModuleConversionPattern<arith::CmpFOp>,
        ExtModuleConversionPattern<arith::CmpIOp>,
        ExtModuleConversionPattern<arith::DivFOp>,
        ExtModuleConversionPattern<arith::DivSIOp>,
        ExtModuleConversionPattern<arith::DivUIOp>,
        ExtModuleConversionPattern<arith::ExtFOp>,
        ExtModuleConversionPattern<arith::ExtSIOp>,
        ExtModuleConversionPattern<arith::ExtUIOp>,
        ExtModuleConversionPattern<arith::FPToSIOp>,
        ExtModuleConversionPattern<arith::FPToUIOp>,
        ExtModuleConversionPattern<arith::FloorDivSIOp>,
        ExtModuleConversionPattern<arith::IndexCastOp>,
        ExtModuleConversionPattern<arith::IndexCastUIOp>,
        ExtModuleConversionPattern<arith::MulFOp>,
        ExtModuleConversionPattern<arith::MulIOp>,
        ExtModuleConversionPattern<arith::NegFOp>,
        ExtModuleConversionPattern<arith::OrIOp>,
        ExtModuleConversionPattern<arith::RemFOp>,
        ExtModuleConversionPattern<arith::RemSIOp>,
        ExtModuleConversionPattern<arith::RemUIOp>,
        ExtModuleConversionPattern<arith::SelectOp>,
        ExtModuleConversionPattern<arith::SIToFPOp>,
        ExtModuleConversionPattern<arith::ShLIOp>,
        ExtModuleConversionPattern<arith::ShRSIOp>,
        ExtModuleConversionPattern<arith::ShRUIOp>,
        ExtModuleConversionPattern<arith::SubFOp>,
        ExtModuleConversionPattern<arith::SubIOp>,
        ExtModuleConversionPattern<arith::TruncFOp>,
        ExtModuleConversionPattern<arith::TruncIOp>,
        ExtModuleConversionPattern<arith::UIToFPOp>,
        ExtModuleConversionPattern<arith::XOrIOp>>(typeConverter,
                                                   funcOp->getContext(), ls);

    // Everything must be converted to operations in the hw dialect
    target.addLegalOp<hw::HWModuleOp, hw::HWModuleExternOp, hw::OutputOp,
                      hw::InstanceOp>();
    target.addIllegalDialect<handshake::HandshakeDialect, arith::ArithDialect,
                             memref::MemRefDialect>();

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
      return signalPassFailure();
  }

private:
  /// Perfoms some simple transformations on the module to make the netlist that
  /// will result from the conversion able to be turned into an RTL design.
  /// NOTE: (RamirezLucas) Ideally, this should be moved to a separate
  /// pre-processing pass.
  LogicalResult preprocessMod();
};

LogicalResult HandshakeToNetListPass::preprocessMod() {
  mlir::ModuleOp modOp = getOperation();
  MLIRContext *ctx = &getContext();
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  RewritePatternSet preprocessPatterns(ctx);
  preprocessPatterns.add<UnpackBufferSlots>(ctx);
  return applyPatternsAndFoldGreedily(modOp, std::move(preprocessPatterns),
                                      config);
}

} // end anonymous namespace

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeToNetlistPass() {
  return std::make_unique<HandshakeToNetListPass>();
}
