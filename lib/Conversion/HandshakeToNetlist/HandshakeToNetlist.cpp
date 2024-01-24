//===- HandshakeToNetlist.cpp - Converts handshake to HW --------*- C++ -*-===//
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
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt::hw;
using namespace dynamatic;

//===----------------------------------------------------------------------===//
// Taken verbatim from CIRCT
//===----------------------------------------------------------------------===//

namespace {
/// A class to be used with getPortInfoForOp. Provides an opaque interface for
/// generating the port names of an operation; handshake operations generate
/// names by the Handshake NamedIOInterface;  and other operations, such as
/// arith ops, are assigned default names.
class HandshakePortNameGenerator {
public:
  explicit HandshakePortNameGenerator(Operation *op)
      : builder(op->getContext()) {
    auto namedOpInterface = dyn_cast<handshake::NamedIOInterface>(op);
    if (namedOpInterface)
      inferFromNamedOpInterface(namedOpInterface);
    else if (auto funcOp = dyn_cast<handshake::FuncOp>(op))
      inferFromFuncOp(funcOp);
    else
      inferDefault(op);
  }

  StringAttr inputName(unsigned idx) { return inputs[idx]; }
  StringAttr outputName(unsigned idx) { return outputs[idx]; }

private:
  using IdxToStrF = const std::function<std::string(unsigned)> &;
  void infer(Operation *op, IdxToStrF &inF, IdxToStrF &outF) {
    llvm::transform(
        llvm::enumerate(op->getOperandTypes()), std::back_inserter(inputs),
        [&](auto it) { return builder.getStringAttr(inF(it.index())); });
    llvm::transform(
        llvm::enumerate(op->getResultTypes()), std::back_inserter(outputs),
        [&](auto it) { return builder.getStringAttr(outF(it.index())); });
  }

  void inferDefault(Operation *op) {
    infer(
        op, [](unsigned idx) { return "in" + std::to_string(idx); },
        [](unsigned idx) { return "out" + std::to_string(idx); });
  }

  void inferFromNamedOpInterface(handshake::NamedIOInterface op) {
    infer(
        op, [&](unsigned idx) { return op.getOperandName(idx); },
        [&](unsigned idx) { return op.getResultName(idx); });
  }

  void inferFromFuncOp(handshake::FuncOp op) {
    auto inF = [&](unsigned idx) { return op.getArgName(idx).str(); };
    auto outF = [&](unsigned idx) { return op.getResName(idx).str(); };
    llvm::transform(
        llvm::enumerate(op.getArgumentTypes()), std::back_inserter(inputs),
        [&](auto it) { return builder.getStringAttr(inF(it.index())); });
    llvm::transform(
        llvm::enumerate(op.getResultTypes()), std::back_inserter(outputs),
        [&](auto it) { return builder.getStringAttr(outF(it.index())); });
  }

  Builder builder;
  llvm::SmallVector<StringAttr> inputs;
  llvm::SmallVector<StringAttr> outputs;
};
} // namespace

/// Wraps a type into a handshake::ChannelType type.
static handshake::ChannelType channelWrapper(Type t) {
  return TypeSwitch<Type, handshake::ChannelType>(t)
      .Case<handshake::ChannelType>([](auto t) { return t; })
      .Case<NoneType>([](NoneType nt) {
        return handshake::ChannelType::get(
            IntegerType::get(nt.getContext(), 0));
      })
      .Default([](Type t) { return handshake::ChannelType::get(t); });
}

static circt::hw::ModulePortInfo
getPortInfoForOpTypes(Operation *op, TypeRange inputs, TypeRange outputs) {
  SmallVector<circt::hw::PortInfo> pinputs, poutputs;

  HandshakePortNameGenerator portNames(op);
  auto *ctx = op->getContext();

  Type i1Type = IntegerType::get(ctx, 1);

  // Add all inputs of funcOp.
  unsigned inIdx = 0;
  for (auto arg : llvm::enumerate(inputs)) {
    pinputs.push_back(
        {{portNames.inputName(arg.index()), channelWrapper(arg.value()),
          circt::hw::ModulePort::Direction::Input},
         arg.index(),
         {}});
    inIdx++;
  }

  // Add all outputs of funcOp.
  for (auto res : llvm::enumerate(outputs)) {
    poutputs.push_back(
        {{portNames.outputName(res.index()), channelWrapper(res.value()),
          circt::hw::ModulePort::Direction::Output},
         res.index(),
         {}});
  }

  // Add clock and reset signals.
  if (op->hasTrait<mlir::OpTrait::HasClock>()) {
    pinputs.push_back({{StringAttr::get(ctx, "clock"), i1Type,
                        circt::hw::ModulePort::Direction::Input},
                       inIdx++,
                       {}});
    pinputs.push_back({{StringAttr::get(ctx, "reset"), i1Type,
                        circt::hw::ModulePort::Direction::Input},
                       inIdx,
                       {}});
  }

  return circt::hw::ModulePortInfo{pinputs, poutputs};
}

//===----------------------------------------------------------------------===//
// Internal data-structures
//===----------------------------------------------------------------------===//

/// Name of port representing the clock signal.
static const std::string CLK_PORT = "clock";
/// Name of port representing the reset signal.
static const std::string RST_PORT = "reset";

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

class ModuleBuilder {
public:
  SmallVector<ModulePort> inputs;
  SmallVector<ModulePort> outputs;
  MLIRContext *ctx;

  ModuleBuilder(MLIRContext *ctx) : ctx(ctx){};

  ModulePortInfo build() {
    // Seems stupid but does not seem to be a way around it
    SmallVector<PortInfo> inputPorts;
    SmallVector<PortInfo> outputPorts;
    for (auto [idx, modPort] : llvm::enumerate(inputs))
      inputPorts.push_back(PortInfo{modPort, idx});
    for (auto [idx, modPort] : llvm::enumerate(outputs))
      outputPorts.push_back(PortInfo{modPort, idx});
    return ModulePortInfo(inputPorts, outputPorts);
  }

  void addInput(const Twine &name, Type type) {
    inputs.push_back(ModulePort{StringAttr::get(ctx, name), type,
                                ModulePort::Direction::Input});
  }

  void addOutput(const Twine &name, Type type) {
    outputs.push_back(ModulePort{StringAttr::get(ctx, name), type,
                                 ModulePort::Direction::Output});
  }

  /// Adds clock and reset ports to a module's inputs ports.
  void addClkAndRstPorts() {
    Type i1Type = IntegerType::get(ctx, 1);
    addInput(CLK_PORT, i1Type);
    addInput(RST_PORT, i1Type);
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

  /// Function's module.
  ModuleBuilder module;

  /// Index mapping between input ports and output ports that correspond to
  /// memory IOs. Each entry in the vector represent a different memory
  /// interface, with NUM_MEM_INPUTS input ports starting at the index contained
  /// in the pair's first value and with NUM_MEM_OUTPUTS output ports starting
  /// at the index contained in the pair's second value.
  std::vector<std::pair<size_t, size_t>> memIO;

  /// Initializes the struct with empty ports.
  FuncModulePortInfo(MLIRContext *ctx) : module(ctx){};

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
  memIO.emplace_back(module.inputs.size(), module.outputs.size());

  // Load data input
  module.addInput(name + "_load_data", dataType);
  // Load enable output
  module.addOutput(name + "_load_en", i1Type);
  // Load address output
  module.addOutput(name + "_load_addr", addrType);
  // Store enable output
  module.addOutput(name + "_store_en", i1Type);
  // Store address output
  module.addOutput(name + "_store_addr", addrType);
  // Store data output
  module.addOutput(name + "_store_data", dataType);
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

/// Extracts the data-carrying type of a value. If the value is a dataflow
/// channel, extracts the data-carrying type, else assumes that the value's type
/// itself is the data-carrying type.
static Type getOperandDataType(Value val) {
  auto valType = val.getType();
  if (auto channelType = valType.dyn_cast<handshake::ChannelType>())
    return channelType.getDataType();
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
      .Case<handshake::JoinOp>([&](auto) {
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
            arith::DivSIOp, arith::DivUIOp, arith::FloorDivSIOp, arith::MaxSIOp,
            arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp, arith::MulFOp,
            arith::MulIOp, arith::NegFOp, arith::OrIOp, arith::RemFOp,
            arith::RemSIOp, arith::RemUIOp, arith::ShLIOp, arith::ShRSIOp,
            arith::ShRUIOp, arith::SubFOp, arith::SubIOp, arith::XOrIOp>(
          [&](auto) {
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
static circt::hw::HWModuleLike findModule(mlir::ModuleOp parentModule,
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
static circt::hw::HWModuleLike getModule(HandshakeLoweringState &ls,
                                         StringRef modName,
                                         ModuleBuilder &module,
                                         Location modLoc) {
  circt::hw::HWModuleLike mod = findModule(ls.parentModule, modName);
  if (!mod)
    return ls.extModBuilder.create<circt::hw::HWModuleExternOp>(
        modLoc, ls.extModBuilder.getStringAttr(modName), module.build());
  return mod;
}

/// Returns a module-like operation that matches the provided name. The
/// functions first attempts to find an existing module with the same name,
/// which it returns if it exists. Failing the latter, the function creates an
/// external module at the provided location and with the provided ports.
static circt::hw::HWModuleLike getModule(HandshakeLoweringState &ls,
                                         StringRef modName,
                                         ModulePortInfo &module,
                                         Location modLoc) {
  circt::hw::HWModuleLike mod = findModule(ls.parentModule, modName);
  if (!mod)
    return ls.extModBuilder.create<circt::hw::HWModuleExternOp>(
        modLoc, ls.extModBuilder.getStringAttr(modName), module);
  return mod;
}

/// Derives port information for an operation so that it can be converted to a
/// hardware module.
static ModulePortInfo getPortInfo(Operation *op) {
  return getPortInfoForOpTypes(op, op->getOperandTypes(), op->getResultTypes());
}

/// Adds the clock and reset arguments of a module to the list of operands of an
/// operation within the module.
static void addClkAndRstOperands(SmallVector<Value> &operands,
                                 circt::hw::HWModuleOp mod) {
  auto numArguments = mod->getNumOperands();
  assert(numArguments >= 2 &&
         "module should have at least a clock and reset arguments");
  assert(mod.getPort(numArguments - 2).getName() == CLK_PORT &&
         "second to last module port should be clock");
  assert(mod.getPort(numArguments - 1).getName() == RST_PORT &&
         "last module port should be clock");
  operands.push_back(mod->getOperand(numArguments - 2));
  operands.push_back(mod->getOperand(numArguments - 1));
}

/// Derives port information for a handshake function so that it can be
/// converted to a hardware module.
static FuncModulePortInfo getFuncPortInfo(handshake::FuncOp funcOp) {

  MLIRContext *ctx = funcOp.getContext();
  FuncModulePortInfo info(ctx);

  // Add all outputs of function
  TypeRange outputs = funcOp.getResultTypes();
  for (auto [idx, res] : llvm::enumerate(outputs)) {
    info.module.addOutput(funcOp.getResName(idx).strref(), channelWrapper(res));
  }
  // Add all inputs of function, replacing memrefs with appropriate ports
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (isa<MemRefType>(arg.getType())) {
      info.addMemIO(dyn_cast<MemRefType>(arg.getType()),
                    funcOp.getArgName(idx).str(), ctx);
    } else {
      info.module.addInput(funcOp.getArgName(idx).strref(),
                           channelWrapper(arg.getType()));
    }
  }

  info.module.addClkAndRstPorts();
  return info;
}

/// Derives port information for a handshake memory controller so that it can be
/// converted to a hardware module. The function needs information from the
/// enclosing function's ports as well as the index of the memory interface
/// within that data-structure to derive the future module's ports.
static ModulePortInfo getMemPortInfo(handshake::MemoryControllerOp memOp,
                                     FuncModulePortInfo &info, size_t memIdx) {

  MLIRContext *ctx = memOp.getContext();
  ModuleBuilder module(ctx);

  auto &[inFuncPortIdx, outFuncPortIdx] = info.memIO[memIdx];

  // Add input ports coming from outside the containing module
  for (size_t i = inFuncPortIdx,
              e = inFuncPortIdx + FuncModulePortInfo::NUM_MEM_INPUTS;
       i < e; i++) {
    auto funcInput = info.module.inputs[i];
    module.addInput(funcInput.name.strref(), funcInput.type);
  }

  // Add input ports corresponding to memory interface operands
  for (auto [idx, arg] : llvm::enumerate(memOp.getMemInputs()))
    module.addInput(memOp.getOperandName(idx + 1),
                    channelWrapper(arg.getType()));

  // Add output ports corresponding to memory interface operands
  for (auto [idx, arg] : llvm::enumerate(memOp.getResults()))
    module.addOutput(memOp.getResultName(idx), channelWrapper(arg.getType()));

  // Add output ports going outside the containing module
  for (size_t i = outFuncPortIdx,
              e = outFuncPortIdx + FuncModulePortInfo::NUM_MEM_OUTPUTS;
       i < e; i++) {
    auto funcOutput = info.module.outputs[i];
    module.addOutput(funcOutput.name.strref(), funcOutput.type);
  }

  module.addClkAndRstPorts();
  return module.build();
}

/// Derives port information for the end operation of a handshake function so
/// that it can be converted to a hardware module. The function needs
/// information from the enclosing function's ports to determine the number of
/// return values in the original function that the future module should
/// output.
static ModulePortInfo getEndPortInfo(handshake::EndOp endOp,
                                     FuncModulePortInfo &info) {

  ModuleBuilder module(endOp.getContext());

  // Add input ports corresponding to end operands
  for (auto [idx, arg] : llvm::enumerate(endOp.getOperands()))
    module.addInput("in" + std::to_string(idx), channelWrapper(arg.getType()));

  // Add output ports corresponding to function return values
  auto numReturnValues = module.outputs.size() - info.getNumMemOutputs();
  auto returnValOperands = endOp.getOperands().take_front(numReturnValues);
  for (auto [idx, arg] : llvm::enumerate(returnValOperands))
    module.addOutput("out" + std::to_string(idx),
                     channelWrapper(arg.getType()));

  module.addClkAndRstPorts();
  return module.build();
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//
namespace {

/// A type converter is needed to perform the in-flight materialization of
/// "raw" (implicit channels) types to their explicit dataflow channel
/// correspondents.
class ChannelTypeConverter : public TypeConverter {
public:
  ChannelTypeConverter() {
    addConversion([](Type type) -> Type { return channelWrapper(type); });

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
/// circt::hw::HWModuleOp or circt::hw::HWModuleExternOp with IO corresponding
/// to the original handshake function. In the case where the matched function
/// is not external, the pattern additionally (1) buffers the function's inputs,
/// (2) converts internal memory interfaces to circt::hw::HWInstanceOp's and
/// connects them to the containing module IO, (3) converts the function's end
/// operation to a circt::hw::HWInstanceOp that also outputs the function's
/// return values, and (4) combines all the module outputs in the
/// circt::hw::HwOutputOp operation at the end of the module.
class FuncOpConversionPattern : public OpConversionPattern<handshake::FuncOp> {
public:
  FuncOpConversionPattern(MLIRContext *ctx, HandshakeLoweringState &ls)
      : OpConversionPattern<handshake::FuncOp>(ctx, 1), ls(ls) {}

  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    FuncModulePortInfo info = getFuncPortInfo(op);

    if (op.isExternal())
      ls.extModBuilder.create<circt::hw::HWModuleExternOp>(
          op.getLoc(), ls.extModBuilder.getStringAttr(op.getName()),
          info.module.build());
    else {
      // Create module for the function
      auto mod = ls.extModBuilder.create<circt::hw::HWModuleOp>(
          op.getLoc(), ls.extModBuilder.getStringAttr(op.getName()),
          info.module.build());

      // Replace uses of function arguments with module inputs
      for (auto it : llvm::zip(op.getArguments(), mod->getOperands()))
        std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

      // Move all operations to the new module
      auto &moduleBlockOps = mod.getBodyBlock()->getOperations();
      moduleBlockOps.splice(moduleBlockOps.begin(),
                            op.getBody().front().getOperations());

      // Insert a start buffer for each channel-typed input
      bufferInputs(mod, rewriter);

      // Convert memory interfaces (and connect them with function IO)
      auto memInstances = convertMemories(mod, info, rewriter);

      // Convert end operation (add results to represent function return
      // values)
      auto endInstance = convertEnd(mod, info, rewriter);

      // Set operands of output operation to match module outputs
      setModuleOutputs(mod, memInstances, endInstance);
    }

    // Original function can safely be deleted before returning
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Lowering state to help in the creation of new hardware
  /// modules/instances.
  HandshakeLoweringState &ls;

  /// Inserts a "start module" that acts as a buffer for all module inputs
  /// that are of type handshake::ChannelType. This is done to match legacy
  /// Dynamatic's implementation of circuits.
  void bufferInputs(HWModuleOp mod, ConversionPatternRewriter &rewriter) const;

  /// Converts memory interfaces within the module into hardware instances
  /// with added IO to handle interactions with external memory through the
  /// module IO. The function returns the list of newly created hardware
  /// instances.
  SmallVector<circt::hw::InstanceOp>
  convertMemories(HWModuleOp mod, FuncModulePortInfo &info,
                  ConversionPatternRewriter &rewriter) const;

  /// Converts the end operation of a handshake function into a corresponding
  /// hardware instance with added outputs to hold the function return values.
  /// The function returns the created hardware instance.
  circt::hw::InstanceOp convertEnd(HWModuleOp mod, FuncModulePortInfo &info,
                                   ConversionPatternRewriter &rewriter) const;

  /// Modifies the operands of the circt::hw::OutputOp operation within the
  /// newly created module to match the latter's outputs.
  void setModuleOutputs(HWModuleOp mod,
                        SmallVector<circt::hw::InstanceOp> memInstances,
                        circt::hw::InstanceOp endInstance) const;
};

/// Converts an operation (of type indicated by the template argument) into an
/// equivalent hardware instance. The method creates an external module to
/// instantiate the new component from if a module with matching IO one does
/// not already exist. Valid/Ready semantics are made explicit thanks to the
/// type converter which converts implicit handshaked types into dataflow
/// channels with a corresponding data-type.
template <typename T>
class ExtModuleConversionPattern : public OpConversionPattern<T> {
public:
  ExtModuleConversionPattern(ChannelTypeConverter &typeConverter,
                             MLIRContext *ctx, HandshakeLoweringState &ls)
      : OpConversionPattern<T>::OpConversionPattern(typeConverter, ctx),
        ls(ls) {}
  using OpAdaptor = typename T::Adaptor;

  LogicalResult
  matchAndRewrite(T op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto modName = getExtModuleName(op);
    auto ports = getPortInfo(op);
    circt::hw::HWModuleLike extModule =
        getModule(ls, modName, ports, op.getLoc());

    // Replace operation with corresponding hardware module instance
    SmallVector<Value> operands = adaptor.getOperands();
    if (op.template hasTrait<mlir::OpTrait::HasClock>())
      addClkAndRstOperands(operands,
                           cast<circt::hw::HWModuleOp>(op->getParentOp()));

    auto instanceOp = rewriter.replaceOpWithNewOp<circt::hw::InstanceOp>(
        op, extModule, rewriter.getStringAttr(ls.nameUniquer(op)), operands);

    // Replace operation results with new instance results
    for (auto it : llvm::zip(op->getResults(), instanceOp->getResults()))
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

    return success();
  }

private:
  /// Lowering state to help in the creation of new hardware
  /// modules/instances.
  HandshakeLoweringState &ls;
};
} // namespace

void FuncOpConversionPattern::bufferInputs(
    HWModuleOp mod, ConversionPatternRewriter &rewriter) const {

  unsigned argIdx = 0;
  rewriter.setInsertionPointToStart(mod.getBodyBlock());

  for (Value arg : mod->getOperands()) {
    auto argType = arg.getType();
    if (!isa<handshake::ChannelType>(argType))
      continue;

    ModuleBuilder module(mod.getContext());

    // Create ports for input buffer
    module.addInput("in0", argType);
    module.addClkAndRstPorts();

    // Module output matches function argument type
    module.addOutput("out0", argType);

    // Check whether we need to create a new external module
    auto argLoc = arg.getLoc();
    std::string modName = "handshake_start_node";
    if (auto channelType = argType.dyn_cast<handshake::ChannelType>())
      modName += "." + getTypeName(channelType.getDataType(), argLoc);

    circt::hw::HWModuleLike extModule = getModule(ls, modName, module, argLoc);

    // Replace operation with corresponding hardware module instance
    llvm::SmallVector<Value> operands;
    operands.push_back(arg);
    addClkAndRstOperands(operands, mod);
    auto instanceOp = rewriter.create<circt::hw::InstanceOp>(
        argLoc, extModule,
        rewriter.getStringAttr("handshake_start_node" +
                               std::to_string(argIdx++)),
        operands);

    rewriter.replaceAllUsesExcept(arg, instanceOp.getResult(0), instanceOp);
  }
}

SmallVector<circt::hw::InstanceOp> FuncOpConversionPattern::convertMemories(
    HWModuleOp mod, FuncModulePortInfo &info,
    ConversionPatternRewriter &rewriter) const {

  SmallVector<circt::hw::InstanceOp> instances;

  for (auto [memIdx, portIndices] : llvm::enumerate(info.memIO)) {
    // Identify the memory controller refering to the memref
    Operation *user = *mod->getOperand(portIndices.first).getUsers().begin();
    assert(user && "old memref value should have a user");
    auto memOp = dyn_cast<handshake::MemoryControllerOp>(user);
    assert(memOp && "user of old memref value should be memory interface");

    std::string memName = getExtModuleName(memOp);
    ModulePortInfo ports = getMemPortInfo(memOp, info, memIdx);

    // Create an external module definition if necessary
    circt::hw::HWModuleLike extModule =
        getModule(ls, memName, ports, memOp->getLoc());

    // Combine memory inputs from the function and internal memory inputs into
    // the new instance operands
    SmallVector<Value> operands;
    for (auto i = portIndices.first,
              e = portIndices.first + FuncModulePortInfo::NUM_MEM_INPUTS;
         i < e; i++)
      operands.push_back(mod->getOperand(i));
    operands.insert(operands.end(), memOp.getMemInputs().begin(),
                    memOp.getMemInputs().end());
    addClkAndRstOperands(operands, mod);

    // Create instance of memory interface
    rewriter.setInsertionPoint(memOp);
    auto instance = rewriter.create<circt::hw::InstanceOp>(
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

circt::hw::InstanceOp
FuncOpConversionPattern::convertEnd(HWModuleOp mod, FuncModulePortInfo &info,
                                    ConversionPatternRewriter &rewriter) const {
  // End operation is guaranteed to exist and be unique
  auto endOp = *mod.getBodyBlock()->getOps<handshake::EndOp>().begin();

  // Create external module
  auto extModule = ls.extModBuilder.create<circt::hw::HWModuleExternOp>(
      endOp->getLoc(), ls.extModBuilder.getStringAttr(getExtModuleName(endOp)),
      getEndPortInfo(endOp, info));

  // Create instance of end operation
  SmallVector<Value> operands(endOp.getOperands());
  addClkAndRstOperands(operands, mod);
  rewriter.setInsertionPoint(endOp);
  auto instance = rewriter.create<circt::hw::InstanceOp>(
      endOp.getLoc(), extModule, rewriter.getStringAttr(ls.nameUniquer(endOp)),
      operands);

  rewriter.eraseOp(endOp);
  return instance;
}

void FuncOpConversionPattern::setModuleOutputs(
    HWModuleOp mod, SmallVector<circt::hw::InstanceOp> memInstances,
    circt::hw::InstanceOp endInstance) const {
  // Output operation is guaranteed to exist and be unique
  auto outputOp = *mod.getBodyBlock()->getOps<circt::hw::OutputOp>().begin();

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
            .Case<handshake::DynamaticReturnOp>(
                [&](handshake::DynamaticReturnOp retOp) -> LogicalResult {
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
/// multiple one-slot buffers. Our RTL buffer components cannot be
/// parameterized with a number of slots (they are all 1-slot) so we have to
/// do this unpacking prior to running the conversion pass.
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
/// The function and all the operations it contains are converted to
/// operations from the HW dialect. Dataflow semantics are made explicit with
/// Handshake channels.
class HandshakeToNetListPass
    : public dynamatic::impl::HandshakeToNetlistBase<HandshakeToNetListPass> {
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
    if (failed(verifyIRMaterialized(funcOp))) {
      funcOp.emitOpError() << ERR_NON_MATERIALIZED_FUNC;
      return signalPassFailure();
    }
    if (failed(verifyAllIndexConcretized(funcOp))) {
      funcOp.emitOpError() << "Lowering to netlist requires that all index "
                              "types in the IR have "
                              "been concretized."
                           << ERR_RUN_CONCRETIZATION;
      return signalPassFailure();
    }
    if (failed(verifyExportToRTL(funcOp)))
      return signalPassFailure();

    // Run a pre-processing pass on the IR
    if (failed(preprocessMod())) {
      mod->emitError() << "Failed to pre-process IR to make it valid for "
                          "netlist conversion";
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
    ChannelTypeConverter typeConverter;
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
    target.addLegalOp<circt::hw::HWModuleOp, circt::hw::HWModuleExternOp,
                      circt::hw::OutputOp, circt::hw::InstanceOp>();
    target.addIllegalDialect<handshake::HandshakeDialect, arith::ArithDialect,
                             memref::MemRefDialect>();

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
      return signalPassFailure();
  }

private:
  /// Perfoms some simple transformations on the module to make the netlist
  /// that will result from the conversion able to be turned into an RTL
  /// design. NOTE: (RamirezLucas) Ideally, this should be moved to a separate
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
