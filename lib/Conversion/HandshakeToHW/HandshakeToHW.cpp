//===- HandshakeToHW.cpp - Convert Handshake to HW --------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts Handshake constructs into equivalent HW constructs.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Conversion/HandshakeToHW.h"
#include "dynamatic/Dialect/HW/HWOpInterfaces.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/HW/HWTypes.h"
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace dynamatic;

/// Name of ports representing the clock and reset signals.
static constexpr llvm::StringLiteral CLK_PORT("clk"), RST_PORT("rst");

//===----------------------------------------------------------------------===//
// Internal data-structures
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

/// Aggregates information to convert a Handshake memory interface into a
/// `hw::InstanceOp`. This must be created during conversion of the Handsahke
/// function containing the interface.
struct MemLoweringState {
  /// Index of first module input corresponding the the interface's inputs.
  size_t inputIdx;
  /// Number of inputs for the memory interface, starting at the `inputIdx`.
  size_t numInputs;
  /// Index of first module output corresponding the the interface's ouputs.
  size_t outputIdx;
  /// Number of outputs for the memory interface, starting at the `outputIdx`.
  size_t numOutputs;

  /// Cache memory port information before modifying the interface, which can
  /// make them impossible to query.
  FuncMemoryPorts ports;
  /// Cache list of operand names before modifying the interface, which can
  /// make them impossible to query.
  SmallVector<std::string> operandNames;
  /// Cache list of result names before modifying the interface, which can
  /// make them impossible to query.
  SmallVector<std::string> resultNames;
  /// Backedges to the containing module's `hw::OutputOp` operation, which must
  /// be set, in order, with the memory interface's results that connect to the
  /// top-level module IO.
  SmallVector<Backedge> backedges;

  /// Needed because we use the class as a value type in a map, which needs to
  /// be default-constructible.
  MemLoweringState() : ports(nullptr) {
    llvm_unreachable("object should never be default-constructed");
  }

  /// Construcst an instance of the object for the provided memory interface.
  /// Input/Output indices refer to the IO of the `hw::HWModuleOp` that is
  /// generated for the Handshake function containing the interface.
  MemLoweringState(handshake::MemoryOpInterface memOp, size_t inputIdx = 0,
                   size_t numInputs = 0, size_t outputIdx = 0,
                   size_t numOutputs = 0);

  /// Returns the module's input ports that connect to the memory interface.
  SmallVector<hw::ModulePort> getMemInputPorts(hw::HWModuleOp modOp);

  /// Returns the module's output ports that the memory interface connects to.
  SmallVector<hw::ModulePort> getMemOutputPorts(hw::HWModuleOp modOp);
};

/// Summarizes information to convert a Handshake function into a
/// `hw::HWModuleOp`.
struct ModuleLoweringState {
  /// Maps each Handshake memory interface in the module with information on how
  /// to convert it into equivalent HW constructs.
  llvm::DenseMap<handshake::MemoryOpInterface, MemLoweringState> memInterfaces;
  /// Backedges to the containing module's `hw::OutputOp` operation, which must
  /// be set, in order, with the results of the `hw::InstanceOp` operation to
  /// which the `handshake::EndOp` operation was converted to.
  SmallVector<Backedge> endBackedges;

  /// Computes the total number of module outputs that are fed by memory
  /// interfaces within the module.
  size_t getNumMemOutputs() {
    size_t numOutputs = 0;
    for (auto &[memOp, info] : memInterfaces)
      numOutputs += info.numOutputs;
    return numOutputs;
  }
};

/// Shared state used during lowering. Captured in a struct to reduce the number
/// of arguments we have to pass around.
struct LoweringState {
  /// Top-level MLIR module.
  mlir::ModuleOp modOp;
  /// Reference to the pass's name analysis, to query unique names for each
  /// operation.
  NameAnalysis &namer;
  /// Allowa to create transient backedges for the `hw::OutputOp` of every
  /// create `hw::HWModuleOp`. We need backedges because, at the moment where a
  /// Handshake function is turned into a HW module, the operands to the
  /// `hw::OutputOp` terminator are not yet available but instead progressively
  /// appear as other oeprations inside the HW module are converted.
  BackedgeBuilder edgeBuilder;

  /// Maps each created `hw::HWModuleOp` during conversion to some lowering
  /// state required to convert the operations nested within it.
  DenseMap<hw::HWModuleOp, ModuleLoweringState> modState;

  /// Creates the lowering state. The builder is passed to a `BackedgeBuilder`
  /// used during the conversion to create transient operands.
  LoweringState(mlir::ModuleOp modOp, NameAnalysis &namer, OpBuilder &builder);
};
} // namespace

MemLoweringState::MemLoweringState(handshake::MemoryOpInterface memOp,
                                   size_t inputIdx, size_t numInputs,
                                   size_t outputIdx, size_t numOutputs)
    : inputIdx(inputIdx), numInputs(numInputs), outputIdx(outputIdx),
      numOutputs(numOutputs), ports(getMemoryPorts(memOp)) {
  // To extract operand/result names from the memory interface
  auto namedMemOp = cast<handshake::NamedIOInterface>(memOp.getOperation());

  // Cache the list of operand and result names of the memory interface's ports,
  // as they may become invalid during conversion
  for (size_t idx = 0; idx < memOp->getNumOperands(); ++idx)
    operandNames.push_back(namedMemOp.getOperandName(idx));
  for (size_t idx = 0; idx < memOp->getNumResults(); ++idx)
    resultNames.push_back(namedMemOp.getResultName(idx));
};

SmallVector<hw::ModulePort>
MemLoweringState::getMemInputPorts(hw::HWModuleOp modOp) {
  if (numInputs == 0)
    return {};
  assert(inputIdx + numInputs <= modOp.getNumInputPorts() &&
         "input index too high");
  SmallVector<hw::ModulePort> ports;
  size_t idx = modOp.getPortIdForInputId(inputIdx);
  ArrayRef<hw::ModulePort> inputPorts = modOp.getModuleType().getPorts();
  return SmallVector<hw::ModulePort>{inputPorts.slice(idx, numInputs)};
}

SmallVector<hw::ModulePort>
MemLoweringState::getMemOutputPorts(hw::HWModuleOp modOp) {
  if (numOutputs == 0)
    return {};
  assert(outputIdx + numOutputs <= modOp.getNumOutputPorts() &&
         "output index too high");
  SmallVector<hw::ModulePort> ports;
  size_t idx = modOp.getPortIdForOutputId(outputIdx);
  ArrayRef<hw::ModulePort> outputPorts = modOp.getModuleType().getPorts();
  return SmallVector<hw::ModulePort>{outputPorts.slice(idx, numOutputs)};
}

LoweringState::LoweringState(mlir::ModuleOp modOp, NameAnalysis &namer,
                             OpBuilder &builder)
    : modOp(modOp), namer(namer), edgeBuilder(builder, modOp.getLoc()){};

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
static std::string getTypeName(Type type) {
  if (isa<IntegerType, FloatType>(type))
    return std::to_string(type.getIntOrFloatBitWidth());
  if (isa<NoneType>(type))
    return "0";
  llvm_unreachable("unsupported data type");
}

/// Adds the clock and reset arguments of a module to the list of operands of an
/// operation within the module.
static void addClkAndRstOperands(SmallVector<Value> &operands,
                                 hw::HWModuleOp mod) {
  unsigned numInputs = mod.getNumInputPorts();
  assert(numInputs >= 2 && "module should have at least clock and reset");
  size_t lastIdx = mod.getPortIdForInputId(numInputs - 1);
  assert(mod.getPort(lastIdx - 1).getName() == CLK_PORT && "expected clock");
  assert(mod.getPort(lastIdx).getName() == RST_PORT && "expected reset");

  auto blockArgs = mod.getBodyBlock()->getArguments();
  operands.push_back(blockArgs.drop_back().back());
  operands.push_back(blockArgs.back());
}

/// Constructs an external module name corresponding to an operation. The
/// returned name is unique with respect to the operation's discriminating
/// parameters.
static std::string getExtModuleName(Operation *oldOp) {
  std::string extModName = getBareExtModuleName(oldOp);
  extModName += "_node.";
  auto types = getDiscriminatingParameters(oldOp);
  SmallVector<Type> &inTypes = types.first;
  SmallVector<Type> &outTypes = types.second;

  llvm::TypeSwitch<Operation *>(oldOp)
      .Case<handshake::OEHBOp>(
          [&](auto) { extModName += "seq_" + getTypeName(outTypes[0]); })
      .Case<handshake::TEHBOp>(
          [&](auto) { extModName += "fifo_" + getTypeName(outTypes[0]); })
      .Case<handshake::ForkOp, handshake::LazyForkOp>([&](auto) {
        // number of outputs
        extModName += std::to_string(outTypes.size());
        // bitwidth
        extModName += "_" + getTypeName(outTypes[0]);
      })
      .Case<handshake::MuxOp>([&](auto) {
        // number of inputs (without select param)
        extModName += std::to_string(inTypes.size() - 1);
        // bitwidth
        extModName += "_" + getTypeName(inTypes[1]);
        // select bitwidth
        extModName += "_" + getTypeName(inTypes[0]);
      })
      .Case<handshake::ControlMergeOp>([&](auto) {
        // number of inputs
        extModName += std::to_string(inTypes.size());
        // bitwidth
        extModName += "_" + getTypeName(inTypes[0]);
        // index result bitwidth
        extModName += "_" + getTypeName(outTypes[outTypes.size() - 1]);
      })
      .Case<handshake::MergeOp>([&](auto) {
        // number of inputs
        extModName += std::to_string(inTypes.size());
        // bitwidth
        extModName += "_" + getTypeName(inTypes[0]);
      })
      .Case<handshake::ConditionalBranchOp>([&](auto) {
        // bitwidth
        extModName += getTypeName(inTypes[1]);
      })
      .Case<handshake::BranchOp, handshake::SinkOp, handshake::SourceOp>(
          [&](auto) {
            // bitwidth
            if (!inTypes.empty())
              extModName += getTypeName(inTypes[0]);
            else
              extModName += getTypeName(outTypes[0]);
          })
      .Case<handshake::LoadOpInterface, handshake::StoreOpInterface>([&](auto) {
        // data bitwidth
        extModName += getTypeName(inTypes[0]);
        // address bitwidth
        extModName += "_" + getTypeName(inTypes[1]);
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
        extModName += "_" + getTypeName(outTypes[0]);
      })
      .Case<handshake::JoinOp>([&](auto) {
        // array of bitwidths
        for (auto inType : inTypes)
          extModName += getTypeName(inType) + "_";
        extModName = extModName.substr(0, extModName.size() - 1);
      })
      .Case<handshake::EndOp>([&](auto) {
        // mem_inputs
        extModName += std::to_string(inTypes.size() - 1);
        // bitwidth
        extModName += "_" + getTypeName(inTypes[0]);
      })
      .Case<handshake::ReturnOp>([&](auto) {
        // bitwidth
        extModName += getTypeName(inTypes[0]);
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
      .Case<handshake::LSQOp>([&](handshake::LSQOp op) {
        FuncMemoryPorts ports = op.getPorts();
        // Data bitwidth
        extModName += std::to_string(ports.dataWidth);
        // Address bitwidth
        extModName += '_' + std::to_string(ports.addrWidth);
        // Port counts
        extModName += '_' + std::to_string(ports.getNumPorts<LoadPort>()) +
                      '_' + std::to_string(ports.getNumPorts<StorePort>()) +
                      '_' + std::to_string(ports.getNumPorts<ControlPort>());
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
            extModName += getTypeName(inTypes[0]);
          })
      .Case<arith::SelectOp>([&](auto) {
        // bitwidth
        extModName += getTypeName(outTypes[0]);
      })
      .Case<arith::CmpFOp, arith::CmpIOp>([&](auto) {
        // predicate
        if (auto cmpOp = dyn_cast<mlir::arith::CmpIOp>(oldOp))
          extModName += stringifyEnum(cmpOp.getPredicate()).str();
        else if (auto cmpOp = dyn_cast<mlir::arith::CmpFOp>(oldOp))
          extModName += stringifyEnum(cmpOp.getPredicate()).str();
        // bitwidth
        extModName += "_" + getTypeName(inTypes[0]);
      })
      .Case<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::FPToSIOp,
            arith::FPToUIOp, arith::SIToFPOp, arith::TruncFOp, arith::TruncIOp,
            arith::UIToFPOp>([&](auto) {
        // input bitwidth
        extModName += getTypeName(inTypes[0]);
        // output bitwidth
        extModName += "_" + getTypeName(outTypes[0]);
      })
      .Default([&](auto) {
        oldOp->emitError() << "No matching component for operation";
      });

  return extModName;
}

namespace {

/// Helper class to build HW modules progressively by adding inputs/outputs one
/// at time.
class ModuleBuilder {
public:
  /// List of module inputs.
  SmallVector<hw::ModulePort> inputs;
  /// List of module outputs.
  SmallVector<hw::ModulePort> outputs;

  /// The MLIR context is used to create string attributes for port names and
  /// types for the clock and reset ports, should they be added.
  ModuleBuilder(MLIRContext *ctx) : ctx(ctx){};

  /// Builds the module port information from the current list of inputs and
  /// outputs.
  hw::ModulePortInfo getPortInfo();

  /// If an external HW module matching the operation already exists, returns
  /// it. Otherwise, creates one at the bottom of the top-level MLIR module
  /// using the currently registered input and output ports and return it.
  hw::HWModuleExternOp getModule(Operation *op,
                                 ConversionPatternRewriter &rewriter);

  /// Adds an input port with the given name and type at the end of the existing
  /// list of input ports.
  void addInput(const Twine &name, Type type);

  /// Adds an output port with the given name and type at the end of the
  /// existing list of output ports.
  void addOutput(const Twine &name, Type type);

  /// Adds clock and reset ports at the end of the existing list of input ports.
  void addClkAndRstPorts();

  /// Returns the MLIR context used by the builder.
  MLIRContext *getContext() { return ctx; }

  /// Attempts to find an external HW module in the MLIR module with the
  /// provided name. Returns it if it exists, otherwise returns `nullptr`.
  static hw::HWModuleExternOp findModule(mlir::ModuleOp modOp, StringRef name);

private:
  /// MLIR context for type/attribute creation.
  MLIRContext *ctx;
};
} // namespace

hw::ModulePortInfo ModuleBuilder::getPortInfo() {
  SmallVector<hw::PortInfo> inputPorts;
  SmallVector<hw::PortInfo> outputPorts;
  for (auto [idx, modPort] : llvm::enumerate(inputs))
    inputPorts.push_back(hw::PortInfo{modPort, idx});
  for (auto [idx, modPort] : llvm::enumerate(outputs))
    outputPorts.push_back(hw::PortInfo{modPort, idx});
  return hw::ModulePortInfo(inputPorts, outputPorts);
}

hw::HWModuleExternOp
ModuleBuilder::getModule(Operation *op, ConversionPatternRewriter &rewriter) {
  StringAttr name = rewriter.getStringAttr(getExtModuleName(op));
  mlir::ModuleOp topLevelModOp = op->getParentOfType<mlir::ModuleOp>();
  assert(topLevelModOp && "missing top-level module");
  if (hw::HWModuleExternOp mod = findModule(topLevelModOp, name))
    return mod;
  rewriter.setInsertionPointToEnd(topLevelModOp.getBody());
  return rewriter.create<hw::HWModuleExternOp>(op->getLoc(), name,
                                               getPortInfo());
}

void ModuleBuilder::addInput(const Twine &name, Type type) {
  inputs.push_back(hw::ModulePort{StringAttr::get(ctx, name), type,
                                  hw::ModulePort::Direction::Input});
}

void ModuleBuilder::addOutput(const Twine &name, Type type) {
  outputs.push_back(hw::ModulePort{StringAttr::get(ctx, name), type,
                                   hw::ModulePort::Direction::Output});
}

void ModuleBuilder::addClkAndRstPorts() {
  Type i1Type = IntegerType::get(ctx, 1);
  addInput(CLK_PORT, i1Type);
  addInput(RST_PORT, i1Type);
}

hw::HWModuleExternOp ModuleBuilder::findModule(mlir::ModuleOp modOp,
                                               StringRef name) {
  if (hw::HWModuleExternOp mod = modOp.lookupSymbol<hw::HWModuleExternOp>(name))
    return mod;
  return nullptr;
}

/// Adds IO to the module builder for the provided memref, using the provided
/// name to unique IO port names. All Handshake memory interfaces referencing
/// the memref inside the function are added to the module lowering state memory
/// interface map, along with helper lowering state.
static void addMemIO(ModuleBuilder &modBuilder, handshake::FuncOp funcOp,
                     TypedValue<MemRefType> memref, StringRef argName,
                     ModuleLoweringState &state) {
  /// Adds a single input port and 5 output ports to the module's IO
  auto addIO = [&]() -> void {
    MLIRContext *ctx = modBuilder.getContext();

    /// TODO: address bus width hardcoded to 32 for now
    Type addrType = IntegerType::get(ctx, 32);
    Type i1Type = IntegerType::get(ctx, 1);
    Type dataType = memref.getType().getElementType();

    // Load data input
    modBuilder.addInput(argName + "_loadData", dataType);
    // Load enable output
    modBuilder.addOutput(argName + "_loadEn", i1Type);
    // Load address output
    modBuilder.addOutput(argName + "_loadAddr", addrType);
    // Store enable output
    modBuilder.addOutput(argName + "_storeEn", i1Type);
    // Store address output
    modBuilder.addOutput(argName + "_storeAddr", addrType);
    // Store data output
    modBuilder.addOutput(argName + "_storeData", dataType);
  };

  // Find all memory interfaces that refer to the memory region
  for (auto memOp : funcOp.getOps<handshake::MemoryOpInterface>()) {
    // The interface must reference this memory region
    if (memOp.getMemRef() != memref)
      continue;

    MemLoweringState info =
        llvm::TypeSwitch<Operation *, MemLoweringState>(memOp.getOperation())
            .Case<handshake::MemoryControllerOp>(
                [&](handshake::MemoryControllerOp mcOp) {
                  auto info = MemLoweringState{memOp, modBuilder.inputs.size(),
                                               1, modBuilder.outputs.size(), 5};
                  addIO();
                  return info;
                })
            .Case<handshake::LSQOp>([&](handshake::LSQOp lsqOp) {
              if (!lsqOp.isConnectedToMC()) {
                // If the LSQ does not connect to an MC, then it
                // connects directly to top-level module IO
                auto info = MemLoweringState{memOp, modBuilder.inputs.size(), 1,
                                             modBuilder.outputs.size(), 5};
                addIO();
                return info;
              }
              return MemLoweringState(memOp);
            })
            .Default([&](auto) {
              llvm_unreachable("unknown memory interface");
              return MemLoweringState(memOp);
            });
    state.memInterfaces.insert({memOp, info});
  }
}

/// Produces the port information for the HW module that will replace the
/// Handshake function. Fills in the lowering state object with information that
/// will allow the conversion pass to connect memory interface to their
/// top-level IO later on.
hw::ModulePortInfo getFuncPortInfo(handshake::FuncOp funcOp,
                                   ModuleLoweringState &state) {
  ModuleBuilder modBuilder(funcOp.getContext());

  // Add all function outputs to the module
  for (auto [idx, res] : llvm::enumerate(funcOp.getResultTypes()))
    modBuilder.addOutput(funcOp.getResName(idx).strref(), channelWrapper(res));

  // Add all function inputs to the module, expanding memory references into a
  // set of individual ports for loads and stores
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    StringAttr argName = funcOp.getArgName(idx);
    if (TypedValue<MemRefType> memref = dyn_cast<TypedValue<MemRefType>>(arg))
      addMemIO(modBuilder, funcOp, memref, argName, state);
    else
      modBuilder.addInput(argName.strref(), channelWrapper(arg.getType()));
  }

  modBuilder.addClkAndRstPorts();
  return modBuilder.getPortInfo();
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//
namespace {

/// A type converter is needed to perform the in-flight materialization of
/// "raw" (implicit channels) types to their explicit dataflow channel
/// correspondents.
class ChannelTypeConverter : public TypeConverter {
public:
  ChannelTypeConverter() {
    addConversion([](Type type) -> Type {
      if (isa<MemRefType>(type))
        return type;
      return channelWrapper(type);
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

/// Converts a Handshake function into a HW module. The pattern creates a
/// `hw::HWModuleOp` or `hw::HWModuleExternOp` with IO corresponding to the
/// original Handshake function. In case of non-external function, the pattern
/// creates a lowering state object associated to the created HW module to
/// control the conversion of other operations within the module.
class ConvertFunc : public OpConversionPattern<handshake::FuncOp> {
public:
  ConvertFunc(ChannelTypeConverter &typeConverter, MLIRContext *ctx,
              LoweringState &lowerState)
      : OpConversionPattern<handshake::FuncOp>(typeConverter, ctx),
        lowerState(lowerState) {}

  LogicalResult
  matchAndRewrite(handshake::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  /// Shared lowering state.
  LoweringState &lowerState;
};
} // namespace

LogicalResult
ConvertFunc::matchAndRewrite(handshake::FuncOp funcOp, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
  ModuleLoweringState state;
  hw::ModulePortInfo modInfo = getFuncPortInfo(funcOp, state);
  StringAttr name = rewriter.getStringAttr(funcOp.getName());

  rewriter.setInsertionPoint(funcOp);

  // External functions simply have to be turned into external HW modules
  if (funcOp.isExternal()) {
    rewriter.replaceOpWithNewOp<hw::HWModuleExternOp>(funcOp, name, modInfo);
    return success();
  }

  // Create non-external HW module to replace the function with
  auto modOp = rewriter.create<hw::HWModuleOp>(funcOp.getLoc(), name, modInfo);

  // Move the block from the Handshake function to the new HW module, after
  // which the Handshake function becomes empty and can be deleted
  Block *funcBlock = funcOp.getBodyBlock();
  Block *modBlock = modOp.getBodyBlock();
  Operation *termOp = modBlock->getTerminator();
  ValueRange modBlockArgs = modBlock->getArguments().drop_back(2);
  rewriter.inlineBlockBefore(funcBlock, termOp, modBlockArgs);
  rewriter.eraseOp(funcOp);

  // Collect output ports of the top-level module for indexed access
  SmallVector<const hw::PortInfo *> outputPorts;
  for (const hw::PortInfo &outPort : modInfo.getOutputs())
    outputPorts.push_back(&outPort);

  // Crerate backege inputs for the module's output operation and associate them
  // to the future operations whose conversion will resolve them
  SmallVector<Value> outputOperands;
  size_t portIdx = 0;
  auto addBackedge = [&](SmallVector<Backedge> &backedges) -> void {
    const hw::PortInfo *port = outputPorts[portIdx++];
    Backedge backedge = lowerState.edgeBuilder.get(port->type);
    outputOperands.push_back(backedge);
    backedges.push_back(backedge);
  };

  size_t numEndResults = modInfo.sizeOutputs() - state.getNumMemOutputs();
  for (size_t i = 0; i < numEndResults; ++i)
    addBackedge(state.endBackedges);

  for (auto &[_, MemLoweringState] : state.memInterfaces) {
    for (size_t i = 0; i < MemLoweringState.numOutputs; ++i)
      addBackedge(MemLoweringState.backedges);
  }

  Operation *outputOp = modOp.getBodyBlock()->getTerminator();
  rewriter.setInsertionPoint(outputOp);
  rewriter.replaceOpWithNewOp<hw::OutputOp>(outputOp, outputOperands);

  // Associate the newly created module to its lowering state object
  lowerState.modState[modOp] = state;
  return success();
}

namespace {
/// Converts the Handshake-level terminator into a HW instance (and an external
/// HW module, potentially). This is special-cased because (1) the operation's
/// IO changes during conversion (essentially copying a subset of its inputs to
/// outputs) and (2) outputs of the HW instance need to connect to the HW-level
/// terminator.
class ConvertEnd : public OpConversionPattern<handshake::EndOp> {
public:
  ConvertEnd(ChannelTypeConverter &typeConverter, MLIRContext *ctx,
             LoweringState &lowerState)
      : OpConversionPattern<handshake::EndOp>(typeConverter, ctx),
        lowerState(lowerState) {}

  LogicalResult
  matchAndRewrite(handshake::EndOp endOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  /// Shared lowering state.
  LoweringState &lowerState;
};
} // namespace

LogicalResult
ConvertEnd::matchAndRewrite(handshake::EndOp endOp, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
  hw::HWModuleOp parentModOp = endOp->getParentOfType<hw::HWModuleOp>();
  ModuleLoweringState &modState = lowerState.modState[parentModOp];
  ModuleBuilder modBuilder(endOp.getContext());

  // Inputs to the module are identical the the original Handshake end
  // operation, plus clock and reset
  for (auto [idx, arg] : llvm::enumerate(endOp.getOperands()))
    modBuilder.addInput("in" + std::to_string(idx),
                        channelWrapper(arg.getType()));
  modBuilder.addClkAndRstPorts();

  // The end operation has one input per memory interface in the function which
  // should not be forwarded to its output ports
  unsigned numMemOperands = modState.memInterfaces.size();
  auto numReturnValues = endOp.getNumOperands() - numMemOperands;
  auto returnValOperands = endOp.getOperands().take_front(numReturnValues);

  // All non-memory inputs to the Handshake end operations should be forwarded
  // to its outputs
  for (auto [idx, arg] : llvm::enumerate(returnValOperands))
    modBuilder.addOutput("out" + std::to_string(idx),
                         channelWrapper(arg.getType()));

  // Replace Handshake end operation with a HW instance
  SmallVector<Value> instOperands(adaptor.getOperands());
  addClkAndRstOperands(instOperands, parentModOp);
  StringAttr name = rewriter.getStringAttr(lowerState.namer.getName(endOp));
  hw::HWModuleLike extModOp = modBuilder.getModule(endOp, rewriter);
  rewriter.setInsertionPoint(endOp);
  ValueRange results =
      rewriter
          .create<hw::InstanceOp>(endOp.getLoc(), extModOp, name, instOperands)
          .getResults();

  // Resolve backedges in the module's terminator that are coming from the end
  for (auto [backedge, res] : llvm::zip_equal(modState.endBackedges, results))
    backedge.setValue(res);

  rewriter.eraseOp(endOp);
  return success();
}

namespace {
/// Converts Handshake memory interfaces into equivalent HW constructs,
/// potentially connecting them to the containing HW module's IO in the process.
/// The latter is enabled by a lowering state data-structure associated to the
/// matched memory interface during conversion of the containing Handshake
/// function, and which is expected to exist when this pattern is invoked.
class ConvertMemInterface
    : public OpInterfaceConversionPattern<handshake::MemoryOpInterface> {
public:
  ConvertMemInterface(ChannelTypeConverter &typeConverter, MLIRContext *ctx,
                      LoweringState &lowerState)
      : OpInterfaceConversionPattern<handshake::MemoryOpInterface>(
            typeConverter, ctx),
        lowerState(lowerState) {}

  LogicalResult
  matchAndRewrite(handshake::MemoryOpInterface memOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;

private:
  /// Shared lowering state.
  LoweringState &lowerState;
};
} // namespace

LogicalResult ConvertMemInterface::matchAndRewrite(
    handshake::MemoryOpInterface memOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {

  MLIRContext *ctx = memOp.getContext();
  hw::HWModuleOp parentModOp = memOp->getParentOfType<hw::HWModuleOp>();
  ModuleLoweringState &modState = lowerState.modState[parentModOp];
  MemLoweringState &memState = modState.memInterfaces[memOp];
  ModuleBuilder modBuilder(ctx);

  // Combine memory inputs from the function and internal memory inputs into
  // the new instance operands
  SmallVector<Value> instOperands;

  // Removes memory region name prefix from the port name.
  auto removePortNamePrefix = [&](const hw::ModulePort &port) -> StringRef {
    StringRef portName = port.name.strref();
    size_t idx = portName.rfind("_");
    if (idx != std::string::npos)
      return portName.substr(idx + 1);
    return portName;
  };

  // The HW instance will be connected to the top-level module through a number
  // of input ports, add those first
  ValueRange blockArgs = parentModOp.getBodyBlock()->getArguments();
  llvm::copy(blockArgs.slice(memState.inputIdx, memState.numInputs),
             std::back_inserter(instOperands));
  for (hw::ModulePort &inputPort : memState.getMemInputPorts(parentModOp))
    modBuilder.addInput(removePortNamePrefix(inputPort), inputPort.type);

  // Appends the converted operand at the specified index to the list of
  // operands of the future HW instance and adds a corresponding input to the
  // module builder.
  auto addInput = [&](size_t idx) -> void {
    Value oprd = operands[idx];
    instOperands.push_back(oprd);
    modBuilder.addInput(memState.operandNames[idx], oprd.getType());
  };

  // Construct the list of operands to the HW instance and the list of input
  // ports at the same time by iterating over the interface's ports
  for (GroupMemoryPorts &groupPorts : memState.ports.groups) {
    if (groupPorts.hasControl()) {
      ControlPort &ctrlPort = *groupPorts.ctrlPort;
      addInput(ctrlPort.getCtrlInputIndex());
    }

    for (auto [portIdx, port] : llvm::enumerate(groupPorts.accessPorts)) {
      if (std::optional<LoadPort> loadPort = dyn_cast<LoadPort>(port)) {
        addInput(loadPort->getAddrInputIndex());
      } else {
        std::optional<StorePort> storePort = dyn_cast<StorePort>(port);
        assert(storePort && "port must be load or store");
        addInput(storePort->getAddrInputIndex());
        addInput(storePort->getDataInputIndex());
      }
    }
  }

  // Finish by clock and reset ports
  modBuilder.addClkAndRstPorts();
  addClkAndRstOperands(instOperands, parentModOp);

  // Add output ports corresponding to memory interface results, then those
  // going outside the top-level HW module
  for (auto [idx, arg] : llvm::enumerate(memOp->getResults())) {
    std::string resName = memState.resultNames[idx];
    modBuilder.addOutput(resName, channelWrapper(arg.getType()));
  }
  for (const hw::ModulePort &outputPort :
       memState.getMemOutputPorts(parentModOp))
    modBuilder.addOutput(removePortNamePrefix(outputPort), outputPort.type);

  // Create HW instance to replace the memory interface with
  StringAttr name = rewriter.getStringAttr(lowerState.namer.getName(memOp));
  hw::HWModuleLike extModOp = modBuilder.getModule(memOp, rewriter);
  rewriter.setInsertionPoint(memOp);
  auto instanceOp = rewriter.create<hw::InstanceOp>(memOp.getLoc(), extModOp,
                                                    name, instOperands);
  size_t numResults = memOp->getNumResults();
  rewriter.replaceOp(memOp, instanceOp->getResults().take_front(numResults));

  // Resolve backedges in the module's terminator that are coming from the
  // memory interface
  ValueRange results = instanceOp->getResults().drop_front(numResults);
  for (auto [backedge, res] : llvm::zip_equal(memState.backedges, results))
    backedge.setValue(res);

  return success();
}

namespace {
/// Converts an operation (of type indicated by the template argument) into an
/// equivalent hardware instance. The method creates an external module to
/// instantiate the new component from if a module with matching IO does not
/// already exist. Valid/Ready semantics are made explicit thanks to the type
/// converter which converts implicit handshaked types into dataflow channels
/// with a corresponding data-type.
template <typename T>
class ExtModuleConversionPattern : public OpConversionPattern<T> {
public:
  ExtModuleConversionPattern(ChannelTypeConverter &typeConverter,
                             MLIRContext *ctx, LoweringState &lowerState)
      : OpConversionPattern<T>::OpConversionPattern(typeConverter, ctx),
        lowerState(lowerState) {}
  using OpAdaptor = typename T::Adaptor;

  /// Always succeeds in replacing the matched operation with an equivalent HW
  /// instance operation, potentially creating an external HW module in the
  /// process.
  LogicalResult
  matchAndRewrite(T op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  /// Shared lowering state.
  LoweringState &lowerState;

  /// Returns an exernal module operation that corresponds to the operation
  /// being converted and from which a HW instance can be created. If no
  /// external module matching the operation exists in the IR, it is created and
  /// then returned.
  hw::HWModuleExternOp getExtModule(T op,
                                    ConversionPatternRewriter &rewriter) const;
};
} // namespace

template <typename T>
LogicalResult ExtModuleConversionPattern<T>::matchAndRewrite(
    T op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  // Retrieve the external module operation that we are going to create an
  // instance from
  hw::HWModuleExternOp extModOp = getExtModule(op, rewriter);

  // If the module needs clock and reset inputs, retrieve them from the parent
  // HW module
  SmallVector<Value> operands = adaptor.getOperands();
  if (op.template hasTrait<mlir::OpTrait::HasClock>())
    addClkAndRstOperands(operands, cast<hw::HWModuleOp>(op->getParentOp()));

  // Replace operation with corresponding hardware module instance
  StringAttr instName = rewriter.getStringAttr(lowerState.namer.getName(op));
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<hw::InstanceOp>(op, extModOp, instName, operands);
  return success();
}

template <typename T>
hw::HWModuleExternOp ExtModuleConversionPattern<T>::getExtModule(
    T op, ConversionPatternRewriter &rewriter) const {
  // If an exernal module with the same name is already present in the top-level
  // MLIR module, then we do not need to recreate it.
  std::string extModName = getExtModuleName(op);
  if (hw::HWModuleExternOp modOp =
          lowerState.modOp.lookupSymbol<hw::HWModuleExternOp>(extModName))
    return modOp;

  // We need to instantiate a new external module for that operation; first
  // derive port information for that module, then create it
  ModuleBuilder modBuilder(op->getContext());
  HandshakePortNameGenerator portNames(op);

  // Add all operation operands to the inputs
  for (auto [idx, type] : llvm::enumerate(op->getOperandTypes()))
    modBuilder.addInput(portNames.inputName(idx).str(), channelWrapper(type));
  if (op.template hasTrait<mlir::OpTrait::HasClock>())
    modBuilder.addClkAndRstPorts();

  // Add all operation results to the inputs
  for (auto [idx, type] : llvm::enumerate(op->getResultTypes()))
    modBuilder.addOutput(portNames.outputName(idx).str(), channelWrapper(type));

  return modBuilder.getModule(op, rewriter);
}

/// Verifies that all the operations inside the function, which may be more
/// general than what we can turn into an RTL design, will be successfully
/// exportable to an RTL design. Fails if at least one operation inside the
/// function is not exportable to RTL.
static LogicalResult verifyExportToRTL(handshake::FuncOp funcOp) {
  if (failed(verifyIRMaterialized(funcOp)))
    return funcOp.emitError() << ERR_NON_MATERIALIZED_FUNC;
  if (failed(verifyAllIndexConcretized(funcOp))) {
    return funcOp.emitError() << "Lowering to HW requires that all index "
                                 "types in the IR have "
                                 "been concretized."
                              << ERR_RUN_CONCRETIZATION;
  }

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

/// Conversion pass driver. The conversion only works on modules containing a
/// single handshake function (handshake::FuncOp) at the moment. The function
/// and all the operations it contains are converted to operations from the HW
/// dialect. Dataflow semantics are made explicit with Handshake channels.
class HandshakeToHWPass
    : public dynamatic::impl::HandshakeToHWBase<HandshakeToHWPass> {
public:
  void runDynamaticPass() override {
    // At this level, all operations already have an intrinsic name so we can
    // disable our naming system
    doNotNameOperations();

    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();

    // We only support one function per module
    auto functions = modOp.getOps<handshake::FuncOp>();
    if (++functions.begin() != functions.end()) {
      modOp->emitOpError()
          << "we currently only support one handshake function per module";
      return signalPassFailure();
    }
    handshake::FuncOp funcOp = *functions.begin();

    // Check that some preconditions are met before doing anything
    if (failed(verifyExportToRTL(funcOp)))
      return signalPassFailure();

    // Helper struct for lowering
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(modOp.getBody(0));
    LoweringState lowerState(modOp, getAnalysis<NameAnalysis>(), builder);
    ChannelTypeConverter typeConverter;

    // Create pattern set
    RewritePatternSet patterns(ctx);
    patterns.insert<ConvertFunc, ConvertEnd, ConvertMemInterface,
                    // Handshake operations
                    ExtModuleConversionPattern<handshake::OEHBOp>,
                    ExtModuleConversionPattern<handshake::TEHBOp>,
                    ExtModuleConversionPattern<handshake::ConditionalBranchOp>,
                    ExtModuleConversionPattern<handshake::BranchOp>,
                    ExtModuleConversionPattern<handshake::MergeOp>,
                    ExtModuleConversionPattern<handshake::ControlMergeOp>,
                    ExtModuleConversionPattern<handshake::MuxOp>,
                    ExtModuleConversionPattern<handshake::SourceOp>,
                    ExtModuleConversionPattern<handshake::ConstantOp>,
                    ExtModuleConversionPattern<handshake::SinkOp>,
                    ExtModuleConversionPattern<handshake::ForkOp>,
                    ExtModuleConversionPattern<handshake::LazyForkOp>,
                    ExtModuleConversionPattern<handshake::ReturnOp>,
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
                    ExtModuleConversionPattern<arith::XOrIOp>>(
        typeConverter, funcOp->getContext(), lowerState);

    // Everything must be converted to operations in the hw dialect
    ConversionTarget target(*ctx);
    target.addLegalOp<hw::HWModuleOp, hw::HWModuleExternOp, hw::InstanceOp,
                      hw::OutputOp, mlir::UnrealizedConversionCastOp>();
    target.addIllegalDialect<handshake::HandshakeDialect, arith::ArithDialect,
                             memref::MemRefDialect>();

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<dynamatic::DynamaticPass> dynamatic::createHandshakeToHWPass() {
  return std::make_unique<HandshakeToHWPass>();
}
