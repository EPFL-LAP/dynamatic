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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <bitset>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <string>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

/// Converts all ExtraSignal types to signless integer.
static SmallVector<ExtraSignal>
lowerExtraSignals(ArrayRef<ExtraSignal> extraSignals) {
  SmallVector<ExtraSignal> newExtraSignals;
  for (const ExtraSignal &extra : extraSignals) {
    unsigned extraWidth = extra.type.getIntOrFloatBitWidth();

    // Convert to integer with the same bit width
    Type newType = IntegerType::get(extra.type.getContext(), extraWidth);

    newExtraSignals.emplace_back(extra.name, newType, extra.downstream);
  }
  return newExtraSignals;
}

/// Makes all (nested) types signless IntegerType's of the same width as the
/// original type. At the HW/RTL level we treat everything as opaque bitvectors,
/// so we no longer want to differentiate types of the same width w.r.t. their
/// intended interpretation.
static Type lowerType(Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<handshake::ChannelType>([&](handshake::ChannelType channelType) {
        // Make sure the data type is signless IntegerType
        unsigned width = channelType.getDataBitWidth();
        Type dataType = IntegerType::get(type.getContext(), width);

        // Convert all ExtraSignals to signless integer
        SmallVector<ExtraSignal> extraSignals =
            lowerExtraSignals(channelType.getExtraSignals());
        return handshake::ChannelType::get(dataType, extraSignals);
      })
      .Case<FloatType, IntegerType>([](auto type) {
        unsigned width = type.getIntOrFloatBitWidth();
        return IntegerType::get(type.getContext(), width);
      })
      .Case<handshake::ControlType>([](handshake::ControlType type) {
        // Convert all ExtraSignals to signless integer
        SmallVector<ExtraSignal> extraSignals =
            lowerExtraSignals(type.getExtraSignals());
        return handshake::ControlType::get(type.getContext(), extraSignals);
      })
      .Default([](auto type) { return nullptr; });
}

namespace {

/// Helper class to build HW modules progressively by adding inputs/outputs
/// one at time.
class ModuleBuilder {
public:
  /// The MLIR context is used to create string attributes for port names
  /// and types for the clock and reset ports, should they be added.
  ModuleBuilder(MLIRContext *ctx) : ctx(ctx){};

  /// Builds the module port information from the current list of inputs and
  /// outputs.
  hw::ModulePortInfo getPortInfo();

  /// Returns the current number of inputs.
  unsigned getNumInputs() { return inputs.size(); }

  /// Returns the current number of outputs.
  unsigned getNumOutputs() { return outputs.size(); }

  /// Adds an input port with the given name and type at the end of the
  /// existing list of input ports.
  void addInput(const Twine &name, Type type) {
    inputs.push_back(hw::ModulePort{StringAttr::get(ctx, name), type,
                                    hw::ModulePort::Direction::Input});
  }

  /// Adds an output port with the given name and type at the end of the
  /// existing list of output ports.
  void addOutput(const Twine &name, Type type) {
    outputs.push_back(hw::ModulePort{StringAttr::get(ctx, name), type,
                                     hw::ModulePort::Direction::Output});
  }

  /// Adds clock and reset ports at the end of the existing list of input
  /// ports.
  void addClkAndRst() {
    Type i1Type = IntegerType::get(ctx, 1);
    addInput(dynamatic::hw::CLK_PORT, i1Type);
    addInput(dynamatic::hw::RST_PORT, i1Type);
  }

  /// Returns the MLIR context used by the builder.
  MLIRContext *getContext() { return ctx; }

private:
  /// MLIR context for type/attribute creation.
  MLIRContext *ctx;
  /// List of module inputs.
  SmallVector<hw::ModulePort> inputs;
  /// List of module outputs.
  SmallVector<hw::ModulePort> outputs;
};

} // namespace

namespace {
/// Aggregates information to convert a Handshake memory interface into a
/// `hw::InstanceOp`. This must be created during conversion of the Handsahke
/// function containing the interface.
struct MemLoweringState {
  /// Memory region's name.
  std::string name;
  /// Data type.
  Type dataType = nullptr;
  /// Cache memory port information before modifying the interface, which can
  /// make them impossible to query.
  FuncMemoryPorts ports;
  /// Generates and stores the interface's port names before starting the
  /// conversion, when those are still queryable.
  handshake::PortNamer portNames;
  /// Backedges to the containing module's `hw::OutputOp` operation, which
  /// must be set, in order, with the memory interface's results that connect
  /// to the top-level module IO.
  SmallVector<Backedge> backedges;

  /// Index of first module input corresponding to the interface's inputs.
  size_t inputIdx = 0;
  /// Number of inputs for the memory interface, starting at the `inputIdx`.
  size_t numInputs = 0;
  /// Index of first module output corresponding to the interface's ouputs.
  size_t outputIdx = 0;
  /// Number of outputs for the memory interface, starting at the `outputIdx`.
  size_t numOutputs = 0;

  /// Needed because we use the class as a value type in a map, which needs to
  /// be default-constructible.
  MemLoweringState() : ports(nullptr), portNames(nullptr) {
    llvm_unreachable("object should never be default-constructed");
  }

  /// Constructs an instance of the object for the provided memory interface.
  MemLoweringState(handshake::MemoryOpInterface memOp, const Twine &name)
      : name(name.str()),
        dataType(lowerType(memOp.getMemRefType().getElementType())),
        ports(getMemoryPorts(memOp)), portNames(memOp) {
    assert(dataType && "unsupported memory element type");
  };

  /// Returns the module's input ports that connect to the memory interface.
  SmallVector<hw::ModulePort> getMemInputPorts(hw::HWModuleOp modOp);

  /// Returns the module's output ports that the memory interface connects to.
  SmallVector<hw::ModulePort> getMemOutputPorts(hw::HWModuleOp modOp);

  /// Determine whether the memory interface connects to top-level IO ports in
  /// the lowered hardware module.
  bool connectsToCircuit() const { return numInputs + numOutputs != 0; }

  /// Adds ports to the module for this memory interface.
  void connectWithCircuit(ModuleBuilder &modBuilder);
};

/// Summarizes information to convert a Handshake function into a
/// `hw::HWModuleOp`.
struct ModuleLoweringState {
  /// Maps each Handshake memory interface in the module with information on
  /// how to convert it into equivalent HW constructs.
  llvm::MapVector<handshake::MemoryOpInterface, MemLoweringState> memInterfaces;
  /// Number of distinct memories in the function's arguments.
  unsigned numMemories = 0;

  /// Default constructor required because we use the class as a map's value,
  /// which must be default constructible.
  ModuleLoweringState() = default;

  /// Constructs the lowering state from the Handshake function to lower.
  ModuleLoweringState(handshake::FuncOp funcOp) {
    numMemories = llvm::count_if(funcOp.getArgumentTypes(), [](Type ty) {
      return isa<mlir::MemRefType>(ty);
    });
  };

  /// Computes the total number of module outputs that are fed by memory
  /// interfaces within the module.
  size_t getNumMemOutputs() {
    size_t numOutputs = 0;
    for (auto &[memOp, info] : memInterfaces)
      numOutputs += info.numOutputs;
    return numOutputs;
  }
};

/// Shared state used during lowering. Captured in a struct to reduce the
/// number of arguments we have to pass around.
struct LoweringState {
  /// Top-level MLIR module.
  mlir::ModuleOp modOp;
  /// Reference to the pass's name analysis, to query unique names for each
  /// operation.
  NameAnalysis &namer;
  /// Allowa to create transient backedges for the `hw::OutputOp` of every
  /// create `hw::HWModuleOp`. We need backedges because, at the moment where
  /// a Handshake function is turned into a HW module, the operands to the
  /// `hw::OutputOp` terminator are not yet available but instead
  /// progressively appear as other oeprations inside the HW module are
  /// converted.
  BackedgeBuilder edgeBuilder;

  /// Maps each created `hw::HWModuleOp` during conversion to some lowering
  /// state required to convert the operations nested within it.
  DenseMap<hw::HWModuleOp, ModuleLoweringState> modState;

  /// Creates the lowering state. The builder is passed to a `BackedgeBuilder`
  /// used during the conversion to create transient operands.
  LoweringState(mlir::ModuleOp modOp, NameAnalysis &namer, OpBuilder &builder);
};
} // namespace

void MemLoweringState::connectWithCircuit(ModuleBuilder &modBuilder) {
  inputIdx = modBuilder.getNumInputs();
  outputIdx = modBuilder.getNumOutputs();

  MLIRContext *ctx = modBuilder.getContext();
  Type i1Type = IntegerType::get(ctx, 1);
  Type addrType = IntegerType::get(ctx, ports.addrWidth);

  // Load data input
  modBuilder.addInput(name + "_loadData", dataType);
  // Load enable output
  modBuilder.addOutput(name + "_loadEn", i1Type);
  // Load address output
  modBuilder.addOutput(name + "_loadAddr", addrType);
  // Store enable output
  modBuilder.addOutput(name + "_storeEn", i1Type);
  // Store address output
  modBuilder.addOutput(name + "_storeAddr", addrType);
  // Store data output
  modBuilder.addOutput(name + "_storeData", dataType);

  numInputs = modBuilder.getNumInputs() - inputIdx;
  numOutputs = modBuilder.getNumOutputs() - outputIdx;
};

SmallVector<hw::ModulePort>
MemLoweringState::getMemInputPorts(hw::HWModuleOp modOp) {
  if (numInputs == 0)
    return {};
  assert(inputIdx + numInputs <= modOp.getNumInputPorts() &&
         "input index too high");
  size_t idx = modOp.getPortIdForInputId(inputIdx);
  ArrayRef<hw::ModulePort> ports = modOp.getModuleType().getPorts();
  return SmallVector<hw::ModulePort>{ports.slice(idx, numInputs)};
}

SmallVector<hw::ModulePort>
MemLoweringState::getMemOutputPorts(hw::HWModuleOp modOp) {
  if (numOutputs == 0)
    return {};
  assert(outputIdx + numOutputs <= modOp.getNumOutputPorts() &&
         "output index too high");
  size_t idx = modOp.getPortIdForOutputId(outputIdx);
  ArrayRef<hw::ModulePort> ports = modOp.getModuleType().getPorts();
  return SmallVector<hw::ModulePort>{ports.slice(idx, numOutputs)};
}

LoweringState::LoweringState(mlir::ModuleOp modOp, NameAnalysis &namer,
                             OpBuilder &builder)
    : modOp(modOp), namer(namer), edgeBuilder(builder, modOp.getLoc()){};

/// Attempts to find an external HW module in the MLIR module with the
/// provided name. Returns it if it exists, otherwise returns `nullptr`.
static hw::HWModuleExternOp findExternMod(mlir::ModuleOp modOp,
                                          StringRef name) {
  if (hw::HWModuleExternOp mod = modOp.lookupSymbol<hw::HWModuleExternOp>(name))
    return mod;
  return nullptr;
}

namespace {
/// Extracts the parameters specific to each type of operation to identify our
/// exact module instantiation needs for RTL emission.
class ModuleDiscriminator {
public:
  /// Identifies the parameters associated to the operation depending on its
  /// type, after which, if the operation is supported, the unique module name
  /// and operation parameters can be queried.
  ModuleDiscriminator(Operation *op);

  /// Same role as the construction which takes an opaque operation but
  /// specialized for memory interfaces, passed through their port information.
  ModuleDiscriminator(FuncMemoryPorts &ports);

  /// Returns the unique external module name for the operation. Two operations
  /// with different parameter values will never receive the same name.
  std::string getDiscriminatedModName() {
    if (modName)
      return *modName;

    auto modOp = op->getParentOfType<mlir::ModuleOp>();
    StringRef opName = op->getName().getStringRef();

    // Try to find an external module with the same RTL name and parameters. If
    // we find one, then we can assign the same external module name to the
    // operation
    auto externalModules = modOp.getOps<hw::HWModuleExternOp>();
    auto extModOp = llvm::find_if(externalModules, [&](auto extModOp) {
      // 1. hw.name (e.g., handshake.fork) must match
      auto nameAttr =
          extModOp->template getAttrOfType<StringAttr>(RTL_NAME_ATTR_NAME);
      if (!nameAttr || nameAttr != opName)
        return false;

      // 2. hw.parameters (a dictionary containing DATA_TYPE, FIFO_DEPTH, etc.)
      // must match
      auto paramsAttr = extModOp->template getAttrOfType<DictionaryAttr>(
          RTL_PARAMETERS_ATTR_NAME);
      if (!paramsAttr)
        return false;

      if (paramsAttr.size() != parameters.size())
        return false;
      for (NamedAttribute param : parameters) {
        auto modParam = paramsAttr.getNamed(param.getName());
        if (!modParam || param.getValue() != modParam->getValue())
          return false;
      }

      // 3. The module's ports must match the operation's inputs and outputs
      // The module's port order is guaranteed to match the operation's inputs
      // and outputs (excluding clk and rst).
      // See ConvertToHWInstance<T>::matchAndRewrite or
      // ConvertMemInterface::matchAndRewrite.
      // Note: This equality check implies we can remove the DATA_TYPE parameter
      // from hw.parameters (checked above).
      unsigned int operandIdx = 0;
      unsigned int resultIdx = 0;
      auto modType = mlir::cast<hw::HWModuleExternOp>(extModOp).getModuleType();
      for (const hw::ModulePort &port : modType.getPorts()) {
        if (port.name == "clk" || port.name == "rst")
          continue;
        if (port.dir == hw::ModulePort::Direction::Input) {
          if (operandIdx >= op->getNumOperands()) {
            // The number of operands is different
            return false;
          }
          if (port.type != op->getOperand(operandIdx).getType()) {
            // The operand's type at operandIdx is different
            return false;
          }
          operandIdx++;
        } else if (port.dir == hw::ModulePort::Direction::Output) {
          if (resultIdx >= op->getNumResults()) {
            // The number of results is different
            return false;
          }
          if (port.type != op->getResult(resultIdx).getType()) {
            // The result's type at resultIdx is different
            return false;
          }
          resultIdx++;
        } else {
          // Inout ports are not used
          llvm_unreachable("Inout ports shouldn't be used");
          return false;
        }
      }

      return true;
    });
    if (extModOp != externalModules.end())
      return (*extModOp).getName().str();

    // Generate a unique name
    std::string name = getOpName() + "_";
    for (size_t i = 0;; ++i) {
      std::string candidateName = name + std::to_string(i);
      if (!modOp.lookupSymbol<hw::HWModuleExternOp>(candidateName))
        return candidateName;
    }
    llvm_unreachable("cannot generate unique name");
    return name;
  }

  /// Sets attribute on the external module (corresponding to the operation the
  /// object was constructed with) to tell the backend how to instantiate the
  /// component.
  void setParameters(hw::HWModuleExternOp modOp);

  /// Whether the operations is currently unsupported. Check after construction
  /// and produce a failure if this returns true.
  bool opUnsupported() { return unsupported; }

  /// Returns the operation the discriminator was created from.
  Operation *getOperation() const { return op; }

private:
  /// The operation whose parameters are being identified.
  Operation *op;
  /// MLIR context to create attributes with.
  MLIRContext *ctx;
  /// The module name may be set explicitly for some operation types or
  /// derived/uniqued automatically based on the RTL parameters.
  std::optional<std::string> modName;
  /// The operation's parameters, as a list of named attributes.
  SmallVector<NamedAttribute> parameters;

  /// Whether the operation is unsupported (set during construction).
  bool unsupported = false;

  /// Adds a parameter.
  void addParam(const Twine &name, Attribute attr) {
    parameters.emplace_back(StringAttr::get(ctx, name), attr);
  }

  /// Adds a boolean-type parameter.
  void addBoolean(const Twine &name, bool value) {
    addParam(name, BoolAttr::get(ctx, value));
  };

  /// Adds a scalar-type parameter.
  void addUnsigned(const Twine &name, unsigned scalar) {
    Type intType = IntegerType::get(ctx, 32, IntegerType::Unsigned);
    addParam(name, IntegerAttr::get(intType, scalar));
  };

  /// Adds a dataflow-type parameter.
  void addType(const Twine &name, Type type) {
    assert((isa<handshake::ControlType>(type) ||
            isa<handshake::ChannelType>(type)) &&
           "incompatible type");
    addParam(name, TypeAttr::get(type));
  };

  /// Adds the value's type as a dataflow-type parameter.
  void addType(const Twine &name, Value val) { addType(name, val.getType()); };

  /// Adds a string parameter.
  void addString(const Twine &name, const Twine &txt) {
    addParam(name, StringAttr::get(ctx, txt));
  };

  std::string getOpName() const {
    std::string opName = op->getName().getStringRef().str();
    std::replace(opName.begin(), opName.end(), '.', '_');
    return opName;
  }

  /// Initializes private fields from the input operation.
  void init(Operation *op) {
    this->op = op;
    ctx = op->getContext();
    auto paramsAttr =
        op->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
    if (!paramsAttr)
      return;
    llvm::copy(paramsAttr.getValue(), std::back_inserter(parameters));
  }
};
} // namespace

ModuleDiscriminator::ModuleDiscriminator(Operation *op) {
  init(op);

  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::InstanceOp>(
          [&](handshake::InstanceOp instOp) { modName = instOp.getModule(); })
      .Case<handshake::ForkOp, handshake::LazyForkOp>([&](auto) {
        // Number of output channels and bitwidth
        addUnsigned("SIZE", op->getNumResults());
        addType("DATA_TYPE", op->getOperand(0));
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        // Number of input data channels, data bitwidth, and select bitwidth
        addUnsigned("SIZE", muxOp.getDataOperands().size());
        addType("DATA_TYPE", muxOp.getResult());
        addType("SELECT_TYPE", muxOp.getSelectOperand());
      })
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp cmergeOp) {
        // Number of input data channels, data bitwidth, and index
        // bitwidth
        addUnsigned("SIZE", cmergeOp.getDataOperands().size());
        addType("DATA_TYPE", cmergeOp.getResult());
        addType("INDEX_TYPE", cmergeOp.getIndex());
      })
      .Case<handshake::MergeOp>([&](auto) {
        // Number of input data channels and data bitwidth
        addUnsigned("SIZE", op->getNumOperands());
        addType("DATA_TYPE", op->getResult(0));
      })
      .Case<handshake::JoinOp, handshake::BlockerOp>([&](auto) {
        // Number of input channels
        addUnsigned("SIZE", op->getNumOperands());
      })
      .Case<handshake::BranchOp, handshake::SinkOp, handshake::BufferOp,
            handshake::NDWireOp>([&](auto) {
        // Bitwidth
        addType("DATA_TYPE", op->getOperand(0));
      })
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp cbrOp) {
            // Bitwidth
            addType("DATA_TYPE", cbrOp.getDataOperand());
          })
      .Case<handshake::SourceOp>([&](auto) {
        // No discrimianting parameters, just to avoid falling into the
        // default case for sources
      })
      .Case<handshake::MemPortOpInterface>(
          [&](handshake::MemPortOpInterface portOp) {
            // Data bitwidth and address bitwidth
            addType("DATA_TYPE", portOp.getDataInput());
            addType("ADDR_TYPE", portOp.getAddressInput());
          })
      .Case<handshake::SharingWrapperOp>(
          [&](handshake::SharingWrapperOp sharingWrapperOp) {
            addType("DATA_WIDTH", sharingWrapperOp.getDataOperands()[0]);

            // In a sharing wrapper, we have the credits as a list of unsigned
            // integers. This will be encoded as a space-separated string and
            // passed to the sharing wrapper generator.

            auto addSpaceSeparatedListOfInt =
                [&](StringRef name, ArrayRef<int64_t> array) -> void {
              std::string strAttr;
              for (unsigned i = 0; i < array.size(); i++) {
                if (i > 0)
                  strAttr += " ";
                strAttr += std::to_string(array[i]);
              }
              addString(name, strAttr);
            };

            addSpaceSeparatedListOfInt("CREDITS",
                                       sharingWrapperOp.getCredits());

            addUnsigned("NUM_SHARED_OPERANDS",
                        sharingWrapperOp.getNumSharedOperands());

            addUnsigned("LATENCY", sharingWrapperOp.getLatency());
          })
      .Case<handshake::ConstantOp>([&](handshake::ConstantOp cstOp) {
        // Bitwidth and binary-encoded constant value
        ChannelType cstType = cstOp.getResult().getType();
        unsigned bitwidth = cstType.getDataBitWidth();
        if (bitwidth > 64) {
          cstOp.emitError() << "Constant value has bitwidth " << bitwidth
                            << ", but we only support up to 64.";
          unsupported = true;
          return;
        }

        // Determine the constant value based on the constant's return type
        // and convert it to a binary string value
        TypedAttr valueAttr = cstOp.getValueAttr();
        std::string bitValue;
        if (auto intType = dyn_cast<IntegerType>(cstType.getDataType())) {
          APInt value = cast<mlir::IntegerAttr>(valueAttr).getValue();

          // Bitset requires a compile-time constant, just use 64 and
          // manually truncate the value after so that it is the exact
          // bitwidth we need
          if (intType.isUnsignedInteger())
            bitValue = std::bitset<64>(value.getZExtValue()).to_string();
          else
            bitValue = std::bitset<64>(value.getSExtValue()).to_string();
          bitValue = bitValue.substr(64 - bitwidth);
        } else if (isa<FloatType>(cstType.getDataType())) {
          mlir::FloatAttr attr = cast<mlir::FloatAttr>(valueAttr);
          APInt floatInt = attr.getValue().bitcastToAPInt();

          // We only support specific bitwidths for floating point numbers
          bitValue = std::bitset<64>(floatInt.getZExtValue()).to_string();
          if (floatInt.getBitWidth() == 32) {
            bitValue = bitValue.substr(32);
          } else if (floatInt.getBitWidth() != 64) {
            cstOp.emitError() << "Constant has unsupported floating point "
                                 "bitwidth. Expected 32 or 64 but got "
                              << bitwidth << ".";
            unsupported = true;
            return;
          }
        } else {
          cstOp->emitError()
              << "Constant type must be integer or floating point.";
          unsupported = true;
          return;
        }

        addString("VALUE", bitValue);
        addUnsigned("DATA_WIDTH", bitwidth);
      })
      .Case<handshake::AddFOp, handshake::AddIOp, handshake::AndIOp,
            handshake::DivFOp, handshake::DivSIOp, handshake::DivUIOp,
            handshake::MaximumFOp, handshake::MinimumFOp, handshake::MulFOp,
            handshake::MulIOp, handshake::NegFOp, handshake::NotOp,
            handshake::OrIOp, handshake::ShLIOp, handshake::ShRSIOp,
            handshake::ShRUIOp, handshake::SubFOp, handshake::SubIOp,
            handshake::XOrIOp, handshake::SIToFPOp, handshake::FPToSIOp,
            handshake::AbsFOp>([&](auto) {
        // Bitwidth
        addType("DATA_TYPE", op->getOperand(0));
      })
      .Case<handshake::SelectOp>([&](handshake::SelectOp selectOp) {
        // Data bitwidth
        addType("DATA_TYPE", selectOp.getTrueValue());
      })
      .Case<handshake::CmpFOp>([&](handshake::CmpFOp cmpFOp) {
        // Predicate and bitwidth
        addString("PREDICATE", stringifyEnum(cmpFOp.getPredicate()));
        addType("DATA_TYPE", cmpFOp.getLhs());
      })
      .Case<handshake::CmpIOp>([&](handshake::CmpIOp cmpIOp) {
        // Predicate and bitwidth
        addString("PREDICATE", stringifyEnum(cmpIOp.getPredicate()));
        addType("DATA_TYPE", cmpIOp.getLhs());
      })
      .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::TruncIOp,
            handshake::ExtFOp, handshake::TruncFOp>([&](auto) {
        // Input bitwidth and output bitwidth
        addType("INPUT_TYPE", op->getOperand(0));
        addType("OUTPUT_TYPE", op->getResult(0));
      })
      .Case<handshake::SpeculatorOp>([&](handshake::SpeculatorOp speculatorOp) {
        addUnsigned("FIFO_DEPTH", speculatorOp.getFifoDepth());
      })
      .Case<handshake::SpecSaveOp, handshake::SpecCommitOp,
            handshake::SpeculatingBranchOp, handshake::NonSpecOp>([&](auto) {
        // No parameters needed for these operations
      })
      .Case<handshake::SpecSaveCommitOp>(
          [&](handshake::SpecSaveCommitOp saveCommitOp) {
            addUnsigned("FIFO_DEPTH", saveCommitOp.getFifoDepth());
          })
      .Default([&](auto) {
        op->emitError() << "This operation cannot be lowered to RTL "
                           "due to a lack of an RTL implementation for it.";
        unsupported = true;
      });
}

ModuleDiscriminator::ModuleDiscriminator(FuncMemoryPorts &ports) {
  init(ports.memOp);

  MLIRContext *ctx = op->getContext();
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::MemoryControllerOp>([&](auto) {
        // There can be at most one of those, and it is a load/store port
        unsigned lsqPort = ports.getNumPorts<LSQLoadStorePort>();
        Type dataType = IntegerType::get(ctx, ports.dataWidth);
        Type addrType = IntegerType::get(ctx, ports.addrWidth);

        // Control port count, load port count, store port count, data
        // bitwidth, and address bitwidth
        addUnsigned("NUM_CONTROLS", ports.getNumPorts<ControlPort>());
        addUnsigned("NUM_LOADS", ports.getNumPorts<LoadPort>() + lsqPort);
        addUnsigned("NUM_STORES", ports.getNumPorts<StorePort>() + lsqPort);
        addType("DATA_TYPE", ChannelType::get(dataType));
        addType("ADDR_TYPE", ChannelType::get(addrType));
      })
      .Case<handshake::LSQOp>([&](auto) {
        LSQGenerationInfo genInfo(ports, getUniqueName(op).str());
        modName = getOpName() + "_" + genInfo.name;

        /// Converts an array into an equivalent MLIR attribute.
        Type intType = IntegerType::get(ctx, 32);
        auto addArrayIntAttr = [&](StringRef name,
                                   ArrayRef<unsigned> array) -> void {
          SmallVector<Attribute> arrayAttr;
          llvm::transform(
              array, std::back_inserter(arrayAttr),
              [&](unsigned elem) { return IntegerAttr::get(intType, elem); });
          addParam(name, ArrayAttr::get(ctx, arrayAttr));
        };

        /// Converts a bi-dimensional array into an equivalent MLIR attribute.
        auto addBiArrayIntAttr =
            [&](StringRef name,
                ArrayRef<SmallVector<unsigned>> biArray) -> void {
          SmallVector<Attribute> biArrayAttr;
          for (ArrayRef<unsigned> array : biArray) {
            SmallVector<Attribute> arrayAttr;
            llvm::transform(
                array, std::back_inserter(arrayAttr),
                [&](unsigned elem) { return IntegerAttr::get(intType, elem); });
            biArrayAttr.push_back(ArrayAttr::get(ctx, arrayAttr));
          }
          addParam(name, ArrayAttr::get(ctx, biArrayAttr));
        };

        addString("name", *modName);
        addBoolean("master", ports.interfacePorts.empty());
        addUnsigned("fifoDepth", genInfo.depth);
        addUnsigned("fifoDepth_L", genInfo.depthLoad);
        addUnsigned("fifoDepth_S", genInfo.depthStore);
        addUnsigned("bufferDepth", genInfo.bufferDepth);
        addUnsigned("dataWidth", genInfo.dataWidth);
        addUnsigned("addrWidth", genInfo.addrWidth);
        addUnsigned("numBBs", genInfo.numGroups);
        addUnsigned("numLoadPorts", genInfo.numLoads);
        addUnsigned("numStorePorts", genInfo.numStores);
        addArrayIntAttr("numLoads", genInfo.loadsPerGroup);
        addArrayIntAttr("numStores", genInfo.storesPerGroup);
        addBiArrayIntAttr("loadOffsets", genInfo.loadOffsets);
        addBiArrayIntAttr("storeOffsets", genInfo.storeOffsets);
        addBiArrayIntAttr("loadPorts", genInfo.loadPorts);
        addBiArrayIntAttr("storePorts", genInfo.storePorts);
        /// Add the attributes needed by the new lsq config file
        addBiArrayIntAttr("ldOrder", genInfo.ldOrder);
        addBiArrayIntAttr("ldPortIdx", genInfo.ldPortIdx);
        addBiArrayIntAttr("stPortIdx", genInfo.stPortIdx);
        addUnsigned("indexWidth", genInfo.indexWidth);
        addUnsigned("numLdChannels", genInfo.numLdChannels);
        addUnsigned("numStChannels", genInfo.numStChannels);
        addUnsigned("stResp", genInfo.stResp);
        addUnsigned("groupMulti", genInfo.groupMulti);
        addUnsigned("pipe0En", genInfo.pipe0En);
        addUnsigned("pipe1En", genInfo.pipe1En);
        addUnsigned("pipeCompEn", genInfo.pipeCompEn);
        addUnsigned("headLagEn", genInfo.headLagEn);
      })
      .Default([&](auto) {
        op->emitError() << "Unsupported memory interface type.";
        unsupported = true;
      });
}

void ModuleDiscriminator::setParameters(hw::HWModuleExternOp modOp) {
  assert(!unsupported && "operation unsupported");

  // The name is used to determine which RTL component to instantiate
  StringRef opName = op->getName().getStringRef();
  modOp->setAttr(RTL_NAME_ATTR_NAME, StringAttr::get(ctx, opName));

  // Parameters are used to determine the concrete version of the RTL
  // component to instantiate
  modOp->setAttr(RTL_PARAMETERS_ATTR_NAME,
                 DictionaryAttr::get(ctx, parameters));
}

namespace {

/// Builder for hardware instances (`hw::InstanceOp`) and their associated
/// external hardware module. The class's methods allows to create the IO of an
/// hardware instance port by port while also creating matching port information
/// for the external hardware module that the instance will be of. When all the
/// instance/module IO has been added, `HWBuilder::createInstance` will create
/// the hardware instance in the IR and, if need be, a matching external module.
class HWBuilder {
public:
  /// Creates the hardware builder.
  HWBuilder(MLIRContext *ctx) : modBuilder(ctx){};

  /// Adds a value to the list of operands for the future instance, and its type
  /// to the future external module's input port information.
  void addInput(const Twine &name, Value oprd) {
    modBuilder.addInput(name, oprd.getType());
    instOperands.push_back(oprd);
  }

  /// Adds a type to the future external module's output port information.
  void addOutput(const Twine &name, Type type) {
    modBuilder.addOutput(name, type);
  }

  /// Adds clock and reset ports from the parent module to the future external
  /// module's input port information and to the operands to the future
  /// hardware instance.
  void addClkAndRst(hw::HWModuleOp modOp);

  /// Creates the instance using all the inputs added so far as operands. If no
  /// external module matching the current port information currently exists,
  /// one is added at the bottom of the top-level MLIR module. Returns the
  /// instance on success, nullptr on failure.
  virtual hw::InstanceOp createInstance(ModuleDiscriminator &discriminator,
                                        const Twine &instName, Location loc,
                                        OpBuilder &builder);

  /// Virtual destructor because of virtual method.
  virtual ~HWBuilder() = default;

protected:
  /// Module builder to optionally create an external module.
  ModuleBuilder modBuilder;
  /// The list of operands that will be used to create the instance
  /// corresponding to the converted operation.
  SmallVector<Value> instOperands;
};

/// Specialization of the hardware builder offering an extra public method to
/// easily replace an existing operation by the created hardware instance.
class HWConverter : public HWBuilder {
public:
  using HWBuilder::HWBuilder;

  /// Replaces the operation with an equivalent instance using all the inputs
  /// added so far as operands. If no external module matching the current port
  /// information currently exists, one is added at the bottom of the top-level
  /// MLIR module. Returns the instance on success, nullptr on failure.
  hw::InstanceOp convertToInstance(Operation *opToConvert,
                                   ConversionPatternRewriter &rewriter) {
    if (modBuilder.getNumOutputs() != opToConvert->getNumResults()) {
      opToConvert->emitError()
          << "Attempting to replace operation with "
          << opToConvert->getNumResults()
          << " results; however, external module will have "
          << modBuilder.getNumOutputs() << " output ports, a mismatch.";
      return nullptr;
    }

    hw::InstanceOp instOp = createInstanceFromOp(opToConvert, rewriter);
    if (!instOp)
      return nullptr;
    rewriter.replaceOp(opToConvert, instOp);
    return instOp;
  }

protected:
  /// Creates the hardware instance using a discrimator, name, and location
  /// derived from an existing operation. Returns the instance on success,
  /// nullptr on failure.
  hw::InstanceOp createInstanceFromOp(Operation *op,
                                      ConversionPatternRewriter &rewriter) {
    ModuleDiscriminator discriminator(op);
    StringRef name = getUniqueName(op);
    Location loc = op->getLoc();
    return createInstance(discriminator, name, loc, rewriter);
  }
};

/// Specialization of the hardware builder for operations implementing the
/// `handshake::MemoryOpInterface` interface.
class HWMemConverter : public HWConverter {
public:
  using HWConverter::HWConverter;

  /// Replaces the memory interface with an equivalent instance using all the
  /// inputs added so far as operands. If no external module matching the
  /// current port information currently exists, one is added at the bottom of
  /// the top-level MLIR module. Returns the instance on success, nullptr on
  /// failure.
  hw::InstanceOp convertToInstance(MemLoweringState &state,
                                   ConversionPatternRewriter &rewriter) {
    handshake::MemoryOpInterface memOp = state.ports.memOp;
    ModuleDiscriminator discriminator(state.ports);
    StringRef name = getUniqueName(memOp);
    Location loc = memOp.getLoc();
    hw::InstanceOp instOp = createInstance(discriminator, name, loc, rewriter);
    if (!instOp)
      return nullptr;

    size_t numResults = memOp->getNumResults();
    rewriter.replaceOp(memOp, instOp->getResults().take_front(numResults));

    // Resolve backedges in the module's terminator that are coming from the
    // memory interface
    ValueRange toModOutput = instOp->getResults().drop_front(numResults);
    for (auto [backedge, res] : llvm::zip_equal(state.backedges, toModOutput))
      backedge.setValue(res);
    return instOp;
  }
};
} // namespace

/// Returns the clock and reset module inputs (which are assumed to be the last
/// two module inputs).
static std::pair<Value, Value> getClkAndRst(hw::HWModuleOp hwModOp) {
  // Check that the parent module's last port are the clock and reset
  // signals we need for the instance operands
  unsigned numInputs = hwModOp.getNumInputPorts();
  assert(numInputs >= 2 && "module should have at least clock and reset");
  size_t lastIdx = hwModOp.getPortIdForInputId(numInputs - 1);
  assert(hwModOp.getPort(lastIdx - 1).getName() == dynamatic::hw::CLK_PORT &&
         "expected clock");
  assert(hwModOp.getPort(lastIdx).getName() == dynamatic::hw::RST_PORT &&
         "expected reset");

  // Add clock and reset to the instance's operands
  ValueRange blockArgs = hwModOp.getBodyBlock()->getArguments();
  return {blockArgs.drop_back().back(), blockArgs.back()};
}

void HWBuilder::addClkAndRst(hw::HWModuleOp hwModOp) {
  // Let the parent class add clock and reset to the input ports
  modBuilder.addClkAndRst();

  // Add clock and reset to the instance's operands
  auto [clkVal, rstVal] = getClkAndRst(hwModOp);
  instOperands.push_back(clkVal);
  instOperands.push_back(rstVal);
}

hw::InstanceOp HWBuilder::createInstance(ModuleDiscriminator &discriminator,
                                         const Twine &instName, Location loc,
                                         OpBuilder &builder) {
  // Fail when the discriminator reports that the operation is unsupported
  if (discriminator.opUnsupported())
    return nullptr;

  // First retrieve or create the external module matching the operation
  mlir::ModuleOp topLevelModOp =
      discriminator.getOperation()->getParentOfType<mlir::ModuleOp>();
  std::string extModName = discriminator.getDiscriminatedModName();
  hw::HWModuleExternOp extModOp = findExternMod(topLevelModOp, extModName);

  if (!extModOp) {
    // The external module does not yet exist, create it
    StringAttr modNameAttr = builder.getStringAttr(extModName);
    RewriterBase::InsertPoint instInsertPoint = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(topLevelModOp.getBody());
    extModOp = builder.create<hw::HWModuleExternOp>(loc, modNameAttr,
                                                    modBuilder.getPortInfo());
    discriminator.setParameters(extModOp);
    builder.restoreInsertionPoint(instInsertPoint);
  }

  // Now create the instance corresponding to the external module
  StringAttr instNameAttr = builder.getStringAttr(instName);
  return builder.create<hw::InstanceOp>(loc, extModOp, instNameAttr,
                                        instOperands);
}

hw::ModulePortInfo ModuleBuilder::getPortInfo() {
  SmallVector<hw::PortInfo> inputPorts;
  SmallVector<hw::PortInfo> outputPorts;
  for (auto [idx, modPort] : llvm::enumerate(inputs))
    inputPorts.push_back(hw::PortInfo{modPort, idx});
  for (auto [idx, modPort] : llvm::enumerate(outputs))
    outputPorts.push_back(hw::PortInfo{modPort, idx});
  return hw::ModulePortInfo(inputPorts, outputPorts);
}

/// Adds IO to the module builder for the provided memref, using the provided
/// name to unique IO port names. All Handshake memory interfaces referencing
/// the memref inside the function are added to the module lowering state
/// memory interface map, along with helper lowering state.
static void addMemIO(ModuleBuilder &modBuilder, handshake::FuncOp funcOp,
                     TypedValue<MemRefType> memref, StringRef memName,
                     ModuleLoweringState &state) {
  for (auto memOp : funcOp.getOps<handshake::MemoryOpInterface>()) {
    // The interface must reference this memory region
    if (memOp.getMemRef() != memref)
      continue;

    MemLoweringState info = MemLoweringState(memOp, memName);
    if (memOp.isMasterInterface())
      info.connectWithCircuit(modBuilder);
    state.memInterfaces.insert({memOp, info});
  }
}

/// Produces the port information for the HW module that will replace the
/// Handshake function. Fills in the lowering state object with information
/// that will allow the conversion pass to connect memory interface to their
/// top-level IO later on.
hw::ModulePortInfo getFuncPortInfo(handshake::FuncOp funcOp,
                                   ModuleLoweringState &state) {
  ModuleBuilder modBuilder(funcOp.getContext());
  handshake::PortNamer portNames(funcOp);

  // Add all function outputs to the module
  for (auto [idx, res] : llvm::enumerate(funcOp.getResultTypes()))
    modBuilder.addOutput(portNames.getOutputName(idx), lowerType(res));

  // Add all function inputs to the module, expanding memory references into a
  // set of individual ports for loads and stores
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    StringAttr argName = funcOp.getArgName(idx);
    Type type = arg.getType();
    if (TypedValue<MemRefType> memref = dyn_cast<TypedValue<MemRefType>>(arg))
      addMemIO(modBuilder, funcOp, memref, argName, state);
    else
      modBuilder.addInput(portNames.getInputName(idx), lowerType(type));
  }

  modBuilder.addClkAndRst();
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
      return lowerType(type);
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

/// Converts a non-external Handshake function into a `hw::HWModuleOp` with IO
/// corresponding to the original Handshake function. The pattern also creates a
/// lowering state object associated to the created HW module to control the
/// conversion of other operations within the module.
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

/// Converts an external Handshake function into a `hw::HWModuleExternOp` with
/// IO corresponding to the original Handshake function.
class ConvertExternalFunc : public OpConversionPattern<handshake::FuncOp> {
public:
  using OpConversionPattern<handshake::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(handshake::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
ConvertFunc::matchAndRewrite(handshake::FuncOp funcOp, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
  if (funcOp.isExternal())
    return failure();

  StringAttr name = rewriter.getStringAttr(funcOp.getName());
  ModuleLoweringState state(funcOp);
  hw::ModulePortInfo modInfo = getFuncPortInfo(funcOp, state);

  // Create non-external HW module to replace the function with
  rewriter.setInsertionPoint(funcOp);
  auto modOp = rewriter.create<hw::HWModuleOp>(funcOp.getLoc(), name, modInfo);

  // Move the block from the Handshake function to the new HW module, after
  // which the Handshake function becomes empty and can be deleted
  Block *funcBlock = funcOp.getBodyBlock();
  Block *modBlock = modOp.getBodyBlock();
  Operation *termOp = modBlock->getTerminator();
  ValueRange modBlockArgs = modBlock->getArguments().drop_back(2);
  rewriter.inlineBlockBefore(funcBlock, termOp, modBlockArgs);
  rewriter.eraseOp(funcOp);

  // Operands for the module's terminators; they are the module's outputs
  SmallVector<Value> outOperands;

  // First outputs operands are identical to the end's, modulo type conversion
  auto endOp = *modOp.getBodyBlock()->getOps<handshake::EndOp>().begin();
  if (failed((rewriter.getRemappedValues(endOp.getOperands(), outOperands)))) {
    return funcOp->emitError()
           << "failed to remap function terminator's operands";
  }

  // Remaining output operands will eventually come from master memory
  // interfaces' outputs; create backedges and resolve them during memory
  // lowering
  auto moduleOutputs = modInfo.getOutputs().begin();
  for (size_t i = 0, e = endOp->getNumOperands(); i < e; ++i, ++moduleOutputs)
    ;
  for (auto &[_, memState] : state.memInterfaces) {
    for (size_t i = 0; i < memState.numOutputs; ++i) {
      const hw::PortInfo &port = *(moduleOutputs++);
      Backedge backedge = lowerState.edgeBuilder.get(port.type);
      outOperands.push_back(backedge);
      memState.backedges.push_back(backedge);
    }
  }

  // Replace the default terminator with one with our operands, and delete the
  // Handshake function's terminator
  Operation *outputOp = modOp.getBodyBlock()->getTerminator();
  rewriter.setInsertionPoint(outputOp);
  rewriter.replaceOpWithNewOp<hw::OutputOp>(outputOp, outOperands);
  rewriter.eraseOp(endOp);

  // Associate the newly created module to its lowering state object
  lowerState.modState.insert({modOp, state});
  return success();
}

LogicalResult ConvertExternalFunc::matchAndRewrite(
    handshake::FuncOp funcOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!funcOp.isExternal())
    return failure();

  StringAttr name = rewriter.getStringAttr(funcOp.getName());
  ModuleBuilder modBuilder(funcOp.getContext());
  handshake::PortNamer portNames(funcOp);

  // Add all function outputs to the module
  for (auto [idx, res] : llvm::enumerate(funcOp.getResultTypes()))
    modBuilder.addOutput(portNames.getOutputName(idx), lowerType(res));

  // Add all function inputs to the module
  for (auto [idx, type] : llvm::enumerate(funcOp.getArgumentTypes())) {
    if (isa<MemRefType>(type)) {
      return funcOp->emitError()
             << "Memory interfaces are not supported for external "
                "functions";
    }
    modBuilder.addInput(portNames.getInputName(idx), lowerType(type));
  }
  modBuilder.addClkAndRst();

  rewriter.setInsertionPoint(funcOp);
  auto modOp = rewriter.replaceOpWithNewOp<hw::HWModuleExternOp>(
      funcOp, name, modBuilder.getPortInfo());
  modOp->setAttr(StringAttr::get(getContext(), RTL_NAME_ATTR_NAME),
                 funcOp.getNameAttr());
  return success();
}

namespace {
/// Converts Handshake memory interfaces into equivalent HW constructs,
/// potentially connecting them to the containing HW module's IO in the
/// process. The latter is enabled by a lowering state data-structure
/// associated to the matched memory interface during conversion of the
/// containing Handshake function, and which is expected to exist when this
/// pattern is invoked.
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
  hw::HWModuleOp parentModOp = memOp->getParentOfType<hw::HWModuleOp>();
  ModuleLoweringState &modState = lowerState.modState[parentModOp];
  MemLoweringState &memState = modState.memInterfaces[memOp];
  HWMemConverter converter(getContext());

  // Removes memory region name prefix from the port name.
  auto removePortNamePrefix = [&](const hw::ModulePort &port) -> StringRef {
    StringRef portName = port.name.strref();
    size_t idx = portName.rfind("_");
    if (idx != std::string::npos)
      return portName.substr(idx + 1);
    return portName;
  };

  // The HW instance will be connected to the top-level module through a
  // number of input ports, add those first, then the regular interface ports,
  // and finally clock and reset
  ValueRange blockArgs = parentModOp.getBodyBlock()->getArguments();
  ValueRange memArgs = blockArgs.slice(memState.inputIdx, memState.numInputs);
  auto inputModPorts = memState.getMemInputPorts(parentModOp);
  for (auto [port, arg] : llvm::zip_equal(inputModPorts, memArgs))
    converter.addInput(removePortNamePrefix(port), arg);
  for (auto [idx, oprd] : llvm::enumerate(operands)) {
    if (!isa<mlir::MemRefType>(oprd.getType()))
      converter.addInput(memState.portNames.getInputName(idx), oprd);
  }
  converter.addClkAndRst(parentModOp);

  // The HW instance will be connected to the top-level module through a
  // number of output ports, add those last after the regular interface ports
  for (auto [idx, res] : llvm::enumerate(memOp->getResults())) {
    converter.addOutput(memState.portNames.getOutputName(idx),
                        lowerType(res.getType()));
  }
  auto outputModPorts = memState.getMemOutputPorts(parentModOp);
  for (const hw::ModulePort &outputPort : outputModPorts)
    converter.addOutput(removePortNamePrefix(outputPort), outputPort.type);

  hw::InstanceOp instOp = converter.convertToInstance(memState, rewriter);
  return instOp ? success() : failure();
}

namespace {
/// Converts an operation (of type indicated by the template argument) into
/// an equivalent hardware instance. The method creates an external module
/// to instantiate the new component from if a module with matching IO does
/// not already exist. Valid/Ready semantics are made explicit thanks to the
/// type converter which converts implicit handshaked types into dataflow
/// channels with a corresponding data-type.
template <typename T>
class ConvertToHWInstance : public OpConversionPattern<T> {
public:
  using OpConversionPattern<T>::OpConversionPattern;
  using OpAdaptor = typename T::Adaptor;

  /// Always succeeds in replacing the matched operation with an equivalent
  /// HW instance operation, potentially creating an external HW module in
  /// the process.
  LogicalResult
  matchAndRewrite(T op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

template <typename T>
LogicalResult ConvertToHWInstance<T>::matchAndRewrite(
    T op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  HWConverter converter(this->getContext());
  handshake::PortNamer portNames(op);

  // Add all operation operands to the inputs
  for (auto [idx, oprd] : llvm::enumerate(adaptor.getOperands()))
    converter.addInput(portNames.getInputName(idx), oprd);
  converter.addClkAndRst(((Operation *)op)->getParentOfType<hw::HWModuleOp>());

  // Add all operation results to the outputs
  for (auto [idx, type] : llvm::enumerate(op->getResultTypes()))
    converter.addOutput(portNames.getOutputName(idx), lowerType(type));

  hw::InstanceOp instOp = converter.convertToInstance(op, rewriter);
  return instOp ? success() : failure();
}

namespace {

/// Converts a Handshake-level instance operation to an equivalent HW-level one.
/// The pattern assumes that the module the Handshake instance references has
/// already been converted to a `hw::HWExternModuleOp`.
class ConvertInstance : public OpConversionPattern<handshake::InstanceOp> {
public:
  using OpConversionPattern<handshake::InstanceOp>::OpConversionPattern;
  using OpAdaptor = typename handshake::InstanceOp::Adaptor;

  /// Always succeeds in replacing the matched operation with an equivalent
  /// HW instance operation.
  LogicalResult
  matchAndRewrite(handshake::InstanceOp instOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
ConvertInstance::matchAndRewrite(handshake::InstanceOp instOp,
                                 OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  SmallVector<Value> instOperands(adaptor.getOperands());
  auto [clk, rst] = getClkAndRst(instOp->getParentOfType<hw::HWModuleOp>());
  instOperands.push_back(clk);
  instOperands.push_back(rst);

  auto topLevelModOp = instOp->getParentOfType<mlir::ModuleOp>();
  hw::HWModuleLike modOp = findExternMod(topLevelModOp, instOp.getModule());
  assert(modOp && "failed to find referenced external module");
  StringAttr instNameAttr = rewriter.getStringAttr(getUniqueName(instOp));
  rewriter.replaceOpWithNewOp<hw::InstanceOp>(instOp, modOp, instNameAttr,
                                              instOperands);
  return success();
}

/// Returns the module's input ports.
static ArrayRef<hw::ModulePort> getModInputs(hw::HWModuleLike modOp) {
  return modOp.getHWModuleType().getPorts().slice(modOp.getPortIdForInputId(0),
                                                  modOp.getNumInputPorts());
}

/// Returns the module's output ports.
static ArrayRef<hw::ModulePort> getModOutputs(hw::HWModuleLike modOp) {
  return modOp.getHWModuleType().getPorts().slice(modOp.getPortIdForOutputId(0),
                                                  modOp.getNumOutputPorts());
}

namespace {

/// Records a mapping between a module's continuous range of output ports and
/// another module's continuous range of input ports. Ranges are identified by
/// their index and must have the same size.
struct IOMapping {
  /// Starting index in the source module's list of output ports.
  size_t srcIdx;
  /// Starting index in the destination module's list of inputs ports.
  size_t dstIdx;
  /// Range size.
  size_t size;

  /// Initialize all fields to 0.
  IOMapping() : srcIdx(0), dstIdx(0), size(0) {}

  /// Member-by-member constructor.
  IOMapping(size_t srcIdx, size_t dstIdx, size_t size)
      : srcIdx(srcIdx), dstIdx(dstIdx), size(size) {}
};

/// Helper class to allow for the creation of a "converter hardware module
/// instance" in between a hardware module's (the "wrapper") top-level IO ports
/// and a module instance (the "circuit") within it.
class ConverterBuilder {
public:
  /// IO mapping from circuit to the converter.
  IOMapping circuitToConverter;
  /// IO mapping from the converter to the wrapper.
  IOMapping converterToWrapper;
  /// IO mapping from the wrapper to the converter.
  IOMapping wrapperToConverter;
  /// IO mapping from the converter to the circuit.
  IOMapping converterToCircuit;

  ConverterBuilder() = default;

  /// Creates the converter builder from the external hardware module which
  /// instances of the converter will reference and the IO mappings between the
  /// converter and the circuit/wrapper.
  ConverterBuilder(hw::HWModuleExternOp converterModOp,
                   const IOMapping &circuitToConverter,
                   const IOMapping &converterToWrapper,
                   const IOMapping &wrapperToConverter,
                   const IOMapping &converterToCircuit)
      : circuitToConverter(circuitToConverter),
        converterToWrapper(converterToWrapper),
        wrapperToConverter(wrapperToConverter),
        converterToCircuit(converterToCircuit), converterModOp(converterModOp) {
  }

private:
  /// Slices the converter module's input ports according to the IO map.
  inline ArrayRef<hw::ModulePort> getSlicedInputs(IOMapping &map) {
    return getModInputs(converterModOp).slice(map.dstIdx, map.size);
  }

  /// Slices the converter module's output ports according to the IO map.
  inline ArrayRef<hw::ModulePort> getSlicedOutputs(IOMapping &map) {
    return getModOutputs(converterModOp).slice(map.srcIdx, map.size);
  }

public:
  /// Adds wrapper inputs that will connect to the converter instance to the
  /// module builder.
  void addWrapperInputs(ModuleBuilder &modBuilder, StringRef baseName) {
    wrapperToConverter.srcIdx = modBuilder.getNumInputs();
    for (const hw::ModulePort port : getSlicedInputs(wrapperToConverter))
      modBuilder.addInput(baseName + "_" + port.name.strref(), port.type);
  }

  /// Adds wrapper output that will originate from the converter instance to the
  /// module builder.
  void addWrapperOutputs(ModuleBuilder &modBuilder, StringRef baseName) {
    converterToWrapper.dstIdx = modBuilder.getNumOutputs();
    for (const hw::ModulePort port : getSlicedOutputs(converterToWrapper))
      modBuilder.addOutput(baseName + "_" + port.name.strref(), port.type);
  }

  /// Adds backedges matching the converter instance's outputs going to the
  /// wrapper to the vector.
  void addWrapperBackedges(BackedgeBuilder &edgeBuilder,
                           SmallVector<Value> &wrapperOutputs) {
    for (const hw::ModulePort port : getSlicedOutputs(converterToWrapper)) {
      wrapperBackedges.push_back(edgeBuilder.get(port.type));
      wrapperOutputs.push_back(wrapperBackedges.back());
    }
  }

  /// Adds backedges matching the converter instance's outputs going to the
  /// circuit to the vector.
  void addCircuitBackedges(BackedgeBuilder &edgeBuilder,
                           SmallVector<Value> &circuitOperands) {
    for (const hw::ModulePort port : getSlicedOutputs(converterToCircuit)) {
      circuitBackedges.push_back(edgeBuilder.get(port.type));
      circuitOperands.push_back(circuitBackedges.back());
    }
  }

public:
  /// Creates an instance of the wrapper between the module's top-level IO ports
  /// and the circuit instance within it. This resolves all internally created
  /// backedges. Returns the converter instance that was inserted.
  hw::InstanceOp createInstance(hw::HWModuleOp wrapperOp,
                                hw::InstanceOp circuitOp, StringRef memName,
                                OpBuilder &builder);

private:
  /// The external hardware module representing the converter.
  hw::HWModuleExternOp converterModOp;
  /// Backedges to the circuit instance's inputs.
  SmallVector<Backedge> circuitBackedges;
  /// Backedges to the wrapper module's outputs.
  SmallVector<Backedge> wrapperBackedges;
};

/// Converter builder between the "simplifier dual-port BRAM interface"
/// implemented by our RTL components and an actual dual-port BRAM interface.
class MemToBRAMConverter : public ConverterBuilder {
public:
  using ConverterBuilder::ConverterBuilder;

  /// RTL module's name (must match one in RTL configuration file).
  static constexpr llvm::StringLiteral HW_NAME = "mem_to_bram";

  /// Constructs from the hardware module that the circuit instance references,
  /// and the memory lowering state object representing the memory interface to
  /// convert.
  MemToBRAMConverter(hw::HWModuleOp circuitMod, const MemLoweringState &state,
                     OpBuilder &builder)
      : ConverterBuilder(buildExternalModule(circuitMod, state, builder),
                         IOMapping(state.outputIdx, 0, 5), IOMapping(0, 0, 8),
                         IOMapping(0, 5, 2), IOMapping(8, state.inputIdx, 1)){};

private:
  /// Creates, inserts, and returns the external harware module corresponding to
  /// the memory converter.
  hw::HWModuleExternOp buildExternalModule(hw::HWModuleOp circuitMod,
                                           const MemLoweringState &memState,
                                           OpBuilder &builder) const;
};

} // namespace

hw::InstanceOp ConverterBuilder::createInstance(hw::HWModuleOp wrapperOp,
                                                hw::InstanceOp circuitOp,
                                                StringRef memName,
                                                OpBuilder &builder) {
  SmallVector<Value> instOperands;

  // Assume that the converter's first inputs come from the wrapper circuit,
  // followed by the wrapper's inputs
  llvm::copy(circuitOp.getResults().slice(circuitToConverter.srcIdx,
                                          circuitToConverter.size),
             std::back_inserter(instOperands));
  llvm::copy(wrapperOp.getBodyBlock()->getArguments().slice(
                 wrapperToConverter.srcIdx, wrapperToConverter.size),
             std::back_inserter(instOperands));

  // Create an instance of the converter
  StringAttr name = builder.getStringAttr("mem_to_bram_converter_" + memName);
  builder.setInsertionPoint(circuitOp);
  hw::InstanceOp converterInstOp = builder.create<hw::InstanceOp>(
      circuitOp.getLoc(), converterModOp, name, instOperands);

  // Resolve backedges in the wrapped circuit operands and in the wrapper's
  // outputs
  ValueRange results = converterInstOp->getResults();
  for (auto [backedge, res] :
       llvm::zip(circuitBackedges, results.slice(converterToCircuit.srcIdx,
                                                 converterToCircuit.size)))
    backedge.setValue(res);
  for (auto [backedge, res] :
       llvm::zip(wrapperBackedges, results.slice(converterToWrapper.srcIdx,
                                                 converterToWrapper.size)))
    backedge.setValue(res);

  return converterInstOp;
}

hw::HWModuleExternOp
MemToBRAMConverter::buildExternalModule(hw::HWModuleOp circuitMod,
                                        const MemLoweringState &memState,
                                        OpBuilder &builder) const {
  std::string extModName =
      HW_NAME.str() + "_" +
      std::to_string(memState.dataType.getIntOrFloatBitWidth()) + "_" +
      std::to_string(memState.ports.addrWidth);
  mlir::ModuleOp topModOp = circuitMod->getParentOfType<mlir::ModuleOp>();
  hw::HWModuleExternOp extModOp = findExternMod(topModOp, extModName);

  if (extModOp)
    return extModOp;

  // The external module does not yet exist, create it
  MLIRContext *ctx = builder.getContext();
  ModuleBuilder modBuilder(ctx);
  Type i1Type = IntegerType::get(ctx, 1);
  Type addrType = IntegerType::get(ctx, memState.ports.addrWidth);

  // Inputs from wrapped circuit
  modBuilder.addInput("loadEn", i1Type);
  modBuilder.addInput("loadAddr", addrType);
  modBuilder.addInput("storeEn", i1Type);
  modBuilder.addInput("storeAddr", addrType);
  modBuilder.addInput("storeData", memState.dataType);

  // Outputs to wrapper
  modBuilder.addOutput("ce0", i1Type);
  modBuilder.addOutput("we0", i1Type);
  modBuilder.addOutput("address0", addrType);
  modBuilder.addOutput("dout0", memState.dataType);
  modBuilder.addOutput("ce1", i1Type);
  modBuilder.addOutput("we1", i1Type);
  modBuilder.addOutput("address1", addrType);
  modBuilder.addOutput("dout1", memState.dataType);

  // Inputs from wrapper
  modBuilder.addInput("din0", memState.dataType);
  modBuilder.addInput("din1", memState.dataType);

  // Outputs to wrapped circuit
  modBuilder.addOutput("loadData", memState.dataType);

  builder.setInsertionPointToEnd(topModOp.getBody());
  StringAttr modNameAttr = builder.getStringAttr(extModName);
  extModOp = builder.create<hw::HWModuleExternOp>(
      circuitMod->getLoc(), modNameAttr, modBuilder.getPortInfo());

  extModOp->setAttr(RTL_NAME_ATTR_NAME, StringAttr::get(ctx, HW_NAME));
  SmallVector<NamedAttribute> parameters;
  Type i32 = IntegerType::get(ctx, 32, IntegerType::Unsigned);
  parameters.emplace_back(
      StringAttr::get(ctx, "DATA_WIDTH"),
      IntegerAttr::get(i32, memState.dataType.getIntOrFloatBitWidth()));
  parameters.emplace_back(StringAttr::get(ctx, "ADDR_WIDTH"),
                          IntegerAttr::get(i32, memState.ports.addrWidth));
  extModOp->setAttr(RTL_PARAMETERS_ATTR_NAME,
                    DictionaryAttr::get(ctx, parameters));
  return extModOp;
}

/// Creates and returns an empty wrapper module. When the function returns,
/// `memConverters `associates each memory interface in the wrapped circuit to a
/// builder for their respective converter; in addition, backedges for the
/// future wrapped circuit results going directly to the wrapper (without
/// passing through a converter) are stored along their corresponding result
/// index inside the `circuitBackedges` vector.
static hw::HWModuleOp createEmptyWrapperMod(
    hw::HWModuleOp circuitOp, LoweringState &state, OpBuilder &builder,
    DenseMap<const MemLoweringState *, ConverterBuilder> &memConverters,
    SmallVector<std::pair<size_t, Backedge>> &circuitBackedges) {

  ModuleLoweringState &modState = state.modState[circuitOp];
  MLIRContext *ctx = builder.getContext();
  ModuleBuilder wrapperBuilder(ctx);

  DenseMap<size_t, const MemLoweringState *> inputToMem, outputToMem;
  for (const auto &[_, memState] : modState.memInterfaces) {
    if (!memState.connectsToCircuit())
      continue;
    inputToMem[memState.inputIdx] = &memState;
    outputToMem[memState.outputIdx] = &memState;
    memConverters[&memState] = MemToBRAMConverter(circuitOp, memState, builder);
  }

  // Create input ports for the wrapper; we need to identify the inputs which
  // map to internal memory interfaces and replace them with an interface for a
  // dual-port BRAM
  ArrayRef<hw::ModulePort> inputPorts = getModInputs(circuitOp);
  for (size_t i = 0, e = inputPorts.size(); i < e;) {
    hw::ModulePort port = inputPorts[i];
    if (auto it = inputToMem.find(i); it != inputToMem.end()) {
      // Beginning of internal mem interface, replace with IO for dual-port BRAM
      const MemLoweringState *memState = it->second;
      ConverterBuilder &converter = memConverters.find(memState)->second;
      converter.addWrapperInputs(wrapperBuilder, memState->name);
      i += memState->numInputs;
    } else {
      // This is a regular argument, just forward it
      wrapperBuilder.addInput(port.name.strref(), port.type);
      ++i;
    }
  }

  // Same as above, but for the wrapper's outputs
  ArrayRef<hw::ModulePort> outputPorts = getModOutputs(circuitOp);
  DenseMap<size_t, ConverterBuilder *> wrapperOutputToMem;
  for (size_t i = 0, e = outputPorts.size(); i < e;) {
    hw::ModulePort port = outputPorts[i];
    if (auto it = outputToMem.find(i); it != outputToMem.end()) {
      // Beginning of internal mem interface, replace with IO for dual-port BRAM
      const MemLoweringState *memState = it->second;
      ConverterBuilder &converter = memConverters.find(memState)->second;
      wrapperOutputToMem[wrapperBuilder.getNumOutputs()] = &converter;
      converter.addWrapperOutputs(wrapperBuilder, memState->name);
      i += memState->numOutputs;
    } else {
      // This is a regular result, just forward it
      wrapperBuilder.addOutput(port.name.strref(), port.type);
      ++i;
    }
  }

  // Create the wrapper
  builder.setInsertionPointToEnd(state.modOp.getBody());
  hw::HWModuleOp wrapperOp = builder.create<hw::HWModuleOp>(
      circuitOp.getLoc(),
      StringAttr::get(ctx, circuitOp.getSymName() + "_wrapper"),
      wrapperBuilder.getPortInfo());
  builder.setInsertionPointToStart(wrapperOp.getBodyBlock());

  // Create backedges for all of the wrapper module's outputs
  SmallVector<Value> modOutputs;
  ArrayRef<hw::ModulePort> wrapperOutputs = getModOutputs(wrapperOp);
  for (size_t i = 0, e = wrapperOutputs.size(); i < e;) {
    hw::ModulePort port = wrapperOutputs[i];
    if (auto it = wrapperOutputToMem.find(i); it != wrapperOutputToMem.end()) {
      // This is the beginning of memory interface outputs that will eventually
      // come from a converter
      it->second->addWrapperBackedges(state.edgeBuilder, modOutputs);
      i += it->second->converterToWrapper.size;
    } else {
      // This is a regular result that will come directly from the wrapped
      // circuit
      circuitBackedges.push_back({i, state.edgeBuilder.get(port.type)});
      modOutputs.push_back(circuitBackedges.back().second);
      i += 1;
    }
  }

  Operation *outputOp = wrapperOp.getBodyBlock()->getTerminator();
  outputOp->setOperands(modOutputs);
  return wrapperOp;
};

/// Creates a wrapper module made up of the hardware module that resulted from
/// Handshake lowering and of memory converters sitting between the latter's
/// memory interfaces and "standard memory interfaces" exposed by the wrapper's
/// module.
static void createWrapper(hw::HWModuleOp circuitOp, LoweringState &state,
                          OpBuilder &builder) {

  DenseMap<const MemLoweringState *, ConverterBuilder> memConverters;
  SmallVector<std::pair<size_t, Backedge>> circuitBackedges;
  hw::HWModuleOp wrapperOp = createEmptyWrapperMod(
      circuitOp, state, builder, memConverters, circuitBackedges);
  builder.setInsertionPointToStart(wrapperOp.getBodyBlock());

  // Operands for the circuit instance inside the wrapper
  SmallVector<Value> circuitOperands;

  DenseMap<size_t, ConverterBuilder *> wrapperInputToConv;
  for (auto &[_, converter] : memConverters)
    wrapperInputToConv[converter.wrapperToConverter.srcIdx] = &converter;

  ArrayRef<BlockArgument> wrapperArgs = wrapperOp.getBody().getArguments();
  for (size_t i = 0, e = wrapperArgs.size(); i < e;) {
    if (auto it = wrapperInputToConv.find(i); it != wrapperInputToConv.end()) {
      ConverterBuilder *converter = it->second;
      converter->addCircuitBackedges(state.edgeBuilder, circuitOperands);
      i += converter->wrapperToConverter.size;
    } else {
      // Argument, just forward it to the operands
      circuitOperands.push_back(wrapperArgs[i]);
      ++i;
    }
  }

  // Create the wrapped circuit instance inside the wrapper
  hw::InstanceOp circuitInstOp = builder.create<hw::InstanceOp>(
      circuitOp.getLoc(), circuitOp,
      builder.getStringAttr(circuitOp.getSymName() + "_wrapped"),
      circuitOperands);

  // Instantiate the memory interface converters inside the wrapper module,
  // which also resolved all backedges meant to originate from the converter
  for (auto &[memState, converter] : memConverters)
    converter.createInstance(wrapperOp, circuitInstOp, memState->name, builder);

  // Resolve backedges coming from the circuit to the wrapper's outputs
  for (auto [resIdx, backedge] : circuitBackedges)
    backedge.setValue(circuitInstOp.getResult(resIdx));
}

namespace {

/// Conversion pass driver. The conversion only works on modules containing
/// a single handshake function (handshake::FuncOp) at the moment. The
/// function and all the operations it contains are converted to operations
/// from the HW dialect. Dataflow semantics are made explicit with Handshake
/// channels.
class HandshakeToHWPass
    : public dynamatic::impl::HandshakeToHWBase<HandshakeToHWPass> {
public:
  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();

    // We only support one function per module
    handshake::FuncOp funcOp = nullptr;
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

    if (funcOp) {
      // Check that some preconditions are met before doing anything
      if (failed(verifyIRMaterialized(funcOp))) {
        funcOp.emitError() << ERR_NON_MATERIALIZED_FUNC;
        return signalPassFailure();
      }
    }

    // Make sure all operations are named
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    namer.nameAllUnnamedOps();

    ChannelTypeConverter typeConverter;
    if (failed(convertExternalFunctions(typeConverter)))
      return signalPassFailure();

    // Helper struct for lowering
    OpBuilder builder(ctx);
    LoweringState lowerState(modOp, namer, builder);

    // Create pattern set
    RewritePatternSet patterns(ctx);
    patterns.insert<ConvertFunc, ConvertMemInterface>(typeConverter, ctx,
                                                      lowerState);
    patterns.insert<ConvertInstance, ConvertToHWInstance<handshake::BufferOp>,
                    ConvertToHWInstance<handshake::NDWireOp>,
                    ConvertToHWInstance<handshake::ConditionalBranchOp>,
                    ConvertToHWInstance<handshake::BranchOp>,
                    ConvertToHWInstance<handshake::MergeOp>,
                    ConvertToHWInstance<handshake::ControlMergeOp>,
                    ConvertToHWInstance<handshake::MuxOp>,
                    ConvertToHWInstance<handshake::JoinOp>,
                    ConvertToHWInstance<handshake::BlockerOp>,
                    ConvertToHWInstance<handshake::SourceOp>,
                    ConvertToHWInstance<handshake::ConstantOp>,
                    ConvertToHWInstance<handshake::SinkOp>,
                    ConvertToHWInstance<handshake::ForkOp>,
                    ConvertToHWInstance<handshake::LazyForkOp>,
                    ConvertToHWInstance<handshake::LoadOp>,
                    ConvertToHWInstance<handshake::StoreOp>,
                    ConvertToHWInstance<handshake::NotOp>,
                    ConvertToHWInstance<handshake::SharingWrapperOp>,

                    // Arith operations
                    ConvertToHWInstance<handshake::AddFOp>,
                    ConvertToHWInstance<handshake::AddIOp>,
                    ConvertToHWInstance<handshake::AndIOp>,
                    ConvertToHWInstance<handshake::CmpFOp>,
                    ConvertToHWInstance<handshake::CmpIOp>,
                    ConvertToHWInstance<handshake::DivFOp>,
                    ConvertToHWInstance<handshake::DivSIOp>,
                    ConvertToHWInstance<handshake::DivUIOp>,
                    ConvertToHWInstance<handshake::ExtSIOp>,
                    ConvertToHWInstance<handshake::ExtUIOp>,
                    ConvertToHWInstance<handshake::MulFOp>,
                    ConvertToHWInstance<handshake::MulIOp>,
                    ConvertToHWInstance<handshake::NegFOp>,
                    ConvertToHWInstance<handshake::OrIOp>,
                    ConvertToHWInstance<handshake::SelectOp>,
                    ConvertToHWInstance<handshake::ShLIOp>,
                    ConvertToHWInstance<handshake::ShRSIOp>,
                    ConvertToHWInstance<handshake::ShRUIOp>,
                    ConvertToHWInstance<handshake::SubFOp>,
                    ConvertToHWInstance<handshake::SubIOp>,
                    ConvertToHWInstance<handshake::TruncIOp>,
                    ConvertToHWInstance<handshake::TruncFOp>,
                    ConvertToHWInstance<handshake::XOrIOp>,
                    ConvertToHWInstance<handshake::SIToFPOp>,
                    ConvertToHWInstance<handshake::FPToSIOp>,
                    ConvertToHWInstance<handshake::ExtFOp>,
                    ConvertToHWInstance<handshake::AbsFOp>,

                    // Speculative operations
                    ConvertToHWInstance<handshake::SpecCommitOp>,
                    ConvertToHWInstance<handshake::SpecSaveOp>,
                    ConvertToHWInstance<handshake::SpecSaveCommitOp>,
                    ConvertToHWInstance<handshake::SpeculatorOp>,
                    ConvertToHWInstance<handshake::SpeculatingBranchOp>,
                    ConvertToHWInstance<handshake::NonSpecOp>>(
        typeConverter, funcOp->getContext());

    // Everything must be converted to operations in the hw dialect
    ConversionTarget target(*ctx);
    target.addLegalOp<hw::HWModuleOp, hw::HWModuleExternOp, hw::InstanceOp,
                      hw::OutputOp>();
    target.addIllegalDialect<handshake::HandshakeDialect,
                             memref::MemRefDialect>();

    if (failed(applyPartialConversion(modOp, target, std::move(patterns))))
      return signalPassFailure();

    // Create memory wrappers around all hardware modules
    for (auto &[circuitOp, _] : lowerState.modState)
      createWrapper(circuitOp, lowerState, builder);

    // At this level all operations already have an intrinsic name so we can
    // disable our naming system
    doNotNameOperations();
  }

private:
  /// Converts all external `handshake::FuncOp` operations into corresponding
  /// `hw::HWModuleExternOp` operations using a partial IR conversion.
  LogicalResult convertExternalFunctions(ChannelTypeConverter &typeConverter) {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<ConvertExternalFunc>(typeConverter, ctx);
    ConversionTarget target(*ctx);
    target.addLegalOp<hw::HWModuleExternOp>();
    target.addDynamicallyLegalOp<handshake::FuncOp>(
        [&](handshake::FuncOp funcOp) { return !funcOp.isExternal(); });
    target.markOpRecursivelyLegal<handshake::FuncOp>();
    return applyPartialConversion(getOperation(), target, std::move(patterns));
  }
};

} // end anonymous namespace

std::unique_ptr<dynamatic::DynamaticPass> dynamatic::createHandshakeToHWPass() {
  return std::make_unique<HandshakeToHWPass>();
}
