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
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/HW/HWTypes.h"
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/RTL.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <bitset>
#include <iterator>
#include <string>

using namespace mlir;
using namespace dynamatic;

/// Name of ports representing the clock and reset signals.
static constexpr llvm::StringLiteral CLK_PORT("clk"), RST_PORT("rst");

//===----------------------------------------------------------------------===//
// Internal data-structures
//===----------------------------------------------------------------------===//

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
    addInput(CLK_PORT, i1Type);
    addInput(RST_PORT, i1Type);
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

/// Provides an opaque interface for generating the port names of an operation;
/// handshake operations generate names by the `handshake::NamedIOInterface`;
/// and other operations, such as arith ops, are assigned default names.
class PortNameGenerator {
public:
  /// Does nohting; no port name will be generated.
  PortNameGenerator() = default;

  /// Derives port names for the operation on object creation.
  PortNameGenerator(Operation *op);

  /// Returs the port name of the input at the specified index.
  StringRef getInputName(unsigned idx) { return inputs[idx]; }

  /// Returs the port name of the output at the specified index.
  StringRef getOutputName(unsigned idx) { return outputs[idx]; }

private:
  /// Maps the index of an input or output to its port name.
  using IdxToStrF = const std::function<std::string(unsigned)> &;

  /// Infers port names for the operation using the provided callbacks.
  void infer(Operation *op, IdxToStrF &inF, IdxToStrF &outF);

  /// Infers default port names when nothing better can be achieved.
  void inferDefault(Operation *op);

  /// Infers port names for an operation implementing the
  /// `handshake::NamedIOInterface` interface.
  void inferFromNamedOpInterface(handshake::NamedIOInterface namedIO);

  /// Infers port names for a Handshake function.
  void inferFromFuncOp(handshake::FuncOp funcOp);

  /// List of input port names.
  SmallVector<std::string> inputs;
  /// List of output port names.
  SmallVector<std::string> outputs;
};
} // namespace

PortNameGenerator::PortNameGenerator(Operation *op) {
  assert(op && "cannot generate port names for null operation");
  if (auto namedOpInterface = dyn_cast<handshake::NamedIOInterface>(op))
    inferFromNamedOpInterface(namedOpInterface);
  else if (auto funcOp = dyn_cast<handshake::FuncOp>(op))
    inferFromFuncOp(funcOp);
  else
    inferDefault(op);
}

void PortNameGenerator::infer(Operation *op, IdxToStrF &inF, IdxToStrF &outF) {
  for (size_t idx = 0, e = op->getNumOperands(); idx < e; ++idx)
    inputs.push_back(inF(idx));
  for (size_t idx = 0, e = op->getNumResults(); idx < e; ++idx)
    outputs.push_back(outF(idx));

  // The Handshake terminator forwards its non-memory inputs to its outputs, so
  // it needs port names for them
  if (handshake::EndOp endOp = dyn_cast<handshake::EndOp>(op)) {
    handshake::FuncOp funcOp = endOp->getParentOfType<handshake::FuncOp>();
    assert(funcOp && "end must be child of handshake function");
    size_t numResults = funcOp.getFunctionType().getNumResults();
    for (size_t idx = 0, e = numResults; idx < e; ++idx)
      outputs.push_back(endOp.getDefaultResultName(idx));
  }
}

void PortNameGenerator::inferDefault(Operation *op) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<arith::AddFOp, arith::AddIOp, arith::AndIOp, arith::CmpIOp,
            arith::CmpFOp, arith::DivFOp, arith::DivSIOp, arith::DivUIOp,
            arith::MaximumFOp, arith::MinimumFOp, arith::MulFOp, arith::MulIOp,
            arith::OrIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
            arith::SubFOp, arith::SubIOp, arith::XOrIOp>([&](auto) {
        infer(
            op, [](unsigned idx) { return idx == 0 ? "lhs" : "rhs"; },
            [](unsigned idx) { return "result"; });
      })
      .Case<arith::ExtSIOp, arith::ExtUIOp, arith::NegFOp, arith::TruncIOp>(
          [&](auto) {
            infer(
                op, [](unsigned idx) { return "ins"; },
                [](unsigned idx) { return "outs"; });
          })
      .Case<arith::SelectOp>([&](auto) {
        infer(
            op,
            [](unsigned idx) {
              switch (idx) {
              case 0:
                return "condition";
              case 1:
                return "trueOut";
              }
              return "falseOut";
            },
            [](unsigned idx) { return "result"; });
      })
      .Default([&](auto) {
        infer(
            op, [](unsigned idx) { return "in" + std::to_string(idx); },
            [](unsigned idx) { return "out" + std::to_string(idx); });
      });
}

void PortNameGenerator::inferFromNamedOpInterface(
    handshake::NamedIOInterface namedIO) {
  auto inF = [&](unsigned idx) { return namedIO.getOperandName(idx); };
  auto outF = [&](unsigned idx) { return namedIO.getResultName(idx); };
  infer(namedIO, inF, outF);
}

void PortNameGenerator::inferFromFuncOp(handshake::FuncOp funcOp) {
  llvm::transform(funcOp.getArgNames(), std::back_inserter(inputs),
                  [](Attribute arg) { return cast<StringAttr>(arg).str(); });
  llvm::transform(funcOp.getResNames(), std::back_inserter(outputs),
                  [](Attribute res) { return cast<StringAttr>(res).str(); });
}

namespace {
/// Aggregates information to convert a Handshake memory interface into a
/// `hw::InstanceOp`. This must be created during conversion of the Handsahke
/// function containing the interface.
struct MemLoweringState {
  /// Memory region's name.
  std::string memName;
  /// Data type.
  Type dataType = nullptr;
  /// Cache memory port information before modifying the interface, which can
  /// make them impossible to query.
  FuncMemoryPorts ports;
  /// Generates and stores the interface's port names before starting the
  /// conversion, when those are still queryable.
  PortNameGenerator portNames;
  /// Backedges to the containing module's `hw::OutputOp` operation, which
  /// must be set, in order, with the memory interface's results that connect
  /// to the top-level module IO.
  SmallVector<Backedge> backedges;

  /// Index of first module input corresponding the the interface's inputs.
  size_t inputIdx = 0;
  /// Number of inputs for the memory interface, starting at the `inputIdx`.
  size_t numInputs = 0;
  /// Index of first module output corresponding the the interface's ouputs.
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
      : memName(name.str()), dataType(memOp.getMemRefType().getElementType()),
        ports(getMemoryPorts(memOp)), portNames(memOp){};

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
  llvm::DenseMap<handshake::MemoryOpInterface, MemLoweringState> memInterfaces;
  /// Generates and stores the end operations's port names before starting the
  /// conversion, when those are still queryable.
  PortNameGenerator endPorts;
  /// Backedges to the containing module's `hw::OutputOp` operation, which
  /// must be set, in order, with the results of the `hw::InstanceOp`
  /// operation to which the `handshake::EndOp` operation was converted to.
  SmallVector<Backedge> endBackedges;

  /// Default constructor required because we use the class as a map's value,
  /// which must be default cosntructible.
  ModuleLoweringState() = default;

  /// Constructs the lowering state from the Handshake function to lower.
  ModuleLoweringState(handshake::FuncOp funcOp)
      : endPorts(funcOp.getBodyBlock()->getTerminator()){};

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
  modBuilder.addInput(memName + "_loadData", dataType);
  // Load enable output
  modBuilder.addOutput(memName + "_loadEn", i1Type);
  // Load address output
  modBuilder.addOutput(memName + "_loadAddr", addrType);
  // Store enable output
  modBuilder.addOutput(memName + "_storeEn", i1Type);
  // Store address output
  modBuilder.addOutput(memName + "_storeAddr", addrType);
  // Store data output
  modBuilder.addOutput(memName + "_storeData", dataType);

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
  /// with different parameter values will never received the same name.
  std::string getDiscriminatedModName() { return modName; }

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
  /// Discriminated module name.
  std::string modName;
  /// The operation's parameters, as a list of named attributes.
  SmallVector<NamedAttribute> parameters;

  /// Whether the operation is unsupported (set during construction).
  bool unsupported = false;

  /// Adds a parameter.
  void addParam(const Twine &name, Attribute attr) {
    parameters.emplace_back(StringAttr::get(ctx, name), attr);
  }

  /// Adds a scalar-type parameter.
  void addUnsigned(const Twine &name, unsigned scalar) {
    Type intType = IntegerType::get(ctx, 32, IntegerType::Unsigned);
    addParam(name, IntegerAttr::get(intType, scalar));
    modName += "_" + std::to_string(scalar);
  };

  /// Adds a bitwdith parameter extracted from a type.
  void addBitwidth(const Twine &name, Type type) {
    addUnsigned(name, getTypeWidth(type));
  };

  /// Adds a bitwdith parameter extracted from a value's type.
  void addBitwidth(const Twine &name, Value val) {
    addUnsigned(name, getTypeWidth(val.getType()));
  };

  /// Adds a string parameter.
  void addString(const Twine &name, const Twine &txt) {
    addParam(name, StringAttr::get(ctx, txt));
    modName += "_" + txt.str();
  };

  /// Returns the module name's prefix from the name of the operation it
  /// represents.
  std::string setModPrefix(Operation *op) {
    std::string prefixModName = op->getName().getStringRef().str();
    std::replace(prefixModName.begin(), prefixModName.end(), '.', '_');
    return prefixModName;
  }

  /// Returns the bitwidth of a type.
  static unsigned getTypeWidth(Type type);
};
} // namespace

ModuleDiscriminator::ModuleDiscriminator(Operation *op)
    : op(op), ctx(op->getContext()), modName(setModPrefix(op)) {

  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::BufferOpInterface>(
          [&](handshake::BufferOpInterface bufOp) {
            // Number of slots and bitwdith
            addUnsigned("SLOTS", bufOp.getSlots());
            addBitwidth("DATA_WIDTH", op->getResult(0));
          })
      .Case<handshake::ForkOp, handshake::LazyForkOp>([&](auto) {
        // Number of output channels and bitwidth
        addUnsigned("SIZE", op->getNumResults());
        addBitwidth("DATA_WIDTH", op->getOperand(0));
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        // Number of input data channels, data bitwidth, and select bitwidth
        addUnsigned("SIZE", muxOp.getDataOperands().size());
        addBitwidth("DATA_WIDTH", muxOp.getResult());
        addBitwidth("SELECT_WIDTH", muxOp.getSelectOperand());
      })
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp cmergeOp) {
        // Number of input data channels, data bitwidth, and index
        // bitwidth
        addUnsigned("SIZE", cmergeOp.getDataOperands().size());
        addBitwidth("DATA_WIDTH", cmergeOp.getResult());
        addBitwidth("INDEX_WIDTH", cmergeOp.getIndex());
      })
      .Case<handshake::MergeOp>([&](auto) {
        // Number of input data channels and data bitwidth
        addUnsigned("SIZE", op->getNumOperands());
        addBitwidth("DATA_WIDTH", op->getResult(0));
      })
      .Case<handshake::BranchOp, handshake::SinkOp>([&](auto) {
        // Bitwidth
        addBitwidth("DATA_WIDTH", op->getOperand(0));
      })
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp cbrOp) {
            // Bitwidth
            addBitwidth("DATA_WIDTH", cbrOp.getDataOperand());
          })
      .Case<handshake::SourceOp>([&](auto) {
        // No discrimianting parameters, just to avoid falling into the
        // default case for sources
      })
      .Case<handshake::LoadOpInterface>([&](handshake::LoadOpInterface loadOp) {
        // Data bitwidth and address bitwidth
        addBitwidth("DATA_WIDTH", loadOp.getDataInput());
        addBitwidth("ADDR_WIDTH", loadOp.getAddressInput());
      })
      .Case<handshake::StoreOpInterface>(
          [&](handshake::StoreOpInterface storeOp) {
            // Data bitwidth and address bitwidth
            addBitwidth("DATA_WIDTH", storeOp.getDataInput());
            addBitwidth("ADDR_WIDTH", storeOp.getAddressInput());
          })
      .Case<handshake::ConstantOp>([&](handshake::ConstantOp cstOp) {
        // Bitwidth and binary-encoded constant value
        Type cstType = cstOp.getResult().getType();
        unsigned bitwidth = cstType.getIntOrFloatBitWidth();
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
        if (isa<IntegerType>(cstType)) {
          APInt value = cast<mlir::IntegerAttr>(valueAttr).getValue();

          // Bitset requires a compile-time constant, just use 64 and
          // manually truncate the value after so that it is the exact
          // bitwidth we need
          if (cstType.isUnsignedInteger())
            bitValue = std::bitset<64>(value.getZExtValue()).to_string();
          else
            bitValue = std::bitset<64>(value.getSExtValue()).to_string();
          bitValue = bitValue.substr(64 - bitwidth);
        } else if (isa<FloatType>(cstType)) {
          mlir::FloatAttr attr = dyn_cast<mlir::FloatAttr>(valueAttr);
          // We only support specific bitwidths for floating point numbers
          if (bitwidth == 32) {
            bitValue =
                std::bitset<32>(attr.getValue().convertToFloat()).to_string();
          } else if (bitwidth == 64) {
            bitValue =
                std::bitset<64>(attr.getValue().convertToDouble()).to_string();
          } else {
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
      .Case<handshake::EndOp>([&](auto) {
        // Number of memory inputs and bitwidth (we assume that there is
        // a single function return value due to our current VHDL
        // limitation)
        addBitwidth("DATA_WIDTH", op->getOperand(0));
        addUnsigned("NUM_MEMORIES", op->getNumOperands() - 1);
      })
      .Case<handshake::ReturnOp>([&](auto) {
        // Bitwidth (we assume that there is a single function return value
        // due to our current VHDL limitation)
        addBitwidth("DATA_WIDTH", op->getOperand(0));
      })
      .Case<arith::AddFOp, arith::AddIOp, arith::AndIOp, arith::DivFOp,
            arith::DivSIOp, arith::DivUIOp, arith::MaximumFOp,
            arith::MinimumFOp, arith::MulFOp, arith::MulIOp, arith::NegFOp,
            arith::OrIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
            arith::SubFOp, arith::SubIOp, arith::XOrIOp>([&](auto) {
        // Bitwidth
        addBitwidth("DATA_WIDTH", op->getOperand(0));
      })
      .Case<arith::SelectOp>([&](arith::SelectOp selectOp) {
        // Data bitwidth
        addBitwidth("DATA_WIDTH", selectOp.getTrueValue());
      })
      .Case<arith::CmpFOp>([&](arith::CmpFOp cmpFOp) {
        // Predicate and bitwidth
        addString("PREDICATE", stringifyEnum(cmpFOp.getPredicate()));
        addBitwidth("DATA_WIDTH", cmpFOp.getLhs());
      })
      .Case<arith::CmpIOp>([&](arith::CmpIOp cmpIOp) {
        // Predicate and bitwidth
        addString("PREDICATE", stringifyEnum(cmpIOp.getPredicate()));
        addBitwidth("DATA_WIDTH", cmpIOp.getLhs());
      })
      .Case<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>([&](auto) {
        // Input bitwidth and output bitwidth
        addBitwidth("INPUT_WIDTH", op->getOperand(0));
        addBitwidth("OUTPUT_WIDTH", op->getResult(0));
      })
      .Default([&](auto) {
        op->emitError() << "This operation cannot be lowered to RTL "
                           "due to a lack of an RTL implementation for it.";
        unsupported = true;
      });
}

ModuleDiscriminator::ModuleDiscriminator(FuncMemoryPorts &ports)
    : op(ports.memOp), ctx(op->getContext()), modName(setModPrefix(op)) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::MemoryControllerOp>([&](auto) {
        // Control port count, load port count, store port count, data
        // bitwidth, and address bitwidth
        addUnsigned("NUM_CONTROL", ports.getNumPorts<ControlPort>());
        addUnsigned("NUM_LOAD", ports.getNumPorts<LoadPort>());
        addUnsigned("NUM_STORE", ports.getNumPorts<StorePort>());
        addUnsigned("DATA_WIDTH", ports.dataWidth);
        addUnsigned("ADDR_WIDTH", ports.addrWidth);
      })
      .Case<handshake::LSQOp>([&](auto) {
        LSQGenerationInfo genInfo(ports, getUniqueName(op).str());
        modName += "_" + genInfo.name;

        SmallVector<NamedAttribute> attributes;

        /// Converts a string into an equivalent MLIR attribute.
        auto stringAttr = [&](StringRef name, StringRef value) -> void {
          attributes.emplace_back(StringAttr::get(ctx, name),
                                  StringAttr::get(ctx, value));
        };

        /// Converts an unsigned number into an equivalent MLIR attribute.
        Type intType = IntegerType::get(ctx, 32);
        auto intAttr = [&](StringRef name, unsigned value) -> void {
          attributes.emplace_back(StringAttr::get(ctx, name),
                                  IntegerAttr::get(intType, value));
        };

        /// Converts an array into an equivalent MLIR attribute.
        auto arrayIntAttr = [&](StringRef name,
                                ArrayRef<unsigned> array) -> void {
          SmallVector<Attribute> arrayAttr;
          llvm::transform(
              array, std::back_inserter(arrayAttr),
              [&](unsigned elem) { return IntegerAttr::get(intType, elem); });
          attributes.emplace_back(StringAttr::get(ctx, name),
                                  ArrayAttr::get(ctx, arrayAttr));
        };

        /// Converts a bi-dimensional array into an equivalent MLIR attribute.
        auto biArrayIntAttr =
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
          attributes.emplace_back(StringAttr::get(ctx, name),
                                  ArrayAttr::get(ctx, biArrayAttr));
        };

        stringAttr("name", modName);
        intAttr("fifoDepth", genInfo.depth);
        intAttr("fifoDepth_L", genInfo.depthLoad);
        intAttr("fifoDepth_S", genInfo.depthStore);
        intAttr("bufferDepth", genInfo.bufferDepth);
        stringAttr("accessType", genInfo.accessType);
        // The Chisel LSQ generator expects this to be a string, not a boolean
        stringAttr("speculation", genInfo.speculation ? "true" : "false");
        intAttr("dataWidth", genInfo.dataWidth);
        intAttr("addrWidth", genInfo.addrWidth);
        intAttr("numBBs", genInfo.numGroups);
        intAttr("numLoadPorts", genInfo.numLoads);
        intAttr("numStorePorts", genInfo.numStores);
        arrayIntAttr("numLoads", genInfo.loadsPerGroup);
        arrayIntAttr("numStores", genInfo.storesPerGroup);
        biArrayIntAttr("loadOffsets",
                       ArrayRef<SmallVector<unsigned>>{genInfo.loadOffsets});
        biArrayIntAttr("storeOffsets",
                       ArrayRef<SmallVector<unsigned>>{genInfo.storeOffsets});
        biArrayIntAttr("loadPorts", genInfo.loadPorts);
        biArrayIntAttr("storePorts", genInfo.storePorts);

        // The LSQ generator expects the JSON containing all the elements to be
        // nested inside a JSON array under  the "specifications" key within the
        // top-level JSON object, make it so
        // {
        //   "specifications": [{
        //    ...all generation info
        //   }]
        // }
        DictionaryAttr dictAttr = DictionaryAttr::get(ctx, attributes);
        ArrayAttr arrayAttr =
            ArrayAttr::get(ctx, SmallVector<Attribute>{dictAttr});
        addParam("specifications", arrayAttr);
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
  modOp->setAttr(RTLRequest::NAME_ATTR, StringAttr::get(ctx, opName));

  // Parameters are used to determine the concrete version of the RTL
  // component to instantiate
  modOp->setAttr(RTLRequest::PARAMETERS_ATTR,
                 DictionaryAttr::get(ctx, parameters));
}

unsigned ModuleDiscriminator::getTypeWidth(Type type) {
  if (isa<IntegerType, FloatType>(type))
    return type.getIntOrFloatBitWidth();
  if (isa<NoneType>(type))
    return 0;
  llvm_unreachable("unsupported data type");
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

/// Specialization of the hardware builder for the `handshake::EndOp`
/// operations.
class HWEndConverter : public HWConverter {
public:
  using HWConverter::HWConverter;

  /// Replaces the end operation with an equivalent instance using all the
  /// inputs added so far as operands. If no external module matching the
  /// current port information currently exists, one is added at the bottom of
  /// the top-level MLIR module. Returns the instance on success, nullptr on
  /// failure.
  hw::InstanceOp convertToInstance(handshake::EndOp endOp,
                                   ModuleLoweringState &state,
                                   ConversionPatternRewriter &rewriter) {
    hw::InstanceOp instOp = createInstanceFromOp(endOp, rewriter);
    if (!instOp)
      return nullptr;

    // Resolve backedges in the module's terminator that are coming from the end
    ValueRange results = instOp.getResults();
    for (auto [backedge, res] : llvm::zip_equal(state.endBackedges, results))
      backedge.setValue(res);
    rewriter.eraseOp(endOp);
    return instOp;
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
    ValueRange results = instOp->getResults().drop_front(numResults);
    for (auto [backedge, res] : llvm::zip_equal(state.backedges, results))
      backedge.setValue(res);
    return instOp;
  }
};
} // namespace

void HWBuilder::addClkAndRst(hw::HWModuleOp hwModOp) {
  // Let the parent class add clock and reset to the input ports
  modBuilder.addClkAndRst();

  // Check that the parent module's last port are the clock and reset
  // signals we need for the instance operands
  unsigned numInputs = hwModOp.getNumInputPorts();
  assert(numInputs >= 2 && "module should have at least clock and reset");
  size_t lastIdx = hwModOp.getPortIdForInputId(numInputs - 1);
  assert(hwModOp.getPort(lastIdx - 1).getName() == CLK_PORT &&
         "expected clock");
  assert(hwModOp.getPort(lastIdx).getName() == RST_PORT && "expected reset");

  // Add clock and reset to the instance's operands
  ValueRange blockArgs = hwModOp.getBodyBlock()->getArguments();
  instOperands.push_back(blockArgs.drop_back().back());
  instOperands.push_back(blockArgs.back());
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
    llvm::TypeSwitch<Operation *, void>(memOp.getOperation())
        .Case<handshake::MemoryControllerOp>(
            [&](handshake::MemoryControllerOp mcOp) {
              info.connectWithCircuit(modBuilder);
            })
        .Case<handshake::LSQOp>([&](handshake::LSQOp lsqOp) {
          if (lsqOp.isConnectedToMC())
            return;

          // If the LSQ does not connect to an MC, then it connects directly to
          // a dual-port BRAM through the top-level module IO
          info.connectWithCircuit(modBuilder);
        })
        .Default([&](auto) { llvm_unreachable("unknown memory interface"); });

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
  PortNameGenerator portNames(funcOp);

  // Add all function outputs to the module
  for (auto [idx, res] : llvm::enumerate(funcOp.getResultTypes()))
    modBuilder.addOutput(portNames.getOutputName(idx), channelWrapper(res));

  // Add all function inputs to the module, expanding memory references into a
  // set of individual ports for loads and stores
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    StringAttr argName = funcOp.getArgName(idx);
    Type type = arg.getType();
    if (TypedValue<MemRefType> memref = dyn_cast<TypedValue<MemRefType>>(arg))
      addMemIO(modBuilder, funcOp, memref, argName, state);
    else
      modBuilder.addInput(portNames.getInputName(idx), channelWrapper(type));
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
/// original Handshake function. In case of non-external function, the
/// pattern creates a lowering state object associated to the created HW
/// module to control the conversion of other operations within the module.
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
  ModuleLoweringState state(funcOp);
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

  // Create backege inputs for the module's output operation and associate
  // them to the future operations whose conversion will resolve them
  SmallVector<Value> outputOperands;
  size_t portIdx = 0;
  auto addBackedge = [&](SmallVector<Backedge> &backedges) -> void {
    const hw::PortInfo *port = outputPorts[portIdx++];
    Backedge backedge = lowerState.edgeBuilder.get(port->type);
    outputOperands.push_back(backedge);
    backedges.push_back(backedge);
  };

  // Function results will come through the Handshake terminator
  size_t numEndResults = modInfo.sizeOutputs() - state.getNumMemOutputs();
  for (size_t i = 0; i < numEndResults; ++i)
    addBackedge(state.endBackedges);
  // Outgoing memory signals will come through memory interfaces
  for (auto &[_, memState] : state.memInterfaces) {
    for (size_t i = 0; i < memState.numOutputs; ++i)
      addBackedge(memState.backedges);
  }

  Operation *outputOp = modOp.getBodyBlock()->getTerminator();
  rewriter.setInsertionPoint(outputOp);
  rewriter.replaceOpWithNewOp<hw::OutputOp>(outputOp, outputOperands);

  // Associate the newly created module to its lowering state object
  lowerState.modState[modOp] = state;
  return success();
}

namespace {
/// Converts the Handshake-level terminator into a HW instance (and an
/// external HW module, potentially). This is special-cased because (1) the
/// operation's IO changes during conversion (essentially copying a subset
/// of its inputs to outputs) and (2) outputs of the HW instance need to
/// connect to the HW-level terminator.
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
  HWEndConverter converter(getContext());

  // Inputs to the module are identical the the original Handshake end
  // operation, plus clock and reset
  for (auto [idx, oprd] : llvm::enumerate(adaptor.getOperands()))
    converter.addInput(modState.endPorts.getInputName(idx), oprd);
  converter.addClkAndRst(parentModOp);

  // The end operation has one input per memory interface in the function
  // which should not be forwarded to its output ports
  unsigned numMemOperands = modState.memInterfaces.size();
  auto numReturnValues = endOp.getNumOperands() - numMemOperands;
  auto returnValOperands = adaptor.getOperands().take_front(numReturnValues);

  // All non-memory inputs to the Handshake end operations should be forwarded
  // to its outputs
  for (auto [idx, oprd] : llvm::enumerate(returnValOperands))
    converter.addOutput(modState.endPorts.getOutputName(idx), oprd.getType());

  hw::InstanceOp instOp =
      converter.convertToInstance(endOp, modState, rewriter);
  return instOp ? success() : failure();
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
  // number of input ports, add those first
  ValueRange blockArgs = parentModOp.getBodyBlock()->getArguments();
  ValueRange memArgs = blockArgs.slice(memState.inputIdx, memState.numInputs);
  SmallVector<hw::ModulePort> memPorts = memState.getMemInputPorts(parentModOp);
  for (auto [port, arg] : llvm::zip_equal(memPorts, memArgs))
    converter.addInput(removePortNamePrefix(port), arg);

  // Adds the operand at the given index to the ports and instance operands.
  auto addInput = [&](size_t idx) -> void {
    Value oprd = operands[idx];
    converter.addInput(memState.portNames.getInputName(idx), oprd);
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
  converter.addClkAndRst(parentModOp);

  // Add output ports corresponding to memory interface results, then those
  // going outside the parent HW module
  for (auto [idx, arg] : llvm::enumerate(memOp->getResults())) {
    StringRef resName = memState.portNames.getOutputName(idx);
    converter.addOutput(resName, channelWrapper(arg.getType()));
  }
  for (const hw::ModulePort &outputPort :
       memState.getMemOutputPorts(parentModOp))
    converter.addOutput(removePortNamePrefix(outputPort), outputPort.type);

  // Create the instance, then replace the original memory interface's result
  // uses with matching results in the instance
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
class ExtModuleConversionPattern : public OpConversionPattern<T> {
public:
  ExtModuleConversionPattern(ChannelTypeConverter &typeConverter,
                             MLIRContext *ctx, LoweringState &lowerState)
      : OpConversionPattern<T>::OpConversionPattern(typeConverter, ctx),
        lowerState(lowerState) {}
  using OpAdaptor = typename T::Adaptor;

  /// Always succeeds in replacing the matched operation with an equivalent
  /// HW instance operation, potentially creating an external HW module in
  /// the process.
  LogicalResult
  matchAndRewrite(T op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  /// Shared lowering state.
  LoweringState &lowerState;
};
} // namespace

template <typename T>
LogicalResult ExtModuleConversionPattern<T>::matchAndRewrite(
    T op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  // We need to instantiate a new external module for that operation; first
  // derive port information for that module, then create it
  HWConverter converter(this->getContext());
  PortNameGenerator portNames(op);

  // Add all operation operands to the inputs
  for (auto [idx, oprd] : llvm::enumerate(adaptor.getOperands()))
    converter.addInput(portNames.getInputName(idx), oprd);
  converter.addClkAndRst(((Operation *)op)->getParentOfType<hw::HWModuleOp>());

  // Add all operation results to the outputs
  for (auto [idx, type] : llvm::enumerate(op->getResultTypes()))
    converter.addOutput(portNames.getOutputName(idx), channelWrapper(type));

  hw::InstanceOp instOp = converter.convertToInstance(op, rewriter);
  return instOp ? success() : failure();
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
/// Replaces Handshake return operations into a sequence of TEHBs, one for each
/// return operand.
struct ReplaceReturnWithTEHB : public OpRewritePattern<handshake::ReturnOp> {
  using OpRewritePattern<handshake::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(returnOp);
    SmallVector<Value> tehbOutputs;
    for (Value oprd : returnOp->getOperands()) {
      auto tehbOp =
          rewriter.create<handshake::TEHBOp>(returnOp.getLoc(), oprd, 1);
      tehbOutputs.push_back(tehbOp.getResult());
    }
    rewriter.replaceOp(returnOp, tehbOutputs);
    return success();
  }
};

/// Conversion pass driver. The conversion only works on modules containing
/// a single handshake function (handshake::FuncOp) at the moment. The
/// function and all the operations it contains are converted to operations
/// from the HW dialect. Dataflow semantics are made explicit with Handshake
/// channels.
class HandshakeToHWPass
    : public dynamatic::impl::HandshakeToHWBase<HandshakeToHWPass> {
public:
  void runDynamaticPass() override {
    // At this level, all operations already have an intrinsic name so we
    // can disable our naming system
    doNotNameOperations();

    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();

    // We only support one function per module
    auto functions = modOp.getOps<handshake::FuncOp>();
    if (functions.empty())
      return;
    if (++functions.begin() != functions.end()) {
      modOp->emitOpError()
          << "we currently only support one handshake function per module";
      return signalPassFailure();
    }
    handshake::FuncOp funcOp = *functions.begin();

    // Check that some preconditions are met before doing anything
    if (failed(verifyExportToRTL(funcOp)))
      return signalPassFailure();

    // Apply some basic IR transformations before actually doing the lowering
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    RewritePatternSet transformPatterns{ctx};
    transformPatterns.add<ReplaceReturnWithTEHB>(ctx);
    if (failed(applyPatternsAndFoldGreedily(modOp, std::move(transformPatterns),
                                            config)))
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
