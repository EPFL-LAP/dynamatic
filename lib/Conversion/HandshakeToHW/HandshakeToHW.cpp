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
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/HW/HWTypes.h"
#include "dynamatic/Dialect/HW/PortImplementation.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/RTL.h"
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
#include <algorithm>
#include <bitset>

using namespace mlir;
using namespace dynamatic;

/// Name of ports representing the clock and reset signals.
static constexpr llvm::StringLiteral CLK_PORT("clk"), RST_PORT("rst");

//===----------------------------------------------------------------------===//
// Internal data-structures
//===----------------------------------------------------------------------===//

namespace {
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
  /// Generates and stores the interface's port names before starting the
  /// conversion, when those are still queryable.
  PortNameGenerator portNames;
  /// Backedges to the containing module's `hw::OutputOp` operation, which
  /// must be set, in order, with the memory interface's results that connect
  /// to the top-level module IO.
  SmallVector<Backedge> backedges;

  /// Needed because we use the class as a value type in a map, which needs to
  /// be default-constructible.
  MemLoweringState() : ports(nullptr), portNames(nullptr) {
    llvm_unreachable("object should never be default-constructed");
  }

  /// Construcst an instance of the object for the provided memory interface.
  /// Input/Output indices refer to the IO of the `hw::HWModuleOp` that is
  /// generated for the Handshake function containing the interface.
  MemLoweringState(handshake::MemoryOpInterface memOp, size_t inputIdx = 0,
                   size_t numInputs = 0, size_t outputIdx = 0,
                   size_t numOutputs = 0)
      : inputIdx(inputIdx), numInputs(numInputs), outputIdx(outputIdx),
        numOutputs(numOutputs), ports(getMemoryPorts(memOp)),
        portNames(memOp){};

  /// Returns the module's input ports that connect to the memory interface.
  SmallVector<hw::ModulePort> getMemInputPorts(hw::HWModuleOp modOp);

  /// Returns the module's output ports that the memory interface connects to.
  SmallVector<hw::ModulePort> getMemOutputPorts(hw::HWModuleOp modOp);
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
  std::string getDiscriminatedModName();

  /// Sets attribute on the external module (corresponding to the operation the
  /// object was constructed with) to tell the backend how to instantiate the
  /// component.
  void setParameters(hw::HWModuleExternOp modOp);

  /// Whether the operations is currently unsupported. Check after construction
  /// and produce a failure if this returns true.
  bool opUnsupported() { return unsupported; }

private:
  /// The operation whose parameters are being identified.
  Operation *op;
  /// The operation's parameters.
  SmallVector<std::pair<std::string, std::string>> parameters;
  /// Whether the operation is unsupported (set during construction).
  bool unsupported = false;

  /// Adds a scalar-type parameter.
  void addUnsigned(const Twine &name, unsigned scalar) {
    parameters.emplace_back(name.str(), RTLUnsignedType::encode(scalar));
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
    parameters.emplace_back(name.str(), RTLStringType::encode(txt.str()));
  };

  /// Returns the bitwidth of a type.
  static unsigned getTypeWidth(Type type);
};
} // namespace

ModuleDiscriminator::ModuleDiscriminator(Operation *op) : op(op) {
  std::string extModName = op->getName().getStringRef().str();
  std::replace(extModName.begin(), extModName.end(), '.', '_');
  extModName += "_";

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
    : op(ports.memOp) {
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
        /// TODO: dummy implemenation, need to connect it to old Chisel
        /// generator which takes JSON inputs
        // Control port count, load port count, store port count, data
        // bitwidth, and address bitwidth
        addUnsigned("NUM_CONTROL", ports.getNumPorts<ControlPort>());
        addUnsigned("NUM_LOAD", ports.getNumPorts<LoadPort>());
        addUnsigned("NUM_STORE", ports.getNumPorts<StorePort>());
        addUnsigned("DATA_WIDTH", ports.dataWidth);
        addUnsigned("ADDR_WIDTH", ports.addrWidth);
      })
      .Default([&](auto) {
        op->emitError() << "This operation cannot be lowered to RTL "
                           "due to a lack of an RTL implementation for it.";
        unsupported = true;
      });
}

std::string ModuleDiscriminator::getDiscriminatedModName() {
  assert(!unsupported && "operation unsupported");
  std::string modName = op->getName().getStringRef().str();
  std::replace(modName.begin(), modName.end(), '.', '_');
  for (auto &[_, paramValue] : parameters)
    modName += "_" + paramValue;
  return modName;
}

void ModuleDiscriminator::setParameters(hw::HWModuleExternOp modOp) {
  assert(!unsupported && "operation unsupported");
  MLIRContext *ctx = op->getContext();

  // The name is used to determine which RTL component to instantiate
  StringRef opName = op->getName().getStringRef();
  modOp->setAttr(RTLMatch::NAME_ATTR, StringAttr::get(ctx, opName));

  // Parameters are used to determine the concrete version of the RTL
  // component to instantiate
  SmallVector<NamedAttribute> paramAttrs;
  llvm::transform(parameters, std::back_inserter(paramAttrs),
                  [&](std::pair<std::string, std::string> &nameAndValue) {
                    return NamedAttribute(
                        StringAttr::get(ctx, nameAndValue.first),
                        StringAttr::get(ctx, nameAndValue.second));
                  });
  modOp->setAttr(RTLMatch::PARAMETERS_ATTR,
                 DictionaryAttr::get(ctx, paramAttrs));
}

unsigned ModuleDiscriminator::getTypeWidth(Type type) {
  if (isa<IntegerType, FloatType>(type))
    return type.getIntOrFloatBitWidth();
  if (isa<NoneType>(type))
    return 0;
  llvm_unreachable("unsupported data type");
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

/// Allows to create/find an external module based on an operation and create an
/// instance of it at the same time.
class ExternalModAndInstanceBuilder {
public:
  /// Constructs from the operation one whishes to convert to an external
  /// module/instance pair.
  ExternalModAndInstanceBuilder(Operation *op)
      : builder(op->getContext()), op(op),
        modOp(op->getParentOfType<hw::HWModuleOp>()) {
    assert(modOp && "no parent HW module for operation");
  };

  /// Adds a value to the list of operands for the future instance, and its type
  /// to the future external module's input port information.
  void addInput(const Twine &name, Value oprd) {
    builder.addInput(name, oprd.getType());
    instOperands.push_back(oprd);
  }

  /// Adds a type to the future external module's output port information.
  void addOutput(const Twine &name, Type type) {
    builder.addOutput(name, type);
  }

  /// Adds clock and reset ports to the future external module's input port
  /// information and operands to the future instance (inherited from the
  /// converted opetation's parent module inputs).
  void addClkAndRst();

  /// Creates the instance corresponding to the converted operation using all
  /// the inputs added so far as operands. The operation is created next to the
  /// converted one, which remains in the IR when the function returns. If no
  /// external module matching the current port information currently exists,
  /// one is added at the bottom of the top-level MLIR module. Returns the
  /// instance on success, nullptr on failure.
  virtual hw::InstanceOp createInstance(const Twine &instName,
                                        ConversionPatternRewriter &rewriter) {
    ModuleDiscriminator discriminator(op);
    return findModAndReplace(discriminator, instName, rewriter, false);
  }

  /// Replaces the converted operation with an equivalent instance using all
  /// the inputs added so far as operands. If no external module matching the
  /// current port information currently exists, one is added at the bottom of
  /// the top-level MLIR module. Returns the instance on success, nullptr on
  /// failure.
  virtual hw::InstanceOp
  replaceWithInstance(const Twine &instName,
                      ConversionPatternRewriter &rewriter) {
    ModuleDiscriminator discriminator(op);
    return findModAndReplace(discriminator, instName, rewriter);
  }

  virtual ~ExternalModAndInstanceBuilder() = default;

protected:
  /// Internal implementation for
  /// `ExternalModAndInstanceBuilder::createInstance` and
  /// `ExternalModAndInstanceBuilder::replaceWithInstance`. The `replace` flag
  /// controls the replacement behavior. Returns the instance on success,
  /// nullptr on failure.
  hw::InstanceOp findModAndReplace(ModuleDiscriminator &discriminator,
                                   const Twine &instName,
                                   ConversionPatternRewriter &rewriter,
                                   bool replace = true);

  /// Underlying module builder to potentially build an external module.
  ModuleBuilder builder;
  /// The operation being converted.
  Operation *op;
  /// The converted operation's parent module, used to get clock and reset
  /// operands from if needed.
  hw::HWModuleOp modOp;
  /// The list of operands that will be used to create the instance
  /// corresponding to the converted operation.
  SmallVector<Value> instOperands;
};

/// Specialization of the external module / instance builder for memory
/// interface operations, which are built through their lowering state object
/// rather than the raw operation.
class MemoryInterfaceBuilder : public ExternalModAndInstanceBuilder {
public:
  /// Constructs the builder from the memory lowering state object associated to
  /// the memory interface to convert.
  MemoryInterfaceBuilder(MemLoweringState &state)
      : ExternalModAndInstanceBuilder(state.ports.memOp), state(state){};

  hw::InstanceOp createInstance(const Twine &instName,
                                ConversionPatternRewriter &rewriter) override {
    ModuleDiscriminator discriminator(state.ports);
    return findModAndReplace(discriminator, instName, rewriter, false);
  }

  hw::InstanceOp
  replaceWithInstance(const Twine &instName,
                      ConversionPatternRewriter &rewriter) override {
    ModuleDiscriminator discriminator(state.ports);
    return findModAndReplace(discriminator, instName, rewriter);
  }

private:
  /// The memory lowering state object associated to the memory interface to
  /// convert.
  MemLoweringState &state;
};
} // namespace

void ExternalModAndInstanceBuilder::addClkAndRst() {
  // Let the parent class add clock and reset to the input ports
  builder.addClkAndRst();

  // Check that the parent module's last port are the clock and reset
  // signals we need for the instance operands
  unsigned numInputs = modOp.getNumInputPorts();
  assert(numInputs >= 2 && "module should have at least clock and reset");
  size_t lastIdx = modOp.getPortIdForInputId(numInputs - 1);
  assert(modOp.getPort(lastIdx - 1).getName() == CLK_PORT && "expected clock");
  assert(modOp.getPort(lastIdx).getName() == RST_PORT && "expected reset");

  // Add clock and reset to the instance's operands
  ValueRange blockArgs = modOp.getBodyBlock()->getArguments();
  instOperands.push_back(blockArgs.drop_back().back());
  instOperands.push_back(blockArgs.back());
}

hw::InstanceOp ExternalModAndInstanceBuilder::findModAndReplace(
    ModuleDiscriminator &discriminator, const Twine &instName,
    ConversionPatternRewriter &rewriter, bool replace) {
  // Fail when the discriminator reports that the operation is unsupported
  if (discriminator.opUnsupported())
    return nullptr;
  if (replace && builder.getNumOutputs() != op->getNumResults()) {
    op->emitError() << "Attempting to replace operation with "
                    << op->getNumResults()
                    << " results; however, external module will have "
                    << builder.getNumOutputs() << " output ports, a mismatch.";
    return nullptr;
  }

  // First retrieve or create the external module matching the operation
  mlir::ModuleOp topLevelModOp = op->getParentOfType<mlir::ModuleOp>();
  std::string extModName = discriminator.getDiscriminatedModName();
  hw::HWModuleExternOp extModOp = findExternMod(topLevelModOp, extModName);
  if (!extModOp) {
    // The external module does not yet exist, create it
    StringAttr modNameAttr = rewriter.getStringAttr(extModName);
    rewriter.setInsertionPointToEnd(topLevelModOp.getBody());
    extModOp = rewriter.create<hw::HWModuleExternOp>(op->getLoc(), modNameAttr,
                                                     builder.getPortInfo());
    discriminator.setParameters(extModOp);
  }

  // Now create the instance corresponding to the external module
  StringAttr instNameAttr = rewriter.getStringAttr(instName);
  rewriter.setInsertionPoint(op);
  hw::InstanceOp instOp = rewriter.create<hw::InstanceOp>(
      op->getLoc(), extModOp, instNameAttr, instOperands);
  if (replace)
    rewriter.replaceOp(op, instOp);
  return instOp;
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
                  auto info =
                      MemLoweringState{memOp, modBuilder.getNumInputs(), 1,
                                       modBuilder.getNumOutputs(), 5};
                  addIO();
                  return info;
                })
            .Case<handshake::LSQOp>([&](handshake::LSQOp lsqOp) {
              if (!lsqOp.isConnectedToMC()) {
                // If the LSQ does not connect to an MC, then it
                // connects directly to top-level module IO
                auto info = MemLoweringState{memOp, modBuilder.getNumInputs(),
                                             1, modBuilder.getNumOutputs(), 5};
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

  // Crerate backege inputs for the module's output operation and associate
  // them to the future operations whose conversion will resolve them
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
  ExternalModAndInstanceBuilder builder(endOp);

  // Inputs to the module are identical the the original Handshake end
  // operation, plus clock and reset
  for (auto [idx, oprd] : llvm::enumerate(adaptor.getOperands()))
    builder.addInput(modState.endPorts.getInputName(idx), oprd);
  builder.addClkAndRst();

  // The end operation has one input per memory interface in the function
  // which should not be forwarded to its output ports
  unsigned numMemOperands = modState.memInterfaces.size();
  auto numReturnValues = endOp.getNumOperands() - numMemOperands;
  auto returnValOperands = adaptor.getOperands().take_front(numReturnValues);

  // All non-memory inputs to the Handshake end operations should be forwarded
  // to its outputs
  for (auto [idx, oprd] : llvm::enumerate(returnValOperands))
    builder.addOutput(modState.endPorts.getOutputName(idx), oprd.getType());

  hw::InstanceOp instOp =
      builder.createInstance(lowerState.namer.getName(endOp), rewriter);
  if (!instOp)
    return failure();

  // Resolve backedges in the module's terminator that are coming from the end
  ValueRange results = instOp.getResults();
  for (auto [backedge, res] : llvm::zip_equal(modState.endBackedges, results))
    backedge.setValue(res);

  rewriter.eraseOp(endOp);
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
  MemoryInterfaceBuilder builder(memState);

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
    builder.addInput(removePortNamePrefix(port), arg);

  // Adds the operand at the given index to the ports and instance operands.
  auto addInput = [&](size_t idx) -> void {
    Value oprd = operands[idx];
    builder.addInput(memState.portNames.getInputName(idx), oprd);
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
  builder.addClkAndRst();

  // Add output ports corresponding to memory interface results, then those
  // going outside the parent HW module
  for (auto [idx, arg] : llvm::enumerate(memOp->getResults())) {
    StringRef resName = memState.portNames.getOutputName(idx);
    builder.addOutput(resName, channelWrapper(arg.getType()));
  }
  for (const hw::ModulePort &outputPort :
       memState.getMemOutputPorts(parentModOp))
    builder.addOutput(removePortNamePrefix(outputPort), outputPort.type);

  // Create the instance, then replace the original memory interface's result
  // uses with matching results in the instance
  hw::InstanceOp instOp =
      builder.createInstance(lowerState.namer.getName(memOp), rewriter);
  if (!instOp)
    return failure();
  size_t numResults = memOp->getNumResults();
  rewriter.replaceOp(memOp, instOp->getResults().take_front(numResults));

  // Resolve backedges in the module's terminator that are coming from the
  // memory interface
  ValueRange results = instOp->getResults().drop_front(numResults);
  for (auto [backedge, res] : llvm::zip_equal(memState.backedges, results))
    backedge.setValue(res);

  return success();
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
  ExternalModAndInstanceBuilder builder(op);
  PortNameGenerator portNames(op);

  // Add all operation operands to the inputs
  for (auto [idx, oprd] : llvm::enumerate(adaptor.getOperands()))
    builder.addInput(portNames.getInputName(idx), oprd);
  if (op.template hasTrait<mlir::OpTrait::HasClock>())
    builder.addClkAndRst();

  // Add all operation results to the outputs
  for (auto [idx, type] : llvm::enumerate(op->getResultTypes()))
    builder.addOutput(portNames.getOutputName(idx), channelWrapper(type));

  hw::InstanceOp instOp =
      builder.replaceWithInstance(lowerState.namer.getName(op), rewriter);
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
