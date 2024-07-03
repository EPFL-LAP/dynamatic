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
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/Backedge.h"
#include "dynamatic/Support/RTL.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <bitset>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <iterator>
#include <string>

#define DEBUG_TYPE "HandshakeToHW"

using namespace mlir;
using namespace dynamatic;

/// Name of ports representing the clock and reset signals.
static constexpr llvm::StringLiteral CLK_PORT("clk"), RST_PORT("rst");

/// NOTE: temporary hack to support external functions as top-level IO.
static unsigned getNumExtInstanceArgs(handshake::EndOp endOp) {
  if (auto attr = endOp->getAttrOfType<IntegerAttr>("hw.funcCutoff"))
    return endOp->getNumOperands() - attr.getUInt();
  return 0;
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
  hw::PortNameGenerator portNames;
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
      : name(name.str()), dataType(memOp.getMemRefType().getElementType()),
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
  llvm::MapVector<handshake::MemoryOpInterface, MemLoweringState> memInterfaces;
  /// Generates and stores the end operations's port names before starting the
  /// conversion, when those are still queryable.
  hw::PortNameGenerator endPorts;
  /// Backedges to the containing module's `hw::OutputOp` operation, which
  /// must be set, in order, with the results of the `hw::InstanceOp`
  /// operation to which the `handshake::EndOp` operation was converted to.
  SmallVector<Backedge> endBackedges;
  /// Backedges to the containing module's `hw::OutputOp` operation, which
  /// must be set, in order, with the original arguments to external function
  /// calls inside the Handshake function.
  SmallVector<Backedge> extInstBackedges;

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

/// Wraps a type into a handshake::ChannelType type.
static handshake::ChannelType channelWrapper(Type t) {
  return TypeSwitch<Type, handshake::ChannelType>(t)
      .Case<handshake::ChannelType>([](auto t) { return t; })
      .Case<NoneType>([](NoneType nt) {
        return handshake::ChannelType::get(
            IntegerType::get(nt.getContext(), 0));
      })
      .Default([](Type t) {
        if (isa<FloatType>(t)) {
          // At the HW/RTL level we treat everything as opaque bitvectors, so we
          // make everything IntegerType's (only the width matters)
          return handshake::ChannelType::get(
              IntegerType::get(t.getContext(), t.getIntOrFloatBitWidth()));
        }
        return handshake::ChannelType::get(t);
      });
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

  /// Adds a boolean-type parameter.
  void addBoolean(const Twine &name, bool value) {
    addParam(name, BoolAttr::get(ctx, value));
    modName += "_" + std::to_string(value);
  };

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

    // Replace all non-alphanumeric characters by an underscore.
    std::string cleanedName = txt.str();
    std::replace_if(
        cleanedName.begin(), cleanedName.end(),
        [](char c) { return !std::isalnum(c); }, '_');

    modName += "_" + cleanedName;
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
      .Case<handshake::InstanceOp>(
          [&](handshake::InstanceOp instOp) { modName = instOp.getModule(); })
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
      .Case<handshake::SharingWrapperOp>(
          [&](handshake::SharingWrapperOp sharingWrapperOp) {
            addBitwidth("DATA_WIDTH", sharingWrapperOp.getDataOperands()[0]);

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
      .Case<handshake::EndOp>([&](handshake::EndOp endOp) {
        // Number of memory inputs and bitwidth (we assume that there is
        // a single function return value due to our current RTL limitation)
        addBitwidth("DATA_WIDTH", op->getOperand(0));
        addUnsigned("NUM_MEMORIES",
                    op->getNumOperands() - getNumExtInstanceArgs(endOp) - 1);
      })
      .Case<handshake::NotOp>([&](handshake::NotOp notOp) {
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
        // There can be at most one of those, and it is a load/store port
        unsigned lsqPort = ports.getNumPorts<LSQLoadStorePort>();

        // Control port count, load port count, store port count, data
        // bitwidth, and address bitwidth
        addUnsigned("NUM_CONTROL", ports.getNumPorts<ControlPort>());
        addUnsigned("NUM_LOAD", ports.getNumPorts<LoadPort>() + lsqPort);
        addUnsigned("NUM_STORE", ports.getNumPorts<StorePort>() + lsqPort);
        addUnsigned("DATA_WIDTH", ports.dataWidth);
        addUnsigned("ADDR_WIDTH", ports.addrWidth);
      })
      .Case<handshake::LSQOp>([&](auto) {
        LSQGenerationInfo genInfo(ports, getUniqueName(op).str());
        std::string lsqName = modName + "_" + genInfo.name;

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

        addString("name", lsqName);
        addBoolean("experimental", true);
        addBoolean("toMC", !ports.interfacePorts.empty());
        addUnsigned("fifoDepth", genInfo.depth);
        addUnsigned("fifoDepth_L", genInfo.depthLoad);
        addUnsigned("fifoDepth_S", genInfo.depthStore);
        addUnsigned("bufferDepth", genInfo.bufferDepth);
        addString("accessType", genInfo.accessType);
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

        // Override the module's name
        modName = lsqName;
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
  assert(hwModOp.getPort(lastIdx - 1).getName() == CLK_PORT &&
         "expected clock");
  assert(hwModOp.getPort(lastIdx).getName() == RST_PORT && "expected reset");

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
  hw::PortNameGenerator portNames(funcOp);

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

  /// NOTE: this is hacky, but only the end arguments that correspond to the
  /// original function results and memory completion signals should go to the
  /// end synchronizer, the rest of the operands go directly to the module's
  /// outputs
  auto endOp = *modOp.getBodyBlock()->getOps<handshake::EndOp>().begin();
  unsigned numExtInstArgs = getNumExtInstanceArgs(endOp);
  size_t numEndResults =
      modInfo.sizeOutputs() - state.getNumMemOutputs() - numExtInstArgs;
  for (size_t i = 0; i < numEndResults; ++i)
    addBackedge(state.endBackedges);
  for (size_t i = 0; i < numExtInstArgs; ++i)
    addBackedge(state.extInstBackedges);

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

LogicalResult ConvertExternalFunc::matchAndRewrite(
    handshake::FuncOp funcOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!funcOp.isExternal())
    return failure();

  StringAttr name = rewriter.getStringAttr(funcOp.getName());
  ModuleBuilder modBuilder(funcOp.getContext());
  hw::PortNameGenerator portNames(funcOp);

  // Add all function outputs to the module
  for (auto [idx, res] : llvm::enumerate(funcOp.getResultTypes()))
    modBuilder.addOutput(portNames.getOutputName(idx), channelWrapper(res));

  // Add all function inputs to the module
  for (auto [idx, type] : llvm::enumerate(funcOp.getArgumentTypes())) {
    if (isa<MemRefType>(type)) {
      return funcOp->emitError()
             << "Memory interfaces are not supported for external "
                "functions";
    }
    modBuilder.addInput(portNames.getInputName(idx), channelWrapper(type));
  }
  modBuilder.addClkAndRst();

  rewriter.setInsertionPoint(funcOp);
  auto modOp = rewriter.replaceOpWithNewOp<hw::HWModuleExternOp>(
      funcOp, name, modBuilder.getPortInfo());
  modOp->setAttr(StringAttr::get(getContext(), RTLRequest::NAME_ATTR),
                 funcOp.getNameAttr());
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

  /// NOTE: this is hacky, but only the end arguments that correspond to the
  /// original function results and memory completion signals should go to the
  /// end synchronizer, the rest of the operands go directly to the module's
  /// outputs
  ValueRange endOperands = adaptor.getOperands();
  unsigned numExtInstArgs = getNumExtInstanceArgs(endOp);
  for (auto [idx, oprd] :
       llvm::enumerate(endOperands.drop_back(numExtInstArgs)))
    converter.addInput(modState.endPorts.getInputName(idx), oprd);
  converter.addClkAndRst(parentModOp);

  // Remaining inputs to the end terminator go directly to the module's outputs
  for (auto [backedge, oprd] : llvm::zip_equal(
           modState.extInstBackedges, endOperands.take_back(numExtInstArgs)))
    backedge.setValue(oprd);

  // The end operation has one input per memory interface in the function
  // which should not be forwarded to its output ports
  unsigned numMemOperands = modState.memInterfaces.size();
  auto numReturnValues = endOperands.size() - numMemOperands - numExtInstArgs;
  auto returnValOperands = endOperands.take_front(numReturnValues);

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
  auto inputModPorts = memState.getMemInputPorts(parentModOp);
  for (auto [port, arg] : llvm::zip_equal(inputModPorts, memArgs))
    converter.addInput(removePortNamePrefix(port), arg);

  auto addInput = [&](size_t idx) -> void {
    Value oprd = operands[idx];
    converter.addInput(memState.portNames.getInputName(idx), oprd);
  };
  auto addOutput = [&](size_t idx) -> void {
    StringRef resName = memState.portNames.getOutputName(idx);
    OpResult res = memOp->getResults()[idx];
    converter.addOutput(resName, channelWrapper(res.getType()));
  };

  // Add all input/output ports corresponding to the memory interface's groups
  for (GroupMemoryPorts &groupPorts : memState.ports.groups) {
    size_t inputIdx = groupPorts.getFirstOperandIndex();
    if (inputIdx != std::string::npos) {
      size_t lastInputIdx = groupPorts.getLastOperandIndex();
      for (; inputIdx <= lastInputIdx; ++inputIdx)
        addInput(inputIdx);
    }

    size_t outputIdx = groupPorts.getFirstResultIndex();
    if (outputIdx != std::string::npos) {
      size_t lastOutputIdx = groupPorts.getLastResultIndex();
      for (; outputIdx <= lastOutputIdx; ++outputIdx)
        addOutput(outputIdx);
    }
  }

  // Add all input/output ports corresponding to the memory interface's ports
  // with other memory interfaces
  for (MemoryPort &memPort : memState.ports.interfacePorts) {
    for (size_t inputIdx : memPort.getOprdIndices())
      addInput(inputIdx);
    for (size_t outputIdx : memPort.getResIndices())
      addOutput(outputIdx);
  }

  // Finish inputs by clock and reset ports
  converter.addClkAndRst(parentModOp);

  // Add output port corresponding to the interface's done signal
  addOutput(memOp->getNumResults() - 1);

  // Add output ports going to the parent HW module
  auto outputModPorts = memState.getMemOutputPorts(parentModOp);
  for (const hw::ModulePort &outputPort : outputModPorts)
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
  hw::PortNameGenerator portNames(op);

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

  extModOp->setAttr(RTLRequest::NAME_ATTR, StringAttr::get(ctx, HW_NAME));
  SmallVector<NamedAttribute> parameters;
  Type i32 = IntegerType::get(ctx, 32, IntegerType::Unsigned);
  parameters.emplace_back(
      StringAttr::get(ctx, "DATA_WIDTH"),
      IntegerAttr::get(i32, memState.dataType.getIntOrFloatBitWidth()));
  parameters.emplace_back(StringAttr::get(ctx, "ADDR_WIDTH"),
                          IntegerAttr::get(i32, memState.ports.addrWidth));
  extModOp->setAttr(RTLRequest::PARAMETERS_ATTR,
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
/// Replaces Handshake return operations into a sequence of TEHBs, one for
/// each return operand.
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
      if (failed(verifyAllIndexConcretized(funcOp))) {
        funcOp.emitError() << "Lowering to HW requires that all index "
                              "types in the IR have "
                              "been concretized."
                           << ERR_RUN_CONCRETIZATION;
        return signalPassFailure();
      }
    }

    if (failed(runPreprocessing()))
      return signalPassFailure();

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
    patterns.insert<ConvertFunc, ConvertEnd, ConvertMemInterface>(
        typeConverter, ctx, lowerState);
    patterns.insert<
        ConvertInstance, ConvertToHWInstance<handshake::OEHBOp>,
        ConvertToHWInstance<handshake::TEHBOp>,
        ConvertToHWInstance<handshake::ConditionalBranchOp>,
        ConvertToHWInstance<handshake::BranchOp>,
        ConvertToHWInstance<handshake::MergeOp>,
        ConvertToHWInstance<handshake::ControlMergeOp>,
        ConvertToHWInstance<handshake::MuxOp>,
        ConvertToHWInstance<handshake::SourceOp>,
        ConvertToHWInstance<handshake::ConstantOp>,
        ConvertToHWInstance<handshake::SinkOp>,
        ConvertToHWInstance<handshake::ForkOp>,
        ConvertToHWInstance<handshake::LazyForkOp>,
        ConvertToHWInstance<handshake::MCLoadOp>,
        ConvertToHWInstance<handshake::LSQLoadOp>,
        ConvertToHWInstance<handshake::MCStoreOp>,
        ConvertToHWInstance<handshake::LSQStoreOp>,
        ConvertToHWInstance<handshake::SharingWrapperOp>,
        // Arith operations
        ConvertToHWInstance<arith::AddFOp>, ConvertToHWInstance<arith::AddIOp>,
        ConvertToHWInstance<arith::AndIOp>,
        ConvertToHWInstance<arith::BitcastOp>,
        ConvertToHWInstance<arith::CeilDivSIOp>,
        ConvertToHWInstance<arith::CeilDivUIOp>,
        ConvertToHWInstance<arith::CmpFOp>, ConvertToHWInstance<arith::CmpIOp>,
        ConvertToHWInstance<arith::DivFOp>, ConvertToHWInstance<arith::DivSIOp>,
        ConvertToHWInstance<arith::DivUIOp>, ConvertToHWInstance<arith::ExtFOp>,
        ConvertToHWInstance<arith::ExtSIOp>,
        ConvertToHWInstance<arith::ExtUIOp>,
        ConvertToHWInstance<arith::FPToSIOp>,
        ConvertToHWInstance<arith::FPToUIOp>,
        ConvertToHWInstance<arith::FloorDivSIOp>,
        ConvertToHWInstance<arith::IndexCastOp>,
        ConvertToHWInstance<arith::IndexCastUIOp>,
        ConvertToHWInstance<arith::MulFOp>, ConvertToHWInstance<arith::MulIOp>,
        ConvertToHWInstance<arith::NegFOp>, ConvertToHWInstance<arith::OrIOp>,
        ConvertToHWInstance<arith::RemFOp>, ConvertToHWInstance<arith::RemSIOp>,
        ConvertToHWInstance<arith::RemUIOp>,
        ConvertToHWInstance<arith::SelectOp>,
        ConvertToHWInstance<arith::SIToFPOp>,
        ConvertToHWInstance<arith::ShLIOp>, ConvertToHWInstance<arith::ShRSIOp>,
        ConvertToHWInstance<arith::ShRUIOp>, ConvertToHWInstance<arith::SubFOp>,
        ConvertToHWInstance<arith::SubIOp>,
        ConvertToHWInstance<arith::TruncFOp>,
        ConvertToHWInstance<arith::TruncIOp>,
        ConvertToHWInstance<arith::UIToFPOp>,
        ConvertToHWInstance<arith::XOrIOp>>(typeConverter,
                                            funcOp->getContext());

    // Everything must be converted to operations in the hw dialect
    ConversionTarget target(*ctx);
    target.addLegalOp<hw::HWModuleOp, hw::HWModuleExternOp, hw::InstanceOp,
                      hw::OutputOp, mlir::UnrealizedConversionCastOp>();
    target.addIllegalDialect<handshake::HandshakeDialect, arith::ArithDialect,
                             memref::MemRefDialect>();

    if (failed(applyPartialConversion(modOp, target, std::move(patterns))))
      return signalPassFailure();

    // Create memory wrappers around all hardware modules
    for (auto [circuitOp, _] : lowerState.modState)
      createWrapper(circuitOp, lowerState, builder);

    // At this level all operations already have an intrinsic name so we can
    // disable our naming system
    doNotNameOperations();
  }

private:
  /// Runs a pre-processiong greedy pattern rewriter on the input module to get
  /// rid of constructs that have no hardware mapping.
  LogicalResult runPreprocessing() {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns{ctx};
    patterns.add<ReplaceReturnWithTEHB>(ctx);
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    return applyPatternsAndFoldGreedily(modOp, std::move(patterns), config);
  }

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
