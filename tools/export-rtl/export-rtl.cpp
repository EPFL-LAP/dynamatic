//===- export-rtl.cpp - Export RTL from HW-level IR -------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exports RTL from HW-level IR. Files corresponding to internal and external
// modules are written inside a provided output directory (which is created if
// necessary).
//
//===----------------------------------------------------------------------===//
#include "dynamatic/Conversion/HandshakeToHW.h"
#include "dynamatic/Dialect/HW/HWDialect.h"
#include "dynamatic/Dialect/HW/HWOpInterfaces.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/RTL/RTL.h"
#include "dynamatic/Support/System.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

static cl::OptionCategory mainCategory("Tool options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> outputDir(cl::Positional, cl::Required,
                                      cl::desc("<output directory>"),
                                      cl::cat(mainCategory));

static cl::opt<std::string> dynamaticPath("dynamatic-path", cl::Optional,
                                          cl::desc("<path to Dynamatic>"),
                                          cl::init("."), cl::cat(mainCategory));

static cl::opt<HDL>
    hdl("hdl", cl::Optional, cl::desc("<hdl to use>"), cl::init(HDL::VHDL),
        cl::values(clEnumValN(HDL::VHDL, "vhdl", "VHDL"),
                   clEnumValN(HDL::VERILOG, "verilog", "Verilog"),
                   clEnumValN(HDL::SMV, "smv", "SMV")),
        cl::cat(mainCategory));

static cl::list<std::string>
    rtlConfigs(cl::Positional, cl::OneOrMore,
               cl::desc("<RTL configuration files...>"), cl::cat(mainCategory));

namespace llvm {
const std::pair<std::string, bool> EMPTY_KEY = {"EMPTY_KEY", false};
const std::pair<std::string, bool> TOMBSTONE_KEY = {"TOMBSTONE_KEY", false};

template <>
struct DenseMapInfo<std::pair<std::string, bool>> {
  static inline std::pair<std::string, bool> getEmptyKey() { return EMPTY_KEY; }

  static inline std::pair<std::string, bool> getTombstoneKey() {
    return TOMBSTONE_KEY;
  }

  static unsigned getHashValue(const std::pair<std::string, bool> &p) {
    return std::hash<std::string>{}(p.first) ^
           (static_cast<unsigned>(p.second) << 1);
  }

  static bool isEqual(const std::pair<std::string, bool> &LHS,
                      const std::pair<std::string, bool> &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

namespace {

using FGenComp =
    std::function<LogicalResult(const RTLRequest &, hw::HWModuleExternOp)>;

/// Aggregates information useful during RTL export. This is to avoid passing
/// many arguments to a bunch of functions.
struct ExportInfo {
  /// The top-level MLIR module.
  mlir::ModuleOp modOp;
  /// The RTL configuration parsed from JSON-formatted files.
  RTLConfiguration &config;
  /// Output directory (without trailing separators).
  StringRef outputPath;

  /// Maps every external hardware module in the IR to its corresponding
  /// heap-allocated match according to the RTL configuration.
  mlir::DenseMap<hw::HWModuleExternOp, RTLMatch *> externals;

  /// Creates export information for the given module and RTL configuration.
  ExportInfo(mlir::ModuleOp modOp, RTLConfiguration &config,
             StringRef outputPath)
      : modOp(modOp), config(config), outputPath(outputPath) {};

  /// Associates every external hardware module to its match according to the
  /// RTL configuration and concretizes each of them inside the output
  /// directory. Fails if any external module does not have a match in the RTL
  /// configuration; succeeds otherwise.
  LogicalResult concretizeExternalModules();

  /// Deallocates all of our RTL matches.
  ~ExportInfo() {
    for (auto [_, match] : externals)
      delete match;
  }
};
} // namespace

LogicalResult ExportInfo::concretizeExternalModules() {
  std::set<std::string> modules;

  FGenComp concretizeComponent =
      [&](const RTLRequest &request,
          hw::HWModuleExternOp extOp) -> LogicalResult {
    // Try to find a matching component
    RTLMatch *match = config.getMatchingComponent(request);
    if (!match) {
      return emitError(request.loc)
             << "Failed to find matching RTL component for external module";
    }
    // If match is not external, it must be freed when function returns
    // we don't like this solution, feel free to propose a better one
    std::unique_ptr<RTLMatch> matchUniquePtr;
    if (extOp)
      externals[extOp] = match;
    else
      matchUniquePtr.reset(match);

    // No need to do anything if a module with the same name already exists
    StringRef concreteModName = match->getConcreteModuleName();
    if (auto [_, isNew] = modules.insert(concreteModName.str()); !isNew)
      return success();

    // First generate dependencies recursively...
    for (StringRef dep : match->component->getDependencies()) {
      RTLDependencyRequest dependencyRequest(dep, request.loc);
      if (failed(concretizeComponent(dependencyRequest, nullptr)))
        return failure();
    }

    // Parameter analysis
    // TODO: Do this at the HW-level analysis
    if (extOp)
      match->registerParameters(extOp);

    // ...then generate the component itself
    return match->concretize(request, dynamaticPath, outputPath);
  };

  for (hw::HWModuleExternOp extOp : modOp.getOps<hw::HWModuleExternOp>()) {
    RTLRequestFromHWModule request(extOp);
    if (failed(concretizeComponent(request, extOp)))
      return failure();
  }
  return success();
}

namespace {

using FGetValueName = std::function<StringRef(Value)>;

/// Aggregates all data one is likely to need when writing a module's RTL
/// implementation to a file on disk.
struct WriteModData {
  /// Module being exported to RTL.
  hw::HWModuleOp modOp;
  /// Stream on which to write the RTL implementation.
  raw_indented_ostream &os;
  /// Maps SSA values to the name of a corresponding RTL signal.
  llvm::MapVector<Value, std::string> signals;
  /// List of SSA values feeding the module's output ports.
  SmallVector<Value> outputs;

  /// Constructs from the module being exported and from the stream to write the
  /// RTL implementation to.
  WriteModData(hw::HWModuleOp modOp, raw_indented_ostream &os)
      : modOp(modOp), os(os) {}

  using PortDeclarationWriter = void (*)(const llvm::Twine &name, PortType dir,
                                         std::optional<unsigned> type,
                                         raw_indented_ostream &os);

  /// Writes the module's port declarations.
  void writeIO(PortDeclarationWriter writeDeclaration, StringRef sep, HDL hdl);

  using SignalDeclarationWriter = void (*)(const llvm::Twine &name,
                                           std::optional<unsigned> type,
                                           raw_indented_ostream &os);

  /// Writes the module's internal signal declarations.
  void writeSignalDeclarations(SignalDeclarationWriter writeDeclaration);

  using SignalAssignmentWriter = void (*)(const llvm::Twine &dst,
                                          const llvm::Twine &src,
                                          raw_indented_ostream &os);

  /// Writes signal assignments between the top-level module's outputs and
  /// the implementation's internal signals.
  void writeSignalAssignments(SignalAssignmentWriter writeAssignment);

  /// Returns a function that maps SSA values to the name of the internal RTl
  /// signal that corresponds to it. The returned function asserts if the value
  /// is unknown.
  FGetValueName getSignalNameFunc() const {
    return [&](Value val) -> StringRef {
      const auto *name = signals.find(val);
      assert(name != signals.end() && "unknown SSA value");
      return name->second;
    };
  }
};

/// Abstract RTL module writer. Contains common logic and data-structures for
/// writing RTL, regardless of the HDL.
class RTLWriter {
public:
  /// The name of a port along with its type (`std::nullopt` for a 1-bit signal,
  /// an unsigned number `n` for a vector of size `n + 1`)
  using IOPort = std::pair<std::string, std::optional<unsigned>>;

  /// Groups all of the top-level entity's IO as pairs of signal name and signal
  /// type. Inputs and outputs are also separated.
  struct EntityIO {
    /// The entity's inputs.
    std::vector<IOPort> inputs;
    /// The entity's outputs.
    std::vector<IOPort> outputs;

    /// Constructs entity IO for the hardware module.
    EntityIO(hw::HWModuleOp modOp);
  };

  /// Represents a component's port; its name and a boolean indicating whether
  /// the port is part of a vector in RTL.
  using Port = std::pair<std::string, bool>;

  /// Associates every port of a component to internal signal names that should
  /// connect to it.
  using IOMap = llvm::MapVector<Port, SmallVector<std::string /*signal name*/>>;

  /// Suffixes for specfic signal types.
  static constexpr StringLiteral VALID_SUFFIX = StringLiteral("_valid"),
                                 READY_SUFFIX = StringLiteral("_ready");

  /// Export information (external modules must have already been concretized).
  ExportInfo &exportInfo;
  // The HDL in which to write the module.
  HDL hdl;

  /// Creates the RTL writer.
  RTLWriter(ExportInfo &exportInfo, HDL hdl)
      : exportInfo(exportInfo), hdl(hdl) {};

  /// Writes the RTL implementation of the module to the output stream. On
  /// failure, the RTL implementation should be considered invalid and/or
  /// incomplete.
  virtual LogicalResult write(hw::HWModuleOp modOp,
                              raw_indented_ostream &os) const = 0;

  /// Associates each SSA value inside the module to internal module signals.
  /// Fails when encoutering an unsupported operation inside the module;
  /// succeeds otherwise.
  virtual LogicalResult createInternalSignals(WriteModData &data) const;

  void fillIOMappings(hw::InstanceOp instOp, const FGetValueName &getValueName,
                      IOMap &mappings) const;

  /// Default destructor.
  virtual ~RTLWriter() = default;

protected:
  using FGetTypedSignalName = std::function<Port(StringRef, SignalType)>;
  using FGetSignalName = std::function<Port(StringRef)>;

  virtual void
  constructIOMappings(hw::InstanceOp instOp, hw::HWModuleLike modOp,
                      const FGetValueName &getValueName,
                      const FGetTypedSignalName &getTypedSignalName,
                      const FGetSignalName &getSignalName,
                      IOMap &mappings) const;
};
} // namespace

using PortMapWriter = void (*)(const RTLWriter::Port &port,
                               ArrayRef<std::string> signalNames,
                               raw_indented_ostream &os);

/// Writes IO mappings to the output stream. Provided with a correct
/// port-mapping function, this works for VHDL, Verilog, and SMV.
/// Mapping examples:
/// - VHDL    (..., ins => constant0_outs, ...)
/// - Verilog (..., .ins (constant0_outs), ...)
/// - SMV     (..., constant0.outs, ...)
static void writeIOMap(const RTLWriter::IOMap &mappings,
                       PortMapWriter writePortMap, const std::string &separator,
                       raw_indented_ostream &os) {
  size_t numIOLeft = mappings.size();
  for (auto &[port, signalNames] : mappings) {
    writePortMap(port, signalNames, os);
    if (--numIOLeft != 0)
      os << separator;
  }
  if (separator.find('\n') != std::string::npos)
    os << '\n';
}

/// Returns the type's inclusive array bound.
static unsigned convertToInclusiveArrayBound(IntegerType intType) {
  return intType.getWidth() - 1;
}

/// Returns the type's inclusive array bound.
static unsigned convertToInclusiveArrayBound(ChannelType channelType) {
  return channelType.getDataBitWidth() - 1;
}

/// Returns the type's "raw" RTL type.
static std::optional<unsigned> getRawType(IntegerType intType) {
  unsigned dataWidth = intType.getIntOrFloatBitWidth();
  assert(dataWidth != 0 && "0-width signals are not allowed");
  if (dataWidth == 1)
    return std::nullopt;
  return convertToInclusiveArrayBound(intType);
}

/// Returns the hardare module the hardware instance is of.
static hw::HWModuleLike getHWModule(hw::InstanceOp instOp) {
  mlir::ModuleOp modOp = instOp->getParentOfType<mlir::ModuleOp>();
  assert(modOp && "cannot find top-level MLIR module");
  Operation *lookup = modOp.lookupSymbol(instOp.getModuleName());
  assert(lookup && "symbol does not reference an operation");
  return cast<hw::HWModuleLike>(lookup);
}

/// Returns the internal signal name for a specific signal type.
static std::string getInternalSignalName(StringRef baseName, SignalType signalType) {
  switch (signalType) {
  case (SignalType::DATA):
    return baseName.str();
  case (SignalType::VALID):
    return baseName.str() + "_valid";
  case (SignalType::READY):
    return baseName.str() + "_ready";
  }
}

/// Returns the internal signal name for an extra channel signal.
static std::string getExtraSignalName(StringRef baseName,
                                      const ExtraSignal &extra) {
  return baseName.str() + "_" + extra.name.str();
}

void WriteModData::writeIO(PortDeclarationWriter writeDeclaration,
                           StringRef sep, HDL hdl) {
  const RTLWriter::EntityIO entityIO(modOp);
  size_t numIOLeft =
      hdl != HDL::SMV
          ? entityIO.inputs.size() + entityIO.outputs.size()
          // In SMV we ignore clk and rst signals and all the output ports
          : entityIO.inputs.size() - 2;

  auto writePortsDir = [&](const std::vector<RTLWriter::IOPort> &io,
                           PortType dir) -> void {
    for (auto &[portName, portType] : io) {
      const bool toPrint = hdl != HDL::SMV
                               ? true
                               : portName != dynamatic::hw::CLK_PORT &&
                                     portName != dynamatic::hw::RST_PORT;
      if (toPrint) {
        writeDeclaration(portName, dir, portType, os);
        if (--numIOLeft != 0)
          os << sep;
        if (hdl != HDL::SMV)
          os << "\n";
      }
    }
  };

  writePortsDir(entityIO.inputs, PortType::IN);
  if (hdl != HDL::SMV)
    writePortsDir(entityIO.outputs, PortType::OUT);
}

void WriteModData::writeSignalDeclarations(
    SignalDeclarationWriter writeDeclaration) {
  auto isNotBlockArg = [](auto valAndName) -> bool {
    return !isa<BlockArgument>(valAndName.first);
  };

  auto addValidReady = [&](StringRef name) -> void {
    writeDeclaration(getInternalSignalName(name, SignalType::VALID),
                     std::nullopt, os);
    writeDeclaration(getInternalSignalName(name, SignalType::READY),
                     std::nullopt, os);
  };
  auto addExtraSignals = [&](StringRef name,
                             ArrayRef<ExtraSignal> extraSignals) -> void {
    for (const ExtraSignal &extra : extraSignals) {
      writeDeclaration(
          getExtraSignalName(name, extra),
          convertToInclusiveArrayBound(cast<IntegerType>(extra.type)), os);
    }
  };

  for (auto &valueAndName : make_filter_range(signals, isNotBlockArg)) {
    llvm::TypeSwitch<Type, void>(valueAndName.first.getType())
        .Case<ChannelType>([&](ChannelType channelType) {
          writeDeclaration(
              getInternalSignalName(valueAndName.second, SignalType::DATA),
              convertToInclusiveArrayBound(channelType), os);
          addValidReady(valueAndName.second);
          addExtraSignals(valueAndName.second, channelType.getExtraSignals());
        })
        .Case<ControlType>([&](auto type) {
          addValidReady(valueAndName.second);
          addExtraSignals(valueAndName.second, type.getExtraSignals());
        })
        .Case<IntegerType>([&](IntegerType intType) {
          writeDeclaration(valueAndName.second, getRawType(intType), os);
        });
  }
}

void WriteModData::writeSignalAssignments(
    SignalAssignmentWriter writeAssignment) {
  auto addValidReady = [&](StringRef name, StringRef signal) -> void {
    writeAssignment(signal + RTLWriter::VALID_SUFFIX,
                    name + RTLWriter::VALID_SUFFIX, os);

    std::string signalName = signal.str();
    // Wehn connecting the ready signal of the top level module a proper
    // internal signal is needed. This signal needs to be named
    // component_name_port_name instead of component_name.port_name
    std::replace(signalName.begin(), signalName.end(), '.', '_');
    writeAssignment(name + RTLWriter::READY_SUFFIX,
                    signalName + RTLWriter::READY_SUFFIX, os);
  };
  auto addExtraSignals = [&](StringRef name, StringRef signal,
                             ArrayRef<ExtraSignal> extraSignals) -> void {
    for (const ExtraSignal &extra : extraSignals) {
      std::string srcName = getExtraSignalName(signal, extra);
      std::string dstName = getExtraSignalName(name, extra);
      if (!extra.downstream)
        std::swap(srcName, dstName);
      writeAssignment(srcName, dstName, os);
    }
  };

  for (auto valAndName : llvm::zip(outputs, modOp.getOutputNamesStr())) {
    Value val = std::get<0>(valAndName);
    StringRef name = std::get<1>(valAndName).strref();
    StringRef signal = signals[val];
    llvm::TypeSwitch<Type, void>(val.getType())
        .Case<ChannelType>([&](ChannelType channelType) {
          writeAssignment(signal, name, os);
          addValidReady(name, signal);
          addExtraSignals(name, signal, channelType.getExtraSignals());
        })
        .Case<ControlType>([&](auto type) {
          addValidReady(name, signal);
          addExtraSignals(name, signal, type.getExtraSignals());
        })
        .Case<IntegerType>([&](IntegerType intType) {
          writeAssignment(signals[val], name, os);
        });
  }
}

RTLWriter::EntityIO::EntityIO(hw::HWModuleOp modOp) {
  auto addValidAndReady = [&](StringRef portName, std::vector<IOPort> &down,
                              std::vector<IOPort> &up) -> void {
    down.emplace_back(getInternalSignalName(portName, SignalType::VALID),
                      std::nullopt);
    up.emplace_back(getInternalSignalName(portName, SignalType::READY),
                    std::nullopt);
  };
  auto addExtraSignals = [&](StringRef portName, std::vector<IOPort> &down,
                             std::vector<IOPort> &up,
                             ArrayRef<ExtraSignal> extraSignals) -> void {
    for (const ExtraSignal &extra : extraSignals) {
      std::vector<IOPort> &portsDir = extra.downstream ? down : up;
      IntegerType ty = cast<IntegerType>(extra.type);
      portsDir.emplace_back(getExtraSignalName(portName, extra),
                            convertToInclusiveArrayBound(ty));
    }
  };

  auto addPortType = [&](Type portType, StringRef portName,
                         std::vector<IOPort> &down, std::vector<IOPort> &up) {
    llvm::TypeSwitch<Type, void>(portType)
        .Case<ChannelType>([&](ChannelType channelType) {
          down.emplace_back(getInternalSignalName(portName, SignalType::DATA),
                            convertToInclusiveArrayBound(channelType));
          addValidAndReady(portName, down, up);
          addExtraSignals(portName, down, up, channelType.getExtraSignals());
        })
        .Case<ControlType>([&](auto type) {
          addValidAndReady(portName, down, up);
          addExtraSignals(portName, down, up, type.getExtraSignals());
        })
        .Case<IntegerType>([&](IntegerType intType) {
          down.emplace_back(portName, getRawType(intType));
        });
  };

  for (auto [arg, portAttr] : llvm::zip_equal(
           modOp.getBodyBlock()->getArguments(), modOp.getInputNamesStr()))
    addPortType(arg.getType(), portAttr.str(), inputs, outputs);

  for (auto [resType, portAttr] :
       llvm::zip_equal(modOp.getOutputTypes(), modOp.getOutputNamesStr()))
    addPortType(resType, portAttr.str(), outputs, inputs);
}

LogicalResult RTLWriter::createInternalSignals(WriteModData &data) const {
  // Create signal names for all block arguments
  for (auto [arg, name] :
       llvm::zip_equal(data.modOp.getBodyBlock()->getArguments(),
                       data.modOp.getInputNamesStr()))
    data.signals[arg] = name.strref();

  // Create signal names for all operation results
  for (Operation &op : data.modOp.getBodyBlock()->getOperations()) {
    LogicalResult res =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<hw::InstanceOp>([&](hw::InstanceOp instOp) {
              // Retrieve the module referenced by the instance
              hw::HWModuleLike refModOp = getHWModule(instOp);
              std::string prefix = instOp.getInstanceName().str() + "_";

              // Associate each instance result with a signal name
              for (auto [res, name] : llvm::zip_equal(
                       instOp->getResults(), refModOp.getOutputNamesStr()))
                data.signals[res] = prefix + name.str();
              return success();
            })
            .Case<hw::OutputOp>([&](hw::OutputOp outputOp) {
              llvm::copy(outputOp->getOperands(),
                         std::back_inserter(data.outputs));
              return success();
            })
            .Default([&](auto) {
              return op.emitOpError()
                     << "Unsupported operation type within module";
            });
    if (failed(res))
      return failure();
  }
  return success();
}

void RTLWriter::constructIOMappings(
    hw::InstanceOp instOp, hw::HWModuleLike modOp,
    const FGetValueName &getValueName,
    const FGetTypedSignalName &getTypedSignalName,
    const FGetSignalName &getSignalName, IOMap &mappings) const {
  auto addValidAndReady = [&](StringRef port, StringRef signal) -> void {
    mappings[getTypedSignalName(port, SignalType::VALID)].push_back(
        getInternalSignalName(signal, SignalType::VALID));
    mappings[getTypedSignalName(port, SignalType::READY)].push_back(
        getInternalSignalName(signal, SignalType::READY));
  };
  auto addExtraSignals = [&](StringRef port, StringRef signal,
                             ArrayRef<ExtraSignal> extraSignals) -> void {
    for (const ExtraSignal &extra : extraSignals) {
      mappings[{getExtraSignalName(port, extra), false}].push_back(
          getExtraSignalName(signal, extra));
    }
  };

  auto addPortType = [&](Type portType, StringRef port, StringRef signal) {
    llvm::TypeSwitch<Type, void>(portType)
        .Case<ChannelType>([&](ChannelType channelType) {
          mappings[getTypedSignalName(port, SignalType::DATA)].push_back(
              getInternalSignalName(signal, SignalType::DATA));
          addValidAndReady(port, signal);
          addExtraSignals(port, signal, channelType.getExtraSignals());
        })
        .Case<ControlType>([&](auto type) {
          addValidAndReady(port, signal);
          addExtraSignals(port, signal, type.getExtraSignals());
        })
        .Case<IntegerType>([&](IntegerType intType) {
          mappings[getSignalName(port)].push_back(signal.str());
        });
  };

  auto ins = llvm::zip_equal(instOp.getOperands(), modOp.getInputNamesStr());
  for (auto [oprd, portAttr] : ins)
    addPortType(oprd.getType(), portAttr.str(), getValueName(oprd));

  auto outs = llvm::zip_equal(instOp.getResults(), modOp.getOutputNamesStr());
  for (auto [oprd, portAttr] : outs)
    addPortType(oprd.getType(), portAttr.str(), getValueName(oprd));
}

void RTLWriter::fillIOMappings(hw::InstanceOp instOp,
                               const FGetValueName &getValueName,
                               IOMap &mappings) const {
  hw::HWModuleLike modOp = getHWModule(instOp);
  if (auto extModOp = dyn_cast<hw::HWModuleExternOp>(modOp.getOperation())) {
    const RTLMatch &match = *exportInfo.externals.at(extModOp);
    FGetTypedSignalName getTypedSignalName = [&](auto port, auto type) -> Port {
      return match.component->getRTLPortName(port, type, ::hdl);
    };
    FGetSignalName getSignalName = [&](auto port) -> Port {
      return match.component->getRTLPortName(port, ::hdl);
    };
    constructIOMappings(instOp, modOp, getValueName, getTypedSignalName,
                        getSignalName, mappings);
  } else {
    FGetTypedSignalName getTypedSignalName = [&](auto port, auto type) -> Port {
      return {getInternalSignalName(port, type), false};
    };
    FGetSignalName getSignalName = [&](auto port) -> Port {
      return {port.str(), false};
    };
    constructIOMappings(instOp, modOp, getValueName, getTypedSignalName,
                        getSignalName, mappings);
  }
}

//===----------------------------------------------------------------------===//
// VHDLWriter
//===----------------------------------------------------------------------===//

namespace {

struct VHDLWriter : public RTLWriter {
  using RTLWriter::RTLWriter;

  /// Architecture name for VHDL modules we create directly.
  static constexpr StringLiteral ARCH_NAME = "behavioral";

  /// Writes the VHDL implementation of the module to the output stream.
  LogicalResult write(hw::HWModuleOp modOp,
                      raw_indented_ostream &os) const override;

private:
  static std::string getVHDLType(std::optional<unsigned> width) {
    if (width)
      return "std_logic_vector(" + std::to_string(*width) + " downto 0)";
    return "std_logic";
  }

  /// Writes all module instantiations inside the entity's architecture.
  void writeModuleInstantiations(WriteModData &data) const;
};

} // namespace

LogicalResult VHDLWriter::write(hw::HWModuleOp modOp,
                                raw_indented_ostream &os) const {
  WriteModData data(modOp, os);
  if (failed(createInternalSignals(data)))
    return failure();

  // Generic imports
  os << "library ieee;\n";
  os << "use ieee.std_logic_1164.all;\n";
  os << "use ieee.numeric_std.all;\n\n";

  // Declare the entity
  os << "entity " << modOp.getSymName() << " is\n";
  os.indent();
  os << "port (\n";
  os.indent();

  data.writeIO(
      [](const llvm::Twine &name, PortType dir, std::optional<unsigned> type,
         raw_indented_ostream &os) {
        os << name << " : ";
        switch (dir) {
        case PortType::IN:
          os << "in ";
          break;
        case PortType::OUT:
          os << "out ";
          break;
        }
        os << getVHDLType(type);
      },
      ";", hdl);

  // Close the entity declaration
  os.unindent();
  os << ");\n";
  os.unindent();
  os << "end entity;\n\n";

  // Open the entity's architecture
  os << "architecture " << ARCH_NAME << " of " << modOp.getSymName()
     << " is\n\n";
  os.indent();

  data.writeSignalDeclarations([](const llvm::Twine &name,
                                  std::optional<unsigned> type,
                                  raw_indented_ostream &os) {
    os << "signal " << name << " : " << getVHDLType(type) << ";\n";
  });
  os.unindent();
  os << "\nbegin\n\n";
  os.indent();

  // Architecture implementation
  data.writeSignalAssignments(
      [](const llvm::Twine &src, const llvm::Twine &dst,
         raw_indented_ostream &os) { os << dst << " <= " << src << ";\n"; });
  os << "\n";
  writeModuleInstantiations(data);

  // Close the entity's architecture
  os.unindent();
  os << "end architecture;\n";
  return success();
}

void VHDLWriter::writeModuleInstantiations(WriteModData &data) const {
  using KeyValuePair = std::pair<StringRef, StringRef>;

  for (hw::InstanceOp instOp : data.modOp.getOps<hw::InstanceOp>()) {
    HDL hdl(HDL::VHDL);
    std::string moduleName;
    std::string archName;
    SmallVector<KeyValuePair> genericParams;

    llvm::TypeSwitch<Operation *, void>(getHWModule(instOp).getOperation())
        .Case<hw::HWModuleOp>([&](hw::HWModuleOp hwModOp) {
          moduleName = hwModOp.getSymName();
          archName = ARCH_NAME;
        })
        .Case<hw::HWModuleExternOp>([&](hw::HWModuleExternOp extModOp) {
          const RTLMatch &match = *exportInfo.externals.at(extModOp);
          hdl = match.component->getHDL();
          moduleName = match.getConcreteModuleName();
          archName = match.getConcreteArchName();
          genericParams = match.getGenericParameterValues().takeVector();
        })
        .Default([&](auto) { llvm_unreachable("unknown module type"); });

    raw_indented_ostream &os = data.os;
    // Declare the instance
    os << instOp.getInstanceName() << " : entity work." << moduleName;
    if (hdl == HDL::VHDL)
      os << "(" << archName << ")";

    // Write generic parameters if there are any
    if (!genericParams.empty()) {
      os << " generic map(";

      for (auto [_, val] : ArrayRef<KeyValuePair>{genericParams}.drop_back())
        os << val << ", ";
      os << genericParams.back().second << ")";
    }
    os << "\n";

    PortMapWriter writePortMap = [](const Port &port,
                                    ArrayRef<std::string> signalNames,
                                    raw_indented_ostream &os) {
      assert(!signalNames.empty() && "no signal name associated to port");
      if (!port.second) {
        assert(signalNames.size() == 1 &&
               "more than one signal for non-array port");
        os << port.first << " => " << signalNames.front();
        return;
      }

      ArrayRef<std::string> signals(signalNames);
      for (auto [idx, sig] : llvm::enumerate(signals.drop_back()))
        os << port.first << "(" << idx << ") => " << sig << ",\n";
      os << port.first << "(" << std::to_string(signals.size() - 1) << ") => "
         << signals.back();
    };

    // Write IO mappings
    os.indent();
    os << "port map(\n";
    os.indent();
    IOMap mappings;
    fillIOMappings(instOp, data.getSignalNameFunc(), mappings);
    writeIOMap(mappings, writePortMap, ",\n", os);
    os.unindent();
    os << ");\n";
    os.unindent();
    os << "\n";
  }
}
//===----------------------------------------------------------------------===//
// VerilogWriter
//===----------------------------------------------------------------------===//

namespace {

struct VerilogWriter : public RTLWriter {
  using RTLWriter::RTLWriter;

  /// Writes the Verilog implementation of the module to the output stream.
  LogicalResult write(hw::HWModuleOp modOp,
                      raw_indented_ostream &os) const override;

private:
  static std::string getVerilogType(std::optional<unsigned> width) {
    if (width)
      return "[" + std::to_string(*width) + ":0]";
    return "";
  }

  /// Writes all module instantiations inside the entity's architecture.
  void writeModuleInstantiations(WriteModData &data) const;
};

} // namespace

LogicalResult VerilogWriter::write(hw::HWModuleOp modOp,
                                   raw_indented_ostream &os) const {
  WriteModData data(modOp, os);
  if (failed(createInternalSignals(data)))
    return failure();

  os << "module " << modOp.getSymName() << "(\n";

  os.indent();
  data.writeIO(
      [](const llvm::Twine &name, PortType dir, std::optional<unsigned> type,
         raw_indented_ostream &os) {
        switch (dir) {
        case PortType::IN:
          os << "input ";
          break;
        case PortType::OUT:
          os << "output ";
          break;
        }
        os << getVerilogType(type) << " " << name;
      },
      ",", hdl);
  os.unindent();

  os << ");\n";
  os.indent();

  data.writeSignalDeclarations([](const llvm::Twine &name,
                                  std::optional<unsigned> type,
                                  raw_indented_ostream &os) {
    os << "wire " << getVerilogType(type) << " " << name << ";\n";
  });

  os << "\n";
  data.writeSignalAssignments([](const llvm::Twine &src, const llvm::Twine &dst,
                                 raw_indented_ostream &os) {
    os << "assign " << dst << " = " << src << ";\n";
  });
  os << "\n";
  writeModuleInstantiations(data);

  os.unindent();
  os << "endmodule\n";
  return success();
}

void VerilogWriter::writeModuleInstantiations(WriteModData &data) const {
  using KeyValuePair = std::pair<StringRef, StringRef>;

  for (hw::InstanceOp instOp : data.modOp.getOps<hw::InstanceOp>()) {
    HDL hdl(dynamatic::HDL::VERILOG);
    std::string moduleName;
    SmallVector<KeyValuePair> genericParams;

    llvm::TypeSwitch<Operation *, void>(getHWModule(instOp).getOperation())
        .Case<hw::HWModuleOp>(
            [&](hw::HWModuleOp hwModOp) { moduleName = hwModOp.getSymName(); })
        .Case<hw::HWModuleExternOp>([&](hw::HWModuleExternOp extModOp) {
          const RTLMatch &match = *exportInfo.externals.at(extModOp);
          hdl = match.component->getHDL();
          moduleName = match.getConcreteModuleName();
          genericParams = match.getGenericParameterValues().takeVector();
        })
        .Default([&](auto) { llvm_unreachable("unknown module type"); });

    raw_indented_ostream &os = data.os;
    os << moduleName << " ";

    // Write generic parameters if there are any
    if (!genericParams.empty()) {
      os << "#(";
      for (auto [name, val] : ArrayRef<KeyValuePair>{genericParams}.drop_back())
        os << "." << name << "(" << val << "), ";
      auto [name, val] = genericParams.back();
      os << "." << name << "(" << val << ")) ";
    }
    os << instOp.getInstanceName() << "(\n";

    PortMapWriter writePortMap = [](const Port &port,
                                    ArrayRef<std::string> signalNames,
                                    raw_indented_ostream &os) {
      assert(!signalNames.empty() && "no signal name associated to port");

      os << "." << port.first << " (";
      if (!port.second) {
        assert(signalNames.size() == 1 &&
               "more than one signal for non-array port");
        os << signalNames.front() << ")";
        return;
      }

      // Signals are stored in increasing port index order but the port
      // mapping expects them in the opposite order, so we reverse the list
      os << "{";
      ArrayRef<std::string> signals(signalNames);
      for (StringRef sig : llvm::reverse(signals.drop_front()))
        os << sig << ", ";
      os << signals.front();
      os << "})";
    };

    // Write IO mappings
    os.indent();
    IOMap mappings;
    fillIOMappings(instOp, data.getSignalNameFunc(), mappings);
    writeIOMap(mappings, writePortMap, ",\n", os);
    os.unindent();
    os << ");\n\n";
  }
}

//===----------------------------------------------------------------------===//
// SMVWriter
//===----------------------------------------------------------------------===//

namespace {

struct SMVWriter : public RTLWriter {
  using RTLWriter::RTLWriter;

  /// Writes the SMV implementation of the module to the output stream.
  LogicalResult write(hw::HWModuleOp modOp,
                      raw_indented_ostream &os) const override;

private:
  /// creates the mapping between ports internal signals
  void constructIOMappings(hw::InstanceOp instOp, hw::HWModuleLike modOp,
                           const FGetValueName &getValueName,
                           const FGetTypedSignalName &getTypedSignalName,
                           const FGetSignalName &getSignalName,
                           IOMap &mappings) const override;

  /// Creates internal signals that in SMV directly reference the unit to be
  /// connected: component_name.port_name
  LogicalResult createInternalSignals(WriteModData &data) const override;
  /// Writes all module instantiations inside the entity's architecture.
  void writeModuleInstantiations(WriteModData &data) const;
  /// Writes the preprocessor directives to include the external modules
  void writeIncludes(WriteModData &data) const;
};

} // namespace

void SMVWriter::writeIncludes(WriteModData &data) const {
  std::unordered_set<std::string> incNames;
  for (auto m : exportInfo.externals) {
    incNames.insert(m.getSecond()->getConcreteModuleName().str());
  }
  for (const std::string &name : incNames) {
    data.os << "#include \"" << name << ".smv\"\n";
  }
}

LogicalResult SMVWriter::createInternalSignals(WriteModData &data) const {
  // Create the names of all the input signals of the RTL module: i.e., the
  // block arguments of the HW MLIR
  for (auto [arg, name] :
       llvm::zip_equal(data.modOp.getBodyBlock()->getArguments(),
                       data.modOp.getInputNamesStr())) {
    data.signals[arg] = name.strref();
  }

  // Create signal names for all operation results
  for (Operation &op : data.modOp.getBodyBlock()->getOperations()) {
    const LogicalResult res =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<hw::InstanceOp>([&](hw::InstanceOp instOp) {
              // Retrieve the module referenced by the instance
              hw::HWModuleLike refModOp = getHWModule(instOp);
              std::string prefix = instOp.getInstanceName().str() + ".";

              // Associate each instance result with a signal name
              for (auto [res, name] : llvm::zip_equal(
                       instOp->getResults(), refModOp.getOutputNamesStr())) {
                data.signals[res] = prefix + name.str();
              }
              return success();
            })
            .Case<hw::OutputOp>([&](hw::OutputOp outputOp) {
              // Create the names of all the output signals of the RTL module:
              // i.e., the input signals connected to the terminator op
              // (hw::OutputOp).
              llvm::copy(outputOp->getOperands(),
                         std::back_inserter(data.outputs));
              return success();
            })
            .Default([&](auto) {
              return op.emitOpError()
                     << "Unsupported operation type within module";
            });
    if (failed(res))
      return failure();
  }
  return success();
}

void SMVWriter::constructIOMappings(
    hw::InstanceOp instOp, hw::HWModuleLike modOp,
    const FGetValueName &getValueName,
    const FGetTypedSignalName &getTypedSignalName,
    const FGetSignalName &getSignalName, IOMap &mappings) const {

  auto addValid = [&](StringRef port, StringRef signal) -> void {
    mappings[getTypedSignalName(port, SignalType::VALID)].push_back(
        getInternalSignalName(signal, SignalType::VALID));
  };
  auto addReady = [&](StringRef port, OpResult res) -> void {
    // To get the name of the ready signal we can't use the inetrnal signal
    // name. We need to get the user of the signal, and search the name of the
    // corresponding port.
    // Example: unit_tx -> [channel] -> unit_rx
    //          The ready signal needs the name of the user unit_rx.ins_ready
    auto *userOp = *res.getUsers().begin();
    unsigned operandIndex = 0;
    for (unsigned i = 0; i < userOp->getNumOperands(); ++i) {
      auto operand = userOp->getOperand(i);
      if (operand == res) {
        operandIndex = i;
        break;
      }
    }
    std::string instName;
    bool instFound = false;
    if (auto argNamesAttr =
            userOp->getAttrOfType<mlir::StringAttr>("instanceName")) {
      instName = argNamesAttr.getValue().str();
      instFound = true;
    }
    std::string argName;
    bool argFound = false;
    if (auto argNamesAttr =
            userOp->getAttrOfType<mlir::ArrayAttr>("argNames")) {
      if (operandIndex < argNamesAttr.size()) {
        if (auto strAttr =
                argNamesAttr[operandIndex].dyn_cast<mlir::StringAttr>()) {
          argName = strAttr.getValue().str();
          argFound = true;
        }
      }
    }

    const std::string signal = instName + "." + argName;

    if (instFound && argFound)
      mappings[getTypedSignalName(port, SignalType::READY)].push_back(
          getInternalSignalName(signal, SignalType::READY));
    else {
      // Connect the ready signal of the top level module: a proper internal
      // signal is needed, with name component_name_port_name instead of
      // component_name.port_name
      std::string signalName = getValueName(res).str();
      std::replace(signalName.begin(), signalName.end(), '.', '_');
      mappings[getTypedSignalName(port, SignalType::READY)].push_back(
          getInternalSignalName(signalName, SignalType::READY));
    }
  };
  auto addExtraSignals = [&](StringRef port, StringRef signal,
                             ArrayRef<ExtraSignal> extraSignals) -> void {
    for (const ExtraSignal &extra : extraSignals) {
      mappings[{getExtraSignalName(port, extra), false}].push_back(
          getExtraSignalName(signal, extra));
    }
  };

  auto addInPortType = [&](Type portType, StringRef port, StringRef signal) {
    llvm::TypeSwitch<Type, void>(portType)
        .Case<ChannelType>([&](ChannelType channelType) {
          mappings[getTypedSignalName(port, SignalType::DATA)].push_back(
              getInternalSignalName(signal, SignalType::DATA));
          addValid(port, signal);
          addExtraSignals(port, signal, channelType.getExtraSignals());
        })
        .Case<ControlType>([&](auto type) {
          addValid(port, signal);
          addExtraSignals(port, signal, type.getExtraSignals());
        })
        .Case<IntegerType>([&](IntegerType intType) {
          if (signal.str() != dynamatic::hw::CLK_PORT &&
              signal.str() != dynamatic::hw::RST_PORT)
            mappings[getSignalName(port)].push_back(signal.str());
        });
  };

  auto addOutPortType = [&](Type portType, StringRef port, OpResult op) {
    llvm::TypeSwitch<Type, void>(portType)
        .Case<ChannelType>([&](ChannelType channelType) { addReady(port, op); })
        .Case<ControlType>([&](ControlType type) { addReady(port, op); });
  };

  auto ins = llvm::zip_equal(instOp.getOperands(), modOp.getInputNamesStr());
  for (auto [oprd, portAttr] : ins)
    addInPortType(oprd.getType(), portAttr.str(), getValueName(oprd));

  auto outs = llvm::zip_equal(instOp.getResults(), modOp.getOutputNamesStr());
  for (auto [oprd, portAttr] : outs)
    addOutPortType(oprd.getType(), portAttr.str(), oprd);
}

LogicalResult SMVWriter::write(hw::HWModuleOp modOp,
                               raw_indented_ostream &os) const {
  WriteModData data(modOp, os);
  if (failed(createInternalSignals(data)))
    return failure();

  writeIncludes(data);
  os << "\n\n";

  os << "MODULE " << modOp.getSymName() << " (";

  data.writeIO([](const llvm::Twine &name, PortType dir,
                  std::optional<unsigned> type,
                  raw_indented_ostream &os) { os << name; },
               ", ", hdl);

  os << ")\n\n";

  writeModuleInstantiations(data);

  os << "\n// output\n";
  data.writeSignalAssignments([](const llvm::Twine &src, const llvm::Twine &dst,
                                 raw_indented_ostream &os) {
    os << "DEFINE " << dst << " := " << src << ";\n";
  });

  return success();
}

void SMVWriter::writeModuleInstantiations(WriteModData &data) const {
  using KeyValuePair = std::pair<StringRef, StringRef>;

  for (hw::InstanceOp instOp : data.modOp.getOps<hw::InstanceOp>()) {
    HDL hdl(HDL::SMV);
    std::string moduleName;
    std::string archName;
    SmallVector<KeyValuePair> genericParams;

    llvm::TypeSwitch<Operation *, void>(getHWModule(instOp).getOperation())
        .Case<hw::HWModuleOp>(
            [&](hw::HWModuleOp hwModOp) { moduleName = hwModOp.getSymName(); })
        .Case<hw::HWModuleExternOp>([&](hw::HWModuleExternOp extModOp) {
          const RTLMatch &match = *exportInfo.externals.at(extModOp);
          hdl = match.component->getHDL();
          moduleName = match.getConcreteModuleName();
          archName = match.getConcreteArchName();
          genericParams = match.getGenericParameterValues().takeVector();
        })
        .Default([&](auto) { llvm_unreachable("unknown module type"); });

    raw_indented_ostream &os = data.os;
    // Declare the instance
    os << "VAR " << instOp.getInstanceName() << " : " << moduleName;

    os << "(";

    PortMapWriter writePortMap = [](const Port &port,
                                    ArrayRef<std::string> signalNames,
                                    raw_indented_ostream &os) {
      assert(!signalNames.empty() && "no signal name associated to port");
      if (!port.second) {
        assert(signalNames.size() == 1 &&
               "more than one signal for non-array port");
        os << signalNames.front();
        return;
      }

      ArrayRef<std::string> signals(signalNames);
      for (const std::string &sig : signals.drop_back())
        os << sig << ", ";
      os << signals.back();
    };

    // Write IO mappings
    IOMap mappings;
    fillIOMappings(instOp, data.getSignalNameFunc(), mappings);
    writeIOMap(mappings, writePortMap, ", ", os);
    os << ");\n";
  }
}

/// Writes the RTL implementation corresponding to the hardware module in a
/// file named like the module inside the output directory. Fails if the
/// output file cannot be created or if the module cannot be converted to RTL;
/// succeeds otherwise.
static LogicalResult writeModule(RTLWriter &writer, hw::HWModuleOp modOp) {
  // Open the file in which we will create the module, it is named like the
  // module itself
  std::string filepath =
      writer.exportInfo.outputPath.str() + sys::path::get_separator().str() +
      modOp.getSymName().str() + "." + getHDLExtension(hdl).str();

  std::error_code ec;
  llvm::raw_fd_ostream fileStream(filepath, ec);
  if (ec.value() != 0) {
    return modOp->emitOpError() << "Failed to create file for export @ \""
                                << filepath << "\": " << ec.message();
  }
  raw_indented_ostream os(fileStream);
  return writer.write(modOp, os);
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Exports a VHDL design corresponding to an input HW-level IR. "
      "JSON-formatted RTL configuration files encode the procedure to "
      "instantiate/generate external HW modules present in the input IR.");

  // Make sure the output path does not end in a file separator
  StringRef outputPath = sys::path::removeTrailingSeparators(outputDir);

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFilename
                 << "': " << error.message() << "\n";
    return 1;
  }

  // We only need the Handshake and HW dialects
  MLIRContext context;
  context.loadDialect<handshake::HandshakeDialect, hw::HWDialect>();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  // Parse the RTL configuration files
  RTLConfiguration config;
  for (StringRef filepath : rtlConfigs) {
    if (failed(config.addComponentsFromJSON(filepath)))
      return 1;
  }

  // Create the (potentially nested) output directory
  if (auto ec = sys::fs::create_directories(outputPath); ec.value() != 0) {
    llvm::errs() << "Failed to create output directory\n" << ec.message();
    return 1;
  }

  // Generate/Pull all external modules into the output directory
  ExportInfo info(*modOp, config, outputPath);
  if (failed(info.concretizeExternalModules()))
    return 1;

  // Create an RTL writer
  RTLWriter *writer;
  switch (hdl) {
  case HDL::VHDL:
    writer = new VHDLWriter(info, hdl);
    break;
  case HDL::VERILOG:
    writer = new VerilogWriter(info, hdl);
    break;
  case HDL::SMV:
    writer = new SMVWriter(info, hdl);
    break;
  }

  // Write each module's RTL implementation to a separate file
  for (hw::HWModuleOp hwModOp : modOp->getOps<hw::HWModuleOp>()) {
    if (failed(writeModule(*writer, hwModOp))) {
      delete writer;
      return 1;
    }
  }

  delete writer;
  return 0;
}
