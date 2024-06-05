//===- export-vhdl.cpp - Export VHDL from HW-level IR -----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Experimental tool that exports VHDL from HW-level IR. Files corresponding to
// internal and external modules are written inside a provided output directory
// (which is created if necessary).
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/HW/HWDialect.h"
#include "dynamatic/Dialect/HW/HWOpInterfaces.h"
#include "dynamatic/Dialect/HW/HWOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/RTL.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
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
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;

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

static cl::list<std::string>
    rtlConfigs(cl::Positional, cl::OneOrMore,
               cl::desc("<RTL configuration files...>"), cl::cat(mainCategory));

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
  /// Maps every external hardware module in the IR to its corresponding match
  /// according to the RTL configuration.
  mlir::DenseMap<hw::HWModuleExternOp, RTLMatch> externals;

  /// Creates export information for the given module and RTL configuration.
  ExportInfo(mlir::ModuleOp modOp, RTLConfiguration &config)
      : modOp(modOp), config(config){};

  /// Associates every external hardware module to its match according to the
  /// RTL configuration and concretizes each of them inside the output
  /// directory. Fails if any external module does not have a match in the RTL
  /// configuration; succeeds otherwise.
  LogicalResult concretizeExternalModules();
};
} // namespace

LogicalResult ExportInfo::concretizeExternalModules() {
  std::set<std::string> entities;

  FGenComp concretizeComponent =
      [&](const RTLRequest &request,
          hw::HWModuleExternOp extOp) -> LogicalResult {
    // Try to find a matching component
    std::optional<RTLMatch> match = config.getMatchingComponent(request);
    if (!match) {
      return emitError(request.loc)
             << "Failed to find matching RTL component for external module";
    }
    if (extOp)
      externals[extOp] = *match;

    // No need to do anything if an entity with the same name already exists
    if (auto [_, isNew] = entities.insert(match->getConcreteModuleName().str());
        !isNew)
      return success();

    // First generate dependencies...
    for (StringRef dep : match->component->getDependencies()) {
      RTLRequest dependencyRequest(dep, request.loc);
      if (failed(concretizeComponent(dependencyRequest, nullptr)))
        return failure();
    }

    // ...then generate the component itself
    return match->concretize(request, dynamaticPath, outputDir);
  };

  for (hw::HWModuleExternOp extOp : modOp.getOps<hw::HWModuleExternOp>()) {
    RTLRequestFromHWModule request(extOp);
    if (failed(concretizeComponent(request, extOp)))
      return failure();
  }

  return success();
}

namespace {

/// A value of channel type.
using ChannelValue = TypedValue<handshake::ChannelType>;

/// Maps a value to its internal signal name.
using FGetValueName = std::function<StringRef(Value)>;

/// Aggregates all data one is likely to need when writing a module's RTL
/// implementation to a file on disk.
struct WriteData {
  /// Module being exported to RTL.
  hw::HWModuleOp modOp;
  /// Stream on which to write the RTL implementation.
  raw_indented_ostream &os;

  /// Maps channel-typed SSA values to the base name of corresponding RTL
  /// signals (data + valid + ready).
  llvm::MapVector<ChannelValue, std::string> dataflowSignals;
  /// Maps non-channel-typed SSA values to the name of a corresponding RTL
  /// signal.
  llvm::MapVector<Value, std::string> signals;
  /// List of SSA values feeding the module's output ports.
  SmallVector<Value> outputs;

  /// Constructs from the module being exported and from the stream to write the
  /// RTL implementation to.
  WriteData(hw::HWModuleOp modOp, raw_indented_ostream &os)
      : modOp(modOp), os(os) {}

  /// Returns a function that maps SSA values to the name of the internal RTl
  /// signal that corresponds to it. The returned function asserts if the value
  /// is unknown.
  FGetValueName getSignalNameFunc() const {
    return [&](Value val) -> StringRef {
      if (auto channelVal = dyn_cast<ChannelValue>(val)) {
        const auto *name = dataflowSignals.find(channelVal);
        assert(name != dataflowSignals.end() && "unknown SSA value");
        return name->second;
      }
      const auto *name = signals.find(val);
      assert(name != signals.end() && "unknown SSA value");
      return name->second;
    };
  }
};

/// RTL file writer. This is currently specialized for VHDL but will in the
/// future probably become abstract with child classes to write VHDL or Verilog.
class RTLWriter {
public:
  /// A pair of strings.
  using IOPort = std::pair<std::string, std::string>;

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

  /// An association between a component port name and an internal signal name.
  struct PortMapPair {
    /// The component port name.
    std::string portName;
    /// The internal signal name.
    std::string signalName;

    /// Creates the association.
    PortMapPair(StringRef portName, StringRef signalName)
        : portName(portName), signalName(signalName){};

    // NOLINTBEGIN(*unused-function)
    // clang-tidy does not see that sorting a vector of `PortMapPair`'s requires
    // the < operator

    /// Makes the struct sortable in a vector.
    friend bool operator<(const PortMapPair &lhs, const PortMapPair &rhs) {
      return lhs.portName < rhs.portName;
    }

    // NOLINTEND(*unused-function)
  };

  /// Associates every port of a component to an internal signal name. This is
  /// useful when instantiating components.
  struct IOMap {
    /// Associations for the component's input ports.
    std::vector<PortMapPair> inputs;
    /// Associations for the component's output ports.
    std::vector<PortMapPair> outputs;

    /// Constructs from the hardware instance of interest, the export
    /// information, and a function to retrieve the internal signal name
    /// corresponding to each SSA value.
    IOMap(hw::InstanceOp instOp, const ExportInfo &info,
          const FGetValueName &getValueName);

  private:
    using FGetTypedSignalName =
        std::function<std::string(StringRef, SignalType)>;
    using FGetSignalName = std::function<std::string(StringRef)>;

    /// Internally called by the constructor to initialize the list of inputs
    /// and outputs.
    void construct(hw::InstanceOp instOp, hw::HWModuleLike modOp,
                   const FGetValueName &getValueName,
                   const FGetTypedSignalName &getTypedSignalName,
                   const FGetSignalName &getSignalName);
  };

  /// Suffixes for specfic signal types.
  static constexpr StringLiteral VALID_SUFFIX = StringLiteral("_valid"),
                                 READY_SUFFIX = StringLiteral("_ready");

  /// Export information (external modules must have already been concretized).
  ExportInfo &exportInfo;

  /// Creates the RTL writer.
  RTLWriter(ExportInfo &exportInfo) : exportInfo(exportInfo){};

  /// Writes the RTL implementation of the module to the output stream. On
  /// failure, the RTL implementation should be considered invalid and/or
  /// incomplete.
  virtual LogicalResult write(hw::HWModuleOp modOp,
                              raw_indented_ostream &os) const = 0;

  /// Default destructor.
  virtual ~RTLWriter() = default;
};
} // namespace

/// Returns the VHDL data type correspnding to the MLIR type.
static std::string getDataType(Type type) {
  unsigned dataWidth = type.getIntOrFloatBitWidth();
  if (dataWidth == 1)
    return "std_logic";
  unsigned signalWidth = dataWidth == 0 ? 0 : dataWidth - 1;
  return "std_logic_vector(" + std::to_string(signalWidth) + " downto 0)";
}

/// Returns the VHDL data type correspnding to the channel type.
static std::string getChannelDataType(handshake::ChannelType type) {
  unsigned dataWidth = type.getDataType().getIntOrFloatBitWidth();
  unsigned signalWidth = dataWidth == 0 ? 0 : dataWidth - 1;
  return "std_logic_vector(" + std::to_string(signalWidth) + " downto 0)";
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
static std::string getInternalSignalName(StringRef baseName, SignalType type) {
  switch (type) {
  case (SignalType::DATA):
    return baseName.str();
  case (SignalType::VALID):
    return baseName.str() + "_valid";
  case (SignalType::READY):
    return baseName.str() + "_ready";
  }
}

RTLWriter::EntityIO::EntityIO(hw::HWModuleOp modOp) {

  for (auto [arg, portAttr] : llvm::zip_equal(
           modOp.getBodyBlock()->getArguments(), modOp.getInputNamesStr())) {
    std::string port = portAttr.str();
    if (auto channelType = dyn_cast<handshake::ChannelType>(arg.getType())) {
      inputs.emplace_back(getInternalSignalName(port, SignalType::DATA),
                          getChannelDataType(channelType));
      inputs.emplace_back(getInternalSignalName(port, SignalType::VALID),
                          "std_logic");
      outputs.emplace_back(getInternalSignalName(port, SignalType::READY),
                           "std_logic");
    } else {
      inputs.emplace_back(port, getDataType(arg.getType()));
    }
  }

  for (auto [resType, portAttr] :
       llvm::zip_equal(modOp.getOutputTypes(), modOp.getOutputNamesStr())) {
    std::string port = portAttr.str();
    if (auto channelType = dyn_cast<handshake::ChannelType>(resType)) {
      outputs.emplace_back(getInternalSignalName(port, SignalType::DATA),
                           getChannelDataType(channelType));
      outputs.emplace_back(getInternalSignalName(port, SignalType::VALID),
                           "std_logic");
      inputs.emplace_back(getInternalSignalName(port, SignalType::READY),
                          "std_logic");
    } else {
      outputs.emplace_back(port, getDataType(resType));
    }
  }
}

RTLWriter::IOMap::IOMap(hw::InstanceOp instOp, const ExportInfo &info,
                        const FGetValueName &getValueName) {
  hw::HWModuleLike modOp = getHWModule(instOp);
  if (auto extModOp = dyn_cast<hw::HWModuleExternOp>(modOp.getOperation())) {
    const RTLMatch &match = info.externals.at(extModOp);
    FGetTypedSignalName getTypedSignalName = [&](auto port, auto type) {
      return match.component->getRTLPortName(port, type);
    };
    FGetSignalName getSignalName = [&](auto port) {
      return match.component->getRTLPortName(port);
    };
    construct(instOp, modOp, getValueName, getTypedSignalName, getSignalName);
  } else {
    FGetTypedSignalName getTypedSignalName = [&](auto port, auto type) {
      return getInternalSignalName(port, type);
    };
    FGetSignalName getSignalName = [&](auto port) { return port.str(); };
    construct(instOp, modOp, getValueName, getTypedSignalName, getSignalName);
  }
}

void RTLWriter::IOMap::construct(hw::InstanceOp instOp, hw::HWModuleLike modOp,
                                 const FGetValueName &getValueName,
                                 const FGetTypedSignalName &getTypedSignalName,
                                 const FGetSignalName &getSignalName) {

  auto ins = llvm::zip_equal(instOp.getOperands(), modOp.getInputNamesStr());
  for (auto [oprd, portAttr] : ins) {
    std::string port = portAttr.str();
    if (isa<ChannelValue>(oprd)) {
      StringRef signal = getValueName(oprd);
      inputs.emplace_back(getTypedSignalName(port, SignalType::DATA),
                          getInternalSignalName(signal, SignalType::DATA));
      inputs.emplace_back(getTypedSignalName(port, SignalType::VALID),
                          getInternalSignalName(signal, SignalType::VALID));
      outputs.emplace_back(getTypedSignalName(port, SignalType::READY),
                           getInternalSignalName(signal, SignalType::READY));
    } else {
      inputs.emplace_back(getSignalName(port), getValueName(oprd));
    }
  }

  auto outs = llvm::zip_equal(instOp.getResults(), modOp.getOutputNamesStr());
  for (auto [oprd, portAttr] : outs) {
    std::string port = portAttr.str();
    if (isa<ChannelValue>(oprd)) {
      StringRef signal = getValueName(oprd);
      outputs.emplace_back(getTypedSignalName(port, SignalType::DATA),
                           getInternalSignalName(signal, SignalType::DATA));
      outputs.emplace_back(getTypedSignalName(port, SignalType::VALID),
                           getInternalSignalName(signal, SignalType::VALID));
      inputs.emplace_back(getTypedSignalName(port, SignalType::READY),
                          getInternalSignalName(signal, SignalType::READY));
    } else {
      outputs.emplace_back(getSignalName(port), getValueName(oprd));
    }
  }
}

namespace {

struct VHDLWriter : public RTLWriter {
  using RTLWriter::RTLWriter;

  /// Architecture name for VHDL modules we create directly.
  static constexpr StringLiteral ARCH_NAME = "behavioral";

  /// Writes the VHDL implementation of the module to the output stream.
  LogicalResult write(hw::HWModuleOp modOp,
                      raw_indented_ostream &os) const override;

private:
  /// Associates each SSA value inside the module to internal module signals.
  /// Fails when encoutering an unsupported operation inside the module;
  /// succeeds otherwise.
  LogicalResult createInternalSignals(WriteData &data) const;

  /// Writes the entry's IO ports.
  void writeEntityIO(WriteData &data) const;

  /// Writes all internal signal declarations inside the entity's architecture.
  void writeInternalSignals(WriteData &data) const;

  /// Writes signal assignments betwween the top-level entity's outputs and the
  /// architecture's internal signals.
  void writeSignalAssignments(WriteData &data) const;

  /// Writes all module instantiations inside the entity's architecture.
  void writeModuleInstantiations(WriteData &data) const;

  /// Writes IO mappings for a component instantiation.
  void writeIOMap(hw::InstanceOp instOp, WriteData &data) const;
};

} // namespace

LogicalResult VHDLWriter::write(hw::HWModuleOp modOp,
                                raw_indented_ostream &os) const {
  WriteData data(modOp, os);
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

  writeEntityIO(data);

  // Close the entity declaration
  os.unindent();
  os << ");\n";
  os.unindent();
  os << "end entity;\n\n";

  // Open the entity's architecture
  os << "architecture " << ARCH_NAME << " of " << modOp.getSymName()
     << " is\n\n";
  os.indent();

  // Declare signals inside the architecture
  writeInternalSignals(data);

  os.unindent();
  os << "\nbegin\n\n";
  os.indent();

  // Architecture implementation
  writeSignalAssignments(data);
  os << "\n";
  writeModuleInstantiations(data);

  // Close the entity's architecture
  os.unindent();
  os << "end architecture;\n";
  return success();
}

LogicalResult VHDLWriter::createInternalSignals(WriteData &data) const {

  // Create signal names for all block arguments
  for (auto [arg, name] :
       llvm::zip_equal(data.modOp.getBodyBlock()->getArguments(),
                       data.modOp.getInputNamesStr())) {
    if (auto channelArg = dyn_cast<ChannelValue>(arg))
      data.dataflowSignals[channelArg] = name.strref();
    else
      data.signals[arg] = name.strref();
  }

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
                       instOp->getResults(), refModOp.getOutputNamesStr())) {
                const Twine &sigName = prefix + name.strref();
                if (auto channelRes = dyn_cast<ChannelValue>(res))
                  data.dataflowSignals[channelRes] = sigName.str();
                else
                  data.signals[res] = sigName.str();
              }
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

void VHDLWriter::writeEntityIO(WriteData &data) const {
  const EntityIO entityIO(data.modOp);
  size_t numIOLeft = entityIO.inputs.size() + entityIO.outputs.size();

  raw_indented_ostream &os = data.os;
  auto writePortsDir = [&](const std::vector<IOPort> &io, StringRef dir,
                           StringRef name) -> void {
    if (!io.empty())
      os << "-- " << name << "\n";
    for (auto &[portName, portType] : io) {
      os << portName << " : " << dir << " " << portType;
      if (--numIOLeft != 0)
        os << ";";
      os << "\n";
    }
  };

  writePortsDir(entityIO.inputs, "in", "inputs");
  writePortsDir(entityIO.outputs, "out", "outputs");
}

void VHDLWriter::writeInternalSignals(WriteData &data) const {
  auto isNotBlockArg = [](auto valAndName) -> bool {
    return !isa<BlockArgument>(valAndName.first);
  };

  raw_indented_ostream &os = data.os;
  for (auto [value, name] :
       make_filter_range(data.dataflowSignals, isNotBlockArg)) {
    os << "signal " << getInternalSignalName(name, SignalType::DATA) << " : "
       << getChannelDataType(value.getType()) << ";\n";
    os << "signal " << getInternalSignalName(name, SignalType::VALID)
       << " : std_logic;\n";
    os << "signal " << getInternalSignalName(name, SignalType::READY)
       << " : std_logic;\n";
  }

  for (auto [value, name] : make_filter_range(data.signals, isNotBlockArg))
    os << "signal " << name << " : " << getDataType(value.getType()) << ";\n";
}

void VHDLWriter::writeSignalAssignments(WriteData &data) const {
  raw_indented_ostream &os = data.os;
  os << "-- entity outputs\n";
  for (auto [val, outputName] :
       llvm::zip(data.outputs, data.modOp.getOutputNamesStr())) {
    StringRef name = outputName.strref();
    if (auto channelVal = dyn_cast<ChannelValue>(val)) {
      StringRef signal = data.dataflowSignals[channelVal];
      os << name << " <= " << signal << ";\n";
      os << name << VALID_SUFFIX << " <= " << signal << VALID_SUFFIX << ";\n";
      os << signal << READY_SUFFIX << " <= " << name << READY_SUFFIX << ";\n";
    } else {
      os << name << " <= " << data.signals[val] << ";\n";
    }
  }
}

void VHDLWriter::writeModuleInstantiations(WriteData &data) const {
  for (hw::InstanceOp instOp : data.modOp.getOps<hw::InstanceOp>()) {
    RTLComponent::HDL hdl(dynamatic::RTLComponent::HDL::VHDL);
    std::string moduleName;
    std::string archName;
    SmallVector<StringRef, 4> genericParams;

    llvm::TypeSwitch<Operation *, void>(getHWModule(instOp).getOperation())
        .Case<hw::HWModuleOp>([&](hw::HWModuleOp hwModOp) {
          moduleName = hwModOp.getSymName();
          archName = ARCH_NAME;
        })
        .Case<hw::HWModuleExternOp>([&](hw::HWModuleExternOp extModOp) {
          const RTLMatch &match = exportInfo.externals.at(extModOp);
          hdl = match.component->getHDL();
          moduleName = match.getConcreteModuleName();
          archName = match.getConcreteArchName();
          genericParams = match.getGenericParameterValues();
        })
        .Default([&](auto) { llvm_unreachable("unknown module type"); });

    raw_indented_ostream &os = data.os;
    // Declare the instance
    os << instOp.getInstanceName() << " : entity work." << moduleName;
    if (hdl == RTLComponent::HDL::VHDL)
      os << "(" << archName << ")";

    // Write generic parameters if there are any
    if (!genericParams.empty()) {
      os << " generic map(";
      for (StringRef param : ArrayRef<StringRef>{genericParams}.drop_back())
        os << param << ", ";
      os << genericParams.back() << ")";
    }
    os << "\n";

    os.indent();
    os << "port map(\n";
    os.indent();

    // Write IO mappings between the hardware instance and the module's internal
    // signals
    writeIOMap(instOp, data);

    os.unindent();
    os << ");\n";
    os.unindent();
    os << "\n";
  }
}

void VHDLWriter::writeIOMap(hw::InstanceOp instOp, WriteData &data) const {
  IOMap ioMap(instOp, exportInfo, data.getSignalNameFunc());
  size_t numIOLeft = ioMap.inputs.size() + ioMap.outputs.size();

  raw_indented_ostream &os = data.os;
  auto writePortsDir = [&](std::vector<PortMapPair> io, StringRef name) {
    // VHDL expects ports belonging to the same array to be mapped contiguously
    // to each other, achieve this by string sorting array elements according to
    // their module port name
    std::sort(io.begin(), io.end(),
              [](auto &firstIO, auto &secondIO) { return firstIO < secondIO; });

    if (!io.empty())
      os << "-- " << name << "\n";
    for (auto &[modPortName, internalSignalName] : io) {
      os << modPortName << " => " << internalSignalName;
      if (--numIOLeft != 0)
        os << ",";
      os << "\n";
    }
  };

  writePortsDir(ioMap.inputs, "inputs");
  writePortsDir(ioMap.outputs, "outputs");
}

/// Writes the RTL implementation corresponding to the hardware module in a file
/// named like the module inside the output directory. Fails if the output file
/// cannot be created or if the module cannot be converted to RTL; succeeds
/// otherwise.
static LogicalResult generateModule(RTLWriter &writer, hw::HWModuleOp modOp) {
  // Open the file in which we will create the module, it is named like the
  // module itself
  const llvm::Twine &filepath =
      outputDir + sys::path::get_separator() + modOp.getSymName() + ".vhd";
  std::error_code ec;
  llvm::raw_fd_ostream fileStream(filepath.str(), ec);
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

  // Make sure the output path does not end in a file separator
  if (outputDir.empty()) {
    llvm::errs() << "Output path is empty\n";
    return 1;
  }
  StringRef sep = sys::path::get_separator();
  if (StringRef{outputDir}.ends_with(sep)) {
    outputDir = outputDir.substr(0, outputDir.size() - sep.size());
  }

  // Create the (potentially nested) output directory
  if (auto ec = sys::fs::create_directories(outputDir); ec.value() != 0) {
    llvm::errs() << "Failed to create output directory\n" << ec.message();
    return 1;
  }

  // Generate/Pull all external modules into the output directory
  ExportInfo info(*modOp, config);
  if (failed(info.concretizeExternalModules()))
    return 1;

  // Write each module's RTL implementation to a separate file
  VHDLWriter writer(info);
  for (hw::HWModuleOp hwModOp : modOp->getOps<hw::HWModuleOp>()) {
    if (failed(generateModule(writer, hwModOp)))
      return 1;
  }

  return 0;
}
