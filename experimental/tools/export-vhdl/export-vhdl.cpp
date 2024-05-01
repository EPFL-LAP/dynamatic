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

/// RTL file writer. This is currently specialized for VHDL but will in the
/// future probably become abstract with child classes to write VHDL or Verilog.
struct RTLWriter {

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
  /// useful when instantiationg components.
  struct IOMap {
    /// Associations for the component's input ports.
    std::vector<PortMapPair> inputs;
    /// Associations for the component's output ports.
    std::vector<PortMapPair> outputs;

    /// Constructs the IO map for an hardware instance corresponding to a known
    /// RTL component. Queries internal signal names from the RTL writer.
    IOMap(hw::InstanceOp instOp, const RTLComponent &rtlComponent,
          RTLWriter &writer);
  };

  /// A value of channel type.
  using ChannelValue = TypedValue<handshake::ChannelType>;

  /// Suffixes for specfic signal types.
  static constexpr StringLiteral VALID_SUFFIX = StringLiteral("_valid"),
                                 READY_SUFFIX = StringLiteral("_ready");

  /// Export information (external modules must have already been concretized).
  ExportInfo exportInfo;
  /// Module being converted to RTL.
  hw::HWModuleOp modOp;
  /// Stream to write to.
  mlir::raw_indented_ostream &os;
  /// Maps channel-typed SSA values to the base name of corresponding RTL
  /// signals (data + valid + ready).
  llvm::MapVector<ChannelValue, std::string> dataflowSignals;
  /// Maps non-channel-typed SSA values to the name of a corresponding RTL
  /// signal.
  llvm::MapVector<Value, std::string> signals;
  /// List of SSA values feeding the entity's output ports.
  SmallVector<Value> outputs;

  /// Creates the RTL writer, which will write the hardware module's
  /// implementation at the output stream.
  RTLWriter(ExportInfo &exportInfo, hw::HWModuleOp modOp,
            mlir::raw_indented_ostream &os)
      : exportInfo(exportInfo), modOp(modOp), os(os){};

  /// Associates each SSA value inside the module to internal module signals.
  /// Fails when encoutering an unsupported operation inside the module;
  /// succeeds otherwise.
  LogicalResult createInternalSignals();

  /// Writes the file header.
  void writeHeader();

  /// Writes the entity declaration corresponding to the module being exported.
  void writeEntityDeclaration();

  /// Writes the enrtty's IO ports.
  void writeEntityIO(const EntityIO &entityIO) const;

  /// Writes the entire entity's architecture.
  void writeArchitecture();

  /// Writes all internal signal declarations inside the entity's architecture.
  void writeInternalSignals();

  /// Writes signal assignments betwween the top-level entity's outputs and the
  /// architecture's internal signals.
  void writeSignalAssignments();

  /// Writes all module instantiations inside the entity's architecture.
  void writeModuleInstantiations();

  /// Writes IO mappings for a component instantiation.
  void writeIOMap(const IOMap &ioMap) const;
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

/// Returns the external hardare module the hardware instance is from.
static hw::HWModuleExternOp getExtModOp(hw::InstanceOp instOp) {
  mlir::ModuleOp modOp = instOp->getParentOfType<mlir::ModuleOp>();
  assert(modOp && "cannot find top-level MLIR module");
  Operation *lookup = modOp.lookupSymbol(instOp.getModuleName());
  assert(lookup && "symbol does not reference an operation");
  return cast<hw::HWModuleExternOp>(lookup);
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

using FGenComp =
    std::function<LogicalResult(const RTLRequest &, hw::HWModuleExternOp)>;

LogicalResult ExportInfo::concretizeExternalModules() {
  mlir::DenseSet<StringRef> entities;

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
    if (auto [_, isNew] = entities.insert(match->getConcreteModuleName());
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

RTLWriter::IOMap::IOMap(hw::InstanceOp instOp, const RTLComponent &rtlComponent,
                        RTLWriter &writer) {
  hw::HWModuleExternOp refModOp = getExtModOp(instOp);

  for (auto [oprd, portAttr] :
       llvm::zip_equal(instOp.getOperands(), refModOp.getInputNamesStr())) {
    std::string port = portAttr.str();
    if (auto channelOprd = dyn_cast<ChannelValue>(oprd)) {
      StringRef signal = writer.dataflowSignals[channelOprd];
      inputs.emplace_back(rtlComponent.getRTLPortName(port, SignalType::DATA),
                          getInternalSignalName(signal, SignalType::DATA));
      inputs.emplace_back(rtlComponent.getRTLPortName(port, SignalType::VALID),
                          getInternalSignalName(signal, SignalType::VALID));
      outputs.emplace_back(rtlComponent.getRTLPortName(port, SignalType::READY),
                           getInternalSignalName(signal, SignalType::READY));
    } else {
      inputs.emplace_back(rtlComponent.getRTLPortName(port),
                          writer.signals[oprd]);
    }
  }
  for (auto [oprd, portAttr] :
       llvm::zip_equal(instOp.getResults(), refModOp.getOutputNamesStr())) {
    std::string port = portAttr.str();
    if (auto channelOprd = dyn_cast<ChannelValue>(oprd)) {
      StringRef signal = writer.dataflowSignals[channelOprd];
      outputs.emplace_back(rtlComponent.getRTLPortName(port, SignalType::DATA),
                           getInternalSignalName(signal, SignalType::DATA));
      outputs.emplace_back(rtlComponent.getRTLPortName(port, SignalType::VALID),
                           getInternalSignalName(signal, SignalType::VALID));
      inputs.emplace_back(rtlComponent.getRTLPortName(port, SignalType::READY),
                          getInternalSignalName(signal, SignalType::READY));
    } else {
      outputs.emplace_back(rtlComponent.getRTLPortName(port),
                           writer.signals[oprd]);
    }
  }
}

LogicalResult RTLWriter::createInternalSignals() {

  // Create signal names for all block arguments
  for (auto [arg, name] : llvm::zip_equal(modOp.getBodyBlock()->getArguments(),
                                          modOp.getInputNamesStr())) {
    if (auto channelArg = dyn_cast<ChannelValue>(arg))
      dataflowSignals[channelArg] = name.strref();
    else
      signals[arg] = name.strref();
  }

  // Create signal names for all operation results
  for (Operation &op : modOp.getBodyBlock()->getOperations()) {
    LogicalResult res =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<hw::InstanceOp>([&](hw::InstanceOp instOp) {
              // Retrieve the module referenced by the instance
              hw::HWModuleExternOp refModOp = getExtModOp(instOp);

              std::string prefix = instOp.getInstanceName().str() + "_";

              // Associate each instance result with a symbol name
              for (auto [res, name] : llvm::zip_equal(
                       instOp->getResults(), refModOp.getOutputNamesStr())) {
                if (auto channelRes = dyn_cast<ChannelValue>(res))
                  dataflowSignals[channelRes] = prefix + name.str();
                else
                  signals[res] = prefix + name.str();
              }
              return success();
            })
            .Case<hw::OutputOp>([&](hw::OutputOp outputOp) {
              llvm::copy(outputOp->getOperands(), std::back_inserter(outputs));
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

void RTLWriter::writeHeader() {
  // Generic imports
  os << "library ieee;\n";
  os << "use ieee.std_logic_1164.all;\n";
  os << "use ieee.numeric_std.all;\n\n";
}

void RTLWriter::writeEntityDeclaration() {

  // Declare the entity
  os << "entity " << modOp.getSymName() << " is\n";
  os.indent();
  os << "port (\n";
  os.indent();

  writeEntityIO(EntityIO{modOp});

  // Close the entity declaration
  os.unindent();
  os << ");\n";
  os.unindent();
  os << "end entity;\n\n";
}

void RTLWriter::writeEntityIO(const EntityIO &entityIO) const {
  size_t numIOLeft = entityIO.inputs.size() + entityIO.outputs.size();

  auto writePortsDir = [&](const std::vector<IOPort> &io, StringRef dir,
                           StringRef name) {
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

void RTLWriter::writeArchitecture() {
  os << "architecture behavioral of " << modOp.getSymName() << " is\n\n";
  os.indent();

  writeInternalSignals();

  os.unindent();
  os << "\nbegin\n\n";
  os.indent();

  // Architecture implementation

  writeSignalAssignments();
  os << "\n";
  writeModuleInstantiations();

  os.unindent();
  os << "end architecture;\n";
}

void RTLWriter::writeInternalSignals() {
  auto isNotBlockArg = [](auto valAndName) -> bool {
    return !isa<BlockArgument>(valAndName.first);
  };

  for (auto [value, name] : make_filter_range(dataflowSignals, isNotBlockArg)) {
    os << "signal " << getInternalSignalName(name, SignalType::DATA) << " : "
       << getChannelDataType(value.getType()) << ";\n";
    os << "signal " << getInternalSignalName(name, SignalType::VALID)
       << " : std_logic;\n";
    os << "signal " << getInternalSignalName(name, SignalType::READY)
       << " : std_logic;\n";
  }

  for (auto [value, name] : make_filter_range(signals, isNotBlockArg))
    os << "signal " << name << " : " << getDataType(value.getType()) << ";\n";
}

void RTLWriter::writeSignalAssignments() {
  os << "-- entity outputs\n";
  for (auto [val, outputName] : llvm::zip(outputs, modOp.getOutputNamesStr())) {
    StringRef name = outputName.strref();
    if (auto channelVal = dyn_cast<ChannelValue>(val)) {
      StringRef signal = dataflowSignals[channelVal];
      os << name << " <= " << signal << ";\n";
      os << name << VALID_SUFFIX << " <= " << signal << VALID_SUFFIX << ";\n";
      os << signal << READY_SUFFIX << " <= " << name << READY_SUFFIX << ";\n";
    } else {
      os << name << " <= " << signals[val] << ";\n";
    }
  }
}

void RTLWriter::writeModuleInstantiations() {
  for (hw::InstanceOp instOp : modOp.getOps<hw::InstanceOp>()) {
    // Retrieve the module referenced by the instance
    hw::HWModuleExternOp refModOp = getExtModOp(instOp);
    const RTLMatch &match = exportInfo.externals[refModOp];

    // Declare the instance
    os << instOp.getInstanceName() << " : ";
    os << "entity work." << match.getConcreteModuleName();
    if (match.component->getHDL() == RTLComponent::HDL::VHDL)
      os << "(" << match.getConcreteArchName() << ")";

    // Write generic parameters if there are any
    SmallVector<StringRef> genericParams = match.getGenericParameterValues();
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

    writeIOMap(IOMap{instOp, *match.component, *this});

    os.unindent();
    os << ");\n";
    os.unindent();
    os << "\n";
  }
}

void RTLWriter::writeIOMap(const IOMap &ioMap) const {
  size_t numIOLeft = ioMap.inputs.size() + ioMap.outputs.size();

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
static LogicalResult generateModule(ExportInfo &info, hw::HWModuleOp modOp) {
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
  mlir::raw_indented_ostream os(fileStream);

  RTLWriter writer(info, modOp, os);

  if (failed(writer.createInternalSignals()))
    return failure();

  writer.writeHeader();
  writer.writeEntityDeclaration();
  writer.writeArchitecture();

  return success();
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

  ExportInfo info(*modOp, config);

  // Generate/Pull all external modules into the output directory
  if (failed(info.concretizeExternalModules()))
    return 1;

  for (hw::HWModuleOp hwModOp : modOp->getOps<hw::HWModuleOp>()) {
    if (failed(generateModule(info, hwModOp)))
      return 1;
  }

  return 0;
}
