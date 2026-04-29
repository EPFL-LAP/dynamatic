//===- BackendGenerator.cpp - RTL and BLIF backend generator ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates RTL (Verilog) or BLIF output for a given Handshake operation by
// looking up its entry in a JSON RTL config, substituting parameters, and
// running the generator command. For the BLIF backend, Yosys synthesis and
// ABC optimization are also performed.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/BackendGenerator.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>

using namespace dynamatic;

// Helper functions
namespace {

// Extracts a string value from a JSON object for a given key, or returns
// std::nullopt if the key is not found or the value is not a string.
static std::optional<std::string> jsonGetString(const llvm::json::Object &obj,
                                                llvm::StringRef key) {
  const llvm::json::Value *v = obj.get(key);
  if (!v)
    return std::nullopt;
  auto s = v->getAsString();
  if (!s)
    return std::nullopt;
  return s->str();
}

// Extracts an array value from a JSON object for a given key, or returns
// nullptr if the key is not found or the value is not an array.
static const llvm::json::Array *jsonGetArray(const llvm::json::Object &obj,
                                             llvm::StringRef key) {
  const llvm::json::Value *v = obj.get(key);
  if (!v)
    return nullptr;
  return v->getAsArray();
}

// Converts a single NamedAttribute value from getRTLParameters() to a string.
static std::string rtlParamToString(mlir::Attribute attr) {
  if (auto ta = mlir::dyn_cast<mlir::TypeAttr>(attr))
    return std::to_string(
        dynamatic::handshake::getHandshakeTypeBitWidth(ta.getValue()));
  if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return std::to_string(ia.getValue().getZExtValue());
  if (auto sa = mlir::dyn_cast<mlir::StringAttr>(attr))
    return sa.str();
  return "";
}

// Replaces all occurrences of `from` in `str` with `to`.
static void replaceAll(std::string &str, const std::string &from,
                       const std::string &to) {
  size_t pos = 0;
  while ((pos = str.find(from, pos)) != std::string::npos) {
    str.replace(pos, from.size(), to);
    pos += to.size();
  }
}

// Checks if a JSON config entry matches the given parameter map. An entry
// matches if all parameters specified in the entry are present in the map with
// the same value.
static bool
entryMatchesParams(const llvm::json::Object &entry,
                   const std::map<std::string, std::string> &paramMap) {
  const auto *params = jsonGetArray(entry, "parameters");
  if (!params)
    return true;
  for (const auto &pval : *params) {
    const auto *pobj = pval.getAsObject();
    if (!pobj)
      continue;
    auto nameOpt = jsonGetString(*pobj, "name");
    auto eqOpt = jsonGetString(*pobj, "eq");
    if (!nameOpt || !eqOpt)
      continue;
    auto it = paramMap.find(*nameOpt);
    if (it == paramMap.end() || it->second != *eqOpt)
      return false;
  }
  return true;
}

// Function to extract the parameters of an operation into a map.
bool getParamMap(mlir::Operation *op,
                 std::map<std::string, std::string> &paramMap) {
  // Retrieve all RTL parameters from the op via the interface, and construct
  // the module name and output directory path based on the operation name and
  // parameter values.
  auto iface = mlir::cast<dynamatic::handshake::RTLAttrInterface>(op);
  auto paramsOrErr = iface.getRTLParameters();
  if (mlir::failed(paramsOrErr)) {
    llvm::errs() << "BackendGenerator: getRTLParameters() failed for `"
                 << op->getName().getStringRef() << "`\n";
    return false;
  }

  paramMap.clear();
  for (const auto &p : *paramsOrErr) {
    std::string name = p.getName().str();
    std::string value = rtlParamToString(p.getValue());
    paramMap[name] = value;
  }
  return true;
}

} // namespace

BackendGenerator::BackendGenerator(Backend backend, Params params)
    : backend(backend), params(std::move(params)) {}

// Main entry point for generating backend output for a given operation.
bool BackendGenerator::generate(mlir::Operation *op) {
  switch (backend) {
  case Backend::Verilog:
    return generateVerilog(op);
  case Backend::BLIF:
    return generateBLIF(op);
  }
  assert(false && "Only Verilog and BLIF backends are supported for now");
  return false;
}

// Function to get the directory path of backend files for a given operation.
std::filesystem::path
BackendGenerator::getOutputDirForOp(mlir::Operation *op,
                                    const std::string &baseDir) {
  std::string opName = op->getName().getStringRef().str();
  std::string opShortName = opName.substr(opName.find('.') + 1);

  auto iface = mlir::cast<dynamatic::handshake::RTLAttrInterface>(op);
  auto paramsOrErr = iface.getRTLParameters();
  if (mlir::failed(paramsOrErr)) {
    llvm::errs() << "BackendGenerator: getRTLParameters() failed for `"
                 << opName << "`\n";
    return "";
  }

  std::string dir = std::filesystem::path(baseDir) / opShortName;
  for (const auto &p : *paramsOrErr)
    dir = std::filesystem::path(dir) / rtlParamToString(p.getValue());
  return dir;
}

// Function to generate a BLIF file for a given operation by running the RTL
// generator, Yosys and ABC.
bool BackendGenerator::generateBLIF(mlir::Operation *op) {
  namespace fs = std::filesystem;

  const auto &p = std::get<BLIFParams>(params);

  BackendGenerator verilogGen(
      Backend::Verilog,
      VerilogParams{p.rtlConfigPath, p.dynamaticRoot, p.outputBaseDir});
  if (!verilogGen.generate(op)) {
    llvm::errs() << "BackendGenerator: Verilog generation failed for `"
                 << op->getName().getStringRef() << "`\n";
    return false;
  }
  std::vector<std::string> verilogFiles = verilogGen.getOutputFiles();
  if (verilogFiles.empty()) {
    llvm::errs() << "BackendGenerator: no Verilog files generated for `"
                 << op->getName().getStringRef() << "`\n";
    return false;
  }
  // Get module name
  moduleName = verilogGen.getModuleName();
  assert(!moduleName.empty() && "Module name cannot be empty");

  // Construct the expected BLIF file path based on the operation name and
  std::string opShortName = op->getName().getStringRef().str();
  opShortName = opShortName.substr(opShortName.find('.') + 1);

  // Get output directory for the operation and create it if it doesn't exist.
  std::string outDirPath = getOutputDirForOp(op, p.outputBaseDir);
  fs::create_directories(outDirPath);

  // Run Yosys to synthesize the Verilog into BLIF.
  std::string yosysOutName = moduleName + "_yosys.blif";
  if (!runYosys(p.yosysExecutable, moduleName, outDirPath, verilogFiles,
                yosysOutName)) {
    llvm::errs() << "BackendGenerator: Yosys failed for `"
                 << op->getName().getStringRef() << "`\n";
    return false;
  }

  // Run ABC to optimize the BLIF file generated by Yosys.
  std::string yosysOutPath = (fs::path(outDirPath) / yosysOutName).string();
  std::string abcOutName = opShortName + ".blif";
  if (!runAbc(p.abcExecutable, yosysOutPath, outDirPath, abcOutName)) {
    llvm::errs() << "BackendGenerator: ABC failed for `"
                 << op->getName().getStringRef() << "`\n";
    return false;
  }

  // Check that the final BLIF file exists and store its path in outputFiles.
  fs::path blifPath = fs::path(outDirPath) / abcOutName;
  if (!fs::exists(blifPath))
    return false;
  outputFiles = {blifPath.string()};
  return true;
}

bool BackendGenerator::generateVerilog(mlir::Operation *op) {

  const auto &verilogParams = std::get<VerilogParams>(params);
  const std::string &rtlConfigPath = verilogParams.rtlConfigPath;
  const std::string &dynamaticRoot = verilogParams.dynamaticRoot;
  const std::string &outputBaseDir = verilogParams.outputBaseDir;

  // Assert parameters are not empty or NULL
  assert(!rtlConfigPath.empty() && "rtlConfigPath cannot be empty");
  assert(!dynamaticRoot.empty() && "dynamaticRoot cannot be empty");
  assert(!outputBaseDir.empty() && "outputBaseDir cannot be empty");

  std::vector<std::string> verilogFiles;
  namespace fs = std::filesystem;

  // Generate the Verilog files using the RTL generator command from the JSON
  // config for the operation.
  std::string opName = op->getName().getStringRef().str();
  std::string opShortName = opName.substr(opName.find('.') + 1);

  // Get the operation parameters into a map for easy lookup when substituting
  // into the generator command.
  std::map<std::string, std::string> paramMap;
  getParamMap(op, paramMap);

  moduleName = opShortName;

  // Get output directory for the operation and create it if it doesn't exist.
  fs::path outDirPath = getOutputDirForOp(op, outputBaseDir);
  fs::create_directories(outDirPath);

  // Add common parameters for generator command substitution.
  paramMap["MODULE_NAME"] = moduleName;
  paramMap["OUTPUT_DIR"] = outDirPath;
  paramMap["DYNAMATIC"] = dynamaticRoot;
  paramMap["EXTRA_SIGNALS"] = "0";
  if (paramMap.find("BITWIDTH") == paramMap.end())
    paramMap["BITWIDTH"] = "0";

  // Read and parse the JSON config file to find the generator command for this
  // operation.
  std::ifstream configFile(rtlConfigPath);
  if (!configFile.is_open()) {
    llvm::errs() << "BackendGenerator: cannot open config: " << rtlConfigPath
                 << "\n";
    return false;
  }
  std::string jsonStr, line;
  while (std::getline(configFile, line))
    jsonStr += line;

  auto parsedOrErr = llvm::json::parse(jsonStr);
  if (!parsedOrErr) {
    llvm::errs() << "BackendGenerator: failed to parse JSON config\n";
    return false;
  }
  const llvm::json::Array *allEntries = parsedOrErr->getAsArray();
  if (!allEntries) {
    llvm::errs() << "BackendGenerator: JSON root is not an array\n";
    return false;
  }

  const llvm::json::Object *matchedEntry = nullptr;
  for (const auto &entryVal : *allEntries) {
    const auto *obj = entryVal.getAsObject();
    if (!obj)
      continue;
    auto nameOpt = jsonGetString(*obj, "name");
    if (!nameOpt || *nameOpt != opName)
      continue;
    if (entryMatchesParams(*obj, paramMap)) {
      matchedEntry = obj;
      break;
    }
  }
  if (!matchedEntry) {
    llvm::errs() << "BackendGenerator: no config entry for `" << opName
                 << "`\n";
    return false;
  }

  auto genCmdOpt = jsonGetString(*matchedEntry, "generator");
  if (!genCmdOpt) {
    llvm::errs() << "BackendGenerator: no 'generator' field for `" << opName
                 << "`\n";
    return false;
  }
  std::string cmd = *genCmdOpt;

  // Substitute longer names first to avoid partial matches.
  std::vector<std::pair<std::string, std::string>> sortedParams(
      paramMap.begin(), paramMap.end());
  std::sort(sortedParams.begin(), sortedParams.end(),
            [](const auto &a, const auto &b) {
              return a.first.size() > b.first.size();
            });

  // Substitute parameters into the generator command.
  for (const auto &[name, value] : sortedParams)
    replaceAll(cmd, "$" + name, value);

  // Check that all variables have been substituted (no '$' should remain) and
  // run the command.
  if (cmd.find('$') != std::string::npos) {
    llvm::errs() << "BackendGenerator: unresolved variables in generator "
                    "command for `"
                 << opName << "`: " << cmd << "\n";
    return false;
  }
  if (system(cmd.c_str()) != 0) {
    llvm::errs() << "BackendGenerator: generator command failed for `" << opName
                 << "`\n";
    return false;
  }

  // Collect the generated Verilog files. The generator command is expected to
  // produce a main file named "<moduleName>.v" in the output directory.
  std::string genVerilog = (outDirPath / (moduleName + ".v")).string();
  if (fs::exists(genVerilog))
    verilogFiles.push_back(genVerilog);
  // Copy any additional .v files that might be used by the Yosys script (e.g.
  // for submodules).
  for (const auto &entryVal : *allEntries) {
    const auto *obj = entryVal.getAsObject();
    if (!obj || obj->get("name"))
      continue;
    auto genOpt = jsonGetString(*obj, "generic");
    if (!genOpt)
      continue;
    std::string path = *genOpt;
    replaceAll(path, "$DYNAMATIC", dynamaticRoot);
    if (fs::exists(path) && std::find(verilogFiles.begin(), verilogFiles.end(),
                                      path) == verilogFiles.end())
      verilogFiles.push_back(path);
  }
  // Check that the list of Verilog files is not empty.
  if (verilogFiles.empty()) {
    llvm::errs() << "Generator command did not produce expected Verilog file: "
                 << genVerilog << "\n";
    return false;
  }
  outputFiles = verilogFiles;
  return true;
}

bool BackendGenerator::runYosys(const std::string &yosysExecutable,
                                const std::string &topModule,
                                const std::string &outputDir,
                                const std::vector<std::string> &verilogFiles,
                                const std::string &outputName) const {
  std::string scriptPath = outputDir + "/run_yosys.sh";
  std::ofstream script(scriptPath);
  if (!script.is_open())
    return false;

  script << "#!/bin/bash\n";
  script << yosysExecutable << " -p \"";
  for (const auto &f : verilogFiles)
    script << "read_verilog -defer " << f << ";\n        ";
  script << "hierarchy -top " << topModule << ";\n";
  script << "        proc;\n";
  script << "        opt -nodffe -nosdff;\n";
  script << "        memory -nomap;\n";
  script << "        techmap;\n";
  script << "        flatten;\n";
  script << "        clean;\n";
  script << "        write_blif " << outputDir << "/" << outputName
         << "\" > /dev/null 2>&1\n";
  script.close();

  std::filesystem::permissions(scriptPath,
                               std::filesystem::perms::owner_exec |
                                   std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::add);
  return system(("bash " + scriptPath).c_str()) == 0;
}

bool BackendGenerator::runAbc(const std::string &abcExecutable,
                              const std::string &inputFile,
                              const std::string &outputDir,
                              const std::string &outputName) const {
  std::string scriptPath = outputDir + "/run_abc.sh";
  std::ofstream script(scriptPath);
  if (!script.is_open())
    return false;

  script << "#!/bin/bash\n";
  script << abcExecutable << " -c \"read_blif " << inputFile << ";\n";
  script << "        strash;\n";
  for (int i = 0; i < 6; ++i) {
    script << "        rewrite;\n";
    script << "        b;\n";
    script << "        refactor;\n";
    script << "        b;\n";
  }
  script << "        write_blif " << outputDir << "/" << outputName
         << "\" > /dev/null 2>&1\n";
  script.close();

  std::filesystem::permissions(scriptPath,
                               std::filesystem::perms::owner_exec |
                                   std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::add);
  return system(("bash " + scriptPath).c_str()) == 0;
}
