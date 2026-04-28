//===- RTLGenerator.cpp - RTL code generator from JSON config ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates a Verilog file for a given Handshake operation by looking up its
// entry in a JSON RTL config, substituting RTL parameters (extracted via
// RTLAttrInterface), and running the configured generator command.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/RTLGenerator.h"
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
#include <sstream>

using namespace dynamatic;

namespace {

// Helper functions for JSON parsing and variable substitution.
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

// Get an array from a JSON object, or nullptr if the key is missing or not an
// array.
static const llvm::json::Array *jsonGetArray(const llvm::json::Object &obj,
                                             llvm::StringRef key) {
  const llvm::json::Value *v = obj.get(key);
  if (!v)
    return nullptr;
  return v->getAsArray();
}

// Convert an RTL parameter attribute to a string for variable substitution.
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

// Replace all occurrences of 'from' in 'str' with 'to'.
static void replaceAll(std::string &str, const std::string &from,
                       const std::string &to) {
  size_t pos = 0;
  while ((pos = str.find(from, pos)) != std::string::npos) {
    str.replace(pos, from.size(), to);
    pos += to.size();
  }
}

// Check if a JSON config entry matches the given parameter map.
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

} // namespace

RTLGenerator::RTLGenerator(const std::string &rtlConfigPath,
                           const std::string &dynamaticRoot,
                           const std::string &outputBaseDir,
                           mlir::Operation *op)
    : rtlConfigPath(rtlConfigPath), dynamaticRoot(dynamaticRoot),
      outputBaseDir(outputBaseDir), op(op) {}

/// Runs the RTL generator command found in the JSON config.
bool RTLGenerator::generate() {
  namespace fs = std::filesystem;

  std::string opName = op->getName().getStringRef().str();
  std::string opShortName = opName.substr(opName.find('.') + 1);

  // Extract parameters from the operation via RTLAttrInterface.
  auto iface = mlir::cast<dynamatic::handshake::RTLAttrInterface>(op);
  auto paramsOrErr = iface.getRTLParameters();
  if (mlir::failed(paramsOrErr)) {
    llvm::errs() << "RTLGenerator: getRTLParameters() failed for `" << opName
                 << "`\n";
    return false;
  }

  // Construct a map from parameter name to value, and a list of parameter
  // values.
  std::map<std::string, std::string> paramMap;
  std::vector<std::string> paramValues;
  for (const auto &p : *paramsOrErr) {
    std::string name = p.getName().str();
    std::string value = rtlParamToString(p.getValue());
    paramMap[name] = value;
    paramValues.push_back(value);
  }

  moduleName = opShortName;
  for (const auto &v : paramValues)
    moduleName += "_" + v;

  outputDir = outputBaseDir;
  fs::path outPath = fs::path(outputBaseDir) / opShortName;
  for (const auto &v : paramValues)
    outPath /= v;
  outputDir = outPath.string();
  fs::create_directories(outPath);

  paramMap["MODULE_NAME"] = moduleName;
  paramMap["OUTPUT_DIR"] = outputDir;
  paramMap["DYNAMATIC"] = dynamaticRoot;
  paramMap["EXTRA_SIGNALS"] = "0";
  if (paramMap.find("BITWIDTH") == paramMap.end())
    paramMap["BITWIDTH"] = "0";

  // Parse the JSON RTL config and find the matching entry for this operation
  // based on its name.
  std::ifstream configFile(rtlConfigPath);
  if (!configFile.is_open()) {
    llvm::errs() << "RTLGenerator cannot open config: " << rtlConfigPath
                 << "\n";
    return false;
  }
  std::string jsonStr, line;
  while (std::getline(configFile, line))
    jsonStr += line;

  auto parsedOrErr = llvm::json::parse(jsonStr);
  if (!parsedOrErr) {
    llvm::errs() << "RTLGenerator failed to parse JSON config\n";
    return false;
  }
  const llvm::json::Array *allEntries = parsedOrErr->getAsArray();
  if (!allEntries) {
    llvm::errs() << "RTLGenerator JSON root is not an array\n";
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
    llvm::errs() << "RTLGenerator: no config entry for `" << opName << "`\n";
    return false;
  }

  // Get the generator command from the config, substitute variables, and
  // execute it.
  auto genCmdOpt = jsonGetString(*matchedEntry, "generator");
  if (!genCmdOpt) {
    llvm::errs() << "RTLGenerator: no 'generator' field for `" << opName
                 << "`\n";
    return false;
  }
  std::string cmd = *genCmdOpt;

  std::vector<std::pair<std::string, std::string>> sortedParams(
      paramMap.begin(), paramMap.end());
  std::sort(sortedParams.begin(), sortedParams.end(),
            [](const auto &a, const auto &b) {
              return a.first.size() > b.first.size();
            });
  for (const auto &[name, value] : sortedParams)
    replaceAll(cmd, "$" + name, value);

  if (cmd.find('$') != std::string::npos) {
    llvm::errs() << "RTLGenerator: unresolved variables in generator command "
                    "for `"
                 << opName << "`: " << cmd << "\n";
    return false;
  }

  if (system(cmd.c_str()) != 0) {
    llvm::errs() << "RTLGenerator: generator command failed for `" << opName
                 << "`\n";
    return false;
  }

  // Collect the generated Verilog file and any support files.
  verilogFiles.clear();
  std::string genVerilog = (outPath / (moduleName + ".v")).string();
  if (fs::exists(genVerilog))
    verilogFiles.push_back(genVerilog);

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

  return true;
}
