//===- BLIFGenerator.cpp - On-demand BLIF file generator --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// C++ version of tools/blif-generator/blif_generator.py.
// Synthesizes a Verilog component to BLIF via Yosys, then optimizes the result
// with ABC.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/BLIFGenerator.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

using namespace dynamatic;

namespace {

// ---------------------------------------------------------------------------
// Helpers for llvm::json
// ---------------------------------------------------------------------------

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

static std::optional<bool> jsonGetBool(const llvm::json::Object &obj,
                                       llvm::StringRef key) {
  const llvm::json::Value *v = obj.get(key);
  if (!v)
    return std::nullopt;
  return v->getAsBoolean();
}

static const llvm::json::Array *jsonGetArray(const llvm::json::Object &obj,
                                             llvm::StringRef key) {
  const llvm::json::Value *v = obj.get(key);
  if (!v)
    return nullptr;
  return v->getAsArray();
}

// ---------------------------------------------------------------------------
// Determine the component name used to identify a JSON entry.
// Priority: module-name -> name (strip "handshake.") -> basename of generic
// path
// ---------------------------------------------------------------------------
static std::string getComponentName(const llvm::json::Object &obj) {
  if (auto mn = jsonGetString(obj, "module-name"))
    return *mn;

  if (auto nameOpt = jsonGetString(obj, "name")) {
    std::string name = *nameOpt;
    auto dot = name.find('.');
    if (dot != std::string::npos)
      return name.substr(dot + 1);
    return name;
  }

  // Support modules have no "name"; use the filename from "generic".
  if (auto genOpt = jsonGetString(obj, "generic")) {
    std::string path = *genOpt;
    auto slash = path.rfind('/');
    std::string filename =
        (slash != std::string::npos) ? path.substr(slash + 1) : path;
    auto dot = filename.rfind('.');
    if (dot != std::string::npos)
      filename = filename.substr(0, dot);
    return filename;
  }
  return "";
}

// ---------------------------------------------------------------------------
// True if a parameter contributes to the Yosys chparam command.
// ---------------------------------------------------------------------------
static bool isRangeParam(const llvm::json::Object &param) {
  auto typeOpt = jsonGetString(param, "type");
  if (!typeOpt)
    return false;
  if (*typeOpt != "unsigned" && *typeOpt != "dataflow")
    return false;

  // Explicit generic:false -> skip.
  auto gen = jsonGetBool(param, "generic");
  if (gen.has_value() && !*gen)
    return false;

  // Fixed values -> not a range parameter.
  if (param.get("eq") || param.get("data-eq"))
    return false;

  return true;
}

// ---------------------------------------------------------------------------
// Replace the first occurrence of `from` with `to` in `str`.
// ---------------------------------------------------------------------------
static void replaceFirst(std::string &str, const std::string &from,
                         const std::string &to) {
  auto pos = str.find(from);
  if (pos != std::string::npos)
    str.replace(pos, from.size(), to);
}

// ---------------------------------------------------------------------------
// Replace all occurrences of `from` with `to` in `str`.
// ---------------------------------------------------------------------------
static void replaceAll(std::string &str, const std::string &from,
                       const std::string &to) {
  size_t pos = 0;
  while ((pos = str.find(from, pos)) != std::string::npos) {
    str.replace(pos, from.size(), to);
    pos += to.size();
  }
}

// ---------------------------------------------------------------------------
// Recursively collect Verilog file paths for a module and its dependencies.
// ---------------------------------------------------------------------------
static void collectVerilogFiles(const llvm::json::Object &config,
                                const llvm::json::Array &allModules,
                                std::vector<std::string> &files,
                                std::set<std::string> &visited,
                                const std::string &dynamaticRoot) {
  // Add this entry's main Verilog file.
  if (auto genOpt = jsonGetString(config, "generic")) {
    std::string path = *genOpt;
    replaceFirst(path, "$DYNAMATIC", dynamaticRoot);
    if (std::filesystem::exists(path) &&
        std::find(files.begin(), files.end(), path) == files.end())
      files.push_back(path);
  }

  const llvm::json::Array *deps = jsonGetArray(config, "dependencies");
  if (!deps)
    return;

  for (const auto &depVal : *deps) {
    auto depStr = depVal.getAsString();
    if (!depStr)
      continue;
    std::string depName = depStr->str();

    if (visited.count(depName))
      continue;
    visited.insert(depName);

    for (const auto &modVal : allModules) {
      const auto *modObj = modVal.getAsObject();
      if (!modObj)
        continue;
      if (getComponentName(*modObj) == depName) {
        collectVerilogFiles(*modObj, allModules, files, visited, dynamaticRoot);
        break;
      }
    }
  }
}

} // namespace

// ---------------------------------------------------------------------------
// BLIFGenerator implementation
// ---------------------------------------------------------------------------

BLIFGenerator::BLIFGenerator(const std::string &blifDirPath,
                             const std::string &yosysExecutable,
                             const std::string &abcExecutable)
    : blifDirPath(blifDirPath), yosysExecutable(yosysExecutable),
      abcExecutable(abcExecutable) {
  namespace fs = std::filesystem;
  // blifDirPath is "<root>/data/blif"; root is two levels up.
  dynamaticRoot =
      (fs::path(blifDirPath) / ".." / "..").lexically_normal().string();
}

bool BLIFGenerator::generate(const std::string &component,
                             const std::vector<std::string> &paramValues) {
  bool debugPrint = false;
  namespace fs = std::filesystem;

  // Build and create the output directory.
  fs::path outputDir = blifDirPath;
  outputDir /= component;
  for (const auto &p : paramValues)
    outputDir /= p;
  fs::create_directories(outputDir);

  // Load and parse JSON config.
  fs::path jsonPath = (fs::path(blifDirPath) / ".." / "rtl-config-verilog.json")
                          .lexically_normal();
  std::ifstream configFile(jsonPath.string());
  if (!configFile.is_open()) {
    llvm::errs() << "BLIFGenerator cannot open config: " << jsonPath.string()
                 << "\n";
    return false;
  }
  std::string jsonStr, line;
  while (std::getline(configFile, line))
    jsonStr += line;

  auto parsedOrErr = llvm::json::parse(jsonStr);
  if (!parsedOrErr) {
    llvm::errs() << "BLIFGenerator failed to parse JSON config\n";
    return false;
  }

  const llvm::json::Array *allModules = parsedOrErr->getAsArray();
  if (!allModules) {
    llvm::errs() << "BLIFGenerator JSON parsing failed: root is not an array\n";
    return false;
  }

  // Find the module config entry for this component.
  // Primary lookup: match by the name derived via getComponentName().
  // Fallback: match by the basename of the "generic" Verilog file path.
  // The fallback handles cases where multiple entries share the same "name"
  // but point to different Verilog files (e.g. mem_controller_storeless vs
  // mem_controller_loadless vs mem_controller all have name
  // "handshake.mem_controller").
  const llvm::json::Object *moduleConfig = nullptr;
  std::string yosysTopModule;
  for (const auto &entry : *allModules) {
    const auto *obj = entry.getAsObject();
    if (!obj)
      continue;
    if (getComponentName(*obj) == component) {
      moduleConfig = obj;
      yosysTopModule = component;
      break;
    }
  }

  if (!moduleConfig) {
    for (const auto &entry : *allModules) {
      const auto *obj = entry.getAsObject();
      if (!obj)
        continue;
      auto genOpt = jsonGetString(*obj, "generic");
      if (!genOpt)
        continue;
      std::string path = *genOpt;
      auto slash = path.rfind('/');
      std::string basename =
          (slash != std::string::npos) ? path.substr(slash + 1) : path;
      auto dot = basename.rfind('.');
      if (dot != std::string::npos)
        basename = basename.substr(0, dot);
      if (basename == component) {
        moduleConfig = obj;
        yosysTopModule = component;
        break;
      }
    }
  }

  if (!moduleConfig) {
    llvm::errs() << "BLIFGenerator component '" << component
                 << "' not found in JSON config\n";
    return false;
  }

  // If the entry has a generator command, run it to produce the Verilog file.
  std::string generatedVerilogFile;
  if (auto genCmdOpt = jsonGetString(*moduleConfig, "generator")) {
    std::string cmd = *genCmdOpt;
    replaceAll(cmd, "$DYNAMATIC", dynamaticRoot);
    replaceAll(cmd, "$OUTPUT_DIR", outputDir.string());
    replaceAll(cmd, "$MODULE_NAME", component);
    // Hard-code special substitutions for constant and cmpi. Missing cmpf, lsq
    // and ram which are unhandled for now
    if (component == "constant")
      replaceAll(cmd, "$VALUE", paramValues.empty() ? "1" : paramValues.back());
    if (component == "cmpi")
      replaceAll(cmd, "$PREDICATE",
                 paramValues.empty() ? "ult" : paramValues.back());

    if (debugPrint) {
      llvm::errs() << "BLIFGenerator generating " << component << "\n";
    }
    if (system(cmd.c_str()) != 0) {
      llvm::errs() << "BLIFGenerator command failed for " << component << "\n";
      return false;
    }
    generatedVerilogFile = (outputDir / (component + ".v")).string();
  }

  // Collect Verilog dependency files.
  std::vector<std::string> verilogFiles;
  std::set<std::string> visited;
  collectVerilogFiles(*moduleConfig, *allModules, verilogFiles, visited,
                      dynamaticRoot);
  if (!generatedVerilogFile.empty() &&
      std::filesystem::exists(generatedVerilogFile))
    verilogFiles.push_back(generatedVerilogFile);

  // Collect ordered generic parameter names from JSON.
  std::vector<std::string> genericParamNames;
  if (const auto *params = jsonGetArray(*moduleConfig, "parameters")) {
    for (const auto &paramVal : *params) {
      const auto *param = paramVal.getAsObject();
      if (!param || !isRangeParam(*param))
        continue;
      if (auto pname = jsonGetString(*param, "name"))
        genericParamNames.push_back(*pname);
    }
  }

  // Build the chparam command.
  // paramValues are right-aligned against genericParamNames so that a single
  // paramValue always maps to the last (innermost) generic parameter.  This
  // handles components like tfifo where the path only encodes DATA_TYPE but
  // the JSON also lists NUM_SLOTS as a range parameter.
  std::string chparam;
  if (!paramValues.empty() && !genericParamNames.empty()) {
    size_t offset = (genericParamNames.size() > paramValues.size())
                        ? (genericParamNames.size() - paramValues.size())
                        : 0;
    chparam = "chparam";
    for (size_t i = 0;
         i < paramValues.size() && (offset + i) < genericParamNames.size(); ++i)
      chparam +=
          " -set " + genericParamNames[offset + i] + " " + paramValues[i];
    chparam += " " + yosysTopModule + ";";
  }

  // Yosys output: <component>_<p0>_<p1>_..._yosys.blif
  std::string nameSuffix;
  for (const auto &p : paramValues)
    nameSuffix += "_" + p;
  std::string yosysOutputName = component + nameSuffix + "_yosys.blif";
  std::string yosysOutputPath = (outputDir / yosysOutputName).string();

  if (!runYosys(yosysTopModule, outputDir.string(), chparam, verilogFiles,
                yosysOutputName)) {
    llvm::errs() << "BLIFGenerator call for Yosys failed for " << component
                 << "\n";
    return false;
  }

  std::string abcOutputName = component + ".blif";
  if (!runAbc(yosysOutputPath, outputDir.string(), abcOutputName)) {
    llvm::errs() << "BLIFGenerator call for ABC failed for " << component
                 << "\n";
    return false;
  }

  return std::filesystem::exists(outputDir / abcOutputName);
}

std::string BLIFGenerator::expandPath(const std::string &path) const {
  std::string result = path;
  replaceFirst(result, "$DYNAMATIC", dynamaticRoot);
  return result;
}

bool BLIFGenerator::runYosys(const std::string &topModule,
                             const std::string &outputDir,
                             const std::string &chparam,
                             const std::vector<std::string> &verilogFiles,
                             const std::string &outputName) const {
  std::string scriptPath = outputDir + "/run_yosys.sh";
  std::ofstream script(scriptPath);
  if (!script.is_open())
    return false;

  script << "#!/bin/bash\n";
  script << yosysExecutable << " -p \"";
  for (const auto &f : verilogFiles)
    script << "read_verilog -defer " << f << "\n        ";
  if (!chparam.empty())
    script << chparam << "\n        ";
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

bool BLIFGenerator::runAbc(const std::string &inputFile,
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
