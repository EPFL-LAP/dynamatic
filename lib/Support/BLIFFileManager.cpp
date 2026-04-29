//===- BLIFFileManager.cpp - Support functions for BLIF importer -*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the function to retrieve the path of the blif file for a
// given handshake operation. The path is created by combining the base path
// containing all the blif files, the operation name, and the parameter values
// returned by getRTLParameters() via the RTLAttrInterface.
// If the file does not exist, an attempt is made to generate it using
// BLIFGenerator (requires Yosys and ABC to be configured at build time).
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/BLIFFileManager.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/BackendGenerator.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <regex>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace dynamatic {

// Converts a single NamedAttribute value from getRTLParameters() to a string.
// TypeAttr(HandshakeType) -> data bitwidth; IntegerAttr -> integer; StringAttr
// -> string.
static std::string rtlParamToString(mlir::Attribute attr) {
  if (auto ta = mlir::dyn_cast<mlir::TypeAttr>(attr))
    return std::to_string(handshake::getHandshakeTypeBitWidth(ta.getValue()));
  if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return std::to_string(ia.getValue().getZExtValue());
  if (auto sa = mlir::dyn_cast<mlir::StringAttr>(attr))
    return sa.str();
  return "";
}

// Function to combine parameter values, module type and blif directory path
// to create the blif file path
std::string BLIFFileManager::combineBlifFilePath(
    const std::string &moduleType, const std::vector<std::string> &paramValues,
    const std::string &extraSuffix) {
  std::string blifFilePath = blifDirPath + "/" + moduleType + extraSuffix;
  for (const auto &paramValue : paramValues) {
    blifFilePath += "/" + paramValue;
  }
  blifFilePath += "/" + moduleType + extraSuffix + ".blif";
  return blifFilePath;
}

// Function to retrieve the path of the blif file for a given handshake
// operation
std::string BLIFFileManager::getBlifFilePathForHandshakeOp(Operation *op) {
  // EndOp has no hardware implementation.
  if (mlir::isa<handshake::EndOp>(op))
    return "";

  // Get op name, strip dialect prefix (e.g. "handshake.addi" -> "addi").
  std::string moduleType = op->getName().getStringRef().str();
  moduleType = moduleType.substr(moduleType.find('.') + 1);

  // Retrieve all RTL parameters from the op via the interface.
  auto iface = mlir::cast<handshake::RTLAttrInterface>(op);
  auto paramsOrErr = iface.getRTLParameters();
  if (mlir::failed(paramsOrErr)) {
    llvm::errs() << "BLIFFileManager: getRTLParameters() failed for `"
                 << moduleType << "`\n";
    return "";
  }

  // Convert all parameter values to strings for path construction.
  std::vector<std::string> paramStrs;
  paramStrs.reserve(paramsOrErr->size());
  for (const auto &p : *paramsOrErr)
    paramStrs.push_back(rtlParamToString(p.getValue()));

  std::string blifFileName = combineBlifFilePath(moduleType, paramStrs);

  // Check if the file exists; if not, attempt to generate it.
  if (!std::filesystem::exists(blifFileName)) {
    llvm::errs() << "BLIF file missing, generating: " << blifFileName << "\n";
    BackendGenerator gen(BackendGenerator::Backend::BLIF,
                         BackendGenerator::BackendParams{
                             rtlJsonFile, dynamaticRootPath, blifDirPath});
    bool ok = gen.generate(op);
    if (!ok || !std::filesystem::exists(blifFileName)) {
      llvm::errs() << "Failed to generate BLIF file for operation `"
                   << getUniqueName(op) << "`: " << blifFileName
                   << ". Ensure that Yosys and ABC are correctly configured.\n";
      assert(false && "BLIF generation failed");
    }
  }
  return blifFileName;
}

// Formats a bit-indexed port name: "sig[bit]".
// When width == 1 the name is returned unchanged (no "[0]" suffix).
// If baseName already ends with "[N]" the indices are linearised.
std::string legalizeDataPortName(StringRef baseName, unsigned bit,
                                 unsigned width) {
  if (width == 1)
    return baseName.str();
  return formatArrayName(baseName.str(), bit, width);
}

// Inserts a suffix string before the first '[' in the input name, or appends
// it at the end if no '[' is found.
// Example: legalizeControlPortName("data[3]", "_valid") -> "data_valid[3]"
std::string legalizeControlPortName(const std::string &name,
                                    const std::string &suffix) {
  std::size_t pos = name.find('[');
  if (pos != std::string::npos) {
    std::string result = name;
    result.insert(pos, suffix);
    return result;
  }
  return name + suffix;
}

// Converts a (root, index) pair into the canonical "root[index]" form.
// If root already contains "[N]", the old index is linearised with
//  arrayWidth before adding index.
// Example: formatArrayName("data[2]", 3, 4) becomes "data[11]"  (2*4 + 3)
std::string formatArrayName(const std::string &root, unsigned index,
                            unsigned arrayWidth) {
  static const std::regex arrayPattern(R"((\w+)\[(\d+)\])");
  std::smatch m;
  if (std::regex_match(root, m, arrayPattern)) {
    assert(arrayWidth != 0 && "arrayWidth required for already-indexed names");
    unsigned linearised = std::stoi(m[2].str()) * arrayWidth + index;
    return m[1].str() + "[" + std::to_string(linearised) + "]";
  }
  return root + "[" + std::to_string(index) + "]";
}

// Legalizes a list of handshake port names by converting the "root_N" index
// pattern (used internally by NamedIOInterface) into the "root[N]" array
// notation expected by the BLIF importer.
// Example: ["data_0", "data_1", "valid"] -> ["data[0]", "data[1]", "valid"]
void legalizeBlifPortNames(SmallVector<std::string> &names) {
  static const std::regex indexPat(R"((\w+)_(\d+))");
  SmallVector<std::string> result;
  result.reserve(names.size());

  for (auto &name : names) {
    std::smatch m;
    if (!std::regex_match(name, m, indexPat)) {
      result.push_back(name);
      continue;
    }
    std::string root = m[1].str();
    unsigned idx = std::stoi(m[2].str());
    if (idx == 0) {
      result.push_back(root);
    } else {
      if (idx == 1) {
        auto *it = std::find(result.begin(), result.end(), root);
        assert(it != result.end() && "index-0 port not found for back-patch");
        *it = formatArrayName(root, 0);
      }
      result.push_back(formatArrayName(root, idx));
    }
  }
  names = std::move(result);
}

} // namespace dynamatic
