//===- HandshakeSpeculation.cpp - Speculative Dataflows ---------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Placement of Speculation components to enable speculative execution.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/Speculation/SpeculationPlacement.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/Logging.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include <fstream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::speculation;

// SpeculationPlacements Methods

bool OpPlacement::operator==(const OpPlacement &other) const {
  return (this->srcOpResult == other.srcOpResult) and
         (this->dstOp == other.dstOp);
}

std::size_t OpPlacement::Hash::operator()(const OpPlacement &p) const {
  std::size_t srcHash = mlir::hash_value(p.srcOpResult);
  std::size_t dstHash = p.dstOp->hashProperties();
  std::size_t hash = llvm::hash_combine(srcHash, dstHash);
  return hash;
}

void SpeculationPlacements::setSpeculator(Value srcOpResult, Operation *dstOp) {
  this->speculator = {srcOpResult, dstOp};
}

void SpeculationPlacements::addSave(Value srcOpResult, Operation *dstOp) {
  this->saves.insert({srcOpResult, dstOp});
}

void SpeculationPlacements::addCommit(Value srcOpResult, Operation *dstOp) {
  this->commits.insert({srcOpResult, dstOp});
}

void SpeculationPlacements::addSaveCommit(Value srcOpResult, Operation *dstOp) {
  this->saveCommits.insert({srcOpResult, dstOp});
}

bool SpeculationPlacements::containsCommit(Value srcOpResult,
                                           Operation *dstOp) {
  return this->commits.count({srcOpResult, dstOp});
}

bool SpeculationPlacements::containsSave(Value srcOpResult, Operation *dstOp) {
  return this->saves.count({srcOpResult, dstOp});
}

void SpeculationPlacements::eraseCommit(Value srcOpResult, Operation *dstOp) {
  this->commits.erase({srcOpResult, dstOp});
}

void SpeculationPlacements::eraseSave(Value srcOpResult, Operation *dstOp) {
  this->saves.erase({srcOpResult, dstOp});
}

OpPlacement SpeculationPlacements::getSpeculatorPlacement() {
  return this->speculator;
}

template <>
const PlacementList &
SpeculationPlacements::getPlacements<handshake::SpecSaveOp>() {
  return this->saves;
}

template <>
const PlacementList &
SpeculationPlacements::getPlacements<handshake::SpecCommitOp>() {
  return this->commits;
}

template <>
const PlacementList &
SpeculationPlacements::getPlacements<handshake::SpecSaveCommitOp>() {
  return this->saveCommits;
}

inline void parseSpeculatorPlacement(
    std::map<StringRef, llvm::SmallVector<PlacementNames>> &placements,
    const llvm::json::Object *components) {
  if (components->find("speculator") != components->end()) {
    const llvm::json::Object *specObj = components->getObject("speculator");
    StringRef srcOpName = specObj->getString("srcOp").value();
    StringRef dstOpName = specObj->getString("dstOp").value();
    placements["speculator"].push_back({srcOpName, dstOpName});
  }
}

inline void parseOperationPlacements(
    StringRef opType,
    std::map<StringRef, llvm::SmallVector<PlacementNames>> &placements,
    const llvm::json::Object *components) {
  if (components->find(opType) != components->end()) {
    const llvm::json::Array *jsonArray = components->getArray(opType);
    for (const llvm::json::Value &element : *jsonArray) {
      const llvm::json::Object *specObj = element.getAsObject();
      StringRef srcOpName = specObj->getString("srcOp").value();
      StringRef dstOpName = specObj->getString("dstOp").value();
      placements[opType].push_back({srcOpName, dstOpName});
    }
  }
}

bool parseJSON(
    const llvm::json::Value &jsonValue,
    std::map<StringRef, llvm::SmallVector<PlacementNames>> &placements) {
  const llvm::json::Object *components = jsonValue.getAsObject();
  if (!components)
    return false;

  parseSpeculatorPlacement(placements, components);
  parseOperationPlacements("saves", placements, components);
  parseOperationPlacements("commits", placements, components);
  parseOperationPlacements("save-commits", placements, components);
  return true;
}

LogicalResult areOpConnected(Operation *srcOp, Operation *dstOp,
                             Value &srcOpResult) {
  unsigned result_idx = 0;
  srcOpResult = srcOp->getResult(result_idx);
  while (not llvm::is_contained(dstOp->getOperands(), srcOpResult)) {
    result_idx++;
    if (result_idx >= srcOp->getNumResults()) {
      return failure(); // srcOp and dstOp are not connected
    }
    srcOpResult = srcOp->getResult(result_idx);
  }
  return success();
}

LogicalResult getOpPlacements(
    SpeculationPlacements &placements,
    std::map<StringRef, llvm::SmallVector<PlacementNames>> &specNameMap,
    NameAnalysis &nameAnalysis) {
  // Add Speculator operation position
  {
    PlacementNames &p = specNameMap["speculator"].front();
    Operation *srcOp = nameAnalysis.getOp(p.srcOpName);
    Operation *dstOp = nameAnalysis.getOp(p.dstOpName);
    Value srcOpResult;
    if (failed(areOpConnected(srcOp, dstOp, srcOpResult)))
      return failure(); // Operations are not connected
    placements.setSpeculator(srcOpResult, dstOp);
  }

  // Add Save Operations position
  for (PlacementNames &p : specNameMap["saves"]) {
    Operation *srcOp = nameAnalysis.getOp(p.srcOpName);
    Operation *dstOp = nameAnalysis.getOp(p.dstOpName);
    Value srcOpResult;
    if (failed(areOpConnected(srcOp, dstOp, srcOpResult)))
      return failure(); // Operations are not connected

    placements.addSave(srcOpResult, dstOp);
  }

  // Add Commit Operations position
  for (PlacementNames &p : specNameMap["commits"]) {
    Operation *srcOp = nameAnalysis.getOp(p.srcOpName);
    Operation *dstOp = nameAnalysis.getOp(p.dstOpName);
    Value srcOpResult;
    if (failed(areOpConnected(srcOp, dstOp, srcOpResult)))
      return failure(); // Operations are not connected

    placements.addCommit(srcOpResult, dstOp);
  }

  // Add Save-Commit Operations position
  for (PlacementNames &p : specNameMap["save-commits"]) {
    Operation *srcOp = nameAnalysis.getOp(p.srcOpName);
    Operation *dstOp = nameAnalysis.getOp(p.dstOpName);
    Value srcOpResult;
    if (failed(areOpConnected(srcOp, dstOp, srcOpResult)))
      return failure(); // Operations are not connected

    placements.addSaveCommit(srcOpResult, dstOp);
  }

  return success();
}

LogicalResult
SpeculationPlacements::readFromJSON(std::string &jsonPath,
                                    SpeculationPlacements &placements,
                                    NameAnalysis &nameAnalysis) {
  // Open the speculation file
  std::ifstream inputFile(jsonPath);
  if (!inputFile.is_open()) {
    llvm::errs() << "Failed to open unit positions file\n";
    return failure();
  }
  // Read the JSON content from the file and into a string
  std::string jsonString;
  std::string line;
  while (std::getline(inputFile, line))
    jsonString += line;

  // Try to parse the string as a JSON
  llvm::Expected<llvm::json::Value> value = llvm::json::parse(jsonString);
  if (!value) {
    llvm::errs() << "Failed to parse unit positions in \"" << jsonPath
                 << "\"\n";
    return failure();
  }

  // Deserialize into a dictionary for operation names
  llvm::json::Path::Root jsonRoot(jsonPath);
  std::map<StringRef, llvm::SmallVector<PlacementNames>> specNameMap;
  if (!parseJSON(*value, specNameMap))
    return failure();

  return getOpPlacements(placements, specNameMap, nameAnalysis);
}