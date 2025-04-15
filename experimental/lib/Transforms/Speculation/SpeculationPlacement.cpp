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

#include "experimental/Transforms/Speculation/SpeculationPlacement.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/Logging.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include <fstream>
#include <map>
#include <string>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::speculation;

// SpeculationPlacements Methods

void SpeculationPlacements::setSpeculator(OpOperand &dstOpOperand) {
  this->speculator = &dstOpOperand;
}

void SpeculationPlacements::addSave(OpOperand &dstOpOperand) {
  this->saves.insert(&dstOpOperand);
}

void SpeculationPlacements::addCommit(OpOperand &dstOpOperand) {
  this->commits.insert(&dstOpOperand);
}

void SpeculationPlacements::addSaveCommit(OpOperand &dstOpOperand) {
  this->saveCommits.insert(&dstOpOperand);
}

bool SpeculationPlacements::containsCommit(OpOperand &dstOpOperand) {
  return this->commits.contains(&dstOpOperand);
}

bool SpeculationPlacements::containsSave(OpOperand &dstOpOperand) {
  return this->saves.contains(&dstOpOperand);
}

bool SpeculationPlacements::containsSaveCommit(OpOperand &dstOpOperand) {
  return this->saveCommits.contains(&dstOpOperand);
}

void SpeculationPlacements::eraseSave(OpOperand &dstOpOperand) {
  this->saves.erase(&dstOpOperand);
}

void SpeculationPlacements::eraseCommit(OpOperand &dstOpOperand) {
  this->commits.erase(&dstOpOperand);
}

OpOperand &SpeculationPlacements::getSpeculatorPlacement() {
  return *this->speculator;
}

template <>
const llvm::DenseSet<OpOperand *> &
SpeculationPlacements::getPlacements<handshake::SpecSaveOp>() {
  return this->saves;
}

template <>
const llvm::DenseSet<OpOperand *> &
SpeculationPlacements::getPlacements<handshake::SpecCommitOp>() {
  return this->commits;
}

template <>
const llvm::DenseSet<OpOperand *> &
SpeculationPlacements::getPlacements<handshake::SpecSaveCommitOp>() {
  return this->saveCommits;
}

static inline void parseSpeculatorPlacement(
    std::map<StringRef, llvm::SmallVector<PlacementOperand>> &placements,
    const llvm::json::Object *components) {
  if (components->find("speculator") != components->end()) {
    const llvm::json::Object *specObj = components->getObject("speculator");
    StringRef opName = specObj->getString("operation-name").value();
    unsigned opIdx = specObj->getInteger("operand-idx").value();
    placements["speculator"].push_back({opName.str(), opIdx});
  }
}

static inline void parseOperationPlacements(
    StringRef opType,
    std::map<StringRef, llvm::SmallVector<PlacementOperand>> &placements,
    const llvm::json::Object *components) {
  if (components->find(opType) != components->end()) {
    const llvm::json::Array *jsonArray = components->getArray(opType);
    for (const llvm::json::Value &element : *jsonArray) {
      const llvm::json::Object *specObj = element.getAsObject();
      StringRef opName = specObj->getString("operation-name").value();
      unsigned opIdx = specObj->getInteger("operand-idx").value();
      placements[opType].push_back({opName.str(), opIdx});
    }
  }
}

// JSON format example:
// {
//   "speculator": {
//     "operation-name": "fork5",
//     "operand-idx": 0
//   },
//   "saves": [
//     {
//       "operation-name": "mc_load0",
//       "operand-idx": 0
//     },
//     {
//       "operation-name": "mc_load1",
//       "operand-idx": 0
//     }
//   ],
//   "commits": [
//     {
//       "operation-name": "cond_br0",
//       "operand-idx": 1
//     }
//   ],
//   "save-commits": [
//     {
//       "operation-name": "buffer10",
//       "operand-idx": 0
//     }
//   ],
//  "buffers": [
//     {
//       "operation-name": "extsi1",
//       "operand-idx": 0
//     }
//   ]
// }
static bool parseJSON(
    const llvm::json::Value &jsonValue,
    std::map<StringRef, llvm::SmallVector<PlacementOperand>> &placements) {
  const llvm::json::Object *components = jsonValue.getAsObject();
  if (!components)
    return false;

  parseSpeculatorPlacement(placements, components);
  parseOperationPlacements("saves", placements, components);
  parseOperationPlacements("commits", placements, components);
  parseOperationPlacements("save-commits", placements, components);
  parseOperationPlacements("buffers", placements, components);
  return true;
}

static LogicalResult getOpPlacements(
    SpeculationPlacements &placements,
    std::map<StringRef, llvm::SmallVector<PlacementOperand>> &specNameMap,
    NameAnalysis &nameAnalysis) {

  OpOperand *dstOpOperand;

  // Check that operations are found by name
  auto getPlacementOps = [&](PlacementOperand &p) {
    Operation *dstOp = nameAnalysis.getOp(p.opName);
    if (!dstOp) {
      llvm::errs() << "Error: operation name " << p.opName << " is not found\n";
      return failure();
    }
    dstOpOperand = &dstOp->getOpOperand(p.opIdx);
    return success();
  };

  // Add Speculator Operation position
  PlacementOperand &p = specNameMap["speculator"].front();
  if (failed(getPlacementOps(p)))
    return failure();
  placements.setSpeculator(*dstOpOperand);

  // Add Save Operations position
  for (PlacementOperand &p : specNameMap["saves"]) {
    if (failed(getPlacementOps(p)))
      return failure();
    placements.addSave(*dstOpOperand);
  }

  // Add Commit Operations position
  for (PlacementOperand &p : specNameMap["commits"]) {
    if (failed(getPlacementOps(p)))
      return failure();
    placements.addCommit(*dstOpOperand);
  }

  // Add Save-Commit Operations position
  for (PlacementOperand &p : specNameMap["save-commits"]) {
    if (failed(getPlacementOps(p)))
      return failure();
    placements.addSaveCommit(*dstOpOperand);
  }

  return success();
}

LogicalResult
SpeculationPlacements::readFromJSON(const std::string &jsonPath,
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
  std::map<StringRef, llvm::SmallVector<PlacementOperand>> specNameMap;
  if (!parseJSON(*value, specNameMap))
    return failure();

  return getOpPlacements(placements, specNameMap, nameAnalysis);
}
