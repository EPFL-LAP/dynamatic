//===- SubjectGraph.cpp - Exp. support for MAPBUF buffer placement --*- C++
//-*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements Subject Graph constructors.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "experimental/Support/SubjectGraph.h"
#include <algorithm>
#include <iterator>
#include <utility>

using namespace dynamatic::experimental;

// Holds the path to the directory of BLIF files
static inline std::string baseBlifPath;

BaseSubjectGraph::BaseSubjectGraph() = default;

BaseSubjectGraph::BaseSubjectGraph(Operation *op) : op(op) {
  moduleMap[op] = this;
  subjectGraphVector.push_back(this);
  uniqueName = getUniqueName(op);
}

void BaseSubjectGraph::connectInputNodesHelper(
    ChannelSignals &currentSignals,
    BaseSubjectGraph *moduleBeforeSubjectGraph) {

  // Get the output nodes of the module before the current module, by retrieving
  // the result number.
  ChannelSignals &moduleBeforeOutputNodes =
      moduleBeforeSubjectGraph->returnOutputNodes(
          inputSubjectGraphToResultNumber[moduleBeforeSubjectGraph]);

  // Connect ready and valid singals. Only 1 bit each.
  Node::connectNodes(moduleBeforeOutputNodes.readySignal,
                     currentSignals.readySignal);
  Node::connectNodes(currentSignals.validSignal,
                     moduleBeforeOutputNodes.validSignal);

  if (isBlackbox) {
    // If the module is a blackbox, we don't connect the data signals.
    for (auto *node : currentSignals.dataSignals) {
      node->convertIOToChannel();
    }
  } else {
    // Connect data signals. Multiple bits.
    for (unsigned int j = 0; j < currentSignals.dataSignals.size(); j++) {
      Node::connectNodes(currentSignals.dataSignals[j],
                         moduleBeforeOutputNodes.dataSignals[j]);
    }
  }
}

void BaseSubjectGraph::retrieveBlif(std::initializer_list<unsigned int> inputs, std::string to_append){
  std::string moduleType;
  std::string fullPath;
  moduleType = op->getName().getStringRef();
  // Erase the dialect name from the moduleType
  moduleType = moduleType.substr(moduleType.find('.') + 1) + to_append;
  
  fullPath = baseBlifPath + "/" + moduleType + "/";

  for (int input : inputs) {
    fullPath += std::to_string(input) + "/";
  }

  fullPath += moduleType + ".blif";
  
  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);
}

// Assigns signals to the variables in ChannelSignals struct
void assignSignals(ChannelSignals &signals, Node *node,
                   const std::string &nodeName) {
  if (nodeName.find("valid") != std::string::npos) {
    signals.validSignal = node;
  } else if (nodeName.find("ready") != std::string::npos) {
    signals.readySignal = node;
  } else {
    signals.dataSignals.push_back(node);
  }
};

// Populate inputSubjectGraphs and outputSubjectGraphs after all of the
// Subject Graphs are created. Retrieves the Result Numbers.
void BaseSubjectGraph::buildSubjectGraphConnections() {
  for (Value inputOperand : op->getOperands()) {
    if (Operation *definingOp = inputOperand.getDefiningOp()) { // Block Arguments has no Defining Operation
      auto *inputSubjectGraph = moduleMap[inputOperand.getDefiningOp()];
      inputSubjectGraphs.push_back(inputSubjectGraph);
      inputSubjectGraphToResultNumber[inputSubjectGraph] =
          inputOperand.cast<OpResult>().getResultNumber();
    }
  }

  for (Value outputOperand : op->getResults()) {
    for (Operation *user : outputOperand.getUsers()) {
      auto *outputSubjectGraph = moduleMap[user];
      outputSubjectGraphs.push_back(outputSubjectGraph);
    }
  }
}

void changeIO(BaseSubjectGraph *newIO, BaseSubjectGraph *prevIO,
              std::vector<BaseSubjectGraph *> &inputOutput) {
  auto it = std::find(inputOutput.begin(), inputOutput.end(), prevIO);
  if (it != inputOutput.end()) {
    // Delete the output from the vector and insert the new output. 
    auto index = std::distance(inputOutput.begin(), it);
    inputOutput.erase(it);
    inputOutput.insert(inputOutput.begin() + index, newIO);
  } else {
    llvm::errs() << "Output not found\n";
  }
}

// ArithSubjectGraph implementation
ArithSubjectGraph::ArithSubjectGraph(Operation *op)
    : BaseSubjectGraph(op) {
  dataWidth = handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());

  retrieveBlif({dataWidth});

  // Ops are mapped to DSP slices if the bitwidth is greater than 4
  if ((dataWidth > 4) &&
      (llvm::isa<handshake::CmpIOp>(op) || llvm::isa<handshake::AddIOp>(op) ||
       llvm::isa<handshake::SubIOp>(op) || llvm::isa<handshake::MulIOp>(op) ||
       llvm::isa<handshake::DivSIOp>(op) ||
       llvm::isa<handshake::DivUIOp>(op))) {
    isBlackbox = true;
  }

  // Assign the nodes to correct signals
  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("result") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
      if (isBlackbox && (nodeName.find("valid") == std::string::npos &&
                         nodeName.find("ready") == std::string::npos)) {
        node->isBlackboxOutput = (true);
      }
    } else if (nodeName.find("lhs") != std::string::npos) {
      assignSignals(lhsNodes, node, nodeName);
    } else if (nodeName.find("rhs") != std::string::npos) {
      assignSignals(rhsNodes, node, nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

void ArithSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(lhsNodes, inputSubjectGraphs[0]);
  connectInputNodesHelper(rhsNodes, inputSubjectGraphs[1]);
}

ChannelSignals &ArithSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// ForkSubjectGraph implementation
ForkSubjectGraph::ForkSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  size = op->getNumResults();
  dataWidth = handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
  outputNodes.resize(size);

  if (dataWidth == 0) {
    retrieveBlif({size}, "_dataless");
  } else {
    retrieveBlif({size, dataWidth}, "_type");
  }

  auto generateNewNameRV =
      [](const std::string &nodeName) -> std::pair<std::string, unsigned int> {
    std::string newName = nodeName;
    std::string number;
    size_t bracketPos = newName.find('[');
    if (bracketPos != std::string::npos) {
      number = newName.substr(bracketPos + 1);
      number = number.substr(0, number.find_first_not_of("0123456789"));
    }
    size_t readyPos = newName.find("ready");
    size_t validPos = newName.find("valid");
    if (readyPos != std::string::npos) {
      newName = newName.substr(0, readyPos) + number + "_ready";
    } else if (validPos != std::string::npos) {
      newName = newName.substr(0, validPos) + number + "_valid";
    }
    return {newName, std::stoi(number)};
  };

  auto generateNewNameData =
      [&](const std::string &nodeName) -> std::pair<std::string, unsigned int> {
    std::string newName = nodeName;
    std::string number;
    size_t bracketPos = newName.find('[');
    if (bracketPos != std::string::npos) {
      number = newName.substr(bracketPos + 1);
      number = number.substr(0, number.find_first_not_of("0123456789"));
      newName = newName.substr(0, bracketPos);
    }
    unsigned int num = std::stoi(number);
    unsigned int newNumber = num / dataWidth;
    unsigned int remainder = (num % dataWidth);
    return {newName + "_" + std::to_string(newNumber) + "[" +
                std::to_string(remainder) + "]",
            newNumber};
  };

  auto generateNewName =
      [&](const std::string &nodeName) -> std::pair<std::string, unsigned int> {
    if (nodeName.find("ready") != std::string::npos ||
        nodeName.find("valid") != std::string::npos) {
      return generateNewNameRV(nodeName);
    }
    return generateNewNameData(nodeName);
  };

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("outs") != std::string::npos) {
      auto [newName, num] = generateNewName(nodeName);
      assignSignals(outputNodes[num], node, newName);
      node->name = (uniqueName + "_" + newName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    } else if (nodeName.find("ins") != std::string::npos &&
               (node->isInput || node->isOutput)) {
      assignSignals(inputNodes, node, nodeName);
    }
  }
}

void ForkSubjectGraph::connectInputNodes() {
  if (!inputSubjectGraphs.empty()) {
    connectInputNodesHelper(inputNodes, inputSubjectGraphs[0]);
  }
}

ChannelSignals &ForkSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return outputNodes[channelIndex];
}

// MuxSubjectGraph implementation
MuxSubjectGraph::MuxSubjectGraph(Operation *op) : BaseSubjectGraph(op)  {
  auto muxOp = llvm::dyn_cast<handshake::MuxOp>(op);
  size = muxOp.getDataOperands().size();
  dataWidth = handshake::getHandshakeTypeBitWidth(muxOp.getResult().getType());
  selectType =
      handshake::getHandshakeTypeBitWidth(muxOp.getSelectOperand().getType());
  inputNodes.resize(size);

  retrieveBlif({size, dataWidth});

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput || node->isOutput)) {
      size_t bracketPos = nodeName.find('[');
      std::string number = nodeName.substr(bracketPos + 1);
      number = number.substr(0, number.find_first_not_of("0123456789"));
      unsigned int num = std::stoi(number);
      if (nodeName.find("ready") == std::string::npos &&
          nodeName.find("valid") == std::string::npos) {
        num = num / dataWidth;
      }
      assignSignals(inputNodes[num], node, nodeName);
    } else if (nodeName.find("index") != std::string::npos) {
      assignSignals(indexNodes, node, nodeName);
    } else if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

void MuxSubjectGraph::connectInputNodes() {
  // index is the first input
  connectInputNodesHelper(indexNodes, inputSubjectGraphs[0]);

  for (unsigned int i = 0; i < inputNodes.size(); i++) {
    connectInputNodesHelper(inputNodes[i], inputSubjectGraphs[i + 1]);
  }
}

ChannelSignals &MuxSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return outputNodes;
}

// ControlMergeSubjectGraph implementation
ControlMergeSubjectGraph::ControlMergeSubjectGraph(Operation *op)
    : BaseSubjectGraph(op) {
  auto cmergeOp = llvm::dyn_cast<handshake::ControlMergeOp>(op);
  size = cmergeOp.getDataOperands().size();
  dataWidth =
      handshake::getHandshakeTypeBitWidth(cmergeOp.getResult().getType());
  indexType =
      handshake::getHandshakeTypeBitWidth(cmergeOp.getIndex().getType());
  inputNodes.resize(size);

  if (dataWidth == 0) {
    retrieveBlif({size, indexType}, "_dataless");
  } else {
    op->emitError("Operation Unsupported");
  }

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput || node->isOutput)) {
      if (size == 1) {
        assignSignals(inputNodes[0], node, nodeName);
        continue;
      }
      size_t bracketPos = nodeName.find('[');
      std::string number = nodeName.substr(bracketPos + 1);
      number = number.substr(0, number.find_first_not_of("0123456789"));
      unsigned int num = std::stoi(number);
      if (nodeName.find("ready") == std::string::npos &&
          nodeName.find("valid") == std::string::npos) {
        num = num / dataWidth;
      }
      assignSignals(inputNodes[num], node, nodeName);
    } else if (nodeName.find("index") != std::string::npos) {
      assignSignals(indexNodes, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

void ControlMergeSubjectGraph::connectInputNodes() {
  for (unsigned int i = 0; i < inputNodes.size(); i++) {
    connectInputNodesHelper(inputNodes[i], inputSubjectGraphs[i]);
  }
}

ChannelSignals &
ControlMergeSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return (channelIndex == 0) ? outputNodes : indexNodes;
}

// ConditionalBranchSubjectGraph implementation
ConditionalBranchSubjectGraph::ConditionalBranchSubjectGraph(Operation *op)
    : BaseSubjectGraph(op) {
  auto cbrOp = llvm::dyn_cast<handshake::ConditionalBranchOp>(op);
  dataWidth =
      handshake::getHandshakeTypeBitWidth(cbrOp.getDataOperand().getType());

  if (dataWidth == 0) {
    retrieveBlif({}, "_dataless");
  } else {
    retrieveBlif({dataWidth});
  }

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("true") != std::string::npos) {
      assignSignals(trueOut, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find("false") != std::string::npos) {
      assignSignals(falseOut, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find("condition") != std::string::npos) {
      assignSignals(conditionNodes, node, nodeName);
    } else if (nodeName.find("data") != std::string::npos) {
      assignSignals(inputNodes, node, nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

void ConditionalBranchSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(conditionNodes, inputSubjectGraphs[0]);
  connectInputNodesHelper(inputNodes, inputSubjectGraphs[1]);
}

ChannelSignals &
ConditionalBranchSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return (channelIndex == 0) ? trueOut : falseOut;
}

// SourceSubjectGraph implementation
SourceSubjectGraph::SourceSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  retrieveBlif({});

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

void SourceSubjectGraph::connectInputNodes() {
  // No input nodes
}

ChannelSignals &SourceSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// LoadSubjectGraph implementation
LoadSubjectGraph::LoadSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  auto loadOp = llvm::dyn_cast<handshake::LoadOp>(op);
  dataWidth =
      handshake::getHandshakeTypeBitWidth(loadOp.getDataInput().getType());
  addrType =
      handshake::getHandshakeTypeBitWidth(loadOp.getAddressInput().getType());
  
  retrieveBlif({addrType, dataWidth});

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("addrIn") != std::string::npos) {
      assignSignals(addrInSignals, node, nodeName);
    } else if (nodeName.find("addrOut") != std::string::npos) {
      assignSignals(addrOutSignals, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find("dataOut") != std::string::npos) {
      assignSignals(dataOutSignals, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

void LoadSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(addrInSignals, inputSubjectGraphs[0]);
}

ChannelSignals &LoadSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return (channelIndex == 0) ? addrOutSignals : dataOutSignals;
}

// StoreSubjectGraph implementation
StoreSubjectGraph::StoreSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  auto storeOp = llvm::dyn_cast<handshake::StoreOp>(op);
  dataWidth =
      handshake::getHandshakeTypeBitWidth(storeOp.getDataInput().getType());
  addrType =
      handshake::getHandshakeTypeBitWidth(storeOp.getAddressInput().getType());

  retrieveBlif({addrType, dataWidth});

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("dataIn") != std::string::npos) {
      assignSignals(dataInSignals, node, nodeName);
    } else if (nodeName.find("addrIn") != std::string::npos) {
      assignSignals(addrInSignals, node, nodeName);
    } else if (nodeName.find("addrOut") != std::string::npos) {
      assignSignals(addrOutSignals, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

void StoreSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(addrInSignals, inputSubjectGraphs[0]);
  connectInputNodesHelper(dataInSignals, inputSubjectGraphs[1]);
}

ChannelSignals &StoreSubjectGraph::returnOutputNodes(unsigned int) {
  return addrOutSignals;
}

// ConstantSubjectGraph implementation
ConstantSubjectGraph::ConstantSubjectGraph(Operation *op)
    : BaseSubjectGraph(op) {

  auto cstOp = llvm::dyn_cast<handshake::ConstantOp>(op);
  handshake::ChannelType cstType = cstOp.getResult().getType();
  dataWidth = cstType.getDataBitWidth();

  retrieveBlif({dataWidth});

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("outs") != std::string::npos) {
      node->name = (uniqueName + "_" + nodeName);
      assignSignals(outputNodes, node, nodeName);
    } else if (nodeName.find("ctrl") != std::string::npos) {
      assignSignals(controlSignals, node, nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

void ConstantSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(controlSignals, inputSubjectGraphs[0]);
}

ChannelSignals &ConstantSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// ExtTruncSubjectGraph implementation
ExtTruncSubjectGraph::ExtTruncSubjectGraph(Operation *op)
    : BaseSubjectGraph(op) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::ExtFOp,
            handshake::TruncIOp, handshake::TruncFOp>([&](auto extOp) {
        inputWidth =
            handshake::getHandshakeTypeBitWidth(extOp.getOperand().getType());
        outputWidth =
            handshake::getHandshakeTypeBitWidth(extOp.getResult().getType());
      });
      
  retrieveBlif({inputWidth, outputWidth});

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput || node->isOutput)) {
      assignSignals(inputNodes, node, nodeName);
    } else if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

void ExtTruncSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(inputNodes, inputSubjectGraphs[0]);
}

ChannelSignals &
ExtTruncSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return outputNodes;
}

// SelectSubjectGraph implementation
SelectSubjectGraph::SelectSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  auto selectOp = llvm::dyn_cast<handshake::SelectOp>(op);
  dataWidth =
      handshake::getHandshakeTypeBitWidth(selectOp->getOperand(1).getType());

  retrieveBlif({dataWidth});


  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("condition") != std::string::npos &&
        (node->isInput || node->isOutput)) {
      assignSignals(condition, node, nodeName);
    } else if (nodeName.find("trueValue") != std::string::npos &&
               (node->isInput || node->isOutput)) {
      assignSignals(trueValue, node, nodeName);
    } else if (nodeName.find("falseValue") != std::string::npos &&
               (node->isInput || node->isOutput)) {
      assignSignals(falseValue, node, nodeName);
    } else if (nodeName.find("result") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

void SelectSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(condition, inputSubjectGraphs[0]);
  connectInputNodesHelper(trueValue, inputSubjectGraphs[1]);
  connectInputNodesHelper(falseValue, inputSubjectGraphs[2]);
}

ChannelSignals &
SelectSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return outputNodes;
}

MergeSubjectGraph::MergeSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  auto mergeOp = llvm::dyn_cast<handshake::MergeOp>(op);
  size = mergeOp.getDataOperands().size();
  dataWidth = handshake::getHandshakeTypeBitWidth(
      mergeOp.getDataOperands()[0].getType());
  inputNodes.resize(size);

  retrieveBlif({size, dataWidth});

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput || node->isOutput)) {
      if (size == 1) {
        assignSignals(inputNodes[0], node, nodeName);
        continue;
      }
      size_t bracketPos = nodeName.find('[');
      std::string number = nodeName.substr(bracketPos + 1);
      number = number.substr(0, number.find_first_not_of("0123456789"));
      unsigned int num = std::stoi(number);
      if (nodeName.find("ready") == std::string::npos &&
          nodeName.find("valid") == std::string::npos) {
        num = num / dataWidth;
      }
      assignSignals(inputNodes[num], node, nodeName);
    } else if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos ||
               nodeName.find("dataReg") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

void MergeSubjectGraph::connectInputNodes() {
  for (unsigned int i = 0; i < inputNodes.size(); i++) {
    connectInputNodesHelper(inputNodes[i], inputSubjectGraphs[i]);
  }
}

ChannelSignals &MergeSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// BranchSinkSubjectGraph implementation
BranchSinkSubjectGraph::BranchSinkSubjectGraph(Operation *op)
    : BaseSubjectGraph(op) {
  dataWidth = handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());

  if (dataWidth == 0) {
    retrieveBlif({}, "_dataless");
  } else {
    retrieveBlif({dataWidth});
  }

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput || node->isOutput)) {
      assignSignals(inputNodes, node, nodeName);
    } else if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos ||
               nodeName.find("dataReg") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

void BranchSinkSubjectGraph::connectInputNodes() {
  if (!inputSubjectGraphs.empty()) {
    connectInputNodesHelper(inputNodes, inputSubjectGraphs[0]);
  }
}

ChannelSignals &BranchSinkSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

void BufferSubjectGraph::initBuffer() {
  std::string fullPath = baseBlifPath;
  if (dataWidth == 0) {
    bufferType += "_dataless";
    fullPath += "/" + bufferType + "/" + bufferType + ".blif";
  } else {
    fullPath += "/" + bufferType + "/" + std::to_string(dataWidth) + "/" + bufferType + ".blif";
  }

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput || node->isOutput)) {
      assignSignals(inputNodes, node, nodeName);
    } else if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->name = (uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos ||
               nodeName.find("dataReg") != std::string::npos) {
      node->name = (uniqueName + "." + nodeName);
    }
  }
}

// BufferSubjectGraph implementations
BufferSubjectGraph::BufferSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  auto bufferOp = llvm::dyn_cast<handshake::BufferOp>(op);
  auto params =
      bufferOp->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
  auto optTiming = params.getNamed(handshake::BufferOp::TIMING_ATTR_NAME);

  if (auto timing = dyn_cast<handshake::TimingAttr>(optTiming->getValue())) {
    handshake::TimingInfo info = timing.getInfo();
    if (info == handshake::TimingInfo::oehb())
      bufferType = "oehb";
    if (info == handshake::TimingInfo::tehb())
      bufferType = "tehb";
  }

  dataWidth = handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
  initBuffer();
}

void BufferSubjectGraph::insertBuffer(BaseSubjectGraph *graph1,
                                      BaseSubjectGraph *graph2) {
  subjectGraphVector.push_back(this);
  inputSubjectGraphs.push_back(graph1);
  outputSubjectGraphs.push_back(graph2);

  unsigned int channelNum = graph2->inputSubjectGraphToResultNumber[graph1];
  inputSubjectGraphToResultNumber[graph1] = channelNum;
  graph2->inputSubjectGraphToResultNumber.erase(graph1);
  graph2->inputSubjectGraphToResultNumber[this] = channelNum;

  ChannelSignals &channel = graph1->returnOutputNodes(channelNum);
  dataWidth = channel.dataSignals.size();
 
  changeIO(this, graph1, graph2->inputSubjectGraphs);
  changeIO(this, graph2, graph1->outputSubjectGraphs);
}

BufferSubjectGraph::BufferSubjectGraph(Operation *op1, Operation *op2,
                                       std::string bufferTypeName)
    : BaseSubjectGraph() {
  static unsigned int bufferCount;
  uniqueName = "oehb_" + std::to_string(bufferCount++);
  bufferType = std::move(bufferTypeName);

  BaseSubjectGraph *graph1 = moduleMap[op1];
  BaseSubjectGraph *graph2 = moduleMap[op2];

  insertBuffer(graph1, graph2);
  initBuffer();
}

BufferSubjectGraph::BufferSubjectGraph(BufferSubjectGraph *graph1,
                                       Operation *op2, std::string bufferTypeName)
    : BaseSubjectGraph() {
  static unsigned int bufferCount;
  uniqueName = "tehb_" + std::to_string(bufferCount++);
  bufferType = std::move(bufferTypeName);

  BaseSubjectGraph *graph2 = moduleMap[op2];

  insertBuffer(graph1, graph2);
  initBuffer();
}

void BufferSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(inputNodes, inputSubjectGraphs[0]);
}

ChannelSignals &BufferSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// SubjectGraphGenerator implementation
SubjectGraphGenerator::SubjectGraphGenerator(handshake::FuncOp funcOp,
                                             StringRef blifFiles) {
  baseBlifPath = blifFiles;
  std::vector<BaseSubjectGraph *> subjectGraphs;

  funcOp.walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::AddIOp, handshake::AndIOp, handshake::CmpIOp,
            handshake::OrIOp, handshake::ShLIOp, handshake::ShRSIOp,
            handshake::ShRUIOp, handshake::SubIOp, handshake::XOrIOp,
            handshake::MulIOp, handshake::DivSIOp, handshake::DivUIOp>([&](auto) { 
        subjectGraphs.push_back(new ArithSubjectGraph(op)); 
      })
      .Case<handshake::BranchOp, handshake::SinkOp>([&](auto) {
        subjectGraphs.push_back(new BranchSinkSubjectGraph(op));
      })
      .Case<handshake::BufferOp, handshake::SinkOp>([&](auto) { 
        subjectGraphs.push_back(new BufferSubjectGraph(op)); 
      })
      .Case<handshake::ConditionalBranchOp>([&](handshake::ConditionalBranchOp cbrOp) {
        subjectGraphs.push_back(new ConditionalBranchSubjectGraph(op));
      })
      .Case<handshake::ConstantOp>([&](handshake::ConstantOp cstOp) {
        subjectGraphs.push_back(new ConstantSubjectGraph(op));
      })
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp cmergeOp) {
        subjectGraphs.push_back(new ControlMergeSubjectGraph(op));
      })
      .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::ExtFOp,
            handshake::TruncIOp, handshake::TruncFOp>([&](auto) {
        subjectGraphs.push_back(new ExtTruncSubjectGraph(op));
      })
      .Case<handshake::ForkOp, handshake::LazyForkOp>([&](auto) { 
        subjectGraphs.push_back(new ForkSubjectGraph(op)); 
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        subjectGraphs.push_back(new MuxSubjectGraph(op));
      })
      .Case<handshake::MergeOp>([&](auto) { 
        subjectGraphs.push_back(new MergeSubjectGraph(op)); 
      })
      .Case<handshake::LoadOp>([&](handshake::LoadOp loadOp) {
        subjectGraphs.push_back(new LoadSubjectGraph(op));
      })
      .Case<handshake::SelectOp>([&](handshake::SelectOp selectOp) {
        subjectGraphs.push_back(new SelectSubjectGraph(op));
      })
      .Case<handshake::SourceOp>([&](auto) { 
        subjectGraphs.push_back(new SourceSubjectGraph(op)); 
      })
      .Case<handshake::StoreOp>([&](handshake::StoreOp storeOp) {
        subjectGraphs.push_back(new StoreSubjectGraph(op));
      })
      .Default([&](auto) { return; });
  });

  // Populate Subject Graph vectors
  for (auto *module : subjectGraphs) {
    module->buildSubjectGraphConnections();
  }
}
