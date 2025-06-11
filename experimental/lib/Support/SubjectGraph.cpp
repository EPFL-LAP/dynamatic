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

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
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

using namespace dynamatic;
using namespace dynamatic::experimental;

// Holds the path to the directory of BLIF files
static inline std::string baseBlifPath;

// Default constructor, used for Subject Graph creation without an MLIR
// Operation.
BaseSubjectGraph::BaseSubjectGraph() { subjectGraphVector.push_back(this); };

// Constructor for a SubjectGraph based on an MLIR Operation.
BaseSubjectGraph::BaseSubjectGraph(Operation *op) : op(op) {
  moduleMap[op] = this;
  subjectGraphVector.push_back(this);
  uniqueName = getUniqueName(op);
}

// Helper function to connect PIs of the Subject Graph with POs of the Subject
// Graph of the preceding module.
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

// Constructs the file path based on Operation name and parameters, calls the
// Blif parser to load the Blif file
void BaseSubjectGraph::loadBlifFile(std::initializer_list<unsigned int> inputs,
                                    std::string toAppend) {
  std::string moduleType;
  std::string fullPath;
  moduleType = op->getName().getStringRef();
  // Erase the dialect name from the moduleType
  moduleType = moduleType.substr(moduleType.find('.') + 1) + toAppend;

  // Append moduleType to the base path
  fullPath = baseBlifPath + "/" + moduleType + "/";

  // Append inputs to the path
  for (int input : inputs) {
    fullPath += std::to_string(input) + "/";
  }

  // Append file format
  fullPath += moduleType + ".blif";

  // Call the parser to load and parse the Blif file
  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);
}

// Assigns signals to the variables in ChannelSignals struct
void assignSignals(ChannelSignals &signals, Node *node,
                   const std::string &nodeName) {
  // If nodeName includes "valid" or "ready", assign it to the respective
  // signal. If it does not, assign it to the data signals.
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
  // Loop over the input operands of the operation to find the input Ops.
  for (Value inputOperand : op->getOperands()) {
    // Block Arguments has no Defining Operation
    if (Operation *definingOp = inputOperand.getDefiningOp()) {
      // Add the Subject Graph of the defining Op to the inputSubjectGraphs.
      auto *inputSubjectGraph = moduleMap[inputOperand.getDefiningOp()];
      inputSubjectGraphs.push_back(inputSubjectGraph);
      // Store the Result Number of the input operand in the
      // inputSubjectGraphToResultNumber map.
      inputSubjectGraphToResultNumber[inputSubjectGraph] =
          inputOperand.cast<OpResult>().getResultNumber();
    }
  }

  // Loop over the output operands of the operation to find the output Ops.
  for (Value outputOperand : op->getResults()) {
    for (Operation *user : outputOperand.getUsers()) {
      // Add the Subject Graph of the user Op to the outputSubjectGraphs.
      auto *outputSubjectGraph = moduleMap[user];
      outputSubjectGraphs.push_back(outputSubjectGraph);
    }
  }
}

// Processes nodes in the BLIF file based on the rules provided.
void BaseSubjectGraph::processNodesWithRules(
    const std::vector<NodeProcessingRule> &rules) {
  for (auto &node : blifData->getAllNodes()) {
    std::string nodeName = node->name;
    for (const auto &rule : rules) {
      if (nodeName.find(rule.pattern) != std::string::npos &&
          (node->isInput || node->isOutput)) {
        assignSignals(rule.signals, node, nodeName);
        if (rule.renameNode) // change the name of the node if set true
          node->name = uniqueName + "_" + nodeName;
        if (rule.extraProcessing) // apply extra processing to node if a
                                  // function is given
          rule.extraProcessing(node);
      } else if (nodeName.find(".") != std::string::npos ||
                 nodeName.find("dataReg") !=
                     std::string::npos) { // Nodes with "." and "dataReg"
                                          // require unique naming to avoid
                                          // naming conflicts
        node->name = (uniqueName + "." + nodeName);
      }
    }
  }
}

// Inserts a new SubjectGraph in between two existing SubjectGraphs.
void BaseSubjectGraph::insertNewSubjectGraph(BaseSubjectGraph *predecessorGraph,
                                             BaseSubjectGraph *successorGraph) {
  // Add predecessorGraph and successorGraph to the input/output vectors of the
  // new SubjectGraph.
  inputSubjectGraphs.push_back(predecessorGraph);
  outputSubjectGraphs.push_back(successorGraph);

  // Get the channel number between the predecessorGraph and successorGraph.
  unsigned int resultNum =
      successorGraph->inputSubjectGraphToResultNumber[predecessorGraph];
  inputSubjectGraphToResultNumber[predecessorGraph] = resultNum;

  // Remove predecessorGraph from the inputSubjectGraphs of the successorGraph
  // and add the new SubjectGraph.
  successorGraph->inputSubjectGraphToResultNumber.erase(predecessorGraph);
  successorGraph->inputSubjectGraphToResultNumber[this] = resultNum;

  // Delete the predecessor and succesor graphs from the input/output vectors of
  // each other and insert the new SubjectGraph.
  auto changeIO = [&](BaseSubjectGraph *prevIO,
                      std::vector<BaseSubjectGraph *> &inputOutput) {
    auto it = std::find(inputOutput.begin(), inputOutput.end(), prevIO);
    auto index = std::distance(inputOutput.begin(), it);
    inputOutput.erase(it);
    inputOutput.insert(inputOutput.begin() + index, this);
  };

  changeIO(predecessorGraph, successorGraph->inputSubjectGraphs);
  changeIO(successorGraph, predecessorGraph->outputSubjectGraphs);
}

// ArithSubjectGraph implementation
ArithSubjectGraph::ArithSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  // Get datawidth of the operation
  dataWidth = handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());

  loadBlifFile({dataWidth});

  // Ops are mapped to DSP slices if the bitwidth is greater than 4
  if ((dataWidth > 4) &&
      (llvm::isa<handshake::CmpIOp>(op) || llvm::isa<handshake::AddIOp>(op) ||
       llvm::isa<handshake::SubIOp>(op) || llvm::isa<handshake::MulIOp>(op) ||
       llvm::isa<handshake::DivSIOp>(op) ||
       llvm::isa<handshake::DivUIOp>(op))) {
    isBlackbox = true;
  }

  // Data signal nodes of blackbox modules need to be set as Blackbox Outputs.
  // Valid and Ready signals are not blackboxed, so they are not set.
  auto setBlackboxBool = [&](Node *node) {
    std::string nodeName = node->name;
    if (isBlackbox && (nodeName.find("valid") == std::string::npos &&
                       nodeName.find("ready") == std::string::npos)) {
      node->isBlackboxOutput = (true);
    }
  };

  std::vector<NodeProcessingRule> rules = {
      {"lhs", lhsNodes, false, nullptr},
      {"rhs", rhsNodes, false, nullptr},
      {"result", resultNodes, true, setBlackboxBool}};

  processNodesWithRules(rules);
}

// lhsNodes are connected to the Subject Graph with index 0, and rhsNodes
// are connected to the Subject Graph with index 1.
void ArithSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(lhsNodes, inputSubjectGraphs[0]);
  connectInputNodesHelper(rhsNodes, inputSubjectGraphs[1]);
}

// Arith modules only have resultNodes as output.
ChannelSignals &ArithSubjectGraph::returnOutputNodes(unsigned int) {
  return resultNodes;
}

void ForkSubjectGraph::processOutOfRuleNodes() {
  // Generates new names for ready and valid signals of the fork module. Since
  // the output signals of the fork module in the Verilog description are
  // implemented as a single array, we divide them into different arrays here.
  auto generateNewNameReadyValid =
      [](const std::string &nodeName) -> std::pair<std::string, unsigned int> {
    std::string newName = nodeName;
    std::string number;
    size_t bracketPos = newName.find('[');
    // get the index of the bit. for example, if node has name outs[7]_ready,
    // number variable will have 7.
    if (bracketPos != std::string::npos) {
      number = newName.substr(bracketPos + 1);
      number = number.substr(0, number.find_first_not_of("0123456789"));
    }
    size_t readyPos = newName.find("ready");
    size_t validPos = newName.find("valid");
    // If the nodeName has "ready" or "valid", append it.
    if (readyPos != std::string::npos) {
      newName = newName.substr(0, readyPos) + number + "_ready";
    } else if (validPos != std::string::npos) {
      newName = newName.substr(0, validPos) + number + "_valid";
    }
    // Return the newName and index of the bit
    return {newName, std::stoi(number)};
  };

  // If fork has 3 outputs with datawidth of 10, in the .blif file output
  // nodes will be named as outs[0], outs[1], ... outs[29]. Here, we divide
  // the output nodes into 3 arrays, each with datawidth of 10. So, the new
  // names will be outs_0[0], outs_0[1], ..., outs_0[9], outs_1[0], outs_1[1],
  // ..., outs_1[9], outs_2[0], outs_2[1], ..., outs_2[9].
  auto generateNewNameData =
      [&](const std::string &nodeName) -> std::pair<std::string, unsigned int> {
    std::string newName = nodeName;
    std::string number;
    size_t bracketPos = newName.find('[');
    // get the index of the bit. for example, if node has name outs[7],
    // number variable will have 7.
    if (bracketPos != std::string::npos) {
      number = newName.substr(bracketPos + 1);
      number = number.substr(0, number.find_first_not_of("0123456789"));
      newName = newName.substr(0, bracketPos);
    }
    unsigned int num = std::stoi(number);
    // If the nodeName is outs[17], and dataWidth is 10, newNumber will be 1,
    // and remainder will be 7. Meaning it is the 8th bit of the 2nd output
    // (first are 0th)
    unsigned int newNumber = num / dataWidth;
    unsigned int remainder = (num % dataWidth);
    return {newName + "_" + std::to_string(newNumber) + "[" +
                std::to_string(remainder) + "]",
            newNumber};
  };

  // Generate new names for the output nodes based on the type of the node,
  // whether it is an RV/Data.
  auto generateNewName =
      [&](const std::string &nodeName) -> std::pair<std::string, unsigned int> {
    if (nodeName.find("ready") != std::string::npos ||
        nodeName.find("valid") != std::string::npos) {
      return generateNewNameReadyValid(nodeName);
    }
    return generateNewNameData(nodeName);
  };

  // Loop over all nodes in the BLIF data and assign signals to the
  // outputNodes based on the generated names.
  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->name;
    if (nodeName.find("outs") != std::string::npos) {
      auto [newName, num] = generateNewName(nodeName);
      assignSignals(outputNodes[num], node, newName);
      node->name = (uniqueName + "_" + newName);
    }
  }
}

// ForkSubjectGraph implementation
ForkSubjectGraph::ForkSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  // Get the size and datawidth parameters of the fork module
  size = op->getNumResults();
  dataWidth = handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
  outputNodes.resize(size);

  if (dataWidth == 0) {
    loadBlifFile({size}, "_dataless");
  } else {
    loadBlifFile({size, dataWidth}, "_type");
  }

  // "outs" case does not obey the rules
  processOutOfRuleNodes();

  std::vector<NodeProcessingRule> rules = {{"ins", inputNodes, false, nullptr}};

  processNodesWithRules(rules);
}

// Fork module only have inputNodes as input.
void ForkSubjectGraph::connectInputNodes() {
  // In the cases where fork modules are Block Arguments, they
  // do not have any input Operations.
  if (!inputSubjectGraphs.empty()) {
    connectInputNodesHelper(inputNodes, inputSubjectGraphs[0]);
  }
}

// Return outputNodes of the ForkSubjectGraph, corresponding to the
// channelIndex provided.
ChannelSignals &ForkSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return outputNodes[channelIndex];
}

void MuxSubjectGraph::processOutOfRuleNodes() {
  // Similar to the ForkSubjectGraph, the Mux module has
  // output nodes named as ins[0], ins[1], ..., ins[size-1]. This function
  // assigns the signals to the inputNodes based on the index of the input
  // node.
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
    }
  }
}

// MuxSubjectGraph implementation
MuxSubjectGraph::MuxSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  auto muxOp = llvm::dyn_cast<handshake::MuxOp>(op);
  // Get the size, datawidth and select type parameters of the mux module
  size = muxOp.getDataOperands().size();
  dataWidth = handshake::getHandshakeTypeBitWidth(muxOp.getResult().getType());
  selectType =
      handshake::getHandshakeTypeBitWidth(muxOp.getSelectOperand().getType());

  inputNodes.resize(size);

  loadBlifFile({size, dataWidth});

  // "ins" case does not obey the rules
  processOutOfRuleNodes();

  std::vector<NodeProcessingRule> rules = {
      {"index", indexNodes, false, nullptr},
      {"outs", outputNodes, true, nullptr}};

  processNodesWithRules(rules);
}

// indexNodes are connected to the first inputSubjectGraph, and rest are
// connected to the inputSubjectGraphs in the order they are defined.
void MuxSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(indexNodes, inputSubjectGraphs[0]);

  for (unsigned int i = 0; i < inputNodes.size(); i++) {
    connectInputNodesHelper(inputNodes[i], inputSubjectGraphs[i + 1]);
  }
}

// Mux module only have outputNodes as output.
ChannelSignals &MuxSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

void ControlMergeSubjectGraph::processOutOfRuleNodes() {
  // Similar to the ForkSubjectGraph, the ControlMerge module has
  // output nodes named as ins[0], ins[1], ..., ins[size-1]. This function
  // assigns the signals to the inputNodes based on the index of the input
  // node.
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
    }
  }
}

// ControlMergeSubjectGraph implementation
ControlMergeSubjectGraph::ControlMergeSubjectGraph(Operation *op)
    : BaseSubjectGraph(op) {
  auto cmergeOp = llvm::dyn_cast<handshake::ControlMergeOp>(op);
  // Get the size, datawidth and index type parameters of the control merge
  size = cmergeOp.getDataOperands().size();
  dataWidth =
      handshake::getHandshakeTypeBitWidth(cmergeOp.getResult().getType());
  indexType =
      handshake::getHandshakeTypeBitWidth(cmergeOp.getIndex().getType());

  inputNodes.resize(size);

  if (dataWidth == 0) {
    loadBlifFile({size, indexType}, "_dataless");
  } else {
    op->emitError("Operation Unsupported");
  }

  // "ins" case does not obey the rules
  processOutOfRuleNodes();

  std::vector<NodeProcessingRule> rules = {
      {"index", indexNodes, false, nullptr},
      {"outs", outputNodes, true, nullptr}};

  processNodesWithRules(rules);
}

// Connects the inputNodes of the ControlMergeSubjectGraph to the
// inputSubjectGraphs in the order that they are defined.
void ControlMergeSubjectGraph::connectInputNodes() {
  for (unsigned int i = 0; i < inputNodes.size(); i++) {
    connectInputNodesHelper(inputNodes[i], inputSubjectGraphs[i]);
  }
}

// outputNodes are at channelIndex 0, and indexNodes are at channelIndex 1.
ChannelSignals &
ControlMergeSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return (channelIndex == 0) ? outputNodes : indexNodes;
}

// ConditionalBranchSubjectGraph implementation
ConditionalBranchSubjectGraph::ConditionalBranchSubjectGraph(Operation *op)
    : BaseSubjectGraph(op) {
  auto cbrOp = llvm::dyn_cast<handshake::ConditionalBranchOp>(op);
  // Get the data width of the Conditional Branch op
  dataWidth =
      handshake::getHandshakeTypeBitWidth(cbrOp.getDataOperand().getType());

  if (dataWidth == 0) {
    loadBlifFile({}, "_dataless");
  } else {
    loadBlifFile({dataWidth});
  }

  std::vector<NodeProcessingRule> rules = {
      {"condition", conditionNodes, false, nullptr},
      {"data", inputNodes, false, nullptr},
      {"true", trueOut, true, nullptr},
      {"false", falseOut, true, nullptr}};

  processNodesWithRules(rules);
}

// conditionNodes are connected to the first inputSubjectGraph, and
// inputNodes are connected to the second inputSubjectGraph.
void ConditionalBranchSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(conditionNodes, inputSubjectGraphs[0]);
  connectInputNodesHelper(inputNodes, inputSubjectGraphs[1]);
}

// trueOut are at channelIndex 0, and falseOut are at channelIndex 1.
ChannelSignals &
ConditionalBranchSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return (channelIndex == 0) ? trueOut : falseOut;
}

// SourceSubjectGraph implementation
SourceSubjectGraph::SourceSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  // Source module has no attributes
  loadBlifFile({});

  std::vector<NodeProcessingRule> rules = {
      {"outs", outputNodes, true, nullptr}};

  processNodesWithRules(rules);
}

// Source module has no input nodes.
void SourceSubjectGraph::connectInputNodes() {}

// Source module only has outputNodes as output.
ChannelSignals &SourceSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// LoadSubjectGraph implementation
LoadSubjectGraph::LoadSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  auto loadOp = llvm::dyn_cast<handshake::LoadOp>(op);
  // Get the data width and address type of the Load operation
  dataWidth =
      handshake::getHandshakeTypeBitWidth(loadOp.getDataInput().getType());
  addrType =
      handshake::getHandshakeTypeBitWidth(loadOp.getAddressInput().getType());

  loadBlifFile({addrType, dataWidth});

  std::vector<NodeProcessingRule> rules = {
      {"addrIn", addrInSignals, false, nullptr},
      {"addrOut", addrOutSignals, true, nullptr},
      {"dataOut", dataOutSignals, true, nullptr}};

  processNodesWithRules(rules);
}

// Load Module has only addrInSignals as input.
void LoadSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(addrInSignals, inputSubjectGraphs[0]);
}

// addrOutSignals are connected to the output module with channelIndex 0,
// and dataOutSignals are connected to the output module with channelIndex 1.
ChannelSignals &LoadSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return (channelIndex == 0) ? addrOutSignals : dataOutSignals;
}

// StoreSubjectGraph implementation
StoreSubjectGraph::StoreSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  auto storeOp = llvm::dyn_cast<handshake::StoreOp>(op);
  // Get the data width and address type of the Store operation
  dataWidth =
      handshake::getHandshakeTypeBitWidth(storeOp.getDataInput().getType());
  addrType =
      handshake::getHandshakeTypeBitWidth(storeOp.getAddressInput().getType());

  loadBlifFile({addrType, dataWidth});

  std::vector<NodeProcessingRule> rules = {
      {"dataIn", dataInSignals, false, nullptr},
      {"addrIn", addrInSignals, false, nullptr},
      {"addrOut", addrOutSignals, true, nullptr}};

  processNodesWithRules(rules);
}

// addrInSignals and dataInSignals are connected to the first and
// second inputSubjectGraphs respectively.
void StoreSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(addrInSignals, inputSubjectGraphs[0]);
  connectInputNodesHelper(dataInSignals, inputSubjectGraphs[1]);
}

// Store module has only addrOutSignals as output.
ChannelSignals &StoreSubjectGraph::returnOutputNodes(unsigned int) {
  return addrOutSignals;
}

// ConstantSubjectGraph implementation
ConstantSubjectGraph::ConstantSubjectGraph(Operation *op)
    : BaseSubjectGraph(op) {
  auto cstOp = llvm::dyn_cast<handshake::ConstantOp>(op);
  handshake::ChannelType cstType = cstOp.getResult().getType();
  // Get the data width of the constant operation
  dataWidth = cstType.getDataBitWidth();

  loadBlifFile({dataWidth});

  std::vector<NodeProcessingRule> rules = {
      {"ctrl", controlSignals, false, nullptr},
      {"outs", outputNodes, true, nullptr}};

  processNodesWithRules(rules);
}

// Constant module has only controlSignals as input.
void ConstantSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(controlSignals, inputSubjectGraphs[0]);
}

// Constant module has only outputNodes as output.
ChannelSignals &ConstantSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// ExtTruncSubjectGraph implementation
ExtTruncSubjectGraph::ExtTruncSubjectGraph(Operation *op)
    : BaseSubjectGraph(op) {
  // Ext and Trunch operations have the same attributes and data types.
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::ExtFOp,
            handshake::TruncIOp, handshake::TruncFOp>([&](auto extTruncOp) {
        inputWidth = handshake::getHandshakeTypeBitWidth(
            extTruncOp.getOperand().getType());
        outputWidth = handshake::getHandshakeTypeBitWidth(
            extTruncOp.getResult().getType());
      });

  loadBlifFile({inputWidth, outputWidth});

  std::vector<NodeProcessingRule> rules = {
      {"ins", inputNodes, false, nullptr},
      {"outs", outputNodes, true, nullptr}};

  processNodesWithRules(rules);
}

// These modules only have a single input, inputNodes.
void ExtTruncSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(inputNodes, inputSubjectGraphs[0]);
}

// These modules only have a single output, outputNodes.
ChannelSignals &ExtTruncSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// SelectSubjectGraph implementation
SelectSubjectGraph::SelectSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  auto selectOp = llvm::dyn_cast<handshake::SelectOp>(op);
  // Get the data width of the Select operation
  dataWidth =
      handshake::getHandshakeTypeBitWidth(selectOp->getOperand(1).getType());

  loadBlifFile({dataWidth});

  std::vector<NodeProcessingRule> rules = {
      {"condition", condition, false, nullptr},
      {"trueValue", trueValue, false, nullptr},
      {"falseValue", falseValue, false, nullptr},
      {"result", outputNodes, true, nullptr},
  };

  processNodesWithRules(rules);
}

// condition are connected to the first inputSubjectGraph, trueValue are
// connected to the second inputSubjectGraph, and falseValue are connected to
// the third inputSubjectGraph.
void SelectSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(condition, inputSubjectGraphs[0]);
  connectInputNodesHelper(trueValue, inputSubjectGraphs[1]);
  connectInputNodesHelper(falseValue, inputSubjectGraphs[2]);
}

// Select module has only outputNodes as output.
ChannelSignals &SelectSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

void MergeSubjectGraph::processOutOfRuleNodes() {
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
    }
  }
}

MergeSubjectGraph::MergeSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  auto mergeOp = llvm::dyn_cast<handshake::MergeOp>(op);
  // Get the size and data width of the merge operation
  size = mergeOp.getDataOperands().size();
  dataWidth = handshake::getHandshakeTypeBitWidth(
      mergeOp.getDataOperands()[0].getType());

  inputNodes.resize(size);

  loadBlifFile({size, dataWidth});

  // "ins" case does not obey the rules
  processOutOfRuleNodes();

  std::vector<NodeProcessingRule> rules = {
      {"outs", outputNodes, true, nullptr}};

  processNodesWithRules(rules);
}

// Connects the inputNodes of the MergeSubjectGraph to the
// inputSubjectGraphs in the order that they are defined.
void MergeSubjectGraph::connectInputNodes() {
  for (unsigned int i = 0; i < inputNodes.size(); i++) {
    connectInputNodesHelper(inputNodes[i], inputSubjectGraphs[i]);
  }
}

// Merge module has only outputNodes as output.
ChannelSignals &MergeSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// BranchSinkSubjectGraph implementation
BranchSinkSubjectGraph::BranchSinkSubjectGraph(Operation *op)
    : BaseSubjectGraph(op) {
  // Get the data width of the Branch or Sink operation
  dataWidth = handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());

  if (dataWidth == 0) {
    loadBlifFile({}, "_dataless");
  } else {
    loadBlifFile({dataWidth});
  }

  std::vector<NodeProcessingRule> rules = {
      {"ins", inputNodes, false, nullptr},
      {"outs", outputNodes, true, nullptr}};

  processNodesWithRules(rules);
}

// BranchSinkSubjectGraph has only inputNodes as input.
void BranchSinkSubjectGraph::connectInputNodes() {
  // Block arguments do not have any input operations.
  if (!inputSubjectGraphs.empty()) {
    connectInputNodesHelper(inputNodes, inputSubjectGraphs[0]);
  }
}

// BranchSinkSubjectGraph has only outputNodes as output.
ChannelSignals &BranchSinkSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// Buffer initialization function. It loads the BLIF file based on the buffer
// type name and parses it. Then processes the buffer nodes.
void BufferSubjectGraph::initBuffer() {

  // We cannot use the loadBlifFile method here because the getName() method on
  // Operations only return the name of the operation. However, we need the
  // buffer type name to load the correct BLIF file.
  std::string fullPath = baseBlifPath;

  if (dataWidth == 0) {
    bufferType += "_dataless";
    fullPath += "/" + bufferType + "/" + bufferType + ".blif";
  } else {
    fullPath += "/" + bufferType + "/" + std::to_string(dataWidth) + "/" +
                bufferType + ".blif";
  }

  // Parse the BLIF file
  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  std::vector<NodeProcessingRule> rules = {
      {"ins", inputNodes, false, nullptr},
      {"outs", outputNodes, true, nullptr}};

  processNodesWithRules(rules);
}

// BufferSubjectGraph constructor from an MLIR Operation
BufferSubjectGraph::BufferSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  auto bufferOp = llvm::dyn_cast<handshake::BufferOp>(op);

  // Get the Buffer type and data width from the operation attributes
  auto params =
      bufferOp->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
  auto bufferTypeNamed =
      params.getNamed(handshake::BufferOp::BUFFER_TYPE_ATTR_NAME);
  auto bufferTypeAttr = dyn_cast<StringAttr>(bufferTypeNamed->getValue());
  bufferType = bufferTypeAttr.getValue().str();

  dataWidth = handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());

  initBuffer();
}

// BufferSubjectGraph constructor without an MLIR operation. This is used in the
// case where we insert a new buffer to the Subject Graph. Data width and
// buffer type parameters are given as input.
BufferSubjectGraph::BufferSubjectGraph(unsigned int inputDataWidth,
                                       std::string bufferTypeName)
    : BaseSubjectGraph(), dataWidth(inputDataWidth),
      bufferType(std::move(bufferTypeName)) {
  // Static buffer count is used to create unique names for buffers. It keeps
  // track of how many new buffers have been created so far.
  static unsigned int bufferCount;
  uniqueName = bufferTypeName + "_" + std::to_string(bufferCount++);

  initBuffer();
}

// Buffers only have inputNodes as input.
void BufferSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(inputNodes, inputSubjectGraphs[0]);
}

// Buffers only have outputNodes as output.
ChannelSignals &BufferSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// This function iterates over the MLIR operations inside a FuncOp, and it
// populates the vector subjectGraphVector with the subject graph of each op.
// Then, it marks the PIs and POs of the subject graph with the corresponding
// dataflow unit port that they represent.
void dynamatic::experimental::subjectGraphGenerator(handshake::FuncOp funcOp,
                                                    StringRef blifFiles) {
  baseBlifPath = blifFiles;
  std::vector<BaseSubjectGraph *> subjectGraphs;

  funcOp.walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<handshake::AddIOp, handshake::AndIOp, handshake::CmpIOp,
              handshake::OrIOp, handshake::ShLIOp, handshake::ShRSIOp,
              handshake::ShRUIOp, handshake::SubIOp, handshake::XOrIOp,
              handshake::MulIOp, handshake::DivSIOp, handshake::DivUIOp>(
            [&](auto) { subjectGraphs.push_back(new ArithSubjectGraph(op)); })
        .Case<handshake::BranchOp, handshake::SinkOp>([&](auto) {
          subjectGraphs.push_back(new BranchSinkSubjectGraph(op));
        })
        .Case<handshake::BufferOp, handshake::SinkOp>(
            [&](auto) { subjectGraphs.push_back(new BufferSubjectGraph(op)); })
        .Case<handshake::ConditionalBranchOp>(
            [&](handshake::ConditionalBranchOp cbrOp) {
              subjectGraphs.push_back(new ConditionalBranchSubjectGraph(op));
            })
        .Case<handshake::ConstantOp>([&](handshake::ConstantOp cstOp) {
          subjectGraphs.push_back(new ConstantSubjectGraph(op));
        })
        .Case<handshake::ControlMergeOp>(
            [&](handshake::ControlMergeOp cmergeOp) {
              subjectGraphs.push_back(new ControlMergeSubjectGraph(op));
            })
        .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::ExtFOp,
              handshake::TruncIOp, handshake::TruncFOp>([&](auto) {
          subjectGraphs.push_back(new ExtTruncSubjectGraph(op));
        })
        .Case<handshake::ForkOp, handshake::LazyForkOp>(
            [&](auto) { subjectGraphs.push_back(new ForkSubjectGraph(op)); })
        .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
          subjectGraphs.push_back(new MuxSubjectGraph(op));
        })
        .Case<handshake::MergeOp>(
            [&](auto) { subjectGraphs.push_back(new MergeSubjectGraph(op)); })
        .Case<handshake::LoadOp>([&](handshake::LoadOp loadOp) {
          subjectGraphs.push_back(new LoadSubjectGraph(op));
        })
        .Case<handshake::SelectOp>([&](handshake::SelectOp selectOp) {
          subjectGraphs.push_back(new SelectSubjectGraph(op));
        })
        .Case<handshake::SourceOp>(
            [&](auto) { subjectGraphs.push_back(new SourceSubjectGraph(op)); })
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
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
