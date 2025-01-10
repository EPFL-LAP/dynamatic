//===- SubjectGraph.cpp - Exp. support for MAPBUF buffer placement -------*- C++
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

#include "experimental/Support/SubjectGraph.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include <algorithm>
#include <cassert>
#include <iterator>
using namespace dynamatic::experimental;

BaseSubjectGraph::BaseSubjectGraph() = default;

BaseSubjectGraph::BaseSubjectGraph(Operation *op) : op(op) {
  moduleMap[op] = this;
  subjectGraphMap[this] = op;
  moduleType = op->getName().getStringRef();
  uniqueName = getUniqueName(op);

  for (Value operand : op->getOperands()) {
    if (Operation *definingOp = operand.getDefiningOp()) {
      unsigned portNumber = operand.cast<OpResult>().getResultNumber();
      inputModules.push_back(definingOp);
      inputModuleToResNum[definingOp] = portNumber;
    }
  }

  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      unsigned portNumber = result.cast<OpResult>().getResultNumber();
      outputModules.push_back(user);
      outputModuleToResNum[user] = portNumber;
    }
  }

  size_t dotPosition = moduleType.find('.');
  if (dotPosition != std::string::npos) {
    moduleType = moduleType.substr(dotPosition + 1);
  } else {
    assert(false && "operation unsupported");
  }
}

BaseSubjectGraph::BaseSubjectGraph(Operation *before, Operation *after) {
  auto *beforeSubjectGraph = moduleMap[before];
  auto *afterSubjectGraph = moduleMap[after];
  inputModules.push_back(before);
  outputModules.push_back(after);
  inputSubjectGraphs.push_back(beforeSubjectGraph);
  outputSubjectGraphs.push_back(afterSubjectGraph);
  inputModuleToResNum[before] = 0;
}

void BaseSubjectGraph::connectInputNodesHelper(
    ChannelSignals &currentSignals,
    BaseSubjectGraph *moduleBeforeSubjectGraph) {
  ChannelSignals &moduleBeforeOutputNodes =
      moduleBeforeSubjectGraph->returnOutputNodes(
          inputSubjectGraphToResNum[moduleBeforeSubjectGraph]);

  connectSignals(moduleBeforeOutputNodes.readySignal,
                 currentSignals.readySignal);
  connectSignals(currentSignals.validSignal,
                 moduleBeforeOutputNodes.validSignal);

  if (isBlackbox) {
    // If the module is a blackbox, we don't need to connect the data signals,
    // as it will result in cut generation for blackbox modules
    for (auto *node : currentSignals.dataSignals) {
      node->setInput(false);
      node->setOutput(false);
    }
  } else {
    for (unsigned int j = 0; j < currentSignals.dataSignals.size(); j++) {
      connectSignals(currentSignals.dataSignals[j],
                     moduleBeforeOutputNodes.dataSignals[j]);
    }
  }
}

void BaseSubjectGraph::assignSignals(ChannelSignals &signals, Node *node,
                                     const std::string &nodeName) {
  if (nodeName.find("valid") != std::string::npos) {
    signals.validSignal = node;
  } else if (nodeName.find("ready") != std::string::npos) {
    signals.readySignal = node;
  } else {
    signals.dataSignals.push_back(node);
  }
};

void BaseSubjectGraph::replaceOpsBySubjectGraph() {
  for (auto *inputModule : inputModules) {
    auto *inputSubjectGraph = moduleMap[inputModule];
    inputSubjectGraphs.push_back(inputSubjectGraph);
    inputSubjectGraphToResNum[inputSubjectGraph] =
        inputModuleToResNum[inputModule];
  }

  for (auto *outputModule : outputModules) {
    auto *outputSubjectGraph = moduleMap[outputModule];
    outputSubjectGraphs.push_back(outputSubjectGraph);
    outputSubjectGraphToResNum[outputSubjectGraph] =
        outputModuleToResNum[outputModule];
  }
}

unsigned int BaseSubjectGraph::getChannelNumber(BaseSubjectGraph *first,
                                                BaseSubjectGraph *second) {
  return first->outputSubjectGraphToResNum[second];
}

void BaseSubjectGraph::changeOutput(BaseSubjectGraph *graph,
                                    BaseSubjectGraph *newOutput,
                                    BaseSubjectGraph *oldOutput) {
  auto it = std::find(graph->outputSubjectGraphs.begin(),
                      graph->outputSubjectGraphs.end(), oldOutput);
  unsigned int channelNumber = 0;
  if (it != graph->outputSubjectGraphs.end()) {
    channelNumber = graph->outputSubjectGraphToResNum[oldOutput];
    auto index = std::distance(graph->outputSubjectGraphs.begin(), it);
    graph->outputSubjectGraphs.erase(it);
    graph->outputSubjectGraphs.insert(
        graph->outputSubjectGraphs.begin() + index, newOutput);
    graph->outputSubjectGraphToResNum.erase(oldOutput);
  } else {
    llvm::errs() << "Output not found\n";
  }
  graph->outputSubjectGraphToResNum[newOutput] = channelNumber;
}

void BaseSubjectGraph::changeInput(BaseSubjectGraph *graph,
                                   BaseSubjectGraph *newInput,
                                   BaseSubjectGraph *oldInput) {
  unsigned int channelNumber = 0;
  auto it = std::find(graph->inputSubjectGraphs.begin(),
                      graph->inputSubjectGraphs.end(), oldInput);
  if (it != graph->inputSubjectGraphs.end()) {
    auto index = std::distance(graph->inputSubjectGraphs.begin(), it);
    graph->inputSubjectGraphs.erase(it);
    graph->inputSubjectGraphs.insert(graph->inputSubjectGraphs.begin() + index,
                                     newInput);
    channelNumber = graph->inputSubjectGraphToResNum[oldInput];
    graph->inputSubjectGraphToResNum.erase(oldInput);
  } else {
    llvm::errs() << "Input not found\n";
  }
  graph->inputSubjectGraphToResNum[newInput] = channelNumber;
}

void BaseSubjectGraph::appendVarsToPath(
    std::initializer_list<unsigned int> inputs) {
  fullPath = baseBlifPath + "/" + moduleType + "/";
  for (int input : inputs) {
    fullPath += std::to_string(input) + "/";
  }
  fullPath += moduleType + ".blif";
}

void BaseSubjectGraph::connectSignals(Node *currentSignal, Node *beforeSignal) {
  beforeSignal->addFanout(currentSignal->getFanouts());
  currentSignal->setInput(false);
  beforeSignal->setOutput(false);
  beforeSignal->setInput(false);
  currentSignal->setOutput(false);
  beforeSignal->setChannelEdge(true);
  currentSignal->setChannelEdge(true);

  if (beforeSignal->isBlackboxOutputNode()) {
    beforeSignal->setInput(true);
  }

  for (auto &fanout : currentSignal->getFanouts()) {
    fanout->getFanins().erase(currentSignal);
    fanout->getFanins().insert(beforeSignal);
  }

  if (beforeSignal->getName().find("ready") != std::string::npos) {
    beforeSignal->setName(currentSignal->getName());
  }

  currentSignal = beforeSignal;
}

BlifData *BaseSubjectGraph::getBlifData() const { return blifData; }

BaseSubjectGraph *BaseSubjectGraph::getSubjectGraph(Operation *op) {
  return moduleMap[op];
}

std::string &BaseSubjectGraph::getUniqueNameGraph() { return uniqueName; }

std::string &BaseSubjectGraph::getModuleType() { return moduleType; }

std::vector<BaseSubjectGraph *> BaseSubjectGraph::getInputSubjectGraphs() {
  return inputSubjectGraphs;
}

std::vector<BaseSubjectGraph *> BaseSubjectGraph::getOutputSubjectGraphs() {
  return outputSubjectGraphs;
}

// ArithSubjectGraph implementation
ArithSubjectGraph::ArithSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::AddIOp, handshake::AndIOp, handshake::OrIOp,
            handshake::ShLIOp, handshake::ShRSIOp, handshake::ShRUIOp,
            handshake::SubIOp, handshake::XOrIOp, handshake::MulIOp,
            handshake::DivSIOp, handshake::DivUIOp>([&](auto) {
        dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        appendVarsToPath({dataWidth});
        if ((dataWidth > 4) && (llvm::isa<handshake::AddIOp>(op) ||
                                llvm::isa<handshake::SubIOp>(op) ||
                                llvm::isa<handshake::MulIOp>(op) ||
                                llvm::isa<handshake::DivSIOp>(op) ||
                                llvm::isa<handshake::DivUIOp>(op))) {
          isBlackbox = true;
        }
      })
      .Case<handshake::AddFOp, handshake::DivFOp, handshake::MaximumFOp,
            handshake::MinimumFOp, handshake::MulFOp, handshake::NegFOp,
            handshake::NotOp, handshake::SubFOp, handshake::SIToFPOp,
            handshake::FPToSIOp, handshake::AbsFOp, handshake::CmpFOp>(
          [&](auto) {
            assert(false && "Float not supported");
            return;
          })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("result") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
      if (isBlackbox && (nodeName.find("valid") == std::string::npos &&
                         nodeName.find("ready") == std::string::npos)) {
        node->setBlackboxOutput(true);
      }
    } else if (nodeName.find("lhs") != std::string::npos) {
      assignSignals(inputNodes[0], node, nodeName);
    } else if (nodeName.find("rhs") != std::string::npos) {
      assignSignals(inputNodes[1], node, nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
    }
  }
}

void ArithSubjectGraph::connectInputNodes() {
  for (unsigned int i = 0; i < inputNodes.size(); i++) {
    connectInputNodesHelper(inputNodes[i], inputSubjectGraphs[i]);
  }
}

ChannelSignals &
ArithSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return outputNodes;
}

// CmpISubjectGraph implementation
CmpISubjectGraph::CmpISubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::CmpIOp>([&](handshake::CmpIOp cmpIOp) {
        dataWidth =
            handshake::getHandshakeTypeBitWidth(cmpIOp.getOperand(0).getType());
        appendVarsToPath({dataWidth});
        if (dataWidth > 4) {
          isBlackbox = true;
        }
      })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("result") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
      if (isBlackbox && (nodeName.find("valid") == std::string::npos &&
                         nodeName.find("ready") == std::string::npos)) {
        node->setBlackboxOutput(true);
      }
    } else if (nodeName.find("lhs") != std::string::npos) {
      assignSignals(inputNodes[0], node, nodeName);
    } else if (nodeName.find("rhs") != std::string::npos) {
      assignSignals(inputNodes[1], node, nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
    }
  }
}

void CmpISubjectGraph::connectInputNodes() {
  for (unsigned int i = 0; i < inputNodes.size(); i++) {
    connectInputNodesHelper(inputNodes[i], inputSubjectGraphs[i]);
  }
}

ChannelSignals &CmpISubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return outputNodes;
}

// ForkSubjectGraph implementation
ForkSubjectGraph::ForkSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::ForkOp>([&](auto) {
        size = op->getNumResults();
        dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());

        if (dataWidth == 0) {
          moduleType += "_dataless";
          appendVarsToPath({size});
        } else {
          moduleType += "_type";
          appendVarsToPath({size, dataWidth});
        }
      })
      .Case<handshake::LazyForkOp>([&](auto) {
        size = op->getNumResults();
        dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());

        if (dataWidth == 0) {
          moduleType += "_dataless";
          appendVarsToPath({size});
        } else {
          appendVarsToPath({size, dataWidth});
        }
      })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

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

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("outs") != std::string::npos) {
      auto [newName, num] = generateNewName(nodeName);
      assignSignals(outputNodes[num], node, newName);
      node->setName(uniqueName + "_" + newName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
    } else if (nodeName.find("ins") != std::string::npos &&
               (node->isInput() || node->isOutput())) {
      assignSignals(inputNodes, node, nodeName);
    }
  }
}

void ForkSubjectGraph::connectInputNodes() {
  if (!inputModules.empty()) {
    connectInputNodesHelper(inputNodes, inputSubjectGraphs[0]);
  }
}

ChannelSignals &ForkSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return outputNodes[channelIndex];
}

// MuxSubjectGraph implementation
MuxSubjectGraph::MuxSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        size = muxOp.getDataOperands().size();
        dataWidth =
            handshake::getHandshakeTypeBitWidth(muxOp.getResult().getType());
        selectType = handshake::getHandshakeTypeBitWidth(
            muxOp.getSelectOperand().getType());

        appendVarsToPath({size, dataWidth});
      })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput() || node->isOutput())) {
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
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
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
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp cmergeOp) {
        size = cmergeOp.getDataOperands().size();
        dataWidth =
            handshake::getHandshakeTypeBitWidth(cmergeOp.getResult().getType());
        indexType =
            handshake::getHandshakeTypeBitWidth(cmergeOp.getIndex().getType());
        if (dataWidth == 0) {
          moduleType += "_dataless";
          appendVarsToPath({size, indexType});
        } else {
          assert(false && "ControlMerge with data width not supported");
        }
      })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput() || node->isOutput())) {
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
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
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
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp cbrOp) {
            dataWidth = handshake::getHandshakeTypeBitWidth(
                cbrOp.getDataOperand().getType());
            if (dataWidth == 0) {
              moduleType += "_dataless";
              appendVarsToPath({});
            } else {
              appendVarsToPath({dataWidth});
            }
          })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("true") != std::string::npos) {
      assignSignals(outputNodes[0], node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find("false") != std::string::npos) {
      assignSignals(outputNodes[1], node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find("condition") != std::string::npos) {
      assignSignals(conditionNodes, node, nodeName);
    } else if (nodeName.find("data") != std::string::npos) {
      assignSignals(inputNodes, node, nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
    }
  }
}

void ConditionalBranchSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(conditionNodes, inputSubjectGraphs[0]);
  connectInputNodesHelper(inputNodes, inputSubjectGraphs[1]);
}

ChannelSignals &
ConditionalBranchSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return outputNodes[channelIndex];
}

// SourceSubjectGraph implementation
SourceSubjectGraph::SourceSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::SourceOp>([&](auto) { appendVarsToPath({}); })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
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
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::LoadOp>([&](handshake::LoadOp loadOp) {
        dataWidth = handshake::getHandshakeTypeBitWidth(
            loadOp.getDataInput().getType());
        addrType = handshake::getHandshakeTypeBitWidth(
            loadOp.getAddressInput().getType());
        appendVarsToPath({addrType, dataWidth});
      })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("addrIn") != std::string::npos) {
      assignSignals(addrInSignals, node, nodeName);
    } else if (nodeName.find("addrOut") != std::string::npos) {
      assignSignals(addrOutSignals, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find("dataOut") != std::string::npos) {
      assignSignals(dataOutSignals, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
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
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::StoreOp>([&](handshake::StoreOp storeOp) {
        dataWidth = handshake::getHandshakeTypeBitWidth(
            storeOp.getDataInput().getType());
        addrType = handshake::getHandshakeTypeBitWidth(
            storeOp.getAddressInput().getType());

        appendVarsToPath({addrType, dataWidth});
      })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("dataIn") != std::string::npos) {
      assignSignals(dataInSignals, node, nodeName);
    } else if (nodeName.find("addrIn") != std::string::npos) {
      assignSignals(addrInSignals, node, nodeName);
    } else if (nodeName.find("addrOut") != std::string::npos) {
      assignSignals(addrOutSignals, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
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
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::ConstantOp>([&](handshake::ConstantOp cstOp) {
        handshake::ChannelType cstType = cstOp.getResult().getType();
        unsigned bitwidth = cstType.getDataBitWidth();
        dataWidth = bitwidth;
        appendVarsToPath({dataWidth});

        if (bitwidth > 64) {
          cstOp.emitError() << "Constant value has bitwidth " << bitwidth
                            << ", but we only support up to 64.";
          return;
        }
      })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("outs") != std::string::npos) {
      node->setName(uniqueName + "_" + nodeName);
      assignSignals(outputNodes, node, nodeName);
    } else if (nodeName.find("ctrl") != std::string::npos) {
      assignSignals(controlSignals, node, nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
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
        appendVarsToPath({inputWidth, outputWidth});
      })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput() || node->isOutput())) {
      assignSignals(inputNodes, node, nodeName);
    } else if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
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
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::SelectOp>([&](auto selectOp) {
        dataWidth = handshake::getHandshakeTypeBitWidth(
            selectOp->getOperand(1).getType());
        appendVarsToPath({dataWidth});
      })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("trueValue") != std::string::npos &&
        (node->isInput() || node->isOutput())) {
      assignSignals(inputNodes[1], node, nodeName);
    } else if (nodeName.find("falseValue") != std::string::npos &&
               (node->isInput() || node->isOutput())) {
      assignSignals(inputNodes[2], node, nodeName);
    } else if (nodeName.find("condition") != std::string::npos &&
               (node->isInput() || node->isOutput())) {
      assignSignals(inputNodes[0], node, nodeName);
    } else if (nodeName.find("result") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
    }
  }
}

void SelectSubjectGraph::connectInputNodes() {
  for (unsigned int i = 0; i < inputNodes.size(); i++) {
    connectInputNodesHelper(inputNodes[i], inputSubjectGraphs[i]);
  }
}

ChannelSignals &
SelectSubjectGraph::returnOutputNodes(unsigned int channelIndex) {
  return outputNodes;
}

MergeSubjectGraph::MergeSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::MergeOp>([&](auto mergeOp) {
        size = mergeOp.getDataOperands().size();
        dataWidth = handshake::getHandshakeTypeBitWidth(
            mergeOp.getDataOperands()[0].getType());
        appendVarsToPath({size, dataWidth});
      })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput() || node->isOutput())) {
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
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos ||
               nodeName.find("dataReg") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
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
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::BranchOp, handshake::SinkOp>([&](auto) {
        dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        if (dataWidth == 0) {
          moduleType += "_dataless";
          appendVarsToPath({});
        } else {
          appendVarsToPath({dataWidth});
        }
      })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput() || node->isOutput())) {
      assignSignals(inputNodes, node, nodeName);
    } else if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos ||
               nodeName.find("dataReg") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
    }
  }
}

void BranchSinkSubjectGraph::connectInputNodes() {
  if (!inputModules.empty()) {
    connectInputNodesHelper(inputNodes, inputSubjectGraphs[0]);
  }
}

ChannelSignals &BranchSinkSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// BufferSubjectGraph implementations
BufferSubjectGraph::BufferSubjectGraph(Operation *op) : BaseSubjectGraph(op) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::BufferOp>([&](handshake::BufferOp bufferOp) {
        auto params =
            bufferOp->getAttrOfType<DictionaryAttr>(RTL_PARAMETERS_ATTR_NAME);
        auto optTiming = params.getNamed(handshake::BufferOp::TIMING_ATTR_NAME);

        if (auto timing =
                dyn_cast<handshake::TimingAttr>(optTiming->getValue())) {
          handshake::TimingInfo info = timing.getInfo();
          if (info == handshake::TimingInfo::oehb())
            moduleType = "oehb";
          if (info == handshake::TimingInfo::tehb())
            moduleType = "tehb";
        }

        dataWidth =
            handshake::getHandshakeTypeBitWidth(op->getOperand(0).getType());
        if (dataWidth == 0) {
          moduleType += "_dataless";
          appendVarsToPath({});
        } else {
          appendVarsToPath({dataWidth});
        }
      })
      .Default([&](auto) {
        assert(false && "Operation does not match any supported type");
        return;
      });

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput() || node->isOutput())) {
      assignSignals(inputNodes, node, nodeName);
    } else if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos ||
               nodeName.find("dataReg") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
    }
  }
}

BufferSubjectGraph::BufferSubjectGraph(Operation *op1, Operation *op2,
                                       std::string bufferType)
    : BaseSubjectGraph() {
  subjectGraphMap[this] = nullptr;
  moduleType = bufferType;

  BaseSubjectGraph *graph1 = moduleMap[op1];
  BaseSubjectGraph *graph2 = moduleMap[op2];
  inputSubjectGraphs.push_back(graph1);
  outputSubjectGraphs.push_back(graph2);

  static unsigned int bufferCount;
  unsigned int channelNum = getChannelNumber(graph1, graph2);
  outputSubjectGraphToResNum[graph2] = channelNum;
  inputSubjectGraphToResNum[graph1] = channelNum;

  ChannelSignals &channel = graph1->returnOutputNodes(channelNum);
  dataWidth = channel.dataSignals.size();
  uniqueName = "oehb_" + std::to_string(bufferCount++);

  changeOutput(graph1, this, graph2);
  changeInput(graph2, this, graph1);

  if (dataWidth == 0) {
    moduleType += "_dataless";
    appendVarsToPath({});
  } else {
    appendVarsToPath({dataWidth});
  }

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput() || node->isOutput())) {
      assignSignals(inputNodes, node, nodeName);
    } else if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos ||
               nodeName.find("dataReg") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
    }
  }
}

BufferSubjectGraph::BufferSubjectGraph(BufferSubjectGraph *graph1,
                                       Operation *op2, std::string bufferType)
    : BaseSubjectGraph() {
  subjectGraphMap[this] = nullptr;
  moduleType = bufferType;

  BaseSubjectGraph *graph2 = moduleMap[op2];
  inputSubjectGraphs.push_back(graph1);
  outputSubjectGraphs.push_back(graph2);

  static unsigned int bufferCount;
  unsigned int channelNum = getChannelNumber(graph1, graph2);
  outputSubjectGraphToResNum[graph2] = channelNum;
  inputSubjectGraphToResNum[graph1] = channelNum;

  ChannelSignals &channel = graph1->returnOutputNodes(channelNum);
  dataWidth = channel.dataSignals.size();
  uniqueName = "tehb_" + std::to_string(bufferCount++);

  changeOutput(graph1, this, graph2);
  changeInput(graph2, this, graph1);

  if (dataWidth == 0) {
    moduleType += "_dataless";
    appendVarsToPath({});
  } else {
    appendVarsToPath({dataWidth});
  }

  experimental::BlifParser parser;
  blifData = parser.parseBlifFile(fullPath);

  for (auto &node : blifData->getAllNodes()) {
    auto nodeName = node->getName();
    if (nodeName.find("ins") != std::string::npos &&
        (node->isInput() || node->isOutput())) {
      assignSignals(inputNodes, node, nodeName);
    } else if (nodeName.find("outs") != std::string::npos) {
      assignSignals(outputNodes, node, nodeName);
      node->setName(uniqueName + "_" + nodeName);
    } else if (nodeName.find(".") != std::string::npos ||
               nodeName.find("dataReg") != std::string::npos) {
      node->setName(uniqueName + "." + nodeName);
    }
  }
}

void BufferSubjectGraph::connectInputNodes() {
  connectInputNodesHelper(inputNodes, inputSubjectGraphs[0]);
}

ChannelSignals &BufferSubjectGraph::returnOutputNodes(unsigned int) {
  return outputNodes;
}

// OperationDifferentiator implementation
OperationDifferentiator::OperationDifferentiator(Operation *ops) : op(ops) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<handshake::InstanceOp>([&](handshake::InstanceOp instOp) {
        // op->emitRemark("Instance Op");
      })
      .Case<handshake::ForkOp, handshake::LazyForkOp>(
          [&](auto) { moduleMap[op] = new ForkSubjectGraph(op); })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        moduleMap[op] = new MuxSubjectGraph(op);
      })
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp cmergeOp) {
        moduleMap[op] = new ControlMergeSubjectGraph(op);
      })
      .Case<handshake::MergeOp>(
          [&](auto) { moduleMap[op] = new MergeSubjectGraph(op); })
      .Case<handshake::JoinOp>([&](auto) { op->emitRemark("Join Op"); })
      .Case<handshake::BranchOp, handshake::SinkOp>(
          [&](auto) { moduleMap[op] = new BranchSinkSubjectGraph(op); })
      .Case<handshake::BufferOp, handshake::SinkOp>(
          [&](auto) { moduleMap[op] = new BufferSubjectGraph(op); })
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp cbrOp) {
            moduleMap[op] = new ConditionalBranchSubjectGraph(op);
          })
      .Case<handshake::SourceOp>(
          [&](auto) { moduleMap[op] = new SourceSubjectGraph(op); })
      .Case<handshake::LoadOp>([&](handshake::LoadOp loadOp) {
        moduleMap[op] = new LoadSubjectGraph(op);
      })
      .Case<handshake::StoreOp>([&](handshake::StoreOp storeOp) {
        moduleMap[op] = new StoreSubjectGraph(op);
      })
      .Case<handshake::SharingWrapperOp>(
          [&](handshake::SharingWrapperOp sharingWrapperOp) {
            assert(false && "operation unsupported");
          })
      .Case<handshake::ConstantOp>([&](handshake::ConstantOp cstOp) {
        moduleMap[op] = new ConstantSubjectGraph(op);
      })
      .Case<handshake::AddFOp, handshake::DivFOp, handshake::MaximumFOp,
            handshake::MinimumFOp, handshake::MulFOp, handshake::NegFOp,
            handshake::NotOp, handshake::SubFOp, handshake::SIToFPOp,
            handshake::FPToSIOp, handshake::AbsFOp, handshake::CmpFOp>(
          [&](auto) {
            op->emitError() << "Float not supported";
            return;
          })
      .Case<handshake::AddIOp, handshake::AndIOp, handshake::OrIOp,
            handshake::ShLIOp, handshake::ShRSIOp, handshake::ShRUIOp,
            handshake::SubIOp, handshake::XOrIOp, handshake::MulIOp,
            handshake::DivSIOp, handshake::DivUIOp>(
          [&](auto) { moduleMap[op] = new ArithSubjectGraph(op); })
      .Case<handshake::SelectOp>([&](handshake::SelectOp selectOp) {
        moduleMap[op] = new SelectSubjectGraph(op);
      })
      .Case<handshake::CmpIOp>([&](handshake::CmpIOp cmpIOp) {
        moduleMap[op] = new CmpISubjectGraph(op);
      })
      .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::ExtFOp,
            handshake::TruncIOp, handshake::TruncFOp>(
          [&](auto) { moduleMap[op] = new ExtTruncSubjectGraph(op); })
      .Default([&](auto) { return; });
}

// SubjectGraphGenerator implementation
SubjectGraphGenerator::SubjectGraphGenerator(handshake::FuncOp funcOp,
                                             StringRef blifFiles) {
  BaseSubjectGraph::setBaseBlifPath(blifFiles);
  funcOp.walk(
      [&](Operation *op) { (experimental::OperationDifferentiator(op)); });

  for (auto &module : experimental::OperationDifferentiator::moduleMap) {
    module.second->replaceOpsBySubjectGraph();
  }
}