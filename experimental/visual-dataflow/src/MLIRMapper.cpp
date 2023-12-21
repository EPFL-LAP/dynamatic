//===- MLIRMapper.cpp - Map MLIR module to Graph ----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the MLIR mapper. The mapper
// produces the graph representation of the input handshake-level IR.
//
//===----------------------------------------------------------------------===//

#include "MLIRMapper.h"
#include "GraphEdge.h"
#include "GraphNode.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Conversion/PassDetails.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::experimental::visual_dataflow;

/// Creates the "in" or "out" attribute of a node from a list of values and a
/// port name (used as prefix to derive numbered port names for all values).
static std::vector<std::string> getIOFromValues(ValueRange values,
                                                std::string &&portType) {
  std::vector<std::string> ports;
  for (auto [idx, val] : llvm::enumerate(values))
    ports.push_back(portType + std::to_string(idx + 1));
  return ports;
}

static size_t findIndexInRange(ValueRange range, Value val) {
  for (auto [idx, res] : llvm::enumerate(range))
    if (res == val)
      return idx;
  assert(false && "value should exist in range");
  return 0;
}

/// Finds the position (block index and operand index) of a value in the
/// inputs of a memory interface.
static std::pair<size_t, size_t> findValueInGroups(FuncMemoryPorts &ports,
                                                   Value val) {
  unsigned numBlocks = ports.getNumGroups();
  unsigned accInputIdx = 0;
  for (size_t blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
    ValueRange blockInputs = ports.getGroupInputs(blockIdx);
    accInputIdx += blockInputs.size();
    for (auto [inputIdx, input] : llvm::enumerate(blockInputs)) {
      if (input == val)
        return std::make_pair(blockIdx, inputIdx);
    }
  }

  // Value must belong to a port with another memory interface, find the one
  ValueRange lastInputs = ports.memOp.getMemOperands().drop_front(accInputIdx);
  for (auto [inputIdx, input] : llvm::enumerate(lastInputs)) {
    if (input == val)
      return std::make_pair(ports.getNumGroups(), inputIdx + accInputIdx);
  }

  llvm_unreachable("value should be an operand to the memory interface");
}

/// Corrects for different output port ordering conventions with legacy
/// Dynamatic.
static size_t fixOutputPortNumber(Operation *op, size_t idx) {
  return llvm::TypeSwitch<Operation *, size_t>(op)
      .Case<handshake::ConditionalBranchOp>([&](auto) {
        // Legacy Dynamatic has the data operand before the condition operand
        return idx;
      })
      .Case<handshake::EndOp>([&](handshake::EndOp endOp) {
        // Legacy Dynamatic has the memory controls before the return values
        auto numReturnValues = endOp.getReturnValues().size();
        auto numMemoryControls = endOp.getMemoryControls().size();
        return (idx < numReturnValues) ? idx + numMemoryControls
                                       : idx - numReturnValues;
      })
      .Case<handshake::LoadOpInterface, handshake::StoreOpInterface>([&](auto) {
        // Legacy Dynamatic has the data operand/result before the address
        // operand/result
        return 1 - idx;
      })
      .Case<handshake::LSQOp>([&](handshake::LSQOp lsqOp) {
        // Legacy Dynamatic places the end control signal before the signals
        // going to the MC, if one is connected
        LSQPorts lsqPorts = lsqOp.getPorts();
        if (!lsqPorts.hasAnyPort<MCLoadStorePort>())
          return idx;

        // End control signal succeeded by laad address, store address, store
        // data
        if (idx == lsqOp.getNumResults() - 1)
          return idx - 3;

        // Signals to MC preceeded by end control signal
        unsigned numLoads = lsqPorts.getNumPorts<LSQLoadPort>();
        if (idx >= numLoads)
          return idx + 1;
        return idx;
      })
      .Default([&](auto) { return idx; });
}

/// Corrects for different input port ordering conventions with legacy
/// Dynamatic.
static size_t fixInputPortNumber(Operation *op, size_t idx) {
  return llvm::TypeSwitch<Operation *, size_t>(op)
      .Case<handshake::ConditionalBranchOp>([&](auto) {
        // Legacy Dynamatic has the data operand before the condition operand
        return 1 - idx;
      })
      .Case<handshake::EndOp>([&](handshake::EndOp endOp) {
        // Legacy Dynamatic has the memory controls before the return values
        auto numReturnValues = endOp.getReturnValues().size();
        auto numMemoryControls = endOp.getMemoryControls().size();
        return (idx < numReturnValues) ? idx + numMemoryControls
                                       : idx - numReturnValues;
      })
      .Case<handshake::LoadOpInterface, handshake::StoreOpInterface>([&](auto) {
        // Legacy Dynamatic has the data operand/result before the address
        // operand/result
        return 1 - idx;
      })
      .Case<handshake::MemoryOpInterface>(
          [&](handshake::MemoryOpInterface memOp) {
            Value val = op->getOperand(idx);

            // Legacy Dynamatic puts all control operands before all data
            // operands, whereas for us each control operand appears just
            // before the data inputs of the group it corresponds to
            FuncMemoryPorts ports = getMemoryPorts(memOp);

            // Determine total number of control operands
            unsigned ctrlCount = ports.getNumPorts<ControlPort>();

            // Figure out where the value lies
            auto [groupIDx, opIdx] = findValueInGroups(ports, val);

            if (groupIDx == ports.getNumGroups()) {
              // If the group index is equal to the number of connected groups,
              // then the operand index points directly to the matching port in
              // legacy Dynamatic's conventions
              return opIdx;
            }

            // Figure out at which index the value would be in legacy
            // Dynamatic's interface
            bool valGroupHasControl = ports.groups[groupIDx].hasControl();
            if (opIdx == 0 && valGroupHasControl) {
              // Value is a control input
              size_t fixedIdx = 0;
              for (size_t i = 0; i < groupIDx; i++)
                if (ports.groups[i].hasControl())
                  fixedIdx++;
              return fixedIdx;
            }

            // Value is a data input
            size_t fixedIdx = ctrlCount;
            for (size_t i = 0; i < groupIDx; i++) {
              // Add number of data inputs corresponding to the group, minus the
              // control input which was already accounted for (if present)
              fixedIdx += ports.groups[i].getNumInputs();
              if (ports.groups[i].hasControl())
                --fixedIdx;
            }
            // Add index offset in the group the value belongs to
            if (valGroupHasControl)
              fixedIdx += opIdx - 1;
            else
              fixedIdx += opIdx;
            return fixedIdx;
          })
      .Default([&](auto) { return idx; });
}

MLIRMapper::MLIRMapper(Graph *graph) : graph(graph) {}

LogicalResult MLIRMapper::mapMLIR(mlir::ModuleOp mod) {
  auto funcs = mod.getOps<handshake::FuncOp>();
  if (funcs.empty())
    return success();
  if (++funcs.begin() != funcs.end()) {
    mod->emitOpError()
        << "we currently only support one handshake function per module";
    return failure();
  }
  handshake::FuncOp funcOp = *funcs.begin();

  std::map<std::string, unsigned> opTypeCntrs;
  DenseMap<Operation *, unsigned> opIDs;

  // Sequentially scan across the operations in the function and assign
  // instance IDs to each operation
  for (auto &op : funcOp.getOps())
    opIDs[&op] = opTypeCntrs[op.getName().getStringRef().str()]++;

  for (auto &op : funcOp.getOps()) {
    // Give a unique name to each operation. Extract operation name without
    // dialect prefix if possible and then append an ID unique to each
    // operation type
    std::string opFullName = op.getName().getStringRef().str();
    auto startIdx = opFullName.find('.');
    if (startIdx == std::string::npos)
      startIdx = 0;
    auto opID = std::to_string(opIDs[&op]);
    opNameMap[&op] = opFullName.substr(startIdx + 1) + opID;

    // Map the operation
    if (failed(mapNode(&op)))
      return failure();
  }

  // Iterate over all uses of all results of all operations
  for (auto &op : funcOp.getOps()) {
    for (auto res : op.getResults())
      for (auto &use : res.getUses()) {
        Operation *useOp = use.getOwner();
        EdgeId edgeId = 0;

        // Map the result
        if (failed(mapEdge(&op, useOp, res, &edgeId)))
          return failure();
      }
  }

  return success();
}

LogicalResult MLIRMapper::mapNode(Operation *op) {
  // Node unique name
  auto opName = opNameMap[op];

  // Node inPorts
  std::vector<std::string> inPorts = getIOFromValues(op->getOperands(), "in");

  // Node outPorts
  std::vector<std::string> outPorts = getIOFromValues(op->getResults(), "out");

  // TODO: Node position
  std::pair<float, float> pos = {0, 0};

  // Create the node
  GraphNode node(opName, pos);

  // Add the ports to the node
  for (auto &inPort : inPorts)
    node.addPort(inPort.back(), true);
  for (auto &outPort : outPorts)
    node.addPort(outPort.back(), false);

  // Add the node to the graph
  graph->addNode(node);

  return success();
}

LogicalResult MLIRMapper::mapEdge(Operation *src, Operation *dst, Value val,
                                  EdgeId *edgeId) {
  // Find the source and destination nodes
  std::string srcNodeName = getNodeName(src);
  std::string dstNodeName = getNodeName(dst);
  GraphNode *srcNodeIt = nullptr;
  GraphNode *dstNodeIt = nullptr;
  if (failed(graph->getNode(srcNodeName, *srcNodeIt)) ||
      failed(graph->getNode(dstNodeName, *dstNodeIt)))
    return failure();

  // Find the source value
  Value srcVal = val.getDefiningOp()->getResult(0);

  // Locate value in source results and destination operands
  auto resIdx = findIndexInRange(src->getResults(), srcVal);
  auto argIdx = findIndexInRange(dst->getOperands(), val);

  // Find the source and destination ports
  int from = fixOutputPortNumber(src, resIdx) + 1;
  int to = fixInputPortNumber(dst, argIdx) + 1;

  // Create the edge
  GraphEdge edge(*edgeId++, *srcNodeIt, *dstNodeIt, from, to, {});

  // Add the edge to the graph
  graph->addEdge(edge);

  return success();
}

std::string MLIRMapper::getNodeName(Operation *op) {
  auto opNameIt = opNameMap.find(op);
  if (opNameIt == opNameMap.end())
    return "";
  return opNameIt->second;
}
