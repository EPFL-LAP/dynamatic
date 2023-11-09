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
namespace {
std::vector<std::string> getIOFromValues(ValueRange values,
                                         std::string &&portType) {
  std::vector<std::string> ports;
  for (auto [idx, val] : llvm::enumerate(values))
    ports.push_back(portType + std::to_string(idx + 1));
  return ports;
}

size_t findIndexInRange(ValueRange range, Value val) {
  for (auto [idx, res] : llvm::enumerate(range))
    if (res == val)
      return idx;
  assert(false && "value should exist in range");
  return 0;
}

/// Finds the position (group index and operand index) of a value in the
/// inputs of a memory interface.
std::pair<size_t, size_t> findValueInGroups(FuncMemoryPorts &ports, Value val) {
  unsigned numBlocks = ports.getNumConnectedBlock();
  for (size_t blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
    for (auto [inputIdx, input] :
         llvm::enumerate(ports.getBlockInputs(blockIdx))) {
      if (input == val)
        return std::make_pair(blockIdx, inputIdx);
    }
  }
  llvm_unreachable("value should be an operand to the memory interface");
}

/// Transforms the port number associated to an edge endpoint to match the
/// operand ordering of legacy Dynamatic.
static size_t fixPortNumber(Operation *op, Value val, size_t idx,
                            bool isSrcOp) {
  return llvm::TypeSwitch<Operation *, size_t>(op)
      .Case<handshake::ConditionalBranchOp>([&](auto) {
        if (isSrcOp)
          return idx;
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
      .Case<handshake::DynamaticLoadOp, handshake::DynamaticStoreOp>([&](auto) {
        // Legacy Dynamatic has the data operand/result before the address
        // operand/result
        return 1 - idx;
      })
      .Case<handshake::MemoryControllerOp>(
          [&](handshake::MemoryControllerOp memOp) {
            if (isSrcOp)
              return idx;

            // Legacy Dynamatic puts all control operands before all data
            // operands, whereas for us each control operand appears just
            // before the data inputs of the block it corresponds to
            FuncMemoryPorts ports = memOp.getPorts();

            // auto groups = memOp.groupInputsByBB();

            // Determine total number of control operands
            unsigned ctrlCount = ports.getNumPorts(MemoryPort::Kind::CONTROL);

            // Figure out where the value lies
            auto [groupIdx, opIdx] = findValueInGroups(ports, val);

            // Figure out at which index the value would be in legacy
            // Dynamatic's interface
            bool valGroupHasControl = ports.blocks[groupIdx].hasControl();
            if (opIdx == 0 && valGroupHasControl) {
              // Value is a control input
              size_t fixedIdx = 0;
              for (size_t i = 0; i < groupIdx; i++)
                if (ports.blocks[i].hasControl())
                  fixedIdx++;
              return fixedIdx;
            }

            // Value is a data input
            size_t fixedIdx = ctrlCount;
            for (size_t i = 0; i < groupIdx; i++)
              // Add number of data inputs corresponding to the block
              if (ports.blocks[i].hasControl())
                fixedIdx += ports.blocks[i].getNumInputs() - 1;
              else
                fixedIdx += ports.blocks[i].getNumInputs();

            // Add index offset in the group the value belongs to
            if (valGroupHasControl)
              fixedIdx += opIdx - 1;
            else
              fixedIdx += opIdx;
            return fixedIdx;
          })
      .Default([&](auto) { return idx; });
}
} // namespace

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
  GraphNode *node = new GraphNode(opName, pos);

  // Add the ports to the node
  for (auto &inPort : inPorts)
    node->addPort(inPort, true);
  for (auto &outPort : outPorts)
    node->addPort(outPort, false);

  // Add the node to the graph
  graph->addNode(*node);

  return success();
}

LogicalResult MLIRMapper::mapEdge(Operation *src, Operation *dst, Value val,
                                  EdgeId *edgeId) {
  // Find the source and destination nodes
  std::string srcNodeName = getNodeName(src);
  std::string dstNodeName = getNodeName(dst);
  GraphNode *srcNodeIt;
  GraphNode *dstNodeIt;
  if (failed(graph->getNode(srcNodeName, *srcNodeIt)) ||
      failed(graph->getNode(dstNodeName, *dstNodeIt)))
    return failure();

  // Find the source value
  Value srcVal = val.getDefiningOp()->getResult(0);

  // Locate value in source results and destination operands
  auto resIdx = findIndexInRange(src->getResults(), srcVal);
  auto argIdx = findIndexInRange(dst->getOperands(), val);

  // Find the source and destination ports
  int from = fixPortNumber(src, srcVal, resIdx, true) + 1;
  int to = fixPortNumber(dst, val, argIdx, false) + 1;

  // Create the edge
  GraphEdge *edge =
      new GraphEdge(*edgeId++, *srcNodeIt, *dstNodeIt, from, to, {});

  // Add the edge to the graph
  graph->addEdge(*edge);

  return success();
}

std::string MLIRMapper::getNodeName(Operation *op) {
  auto opNameIt = opNameMap.find(op);
  if (opNameIt == opNameMap.end())
    return "";
  return opNameIt->second;
}
