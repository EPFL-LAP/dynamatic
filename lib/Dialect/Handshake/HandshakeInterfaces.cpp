//===- HandshakeInterfaces.cpp - Handshake interfaces -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of Handshake dialect's interfaces' methods for specific
// Handshake operations.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/ErrorHandling.h"

using namespace dynamatic;

//===----------------------------------------------------------------------===//
// NamedIOInterface (getOperandName/getResultName)
//===----------------------------------------------------------------------===//

static inline std::string getArrayElemName(const Twine &name, unsigned idx) {
  return name.str() + "_" + std::to_string(idx);
}

std::string handshake::MuxOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");
  return idx == 0 ? "index" : getDefaultOperandName(idx - 1);
}

std::string handshake::ControlMergeOp::getResultName(unsigned idx) {
  assert(idx < getNumResults() && "index too high");
  return idx == 0 ? "index" : getDefaultResultName(idx - 1);
}

std::string handshake::ConditionalBranchOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");
  return idx == 0 ? "condition" : "ins";
}

std::string handshake::ConditionalBranchOp::getResultName(unsigned idx) {
  assert(idx < getNumResults() && "index too high");
  return idx == ConditionalBranchOp::trueIndex ? "trueOut" : "falseOut";
}

std::string handshake::ConstantOp::getOperandName(unsigned idx) {
  assert(idx == 0 && "index too high");
  return "control";
}

std::string handshake::EndOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");
  handshake::FuncOp funcOp = (*this)->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "end must be child of handshake function");

  unsigned numResults = funcOp.getFunctionType().getNumResults();
  if (idx < numResults)
    return getDefaultOperandName(idx);
  return "memDone_" + std::to_string(idx - numResults);
}

/// Load/Store base signal names common to all memory interfaces
static constexpr llvm::StringLiteral MEMREF("memref"), CTRL("ctrl"),
    LD_ADDR("ldAddr"), LD_DATA("ldData"), ST_ADDR("stAddr"), ST_DATA("stData");

/// Common operand naming logic for memory controllers and LSQs.
static std::string getMemOperandName(const FuncMemoryPorts &ports,
                                     unsigned idx) {
  // Iterate through all memory ports to find out the type of the operand
  unsigned ctrlIdx = 0, loadIdx = 0, storeIdx = 0;
  for (const GroupMemoryPorts &blockPorts : ports.groups) {
    if (blockPorts.hasControl()) {
      if (idx == blockPorts.ctrlPort->getCtrlInputIndex())
        return getArrayElemName(CTRL, ctrlIdx);
      ++ctrlIdx;
    }
    for (const MemoryPort &accessPort : blockPorts.accessPorts) {
      if (std::optional<LoadPort> loadPort = dyn_cast<LoadPort>(accessPort)) {
        if (loadPort->getAddrInputIndex() == idx)
          return getArrayElemName(LD_ADDR, loadIdx);
        ++loadIdx;
      } else {
        std::optional<StorePort> storePort = cast<StorePort>(accessPort);
        if (storePort->getAddrInputIndex() == idx)
          return getArrayElemName(ST_ADDR, storeIdx);
        if (storePort->getDataInputIndex() == idx)
          return getArrayElemName(ST_DATA, storeIdx);
        ++storeIdx;
      }
    }
  }

  return "";
}

/// Common result naming logic for memory controllers and LSQs.
static std::string getMemResultName(const FuncMemoryPorts &ports,
                                    unsigned idx) {
  if (idx == ports.memOp->getNumResults() - 1)
    return "memDone";

  // Iterate through all memory ports to find out the type of the
  // operand
  unsigned loadIdx = 0;
  for (const GroupMemoryPorts &blockPorts : ports.groups) {
    for (const MemoryPort &accessPort : blockPorts.accessPorts) {
      if (std::optional<LoadPort> loadPort = dyn_cast<LoadPort>(accessPort)) {
        if (loadPort->getDataOutputIndex() == idx)
          return getArrayElemName(LD_DATA, loadIdx);
        ++loadIdx;
      }
    }
  }
  return "";
}

std::string handshake::MemoryControllerOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");

  if (idx == 0)
    return MEMREF.str();

  // Try to get the operand name from the regular ports
  MCPorts mcPorts = getPorts();
  if (std::string name = getMemOperandName(mcPorts, idx); !name.empty())
    return name;

  // Try to get the operand name from a potential LSQ port
  if (mcPorts.hasConnectionToLSQ()) {
    LSQLoadStorePort lsqPort = mcPorts.getLSQPort();
    if (lsqPort.getLoadAddrInputIndex() == idx)
      return "lsqLdAddr";
    if (lsqPort.getStoreAddrInputIndex() == idx)
      return "lsqStAddr";
    if (lsqPort.getStoreDataInputIndex() == idx)
      return "lsqStData";
  }
  llvm_unreachable("faulty port logic");
}

std::string handshake::MemoryControllerOp::getResultName(unsigned idx) {
  assert(idx < getNumResults() && "index too high");

  // Try to get the operand name from the regular ports
  MCPorts mcPorts = getPorts();
  if (std::string name = getMemResultName(mcPorts, idx); !name.empty())
    return name;

  // Try to get the result name from a potential LSQ port
  if (mcPorts.hasConnectionToLSQ()) {
    LSQLoadStorePort lsqPort = mcPorts.getLSQPort();
    if (lsqPort.getLoadDataOutputIndex() == idx)
      return "lsqLdData";
  }
  llvm_unreachable("faulty port logic");
}

std::string handshake::LSQOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");

  bool connectsToMC = isConnectedToMC();
  if (idx == 0 && !connectsToMC)
    return MEMREF.str();

  // Try to get the operand name from the regular ports
  LSQPorts lsqPorts = getPorts();
  if (std::string name = getMemOperandName(lsqPorts, idx); !name.empty())
    return name;

  // Try to get the operand name from a potential MC port
  if (connectsToMC) {
    MCLoadStorePort mcPort = lsqPorts.getMCPort();
    if (mcPort.getLoadDataInputIndex() == idx)
      return "mcLdData";
  }
  llvm_unreachable("faulty port logic");
}

std::string handshake::LSQOp::getResultName(unsigned idx) {
  assert(idx < getNumResults() && "index too high");

  // Try to get the operand name from the regular ports
  LSQPorts lsqPorts = getPorts();
  if (std::string name = getMemResultName(lsqPorts, idx); !name.empty())
    return name;

  // Go through ports to other memory interfaces
  // Try to get the operand name from a potential MC port
  if (lsqPorts.hasConnectionToMC()) {
    MCLoadStorePort mcPort = lsqPorts.getMCPort();
    if (mcPort.getLoadAddrOutputIndex() == idx)
      return "mcLdAddr";
    if (mcPort.getStoreAddrOutputIndex() == idx)
      return "mcStAddr";
    if (mcPort.getStoreDataOutputIndex() == idx)
      return "mcStData";
  }
  llvm_unreachable("faulty port logic");
}

//===----------------------------------------------------------------------===//
// ControlInterface
//===----------------------------------------------------------------------===//

bool dynamatic::handshake::isControlOpImpl(Operation *op) {
  if (SOSTInterface sostInterface = dyn_cast<SOSTInterface>(op); sostInterface)
    return sostInterface.sostIsControl();
  return false;
}

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.cpp.inc"
