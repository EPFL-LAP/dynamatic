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
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

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
  return idx == 0 ? "outs" : "index";
}

std::string handshake::ConditionalBranchOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");
  return idx == 0 ? "condition" : "data";
}

std::string handshake::ConditionalBranchOp::getResultName(unsigned idx) {
  assert(idx < getNumResults() && "index too high");
  return idx == ConditionalBranchOp::trueIndex ? "trueOut" : "falseOut";
}

std::string handshake::ConstantOp::getOperandName(unsigned idx) {
  assert(idx == 0 && "index too high");
  return "ctrl";
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

std::string handshake::SelectOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");
  if (idx == 0)
    return "condition";
  return (idx == 1) ? "lhs" : "rhs";
}

std::string handshake::SelectOp::getResultName(unsigned idx) {
  assert(idx == 0 && "index too high");
  return "result";
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

  // Get the operand name from a port to an LSQ
  assert(mcPorts.hasConnectionToLSQ() && "expected MC to connect to LSQ");
  LSQLoadStorePort lsqPort = mcPorts.getLSQPort();
  if (lsqPort.getLoadAddrInputIndex() == idx)
    return getArrayElemName(LD_ADDR, mcPorts.getNumPorts<LoadPort>());
  if (lsqPort.getStoreAddrInputIndex() == idx)
    return getArrayElemName(ST_ADDR, mcPorts.getNumPorts<StorePort>());
  assert(lsqPort.getStoreDataInputIndex() == idx && "unknown MC/LSQ operand");
  return getArrayElemName(ST_DATA, mcPorts.getNumPorts<StorePort>());
}

std::string handshake::MemoryControllerOp::getResultName(unsigned idx) {
  assert(idx < getNumResults() && "index too high");

  // Try to get the operand name from the regular ports
  MCPorts mcPorts = getPorts();
  if (std::string name = getMemResultName(mcPorts, idx); !name.empty())
    return name;

  // Get the operand name from a port to an LSQ
  assert(mcPorts.hasConnectionToLSQ() && "expected MC to connect to LSQ");
  LSQLoadStorePort lsqPort = mcPorts.getLSQPort();
  assert(lsqPort.getLoadDataOutputIndex() == idx && "unknown MC/LSQ result");
  return getArrayElemName(LD_DATA, mcPorts.getNumPorts<LoadPort>());
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

  // Get the operand name from a port to a memory controller
  assert(lsqPorts.hasConnectionToMC() && "expected LSQ to connect to MC");
  assert(lsqPorts.getMCPort().getLoadDataInputIndex() == idx &&
         "unknown LSQ/MC operand");
  return "ldDataFromMC";
}

std::string handshake::LSQOp::getResultName(unsigned idx) {
  assert(idx < getNumResults() && "index too high");

  // Try to get the operand name from the regular ports
  LSQPorts lsqPorts = getPorts();
  if (std::string name = getMemResultName(lsqPorts, idx); !name.empty())
    return name;

  // Get the operand name from a port to a memory controller
  assert(lsqPorts.hasConnectionToMC() && "expected LSQ to connect to MC");
  MCLoadStorePort mcPort = lsqPorts.getMCPort();
  if (mcPort.getLoadAddrOutputIndex() == idx)
    return "ldAddrToMC";
  if (mcPort.getStoreAddrOutputIndex() == idx)
    return "stAddrToMC";
  assert(mcPort.getStoreDataOutputIndex() == idx && "unknown LSQ/MC result");
  return "stDataToMC";
}

//===----------------------------------------------------------------------===//
// ControlInterface
//===----------------------------------------------------------------------===//

bool dynamatic::handshake::isControlOpImpl(Operation *op) {
  if (SOSTInterface sostInterface = dyn_cast<SOSTInterface>(op); sostInterface)
    return sostInterface.sostIsControl();
  return false;
}

//===----------------------------------------------------------------------===//
// PreservesExtraSignals
//===----------------------------------------------------------------------===//

LogicalResult
dynamatic::handshake::detail::verifyPreservesExtraSignals(Operation *op) {
  std::optional<ArrayRef<ExtraSignal>> refExtras;

  /// Identify all channel-typed operands and results
  auto checkCompatible = [&](ValueRange values) -> LogicalResult {
    for (Value val : values) {
      auto channelType = dyn_cast<handshake::ChannelType>(val.getType());
      if (!channelType)
        continue;
      if (!refExtras) {
        refExtras = channelType.getExtraSignals();
        continue;
      }
      ArrayRef<ExtraSignal> extras = channelType.getExtraSignals();
      if (refExtras->size() != extras.size())
        return op->emitError() << "incompatible number of extra signals "
                                  "between two operand/result channel types";
      auto signalsZip = llvm::zip(*refExtras, extras);
      for (const auto &[idx, signals] : llvm::enumerate(signalsZip)) {
        auto &[refSig, sig] = signals;
        if (refSig != sig)
          return op->emitError()
                 << "different " << idx
                 << "-th extra signal between two operand/result channel types";
      }
    }
    return success();
  };

  return failure(failed(checkCompatible(op->getOperands())) ||
                 failed(checkCompatible(op->getResults())));
}

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.cpp.inc"
