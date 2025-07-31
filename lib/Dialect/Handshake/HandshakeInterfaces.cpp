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

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include <string>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

//===----------------------------------------------------------------------===//
// PortNameGenerator (uses NamedIOInterface)
//===----------------------------------------------------------------------===//

PortNamer::PortNamer(Operation *op) {
  assert(op && "cannot generate port names for null operation");
  if (auto funcOp = dyn_cast<handshake::FuncOp>(op)){
    inferFromFuncOp(funcOp);
  } else {
    inferFromInterface(op);
  }
}

void PortNamer::inferFromInterface(Operation *op) {
  auto customNamedOpInterface = dyn_cast<handshake::CustomNamedIOInterface>(op);
  auto simpleNamedOpInterface = dyn_cast<handshake::SimpleNamedIOInterface>(op);

  IdxToStrF inF, outF;
  if(customNamedOpInterface){
    inF = [&](unsigned idx) { return customNamedOpInterface.getOperandName(idx); };
    outF = [&](unsigned idx) { return customNamedOpInterface.getResultName(idx); };
  } else if (simpleNamedOpInterface) {
    inF = [&](unsigned idx) { return simpleNamedOpInterface.getOperandName(idx); };
    outF = [&](unsigned idx) { return simpleNamedOpInterface.getResultName(idx); };
  } else {
    op->emitError("all normal operations must specify port names");
    assert(false);
  }

  for (size_t idx = 0, e = op->getNumOperands(); idx < e; ++idx)
    inputs.push_back(inF(idx));
  for (size_t idx = 0, e = op->getNumResults(); idx < e; ++idx)
    outputs.push_back(outF(idx));

  // The Handshake terminator forwards its non-memory inputs to its outputs, so
  // it needs port names for them
  if (handshake::EndOp endOp = dyn_cast<handshake::EndOp>(op)) {
    handshake::FuncOp funcOp = endOp->getParentOfType<handshake::FuncOp>();
    assert(funcOp && "end must be child of handshake function");
    size_t numResults = funcOp.getFunctionType().getNumResults();
    for (size_t idx = 0, e = numResults; idx < e; ++idx)
      outputs.push_back(endOp.getDefaultResultName(idx));
  }
}

void PortNamer::inferFromFuncOp(handshake::FuncOp funcOp) {
  llvm::transform(funcOp.getArgNames(), std::back_inserter(inputs),
                  [](Attribute arg) { return cast<StringAttr>(arg).str(); });
  llvm::transform(funcOp.getResNames(), std::back_inserter(outputs),
                  [](Attribute res) { return cast<StringAttr>(res).str(); });
}

//===----------------------------------------------------------------------===//
// NamedIOInterface (getOperandName/getResultName)
//===----------------------------------------------------------------------===//

static inline std::string getArrayElemName(const Twine &name, unsigned idx) {
  return name.str() + "_" + std::to_string(idx);
}

std::string handshake::MuxOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");
  return idx == 0 ? "index" : "ins_" + std::to_string(idx - 1);
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
    return "ins_" + std::to_string(idx);
  return "memDone_" + std::to_string(idx - numResults);
}

std::string handshake::SelectOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");
  if (idx == 0)
    return "condition";
  return (idx == 1) ? "trueValue" : "falseValue";
}

std::string handshake::SelectOp::getResultName(unsigned idx) {
  assert(idx == 0 && "index too high");
  return "result";
}

/// Load/Store base signal names common to all memory interfaces
static constexpr llvm::StringLiteral MEMREF("memref"), MEM_START("memStart"),
    MEM_END("memEnd"), CTRL_END("ctrlEnd"), CTRL("ctrl"), LD_ADDR("ldAddr"),
    LD_DATA("ldData"), ST_ADDR("stAddr"), ST_DATA("stData");

static StringRef getIfControlOprd(MemoryOpInterface memOp, unsigned idx) {
  if (!memOp.isMasterInterface())
    return "";
  switch (idx) {
  case 0:
    return MEMREF;
  case 1:
    return MEM_START;
  default:
    return idx == memOp->getNumOperands() - 1 ? CTRL_END : "";
  }
}

static StringRef getIfControlRes(MemoryOpInterface memOp, unsigned idx) {
  if (memOp.isMasterInterface() && idx == memOp->getNumResults() - 1)
    return MEM_END;
  return "";
}

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
static std::string getMemResultName(FuncMemoryPorts &ports, unsigned idx) {
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

  if (StringRef name = getIfControlOprd(*this, idx); !name.empty())
    return name.str();

  // Try to get the operand name from the regular ports
  MCPorts mcPorts = getPorts();
  if (std::string name = getMemOperandName(mcPorts, idx); !name.empty())
    return name;

  // Get the operand name from a port to an LSQ
  assert(mcPorts.connectsToLSQ() && "expected MC to connect to LSQ");
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

  if (StringRef name = getIfControlRes(*this, idx); !name.empty())
    return name.str();

  // Try to get the operand name from the regular ports
  MCPorts mcPorts = getPorts();
  if (std::string name = getMemResultName(mcPorts, idx); !name.empty())
    return name;

  // Get the operand name from a port to an LSQ
  assert(mcPorts.connectsToLSQ() && "expected MC to connect to LSQ");
  LSQLoadStorePort lsqPort = mcPorts.getLSQPort();
  assert(lsqPort.getLoadDataOutputIndex() == idx && "unknown MC/LSQ result");
  return getArrayElemName(LD_DATA, mcPorts.getNumPorts<LoadPort>());
}

std::string handshake::LSQOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");

  if (StringRef name = getIfControlOprd(*this, idx); !name.empty())
    return name.str();

  // Try to get the operand name from the regular ports
  LSQPorts lsqPorts = getPorts();
  if (std::string name = getMemOperandName(lsqPorts, idx); !name.empty())
    return name;

  // Get the operand name from a port to a memory controller
  assert(lsqPorts.connectsToMC() && "expected LSQ to connect to MC");
  assert(lsqPorts.getMCPort().getLoadDataInputIndex() == idx &&
         "unknown LSQ/MC operand");
  return "ldDataFromMC";
}

std::string handshake::LSQOp::getResultName(unsigned idx) {
  assert(idx < getNumResults() && "index too high");

  if (StringRef name = getIfControlRes(*this, idx); !name.empty())
    return name.str();

  // Try to get the operand name from the regular ports
  LSQPorts lsqPorts = getPorts();
  if (std::string name = getMemResultName(lsqPorts, idx); !name.empty())
    return name;

  // Get the operand name from a port to a memory controller
  assert(lsqPorts.connectsToMC() && "expected LSQ to connect to MC");
  MCLoadStorePort mcPort = lsqPorts.getMCPort();
  if (mcPort.getLoadAddrOutputIndex() == idx)
    return "ldAddrToMC";
  if (mcPort.getStoreAddrOutputIndex() == idx)
    return "stAddrToMC";
  assert(mcPort.getStoreDataOutputIndex() == idx && "unknown LSQ/MC result");
  return "stDataToMC";
}

std::string handshake::SharingWrapperOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");
  if (idx < getNumSharedOperands() * getNumSharedOperations()) {
    return "op" + std::to_string(idx / getNumSharedOperands()) + "in" +
           std::to_string(idx % getNumSharedOperands());
  }
  return "fromSharedUnitOut0";
}

std::string handshake::SharingWrapperOp::getResultName(unsigned idx) {
  assert(idx < getNumResults() && "index too high");
  if (idx < getNumSharedOperations())
    return "op" + std::to_string(idx) + "out0";
  return "toSharedUnitIn" + std::to_string(idx - getNumSharedOperations());
}

//===----------------------------------------------------------------------===//
// MemoryOpInterface
//===----------------------------------------------------------------------===//

bool MemoryControllerOp::isMasterInterface() { return true; }

bool LSQOp::isMasterInterface() { return !isConnectedToMC(); }

TypedValue<MemRefType> LSQOp::getMemRef() {
  if (handshake::MemoryControllerOp mcOp = getConnectedMC())
    return mcOp.getMemRef();
  return cast<TypedValue<MemRefType>>(getInputs().front());
}

TypedValue<ControlType> LSQOp::getMemStart() {
  if (MemoryControllerOp mcOp = getConnectedMC())
    return mcOp.getMemStart();
  return cast<TypedValue<ControlType>>(getOperand(1));
}

TypedValue<ControlType> LSQOp::getMemEnd() {
  if (MemoryControllerOp mcOp = getConnectedMC())
    return mcOp.getMemStart();
  return cast<TypedValue<ControlType>>(getResults().back());
}

TypedValue<ControlType> LSQOp::getCtrlEnd() {
  if (MemoryControllerOp mcOp = getConnectedMC())
    return mcOp.getCtrlEnd();
  return cast<TypedValue<ControlType>>(getOperands().back());
}


#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.cpp.inc"
