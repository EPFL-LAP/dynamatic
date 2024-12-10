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
  if (auto namedOpInterface = dyn_cast<handshake::NamedIOInterface>(op))
    inferFromNamedOpInterface(namedOpInterface);
  else if (auto funcOp = dyn_cast<handshake::FuncOp>(op))
    inferFromFuncOp(funcOp);
  else
    inferDefault(op);
}

void PortNamer::infer(Operation *op, IdxToStrF &inF, IdxToStrF &outF) {
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

void PortNamer::inferDefault(Operation *op) {
  llvm::TypeSwitch<Operation *, void>(op)
      .Case<arith::AddFOp, arith::AddIOp, arith::AndIOp, arith::CmpIOp,
            arith::CmpFOp, arith::DivFOp, arith::DivSIOp, arith::DivUIOp,
            arith::MaximumFOp, arith::MinimumFOp, arith::MulFOp, arith::MulIOp,
            arith::OrIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
            arith::SubFOp, arith::SubIOp, arith::XOrIOp>([&](auto) {
        infer(
            op, [](unsigned idx) { return idx == 0 ? "lhs" : "rhs"; },
            [](unsigned idx) { return "result"; });
      })
      .Case<arith::ExtSIOp, arith::ExtUIOp, arith::NegFOp, arith::TruncIOp>(
          [&](auto) {
            infer(
                op, [](unsigned idx) { return "ins"; },
                [](unsigned idx) { return "outs"; });
          })
      .Case<arith::SelectOp>([&](auto) {
        infer(
            op,
            [](unsigned idx) {
              if (idx == 0)
                return "condition";
              if (idx == 1)
                return "trueValue";
              return "falseValue";
            },
            [](unsigned idx) { return "result"; });
      })
      .Default([&](auto) {
        infer(
            op, [](unsigned idx) { return "in" + std::to_string(idx); },
            [](unsigned idx) { return "out" + std::to_string(idx); });
      });
}

void PortNamer::inferFromNamedOpInterface(handshake::NamedIOInterface namedIO) {
  auto inF = [&](unsigned idx) { return namedIO.getOperandName(idx); };
  auto outF = [&](unsigned idx) { return namedIO.getResultName(idx); };
  infer(namedIO, inF, outF);
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

std::string handshake::SpeculatingBranchOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");
  return idx == 0 ? "spec_tag_data" : "data";
}

std::string handshake::SpeculatingBranchOp::getResultName(unsigned idx) {
  assert(idx < getNumResults() && "index too high");
  return idx == SpeculatingBranchOp::trueIndex ? "trueOut" : "falseOut";
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
  return (idx == 1) ? "trueValue" : "falseValue";
}

std::string handshake::SelectOp::getResultName(unsigned idx) {
  assert(idx == 0 && "index too high");
  return "result";
}

std::string handshake::SpeculatorOp::getOperandName(unsigned idx) {
  assert(idx < getNumOperands() && "index too high");
  return idx == 0 ? "ins" : "enable";
}

std::string handshake::SpeculatorOp::getResultName(unsigned idx) {
  assert(idx < getNumResults() && "index too high");
  switch (idx) {
  case 0:
    return "outs";
  case 1:
    return "ctrl_save";
  case 2:
    return "ctrl_commit";
  case 3:
    return "ctrl_sc_save";
  case 4:
    return "ctrl_sc_commit";
  default:
    return "ctrl_sc_branch";
  }
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

//===----------------------------------------------------------------------===//
// SameExtraSignalsInterface
//===----------------------------------------------------------------------===//

namespace {
using ChannelVal = TypedValue<handshake::ChannelType>;
} // namespace

static inline ChannelVal toChannel(Value val) { return cast<ChannelVal>(val); }

static void insertChannels(ValueRange values,
                           SmallVectorImpl<ChannelVal> &channels) {
  for (Value val : values) {
    if (auto channelVal = dyn_cast<ChannelVal>(val))
      channels.push_back(channelVal);
  }
}

SmallVector<ChannelVal>
dynamatic::handshake::detail::getChannelsWithSameExtraSignals(Operation *op) {
  SmallVector<ChannelVal> channels;
  insertChannels(op->getOperands(), channels);
  insertChannels(op->getResults(), channels);
  return channels;
}

LogicalResult dynamatic::handshake::detail::verifySameExtraSignalsInterface(
    Operation *op, ArrayRef<ChannelVal> channels) {
  std::optional<ArrayRef<ExtraSignal>> refExtras;

  for (TypedValue<ChannelType> chan : channels) {
    if (!refExtras) {
      refExtras = chan.getType().getExtraSignals();
      continue;
    }
    ArrayRef<ExtraSignal> extras = chan.getType().getExtraSignals();
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
}

SmallVector<ChannelVal> SelectOp::getChannelsWithSameExtraSignals() {
  return {getTrueValue(), getFalseValue(), getResult()};
}

//===----------------------------------------------------------------------===//
// ReshapableChannelsInterface
//===----------------------------------------------------------------------===//

std::pair<handshake::ChannelType, bool>
dynamatic::handshake::detail::getReshapableChannelType(Operation *op) {
  return {dyn_cast<ChannelType>(op->getOperands().front().getType()), false};
}

std::pair<ChannelType, bool> MergeOp::getReshapableChannelType() {
  return {dyn_cast<ChannelType>(getDataOperands().front().getType()), true};
}

std::pair<ChannelType, bool> MuxOp::getReshapableChannelType() {
  return {dyn_cast<ChannelType>(getDataOperands().front().getType()), true};
}

std::pair<ChannelType, bool> ControlMergeOp::getReshapableChannelType() {
  return {dyn_cast<ChannelType>(getDataOperands().front().getType()), true};
}

std::pair<ChannelType, bool> BranchOp::getReshapableChannelType() {
  return {dyn_cast<ChannelType>(getOperand().getType()), true};
}

std::pair<ChannelType, bool> ConditionalBranchOp::getReshapableChannelType() {
  return {dyn_cast<ChannelType>(getDataOperand().getType()), true};
}

std::pair<ChannelType, bool> SelectOp::getReshapableChannelType() {
  return {getTrueValue().getType(), true};
}

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.cpp.inc"
