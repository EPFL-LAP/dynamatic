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
#include "dynamatic/Analysis/NameAnalysis.h"
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
// EagerForkLikeOpInterface
//===----------------------------------------------------------------------===//

int ForkOp::getNumEagerOutputs() { return getNumResults(); }
std::vector<EagerForkSentNamer> ForkOp::getInternalSentStateNamers() {
  std::vector<EagerForkSentNamer> ret;
  StringAttr nameAttr =
      getOperation()->getAttrOfType<mlir::StringAttr>(NameAnalysis::ATTR_NAME);
  assert(nameAttr &&
         "Cannot get names of sent states for operation without name");
  for (auto [i, res] : llvm::enumerate(getResults())) {
    unsigned width = 0;
    if (auto ct = dyn_cast<handshake::ChannelType>(res.getType())) {
      width = ct.getDataBitWidth();
    }
    EagerForkSentNamer state(nameAttr.str(), getResultName(i), width);
    ret.push_back(state);
  }
  return ret;
}

int ControlMergeOp::getNumEagerOutputs() { return 2; }
std::vector<EagerForkSentNamer> ControlMergeOp::getInternalSentStateNamers() {
  std::vector<EagerForkSentNamer> ret;
  StringAttr nameAttr =
      getOperation()->getAttrOfType<mlir::StringAttr>(NameAnalysis::ATTR_NAME);
  assert(nameAttr &&
         "Cannot get names of sent states for operation without name");

  for (auto [i, res] : llvm::enumerate(getResults())) {
    unsigned width = 0;
    if (auto ct = dyn_cast<handshake::ChannelType>(res.getType())) {
      width = ct.getDataBitWidth();
    }

    EagerForkSentNamer state(nameAttr.str(), getResultName(i), width);
    ret.push_back(state);
  }
  return ret;
}

//===----------------------------------------------------------------------===//
// BufferLikeOpInterface
//===----------------------------------------------------------------------===//

std::vector<BufferSlotFullNamer> ControlMergeOp::getInternalSlotStateNamers() {
  std::vector<BufferSlotFullNamer> ret(1);
  StringAttr nameAttr =
      getOperation()->getAttrOfType<mlir::StringAttr>(NameAnalysis::ATTR_NAME);
  assert(nameAttr &&
         "Cannot get names of slot states for operation without name");
  handshake::ChannelType ct = getIndex().getType();

  ret[0] = BufferSlotFullNamer(nameAttr.str(), "slot_full", "data",
                               ct.getDataBitWidth());
  return ret;
}

std::vector<BufferSlotFullNamer> LoadOp::getInternalSlotStateNamers() {
  std::vector<BufferSlotFullNamer> ret(2);
  StringAttr nameAttr =
      getOperation()->getAttrOfType<mlir::StringAttr>(NameAnalysis::ATTR_NAME);
  assert(nameAttr &&
         "Cannot get names of slot states for operation without name");

  ret[0] = BufferSlotFullNamer(nameAttr.str(), ADDR_SLOT_LIT.str() + "_full",
                               "NOT_ACCESSIBLE",
                               getAddress().getType().getDataBitWidth());
  ret[1] = BufferSlotFullNamer(nameAttr.str(), DATA_SLOT_LIT.str() + "_full",
                               "NOT_ACCESSIBLE",
                               getData().getType().getDataBitWidth());
  return ret;
}

std::vector<BufferSlotFullNamer> BufferOp::getInternalSlotStateNamers() {
  std::vector<BufferSlotFullNamer> ret(getNumSlots());
  StringAttr nameAttr =
      getOperation()->getAttrOfType<mlir::StringAttr>(NameAnalysis::ATTR_NAME);
  assert(nameAttr &&
         "Cannot get names of slot states for operation without name");
  unsigned width = 0;
  getBufferType();
  if (auto ct = dyn_cast<handshake::ChannelType>(getOperand().getType())) {
    width = ct.getDataBitWidth();
  } else if (auto ct =
                 dyn_cast<handshake::ControlType>(getOperand().getType())) {
    width = 0;
  } else {
    llvm::errs() << nameAttr << getOperand().getType() << "\n";
    assert(false && "Operand of BufferOp is not a channel");
  }

  BufferType t = getBufferType();
  switch (t) {
  case BufferType::ONE_SLOT_BREAK_DV:
    assert(ret.size() == 1);
    ret[0] = BufferSlotFullNamer(nameAttr.str(), /*full value=*/"outs_valid_i",
                                 /*data value=*/"data", width);
    break;
  case BufferType::ONE_SLOT_BREAK_R:
    assert(ret.size() == 1);
    ret[0] = BufferSlotFullNamer(nameAttr.str(), "full", "data", width);
    break;
  case BufferType::FIFO_BREAK_NONE:
    if (ret.size() == 1) {
      ret[0] = BufferSlotFullNamer(nameAttr.str(), "full", "reg", width);
    } else {
      for (size_t i = 0; i < ret.size(); ++i) {
        ret[i] = BufferSlotFullNamer(nameAttr.str(),
                                     llvm::formatv("b{0}.full", i).str(),
                                     llvm::formatv("b{0}.reg", i).str(), width);
      }
    }
    break;
  case BufferType::FIFO_BREAK_DV:
  case BufferType::ONE_SLOT_BREAK_DVR:
  case BufferType::SHIFT_REG_BREAK_DV:
  case BufferType::COUNTER_BUFFER:
    llvm::report_fatal_error(
        llvm::formatv("no name for buffer slot of type {0}", t));
  }
  return ret;
}
std::vector<BufferSlotFullNamer> DeadBufferOp::getInternalSlotStateNamers() {
  StringAttr nameAttr =
      getOperation()->getAttrOfType<mlir::StringAttr>(NameAnalysis::ATTR_NAME);
  assert(nameAttr &&
         "Cannot get names of slot states for operation without name");
  std::vector<BufferSlotFullNamer> ret;
  ret.emplace_back(nameAttr.str(), "full", "", 0);
  return ret;
}

//===----------------------------------------------------------------------===//
// ShiftLikeArithOpInterface
//===----------------------------------------------------------------------===//

static bool isShiftByConstantImpl(Operation *op) {
  auto rhs = op->getOperand(1);
  // Recursively visit the predecessor
  std::function<bool(Operation *)> isShiftByConstantRecursive =
      [&](Operation *op) {
        if (isa<
                // clang-format off
                handshake::TruncIOp,
                handshake::ExtSIOp,
                handshake::ExtUIOp,
                handshake::ForkOp
                // clang-format on
                >(op)) {

          Value oprd = op->getOperand(0);
          if (Operation *defOp = oprd.getDefiningOp(); defOp)
            return isShiftByConstantRecursive(defOp);
          assert(isa<BlockArgument>(oprd));
          return false;
        }
        return isa<handshake::ConstantOp>(op);
      };
  return isShiftByConstantRecursive(rhs.getDefiningOp());
}

bool ShLIOp::isShiftByConstant() {
  return isShiftByConstantImpl(this->getOperation());
}

bool ShRSIOp::isShiftByConstant() {
  return isShiftByConstantImpl(this->getOperation());
}

bool ShRUIOp::isShiftByConstant() {
  return isShiftByConstantImpl(this->getOperation());
}

//===----------------------------------------------------------------------===//
// RetimingPathsOpInterface
//===----------------------------------------------------------------------===//

SmallVector<buffer::RetimingPath> buffer::getRetimingPaths(Operation *unit) {
  if (auto pathsOp = dyn_cast<RetimingPathsOpInterface>(unit))
    return pathsOp.getRetimingPaths();
  return {buffer::RetimingPath(unit)};
}

/// SpeculatorOp has two independent retiming paths:
///   - trigger (operand 1) feeds dataOut (result 0) and issueCtrl (result 1):
///     these are produced when the speculator decides to issue a speculative
///     token.
///   - dataIn (operand 0) feeds historyCtrl (result 2) and commitCtrl
///     (result 3): once the real data arrives the speculator resolves the
///     speculation and emits the history-control and commit-control signals.
SmallVector<buffer::RetimingPath> SpeculatorOp::getRetimingPaths() {
  SmallVector<buffer::RetimingPath> paths;

  buffer::RetimingPath triggerPath;
  triggerPath.operands.insert(1); // trigger
  triggerPath.results.insert(0);  // dataOut
  triggerPath.results.insert(1);  // issueCtrl
  paths.push_back(triggerPath);

  buffer::RetimingPath dataInPath;
  dataInPath.operands.insert(0); // dataIn
  dataInPath.results.insert(2);  // historyCtrl
  dataInPath.results.insert(3);  // commitCtrl
  paths.push_back(dataInPath);

  return paths;
}

/// SpecSaveCommitOp has two independent retiming paths:
///   - dataIn (operand 0) and issueCtrl (operand 1) feed dataOut (result 0):
///     the save-commit issues output tokens when both data and the issue
///     control are present.
///   - historyCtrl (operand 2) is consumed independently to advance the
///     internal head pointer. It is on a path with no outputs so its retiming
///     variable is decoupled from the data path.
SmallVector<buffer::RetimingPath> SpecSaveCommitOp::getRetimingPaths() {
  SmallVector<buffer::RetimingPath> paths;

  buffer::RetimingPath dataPath;
  dataPath.operands.insert(0); // dataIn
  dataPath.operands.insert(1); // issueCtrl
  dataPath.results.insert(0);  // dataOut
  paths.push_back(dataPath);

  buffer::RetimingPath historyPath;
  historyPath.operands.insert(2); // historyCtrl
  paths.push_back(historyPath);

  return paths;
}

#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.cpp.inc"