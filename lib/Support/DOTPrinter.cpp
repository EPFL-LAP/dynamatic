//===- DOTPrinter.cpp - Print DOT to standard output ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the DOT printer. The printer
// produces, on standard output, Graphviz-formatted output which contains the
// graph representation of the input handshake-level IR.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/DOTPrinter.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/MemoryInterfaces.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "experimental/Transforms/Speculation/SpecAnnotatePaths.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <iomanip>
#include <string>
#include <utility>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

namespace {

/// A list of ports (name and value).
using PortsData = std::vector<std::pair<std::string, Value>>;
/// A list of ports for memory interfaces (name, value, and potential name
/// suffix).
using MemPortsData = std::vector<std::tuple<std::string, Value, std::string>>;
/// In legacy mode, a port represented with a unique name and a bitwidth
using RawPort = std::pair<std::string, unsigned>;

} // namespace

// ============================================================================
// Legacy node/edge attributes
// ============================================================================

/// Maps name of arith/math operations to "op" attribute.
static llvm::StringMap<StringRef> compNameToOpName{
    {arith::AddFOp::getOperationName(), "fadd_op"},
    {arith::AddIOp::getOperationName(), "add_op"},
    {arith::AndIOp::getOperationName(), "and_op"},
    {arith::DivFOp::getOperationName(), "fdiv_op"},
    {arith::DivSIOp::getOperationName(), "sdiv_op"},
    {arith::DivUIOp::getOperationName(), "udiv_op"},
    {arith::ExtSIOp::getOperationName(), "sext_op"},
    {arith::ExtUIOp::getOperationName(), "zext_op"},
    {arith::ExtFOp::getOperationName(), "fext_op"},
    {arith::MulFOp::getOperationName(), "fmul_op"},
    {arith::MulIOp::getOperationName(), "mul_op"},
    {arith::OrIOp::getOperationName(), "or_op"},
    {arith::RemFOp::getOperationName(), "frem_op"},
    {arith::RemSIOp::getOperationName(), "srem_op"},
    {arith::RemUIOp::getOperationName(), "urem_op"},
    {arith::SelectOp::getOperationName(), "select_op"},
    {arith::ShLIOp::getOperationName(), "shl_op"},
    {arith::SubIOp::getOperationName(), "sub_op"},
    {arith::SubFOp::getOperationName(), "fsub_op"},
    {arith::ShRSIOp::getOperationName(), "ashr_op"},
    {arith::SIToFPOp::getOperationName(), "sitofp_op"},
    {arith::TruncIOp::getOperationName(), "trunc_op"},
    {arith::XOrIOp::getOperationName(), "xor_op"},
    {math::CosOp::getOperationName(), "cosf_op"},
    {math::ExpOp::getOperationName(), "expf_op"},
    {math::Exp2Op::getOperationName(), "exp2f_op"},
    {math::LogOp::getOperationName(), "logf_op"},
    {math::Log2Op::getOperationName(), "log2f_op"},
    {math::Log10Op::getOperationName(), "log10f_op"},
    {math::SinOp::getOperationName(), "sinf_op"},
    {math::SqrtOp::getOperationName(), "sqrtf_op"},
};

/// Maps name of integer comparison type to "op" attribute.
static DenseMap<arith::CmpIPredicate, StringRef> cmpINameToOpName{
    {arith::CmpIPredicate::eq, "icmp_eq_op"},
    {arith::CmpIPredicate::ne, "icmp_ne_op"},
    {arith::CmpIPredicate::slt, "icmp_slt_op"},
    {arith::CmpIPredicate::sle, "icmp_sle_op"},
    {arith::CmpIPredicate::sgt, "icmp_sgt_op"},
    {arith::CmpIPredicate::sge, "icmp_sge_op"},
    {arith::CmpIPredicate::ult, "icmp_ult_op"},
    {arith::CmpIPredicate::ule, "icmp_ule_op"},
    {arith::CmpIPredicate::ugt, "icmp_ugt_op"},
    {arith::CmpIPredicate::uge, "icmp_uge_op"},
};

/// Maps name of floating-point comparison type to "op" attribute.
static DenseMap<arith::CmpFPredicate, StringRef> cmpFNameToOpName{
    {arith::CmpFPredicate::AlwaysFalse, "fcmp_false_op"},
    {arith::CmpFPredicate::OEQ, "fcmp_oeq_op"},
    {arith::CmpFPredicate::OGT, "fcmp_ogt_op"},
    {arith::CmpFPredicate::OGE, "fcmp_oge_op"},
    {arith::CmpFPredicate::OLT, "fcmp_olt_op"},
    {arith::CmpFPredicate::OLE, "fcmp_ole_op"},
    {arith::CmpFPredicate::ONE, "fcmp_one_op"},
    {arith::CmpFPredicate::ORD, "fcmp_orq_op"},
    {arith::CmpFPredicate::UEQ, "fcmp_ueq_op"},
    {arith::CmpFPredicate::UGT, "fcmp_ugt_op"},
    {arith::CmpFPredicate::UGE, "fcmp_uge_op"},
    {arith::CmpFPredicate::ULT, "fcmp_ult_op"},
    {arith::CmpFPredicate::ULE, "fcmp_ule_op"},
    {arith::CmpFPredicate::UNE, "fcmp_une_op"},
    {arith::CmpFPredicate::UNO, "fcmp_uno_op"},
    {arith::CmpFPredicate::AlwaysTrue, "fcmp_true_op"},
};

/// Returns the width of a "handshake type" (i.e., width of data signal). In
/// particular, returns 0 for NoneType's.
static unsigned getWidth(Type dataType) {
  if (isa<NoneType>(dataType))
    return 0;
  return dataType.getIntOrFloatBitWidth();
}

/// Returns the width of a handshake value (i.e., width of data signal). In
/// particular, returns 0 for NoneType's.
static unsigned getWidth(Value value) { return getWidth(value.getType()); }

/// Turns a list of ports into a string that can be used as the value for the
/// "in" or "out" attribute of a node.
static std::string getIOFromPorts(PortsData ports) {
  std::stringstream stream;
  for (auto [idx, port] : llvm::enumerate(ports)) {
    stream << port.first << ":" << getWidth(port.second);
    if (idx != (ports.size() - 1))
      stream << " ";
  }
  return stream.str();
}

/// Turns a list of ports into a string that can be used as the value for the
/// "in" or "out" attribute of a node.
static std::string getIOFromPorts(MemPortsData ports) {
  std::stringstream stream;
  for (auto [idx, port] : llvm::enumerate(ports)) {
    stream << std::get<0>(port) << ":" << getWidth(std::get<1>(port));
    if (auto suffix = std::get<2>(port); !suffix.empty())
      stream << "*" << suffix;
    if (idx != (ports.size() - 1))
      stream << " ";
  }
  return stream.str();
}

/// Creates the "in" or "out" attribute of a node from a list of values and a
/// port name (used as prefix to derive numbered port names for all values).
static std::string getIOFromValues(ValueRange values, std::string &&portType) {
  PortsData ports;
  for (auto [idx, val] : llvm::enumerate(values))
    ports.emplace_back(portType + std::to_string(idx + 1), val);
  return getIOFromPorts(ports);
}

/// Produces the "in" attribute value of a handshake::MuxOp.
static std::string getInputForMux(handshake::MuxOp op) {
  PortsData ports;
  ports.emplace_back("in1?", op.getSelectOperand());
  for (auto [idx, val] : llvm::enumerate(op->getOperands().drop_front(1)))
    ports.emplace_back("in" + std::to_string(idx + 2), val);
  return getIOFromPorts(ports);
}

/// Produces the "out" attribute value of a handshake::ControlMergeOp.
static std::string getOutputForControlMerge(handshake::ControlMergeOp op) {
  return "out1:" + std::to_string(getWidth(op.getResult())) +
         " out2?:" + std::to_string((int)ceil(log2(op->getNumOperands())));
}

/// Produces the "in" attribute value of a handshake::ConditionalBranchOp.
static std::string getInputForCondBranch(handshake::ConditionalBranchOp op) {
  PortsData ports;
  ports.emplace_back("in1", op.getDataOperand());
  ports.emplace_back("in2?", op.getConditionOperand());
  return getIOFromPorts(ports);
}

/// Produces the "out" attribute value of a handshake::ConditionalBranchOp.
static std::string getOutputForCondBranch(handshake::ConditionalBranchOp op) {
  PortsData ports;
  ports.emplace_back("out1+", op.getTrueResult());
  ports.emplace_back("out2-", op.getFalseResult());
  return getIOFromPorts(ports);
}

/// Produces the "in" attribute value of a handshake::EndOp.
static std::string getInputForEnd(handshake::EndOp op) {
  MemPortsData ports;
  unsigned idx = 1;
  for (auto val : op.getMemoryControls())
    ports.emplace_back("in" + std::to_string(idx++), val, "e");
  for (auto val : op.getReturnValues())
    ports.emplace_back("in" + std::to_string(idx++), val, "");
  return getIOFromPorts(ports);
}

/// Produces the "in" attribute value of a handshake::LoadOpInterface.
static std::string getInputForLoadOp(handshake::LoadOpInterface op) {
  PortsData ports;
  ports.emplace_back("in1", op.getDataInput());
  ports.emplace_back("in2", op.getAddressInput());
  return getIOFromPorts(ports);
}

/// Produces the "out" attribute value of a handshake::LoadOpInterface.
static std::string getOutputForLoadOp(handshake::LoadOpInterface op) {
  PortsData ports;
  ports.emplace_back("out1", op.getDataOutput());
  ports.emplace_back("out2", op.getAddressOutput());
  return getIOFromPorts(ports);
}

/// Produces the "in" attribute value of a handshake::StoreOpInterface.
static std::string getInputForStoreOp(handshake::StoreOpInterface op) {
  PortsData ports;
  ports.emplace_back("in1", op.getDataInput());
  ports.emplace_back("in2", op.getAddressInput());
  return getIOFromPorts(ports);
}

/// Produces the "out" attribute value of a handshake::StoreOpInterface.
static std::string getOutputForStoreOp(handshake::StoreOpInterface op) {
  PortsData ports;
  ports.emplace_back("out1", op.getDataOutput());
  ports.emplace_back("out2", op.getAddressOutput());
  return getIOFromPorts(ports);
}

/// Produces the "in" attribute value of a memory interface.
static std::string getInputForMemInterface(FuncMemoryPorts &funcPorts) {
  MemPortsData portsData;
  unsigned ctrlIdx = 0, ldIdx = 0, stIdx = 0, inputIdx = 1;
  MemoryOpInterface memOp = funcPorts.memOp;
  ValueRange inputs = memOp->getOperands();

  // Add all control signals first
  for (GroupMemoryPorts &blockPorts : funcPorts.groups) {
    if (blockPorts.hasControl())
      portsData.emplace_back("in" + std::to_string(inputIdx++),
                             inputs[blockPorts.ctrlPort->getCtrlInputIndex()],
                             "c" + std::to_string(ctrlIdx++));
  }

  // Then all memory access signals
  for (GroupMemoryPorts &blockPorts : funcPorts.groups) {
    // Add loads and stores, in program order
    for (MemoryPort &port : blockPorts.accessPorts) {
      if (std::optional<LoadPort> loadPort = dyn_cast<LoadPort>(port)) {
        portsData.emplace_back("in" + std::to_string(inputIdx++),
                               inputs[loadPort->getAddrInputIndex()],
                               "l" + std::to_string(ldIdx++) + "a");
      } else {
        std::optional<StorePort> storePort = dyn_cast<StorePort>(port);
        assert(storePort && "port must be load or store");
        // Address signal first, then data signal
        portsData.emplace_back("in" + std::to_string(inputIdx++),
                               inputs[storePort->getAddrInputIndex()],
                               "s" + std::to_string(stIdx) + "a");
        portsData.emplace_back("in" + std::to_string(inputIdx++),
                               inputs[storePort->getDataInputIndex()],
                               "s" + std::to_string(stIdx++) + "d");
      }
    }
  }

  // Finally, all signals from other interfaces
  for (MemoryPort &port : funcPorts.interfacePorts) {
    if (std::optional<LSQLoadStorePort> lsqPort =
            dyn_cast<LSQLoadStorePort>(port)) {
      // Load address, then store address, then store data
      portsData.emplace_back("in" + std::to_string(inputIdx++),
                             inputs[lsqPort->getLoadAddrInputIndex()],
                             "l" + std::to_string(ldIdx++) + "a");
      portsData.emplace_back("in" + std::to_string(inputIdx++),
                             inputs[lsqPort->getStoreAddrInputIndex()],
                             "s" + std::to_string(stIdx) + "a");
      portsData.emplace_back("in" + std::to_string(inputIdx++),
                             inputs[lsqPort->getStoreDataInputIndex()],
                             "s" + std::to_string(stIdx++) + "d");
    } else {
      std::optional<MCLoadStorePort> mcPort = dyn_cast<MCLoadStorePort>(port);
      assert(mcPort && "interface port must be lsq or mc");
      // Load data
      portsData.emplace_back("in" + std::to_string(inputIdx++),
                             inputs[mcPort->getLoadDataInputIndex()], "x0d");
    }
  }

  // Add data ports after control ports
  return getIOFromPorts(portsData);
}

/// Produces the "out" attribute value of a memory interface.
static std::string getOutputForMemInterface(FuncMemoryPorts &funcPorts) {
  MemPortsData portsData;
  MemoryOpInterface memOp = funcPorts.memOp;
  ValueRange results = memOp->getResults();
  unsigned outputIdx = 1, ldIdx = 0;

  // Control signal to end
  auto addCtrlOutput = [&]() -> void {
    portsData.emplace_back("out" + std::to_string(outputIdx++),
                           memOp->getResults().back(), "e");
  };

  // Load data results, in program order
  for (GroupMemoryPorts &blockPorts : funcPorts.groups) {
    for (MemoryPort &port : blockPorts.accessPorts) {
      if (std::optional<LoadPort> loadPort = dyn_cast<LoadPort>(port)) {
        portsData.emplace_back("out" + std::to_string(outputIdx++),
                               results[loadPort->getDataOutputIndex()],
                               "l" + std::to_string(ldIdx++) + "d");
      }
    }
  }

  // For LSQs, control signal comes before interface outputs (why not?)
  if (isa<handshake::LSQOp>(memOp))
    addCtrlOutput();

  // Finally, all signals to other interfaces
  for (MemoryPort &port : funcPorts.interfacePorts) {
    if (std::optional<LSQLoadStorePort> lsqPort =
            dyn_cast<LSQLoadStorePort>(port)) {
      // Load data
      portsData.emplace_back("out" + std::to_string(outputIdx++),
                             results[lsqPort->getLoadDataOutputIndex()],
                             "l" + std::to_string(ldIdx++) + "d");
    } else {
      std::optional<MCLoadStorePort> mcPort = dyn_cast<MCLoadStorePort>(port);
      assert(mcPort && "interface port must be lsq or mc");
      // Load address, then store address, then store data
      portsData.emplace_back("out" + std::to_string(outputIdx++),
                             results[mcPort->getLoadAddrOutputIndex()], "x0a");
      portsData.emplace_back("out" + std::to_string(outputIdx++),
                             results[mcPort->getStoreAddrOutputIndex()], "y0a");
      portsData.emplace_back("out" + std::to_string(outputIdx++),
                             results[mcPort->getStoreDataOutputIndex()], "y0d");
    }
  }

  // For MCs, control signal comes at the very end
  if (isa<handshake::MemoryControllerOp>(memOp))
    addCtrlOutput();

  return getIOFromPorts(portsData);
}

/// Finds the memory interface which the provided address channel connects to.
static handshake::MemoryOpInterface
getConnectedMemInterface(Value addressToMem) {
  // Find the memory interface that the address goes to (should be the only use)
  auto users = addressToMem.getUsers();
  assert(!users.empty() && "address should have exactly one use");
  if (++users.begin() != users.end())
    assert(false && "address should have exactly one use");
  auto memOp = dyn_cast<handshake::MemoryOpInterface>(*users.begin());
  assert(memOp && "address user must be memory interface");
  return memOp;
}

/// Determines the memory port associated with the address result value of a
/// memory operation ("portId" attribute).
static unsigned findMemoryPort(Value addressToMem) {
  // Iterate over memory accesses to find the one that matches the address
  // value
  handshake::MemoryOpInterface memOp = getConnectedMemInterface(addressToMem);
  ValueRange memInputs = memOp->getOperands();
  FuncMemoryPorts ports = getMemoryPorts(memOp);
  for (GroupMemoryPorts &groupPorts : ports.groups) {
    for (auto [portIdx, port] : llvm::enumerate(groupPorts.accessPorts)) {
      if (std::optional<LoadPort> loadPort = dyn_cast<LoadPort>(port)) {
        if (memInputs[loadPort->getAddrInputIndex()] == addressToMem)
          return portIdx;
      } else {
        std::optional<StorePort> storePort = dyn_cast<StorePort>(port);
        assert(storePort && "port must be load or store");
        if (memInputs[storePort->getAddrInputIndex()] == addressToMem)
          return portIdx;
      }
    }
  }

  llvm_unreachable("can't determine memory port");
}

static size_t findIndexInRange(ValueRange range, Value val) {
  for (auto [idx, res] : llvm::enumerate(range))
    if (res == val)
      return idx;
  llvm_unreachable("value should exist in range");
}

/// Finds the position (block index and operand index) of a value in the
/// inputs of a memory interface.
static std::pair<size_t, size_t> findValueInGroups(FuncMemoryPorts &ports,
                                                   Value val) {
  unsigned numGroups = ports.getNumGroups();
  unsigned accInputIdx = 0;
  for (size_t groupIdx = 0; groupIdx < numGroups; ++groupIdx) {
    ValueRange groupInputs = ports.getGroupInputs(groupIdx);
    accInputIdx += groupInputs.size();
    for (auto [inputIdx, input] : llvm::enumerate(groupInputs)) {
      if (input == val)
        return std::make_pair(groupIdx, inputIdx);
    }
  }

  // Value must belong to a port with another memory interface, find the one
  for (auto [inputIdx, input] : llvm::enumerate(ports.getInterfacesInputs())) {
    if (input == val)
      return std::make_pair(numGroups, inputIdx + accInputIdx);
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

/// Determines whether a bitwidth modification operation (extension or
/// truncation) is located "between two blocks" (i.e., it is after a branch-like
/// operation or before a merge-like operation). Such bitwidth modifiers trip up
/// legacy Dynamatic and should thus be ignore in the DOT.
static bool isBitModBetweenBlocks(Operation *op) {
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) {
    Operation *srcOp = op->getOperand(0).getDefiningOp();
    Operation *dstOp = *op->getResult(0).getUsers().begin();
    return (srcOp && dstOp) &&
           (isa<handshake::BranchOp, handshake::ConditionalBranchOp>(srcOp) ||
            isa<handshake::MergeOp, handshake::ControlMergeOp,
                handshake::MuxOp>(dstOp));
  }
  return false;
}

/// Performs some very hacky transformations in the function to satisfy legacy
/// Dynamatic's buffer placement tool.
static void patchUpIRForLegacyBuffers(handshake::FuncOp funcOp) {
  // Remove the BB attribute of all forks "between basic blocks"
  for (Operation &forkOp : funcOp.getOps()) {
    if (!isa<handshake::ForkOp, handshake::LazyForkOp>(forkOp))
      continue;
    // Only operate on forks which belong to a basic block
    std::optional<unsigned> optForkBB = getLogicBB(&forkOp);
    if (!optForkBB.has_value())
      continue;
    unsigned forkBB = optForkBB.value();

    // Backtrack through extension operations
    Value val = forkOp.getOperand(0);
    while (Operation *defOp = val.getDefiningOp())
      if (isa<arith::ExtSIOp, arith::ExtUIOp>(defOp))
        val = defOp->getOperand(0);
      else
        break;

    // Check whether any of the result's users is a merge-like operation in a
    // different block
    auto isMergeInDiffBlock = [&](OpResult res) {
      Operation *user = *res.getUsers().begin();
      return isa<handshake::MergeLikeOpInterface>(user) &&
             getLogicBB(user).has_value() && getLogicBB(user).value() != forkBB;
    };

    if (isa_and_nonnull<handshake::ConditionalBranchOp>(val.getDefiningOp()) ||
        llvm::any_of(forkOp.getResults(), isMergeInDiffBlock))
      // Fork is located after a branch in the same block or before a merge-like
      // operation in a different block
      forkOp.removeAttr(BB_ATTR_NAME);
  }
}

/// Converts an array of unsigned numbers to a string of the following format:
/// "{array[0];array[1];...;array[size - 1]}".
static std::string arrayToString(ArrayRef<unsigned> array) {
  std::stringstream ss;
  ss << "{";
  for (unsigned num : array.drop_back())
    ss << num << ";";
  ss << array.back();
  ss << "}";
  return ss.str();
}

/// Converts a bidimensional array of unsigned numbers to a string of the
/// following format: "{biArray[0];biArray[1];...;biArray[size - 1]}" where each
/// `biArray` element is represented using `arrayToString`.
static std::string biArrayToString(ArrayRef<SmallVector<unsigned>> biArray) {
  std::stringstream ss;
  ss << "{";
  if (!biArray.empty()) {
    for (ArrayRef<unsigned> array : biArray.drop_back())
      ss << arrayToString(array) << ";";
    ss << arrayToString(biArray.back());
  }
  ss << "}";
  return ss.str();
}

LogicalResult DOTPrinter::annotateNode(Operation *op,
                                       mlir::raw_indented_ostream &os) {
  /// Set common attributes for memory interfaces
  auto setMemInterfaceAttr = [&](DOTNode &info, FuncMemoryPorts &ports,
                                 Value memref) -> void {
    info.stringAttr["in"] = getInputForMemInterface(ports);
    info.stringAttr["out"] = getOutputForMemInterface(ports);

    // Set memory name
    size_t argIdx = cast<BlockArgument>(memref).getArgNumber();
    info.stringAttr["memory"] =
        op->getParentOfType<handshake::FuncOp>().getArgName(argIdx).str();

    unsigned lsqLdSt = ports.getNumPorts<LSQLoadStorePort>();
    info.intAttr["bbcount"] = ports.getNumPorts<ControlPort>();
    info.intAttr["ldcount"] = ports.getNumPorts<LoadPort>() + lsqLdSt;
    info.intAttr["stcount"] = ports.getNumPorts<StorePort>() + lsqLdSt;
  };

  auto setLoadOpAttr = [&](DOTNode &info,
                           handshake::LoadOpInterface loadOp) -> void {
    info.stringAttr["in"] = getInputForLoadOp(loadOp);
    info.stringAttr["out"] = getOutputForLoadOp(loadOp);
    info.intAttr["portId"] = findMemoryPort(loadOp.getAddressOutput());
  };

  auto setStoreOpAttr = [&](DOTNode &info,
                            handshake::StoreOpInterface storeOp) -> void {
    info.stringAttr["in"] = getInputForStoreOp(storeOp);
    info.stringAttr["out"] = getOutputForStoreOp(storeOp);
    info.intAttr["portId"] = findMemoryPort(storeOp.getAddressOutput());
  };

  DOTNode info =
      llvm::TypeSwitch<Operation *, DOTNode>(op)
          .Case<handshake::InstanceOp>([&](auto) {
            /// NOTE: this is not actually supported, I just need the DOT
            /// printer in legacy mode with Handshake instances. People should
            /// know that the old backend doesn't support it.
            return DOTNode("Instance");
          })
          .Case<handshake::NotOp>([&](auto) {
            /// NOTE: this is not actually supported
            return DOTNode("Not");
          })
          .Case<handshake::MergeOp>([&](auto) { return DOTNode("Merge"); })
          .Case<handshake::MuxOp>([&](handshake::MuxOp op) {
            auto info = DOTNode("Mux");
            info.stringAttr["in"] = getInputForMux(op);
            return info;
          })
          .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp op) {
            auto info = DOTNode("CntrlMerge");
            info.stringAttr["out"] = getOutputForControlMerge(op);
            return info;
          })
          .Case<handshake::ConditionalBranchOp>(
              [&](handshake::ConditionalBranchOp op) {
                auto info = DOTNode("Branch");
                info.stringAttr["in"] = getInputForCondBranch(op);
                info.stringAttr["out"] = getOutputForCondBranch(op);
                return info;
              })
          .Case<handshake::OEHBOp>([&](handshake::OEHBOp oehbOp) {
            /// NOTE: Explicitly set the type to OEHB when the buffer has a
            /// single slot so that the legach backend respects our wishes
            bool singleSlot = oehbOp.getSlots() == 1;
            auto info = DOTNode(singleSlot ? "OEHB" : "Buffer");
            info.intAttr["slots"] = oehbOp.getSlots();
            if (!singleSlot)
              info.stringAttr["transparent"] = "false";
            return info;
          })
          .Case<handshake::TEHBOp>([&](handshake::TEHBOp tehbOp) {
            /// NOTE: Explicitly set the type to TEHB when the buffer has a
            /// single slot so that the legach backend respects our wishes
            bool singleSlot = tehbOp.getSlots() == 1;
            auto info = DOTNode(singleSlot ? "TEHB" : "Buffer");
            info.intAttr["slots"] = tehbOp.getSlots();
            if (!singleSlot)
              info.stringAttr["transparent"] = "true";
            return info;
          })
          .Case<handshake::MemoryControllerOp>(
              [&](handshake::MemoryControllerOp mcOp) {
                auto info = DOTNode("MC");
                MCPorts ports = mcOp.getPorts();
                setMemInterfaceAttr(info, ports, mcOp.getMemRef());
                return info;
              })
          .Case<handshake::LSQOp>([&](handshake::LSQOp lsqOp) {
            auto info = DOTNode("LSQ");
            LSQPorts ports = lsqOp.getPorts();
            setMemInterfaceAttr(info, ports, lsqOp.getMemRef());

            // Set LSQ attributes
            LSQGenerationInfo gen(lsqOp);
            info.intAttr["fifoDepth"] = gen.depth;
            info.intAttr["fifoDepth_L"] = gen.depthLoad;
            info.intAttr["fifoDepth_S"] = gen.depthStore;
            info.stringAttr["numLoads"] = arrayToString(gen.loadsPerGroup);
            info.stringAttr["numStores"] = arrayToString(gen.storesPerGroup);
            info.stringAttr["loadOffsets"] = biArrayToString(gen.loadOffsets);
            info.stringAttr["storeOffsets"] = biArrayToString(gen.storeOffsets);
            info.stringAttr["loadPorts"] = biArrayToString(gen.loadPorts);
            info.stringAttr["storePorts"] = biArrayToString(gen.storePorts);
            return info;
          })
          .Case<handshake::MCLoadOp>([&](handshake::MCLoadOp loadOp) {
            auto info = DOTNode("Operator");
            info.stringAttr["op"] = "mc_load_op";
            setLoadOpAttr(info, loadOp);
            return info;
          })
          .Case<handshake::LSQLoadOp>([&](handshake::LSQLoadOp loadOp) {
            auto info = DOTNode("Operator");
            info.stringAttr["op"] = "lsq_load_op";
            setLoadOpAttr(info, loadOp);
            return info;
          })
          .Case<handshake::MCStoreOp>([&](handshake::MCStoreOp storeOp) {
            auto info = DOTNode("Operator");
            info.stringAttr["op"] = "mc_store_op";
            setStoreOpAttr(info, storeOp);
            return info;
          })
          .Case<handshake::LSQStoreOp>([&](handshake::LSQStoreOp storeOp) {
            auto info = DOTNode("Operator");
            info.stringAttr["op"] = "lsq_store_op";
            setStoreOpAttr(info, storeOp);
            return info;
          })
          .Case<handshake::ForkOp>([&](auto) { return DOTNode("Fork"); })
          .Case<handshake::LazyForkOp>(
              [&](auto) { return DOTNode("LazyFork"); })
          .Case<handshake::SourceOp>([&](auto) {
            auto info = DOTNode("Source");
            info.stringAttr["out"] = getIOFromValues(op->getResults(), "out");
            return info;
          })
          .Case<handshake::SinkOp>([&](auto) {
            auto info = DOTNode("Sink");
            info.stringAttr["in"] = getIOFromValues(op->getOperands(), "in");
            return info;
          })
          .Case<handshake::ConstantOp>([&](handshake::ConstantOp cstOp) {
            auto info = DOTNode("Constant");

            // Convert the value to an hexadecimal string value
            std::stringstream stream;
            Type cstType = cstOp.getResult().getType();
            unsigned bitwidth = cstType.getIntOrFloatBitWidth();
            size_t hexLength =
                (bitwidth >> 2) + ((bitwidth & 0b11) != 0 ? 1 : 0);
            stream << "0x" << std::setfill('0') << std::setw(hexLength)
                   << std::hex;

            // Determine the constant value based on the constant's return type
            TypedAttr valueAttr = cstOp.getValueAttr();
            if (isa<IntegerType>(cstType)) {
              APInt value = cast<mlir::IntegerAttr>(valueAttr).getValue();
              if (cstType.isUnsignedInteger())
                stream << value.getZExtValue();
              else
                stream << value.getSExtValue();
            } else if (isa<FloatType>(cstType)) {
              mlir::FloatAttr attr = dyn_cast<mlir::FloatAttr>(valueAttr);
              stream << attr.getValue().convertToDouble();
            } else {
              return DOTNode("");
            }

            info.stringAttr["value"] = stream.str();

            // Legacy Dynamatic uses the output width of the operations also
            // as input width for some reason, make it so
            info.stringAttr["in"] = getIOFromValues(op->getResults(), "in");
            info.stringAttr["out"] = getIOFromValues(op->getResults(), "out");
            return info;
          })
          .Case<handshake::ReturnOp>([&](auto) {
            auto info = DOTNode("Operator");
            info.stringAttr["op"] = "ret_op";
            return info;
          })
          .Case<handshake::EndOp>([&](handshake::EndOp op) {
            auto info = DOTNode("Exit");
            info.stringAttr["in"] = getInputForEnd(op);

            // Output ports of end node are determined by function result
            // types
            std::stringstream stream;
            auto funcOp = op->getParentOfType<handshake::FuncOp>();
            for (auto [idx, res] : llvm::enumerate(funcOp.getResultTypes()))
              stream << "out" << (idx + 1) << ":" << getWidth(res);
            info.stringAttr["out"] = stream.str();
            return info;
          })
          .Case<arith::CmpIOp>([&](arith::CmpIOp op) {
            auto info = DOTNode("Operator");
            info.stringAttr["op"] = cmpINameToOpName[op.getPredicate()];
            return info;
          })
          .Case<arith::CmpFOp>([&](arith::CmpFOp op) {
            auto info = DOTNode("Operator");
            info.stringAttr["op"] = cmpFNameToOpName[op.getPredicate()];
            return info;
          })
          .Case<handshake::SpeculationOpInterface>([&](Operation *op) {
            auto info = DOTNode(op->getName().stripDialect().str());
            info.stringAttr["in"] = getIOFromValues(op->getOperands(), "in");
            info.stringAttr["out"] = getIOFromValues(op->getResults(), "out");
            return info;
          })
          .Default([&](auto) {
            // All our supported "mathematical" operations are stored in a map,
            // query it to see if we support this particular operation
            auto opName = compNameToOpName.find(op->getName().getStringRef());
            if (opName == compNameToOpName.end())
              return DOTNode("");
            DOTNode info("Operator");
            info.stringAttr["op"] = opName->second;

            // Among mathematical operations, only the select operation has
            // special input port logic
            if (arith::SelectOp selOp = dyn_cast<arith::SelectOp>(op)) {
              PortsData ports;
              ports.emplace_back("in1?", selOp.getCondition());
              ports.emplace_back("in2+", selOp.getTrueValue());
              ports.emplace_back("in3-", selOp.getFalseValue());
              info.stringAttr["in"] = getIOFromPorts(ports);
            }
            return info;
          });

  if (info.type.empty())
    return op->emitOpError("unsupported in legacy mode");

  assert((info.type != "Operator" ||
          info.stringAttr.find("op") != info.stringAttr.end()) &&
         "operation marked with \"Operator\" type must have an op attribute");

  // Basic block ID is 0 for out-of-blocks components, something positive
  // otherwise
  if (auto bbID = op->getAttrOfType<mlir::IntegerAttr>(BB_ATTR_NAME); bbID)
    info.intAttr["bbID"] = bbID.getValue().getZExtValue() + 1;
  else
    info.intAttr["bbID"] = 0;

  // Add default IO if not specified
  if (!isa<handshake::SourceOp, handshake::SinkOp>(op)) {
    if (info.stringAttr.find("in") == info.stringAttr.end())
      info.stringAttr["in"] = getIOFromValues(op->getOperands(), "in");
    if (info.stringAttr.find("out") == info.stringAttr.end())
      info.stringAttr["out"] = getIOFromValues(op->getResults(), "out");
  }

  // Add default latency for operators if not specified
  info.stringAttr["delay"] = getNodeDelayAttr(op);
  if (info.type == "Operator")
    info.stringAttr["latency"] = getNodeLatencyAttr(op);

  // II is 1 for all operators
  if (info.type == "Operator")
    info.intAttr["II"] = 1;

  if (experimental::speculation::isSpeculative(op, true))
    info.intAttr["speculative"] = 1;

  info.print(os);
  return success();
}

LogicalResult DOTPrinter::annotateArgumentNode(handshake::FuncOp funcOp,
                                               size_t idx,
                                               mlir::raw_indented_ostream &os) {
  BlockArgument arg = funcOp.getArgument(idx);
  DOTNode info("Entry");
  info.stringAttr["in"] = getIOFromValues(ValueRange(arg), "in");
  info.stringAttr["out"] = getIOFromValues(ValueRange(arg), "out");
  info.intAttr["bbID"] = 1;
  if (isa<NoneType>(arg.getType()))
    info.stringAttr["control"] = "true";

  info.print(os);
  return success();
}

LogicalResult DOTPrinter::annotateEdge(OpOperand &oprd,
                                       mlir::raw_indented_ostream &os) {
  Value val = oprd.get();
  Operation *src = val.getDefiningOp();
  Operation *dst = oprd.getOwner();

  bool legacyBuffers = mode == Mode::LEGACY_BUFFERS;
  // In legacy-buffers mode, skip edges from branch-like operations to bitwidth
  // modifiers in between blocks
  if (legacyBuffers && isBitModBetweenBlocks(dst))
    return success();

  DOTEdge info;

  // "Jump over" bitwidth modification operations that go to a merge-like
  // operation in a different block
  Value srcVal = val;
  if (legacyBuffers && isBitModBetweenBlocks(src)) {
    srcVal = src->getOperand(0);
    src = srcVal.getDefiningOp();
  }

  // Locate value in source results and destination operands
  unsigned resIdx = findIndexInRange(src->getResults(), srcVal);
  unsigned argIdx = findIndexInRange(dst->getOperands(), val);

  // Handle to and from attributes (with special cases). Also add 1 to each
  // index since first ports are called in1/out1
  info.from = fixOutputPortNumber(src, resIdx) + 1;
  info.to = fixInputPortNumber(dst, argIdx) + 1;

  // Handle the mem_address optional attribute
  if (isa<handshake::MemoryOpInterface>(src)) {
    if (isa<handshake::LoadOpInterface, handshake::StoreOpInterface>(dst)) {
      info.memAddress = false;
    } else if (LSQOp lsqOp = dyn_cast<LSQOp>(src);
               lsqOp && isa<MemoryControllerOp>(dst)) {
      MCLoadStorePort mcPorts = lsqOp.getPorts().getMCPort();
      ValueRange lsqResults = lsqOp.getResults();
      info.memAddress = lsqResults[mcPorts.getLoadAddrOutputIndex()] == val ||
                        lsqResults[mcPorts.getStoreAddrOutputIndex()] == val;
    }
  } else if (isa<handshake::MemoryOpInterface>(dst)) {
    if (isa<handshake::LoadOpInterface, handshake::StoreOpInterface>(src))
      // Is val the address result of the memory operation?
      info.memAddress = val == src->getResult(0);
  }

  info.print(os);
  return success();
}

LogicalResult DOTPrinter::annotateArgumentEdge(handshake::FuncOp funcOp,
                                               size_t idx, Operation *dst,
                                               mlir::raw_indented_ostream &os) {
  BlockArgument arg = funcOp.getArgument(idx);
  DOTEdge info;

  // Locate value in destination operands
  auto argIdx = findIndexInRange(dst->getOperands(), arg);

  // Handle to and from attributes (with special cases). Also add 1 to each
  // index since first ports are called in1/out1
  info.from = 1;
  info.to = fixInputPortNumber(dst, argIdx) + 1;

  auto argName = getArgumentName(funcOp, idx);
  info.print(os);
  return success();
}

std::string DOTPrinter::getNodeDelayAttr(Operation *op) {
  const TimingModel *model = timingDB->getModel(op);
  if (!model)
    return "0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000";

  double dataDelay;
  if (failed(model->dataDelay.getCeilMetric(op, dataDelay)))
    dataDelay = 0.0;

  std::stringstream stream;
  stream << std::fixed << std::setprecision(3) << dataDelay << " "
         << model->validDelay << " " << model->readyDelay << " "
         << model->validToReady << " " << model->condToValid << " "
         << model->condToReady << " " << model->validToCond << " "
         << model->validToData;
  return stream.str();
}

std::string DOTPrinter::getNodeLatencyAttr(Operation *op) {
  double latency;
  if (failed(timingDB->getLatency(op, SignalType::DATA, latency)))
    return "0";
  return std::to_string(static_cast<unsigned>(latency));
}

// ============================================================================
// Printing
// ============================================================================

static constexpr StringLiteral DOTTED("dotted"), SOLID("solid"), DOT("dot"),
    NORMAL("normal");

/// Determines the "arrowhead" attribute of the edge corresponding to the
/// operand.
static StringRef getArrowheadStyle(OpOperand &oprd) {
  Value val = oprd.get();
  Operation *ownerOp = oprd.getOwner();
  if (auto muxOp = dyn_cast<handshake::MuxOp>(ownerOp))
    return val == muxOp.getSelectOperand() ? DOT : NORMAL;
  if (auto condBrOp = dyn_cast<handshake::ConditionalBranchOp>(ownerOp))
    return val == condBrOp.getConditionOperand() ? DOT : NORMAL;
  return NORMAL;
}

/// Determines cosmetic attributes of the edge corresponding to the operand.
static std::string getEdgeStyle(OpOperand &oprd) {
  std::string attributes;
  StringRef style = isa<NoneType>(oprd.get().getType()) ? DOTTED : SOLID;
  // StringRef arrowhead =
  return "style=\"" + style.str() +
         R"(", dir="both", arrowtail="none", arrowhead=")" +
         getArrowheadStyle(oprd).str() + "\", ";
}

/// Returns the name of the function argument that corresponds to the memref
/// operand of a memory interface. Returns an empty reference if the memref
/// cannot be found in the arguments.
static StringRef getMemName(Value memref) {
  Operation *parentOp = memref.getParentBlock()->getParentOp();
  handshake::FuncOp funcOp = dyn_cast<handshake::FuncOp>(parentOp);
  for (auto [name, funArg] :
       llvm::zip(funcOp.getArgNames(), funcOp.getArguments())) {
    if (funArg == memref)
      return cast<StringAttr>(name).getValue();
  }
  return StringRef();
}

/// Returns the name of the function argument that corresponds to the memory
/// interface that the memory port operation connects to.
template <typename Op>
static StringRef getMemNameForPort(Op memPortOp) {
  Value addrRes = memPortOp.getAddressOutput();
  auto addrUsers = addrRes.getUsers();
  if (addrUsers.empty())
    return "";
  Operation *userOp = *addrUsers.begin();
  while (isa_and_present<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
                         handshake::ForkOp>(userOp)) {
    auto users = userOp->getResult(0).getUsers();
    if (users.empty())
      return "";
    userOp = *users.begin();
  }
  // We should have reached a memory interface
  if (handshake::LSQOp lsqOp = dyn_cast<handshake::LSQOp>(userOp))
    return getMemName(lsqOp.getMemRef());
  if (handshake::MemoryControllerOp mcOp =
          dyn_cast<handshake::MemoryControllerOp>(userOp))
    return getMemName(mcOp.getMemRef());
  llvm_unreachable("cannot reach memory interface");
}

/// Returns the pretty-field version of a label fro a memory-related operation.
static inline std::string getMemLabel(StringRef baseName, StringRef memName) {
  return (baseName + (memName.empty() ? "" : " (" + memName.str() + ")")).str();
}

/// Returns the pretty-fied version of the DOT node's label corresponding  to
/// the operation.
static std::string getPrettyNodeLabel(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::string>(op)
      // handshake operations
      .Case<handshake::ConstantOp>(
          [&](handshake::ConstantOp cstOp) -> std::string {
            Type cstType = cstOp.getResult().getType();
            TypedAttr valueAttr = cstOp.getValueAttr();
            if (isa<IntegerType>(cstType)) {
              // Special case boolean attribute (which would result in an i1
              // constant integer results) to print true/false instead of 1/0
              if (auto boolAttr = dyn_cast<mlir::BoolAttr>(valueAttr))
                return boolAttr.getValue() ? "true" : "false";

              APInt value = cast<mlir::IntegerAttr>(valueAttr).getValue();
              if (cstType.isUnsignedInteger())
                return std::to_string(value.getZExtValue());
              return std::to_string(value.getSExtValue());
            }
            if (isa<FloatType>(cstType)) {
              mlir::FloatAttr attr = dyn_cast<mlir::FloatAttr>(valueAttr);
              return std::to_string(attr.getValue().convertToDouble());
            }
            // Fallback on an empty string
            return std::string("");
          })
      .Case<handshake::OEHBOp>([&](handshake::OEHBOp oehbOp) {
        return "oehb [" + std::to_string(oehbOp.getSlots()) + "]";
      })
      .Case<handshake::TEHBOp>([&](handshake::TEHBOp tehbOp) {
        return "tehb [" + std::to_string(tehbOp.getSlots()) + "]";
      })
      .Case<handshake::MemoryControllerOp>([&](MemoryControllerOp mcOp) {
        return getMemLabel("MC", getMemName(mcOp.getMemRef()));
      })
      .Case<handshake::LSQOp>([&](handshake::LSQOp lsqOp) {
        return getMemLabel("LSQ", getMemName(lsqOp.getMemRef()));
      })
      .Case<handshake::MCLoadOp, handshake::LSQLoadOp>([&](auto) {
        StringRef memName = getMemNameForPort(dyn_cast<LoadOpInterface>(op));
        return getMemLabel("LD", memName);
      })
      .Case<handshake::MCStoreOp, handshake::LSQStoreOp>([&](auto) {
        StringRef memName = getMemNameForPort(dyn_cast<StoreOpInterface>(op));
        return getMemLabel("ST", memName);
      })
      .Case<handshake::ControlMergeOp>([&](auto) { return "cmerge"; })
      .Case<handshake::BranchOp>([&](auto) { return "branch"; })
      .Case<handshake::ConditionalBranchOp>([&](auto) { return "cbranch"; })
      .Case<handshake::ReturnOp>([&](auto) { return "return"; })
      // arith operations
      .Case<arith::AddIOp, arith::AddFOp>([&](auto) { return "+"; })
      .Case<arith::SubIOp, arith::SubFOp>([&](auto) { return "-"; })
      .Case<arith::AndIOp>([&](auto) { return "&"; })
      .Case<arith::OrIOp>([&](auto) { return "|"; })
      .Case<arith::XOrIOp>([&](auto) { return "^"; })
      .Case<arith::MulIOp, arith::MulFOp>([&](auto) { return "*"; })
      .Case<arith::DivUIOp, arith::DivSIOp, arith::DivFOp>(
          [&](auto) { return "div"; })
      .Case<arith::ShRSIOp, arith::ShRUIOp>([&](auto) { return ">>"; })
      .Case<arith::ShLIOp>([&](auto) { return "<<"; })
      .Case<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, arith::TruncIOp,
            arith::TruncFOp>([&](auto) {
        unsigned opWidth = op->getOperand(0).getType().getIntOrFloatBitWidth();
        unsigned resWidth = op->getResult(0).getType().getIntOrFloatBitWidth();
        return "[" + std::to_string(opWidth) + "..." +
               std::to_string(resWidth) + "]";
      })
      .Case<arith::CmpIOp>([&](arith::CmpIOp op) {
        switch (op.getPredicate()) {
        case arith::CmpIPredicate::eq:
          return "==";
        case arith::CmpIPredicate::ne:
          return "!=";
        case arith::CmpIPredicate::uge:
        case arith::CmpIPredicate::sge:
          return ">=";
        case arith::CmpIPredicate::ugt:
        case arith::CmpIPredicate::sgt:
          return ">";
        case arith::CmpIPredicate::ule:
        case arith::CmpIPredicate::sle:
          return "<=";
        case arith::CmpIPredicate::ult:
        case arith::CmpIPredicate::slt:
          return "<";
        }
      })
      .Case<arith::CmpFOp>([&](arith::CmpFOp op) {
        switch (op.getPredicate()) {
        case arith::CmpFPredicate::OEQ:
        case arith::CmpFPredicate::UEQ:
          return "==";
        case arith::CmpFPredicate::ONE:
        case arith::CmpFPredicate::UNE:
          return "!=";
        case arith::CmpFPredicate::OGE:
        case arith::CmpFPredicate::UGE:
          return ">=";
        case arith::CmpFPredicate::OGT:
        case arith::CmpFPredicate::UGT:
          return ">";
        case arith::CmpFPredicate::OLE:
        case arith::CmpFPredicate::ULE:
          return "<=";
        case arith::CmpFPredicate::OLT:
        case arith::CmpFPredicate::ULT:
          return "<";
        case arith::CmpFPredicate::ORD:
          return "ordered?";
        case arith::CmpFPredicate::UNO:
          return "unordered?";
        case arith::CmpFPredicate::AlwaysFalse:
          return "false";
        case arith::CmpFPredicate::AlwaysTrue:
          return "true";
        }
      })
      .Default([&](auto) {
        StringRef dialect = op->getDialect()->getNamespace();
        std::string label = op->getName().getStringRef().str();
        label.erase(0, dialect.size() + 1);
        return label;
      });
}

static StringRef getNodeColor(Operation *op) {
  return llvm::TypeSwitch<Operation *, StringRef>(op)
      .Case<handshake::ForkOp, handshake::LazyForkOp, handshake::JoinOp>(
          [&](auto) { return "lavender"; })
      .Case<handshake::BufferOpInterface>([&](auto) { return "lightgreen"; })
      .Case<handshake::ReturnOp, handshake::EndOp>([&](auto) { return "gold"; })
      .Case<handshake::SourceOp, handshake::SinkOp>(
          [&](auto) { return "gainsboro"; })
      .Case<handshake::ConstantOp>([&](auto) { return "plum"; })
      .Case<handshake::MemoryOpInterface, handshake::LoadOpInterface,
            handshake::StoreOpInterface>([&](auto) { return "coral"; })
      .Case<handshake::MergeOp, handshake::ControlMergeOp, handshake::MuxOp>(
          [&](auto) { return "lightblue"; })
      .Case<handshake::BranchOp, handshake::ConditionalBranchOp>(
          [&](auto) { return "tan2"; })
      .Case<handshake::SpeculationOpInterface>([&](auto) { return "salmon"; })
      .Default([&](auto) { return "moccasin"; });
}

DOTPrinter::DOTPrinter(Mode mode, EdgeStyle edgeStyle, TimingDatabase *timingDB)
    : mode(mode), edgeStyle(edgeStyle), timingDB(timingDB) {
  assert(!inLegacyMode() ||
         timingDB && "timing database must exist in legacy mode");
};

LogicalResult DOTPrinter::print(mlir::ModuleOp mod,
                                mlir::raw_indented_ostream *os) {
  // We support at most one function per module
  auto funcs = mod.getOps<handshake::FuncOp>();
  if (funcs.empty())
    return success();

  // We only support one function per module
  handshake::FuncOp funcOp = nullptr;
  for (auto op : mod.getOps<handshake::FuncOp>()) {
    if (op.isExternal())
      continue;
    if (funcOp) {
      return mod->emitOpError() << "we currently only support one non-external "
                                   "handshake function per module";
    }
    funcOp = op;
  }

  // Name all operations in the IR
  NameAnalysis nameAnalysis = NameAnalysis(mod);
  if (!nameAnalysis.isAnalysisValid())
    return failure();
  nameAnalysis.nameAllUnnamedOps();

  if (inLegacyMode()) {
    // In legacy mode, the IR must respect certain additional constraints for it
    // to be compatible with legacy Dynamatic
    if (failed(verifyIRMaterialized(funcOp)))
      return funcOp.emitOpError() << ERR_NON_MATERIALIZED_FUNC;
    if (failed(verifyAllIndexConcretized(funcOp)))
      return funcOp.emitOpError()
             << "In legacy mode, all index types in the IR must be concretized "
                "to ensure that the DOT is compatible with legacy Dynamatic. "
             << ERR_RUN_CONCRETIZATION;

    if (mode == Mode::LEGACY_BUFFERS)
      patchUpIRForLegacyBuffers(funcOp);
  }

  // Print the graph
  if (os)
    return printFunc(funcOp, *os);
  mlir::raw_indented_ostream stdOs(llvm::outs());
  return printFunc(funcOp, stdOs);
}

std::string DOTPrinter::getArgumentName(handshake::FuncOp funcOp, size_t idx) {
  auto numArgs = funcOp.getNumArguments();
  assert(idx < numArgs && "argument index too high");
  if (idx == numArgs - 1 && inLegacyMode())
    // Legacy Dynamatic expects the start signal to be called start_0
    return "start_0";
  return funcOp.getArgName(idx).getValue().str();
}

void DOTPrinter::openSubgraph(std::string &name, std::string &label,
                              mlir::raw_indented_ostream &os) {
  os << "subgraph \"" << name << "\" {\n";
  os.indent();
  os << "label=\"" << label << "\"\n";
}

void DOTPrinter::closeSubgraph(mlir::raw_indented_ostream &os) {
  os.unindent();
  os << "}\n";
}

LogicalResult DOTPrinter::printNode(Operation *op,
                                    mlir::raw_indented_ostream &os) {
  // The node's DOT name
  std::string opName = getUniqueName(op).str();

  // The node's DOT "mlir_op" attribute
  std::string mlirOpName = op->getName().getStringRef().str();
  std::string prettyLabel = getPrettyNodeLabel(op);
  if (isa<arith::CmpIOp, arith::CmpFOp>(op))
    mlirOpName += prettyLabel;

  // The node's DOT "shape" attribute
  StringRef dialect = op->getDialect()->getNamespace();
  StringRef shape = dialect == "handshake" ? "box" : "oval";

  // The node's DOT "label" attribute
  StringRef label = inLegacyMode() ? opName : prettyLabel;

  // The node's DOT "style" attribute
  std::string style = "filled";
  if (auto controlInterface = dyn_cast<handshake::ControlInterface>(op)) {
    if (controlInterface.isControl())
      style += ", " + DOTTED.str();
  }

  // Write the node
  os << "\"" << opName << "\""
     << " [mlir_op=\"" << mlirOpName << "\", label=\"" << label
     << "\", fillcolor=" << getNodeColor(op) << ", shape=\"" << shape
     << "\", style=\"" << style << "\", ";
  if (inLegacyMode() && failed(annotateNode(op, os)))
    return failure();
  os << "]\n";
  return success();
}

LogicalResult DOTPrinter::printEdge(OpOperand &oprd,
                                    mlir::raw_indented_ostream &os) {
  Value val = oprd.get();
  Operation *src = val.getDefiningOp();
  Operation *dst = oprd.getOwner();

  bool legacyBuffers = mode == Mode::LEGACY_BUFFERS;
  // In legacy-buffers mode, skip edges from branch-like operations to bitwidth
  // modifiers in between blocks
  if (legacyBuffers && isBitModBetweenBlocks(dst))
    return success();

  // "Jump over" bitwidth modification operations that go to a merge-like
  // operation in a different block
  bool legacy = inLegacyMode();
  std::string srcNodeName =
      legacyBuffers && isBitModBetweenBlocks(src)
          ? getUniqueName(src->getOperand(0).getDefiningOp()).str()
          : getUniqueName(src).str();
  std::string dstNodeName = getUniqueName(dst).str();

  os << "\"" << srcNodeName << "\" -> \"" << dstNodeName << "\" ["
     << getEdgeStyle(oprd);
  if (legacy && failed(annotateEdge(oprd, os)))
    return failure();
  if (isBackedge(val, dst))
    os << (legacy ? ", " : "") << " color=\"blue\"";
  // Print speculative edge attribute
  if (experimental::speculation::isSpeculative(oprd, true))
    os << ((legacy || isBackedge(val, dst)) ? ", " : "") << " speculative=1";
  os << "]\n";
  return success();
}

LogicalResult DOTPrinter::printFunc(handshake::FuncOp funcOp,
                                    mlir::raw_indented_ostream &os) {
  bool legacy = inLegacyMode();

  std::string splines;
  if (edgeStyle == EdgeStyle::SPLINE)
    splines = "spline";
  else
    splines = "ortho";

  os << "Digraph G {\n";
  os.indent();
  os << "splines=" << splines << ";\n";
  os << "compound=true; // Allow edges between clusters\n";

  /// Prints all nodes corresponding to function arguments.
  auto printArgNodes = [&]() -> LogicalResult {
    os << "// Units from function arguments\n";
    for (const auto &arg : enumerate(funcOp.getArguments())) {
      if (isa<MemRefType>(arg.value().getType()))
        // Arguments with memref types are represented by memory interfaces
        // inside the function so they are not displayed
        continue;

      std::string argLabel = getArgumentName(funcOp, arg.index());
      StringRef style = isa<NoneType>(arg.value().getType()) ? DOTTED : SOLID;
      os << "\"" << argLabel
         << R"(" [mlir_op="handshake.func", shape=diamond, )"
         << "label=\"" << argLabel << "\", style=\"" << style << "\", ";
      if (legacy && failed(annotateArgumentNode(funcOp, arg.index(), os)))
        return failure();
      os << "]\n";
    }
    return success();
  };

  /// Prints all edges incoming from function arguments.
  auto printArgEdges = [&]() -> LogicalResult {
    os << "// Channels from function arguments\n";
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (isa<MemRefType>(arg.getType()))
        continue;

      for (OpOperand &oprd : arg.getUses()) {
        Operation *ownerOp = oprd.getOwner();
        std::string argLabel = getArgumentName(funcOp, idx);
        os << "\"" << argLabel << "\" -> \"" << getUniqueName(ownerOp) << "\" ["
           << getEdgeStyle(oprd);
        if (legacy && failed(annotateArgumentEdge(funcOp, idx, ownerOp, os)))
          return failure();
        os << "]\n";
      }
    }
    return success();
  };

  // Get function's "blocks". These leverage the "bb" attributes attached to
  // operations in handshake functions to display operations belonging to the
  // same original basic block together
  LogicBBs blocks = getLogicBBs(funcOp);

  // Whether nodes and edges corresponding to function arguments have already
  // been handled
  bool areArgsHandled = false;

  // Collect all edges that do not connect two nodes in the same block
  llvm::MapVector<unsigned, std::vector<OpOperand *>> outgoingEdges;

  // We print the function "block-by-block" by grouping nodes in the same block
  // (as well as edges between nodes of the same block) within DOT clusters
  for (auto &[blockID, blockOps] : blocks.blocks) {
    areArgsHandled |= blockID == ENTRY_BB;

    // Open the subgraph
    os << "// Units/Channels in BB " << blockID << "\n";
    std::string graphName = "cluster" + std::to_string(blockID);
    std::string graphLabel = "block" + std::to_string(blockID);
    openSubgraph(graphName, graphLabel, os);

    // For entry block, also add all nodes corresponding to function arguments
    if (blockID == ENTRY_BB && failed(printArgNodes()))
      return failure();

    os << "// Units in BB " << blockID << "\n";
    for (Operation *op : blockOps) {
      // In legacy-buffers mode, do not print bitwidth modification operation
      // between branch-like and merge-like operations
      if (mode == Mode::LEGACY_BUFFERS && isBitModBetweenBlocks(op))
        continue;
      if (failed(printNode(op, os)))
        return failure();
    }

    // For entry block, also add all edges incoming from function arguments
    if (blockID == ENTRY_BB && failed(printArgEdges()))
      return failure();

    os << "// Channels in BB " << blockID << "\n";
    for (Operation *op : blockOps) {
      for (OpResult res : op->getResults()) {
        for (OpOperand &oprd : res.getUses()) {
          Operation *userOp = oprd.getOwner();
          std::optional<unsigned> bb = getLogicBB(userOp);
          if (bb && *bb == blockID) {
            if (failed(printEdge(oprd, os)))
              return failure();
          } else {
            outgoingEdges[blockID].push_back(&oprd);
          }
        }
      }
    }

    // Close the subgraph
    closeSubgraph(os);
  }

  // Print edges coming from function arguments if they haven't been so far
  if (!areArgsHandled && failed(printArgNodes()))
    return failure();

  os << "// Units outside of all basic blocks\n";
  for (Operation *op : blocks.outOfBlocks) {
    // In legacy-buffers mode, do not print bitwidth modification operation
    // between branch-like and merge-like operations
    if (mode == Mode::LEGACY_BUFFERS && isBitModBetweenBlocks(op))
      continue;
    if (failed(printNode(op, os)))
      return failure();
  }

  // Print outgoing edges for each block
  for (auto &[blockID, blockEdges] : outgoingEdges) {
    os << "// Channels outgoing of BB " << blockID << "\n";
    for (OpOperand *oprd : blockEdges) {
      if (failed(printEdge(*oprd, os)))
        return failure();
    }
  }

  // Print edges coming from function arguments if they haven't been so far
  if (!areArgsHandled && failed(printArgEdges()))
    return failure();

  os << "// Channels outside of all basic blocks\n";
  for (Operation *op : blocks.outOfBlocks) {
    for (OpResult res : op->getResults()) {
      for (OpOperand &oprd : res.getUses()) {
        if (failed(printEdge(oprd, os)))
          return failure();
      }
    }
  }
  os.unindent();
  os << "}\n";
  return success();
}

void DOTNode::print(mlir::raw_indented_ostream &os) {
  // Print type
  os << "type=\"" << type << "\"";
  if (!stringAttr.empty() || !intAttr.empty())
    os << ", ";

  // Print all attributes
  for (auto [idx, attr] : llvm::enumerate(stringAttr)) {
    auto &[name, value] = attr;
    os << name << "=\"" << value << "\"";
    if (idx != stringAttr.size() - 1)
      os << ", ";
  }
  if (!intAttr.empty())
    os << ", ";
  for (auto [idx, attr] : llvm::enumerate(intAttr)) {
    auto &[name, value] = attr;
    os << name << "=" << value;
    if (idx != intAttr.size() - 1)
      os << ", ";
  }
}

void DOTEdge::print(mlir::raw_indented_ostream &os) {
  os << "from=\"out" << from << "\", to=\"in" << to << "\"";
  if (memAddress.has_value())
    os << ", mem_address=\"" << (memAddress.value() ? "true" : "false") << "\"";
}
