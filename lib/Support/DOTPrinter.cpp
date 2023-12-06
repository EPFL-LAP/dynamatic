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
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Conversion/PassDetails.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <iomanip>
#include <string>
#include <utility>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

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

/// Maps name of arithmetic operation to "op" attribute.
static std::unordered_map<std::string, std::string> arithNameToOpName{
    {"arith.addi", "add_op"},      {"arith.addf", "fadd_op"},
    {"arith.subi", "sub_op"},      {"arith.subf", "fsub_op"},
    {"arith.andi", "and_op"},      {"arith.ori", "or_op"},
    {"arith.xori", "xor_op"},      {"arith.muli", "mul_op"},
    {"arith.mulf", "fmul_op"},     {"arith.divui", "udiv_op"},
    {"arith.divsi", "sdiv_op"},    {"arith.divf", "fdiv_op"},
    {"arith.sitofp", "sitofp_op"}, {"arith.remsi", "urem_op"},
    {"arith.extsi", "sext_op"},    {"arith.extui", "zext_op"},
    {"arith.trunci", "trunc_op"},  {"arith.shrsi", "ashr_op"},
    {"arith.shli", "shl_op"},      {"arith.select", "select_op"}};

/// Maps name of integer comparison type to "op" attribute.
static std::unordered_map<arith::CmpIPredicate, std::string> cmpINameToOpName{
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
static std::unordered_map<arith::CmpFPredicate, std::string> cmpFNameToOpName{
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

/// Produces the "in" attribute value of a handshake::SelectOp.
static std::string getInputForSelect(arith::SelectOp op) {
  PortsData ports;
  ports.emplace_back("in1?", op.getCondition());
  ports.emplace_back("in2+", op.getTrueValue());
  ports.emplace_back("in3-", op.getFalseValue());
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
  ValueRange inputs = memOp.getMemOperands();

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
  ValueRange results = memOp.getMemResults();
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
  ValueRange memInputs = memOp.getMemOperands();
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
  for (auto forkOp : funcOp.getOps<handshake::ForkOp>()) {
    // Only operate on forks which belong to a basic block
    std::optional<unsigned> optForkBB = getLogicBB(forkOp);
    if (!optForkBB.has_value())
      continue;
    unsigned forkBB = optForkBB.value();

    // Backtrack through extension operations
    Value val = forkOp.getOperand();
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
        llvm::any_of(forkOp->getResults(), isMergeInDiffBlock))
      // Fork is located after a branch in the same block or before a merge-like
      // operation in a different block
      forkOp->removeAttr(BB_ATTR);
  }
}

/// Converts an array of unsigned numbers to a string of the following format:
/// "{array[0];array[1];...;array[size - 1];0;0;...;0}". The "0"w are generated
/// dynamically to reach `length` elements based on size of the array; the
/// latter of which cannot exceed the length.
static std::string arrayToString(ArrayRef<unsigned> array, unsigned length) {
  std::stringstream ss;
  assert(array.size() <= length && "vector too large");
  ss << "{";
  if (!array.empty()) {
    for (unsigned num : array.drop_back())
      ss << num << ";";
    ss << array.back();
    for (size_t i = array.size(); i < length; ++i)
      ss << ";0";
  } else {
    for (size_t i = 0; i < length - 1; ++i)
      ss << "0;";
    ss << "0";
  }
  ss << "}";
  return ss.str();
}

/// Converts a bidimensional array of unsigned numbers to a string of the
/// following format: "{biArray[0];biArray[1];...;biArray[size - 1]}" where each
/// `biArray` element is represented using `arrayToString` with the provided
/// length.
static std::string biArrayToString(ArrayRef<SmallVector<unsigned>> biArray,
                                   unsigned length) {
  std::stringstream ss;
  ss << "{";
  if (!biArray.empty()) {
    for (ArrayRef<unsigned> array : biArray.drop_back())
      ss << arrayToString(array, length) << ";";
    ss << arrayToString(biArray.back(), length);
  }
  ss << "}";
  return ss.str();
}

LogicalResult DOTPrinter::annotateNode(Operation *op,
                                       mlir::raw_indented_ostream &os) {
  /// Set common attributes for memory interfaces
  auto setMemInterfaceAttr = [&](NodeInfo &info, FuncMemoryPorts &ports,
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

  auto setLoadOpAttr = [&](NodeInfo &info,
                           handshake::LoadOpInterface loadOp) -> void {
    info.stringAttr["in"] = getInputForLoadOp(loadOp);
    info.stringAttr["out"] = getOutputForLoadOp(loadOp);
    info.intAttr["portId"] = findMemoryPort(loadOp.getAddressOutput());
  };

  auto setStoreOpAttr = [&](NodeInfo &info,
                            handshake::StoreOpInterface storeOp) -> void {
    info.stringAttr["in"] = getInputForStoreOp(storeOp);
    info.stringAttr["out"] = getOutputForStoreOp(storeOp);
    info.intAttr["portId"] = findMemoryPort(storeOp.getAddressOutput());
  };

  NodeInfo info =
      llvm::TypeSwitch<Operation *, NodeInfo>(op)
          .Case<handshake::MergeOp>([&](auto) { return NodeInfo("Merge"); })
          .Case<handshake::MuxOp>([&](handshake::MuxOp op) {
            auto info = NodeInfo("Mux");
            info.stringAttr["in"] = getInputForMux(op);
            return info;
          })
          .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp op) {
            auto info = NodeInfo("CntrlMerge");
            info.stringAttr["out"] = getOutputForControlMerge(op);
            return info;
          })
          .Case<handshake::ConditionalBranchOp>(
              [&](handshake::ConditionalBranchOp op) {
                auto info = NodeInfo("Branch");
                info.stringAttr["in"] = getInputForCondBranch(op);
                info.stringAttr["out"] = getOutputForCondBranch(op);
                return info;
              })
          .Case<handshake::BufferOp>([&](handshake::BufferOp bufOp) {
            auto info = NodeInfo("Buffer");
            info.intAttr["slots"] = bufOp.getNumSlots();
            info.stringAttr["transparent"] =
                bufOp.getBufferType() == BufferTypeEnum::fifo ? "true"
                                                              : "false";
            return info;
          })
          .Case<handshake::MemoryControllerOp>(
              [&](handshake::MemoryControllerOp mcOp) {
                auto info = NodeInfo("MC");
                MCPorts ports = mcOp.getPorts();
                setMemInterfaceAttr(info, ports, mcOp.getMemRef());
                return info;
              })
          .Case<handshake::LSQOp>([&](handshake::LSQOp lsqOp) {
            auto info = NodeInfo("LSQ");
            LSQPorts ports = lsqOp.getPorts();
            setMemInterfaceAttr(info, ports, lsqOp.getMemRef());
            unsigned depth = 16;
            info.intAttr["fifoDepth"] = depth;

            // Create port information for the LSQ generator

            // Number of load and store ports per block
            SmallVector<unsigned> numLoads, numStores;

            // Offset and (block-relative) port indices for loads and stores.
            // Note that the offsets are semantically undimensional vectors (one
            // logical value per block); however, in legacy DOTs they are stored
            // as bi-dimensional arrays therefore we use the same data-structure
            // here
            SmallVector<SmallVector<unsigned>> loadOffsets, storeOffsets,
                loadPorts, storePorts;

            unsigned loadIdx = 0, storeIdx = 0;
            for (GroupMemoryPorts &blockPorts : ports.groups) {
              // Number of load and store ports per block
              numLoads.push_back(blockPorts.getNumPorts<LoadPort>());
              numStores.push_back(blockPorts.getNumPorts<StorePort>());

              // Offsets of first load/store in the block and indices of each
              // load/store port
              std::optional<unsigned> firstLoadOffset, firstStoreOffset;
              SmallVector<unsigned> blockLoadPorts, blockStorePorts;
              for (auto [portIdx, accessPort] :
                   llvm::enumerate(blockPorts.accessPorts)) {
                if (isa<LoadPort>(accessPort)) {
                  if (!firstLoadOffset)
                    firstLoadOffset = portIdx;
                  blockLoadPorts.push_back(loadIdx++);
                } else {
                  // This is a StorePort
                  assert(isa<StorePort>(accessPort) &&
                         "access port must be load or store");
                  if (!firstStoreOffset)
                    firstStoreOffset = portIdx;
                  blockStorePorts.push_back(storeIdx++);
                }
              }

              // If there are no loads or no stores in the block, set the
              // corresponding offset to 0
              loadOffsets.push_back(
                  SmallVector<unsigned>{firstLoadOffset.value_or(0)});
              storeOffsets.push_back(
                  SmallVector<unsigned>{firstStoreOffset.value_or(0)});

              loadPorts.push_back(blockLoadPorts);
              storePorts.push_back(blockStorePorts);
            }

            // Set LSQ attributes
            info.stringAttr["numLoads"] =
                arrayToString(numLoads, numLoads.size());
            info.stringAttr["numStores"] =
                arrayToString(numStores, numStores.size());
            info.stringAttr["loadOffsets"] =
                biArrayToString(loadOffsets, depth);
            info.stringAttr["storeOffsets"] =
                biArrayToString(storeOffsets, depth);
            info.stringAttr["loadPorts"] = biArrayToString(loadPorts, depth);
            info.stringAttr["storePorts"] = biArrayToString(storePorts, depth);

            return info;
          })
          .Case<handshake::MCLoadOp>([&](handshake::MCLoadOp loadOp) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = "mc_load_op";
            setLoadOpAttr(info, loadOp);
            return info;
          })
          .Case<handshake::LSQLoadOp>([&](handshake::LSQLoadOp loadOp) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = "lsq_load_op";
            setLoadOpAttr(info, loadOp);
            return info;
          })
          .Case<handshake::MCStoreOp>([&](handshake::MCStoreOp storeOp) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = "mc_store_op";
            setStoreOpAttr(info, storeOp);
            return info;
          })
          .Case<handshake::LSQStoreOp>([&](handshake::LSQStoreOp storeOp) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = "lsq_store_op";
            setStoreOpAttr(info, storeOp);
            return info;
          })
          .Case<handshake::ForkOp>([&](auto) { return NodeInfo("Fork"); })
          .Case<handshake::SourceOp>([&](auto) {
            auto info = NodeInfo("Source");
            info.stringAttr["out"] = getIOFromValues(op->getResults(), "out");
            return info;
          })
          .Case<handshake::SinkOp>([&](auto) {
            auto info = NodeInfo("Sink");
            info.stringAttr["in"] = getIOFromValues(op->getOperands(), "in");
            return info;
          })
          .Case<handshake::ConstantOp>([&](handshake::ConstantOp cstOp) {
            auto info = NodeInfo("Constant");

            // Determine the constant value and its bitwidth based on the
            // vondtnat's value attribute
            long int value = 0;
            unsigned bitwidth = 0;
            TypedAttr valueAttr = cstOp.getValueAttr();
            if (auto intAttr = dyn_cast<mlir::IntegerAttr>(valueAttr)) {
              value = intAttr.getValue().getSExtValue();
              bitwidth = intAttr.getValue().getBitWidth();
            } else if (auto boolAttr = dyn_cast<mlir::BoolAttr>(valueAttr)) {
              value = boolAttr.getValue() ? 1 : 0;
              bitwidth = 1;
            } else {
              llvm_unreachable("unsupported constant type");
            }

            // Convert the value to hexadecimal format
            std::stringstream stream;
            int hexLength = (bitwidth >> 2) + ((bitwidth & 0b11) != 0 ? 1 : 0);
            stream << "0x" << std::setfill('0') << std::setw(hexLength)
                   << std::hex << value;
            info.stringAttr["value"] = stream.str();

            // Legacy Dynamatic uses the output width of the operations also
            // as input width for some reason, make it so
            info.stringAttr["in"] = getIOFromValues(op->getResults(), "in");
            info.stringAttr["out"] = getIOFromValues(op->getResults(), "out");
            return info;
          })
          .Case<handshake::DynamaticReturnOp>([&](auto) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = "ret_op";
            return info;
          })
          .Case<handshake::EndOp>([&](handshake::EndOp op) {
            auto info = NodeInfo("Exit");
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
          .Case<arith::SelectOp>([&](arith::SelectOp op) {
            auto info = NodeInfo("Operator");
            auto opName = op->getName().getStringRef().str();
            info.stringAttr["op"] = arithNameToOpName[opName];
            info.stringAttr["in"] = getInputForSelect(op);
            return info;
          })
          .Case<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp,
                arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::MulIOp,
                arith::MulFOp, arith::DivUIOp, arith::DivSIOp, arith::DivFOp,
                arith::SIToFPOp, arith::RemSIOp, arith::ExtSIOp, arith::ExtUIOp,
                arith::TruncIOp, arith::ShRSIOp, arith::ShLIOp>([&](auto) {
            auto info = NodeInfo("Operator");
            auto opName = op->getName().getStringRef().str();
            info.stringAttr["op"] = arithNameToOpName[opName];
            return info;
          })
          .Case<arith::CmpIOp>([&](arith::CmpIOp op) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = cmpINameToOpName[op.getPredicate()];
            return info;
          })
          .Case<arith::CmpFOp>([&](arith::CmpFOp op) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = cmpFNameToOpName[op.getPredicate()];
            return info;
          })
          .Case<arith::IndexCastOp>([&](auto) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = "zext_op";
            return info;
          })
          .Default([&](auto) { return NodeInfo(""); });

  if (info.type.empty())
    return op->emitOpError("unsupported in legacy mode");

  assert((info.type != "Operator" ||
          info.stringAttr.find("op") != info.stringAttr.end()) &&
         "operation marked with \"Operator\" type must have an op attribute");

  // Basic block ID is 0 for out-of-blocks components, something positive
  // otherwise
  if (auto bbID = op->getAttrOfType<mlir::IntegerAttr>(BB_ATTR); bbID)
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

  info.print(os);
  return success();
}

LogicalResult DOTPrinter::annotateArgumentNode(handshake::FuncOp funcOp,
                                               size_t idx,
                                               mlir::raw_indented_ostream &os) {
  BlockArgument arg = funcOp.getArgument(idx);
  NodeInfo info("Entry");
  info.stringAttr["in"] = getIOFromValues(ValueRange(arg), "in");
  info.stringAttr["out"] = getIOFromValues(ValueRange(arg), "out");
  info.intAttr["bbID"] = 1;
  if (isa<NoneType>(arg.getType()))
    info.stringAttr["control"] = "true";

  info.print(os);
  return success();
}

LogicalResult DOTPrinter::annotateEdge(Operation *src, Operation *dst,
                                       Value val,
                                       mlir::raw_indented_ostream &os) {
  bool legacyBuffers = mode == Mode::LEGACY_BUFFERS;
  // In legacy-buffers mode, skip edges from branch-like operations to bitwidth
  // modifiers in between blocks
  if (legacyBuffers && isBitModBetweenBlocks(dst))
    return success();

  EdgeInfo info;

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
      ValueRange lsqOutputs = lsqOp.getMemResults();
      info.memAddress = lsqOutputs[mcPorts.getLoadAddrOutputIndex()] == val ||
                        lsqOutputs[mcPorts.getStoreAddrOutputIndex()] == val;
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
  EdgeInfo info;

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

/// Style attribute value for control nodes/edges.
static const std::string CONTROL_STYLE = "dashed";

/// Determines the style attribute of a value.
static std::string getStyleOfValue(Value result) {
  return isa<NoneType>(result.getType()) ? "style=" + CONTROL_STYLE + ", " : "";
}

/// Determines the pretty-printed version of a node label.
static std::string getPrettyPrintedNodeLabel(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::string>(op)
      // handshake operations
      .Case<handshake::ConstantOp>([&](auto op) {
        // Try to get the constant value as a boolean
        if (mlir::BoolAttr boolAttr =
                op->template getAttrOfType<mlir::BoolAttr>("value"))
          return std::to_string(boolAttr.getValue());

        // Try to get the constant value as an integer
        if (mlir::IntegerAttr intAttr =
                op->template getAttrOfType<mlir::IntegerAttr>("value")) {
          Type inType = intAttr.getType();
          if (!isa<IndexType>(inType) && inType.getIntOrFloatBitWidth() == 0)
            return std::string("null");
          APInt ap = intAttr.getValue();
          return ap.isNegative() ? std::to_string(ap.getSExtValue())
                                 : std::to_string(ap.getZExtValue());
        }

        // Try to get the constant value as floating point
        if (mlir::FloatAttr floatAttr =
                op->template getAttrOfType<mlir::FloatAttr>("value"))
          return std::to_string(floatAttr.getValue().convertToFloat());

        // Fallback on a generic string
        return std::string("constant");
      })
      .Case<handshake::ControlMergeOp>([&](auto) { return "cmerge"; })
      .Case<handshake::ConditionalBranchOp>([&](auto) { return "cbranch"; })
      .Case<handshake::BufferOp>([&](handshake::BufferOp bufOp) {
        return stringifyEnum(bufOp.getBufferType()).str() + " [" +
               std::to_string(bufOp.getNumSlots()) + "]";
      })
      .Case<handshake::BranchOp>([&](auto) { return "branch"; })
      // handshake operations (dynamatic)
      .Case<handshake::MCLoadOp>([&](auto) { return "mc_load"; })
      .Case<handshake::MCStoreOp>([&](auto) { return "mc_store"; })
      .Case<handshake::LSQLoadOp>([&](auto) { return "lsq_load"; })
      .Case<handshake::LSQStoreOp>([&](auto) { return "lsq_store"; })
      .Case<handshake::MemoryControllerOp>([&](auto) { return "MC"; })
      .Case<handshake::LSQOp>([&](auto) { return "LSQ"; })
      .Case<handshake::DynamaticReturnOp>([&](auto) { return "return"; })
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
      .Case<arith::SelectOp>([&](auto) { return "select"; })
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
        llvm_unreachable("unhandled cmpi predicate");
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
        llvm_unreachable("unhandled cmpf predicate");
      })
      .Default([&](auto op) {
        auto opDialect = op->getDialect()->getNamespace();
        std::string label = op->getName().getStringRef().str();
        if (opDialect == "handshake")
          label.erase(0, StringLiteral("handshake.").size());

        return label;
      });
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
  if (++funcs.begin() != funcs.end()) {
    mod->emitOpError()
        << "we currently only support one handshake function per module";
    return failure();
  }

  // Name all operations in the IR
  NameAnalysis nameAnalysis = NameAnalysis(mod);
  if (!nameAnalysis.isAnalysisValid())
    return failure();
  nameAnalysis.nameAllUnnamedOps();

  handshake::FuncOp funcOp = *funcs.begin();

  if (inLegacyMode()) {
    // In legacy mode, the IR must respect certain additional constraints for it
    // to be compatible with legacy Dynamatic
    if (failed(verifyAllValuesHasOneUse(funcOp)))
      return funcOp.emitOpError()
             << "In legacy mode, all values in the IR must have exactly one "
                "use to ensure that the DOT is compatible with legacy "
                "Dynamatic. Run the --handshake-materialize-forks-sinks pass "
                "before to insert forks and sinks in the IR and make every "
                "value used exactly once.";
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

  std::string prettyLabel = getPrettyPrintedNodeLabel(op);
  std::string canonicalName =
      op->getName().getStringRef().str() +
      (isa<arith::CmpIOp, arith::CmpFOp>(op) ? prettyLabel : "");

  // Print node name
  std::string opName = getUniqueName(op);
  if (inLegacyMode()) {
    // LSQ must be capitalized in legacy modes for dot2vhdl to recognize it
    if (size_t idx = opName.find("lsq"); idx != std::string::npos)
      opName = "LSQ" + opName.substr(3);
  }
  os << "\"" << opName << "\""
     << " [mlir_op=\"" << canonicalName << "\", ";

  // Determine fill color
  os << "fillcolor=";
  os << llvm::TypeSwitch<Operation *, std::string>(op)
            .Case<handshake::ForkOp, handshake::LazyForkOp, handshake::JoinOp>(
                [&](auto) { return "lavender"; })
            .Case<handshake::BufferOp>([&](auto) { return "lightgreen"; })
            .Case<handshake::DynamaticReturnOp, handshake::EndOp>(
                [&](auto) { return "gold"; })
            .Case<handshake::SourceOp, handshake::SinkOp>(
                [&](auto) { return "gainsboro"; })
            .Case<handshake::ConstantOp>([&](auto) { return "plum"; })
            .Case<handshake::MemoryOpInterface, handshake::LoadOpInterface,
                  handshake::StoreOpInterface>([&](auto) { return "coral"; })
            .Case<handshake::MergeOp, handshake::ControlMergeOp,
                  handshake::MuxOp>([&](auto) { return "lightblue"; })
            .Case<handshake::BranchOp, handshake::ConditionalBranchOp>(
                [&](auto) { return "tan2"; })
            .Default([&](auto) { return "moccasin"; });

  // Determine shape
  os << ", shape=";
  if (op->getDialect()->getNamespace() == "handshake")
    os << "box";
  else
    os << "oval";

  // Determine label
  os << ", label=\"" << (inLegacyMode() ? opName : prettyLabel) << "\"";

  // Determine style
  os << ", style=\"filled";
  if (auto controlInterface = dyn_cast<handshake::ControlInterface>(op);
      controlInterface && controlInterface.isControl())
    os << ", " + CONTROL_STYLE;
  os << "\", ";
  if (inLegacyMode() && failed(annotateNode(op, os)))
    return failure();
  os << "]\n";

  return success();
}

LogicalResult DOTPrinter::printEdge(Operation *src, Operation *dst, Value val,
                                    mlir::raw_indented_ostream &os) {
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
          ? getUniqueName(src->getOperand(0).getDefiningOp())
          : getUniqueName(src);
  std::string dstNodeName = getUniqueName(dst);
  if (inLegacyMode()) {
    // LSQ must be capitalized in legacy modes for dot2vhdl to recognize it
    if (size_t idx = srcNodeName.find("lsq"); idx != std::string::npos)
      srcNodeName = "LSQ" + srcNodeName.substr(3);
    if (size_t idx = dstNodeName.find("lsq"); idx != std::string::npos)
      dstNodeName = "LSQ" + dstNodeName.substr(3);
  }

  os << "\"" << srcNodeName << "\" -> \"" << dstNodeName << "\" ["
     << getStyleOfValue(val);
  if (legacy && failed(annotateEdge(src, dst, val, os)))
    return failure();
  if (isBackedge(val, dst))
    os << (legacy ? ", " : "") << " color=\"blue\"";
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

  // Print nodes corresponding to function arguments
  os << "// Function arguments\n";
  for (const auto &arg : enumerate(funcOp.getArguments())) {
    if (isa<MemRefType>(arg.value().getType()))
      // Arguments with memref types are represented by memory interfaces
      // inside the function so they are not displayed
      continue;

    auto argLabel = getArgumentName(funcOp, arg.index());
    os << "\"" << argLabel << R"(" [mlir_op="handshake.arg", shape=diamond, )"
       << getStyleOfValue(arg.value()) << "label=\"" << argLabel << "\", ";
    if (legacy && failed(annotateArgumentNode(funcOp, arg.index(), os)))
      return failure();
    os << "]\n";
  }

  // Print nodes corresponding to function operations
  os << "// Function operations\n";
  for (auto &op : funcOp.getOps()) {
    // In legacy-buffers mode, do not print bitwidth modification operation
    // between branch-like and merge-like operations
    if (mode == Mode::LEGACY_BUFFERS && isBitModBetweenBlocks(&op))
      continue;

    // Print the operation
    if (failed(printNode(&op, os)))
      return failure();
  }

  // Get function's "blocks". These leverage the "bb" attributes attached to
  // operations in handshake functions to display operations belonging to the
  // same original basic block together
  auto handshakeBlocks = getLogicBBs(funcOp);

  // Print all edges incoming from operations in a block
  bool argEdgesAdded = false;
  auto addArgEdges = [&]() -> LogicalResult {
    argEdgesAdded = true;
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments()))
      if (!isa<MemRefType>(arg.getType()))
        for (Operation *user : arg.getUsers()) {
          auto argLabel = getArgumentName(funcOp, idx);
          os << "\"" << argLabel << "\" -> \"" << getUniqueName(user) << "\" ["
             << getStyleOfValue(arg);
          if (legacy && failed(annotateArgumentEdge(funcOp, idx, user, os)))
            return failure();
          os << "]\n";
        }
    return success();
  };

  for (auto &[blockID, ops] : handshakeBlocks.blocks) {

    // For each block, we create a subgraph to contain all edges between two
    // operations of that block
    auto blockStrID = std::to_string(blockID);
    os << "// Edges within basic block " << blockStrID << "\n";
    std::string graphName = "cluster" + blockStrID;
    std::string graphLabel = "block" + blockStrID;
    openSubgraph(graphName, graphLabel, os);

    // Collect all edges leaving the block and print them after the subgraph
    std::vector<std::tuple<Operation *, Operation *, OpResult>> outgoingEdges;

    // Determines whether an edge to a destination operation should be inside
    // of the source operation's basic block
    auto isEdgeInSubgraph = [](Operation *useOp, unsigned currentBB) {
      // Sink operations are always displayed outside of blocks
      if (isa<handshake::SinkOp>(useOp))
        return false;

      auto bb = useOp->getAttrOfType<mlir::IntegerAttr>(BB_ATTR);
      return bb && bb.getValue().getZExtValue() == currentBB;
    };

    // Iterate over all uses of all results of all operations inside the
    // block
    for (auto *op : ops) {
      for (auto res : op->getResults())
        for (auto &use : res.getUses()) {
          // Add edge to subgraph or outgoing edges depending on the block of
          // the operation using the result
          Operation *useOp = use.getOwner();
          if (isEdgeInSubgraph(useOp, blockID)) {
            if (failed(printEdge(op, useOp, res, os)))
              return failure();
          } else {
            outgoingEdges.emplace_back(op, useOp, res);
          }
        }
    }

    // For entry block, also add all edges incoming from function arguments
    if (blockID == 0 && failed(addArgEdges()))
      return failure();

    // Close the subgraph
    closeSubgraph(os);

    // Print outgoing edges for this block
    if (!outgoingEdges.empty())
      os << "// Edges outgoing of basic block " << blockStrID << "\n";
    for (auto &[op, useOp, res] : outgoingEdges)
      if (failed(printEdge(op, useOp, res, os)))
        return failure();
  }

  // Print all edges incoming from operations not belonging to any block
  // outside of all subgraphs
  os << "// Edges outside of all basic blocks\n";
  // Print edges coming from function arguments if they haven't been so far
  if (!argEdgesAdded && failed(addArgEdges()))
    return failure();
  for (auto *op : handshakeBlocks.outOfBlocks)
    for (auto res : op->getResults())
      for (auto &use : res.getUses())
        if (failed(printEdge(op, use.getOwner(), res, os)))
          return failure();

  os.unindent();
  os << "}\n";

  return success();
}

void NodeInfo::print(mlir::raw_indented_ostream &os) {
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

void EdgeInfo::print(mlir::raw_indented_ostream &os) {
  os << "from=\"out" << from << "\", to=\"in" << to << "\"";
  if (memAddress.has_value())
    os << ", mem_address=\"" << (memAddress.value() ? "true" : "false") << "\"";
}
