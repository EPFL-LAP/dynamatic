//===- DOTPrinter.cpp - Print DOT to standard output ------------*- C++ -*-===//
//
// This file contains the implementation of the DOT printer. The printer
// produces, on standard output, Graphviz-formatted output which contains the
// graph representation of the input handshake-level IR.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/DOTPrinter.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Conversion/PassDetails.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/HandshakeConcretizeIndexType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iomanip>
#include <string>

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
    ports.push_back(std::make_pair(portType + std::to_string(idx + 1), val));
  return getIOFromPorts(ports);
}

/// Produces the "in" attribute value of a handshake::MuxOp.
static std::string getInputForMux(handshake::MuxOp op) {
  PortsData ports;
  ports.push_back(std::make_pair("in1?", op.getSelectOperand()));
  for (auto [idx, val] : llvm::enumerate(op->getOperands().drop_front(1)))
    ports.push_back(std::make_pair("in" + std::to_string(idx + 2), val));
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
  ports.push_back(std::make_pair("in1", op.getDataOperand()));
  ports.push_back(std::make_pair("in2?", op.getConditionOperand()));
  return getIOFromPorts(ports);
}

/// Produces the "out" attribute value of a handshake::ConditionalBranchOp.
static std::string getOutputForCondBranch(handshake::ConditionalBranchOp op) {
  PortsData ports;
  ports.push_back(std::make_pair("out1+", op.getTrueResult()));
  ports.push_back(std::make_pair("out2-", op.getFalseResult()));
  return getIOFromPorts(ports);
}

/// Produces the "in" attribute value of a handshake::SelectOp.
static std::string getInputForSelect(arith::SelectOp op) {
  PortsData ports;
  ports.push_back(std::make_pair("in1?", op.getCondition()));
  ports.push_back(std::make_pair("in2+", op.getTrueValue()));
  ports.push_back(std::make_pair("in3-", op.getFalseValue()));
  return getIOFromPorts(ports);
}

/// Produces the "in" attribute value of a handshake::EndOp.
static std::string getInputForEnd(handshake::EndOp op) {
  MemPortsData ports;
  unsigned idx = 1;
  for (auto val : op.getMemoryControls())
    ports.push_back(std::make_tuple("in" + std::to_string(idx++), val, "e"));
  for (auto val : op.getReturnValues())
    ports.push_back(std::make_tuple("in" + std::to_string(idx++), val, ""));
  return getIOFromPorts(ports);
}

/// Produces the "in" attribute value of a handshake::DynamaticLoadOp.
static std::string getInputForLoadOp(handshake::DynamaticLoadOp op) {
  PortsData ports;
  ports.push_back(std::make_pair("in1", op.getData()));
  ports.push_back(std::make_pair("in2", op.getAddress()));
  return getIOFromPorts(ports);
}

/// Produces the "out" attribute value of a handshake::DynamaticLoadOp.
static std::string getOutputForLoadOp(handshake::DynamaticLoadOp op) {
  PortsData ports;
  ports.push_back(std::make_pair("out1", op.getDataResult()));
  ports.push_back(std::make_pair("out2", op.getAddressResult()));
  return getIOFromPorts(ports);
}

/// Produces the "in" attribute value of a handshake::DynamaticStoreOp.
static std::string getInputForStoreOp(handshake::DynamaticStoreOp op) {
  PortsData ports;
  ports.push_back(std::make_pair("in1", op.getData()));
  ports.push_back(std::make_pair("in2", op.getAddress()));
  return getIOFromPorts(ports);
}

/// Produces the "out" attribute value of a handshake::DynamaticStoreOp.
static std::string getOutputForStoreOp(handshake::DynamaticStoreOp op) {
  PortsData ports;
  ports.push_back(std::make_pair("out1", op.getDataResult()));
  ports.push_back(std::make_pair("out2", op.getAddressResult()));
  return getIOFromPorts(ports);
}

/// Produces the "in" attribute value of a handshake::MemoryControllerOp.
static std::string getInputForMC(handshake::MemoryControllerOp op) {
  MemPortsData allPorts, dataPorts;
  unsigned ctrlIdx = 0, ldIdx = 0, stIdx = 0, inputIdx = 1;
  size_t operandIdx = 0;
  ValueRange inputs = op.getInputs();

  // Add all control signals first
  for (auto [idx, blockAccesses] : llvm::enumerate(op.getAccesses()))
    if (op.bbHasControl(idx))
      allPorts.push_back(std::make_tuple("in" + std::to_string(inputIdx++),
                                         inputs[operandIdx++],
                                         "c" + std::to_string(ctrlIdx++)));

  // Then all memory access signals
  for (auto [idx, blockAccesses] : llvm::enumerate(op.getAccesses())) {
    // Add loads and stores, in program order
    for (auto &access : cast<mlir::ArrayAttr>(blockAccesses))
      if (cast<AccessTypeEnumAttr>(access).getValue() == AccessTypeEnum::Load)
        dataPorts.push_back(std::make_tuple(
            "in" + std::to_string(inputIdx++), inputs[operandIdx++],
            "l" + std::to_string(ldIdx++) + "a"));
      else {
        // Address signal first, then data signal
        dataPorts.push_back(std::make_tuple("in" + std::to_string(inputIdx++),
                                            inputs[operandIdx++],
                                            "s" + std::to_string(stIdx) + "a"));
        dataPorts.push_back(std::make_tuple(
            "in" + std::to_string(inputIdx++), inputs[operandIdx++],
            "s" + std::to_string(stIdx++) + "d"));
      }
  }

  // Add data ports after control ports
  allPorts.insert(allPorts.end(), dataPorts.begin(), dataPorts.end());
  return getIOFromPorts(allPorts);
}

/// Produces the "out" attribute value of a handshake::MemoryControllerOp.
static std::string getOutputForMC(handshake::MemoryControllerOp op) {
  MemPortsData ports;
  for (auto [idx, res] : llvm::enumerate(op->getResults().drop_back(1)))
    ports.push_back((std::make_tuple("out" + std::to_string(idx + 1), res,
                                     "l" + std::to_string(idx) + "d")));
  ports.push_back((std::make_tuple("out" + std::to_string(op.getNumResults()),
                                   op->getResults().back(), "e")));
  return getIOFromPorts(ports);
}

/// Determines the memory port associated with the address result value of a
/// memory operation ("portId" attribute).
static unsigned findMemoryPort(Value addressToMem) {
  // Find the memory interface that the address goes to (should be the only use)
  auto users = addressToMem.getUsers();
  assert(!users.empty() && "address should have exactly one use");
  if (++users.begin() != users.end())
    assert(false && "address should have exactly one use");
  auto memOp = dyn_cast<handshake::MemoryControllerOp>(*users.begin());
  assert(memOp && "address user must be MemoryControllerOp");

  // Iterate over memory accesses to find the one that matches the address
  // value
  size_t inputIdx = 0;
  auto memInputs = memOp.getInputs();
  auto accesses = memOp.getAccesses();
  for (auto [idx, bbAccesses] : llvm::enumerate(accesses)) {
    if (memOp.bbHasControl(idx))
      // Skip over the control value
      inputIdx++;

    for (auto [portIdx, access] :
         llvm::enumerate(cast<mlir::ArrayAttr>(bbAccesses))) {
      // Check whether this is our port
      if (memInputs[inputIdx] == addressToMem)
        return portIdx;

      // Go to the index corresponding to the next memory operation in the
      // block
      if (cast<AccessTypeEnumAttr>(access).getValue() == AccessTypeEnum::Load)
        inputIdx++;
      else
        inputIdx += 2;
    }
  }

  assert(false && "can't determine memory port");
  return 0;
}

static size_t findIndexInRange(ValueRange range, Value val) {
  for (auto [idx, res] : llvm::enumerate(range))
    if (res == val)
      return idx;
  assert(false && "value should exist in range");
  return 0;
}

/// Finds the position (group index and operand index) of a value in the
/// inputs of a memory interface.
static std::pair<size_t, size_t>
findValueInGroups(SmallVector<SmallVector<Value>> &groups, Value val) {
  for (auto [groupIdx, bbOperands] : llvm::enumerate(groups))
    for (auto [opIdx, operand] : llvm::enumerate(bbOperands))
      if (val == operand)
        return std::make_pair(groupIdx, opIdx);
  assert(false && "value should be an operand to the memory interface");
  return std::make_pair(0, 0);
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
            auto groups = memOp.groupInputsByBB();

            // Determine total number of control operands
            unsigned ctrlCount = 0;
            for (size_t i = 0, e = groups.size(); i < e; i++)
              if (memOp.bbHasControl(i))
                ctrlCount++;

            // Figure out where the value lies
            auto [groupIdx, opIdx] = findValueInGroups(groups, val);

            // Figure out at which index the value would be in legacy
            // Dynamatic's interface
            bool valGroupHasControl = memOp.bbHasControl(groupIdx);
            if (opIdx == 0 && valGroupHasControl) {
              // Value is a control input
              size_t fixedIdx = 0;
              for (size_t i = 0; i < groupIdx; i++)
                if (memOp.bbHasControl(i))
                  fixedIdx++;
              return fixedIdx;
            }

            // Value is a data input
            size_t fixedIdx = ctrlCount;
            for (size_t i = 0; i < groupIdx; i++)
              // Add number of data inputs corresponding to the block
              if (memOp.bbHasControl(i))
                fixedIdx += groups[i].size() - 1;
              else
                fixedIdx += groups[i].size();

            // Add index offset in the group the value belongs to
            if (valGroupHasControl)
              fixedIdx += opIdx - 1;
            else
              fixedIdx += opIdx;
            return fixedIdx;
          })
      .Default([&](auto) { return idx; });
}

/// Derives a raw port from a port string in the format
/// "<port_name>:<port_width>". Uses the name passed as argument to derive a
/// globally unique name for the returned raw port.
static RawPort splitPortStr(std::string &portStr, std::string &nodeName) {
  auto colonIdx = portStr.find(":");
  assert(colonIdx != std::string::npos && "port string has incorrect format");

  // Take out last special character from the port name if present
  auto portName = portStr.substr(0, colonIdx);
  auto lastChar = portName[colonIdx - 1];
  if (lastChar == '?' || lastChar == '+' || lastChar == '-')
    portName = portName.substr(0, colonIdx - 1);

  auto width = (unsigned)std::stoul(portStr.substr(colonIdx + 1));
  return std::make_pair(nodeName + "_" + portName, width);
}

/// Extracts a list of raw ports from the "in" or "out" attribute of a node in
/// the graph. Each returned raw port is uniquely named globally using the
/// name passed as argument.
static SmallVector<RawPort> extractPortsFromString(std::string &portsInfo,
                                                   std::string &nodeName) {
  SmallVector<RawPort> ports;
  if (portsInfo.empty())
    return ports;

  size_t last = 0, next = 0;
  while ((next = portsInfo.find(" ", last)) != std::string::npos) {
    std::string subPorts = portsInfo.substr(last, next - last);
    ports.push_back(splitPortStr(subPorts, nodeName));
    last = next + 1;
  }
  std::string subPorts = portsInfo.substr(last);
  ports.push_back(splitPortStr(subPorts, nodeName));
  return ports;
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
/// Dynamatic's toolchain.
static void patchUpIRForLegacy(handshake::FuncOp funcOp) {
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

LogicalResult DOTPrinter::legacyRegisterPorts(NodeInfo &info,
                                              std::string &nodeName,
                                              bool skipInputs,
                                              bool skipOutputs) {
  if (!skipInputs && info.stringAttr.find("in") != info.stringAttr.end())
    for (auto &in : extractPortsFromString(info.stringAttr["in"], nodeName))
      if (auto [_, newPort] = legacyPorts.insert(in); !newPort)
        return failure();
  if (!skipOutputs && info.stringAttr.find("out") != info.stringAttr.end())
    for (auto &out : extractPortsFromString(info.stringAttr["out"], nodeName))
      if (auto [_, newPort] = legacyPorts.insert(out); !newPort)
        return failure();
  return success();
}

LogicalResult DOTPrinter::legacyRegisterChannel(EdgeInfo &info,
                                                std::string &srcName,
                                                std::string &dstName) {
  auto srcPort = srcName + "_out" + std::to_string(info.from);
  auto dstPort = dstName + "_in" + std::to_string(info.to);
  if (auto [_, newChannel] =
          legacyChannels.insert(std::make_pair(srcPort, dstPort));
      !newChannel)
    return failure();
  return success();
}

LogicalResult DOTPrinter::verifyDOT(handshake::FuncOp funcOp,
                                    bool failOnWidthMismatch) {

  // Create a set of all port names to keep track of the ones that haven't
  // been matched so far
  std::set<std::string> unmatchedPorts;
  for (auto &[name, _] : legacyPorts)
    unmatchedPorts.insert(name);

  // Iterate over channels and check for correctness
  for (auto &channel : legacyChannels) {
    // Both ports must exist
    auto srcPortIt = unmatchedPorts.find(channel.first);
    if (srcPortIt == unmatchedPorts.end())
      return funcOp->emitError()
             << "port " << channel.first
             << " is referenced by channel but does not exist\n";
    auto dstPortIt = unmatchedPorts.find(channel.second);
    if (dstPortIt == unmatchedPorts.end())
      return funcOp->emitError()
             << "port " << channel.second
             << " is referenced by channel but does not exist\n";

    auto srcPort = *srcPortIt, dstPort = *dstPortIt;

    // Port widths must match
    if (failOnWidthMismatch)
      if (legacyPorts[srcPort] != legacyPorts[dstPort])
        return funcOp->emitError()
               << "port widths do not match between " << srcPort << " and "
               << dstPort << " (" << legacyPorts[srcPort]
               << " != " << legacyPorts[dstPort] << ")\n";

    // Remove ports from set of unmatched ports
    unmatchedPorts.erase(srcPort);
    unmatchedPorts.erase(dstPort);
  }

  for (auto &name : unmatchedPorts)
    return funcOp.emitError()
           << "port " << name << " isn't wired to any other port\n";

  return success();
}

LogicalResult DOTPrinter::annotateNode(Operation *op) {
  auto info =
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
              [&](handshake::MemoryControllerOp memOp) {
                auto info = NodeInfo("MC");
                info.stringAttr["in"] = getInputForMC(memOp);
                info.stringAttr["out"] = getOutputForMC(memOp);
                info.stringAttr["memory"] =
                    "mem" + std::to_string(memOp.getId());

                // Compute the number of basic blocks with a control signal to
                // the MC
                unsigned numControls = 0;
                for (size_t i = 0, e = memOp.getBBCount(); i < e; ++i)
                  if (memOp.bbHasControl(i))
                    ++numControls;

                info.intAttr["bbcount"] = numControls;
                info.intAttr["ldcount"] = memOp.getLdCount();
                info.intAttr["stcount"] = memOp.getStCount();
                return info;
              })
          .Case<handshake::DynamaticLoadOp>([&](handshake::DynamaticLoadOp op) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = "mc_load_op";
            info.stringAttr["in"] = getInputForLoadOp(op);
            info.stringAttr["out"] = getOutputForLoadOp(op);
            info.intAttr["portId"] = findMemoryPort(op.getAddressResult());
            return info;
          })
          .Case<handshake::DynamaticStoreOp>(
              [&](handshake::DynamaticStoreOp op) {
                auto info = NodeInfo("Operator");
                info.stringAttr["op"] = "mc_store_op";
                info.stringAttr["in"] = getInputForStoreOp(op);
                info.stringAttr["out"] = getOutputForStoreOp(op);
                info.intAttr["portId"] = findMemoryPort(op.getAddressResult());
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
          .Case<handshake::ConstantOp>([&](auto) {
            auto info = NodeInfo("Constant");
            // Try to get the constant value as an integer
            int value = 0;
            int length = 8;
            if (mlir::IntegerAttr intAttr =
                    op->template getAttrOfType<mlir::IntegerAttr>("value");
                intAttr)
              value = intAttr.getValue().getSExtValue();
            // Try to get the constant value as an integer
            if (mlir::BoolAttr boolAttr =
                    op->template getAttrOfType<mlir::BoolAttr>("value");
                boolAttr && boolAttr.getValue()) {
              value = 1;
              length = 1;
            }

            // Convert the value to hexadecimal format
            std::stringstream stream;
            stream << "0x" << std::setfill('0') << std::setw(length) << std::hex
                   << value;

            // Legacy Dynamatic uses the output width of the operations also
            // as input width for some reason, make it so
            info.stringAttr["in"] = getIOFromValues(op->getResults(), "in");
            info.stringAttr["out"] = getIOFromValues(op->getResults(), "out");

            info.stringAttr["value"] = stream.str();
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

  bool skipOutputs = isa<handshake::EndOp>(op);
  if (failed(legacyRegisterPorts(info, opNameMap[op], false, skipOutputs)))
    return op->emitError()
           << "failed to register node due to duplicated port name";
  info.print(os);
  return success();
}

LogicalResult DOTPrinter::annotateArgumentNode(handshake::FuncOp funcOp,
                                               size_t idx) {
  BlockArgument arg = funcOp.getArgument(idx);
  NodeInfo info("Entry");
  info.stringAttr["in"] = getIOFromValues(ValueRange(arg), "in");
  info.stringAttr["out"] = getIOFromValues(ValueRange(arg), "out");
  info.intAttr["bbID"] = 1;
  if (isa<NoneType>(arg.getType()))
    info.stringAttr["control"] = "true";

  auto argName = getArgumentName(funcOp, idx);
  if (failed(legacyRegisterPorts(info, argName, true, false)))
    return funcOp.emitError() << "failed to register argument node " << idx
                              << " due to duplicated port name";
  info.print(os);
  return success();
}

LogicalResult DOTPrinter::annotateEdge(Operation *src, Operation *dst,
                                       Value val) {
  // In legacy mode, skip edges from branch-like operations to bitwidth
  // modifiers in between blocks
  if (legacy && isBitModBetweenBlocks(dst))
    return success();

  EdgeInfo info;

  // "Jump over" bitwidth modification operations that go to a merge-like
  // operation in a different block
  Value srcVal = val;
  if (legacy && isBitModBetweenBlocks(src)) {
    srcVal = src->getOperand(0);
    src = srcVal.getDefiningOp();
  }

  // Locate value in source results and destination operands
  auto resIdx = findIndexInRange(src->getResults(), srcVal);
  auto argIdx = findIndexInRange(dst->getOperands(), val);

  // Handle to and from attributes (with special cases). Also add 1 to each
  // index since first ports are called in1/out1
  info.from = fixPortNumber(src, srcVal, resIdx, true) + 1;
  info.to = fixPortNumber(dst, val, argIdx, false) + 1;

  // Handle the mem_address optional attribute
  if (auto srcMem = dyn_cast<handshake::MemoryControllerOp>(src); srcMem) {
    if (isa<handshake::DynamaticLoadOp, handshake::DynamaticStoreOp>(dst))
      info.memAddress = false;
  } else if (auto dstMem = dyn_cast<handshake::MemoryControllerOp>(dst); dstMem)
    if (isa<handshake::DynamaticLoadOp, handshake::DynamaticStoreOp>(src))
      // Is val the address result of the memory operation?
      info.memAddress = val == src->getResult(0);

  if (failed(legacyRegisterChannel(info, opNameMap[src], opNameMap[dst])))
    return src->emitError()
           << "failed to register channel to destination node "
           << opNameMap[dst] << " due to duplicated channel name";
  info.print(os);
  return success();
}

LogicalResult DOTPrinter::annotateArgumentEdge(handshake::FuncOp funcOp,
                                               size_t idx, Operation *dst) {
  BlockArgument arg = funcOp.getArgument(idx);
  EdgeInfo info;

  // Locate value in destination operands
  auto argIdx = findIndexInRange(dst->getOperands(), arg);

  // Handle to and from attributes (with special cases). Also add 1 to each
  // index since first ports are called in1/out1
  info.from = 1;
  info.to = fixPortNumber(dst, arg, argIdx, false) + 1;

  auto argName = getArgumentName(funcOp, idx);
  if (failed(legacyRegisterChannel(info, argName, opNameMap[dst])))
    return funcOp.emitError()
           << "failed to register channel from argument node " << idx << " to "
           << opNameMap[dst] << " due to duplicated channel name";
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
  if (failed(timingDB->getLatency(op, latency)))
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
        // Try to get the constant value as an integer
        if (mlir::BoolAttr boolAttr =
                op->template getAttrOfType<mlir::BoolAttr>("value");
            boolAttr)
          return std::to_string(boolAttr.getValue());

        // Try to get the constant value as an integer
        if (mlir::IntegerAttr intAttr =
                op->template getAttrOfType<mlir::IntegerAttr>("value");
            intAttr)
          return std::to_string(intAttr.getValue().getSExtValue());

        // Try to get the constant value as floating point
        if (mlir::FloatAttr floatAttr =
                op->template getAttrOfType<mlir::FloatAttr>("value");
            floatAttr)
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
      .Case<handshake::DynamaticLoadOp>([&](auto) { return "load"; })
      .Case<handshake::DynamaticStoreOp>([&](auto) { return "store"; })
      .Case<handshake::MemoryControllerOp>([&](auto) { return "MC"; })
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

DOTPrinter::DOTPrinter(bool legacy, bool debug, TimingDatabase *timingDB)
    : legacy(legacy), debug(debug), timingDB(timingDB), os(llvm::outs()) {
  assert(!legacy || timingDB && "timing database must exist in legacy mode");
};

LogicalResult DOTPrinter::printDOT(mlir::ModuleOp mod) {
  // We support at most one function per module
  auto funcs = mod.getOps<handshake::FuncOp>();
  if (funcs.empty())
    return success();
  if (++funcs.begin() != funcs.end()) {
    mod->emitOpError()
        << "we currently only support one handshake function per module";
    return failure();
  }
  handshake::FuncOp funcOp = *funcs.begin();

  if (legacy) {
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

    patchUpIRForLegacy(funcOp);
  }

  mlir::raw_indented_ostream os(llvm::outs());

  // Print the graph
  return printFunc(funcOp);
}

std::string DOTPrinter::getNodeName(Operation *op) {
  auto opNameIt = opNameMap.find(op);
  assert(opNameIt != opNameMap.end() &&
         "No name registered for the operation!");
  return opNameIt->second;
}

std::string DOTPrinter::getArgumentName(handshake::FuncOp funcOp, size_t idx) {
  auto numArgs = funcOp.getNumArguments();
  assert(idx < numArgs && "argument index too high");
  if (idx == numArgs - 1 && legacy)
    // Legacy Dynamatic expects the start signal to be called start_0
    return "start_0";
  return funcOp.getArgName(idx).getValue().str();
}

void DOTPrinter::openSubgraph(std::string &name, std::string &label) {
  os << "subgraph \"" << name << "\" {\n";
  os.indent();
  os << "label=\"" << label << "\"\n";
}

void DOTPrinter::closeSubgraph() {
  os.unindent();
  os << "}\n";
}

LogicalResult DOTPrinter::printNode(Operation *op) {

  std::string prettyLabel = getPrettyPrintedNodeLabel(op);
  std::string canonicalName =
      op->getName().getStringRef().str() +
      (isa<arith::CmpIOp, arith::CmpFOp>(op) ? prettyLabel : "");

  // Print node name
  auto opName = opNameMap[op];
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
            .Case<handshake::MemoryControllerOp, handshake::DynamaticLoadOp,
                  handshake::DynamaticStoreOp>([&](auto) { return "coral"; })
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
  os << ", label=\"";
  if (!debug)
    os << prettyLabel;
  else {
    os << opName;
  }
  os << "\"";

  // Determine style
  os << ", style=\"filled";
  if (auto controlInterface = dyn_cast<handshake::ControlInterface>(op);
      controlInterface && controlInterface.isControl())
    os << ", " + CONTROL_STYLE;
  os << "\", ";
  if (legacy && failed(annotateNode(op)))
    return failure();
  os << "]\n";

  return success();
}

LogicalResult DOTPrinter::printEdge(Operation *src, Operation *dst, Value val) {

  // In legacy mode, skip edges from branch-like operations to bitwidth
  // modifiers in between blocks
  if (legacy && isBitModBetweenBlocks(dst))
    return success();

  // "Jump over" bitwidth modification operations that go to a merge-like
  // operation in a different block
  std::string srcNodeName =
      legacy && isBitModBetweenBlocks(src)
          ? getNodeName(src->getOperand(0).getDefiningOp())
          : getNodeName(src);

  os << "\"" << srcNodeName << "\" -> \"" << getNodeName(dst) << "\" ["
     << getStyleOfValue(val);
  if (legacy && failed(annotateEdge(src, dst, val)))
    return failure();
  os << "]\n";
  return success();
}

LogicalResult DOTPrinter::printFunc(handshake::FuncOp funcOp) {
  std::map<std::string, unsigned> opTypeCntrs;
  DenseMap<Operation *, unsigned> opIDs;

  os << "Digraph G {\n";
  os.indent();
  os << "splines=spline;\n";
  os << "compound=true; // Allow edges between clusters\n";

  // Sequentially scan across the operations in the function and assign
  // instance IDs to each operation
  for (auto &op : funcOp.getOps())
    if (auto memOp = dyn_cast<handshake::MemoryControllerOp>(op))
      // Memories already have unique IDs, so make their name match it
      opIDs[&op] = memOp.getId();
    else
      opIDs[&op] = opTypeCntrs[op.getName().getStringRef().str()]++;

  os << "node [shape=box, style=filled, fillcolor=\"white\"]\n";

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
    if (legacy && failed(annotateArgumentNode(funcOp, arg.index())))
      return failure();
    os << "]\n";
  }

  // Print nodes corresponding to function operations
  os << "// Function operations\n";
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

    // In legacy mode, do not print bitwidth modification operation between
    // branch-like and merge-like operations
    if (legacy && isBitModBetweenBlocks(&op))
      continue;

    // Print the operation
    if (failed(printNode(&op)))
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
        for (auto *user : arg.getUsers()) {
          auto argLabel = getArgumentName(funcOp, idx);
          os << "\"" << argLabel << "\" -> \"" << getNodeName(user) << "\" ["
             << getStyleOfValue(arg);
          if (legacy && failed(annotateArgumentEdge(funcOp, idx, user)))
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
    openSubgraph(graphName, graphLabel);

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
            if (failed(printEdge(op, useOp, res)))
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
    closeSubgraph();

    // Print outgoing edges for this block
    if (!outgoingEdges.empty())
      os << "// Edges outgoing of basic block " << blockStrID << "\n";
    for (auto &[op, useOp, res] : outgoingEdges)
      if (failed(printEdge(op, useOp, res)))
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
        if (failed(printEdge(op, use.getOwner(), res)))
          return failure();

  // Verify that annotations are valid in legacy mode
  if (legacy && failed(verifyDOT(funcOp)))
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
