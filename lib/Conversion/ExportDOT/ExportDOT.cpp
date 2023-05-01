//===- ExportDOT.cpp - Export handshake to DOT pass -------------*- C++ -*-===//
//
// This file contains the implementation of the export to DOT pass. It
// produces a .dot file (in the DOT language) parsable by Graphviz and
// containing the graph representation of the input handshake-level IR. The pass
// leaves the actual handshake-level IR unchanged.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Conversion/PassDetails.h"
#include "dynamatic/Conversion/Passes.h"
#include "dynamatic/Conversion/StandardToHandshakeFPGA18.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <iomanip>
#include <string>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

namespace {
struct ExportDOTPass : public ExportDOTBase<ExportDOTPass> {

  ExportDOTPass(bool legacy) { this->legacy = legacy; }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    markAllAnalysesPreserved();

    // We support at most one function per module
    auto funcs = mod.getOps<handshake::FuncOp>();
    if (funcs.empty())
      return;
    if (++funcs.begin() != funcs.end()) {
      mod->emitOpError()
          << "we currently only support one handshake function per module";
      return signalPassFailure();
    }
    handshake::FuncOp funcOp = *funcs.begin();

    // Create the file to store the graph
    std::error_code ec;
    llvm::raw_fd_ostream outfile(funcOp.getNameAttr().str() + ".dot", ec);
    mlir::raw_indented_ostream os(outfile);

    // Print the graph
    os << "Digraph G {\n";
    os.indent();
    os << "splines=spline;\n";
    os << "compound=true; // Allow edges between clusters\n";
    if (failed(printFunc(os, funcOp))) {
      funcOp.emitOpError() << "failed to export to DOT";
      outfile.close();
      return signalPassFailure();
    }
    os.unindent();
    os << "}\n";

    outfile.close();
  };

private:
  /// Maintain a mapping of module names and the number of times one of those
  /// modules have been instantiated in the design. This is used to generate
  /// unique names in the output graph.
  std::map<std::string, unsigned> instanceIdMap;

  /// A mapping between operations and their unique name in the .dot file.
  DenseMap<Operation *, std::string> opNameMap;

  /// Returns the name of a function's argument given its index.
  std::string getArgumentName(handshake::FuncOp funcOp, size_t idx);

  /// Returns the name of the node representing the operation.
  std::string getNodeName(Operation *op);

  /// Prints a node corresponding to an operation and, on success, returns a
  /// unique name for the operation in the outName argument.
  LogicalResult printNode(mlir::raw_indented_ostream &os, Operation *op,
                          unsigned opID, std::string &outName);

  /// Prints an instance of a handshake.func to the graph.
  LogicalResult printFunc(mlir::raw_indented_ostream &os,
                          handshake::FuncOp funcOp);

  /// Prints an edge between a source and destination operation, which are
  /// linked by a result of the source that the destination uses as an
  /// operand.
  template <typename Stream>
  void printEdge(Stream &os, Operation *src, Operation *dst, Value val);

  /// Opens a subgraph in the DOT file using the provided name and label.
  void openSubgraph(mlir::raw_indented_ostream &os, std::string name,
                    std::string label);

  /// Closes a subgraph in the DOT file.
  void closeSubgraph(mlir::raw_indented_ostream &os);
};

/// Holds information about data attributes for a DOT node.
struct NodeInfo {
  /// The node's type.
  std::string type;
  /// A mapping between attribute name and value (printed between "").
  std::map<std::string, std::string> stringAttr;
  /// A mapping between attribute name and value (printed without "").
  std::map<std::string, int> intAttr;

  /// Constructs a NodeInfo with a specific type.
  NodeInfo(std::string type) : type(type){};

  /// Prints all stored data attributes on the output stream. The function
  /// doesn't insert [brackets] around the attributes; it is the responsibility
  /// of the caller of this method to insert an opening bracket before the call
  /// and a closing bracket after the call.
  void print(mlir::raw_indented_ostream &os) {
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
};

/// Holds information about data attributes for a DOT edge.
struct EdgeInfo {
  /// The port number of the edge's source node.
  size_t from;
  /// The port number of the edge's destination node.
  size_t to;
  /// If the edge is between a memory operation and a memory interface,
  /// indicates whether the edge represents an address or a data value.
  std::optional<bool> memAddress;

  /// Prints all stored data attributes on the output stream. The function
  /// doesn't insert [brackets] around the attributes; it is the responsibility
  /// of the caller of this method to insert an opening bracket before the call
  /// and a closing bracket after the call.
  template <typename Stream> void print(Stream &stream) {
    stream << "from=\"out" << from << "\", to=\"in" << to << "\"";
    if (memAddress.has_value())
      stream << ", mem_address=\"" << (memAddress.value() ? "true" : "false")
             << "\"";
  }
};

/// A list of ports (name and value).
using PortsData = std::vector<std::pair<std::string, Value>>;
/// A list of ports for memory interfaces (name, value, and potential name
/// suffix).
using MemPortsData = std::vector<std::tuple<std::string, Value, std::string>>;

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
    {"arith.sext", "sext_op"},     {"arith.extui", "zext_op"},
    {"arith.trunci", "trunc_op"},  {"arith.shrsi", "ashr_op"}};

/// Delay information for arith.addi and arith.subi operations.
static const std::string DELAY_ADD_SUB =
    "2.287 1.397 1.400 1.409 100.000 100.000 100.000 100.000";
/// Delay information for arith.muli and arith.divi operations.
static const std::string DELAY_MUL_DIV =
    "2.287 1.397 1.400 1.409 100.000 100.000 100.000 100.000";
/// Delay information for arith.subf and arith.mulf operations.
static const std::string DELAY_SUBF_MULF =
    "0.000 0.000 1.400 1.411 100.000 100.000 100.000 100.000";
/// Delay information for arith.andi, arith.ori, and arith.xori operations.
static const std::string DELAY_LOGIC_OP =
    "1.397 1.397 1.400 1.409 100.000 100.000 100.000 100.000";
/// Delay information for arith.sitofp and arith.remsi operations.
static const std::string DELAY_SITOFP_REMSI =
    "1.412 1.397 0.000 1.412 1.397 1.412 100.000 100.000";
/// Delay information for extension and truncation operations.
static const std::string DELAY_EXT_TRUNC =
    "0.672 0.672 1.397 1.397 100.000 100.000 100.000 100.000";

/// Maps name of arithmetic operation to "delay" attribute.
static std::unordered_map<std::string, std::string> arithNameToDelay{
    {"arith.addi", DELAY_ADD_SUB},
    {"arith.subi", DELAY_ADD_SUB},
    {"arith.muli", DELAY_MUL_DIV},
    {"arith.addf", "0.000,0.000,0.000,100.000,100.000,100.000,100.000,100.000"},
    {"arith.subf", DELAY_SUBF_MULF},
    {"arith.mulf", DELAY_SUBF_MULF},
    {"arith.divui", DELAY_MUL_DIV},
    {"arith.divsi", DELAY_MUL_DIV},
    {"arith.divf", "0.000 0.000 1.400 100.000 100.000 100.000 100.000 100.000"},
    {"arith.andi", DELAY_LOGIC_OP},
    {"arith.ori", DELAY_LOGIC_OP},
    {"arith.xori", DELAY_LOGIC_OP},
    {"arith.sitofp", DELAY_SITOFP_REMSI},
    {"arith.remsi", DELAY_SITOFP_REMSI},
    {"arith.sext", DELAY_EXT_TRUNC},
    {"arith.extui", DELAY_EXT_TRUNC},
    {"arith.trunci", DELAY_EXT_TRUNC},
    {"arith.shrsi", DELAY_EXT_TRUNC}};

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
  return llvm::TypeSwitch<Type, unsigned>(dataType)
      .Case<NoneType>([&](auto) { return 0; })
      .Case<IndexType>([&](auto) { return 32; })
      .Default([&](auto) { return dataType.getIntOrFloatBitWidth(); });
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
static std::string getIOFromValues(ValueRange values, std::string portType) {
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
  // Legacy Dynamatic always forces the index output to have a width of 1
  return "out1:" + std::to_string(getWidth(op.getResult())) + " out2?:" + "1";
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
static std::string getInputForSelect(handshake::SelectOp op) {
  PortsData ports;
  ports.push_back(std::make_pair("in1?", op.getCondOperand()));
  ports.push_back(std::make_pair("in2+", op.getTrueOperand()));
  ports.push_back(std::make_pair("in3-", op.getFalseOperand()));
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

  for (auto [idx, blockAccesses] : llvm::enumerate(op.getAccesses())) {
    // Add control signal if present
    if (op.bbHasControl(idx))
      allPorts.push_back(std::make_tuple("in" + std::to_string(inputIdx++),
                                         inputs[operandIdx++],
                                         "c" + std::to_string(ctrlIdx++)));

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
  assert(!users.empty() && (++users.begin() == users.end()) &&
         "address should only have one use");
  auto memOp = dyn_cast<handshake::MemoryControllerOp>(*users.begin());
  assert(memOp && "address user must be MemoryControllerOp");

  // Iterate over memory accesses to find the one that matches the address value
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

      // Go to the index corresponding to the next memory operation in the block
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

/// Finds the position (group index and operand index) of a value in the inputs
/// of a memory interface.
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
        return (idx == 1) ? (size_t)0 : (size_t)1;
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
        return (idx == 1) ? 0 : 1;
      })
      .Case<handshake::MemoryControllerOp>(
          [&](handshake::MemoryControllerOp memOp) {
            if (isSrcOp)
              return idx;

            // Legacy Dynamatic puts all control operands before all data
            // operands, whereas for us each control operand appears just before
            // the data inputs of the block it corresponds to
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

/// Computes all data attributes of an operation for use in legacy Dynamatic and
/// prints them to the output stream; it is the responsibility of the caller of
/// this method to insert an opening bracket before the call and a closing
/// bracket after the call.
static LogicalResult annotateNode(mlir::raw_indented_ostream &os,
                                  Operation *op) {
  auto info =
      llvm::TypeSwitch<Operation *, NodeInfo>(op)
          .Case<handshake::MergeOp>([&](auto) {
            auto info = NodeInfo("Merge");
            info.stringAttr["delay"] =
                "1.397 1.412 0.000 100.000 100.000 100.000 100.000 100.000";
            return info;
          })
          .Case<handshake::MuxOp>([&](handshake::MuxOp op) {
            auto info = NodeInfo("Mux");
            info.stringAttr["in"] = getInputForMux(op);
            info.stringAttr["delay"] =
                "1.412 1.397 0.000 1.412 1.397 1.412 100.000 100.000";
            return info;
          })
          .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp op) {
            auto info = NodeInfo("CntrlMerge");
            info.stringAttr["out"] = getOutputForControlMerge(op);
            info.stringAttr["delay"] =
                "0.000 1.397 0.000 100.000 100.000 100.000 100.000 100.000";
            return info;
          })
          .Case<handshake::ConditionalBranchOp>(
              [&](handshake::ConditionalBranchOp op) {
                auto info = NodeInfo("Branch");
                info.stringAttr["in"] = getInputForCondBranch(op);
                info.stringAttr["out"] = getOutputForCondBranch(op);
                info.stringAttr["delay"] =
                    "0.000 1.409 1.411 1.412 1.400 1.412 100.000 100.000";
                return info;
              })
          .Case<handshake::BufferOp>([&](handshake::BufferOp bufOp) {
            auto info = NodeInfo("Buffer");
            info.intAttr["slots"] = bufOp.getNumSlots();
            info.stringAttr["tansparent"] =
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
            info.intAttr["latency"] = 2;
            info.stringAttr["delay"] =
                "1.412 1.409 0.000 100.000 100.000 100.000 100.000 100.000";
            return info;
          })
          .Case<handshake::DynamaticStoreOp>(
              [&](handshake::DynamaticStoreOp op) {
                auto info = NodeInfo("Operator");
                info.stringAttr["op"] = "mc_store_op";
                info.stringAttr["in"] = getInputForStoreOp(op);
                info.stringAttr["out"] = getOutputForStoreOp(op);
                info.intAttr["portId"] = findMemoryPort(op.getAddressResult());
                info.stringAttr["delay"] =
                    "0.672 1.397 1.400 1.409 100.000 100.000 100.000 100.000";
                return info;
              })
          .Case<handshake::ForkOp>([&](auto) {
            auto info = NodeInfo("Fork");
            info.stringAttr["delay"] =
                "0.000 0.100 0.100 100.000 100.000 100.000 100.000 100.000";
            return info;
          })
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

            // Legacy Dynamatic uses the output width of the operations also as
            // input width for some reason, make it so
            info.stringAttr["in"] = getIOFromValues(op->getResults(), "in");
            info.stringAttr["out"] = getIOFromValues(op->getResults(), "out");

            info.stringAttr["value"] = stream.str();
            info.stringAttr["delay"] =
                "0.000 0.000 0.000 100.000 100.000 100.000 100.000 100.000";
            return info;
          })
          .Case<handshake::SelectOp>([&](handshake::SelectOp op) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = "select_op";
            info.stringAttr["in"] = getInputForSelect(op);
            info.stringAttr["delay"] =
                "1.397 1.397 1.412 2.061 100.000 100.000 100.000 100.000";
            return info;
          })
          .Case<handshake::DynamaticReturnOp>([&](auto) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = "ret_op";
            info.stringAttr["delay"] =
                "1.412 1.409 0.000 100.000 100.000 100.000 100.000 100.000";
            return info;
          })
          .Case<handshake::EndOp>([&](handshake::EndOp op) {
            auto info = NodeInfo("Exit");
            info.stringAttr["in"] = getInputForEnd(op);

            // Output ports of end node are determined by function result types
            std::stringstream stream;
            auto funcOp = op->getParentOfType<handshake::FuncOp>();
            for (auto [idx, res] : llvm::enumerate(funcOp.getResultTypes()))
              stream << "out" << (idx + 1) << ":" << getWidth(res);
            info.stringAttr["out"] = stream.str();

            info.stringAttr["delay"] =
                "1.397 0.000 1.397 1.409 100.000 100.000 100.000 100.000";
            return info;
          })
          .Case<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp,
                arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::MulIOp,
                arith::MulFOp, arith::DivUIOp, arith::DivSIOp, arith::DivFOp,
                arith::SIToFPOp, arith::RemSIOp, arith::ExtSIOp, arith::ExtUIOp,
                arith::TruncIOp, arith::ShRSIOp>([&](auto) {
            auto info = NodeInfo("Operator");
            auto opName = op->getName().getStringRef().str();
            info.stringAttr["op"] = arithNameToOpName[opName];
            info.stringAttr["delay"] = arithNameToDelay[opName];

            // Set non-zero latencies
            if (opName == "arith.divui" || opName == "arith.divsi")
              info.intAttr["latency"] = 36;
            else if (opName == "arith.muli")
              info.intAttr["latency"] = 4;
            else if (opName == "arith.fadd" || opName == "arith.fsub")
              info.intAttr["latency"] = 10;
            else if (opName == "arith.divf")
              info.intAttr["latency"] = 30;
            else if (opName == "arith.mulf")
              info.intAttr["latency"] = 6;

            return info;
          })
          .Case<arith::CmpIOp>([&](arith::CmpIOp op) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = cmpINameToOpName[op.getPredicate()];
            info.stringAttr["delay"] =
                "1.907 1.397 1.400 1.409 100.000 100.000 100.000 100.000";
            return info;
          })
          .Case<arith::CmpFOp>([&](arith::CmpFOp op) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = cmpFNameToOpName[op.getPredicate()];
            info.intAttr["latency"] = 2;
            info.stringAttr["latency"] =
                "1.895 1.397 1.406 1.411 100.000 100.000 100.000 100.000";
            return info;
          })
          .Case<arith::IndexCastOp>([&](auto) {
            auto info = NodeInfo("Operator");
            info.stringAttr["op"] = "zext_op";
            info.stringAttr["delay"] = DELAY_EXT_TRUNC;
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
  if (info.intAttr.find("latency") == info.intAttr.end() &&
      info.type == "Operator")
    info.intAttr["latency"] = 0;

  // II is 1 for all operators
  if (info.type == "Operator")
    info.intAttr["II"] = 1;

  // Print to output
  info.print(os);
  return success();
}

/// Computes all data attributes of a function argument for use in legacy
/// Dynamatic and prints them to the output stream; it is the responsibility of
/// the caller of this method to insert an opening bracket before the call and a
/// closing bracket after the call.
static void annotateArgument(mlir::raw_indented_ostream &os,
                             BlockArgument arg) {
  NodeInfo info("Entry");
  info.stringAttr["in"] = getIOFromValues(ValueRange(arg), "in");
  info.stringAttr["out"] = getIOFromValues(ValueRange(arg), "out");
  info.intAttr["bbID"] = 1;
  if (isa<NoneType>(arg.getType()))
    info.stringAttr["control"] = "true";
  info.print(os);
}

/// Computes all data attributes of an edge for use in legacy Dynamatic and
/// prints them to the output stream; it is the responsibility of the caller of
/// this method to insert an opening bracket before the call and a closing
/// bracket after the call.
template <typename Stream>
static void annotateEdge(Stream &os, Operation *src, Operation *dst,
                         Value val) {
  EdgeInfo info;

  // Locate value in source results and destination operands
  auto resIdx = findIndexInRange(src->getResults(), val);
  auto argIdx = findIndexInRange(dst->getOperands(), val);

  // Handle to and from attributes (with special cases). Also add 1 to each
  // index since first ports are called in1/out1
  info.from = fixPortNumber(src, val, resIdx, true) + 1;
  info.to = fixPortNumber(dst, val, argIdx, false) + 1;

  // Handle the mem_address optional attribute
  if (auto srcMem = dyn_cast<handshake::MemoryControllerOp>(src); srcMem) {
    if (isa<handshake::DynamaticLoadOp, handshake::DynamaticStoreOp>(dst))
      info.memAddress = false;
  } else if (auto dstMem = dyn_cast<handshake::MemoryControllerOp>(dst); dstMem)
    if (isa<handshake::DynamaticLoadOp, handshake::DynamaticStoreOp>(src))
      // Is val the address result of the memory operation?
      info.memAddress = val == src->getResult(0);

  info.print(os);
}

static void annotateArgumentEdge(mlir::raw_indented_ostream &os,
                                 BlockArgument arg, Operation *dst) {
  EdgeInfo info;

  // Locate value in destination operands
  auto argIdx = findIndexInRange(dst->getOperands(), arg);

  // Handle to and from attributes (with special cases). Also add 1 to each
  // index since first ports are called in1/out1
  info.from = 1;
  info.to = fixPortNumber(dst, arg, argIdx, false) + 1;

  info.print(os);
}

// ============================================================================
// DOT export
// ============================================================================

/// Style attribute value for control nodes/edges.
static const std::string CONTROL_STYLE = "dashed";

/// Determines the style attribute of a value.
static std::string getStyleOfValue(Value result) {
  return isa<NoneType>(result.getType()) ? "style=" + CONTROL_STYLE + ", " : "";
}

std::string ExportDOTPass::getNodeName(Operation *op) {
  auto opNameIt = opNameMap.find(op);
  assert(opNameIt != opNameMap.end() &&
         "No name registered for the operation!");
  return opNameIt->second;
}

std::string ExportDOTPass::getArgumentName(handshake::FuncOp funcOp,
                                           size_t idx) {
  auto numArgs = funcOp.getNumArguments();
  assert(idx < numArgs && "argument index too high");
  if (idx == numArgs - 1 && legacy)
    // Legacy Dynamatic expects the start signal to be called start_0
    return "start_0";
  else
    return funcOp.getArgName(idx).getValue().str();
}

LogicalResult ExportDOTPass::printNode(mlir::raw_indented_ostream &os,
                                       Operation *op, unsigned opID,
                                       std::string &outName) {

  // Determine node name
  std::string opFullName = op->getName().getStringRef().str();
  std::replace(opFullName.begin(), opFullName.end(), '.', '_');
  std::string opName = opFullName + std::to_string(opID);
  os << "\"" << opName << "\""
     << " [";

  // Determine fill color
  os << "fillcolor=";
  os << llvm::TypeSwitch<Operation *, std::string>(op)
            .Case<handshake::ForkOp, handshake::LazyForkOp, handshake::MuxOp,
                  handshake::JoinOp>([&](auto) { return "lavender"; })
            .Case<handshake::BufferOp>([&](auto) { return "lightgreen"; })
            .Case<handshake::DynamaticReturnOp, handshake::EndOp>(
                [&](auto) { return "gold"; })
            .Case<handshake::SinkOp, handshake::ConstantOp>(
                [&](auto) { return "gainsboro"; })
            .Case<handshake::MemoryControllerOp, handshake::DynamaticLoadOp,
                  handshake::DynamaticStoreOp>([&](auto) { return "coral"; })
            .Case<handshake::MergeOp, handshake::ControlMergeOp,
                  handshake::BranchOp, handshake::ConditionalBranchOp>(
                [&](auto) { return "lightblue"; })
            .Default([&](auto) { return "moccasin"; });

  // Determine shape
  os << ", shape=";
  if (op->getDialect()->getNamespace() == "handshake")
    os << "box";
  else
    os << "oval";

  // Determine label
  os << ", label=\"";
  os << llvm::TypeSwitch<Operation *, std::string>(op)
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
            .Case<handshake::ConditionalBranchOp>(
                [&](auto) { return "cbranch"; })
            .Case<handshake::BufferOp>([&](auto op) {
              std::string n = "buffer ";
              n += stringifyEnum(op.getBufferType());
              return n;
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
                return "cmp-false";
              case arith::CmpFPredicate::AlwaysTrue:
                return "cmp-true";
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
  os << "\"";

  // Determine style
  os << ", style=\"filled";
  if (auto controlInterface = dyn_cast<handshake::ControlInterface>(op);
      controlInterface && controlInterface.isControl())
    os << ", " + CONTROL_STYLE;
  os << "\", ";
  if (legacy && failed(annotateNode(os, op)))
    return failure();
  os << "]\n";

  outName = opName;
  return success();
}

template <typename Stream>
void ExportDOTPass::printEdge(Stream &stream, Operation *src, Operation *dst,
                              Value val) {
  stream << "\"" << getNodeName(src) << "\" -> \"" << getNodeName(dst) << "\" ["
         << getStyleOfValue(val);
  if (legacy)
    annotateEdge(stream, src, dst, val);
  stream << "]\n";
}

void ExportDOTPass::openSubgraph(mlir::raw_indented_ostream &os,
                                 std::string name, std::string label) {
  os << "subgraph \"" << name << "\" {\n";
  os.indent();
  os << "label=\"" << label << "\"\n";
}

void ExportDOTPass::closeSubgraph(mlir::raw_indented_ostream &os) {
  os.unindent();
  os << "}\n";
}

LogicalResult ExportDOTPass::printFunc(mlir::raw_indented_ostream &os,
                                       handshake::FuncOp funcOp) {
  std::map<std::string, unsigned> opTypeCntrs;
  DenseMap<Operation *, unsigned> opIDs;

  // Sequentially scan across the operations in the function and assign
  // instance IDs to each operation
  for (auto &op : funcOp.getOps())
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
    os << "\"" << argLabel << "\" [shape=diamond, "
       << getStyleOfValue(arg.value()) << "label=\"" << argLabel << "\", ";
    if (legacy)
      annotateArgument(os, arg.value());
    os << "]\n";
  }

  // Print nodes corresponding to function operations
  os << "// Function operations\n";
  for (auto &op : funcOp.getOps())
    if (auto instOp = dyn_cast<handshake::InstanceOp>(op); instOp)
      assert(false && "multiple functions are not supported");
    else {
      std::string name;
      if (failed(printNode(os, &op, opIDs[&op], name)))
        return failure();
      opNameMap[&op] = name;
    }
  // Get function's "blocks". These leverage the "bb" attributes attached to
  // operations in handshake functions to display operations belonging to the
  // same original basic block together
  auto handshakeBlocks = getHandshakeBlocks(funcOp);

  // Print all edges incoming from operations in a block
  for (auto &[blockID, ops] : handshakeBlocks.blocks) {

    // For each block, we create a subgraph to contain all edges between two
    // operations of that block
    auto blockStrID = std::to_string(blockID);
    os << "// Edges within basic block " << blockStrID << "\n";
    openSubgraph(os, "cluster" + blockStrID, "block" + blockStrID);

    // Collect all edges leaving the block and print them after the subgraph
    std::vector<std::string> outgoingEdges;

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
    for (auto op : ops) {
      for (auto res : op->getResults())
        for (auto &use : res.getUses()) {
          // Add edge to subgraph or outgoing edges depending on the block of
          // the operation using the result
          Operation *useOp = use.getOwner();
          if (isEdgeInSubgraph(useOp, blockID))
            printEdge(os, op, useOp, res);
          else {
            std::stringstream edge;
            printEdge(edge, op, useOp, res);
            outgoingEdges.push_back(edge.str());
          }
        }
    }

    // For entry block, also add all edges incoming from function arguments
    if (blockID == 0)
      for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments()))
        if (!isa<MemRefType>(arg.getType()))
          for (auto user : arg.getUsers()) {
            auto argLabel = getArgumentName(funcOp, idx);
            os << "\"" << argLabel << "\" -> \"" << getNodeName(user) << "\" ["
               << getStyleOfValue(arg);
            if (legacy)
              annotateArgumentEdge(os, arg, user);
            os << "]\n";
          }

    // Close the subgraph
    closeSubgraph(os);

    // Print outgoing edges for this block
    if (!outgoingEdges.empty())
      os << "// Edges outgoing of basic block " << blockStrID << "\n";
    for (auto &edge : outgoingEdges)
      os << edge;
  }

  // Print all edges incoming from operations not belonging to any block
  // outside of all subgraphs
  os << "// Edges outside of all basic blocks\n";
  for (auto op : handshakeBlocks.outOfBlocks)
    for (auto res : op->getResults())
      for (auto &use : res.getUses())
        printEdge(os, op, use.getOwner(), res);

  return success();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createExportDOTPass(bool legacy) {
  return std::make_unique<ExportDOTPass>(legacy);
}
