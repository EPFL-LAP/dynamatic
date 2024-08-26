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
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "experimental/Transforms/Speculation/SpecAnnotatePaths.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
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
    {handshake::AddFOp::getOperationName(), "fadd_op"},
    {handshake::AddIOp::getOperationName(), "add_op"},
    {handshake::AndIOp::getOperationName(), "and_op"},
    {handshake::DivFOp::getOperationName(), "fdiv_op"},
    {handshake::DivSIOp::getOperationName(), "sdiv_op"},
    {handshake::DivUIOp::getOperationName(), "udiv_op"},
    {handshake::ExtSIOp::getOperationName(), "sext_op"},
    {handshake::ExtUIOp::getOperationName(), "zext_op"},
    {handshake::MulFOp::getOperationName(), "fmul_op"},
    {handshake::MulIOp::getOperationName(), "mul_op"},
    {handshake::OrIOp::getOperationName(), "or_op"},
    {handshake::SelectOp::getOperationName(), "select_op"},
    {handshake::ShLIOp::getOperationName(), "shl_op"},
    {handshake::SubIOp::getOperationName(), "sub_op"},
    {handshake::SubFOp::getOperationName(), "fsub_op"},
    {handshake::ShRSIOp::getOperationName(), "ashr_op"},
    {handshake::TruncIOp::getOperationName(), "trunc_op"},
    {handshake::XOrIOp::getOperationName(), "xor_op"},
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
static DenseMap<handshake::CmpIPredicate, StringRef> cmpINameToOpName{
    {handshake::CmpIPredicate::eq, "icmp_eq_op"},
    {handshake::CmpIPredicate::ne, "icmp_ne_op"},
    {handshake::CmpIPredicate::slt, "icmp_slt_op"},
    {handshake::CmpIPredicate::sle, "icmp_sle_op"},
    {handshake::CmpIPredicate::sgt, "icmp_sgt_op"},
    {handshake::CmpIPredicate::sge, "icmp_sge_op"},
    {handshake::CmpIPredicate::ult, "icmp_ult_op"},
    {handshake::CmpIPredicate::ule, "icmp_ule_op"},
    {handshake::CmpIPredicate::ugt, "icmp_ugt_op"},
    {handshake::CmpIPredicate::uge, "icmp_uge_op"},
};

/// Maps name of floating-point comparison type to "op" attribute.
static DenseMap<handshake::CmpFPredicate, StringRef> cmpFNameToOpName{
    {handshake::CmpFPredicate::AlwaysFalse, "fcmp_false_op"},
    {handshake::CmpFPredicate::OEQ, "fcmp_oeq_op"},
    {handshake::CmpFPredicate::OGT, "fcmp_ogt_op"},
    {handshake::CmpFPredicate::OGE, "fcmp_oge_op"},
    {handshake::CmpFPredicate::OLT, "fcmp_olt_op"},
    {handshake::CmpFPredicate::OLE, "fcmp_ole_op"},
    {handshake::CmpFPredicate::ONE, "fcmp_one_op"},
    {handshake::CmpFPredicate::ORD, "fcmp_orq_op"},
    {handshake::CmpFPredicate::UEQ, "fcmp_ueq_op"},
    {handshake::CmpFPredicate::UGT, "fcmp_ugt_op"},
    {handshake::CmpFPredicate::UGE, "fcmp_uge_op"},
    {handshake::CmpFPredicate::ULT, "fcmp_ult_op"},
    {handshake::CmpFPredicate::ULE, "fcmp_ule_op"},
    {handshake::CmpFPredicate::UNE, "fcmp_une_op"},
    {handshake::CmpFPredicate::UNO, "fcmp_uno_op"},
    {handshake::CmpFPredicate::AlwaysTrue, "fcmp_true_op"},
};

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

static StringRef getStyle(Value val) {
  return isa<handshake::ControlType>(val.getType()) ? DOTTED : SOLID;
}

/// Determines cosmetic attributes of the edge corresponding to the operand.
static std::string getEdgeStyle(OpOperand &oprd) {
  std::string attributes;
  // StringRef arrowhead =
  return "style=\"" + getStyle(oprd.get()).str() +
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
  while (isa_and_present<handshake::ExtSIOp, handshake::ExtUIOp,
                         handshake::TruncIOp, handshake::ForkOp>(userOp)) {
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
            ChannelType cstType = cstOp.getResult().getType();
            TypedAttr valueAttr = cstOp.getValueAttr();
            if (auto intType = dyn_cast<IntegerType>(cstType.getDataType())) {
              // Special case boolean attribute (which would result in an i1
              // constant integer results) to print true/false instead of 1/0
              if (auto boolAttr = dyn_cast<mlir::BoolAttr>(valueAttr))
                return boolAttr.getValue() ? "true" : "false";

              APInt value = cast<mlir::IntegerAttr>(valueAttr).getValue();
              if (intType.isUnsignedInteger())
                return std::to_string(value.getZExtValue());
              return std::to_string(value.getSExtValue());
            }
            if (isa<FloatType>(cstType.getDataType())) {
              mlir::FloatAttr attr = dyn_cast<mlir::FloatAttr>(valueAttr);
              return std::to_string(attr.getValue().convertToDouble());
            }
            // Fallback on an empty string
            return std::string("");
          })
      .Case<handshake::BufferOp>(
          [&](handshake::BufferOp bufferOp) -> std::string {
            // Try to infer the buffer type from HW parameters, if present
            auto params = bufferOp->getAttrOfType<DictionaryAttr>(
                RTL_PARAMETERS_ATTR_NAME);
            if (!params)
              return "buffer";
            auto optSlots = params.getNamed(BufferOp::NUM_SLOTS_ATTR_NAME);
            std::string numSlotsStr = "";
            if (optSlots) {
              if (auto numSlots = dyn_cast<IntegerAttr>(optSlots->getValue())) {
                if (numSlots.getType().isUnsignedInteger())
                  numSlotsStr = " [" + std::to_string(numSlots.getUInt()) + "]";
              }
            }
            auto optTiming = params.getNamed(BufferOp::TIMING_ATTR_NAME);
            if (!optTiming)
              return "buffer" + numSlotsStr;
            if (auto timing =
                    dyn_cast<handshake::TimingAttr>(optTiming->getValue())) {
              TimingInfo info = timing.getInfo();
              if (info == TimingInfo::oehb())
                return "oehb" + numSlotsStr;
              if (info == TimingInfo::tehb())
                return "tehb" + numSlotsStr;
            }
            return "buffer" + numSlotsStr;
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
      .Case<handshake::AddIOp, handshake::AddFOp>([&](auto) { return "+"; })
      .Case<handshake::SubIOp, handshake::SubFOp>([&](auto) { return "-"; })
      .Case<handshake::AndIOp>([&](auto) { return "&"; })
      .Case<handshake::OrIOp>([&](auto) { return "|"; })
      .Case<handshake::XOrIOp>([&](auto) { return "^"; })
      .Case<handshake::MulIOp, handshake::MulFOp>([&](auto) { return "*"; })
      .Case<handshake::DivUIOp, handshake::DivSIOp, handshake::DivFOp>(
          [&](auto) { return "div"; })
      .Case<handshake::ShRSIOp, handshake::ShRUIOp>([&](auto) { return ">>"; })
      .Case<handshake::ShLIOp>([&](auto) { return "<<"; })
      .Case<handshake::ExtSIOp, handshake::ExtUIOp, handshake::TruncIOp>(
          [&](auto) {
            unsigned opWidth = cast<ChannelType>(op->getOperand(0).getType())
                                   .getDataBitWidth();
            unsigned resWidth =
                cast<ChannelType>(op->getResult(0).getType()).getDataBitWidth();
            return "[" + std::to_string(opWidth) + "..." +
                   std::to_string(resWidth) + "]";
          })
      .Case<handshake::CmpIOp>([&](handshake::CmpIOp op) {
        switch (op.getPredicate()) {
        case handshake::CmpIPredicate::eq:
          return "==";
        case handshake::CmpIPredicate::ne:
          return "!=";
        case handshake::CmpIPredicate::uge:
        case handshake::CmpIPredicate::sge:
          return ">=";
        case handshake::CmpIPredicate::ugt:
        case handshake::CmpIPredicate::sgt:
          return ">";
        case handshake::CmpIPredicate::ule:
        case handshake::CmpIPredicate::sle:
          return "<=";
        case handshake::CmpIPredicate::ult:
        case handshake::CmpIPredicate::slt:
          return "<";
        }
      })
      .Case<handshake::CmpFOp>([&](handshake::CmpFOp op) {
        switch (op.getPredicate()) {
        case handshake::CmpFPredicate::OEQ:
        case handshake::CmpFPredicate::UEQ:
          return "==";
        case handshake::CmpFPredicate::ONE:
        case handshake::CmpFPredicate::UNE:
          return "!=";
        case handshake::CmpFPredicate::OGE:
        case handshake::CmpFPredicate::UGE:
          return ">=";
        case handshake::CmpFPredicate::OGT:
        case handshake::CmpFPredicate::UGT:
          return ">";
        case handshake::CmpFPredicate::OLE:
        case handshake::CmpFPredicate::ULE:
          return "<=";
        case handshake::CmpFPredicate::OLT:
        case handshake::CmpFPredicate::ULT:
          return "<";
        case handshake::CmpFPredicate::ORD:
          return "ordered?";
        case handshake::CmpFPredicate::UNO:
          return "unordered?";
        case handshake::CmpFPredicate::AlwaysFalse:
          return "false";
        case handshake::CmpFPredicate::AlwaysTrue:
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
      .Case<handshake::BufferOp>([&](auto) { return "lightgreen"; })
      .Case<handshake::EndOp>([&](auto) { return "gold"; })
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

DOTPrinter::DOTPrinter(EdgeStyle edgeStyle) : edgeStyle(edgeStyle) {}

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

  // Print the graph
  if (os)
    return printFunc(funcOp, *os);
  mlir::raw_indented_ostream stdOs(llvm::outs());
  return printFunc(funcOp, stdOs);
}

std::string DOTPrinter::getArgumentName(handshake::FuncOp funcOp, size_t idx) {
  auto numArgs = funcOp.getNumArguments();
  assert(idx < numArgs && "argument index too high");
  return funcOp.getArgName(idx).getValue().str();
}

std::string DOTPrinter::getResultName(handshake::FuncOp funcOp, size_t idx) {
  auto numResults = funcOp.getFunctionType().getNumResults();
  assert(idx < numResults && "result index too high");
  return funcOp.getResName(idx).getValue().str();
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
  if (isa<handshake::EndOp>(op))
    return success();

  // The node's DOT name
  std::string opName = getUniqueName(op).str();

  // The node's DOT "mlir_op" attribute
  std::string mlirOpName = op->getName().getStringRef().str();
  std::string prettyLabel = getPrettyNodeLabel(op);
  if (isa<handshake::CmpIOp, handshake::CmpFOp>(op))
    mlirOpName += prettyLabel;

  // The node's DOT "shape" attribute
  StringRef shape;
  if (isa<handshake::ArithOpInterface>(op))
    shape = "oval";
  else
    shape = "box";

  // The node's DOT "style" attribute
  std::string style = "filled";
  if (auto controlInterface = dyn_cast<handshake::ControlInterface>(op)) {
    if (controlInterface.isControl())
      style += ", " + DOTTED.str();
  }

  // Write the node
  os << "\"" << opName << "\""
     << " [mlir_op=\"" << mlirOpName << "\", label=\"" << prettyLabel
     << "\", fillcolor=" << getNodeColor(op) << ", shape=\"" << shape
     << "\", style=\"" << style << "\"]\n";
  return success();
}

LogicalResult DOTPrinter::printEdge(OpOperand &oprd,
                                    mlir::raw_indented_ostream &os) {
  Value val = oprd.get();
  Operation *src = val.getDefiningOp();
  Operation *dst = oprd.getOwner();

  std::string srcNodeName = getUniqueName(src).str();
  std::string dstNodeName;
  if (isa<handshake::EndOp>(dst)) {
    dstNodeName = getResultName(dst->getParentOfType<handshake::FuncOp>(),
                                oprd.getOperandNumber());
  } else {
    dstNodeName = getUniqueName(dst).str();
  }

  os << "\"" << srcNodeName << "\" -> \"" << dstNodeName << "\" ["
     << getEdgeStyle(oprd);
  if (isBackedge(val, dst))
    os << " color=\"blue\"";
  // Print speculative edge attribute
  if (experimental::speculation::isSpeculative(oprd, true))
    os << ((isBackedge(val, dst)) ? ", " : "") << " speculative=1";
  os << "]\n";
  return success();
}

LogicalResult DOTPrinter::printFunc(handshake::FuncOp funcOp,
                                    mlir::raw_indented_ostream &os) {
  std::string splines;
  if (edgeStyle == EdgeStyle::SPLINE)
    splines = "spline";
  else
    splines = "ortho";

  os << "Digraph G {\n";
  os.indent();
  os << "splines=" << splines << ";\n";
  os << "compound=true; // Allow edges between clusters\n";

  // Function arguments do not belong to any basic block
  os << "// Function arguments\n";
  for (const auto &[idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (isa<MemRefType>(arg.getType()))
      // Arguments with memref types are represented by memory interfaces
      // inside the function so they are not displayed
      continue;

    std::string argLabel = getArgumentName(funcOp, idx);
    os << "\"" << argLabel << R"(" [mlir_op="handshake.func", shape=diamond, )"
       << "label=\"" << argLabel << "\", style=\"" << getStyle(arg) << "\"]\n";
  }

  // Function results do not belong to any basic block
  os << "// Function results\n";
  ValueRange results = funcOp.getBodyBlock()->getTerminator()->getOperands();
  for (const auto &[idx, res] : llvm::enumerate(results)) {
    std::string resLabel = getResultName(funcOp, idx);
    os << "\"" << resLabel << R"(" [mlir_op="handshake.func", shape=diamond, )"
       << "label=\"" << resLabel << "\", style=\"" << getStyle(res) << "\"]\n";
  }

  // Get function's "blocks". These leverage the "bb" attributes attached to
  // operations in handshake functions to display operations belonging to the
  // same original basic block together
  LogicBBs blocks = getLogicBBs(funcOp);

  // Collect all edges that do not connect two nodes in the same block
  llvm::MapVector<unsigned, std::vector<OpOperand *>> outgoingEdges;

  // We print the function "block-by-block" by grouping nodes in the same block
  // (as well as edges between nodes of the same block) within DOT clusters
  for (auto &[blockID, blockOps] : blocks.blocks) {
    // Open the subgraph
    os << "// Units/Channels in BB " << blockID << "\n";
    std::string graphName = "cluster" + std::to_string(blockID);
    std::string graphLabel = "block" + std::to_string(blockID);
    openSubgraph(graphName, graphLabel, os);

    os << "// Units in BB " << blockID << "\n";
    for (Operation *op : blockOps) {
      if (failed(printNode(op, os)))
        return failure();
    }

    os << "// Channels in BB " << blockID << "\n";
    for (Operation *op : blockOps) {
      for (OpResult res : op->getResults()) {
        for (OpOperand &oprd : res.getUses()) {
          Operation *userOp = oprd.getOwner();
          std::optional<unsigned> bb = getLogicBB(userOp);
          if (bb && *bb == blockID && !isa<handshake::EndOp>(userOp)) {
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

  os << "// Units outside of all basic blocks\n";
  for (Operation *op : blocks.outOfBlocks) {
    if (failed(printNode(op, os)))
      return failure();
  }

  // Print edges coming from function arguments if they haven't been so far
  os << "// Channels from function arguments\n";
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (isa<MemRefType>(arg.getType()))
      continue;

    for (OpOperand &oprd : arg.getUses()) {
      Operation *ownerOp = oprd.getOwner();
      std::string argLabel = getArgumentName(funcOp, idx);
      os << "\"" << argLabel << "\" -> \"" << getUniqueName(ownerOp) << "\" ["
         << getEdgeStyle(oprd) << "]\n";
    }
  }

  // Print outgoing edges for each block
  for (auto &[blockID, blockEdges] : outgoingEdges) {
    os << "// Channels outgoing of BB " << blockID << "\n";
    for (OpOperand *oprd : blockEdges) {
      if (failed(printEdge(*oprd, os)))
        return failure();
    }
  }

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
