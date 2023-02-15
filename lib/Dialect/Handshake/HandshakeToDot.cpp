//===- HandshakeToDot.cpp - Handshale to DOT pass ---------------*- C++ -*-===//
//
// This file contains the implementation of the handshake to DOT pass. It
// produces a .dot file (in the DOT language) parsable by Graphviz and
// containing the graph representation of the input handshake-level IR. The pass
// leaves the actual handshake-level IR unchanged.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Conversion/StandardToHandshakeFPGA18.h"
#include "dynamatic/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Dialect/Handshake/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

namespace {
struct HandshakeToDotPass : public HandshakeToDotBase<HandshakeToDotPass> {

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Resolve the instance graph to get a top-level module.
    std::string topLevel;
    handshake::InstanceGraph uses;
    SmallVector<std::string> sortedFuncs;
    if (resolveInstanceGraph(m, uses, topLevel, sortedFuncs).failed()) {
      signalPassFailure();
      return;
    }

    handshake::FuncOp topLevelOp =
        cast<handshake::FuncOp>(m.lookupSymbol(topLevel));

    // Create top-level graph.
    std::error_code ec;
    llvm::raw_fd_ostream outfile(topLevel + ".dot", ec);
    mlir::raw_indented_ostream os(outfile);

    os << "Digraph G {\n";
    os.indent();
    os << "splines=spline;\n";
    os << "compound=true; // Allow edges between clusters\n";
    dotPrint(os, "TOP", topLevelOp, /*isTop=*/true);
    os.unindent();
    os << "}\n";
    outfile.close();
  };

private:
  /// Prints an instance of a handshake.func to the graph. Returns the unique
  /// name that was assigned to the instance.
  std::string dotPrint(mlir::raw_indented_ostream &os, StringRef parentName,
                       handshake::FuncOp f, bool isTop);

  /// Maintain a mapping of module names and the number of times one of those
  /// modules have been instantiated in the design. This is used to generate
  /// unique names in the output graph.
  std::map<std::string, unsigned> instanceIdMap;

  /// A mapping between operations and their unique name in the .dot file.
  DenseMap<Operation *, std::string> opNameMap;

  /// Returns the name of the vertex representing the operation.
  std::string getNodeName(Operation *op);

  /// Prints an edge between a source and destination operation, which are
  /// linked by aresult of the source that the destination uses as an operand.
  template <typename Stream>
  void printEdge(Stream &os, Operation *src, Operation *dst, Value val);

  void openSubgraph(mlir::raw_indented_ostream &os, std::string name,
                    std::string label);

  void closeSubgraph(mlir::raw_indented_ostream &os);
};

} // namespace

static bool isControlOp(Operation *op) {
  auto controlInterface = dyn_cast<handshake::ControlInterface>(op);
  return controlInterface && controlInterface.isControl();
}

static const std::string CONTROL_STYLE = " style=dashed ";

static std::string getNodeStyle(Value val) {
  return isa<NoneType>(val.getType()) ? CONTROL_STYLE : "";
}

static std::string getEdgeStyle(Value result) {
  return isa<NoneType>(result.getType()) ? CONTROL_STYLE : "";
}

/// Prints an operation to the dot file and returns the unique name for the
/// operation within the graph.
static std::string dotPrintNode(mlir::raw_indented_ostream &outfile,
                                StringRef instanceName, Operation *op,
                                DenseMap<Operation *, unsigned> &opIDs) {

  // Determine node name
  // We use "." to distinguish hierarchy in the dot file, but an op by default
  // prints using "." between the dialect name and the op name. Replace uses of
  // "." with "_".
  std::string opDialectName = op->getName().getStringRef().str();
  std::replace(opDialectName.begin(), opDialectName.end(), '.', '_');
  std::string opName =
      (instanceName + "." + opDialectName + std::to_string(opIDs[op])).str();
  outfile << "\"" << opName << "\""
          << " [";

  // Determine fill color
  outfile << "fillcolor = ";
  outfile << llvm::TypeSwitch<Operation *, std::string>(op)
                 .Case<handshake::ForkOp, handshake::LazyForkOp,
                       handshake::MuxOp, handshake::JoinOp>(
                     [&](auto) { return "lavender"; })
                 .Case<handshake::BufferOp>([&](auto) { return "lightgreen"; })
                 .Case<handshake::DynamaticReturnOp, handshake::EndOp>(
                     [&](auto) { return "gold"; })
                 .Case<handshake::SinkOp, handshake::ConstantOp>(
                     [&](auto) { return "gainsboro"; })
                 .Case<handshake::MemoryControllerOp,
                       handshake::DynamaticLoadOp, handshake::DynamaticStoreOp>(
                     [&](auto) { return "coral"; })
                 .Case<handshake::MergeOp, handshake::ControlMergeOp,
                       handshake::BranchOp, handshake::ConditionalBranchOp>(
                     [&](auto) { return "lightblue"; })
                 .Default([&](auto) { return "moccasin"; });

  // Determine shape
  outfile << ", shape=";
  if (op->getDialect()->getNamespace() == "handshake")
    outfile << "box";
  else
    outfile << "oval";

  // Determine label
  outfile << ", label=\"";
  outfile
      << llvm::TypeSwitch<Operation *, std::string>(op)
             // handshake operations
             .Case<handshake::ConstantOp>([&](auto op) {
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
                 [&](auto) { return "/"; })
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
  outfile << "\"";

  // Display control nodes with a dashed border
  outfile << ", style=\"filled";
  if (isControlOp(op))
    outfile << ", dashed";
  outfile << "\"";
  outfile << "]\n";

  return opName;
}

static std::string getLocalName(StringRef instanceName, StringRef suffix) {
  return (instanceName + "." + suffix).str();
}

static std::string getArgName(handshake::FuncOp op, unsigned index) {
  return op.getArgName(index).getValue().str();
}

static std::string getUniqueArgName(StringRef instanceName,
                                    handshake::FuncOp op, unsigned index) {
  return getLocalName(instanceName, getArgName(op, index));
}

std::string HandshakeToDotPass::getNodeName(Operation *op) {
  auto opNameIt = opNameMap.find(op);
  assert(opNameIt != opNameMap.end() &&
         "No name registered for the operation!");
  return opNameIt->second;
}

template <typename Stream>
void HandshakeToDotPass::printEdge(Stream &stream, Operation *src,
                                   Operation *dst, Value val) {
  stream << "\"" << getNodeName(src) << "\" -> \"" << getNodeName(dst) << "\" ["
         << getEdgeStyle(val) << "]\n";
}

void HandshakeToDotPass::openSubgraph(mlir::raw_indented_ostream &os,
                                      std::string name, std::string label) {
  os << "subgraph \"" << name << "\" {\n";
  os.indent();
  os << "label=\"" << label << "\"\n";
}

void HandshakeToDotPass::closeSubgraph(mlir::raw_indented_ostream &os) {
  os.unindent();
  os << "}\n";
}

std::string HandshakeToDotPass::dotPrint(mlir::raw_indented_ostream &os,
                                         StringRef parentName,
                                         handshake::FuncOp funcOp, bool isTop) {
  std::map<std::string, unsigned> opTypeCntrs;
  DenseMap<Operation *, unsigned> opIDs;
  auto name = funcOp.getName();
  unsigned thisId = instanceIdMap[name.str()]++;
  std::string instanceName = parentName.str() + "." + name.str();
  if (!isTop)
    instanceName += std::to_string(thisId);

  // Sequentially scan across the operations in the function and assign instance
  // IDs to each operation
  for (auto &op : funcOp.getOps())
    opIDs[&op] = opTypeCntrs[op.getName().getStringRef().str()]++;

  if (!isTop) {
    os << "// Subgraph for instance of " << name << "\n";
    openSubgraph(os, "cluster_" + instanceName, name.str());
    os << "labeljust=\"l\"\n";
    os << "color = \"darkgreen\"\n";
  }
  os << "node [shape=box style=filled fillcolor=\"white\"]\n";

  // Print nodes corresponding to function arguments
  os << "// Function argument nodes\n";
  for (const auto &arg : enumerate(funcOp.getArguments())) {
    if (isa<MemRefType>(arg.value().getType()))
      // Arguments with memref types are represented by memory interfaces inside
      // the function so they are not displayed
      continue;

    auto argLabel = getArgName(funcOp, arg.index());
    auto argNodeName = getLocalName(instanceName, argLabel);
    os << "\"" << argNodeName << "\" [shape=diamond"
       << getNodeStyle(arg.value()) << " label=\"" << argLabel << "\"]\n";
  }

  // Print nodes corresponding to function operations
  for (auto &op : funcOp.getOps())
    if (auto instOp = dyn_cast<handshake::InstanceOp>(op); instOp)
      assert(false && "not supported yet");
    else
      opNameMap[&op] = dotPrintNode(os, instanceName, &op, opIDs);

  // Get function's "blocks". These leverage the "bb" attributes attached to
  // operations in handshake functions to display operations belonging to the
  // same original basic block together
  auto handshakeBlocks = dynamatic::getHandshakeBlocks(funcOp);

  // Print all edges incoming from operations in a block
  for (auto &[blockID, ops] : handshakeBlocks.blocks) {

    // For each block, we create a subgraph to contain all edges between two
    // operations of that block
    auto blockStrID = std::to_string(blockID);
    openSubgraph(os, "cluster" + blockStrID, "block" + blockStrID);

    // Collect all edges leaving the block and print them after the subgraph
    std::vector<std::string> outgoingEdges;

    // Iterate over all uses of all results of all operations inside the block
    for (auto op : ops) {
      for (auto res : op->getResults())
        for (auto &use : res.getUses()) {
          // Add edge to subgraph or outgoing edges depending on the block of
          // the operation using the result
          Operation *useOp = use.getOwner();
          if (auto bb = useOp->getAttrOfType<mlir::IntegerAttr>(BB_ATTR);
              bb && bb.getValue().getZExtValue() == blockID)
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
      for (auto arg : llvm::enumerate(funcOp.getArguments()))
        if (!isa<MemRefType>(arg.value().getType()))
          for (auto &use : arg.value().getUses()) {
            Operation *useOp = use.getOwner();
            auto argName = getUniqueArgName(instanceName, funcOp, arg.index());
            os << "\"" << argName << "\" -> \"" << getNodeName(useOp) << "\" ["
               << getEdgeStyle(arg.value()) << "]\n";
          }

    // Close the subgraph
    closeSubgraph(os);

    // Print outgoing edges for this block
    if (!outgoingEdges.empty())
      os << "// Outgoing edges of block " << blockID << "\n";
    for (auto &edge : outgoingEdges)
      os << edge;
  }

  // Print all edges incoming from operations not belonging to any block outside
  // of all subgraphs
  for (auto op : handshakeBlocks.outOfBlocks)
    for (auto res : op->getResults())
      for (auto &use : res.getUses())
        printEdge(os, op, use.getOwner(), res);

  if (!isTop)
    closeSubgraph(os);

  return instanceName;
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakeToDotPass() {
  return std::make_unique<HandshakeToDotPass>();
}
