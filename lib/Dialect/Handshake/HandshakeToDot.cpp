//===- HandshakeToDot.cpp - Handshale to DOT conversion ---------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Dialect/Handshake/PassDetails.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;

static bool isControlOp(Operation *op) {
  auto controlInterface = dyn_cast<handshake::ControlInterface>(op);
  return controlInterface && controlInterface.isControl();
}

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

  /// A mapping between block arguments and their unique name in the .dot file.
  DenseMap<Value, std::string> argNameMap;

  void setUsedByMapping(Value v, Operation *op, StringRef node);
  void setProducedByMapping(Value v, Operation *op, StringRef node);

  /// Returns the name of the vertex using 'v' through 'consumer'.
  std::string getUsedByNode(Value v, Operation *consumer);
  /// Returns the name of the vertex producing 'v' through 'producer'.
  std::string getProducedByNode(Value v, Operation *producer);

  /// Maintain mappings between a value, the operation which (uses/produces) it,
  /// and the node name which the (tail/head) of an edge should refer to. This
  /// is used to resolve edges across handshake.instance's.
  // "'value' used by 'operation*' is used by the 'string' vertex"
  DenseMap<Value, std::map<Operation *, std::string>> usedByMapping;
  // "'value' produced by 'operation*' is produced from the 'string' vertex"
  DenseMap<Value, std::map<Operation *, std::string>> producedByMapping;
};

} // namespace

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

/// Returns true if v is used as a control operand in op
static bool isControlOperand(Operation *op, Value v) {
  if (isControlOp(op))
    return true;

  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<handshake::MuxOp, handshake::ConditionalBranchOp>(
          [&](auto op) { return v == op.getOperand(0); })
      .Case<handshake::ControlMergeOp>([&](auto) { return true; })
      .Default([](auto) { return false; });
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

static std::string getResName(handshake::FuncOp op, unsigned index) {
  return op.getResName(index).getValue().str();
}

static std::string getUniqueResName(StringRef instanceName,
                                    handshake::FuncOp op, unsigned index) {
  return getLocalName(instanceName, getResName(op, index));
}

void HandshakeToDotPass::setUsedByMapping(Value v, Operation *op,
                                          StringRef node) {
  usedByMapping[v][op] = node;
}
void HandshakeToDotPass::setProducedByMapping(Value v, Operation *op,
                                              StringRef node) {
  producedByMapping[v][op] = node;
}

std::string HandshakeToDotPass::getUsedByNode(Value v, Operation *consumer) {
  // Check if there is any mapping registerred for the value-use relation.
  auto it = usedByMapping.find(v);
  if (it != usedByMapping.end()) {
    auto it2 = it->second.find(consumer);
    if (it2 != it->second.end())
      return it2->second;
  }

  // fallback to the registerred name for the operation
  auto opNameIt = opNameMap.find(consumer);
  assert(opNameIt != opNameMap.end() &&
         "No name registered for the operation!");
  return opNameIt->second;
}

std::string HandshakeToDotPass::getProducedByNode(Value v,
                                                  Operation *producer) {
  // Check if there is any mapping registerred for the value-produce relation.
  auto it = producedByMapping.find(v);
  if (it != producedByMapping.end()) {
    auto it2 = it->second.find(producer);
    if (it2 != it->second.end())
      return it2->second;
  }

  // fallback to the registerred name for the operation
  auto opNameIt = opNameMap.find(producer);
  assert(opNameIt != opNameMap.end() &&
         "No name registered for the operation!");
  return opNameIt->second;
}

/// Emits additional, non-graphviz information about the connection between
/// from- and to. This does not have any effect on the graph itself, but may be
/// used by other tools to reason about the connectivity between nodes.
static void tryAddExtraEdgeInfo(mlir::raw_indented_ostream &os, Operation *from,
                                Value result, Operation *to) {
  os << " // ";

  if (from) {
    // Output port information
    auto results = from->getResults();
    unsigned resIdx =
        std::distance(results.begin(), llvm::find(results, result));
    auto fromNamedOpInterface = dyn_cast<handshake::NamedIOInterface>(from);
    if (fromNamedOpInterface) {
      auto resName = fromNamedOpInterface.getResultName(resIdx);
      os << " output=\"" << resName << "\"";
    } else
      os << " output=\"out" << resIdx << "\"";
  }

  if (to) {
    // Input port information
    auto ops = to->getOperands();
    unsigned opIdx = std::distance(ops.begin(), llvm::find(ops, result));
    auto toNamedOpInterface = dyn_cast<handshake::NamedIOInterface>(to);
    if (toNamedOpInterface) {
      auto opName = toNamedOpInterface.getOperandName(opIdx);
      os << " input=\"" << opName << "\"";
    } else
      os << " input=\"in" << opIdx << "\"";
  }
}

std::string HandshakeToDotPass::dotPrint(mlir::raw_indented_ostream &os,
                                         StringRef parentName,
                                         handshake::FuncOp f, bool isTop) {
  // Prints DOT representation of the dataflow graph, used for debugging.
  std::map<std::string, unsigned> opTypeCntrs;
  DenseMap<Operation *, unsigned> opIDs;
  auto name = f.getName();
  unsigned thisId = instanceIdMap[name.str()]++;
  std::string instanceName = parentName.str() + "." + name.str();
  // Follow submodule naming convention from FIRRTL lowering:
  if (!isTop)
    instanceName += std::to_string(thisId);

  /// Maintain a reference to any node in the args, body and result. These are
  /// used to generate cluster edges at the end of this function, to facilitate
  /// a  nice layout.
  std::optional<std::string> anyArg, anyBody, anyRes;

  // Sequentially scan across the operations in the function and assign instance
  // IDs to each operation.
  for (Block &block : f)
    for (Operation &op : block)
      opIDs[&op] = opTypeCntrs[op.getName().getStringRef().str()]++;

  if (!isTop) {
    os << "// Subgraph for instance of " << name << "\n";
    os << "subgraph \"cluster_" << instanceName << "\" {\n";
    os.indent();
    os << "label = \"" << name << "\"\n";
    os << "labeljust=\"l\"\n";
    os << "color = \"darkgreen\"\n";
  }
  os << "node [shape=box style=filled fillcolor=\"white\"]\n";

  Block *bodyBlock = &f.getBody().front();

  /// Print function arg and res nodes.
  os << "// Function argument nodes\n";
  std::string argsCluster = "cluster_" + instanceName + "_args";
  os << "subgraph \"" << argsCluster << "\" {\n";
  os.indent();
  // No label or border; the subgraph just forces args to stay together in the
  // diagram.
  os << "label=\"\"\n";
  os << "peripheries=0\n";
  for (const auto &barg : enumerate(bodyBlock->getArguments())) {
    auto argName = getArgName(f, barg.index());
    auto localArgName = getLocalName(instanceName, argName);
    os << "\"" << localArgName << "\" [shape=diamond";
    if (barg.index() == bodyBlock->getNumArguments() - 1) // ctrl
      os << ", style=dashed";
    os << " label=\"" << argName << "\"";
    os << "]\n";
    if (!anyArg.has_value())
      anyArg = localArgName;
  }
  os.unindent();
  os << "}\n";

  os << "// Function return nodes\n";
  std::string resCluster = "cluster_" + instanceName + "_res";
  os << "subgraph \"" << resCluster << "\" {\n";
  os.indent();
  // No label or border; the subgraph just forces args to stay together in the
  // diagram.
  os << "label=\"\"\n";
  os << "peripheries=0\n";
  // Get the return op; a handshake.func always has a terminator, making this
  // safe.
  auto endOp = *f.getBody().getOps<handshake::EndOp>().begin();
  for (const auto &res : llvm::enumerate(endOp.getOperands())) {
    auto resName = std::string("out") + std::to_string(res.index());
    auto uniqueResName = instanceName + "." + resName;
    os << "\"" << uniqueResName << "\" [shape=diamond";
    if (res.value().getType().isa<NoneType>()) // ctrl
      os << ", style=dashed";
    os << " label=\"" << resName << "\"";
    os << "]\n";

    // Create a mapping between the return op argument uses and the return
    // nodes.
    setUsedByMapping(res.value(), endOp, uniqueResName);

    if (!anyRes.has_value())
      anyRes = uniqueResName;
  }
  os.unindent();
  os << "}\n";

  /// Print operation nodes.
  std::string opsCluster = "cluster_" + instanceName + "_ops";
  os << "subgraph \"" << opsCluster << "\" {\n";
  os.indent();
  // No label or border; the subgraph just forces args to stay together in the
  // diagram.
  os << "label=\"\"\n";
  os << "peripheries=0\n";
  for (Operation &op : *bodyBlock) {
    if (!isa<handshake::InstanceOp, handshake::EndOp>(op)) {
      // Regular node in the diagram.
      opNameMap[&op] = dotPrintNode(os, instanceName, &op, opIDs);
      continue;
    }
    auto instOp = dyn_cast<handshake::InstanceOp>(op);
    if (instOp) {
      // Recurse into instantiated submodule.
      auto calledFuncOp =
          instOp->getParentOfType<ModuleOp>().lookupSymbol<handshake::FuncOp>(
              instOp.getModule());
      assert(calledFuncOp);
      auto subInstanceName = dotPrint(os, instanceName, calledFuncOp, false);

      // Create a mapping between the instance arguments and the arguments to
      // the module which it instantiated.
      for (const auto &arg : llvm::enumerate(instOp.getOperands())) {
        setUsedByMapping(
            arg.value(), instOp,
            getUniqueArgName(subInstanceName, calledFuncOp, arg.index()));
      }
      // Create a  mapping between the instance results and the results from the
      // module which it instantiated.
      for (const auto &res : llvm::enumerate(instOp.getResults())) {
        setProducedByMapping(
            res.value(), instOp,
            getUniqueResName(subInstanceName, calledFuncOp, res.index()));
      }
    }
  }
  if (!opNameMap.empty())
    anyBody = opNameMap.begin()->second;

  os.unindent();
  os << "}\n";

  /// Print operation result edges.
  os << "// Operation result edges\n";
  for (Operation &op : *bodyBlock) {
    for (auto result : op.getResults()) {
      for (auto &u : result.getUses()) {
        Operation *useOp = u.getOwner();
        if (useOp->getBlock() == bodyBlock) {
          os << "\"" << getProducedByNode(result, &op);
          os << "\" -> \"";
          os << getUsedByNode(result, useOp) << "\"";
          if (isControlOp(&op) || isControlOperand(useOp, result))
            os << " [style=\"dashed\"]";

          // Add extra, non-graphviz info to the edge.
          tryAddExtraEdgeInfo(os, &op, result, useOp);

          os << "\n";
        }
      }
    }
  }

  if (!isTop)
    os << "}\n";

  /// Print edges for function argument uses.
  os << "// Function argument edges\n";
  for (const auto &barg : enumerate(bodyBlock->getArguments())) {
    auto argName = getArgName(f, barg.index());
    os << "\"" << getLocalName(instanceName, argName) << "\" [shape=diamond";
    if (barg.index() == bodyBlock->getNumArguments() - 1)
      os << ", style=dashed";
    os << "]\n";
    for (auto *useOp : barg.value().getUsers()) {
      os << "\"" << getLocalName(instanceName, argName) << "\" -> \""
         << getUsedByNode(barg.value(), useOp) << "\"";
      if (isControlOperand(useOp, barg.value()))
        os << " [style=\"dashed\"]";

      tryAddExtraEdgeInfo(os, nullptr, barg.value(), useOp);
      os << "\n";
    }
  }

  /// Print edges from arguments cluster to ops cluster and ops cluster to
  /// results cluser, to coerce a nice layout.
  if (anyArg.has_value() && anyBody.has_value())
    os << "\"" << anyArg.value() << "\" -> \"" << anyBody.value()
       << "\" [lhead=\"" << opsCluster << "\" ltail=\"" << argsCluster
       << "\" style=invis]\n";
  if (anyBody.has_value() && anyRes.has_value())
    os << "\"" << anyBody.value() << "\" -> \"" << anyRes.value()
       << "\" [lhead=\"" << resCluster << "\" ltail=\"" << opsCluster
       << "\" style=invis]\n";

  os.unindent();
  return instanceName;
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::createHandshakeToDotPass() {
  return std::make_unique<HandshakeToDotPass>();
}
