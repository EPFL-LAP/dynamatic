//===- export-dot.cpp - Export Handshake-level IR to DOT --------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the export-dot tool, which outputs on stdout the
// Graphviz-formatted representation on an input Handshake-level IR. The tool
// may be configured so that its output is compatible with .dot files expected
// by legacy Dynamatic, assuming the the inpur IR respects some constraints
// imposed in legacy dataflow circuits. This tools enables the creation of a
// bridge between Dynamatic and legacy Dynamatic, which is very useful in
// practice.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/DOT.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <string>

using namespace llvm;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

static cl::OptionCategory mainCategory("Application options");

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> timingDBFilepath(
    "timing-models", cl::Optional,
    cl::desc(
        "Relative path to JSON-formatted file containing timing models for "
        "dataflow components. The tool only tries to read from this file if it "
        "is ran in one of the legacy-compatible modes, where timing "
        "annotations are given to all nodes in the graph. By default, contains "
        "the relative path (from the project's top-level directory) to the "
        "file defining the default timing models in Dynamatic."),
    cl::init("data/components.json"), cl::cat(mainCategory));

static cl::opt<DOTGraph::EdgeStyle> edgeStyle(
    "edge-style", cl::Optional,
    cl::desc(
        "Style in which to render edges in the resulting DOTs (this is "
        "essentially the 'splines' attribute of the top-level DOT graph):"),
    cl::values(clEnumValN(DOTGraph::EdgeStyle::SPLINE, "spline",
                          "splines, default"),
               clEnumValN(DOTGraph::EdgeStyle::ORTHO, "ortho",
                          "orthogonal polylines")),
    cl::init(DOTGraph::EdgeStyle::SPLINE), cl::cat(mainCategory));

enum class LabelType { TYPE, UNAME };

static cl::opt<LabelType>
    labelType("label-type", cl::Optional,
              cl::desc("Information to use as node labels:"),
              cl::values(clEnumValN(LabelType::TYPE, "type",
                                    "type of the operation, default"),
                         clEnumValN(LabelType::UNAME, "uname",
                                    "unique name of the operation")),
              cl::init(LabelType::TYPE), cl::cat(mainCategory));

static constexpr StringLiteral DOTTED("dotted"), SOLID("solid"), DOT("dot"),
    NORMAL("normal");

static StringRef getStyle(Value val) {
  return isa<handshake::ControlType>(val.getType()) ? DOTTED : SOLID;
}

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
            auto optBufferType =
                params.getNamed(BufferOp::BUFFER_TYPE_ATTR_NAME);
            if (!optBufferType)
              return "buffer" + numSlotsStr;
            if (auto bufferTypeAttr =
                    dyn_cast<StringAttr>(optBufferType->getValue())) {
              llvm::StringRef bufferTypeStr = bufferTypeAttr.getValue();
              if (bufferTypeStr == BufferOp::ONE_SLOT_BREAK_DV) {
                return "DV" + numSlotsStr;
              } else if (bufferTypeStr == BufferOp::ONE_SLOT_BREAK_R) {
                return "R" + numSlotsStr;
              } else if (bufferTypeStr == BufferOp::FIFO_BREAK_DV) {
                return "DV" + numSlotsStr;
              } else if (bufferTypeStr == BufferOp::FIFO_BREAK_NONE) {
                return "NONE" + numSlotsStr;
              } else if (bufferTypeStr == BufferOp::ONE_SLOT_BREAK_DVR) {
                return "DVR" + numSlotsStr;
              } else if (bufferTypeStr == BufferOp::SHIFT_REG_BREAK_DV) {
                return "SRDV" + numSlotsStr;
              }
            }
            return "buffer" + numSlotsStr;
          })
      .Case<handshake::MemoryControllerOp>([&](MemoryControllerOp mcOp) {
        return getMemLabel("MC", getMemName(mcOp.getMemRef()));
      })
      .Case<handshake::LSQOp>([&](handshake::LSQOp lsqOp) {
        return getMemLabel("LSQ", getMemName(lsqOp.getMemRef()));
      })
      .Case<handshake::LoadOp>([&](handshake::LoadOp loadOp) {
        auto memOp = findMemInterface(loadOp.getAddressResult());
        StringRef memName = memOp ? getMemName(memOp.getMemRef()) : "";
        return getMemLabel("LD", memName);
      })
      .Case<handshake::StoreOp>([&](handshake::StoreOp storeOp) {
        auto memOp = findMemInterface(storeOp.getAddressResult());
        StringRef memName = memOp ? getMemName(memOp.getMemRef()) : "";
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
      .Case<handshake::BlockerOp>([&](auto) { return "cyan"; })
      .Case<handshake::BufferOp>([&](auto) { return "palegreen"; })
      .Case<handshake::EndOp>([&](auto) { return "gold"; })
      .Case<handshake::SourceOp, handshake::SinkOp>(
          [&](auto) { return "gainsboro"; })
      .Case<handshake::ConstantOp>([&](auto) { return "plum"; })
      .Case<handshake::MemoryOpInterface, handshake::LoadOp,
            handshake::StoreOp>([&](auto) { return "coral"; })
      .Case<handshake::MergeOp, handshake::ControlMergeOp, handshake::MuxOp>(
          [&](auto) { return "lightblue"; })
      .Case<handshake::BranchOp, handshake::ConditionalBranchOp>(
          [&](auto) { return "tan2"; })
      .Case<handshake::SpeculatorOp, handshake::SpecCommitOp,
            handshake::SpecSaveOp, handshake::SpecSaveCommitOp,
            handshake::SpeculatingBranchOp, handshake::NonSpecOp>(
          [&](auto) { return "salmon"; })
      .Default([&](auto) { return "moccasin"; });
}

static LogicalResult getDOTGraph(handshake::FuncOp funcOp, DOTGraph &graph) {
  DOTGraph::Builder builder(graph);
  mlir::DenseMap<unsigned, DOTGraph::Subgraph *> bbSubgraphs;
  DOTGraph::Subgraph *root = &builder.getRoot();

  // Collect port names for all operations and the top-level function
  using PortNames = mlir::DenseMap<Operation *, handshake::PortNamer>;
  PortNames portNames;
  portNames.try_emplace(funcOp, funcOp);
  for (Operation &op : funcOp.getOps())
    portNames.try_emplace(&op, &op);

  auto addNode = [&](Operation *op,
                     DOTGraph::Subgraph &subgraph) -> LogicalResult {
    // The node's DOT "mlir_op" attribute
    std::string mlirOpName = op->getName().getStringRef().str();
    std::string prettyLabel;
    switch (labelType) {
    case LabelType::TYPE:
      prettyLabel = getPrettyNodeLabel(op);
      break;
    case LabelType::UNAME:
      prettyLabel = getUniqueName(op);
      break;
    }
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

    DOTGraph::Node *node = builder.addNode(getUniqueName(op), subgraph);
    if (!node)
      return op->emitError() << "failed to create node for operation";
    node->addAttr("mlir_op", mlirOpName);
    node->addAttr("label", prettyLabel);
    node->addAttr("fillcolor", getNodeColor(op));
    node->addAttr("shape", shape);
    node->addAttr("style", style);
    return success();
  };

  auto addEdge = [&](OpOperand &oprd, DOTGraph::Subgraph &subgraph) -> void {
    Value val = oprd.get();
    Operation *dstOp = oprd.getOwner();

    // Determine the edge's source
    std::string srcNodeName, srcPortName;
    unsigned srcIdx;
    if (auto res = dyn_cast<OpResult>(val)) {
      Operation *srcOp = res.getDefiningOp();
      srcNodeName = getUniqueName(srcOp).str();
      srcIdx = res.getResultNumber();
      srcPortName = portNames.at(srcOp).getOutputName(srcIdx);
    } else {
      Operation *parentOp = val.getParentBlock()->getParentOp();
      srcIdx = cast<BlockArgument>(val).getArgNumber();
      srcNodeName = srcPortName = portNames.at(parentOp).getInputName(srcIdx);
    }

    // Determine the edge's destination
    std::string dstNodeName, dstPortName;
    unsigned dstIdx;
    if (isa<handshake::EndOp>(dstOp)) {
      Operation *parentOp = dstOp->getParentOp();
      dstIdx = oprd.getOperandNumber();
      dstNodeName = dstPortName = portNames.at(parentOp).getOutputName(dstIdx);
    } else {
      dstNodeName = getUniqueName(dstOp).str();
      dstIdx = oprd.getOperandNumber();
      dstPortName = portNames.at(dstOp).getInputName(dstIdx);
    }

    DOTGraph::Edge &edge = builder.addEdge(srcNodeName, dstNodeName, subgraph);
    edge.addAttr("from", srcPortName);
    edge.addAttr("from_idx", std::to_string(srcIdx));
    edge.addAttr("to", dstPortName);
    edge.addAttr("to_idx", std::to_string(dstIdx));
    edge.addAttr("style", getStyle(oprd.get()));
    edge.addAttr("dir", "both");
    edge.addAttr("arrowtail", "none");
    edge.addAttr("arrowhead", getArrowheadStyle(oprd));
    if (isBackedge(val, dstOp))
      edge.addAttr("color", "blue");
  };

  // Create nodes and outgoing edges for all function arguments
  std::string funcOpName = funcOp->getName().getStringRef().str();
  for (const auto &[idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (isa<MemRefType>(arg.getType()))
      // Arguments with memref types are represented by memory interfaces
      // inside the function so they are not displayed
      continue;

    // Create a node for the argument
    StringRef argName = portNames.at(funcOp).getInputName(idx);
    DOTGraph::Node *node = builder.addNode(argName, *root);
    if (!node)
      return funcOp.emitError() << "failed to create node for argument " << idx;
    node->addAttr("label", argName);
    node->addAttr("mlir_op", funcOpName);
    node->addAttr("shape", "diamond");
    node->addAttr("style", getStyle(arg));

    // Create edges between the argument and operand uses in the function
    for (OpOperand &oprd : arg.getUses())
      addEdge(oprd, *root);
  }

  // Create nodes for all function results
  ValueRange results = funcOp.getBodyBlock()->getTerminator()->getOperands();
  for (const auto &[idx, res] : llvm::enumerate(results)) {
    StringRef resName = portNames.at(funcOp).getOutputName(idx);
    DOTGraph::Node *node = builder.addNode(resName, *root);
    if (!node)
      return funcOp.emitError() << "failed to create node for argument " << idx;
    node->addAttr("label", resName);
    node->addAttr("mlir_op", funcOpName);
    node->addAttr("shape", "diamond");
    node->addAttr("style", getStyle(res));
  }

  for (Operation &op : funcOp.getOps()) {
    if (isa<handshake::EndOp>(op))
      continue;

    // Determine the subgraph in which to insert the operation
    DOTGraph::Subgraph *bbSub = root;
    std::optional<unsigned> bb = getLogicBB(&op);
    if (bb) {
      if (auto subIt = bbSubgraphs.find(*bb); subIt != bbSubgraphs.end()) {
        bbSub = subIt->second;
      } else {
        std::string name = "cluster" + std::to_string(*bb);
        bbSub = &builder.addSubgraph(name, *root);
        bbSub->addAttr("label", "BB " + std::to_string(*bb));
        bbSubgraphs.insert({*bb, bbSub});
      }
    }

    // Create a node for the operation
    if (failed(addNode(&op, *bbSub)))
      return failure();

    // Create an edge for each use of each result of the operation
    for (OpResult res : op.getResults()) {
      for (OpOperand &oprd : res.getUses()) {
        // Determine the subgraph in which to insert the edge
        DOTGraph::Subgraph *edgeSub = root;
        if (bbSub != root) {
          Operation *userOp = oprd.getOwner();
          std::optional<unsigned> userBB = getLogicBB(userOp);
          if (userBB && bb == userBB && !isa<handshake::EndOp>(userOp))
            edgeSub = bbSub;
        }
        addEdge(oprd, *edgeSub);
      }
    }
  }

  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Exports a DOT graph corresponding to the module for visualization\n"
      "and legacy-compatibility purposes.The pass only supports exporting\n"
      "the graph of a single Handshake function at the moment, and will fail\n"
      "if there is more than one Handhsake function in the module.");

  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFileName
                 << "': " << error.message() << "\n";
    return 1;
  }

  // Functions feeding into HLS tools might have attributes from high(er) level
  // dialects or parsers. Allow unregistered dialects to not fail in these
  // cases
  MLIRContext context;
  context.loadDialect<memref::MemRefDialect, arith::ArithDialect,
                      handshake::HandshakeDialect, math::MathDialect>();
  context.allowUnregisteredDialects();

  // Load the MLIR module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> modOp(
      mlir::parseSourceFile<ModuleOp>(sourceMgr, &context));
  if (!modOp)
    return 1;

  // We only support one function per module
  handshake::FuncOp funcOp = nullptr;
  for (auto op : modOp->getOps<handshake::FuncOp>()) {
    if (op.isExternal())
      continue;
    if (funcOp) {
      modOp->emitOpError() << "we currently only support one non-external "
                              "handshake function per module";
      return 1;
    }
    funcOp = op;
  }

  // Name all operations in the IR
  NameAnalysis nameAnalysis = NameAnalysis(*modOp);
  if (!nameAnalysis.isAnalysisValid())
    return 1;
  nameAnalysis.nameAllUnnamedOps();

  DOTGraph graph;
  if (failed(getDOTGraph(funcOp, graph)))
    return 1;

  graph.print(llvm::outs(), edgeStyle);
  return 0;
}
