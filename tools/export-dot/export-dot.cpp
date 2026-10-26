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
#include "dynamatic/Dialect/Handshake/HandshakeDialect.h"
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
#include "llvm/ADT/StringSet.h"
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
              cl::init(LabelType::UNAME), cl::cat(mainCategory));

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
  // a RAM op's memory is named after the RAM op
  if (auto ramOp = memref.getDefiningOp<handshake::RAMOp>())
    return getUniqueName(ramOp);

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

            std::string numSlots = std::to_string(bufferOp.getNumSlots());
            std::string bufferType =
                stringifyEnum(bufferOp.getBufferType()).str();
            return bufferType + " [" + numSlots + "]";
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
      .Case<handshake::BufferOp>(
          [&](handshake::BufferOp bufferOp) -> StringRef {
            return bufferOp.isBypassDV() ? "palegreen" : "mediumseagreen";
          })
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

  // The anonymous subgraphs pinned to the top and bottom rank of each block
  // cluster, holding the routing dots of the edges crossing blocks
  mlir::DenseMap<unsigned, DOTGraph::Subgraph *> bbTopRank, bbBottomRank;
  unsigned numRoutingDots = 0;

  // newrank lets dot honour rank constraints inside clusters; the two
  // spacings are in inches
  root->addAttr("newrank", "true");
  root->addAttr("nodesep", "0.5");
  root->addAttr("ranksep", "0.75");
  // More crossing-reduction iterations than dot's default
  root->addAttr("mclimit", "4");

  // Returns the cluster of a block, creating it on first use with its two
  // rank subgraphs
  auto getBlockCluster = [&](unsigned bb) -> DOTGraph::Subgraph & {
    if (auto subIt = bbSubgraphs.find(bb); subIt != bbSubgraphs.end())
      return *subIt->second;
    std::string name = "cluster" + std::to_string(bb);
    DOTGraph::Subgraph &bbSub = builder.addSubgraph(name, *root);
    bbSub.addAttr("label", "BB " + std::to_string(bb));
    bbSub.addAttr("style", "rounded");
    bbSubgraphs.insert({bb, &bbSub});

    DOTGraph::Subgraph &top = builder.addSubgraph("", bbSub);
    top.addAttr("rank", "min");
    bbTopRank.insert({bb, &top});
    DOTGraph::Subgraph &bottom = builder.addSubgraph("", bbSub);
    bottom.addAttr("rank", "max");
    bbBottomRank.insert({bb, &bottom});
    return bbSub;
  };

  // Adds a routing dot to a rank subgraph and returns its name
  auto addRoutingDot = [&](DOTGraph::Subgraph &rankSub) -> std::string {
    std::string name = "dot_" + std::to_string(numRoutingDots++);
    DOTGraph::Node *dot = builder.addNode(name, rankSub);
    dot->addAttr("shape", "point");
    dot->addAttr("width", "0.08");
    dot->addAttr("color", "blue");
    return name;
  };

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

    // The node's DOT "shape" attribute. A sink is drawn as a stub, a point
    // without a label, since it only marks a dropped output and a full box
    // would pull the layout apart around branches with an unused side.
    StringRef shape;
    bool isStub = isa<handshake::SinkOp>(op);
    if (isStub) {
      shape = "point";
      prettyLabel = "";
    } else if (isa<handshake::ArithOpInterface>(op)) {
      shape = "oval";
    } else {
      shape = "box";
    }

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
    if (isStub)
      node->addAttr("width", "0.15");
    return success();
  };

  // A memory controller is not drawn as one node but as abstractions placed
  // in the blocks they belong to: one node per block storing to it, taking
  // that block's count and the stores' addresses and data; one node per
  // load, taking the address and giving the data; and one node for the end,
  // taking the controller's start and end controls and giving its memory
  // end. These map each controller operand and result to its node, and each
  // abstraction node to the block it is drawn in.
  llvm::DenseMap<std::pair<Operation *, unsigned>, std::string> mcOperandNodes;
  llvm::DenseMap<std::pair<Operation *, unsigned>, std::string> mcResultNodes;
  llvm::StringMap<std::optional<unsigned>> abstractionBlocks;

  // A function argument is drawn once per block that consumes it, so each
  // use knows the name of the copy it connects to
  llvm::DenseMap<OpOperand *, std::string> argUseNodes;

  // The block an operation is drawn in: its own, or for a sink or a buffer
  // without one the block of the first operation with a block up its operand
  // chain, since buffers inherit the block of their consumer and a buffer
  // feeding a memory controller gets none
  auto getDrawBlock = [&](Operation *op) -> std::optional<unsigned> {
    std::optional<unsigned> bb = getLogicBB(op);
    while (!bb && isa<handshake::SinkOp, handshake::BufferOp>(op)) {
      op = op->getOperand(0).getDefiningOp();
      if (!op)
        break;
      bb = getLogicBB(op);
    }
    return bb;
  };

  // The block the consumer side of a use is drawn in. A function result is
  // drawn in the block of its producer.
  auto getUseBlock = [&](OpOperand &oprd) -> std::optional<unsigned> {
    Operation *userOp = oprd.getOwner();
    if (isa<handshake::MemoryControllerOp>(userOp))
      return abstractionBlocks[mcOperandNodes[{userOp,
                                               oprd.getOperandNumber()}]];
    if (isa<handshake::EndOp>(userOp)) {
      if (Operation *producer = oprd.get().getDefiningOp())
        return getDrawBlock(producer);
      return std::nullopt;
    }
    return getDrawBlock(userOp);
  };

  // The DOT endpoints of an operand edge: node and port names and indices
  // on both sides
  struct EdgeEndpoints {
    std::string srcNodeName, srcPortName, dstNodeName, dstPortName;
    unsigned srcIdx, dstIdx;
  };

  auto getEndpoints = [&](OpOperand &oprd) -> EdgeEndpoints {
    Value val = oprd.get();
    Operation *dstOp = oprd.getOwner();
    EdgeEndpoints ends;

    // Determine the edge's source
    if (auto res = dyn_cast<OpResult>(val)) {
      Operation *srcOp = res.getDefiningOp();
      ends.srcIdx = res.getResultNumber();
      if (isa<handshake::MemoryControllerOp>(srcOp))
        ends.srcNodeName = mcResultNodes[{srcOp, ends.srcIdx}];
      else
        ends.srcNodeName = getUniqueName(srcOp).str();
      ends.srcPortName =
          cast<handshake::NamedIOInterface>(srcOp).getResultName(ends.srcIdx);
    } else {
      Operation *parentOp = val.getParentBlock()->getParentOp();
      ends.srcIdx = cast<BlockArgument>(val).getArgNumber();
      ends.srcPortName =
          cast<handshake::NamedIOInterface>(parentOp).getOperandName(
              ends.srcIdx);
      ends.srcNodeName = argUseNodes.lookup(&oprd);
      if (ends.srcNodeName.empty())
        ends.srcNodeName = ends.srcPortName;
    }

    // Determine the edge's destination
    ends.dstIdx = oprd.getOperandNumber();
    if (isa<handshake::EndOp>(dstOp)) {
      Operation *parentOp = dstOp->getParentOp();
      ends.dstNodeName = ends.dstPortName =
          cast<handshake::NamedIOInterface>(parentOp).getResultName(
              ends.dstIdx);
    } else if (isa<handshake::MemoryControllerOp>(dstOp)) {
      ends.dstNodeName = mcOperandNodes[{dstOp, ends.dstIdx}];
      ends.dstPortName =
          cast<handshake::NamedIOInterface>(dstOp).getOperandName(ends.dstIdx);
    } else {
      ends.dstNodeName = getUniqueName(dstOp).str();
      ends.dstPortName =
          cast<handshake::NamedIOInterface>(dstOp).getOperandName(ends.dstIdx);
    }
    return ends;
  };

  // The attributes every edge of an operand's connection gets. The port
  // attributes go on the whole connection, so a routed connection has them
  // on each of its segments.
  auto setEdgeAttrs = [&](DOTGraph::Edge &edge, OpOperand &oprd,
                          const EdgeEndpoints &ends) -> void {
    edge.addAttr("from", ends.srcPortName);
    edge.addAttr("from_idx", std::to_string(ends.srcIdx));
    edge.addAttr("to", ends.dstPortName);
    edge.addAttr("to_idx", std::to_string(ends.dstIdx));
    edge.addAttr("style", getStyle(oprd.get()));
    edge.addAttr("penwidth", "2");
    edge.addAttr("dir", "both");
    edge.addAttr("arrowtail", "none");
    edge.addAttr("arrowhead", getArrowheadStyle(oprd));
    if (isBackedge(oprd.get(), oprd.getOwner()))
      edge.addAttr("color", "blue");
  };

  auto addEdge = [&](OpOperand &oprd, DOTGraph::Subgraph &subgraph) -> void {
    EdgeEndpoints ends = getEndpoints(oprd);
    DOTGraph::Edge &edge =
        builder.addEdge(ends.srcNodeName, ends.dstNodeName, subgraph);
    setEdgeAttrs(edge, oprd, ends);
    // Edges inside a block are weighted so that dot keeps them short and
    // vertical, widening the drawing rather than bending them
    if (&subgraph != root)
      edge.addAttr("weight", "4");
  };

  // An edge between two blocks is routed through a dot on the bottom rank
  // of the source block and a dot on the top rank of the destination block,
  // so that it leaves the source cluster at its bottom, enters the
  // destination cluster at its top, and keeps the clusters apart
  auto addRoutedEdge = [&](OpOperand &oprd, unsigned srcBB,
                           unsigned dstBB) -> void {
    EdgeEndpoints ends = getEndpoints(oprd);
    std::string bottomDot = addRoutingDot(*bbBottomRank[srcBB]);
    std::string topDot = addRoutingDot(*bbTopRank[dstBB]);

    // All three segments are weighted like edges inside a block, so that the
    // whole connection stays vertical
    DOTGraph::Edge &toBottom =
        builder.addEdge(ends.srcNodeName, bottomDot, *bbSubgraphs[srcBB]);
    setEdgeAttrs(toBottom, oprd, ends);
    toBottom.addAttr("arrowhead", "none");
    toBottom.addAttr("weight", "4");

    DOTGraph::Edge &across = builder.addEdge(bottomDot, topDot, *root);
    setEdgeAttrs(across, oprd, ends);
    across.addAttr("arrowhead", "none");
    across.addAttr("minlen", "4");
    across.addAttr("weight", "4");

    DOTGraph::Edge &toDst =
        builder.addEdge(topDot, ends.dstNodeName, *bbSubgraphs[dstBB]);
    setEdgeAttrs(toDst, oprd, ends);
    toDst.addAttr("weight", "4");
  };

  // Adds an abstraction node of a memory controller to the cluster of its
  // block, or to the root when it has none
  auto addAbstractionNode = [&](StringRef name, StringRef label,
                                handshake::MemoryControllerOp mcOp,
                                std::optional<unsigned> bb) -> void {
    DOTGraph::Subgraph &subgraph = bb ? getBlockCluster(*bb) : *root;
    DOTGraph::Node *node = builder.addNode(name, subgraph);
    node->addAttr("mlir_op", mcOp->getName().getStringRef());
    node->addAttr("label", label);
    node->addAttr("fillcolor", "lightsalmon");
    node->addAttr("shape", "box");
    node->addAttr("style", "filled, dashed");
    abstractionBlocks[name] = bb;
  };

  for (auto mcOp : funcOp.getOps<handshake::MemoryControllerOp>()) {
    std::string mcName = getUniqueName(mcOp).str();
    MCPorts ports = mcOp.getPorts();

    for (MCBlock &block : ports.getBlocks()) {
      // A block with only loads has no count and no store, so no block node
      std::string name = mcName + "_bb" + std::to_string(block.blockID);
      std::string label = mcName + "\\nbb" + std::to_string(block.blockID);
      if (block->ctrlPort || block->hasAnyPort<StorePort>())
        addAbstractionNode(name, label, mcOp, block.blockID);
      if (block->ctrlPort)
        mcOperandNodes[{mcOp, block->ctrlPort->getCtrlInputIndex()}] = name;
      for (MemoryPort &port : block->accessPorts) {
        if (auto storePort = dyn_cast<StorePort>(port)) {
          mcOperandNodes[{mcOp, storePort->getAddrInputIndex()}] = name;
          mcOperandNodes[{mcOp, storePort->getDataInputIndex()}] = name;
        } else if (auto loadPort = dyn_cast<LoadPort>(port)) {
          // A load gets its own node, since its data comes back out
          std::string loadName =
              mcName + "_" + getUniqueName(loadPort->getLoadOp()).str();
          addAbstractionNode(loadName, mcName + "\\nload", mcOp,
                             getDrawBlock(loadPort->getLoadOp()));
          mcOperandNodes[{mcOp, loadPort->getAddrInputIndex()}] = loadName;
          mcResultNodes[{mcOp, loadPort->getDataOutputIndex()}] = loadName;
        }
      }
    }

    // A controller behind an LSQ takes the LSQ's load address, store address
    // and store data and gives it the load data back. That side is one node,
    // drawn where the LSQ is: outside the blocks.
    if (ports.connectsToLSQ()) {
      LSQLoadStorePort lsqPort = ports.getLSQPort();
      std::string lsqName = mcName + "_lsq";
      addAbstractionNode(lsqName, mcName + "\\nlsq", mcOp, std::nullopt);
      mcOperandNodes[{mcOp, lsqPort.getLoadAddrInputIndex()}] = lsqName;
      mcOperandNodes[{mcOp, lsqPort.getStoreAddrInputIndex()}] = lsqName;
      mcOperandNodes[{mcOp, lsqPort.getStoreDataInputIndex()}] = lsqName;
      mcResultNodes[{mcOp, lsqPort.getLoadDataOutputIndex()}] = lsqName;
    }

    // The end node takes the end control and gives the memory end; it is
    // drawn where that control comes from.
    std::string endName = mcName + "_end";
    std::optional<unsigned> endBlock;
    if (Operation *endProducer = mcOp.getCtrlEnd().getDefiningOp())
      endBlock = getDrawBlock(endProducer);
    addAbstractionNode(endName, mcName + "\\nend", mcOp, endBlock);
    for (OpOperand &oprd : mcOp->getOpOperands()) {
      if (oprd.get() == mcOp.getCtrlEnd())
        mcOperandNodes[{mcOp, oprd.getOperandNumber()}] = endName;
    }
    mcResultNodes[{mcOp, cast<OpResult>(mcOp.getMemEnd()).getResultNumber()}] =
        endName;

    // The start node takes the memory start, which comes from a function
    // argument, either directly or through a fork when the memory start also
    // feeds something else; it is drawn in the entry block
    std::string startName = mcName + "_start";
    addAbstractionNode(startName, mcName + "\\nstart", mcOp, /*bb=*/0);
    for (OpOperand &oprd : mcOp->getOpOperands()) {
      if (oprd.get() == mcOp.getMemStart())
        mcOperandNodes[{mcOp, oprd.getOperandNumber()}] = startName;
    }
  }

  // A back edge is routed so that every segment still points down in rank,
  // which keeps dot from reordering the loop: the source goes down to a dot
  // on its block's bottom rank, an invisible dot outside all clusters joins
  // that dot and a dot on the destination block's top rank, and the latter
  // feeds the destination. The segments keep the blue of back edges.
  auto addBackEdge = [&](OpOperand &oprd, unsigned srcBB,
                         unsigned dstBB) -> void {
    EdgeEndpoints ends = getEndpoints(oprd);
    std::string bottomDot = addRoutingDot(*bbBottomRank[srcBB]);
    std::string topDot = addRoutingDot(*bbTopRank[dstBB]);
    std::string outerDot = "dot_" + std::to_string(numRoutingDots++);
    DOTGraph::Node *outer = builder.addNode(outerDot, *root);
    outer->addAttr("shape", "point");
    outer->addAttr("width", "0");
    outer->addAttr("height", "0");
    outer->addAttr("style", "invis");

    DOTGraph::Edge &toBottom =
        builder.addEdge(ends.srcNodeName, bottomDot, *bbSubgraphs[srcBB]);
    setEdgeAttrs(toBottom, oprd, ends);
    toBottom.addAttr("arrowhead", "none");

    DOTGraph::Edge &outerToBottom = builder.addEdge(outerDot, bottomDot, *root);
    setEdgeAttrs(outerToBottom, oprd, ends);
    outerToBottom.addAttr("arrowhead", "none");

    DOTGraph::Edge &topToOuter = builder.addEdge(topDot, outerDot, *root);
    setEdgeAttrs(topToOuter, oprd, ends);
    topToOuter.addAttr("arrowhead", "none");

    DOTGraph::Edge &toDst =
        builder.addEdge(topDot, ends.dstNodeName, *bbSubgraphs[dstBB]);
    setEdgeAttrs(toDst, oprd, ends);
  };

  // Create nodes and outgoing edges for all function arguments. An argument
  // is drawn once in every block that consumes it, and once in the root for
  // its consumers outside blocks.
  std::string funcOpName = funcOp->getName().getStringRef().str();
  for (const auto &[idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    // Arguments with memref types are represented by memory interfaces
    // inside the function, and are not displayed
    if (isa<MemRefType>(arg.getType()))
      continue;

    StringRef argName = funcOp.getArgName(idx);
    llvm::StringSet<> created;
    for (OpOperand &oprd : arg.getUses()) {
      std::optional<unsigned> useBlock = getUseBlock(oprd);
      std::string nodeName = argName.str();
      DOTGraph::Subgraph *subgraph = root;
      if (useBlock) {
        nodeName += "_bb" + std::to_string(*useBlock);
        subgraph = &getBlockCluster(*useBlock);
      }
      if (created.insert(nodeName).second) {
        DOTGraph::Node *node = builder.addNode(nodeName, *subgraph);
        if (!node)
          return funcOp.emitError()
                 << "failed to create node for argument " << idx;
        node->addAttr("label", argName);
        node->addAttr("mlir_op", funcOpName);
        node->addAttr("shape", "diamond");
        node->addAttr("style", getStyle(arg));
      }
      argUseNodes[&oprd] = nodeName;
      addEdge(oprd, *subgraph);
    }
  }

  // Create nodes for all function results, each in the block of its
  // producer. A memory end is drawn in the block of its memory controller's
  // end node.
  ValueRange results = funcOp.getBodyBlock()->getTerminator()->getOperands();
  for (const auto &[idx, res] : llvm::enumerate(results)) {
    StringRef resName = funcOp.getResName(idx);
    DOTGraph::Subgraph *subgraph = root;
    if (Operation *producer = res.getDefiningOp()) {
      std::optional<unsigned> bb;
      if (isa<handshake::MemoryControllerOp>(producer))
        bb = abstractionBlocks[mcResultNodes[{
            producer, cast<OpResult>(res).getResultNumber()}]];
      else
        bb = getDrawBlock(producer);
      if (bb)
        subgraph = &getBlockCluster(*bb);
    }
    DOTGraph::Node *node = builder.addNode(resName, *subgraph);
    if (!node)
      return funcOp.emitError() << "failed to create node for argument " << idx;
    node->addAttr("label", resName);
    node->addAttr("mlir_op", funcOpName);
    node->addAttr("shape", "diamond");
    node->addAttr("style", getStyle(res));
  }

  for (Operation &op : funcOp.getOps()) {
    // The end is drawn as the function's result nodes. A source only
    // triggers a constant; it is not drawn and neither is its edge, so the
    // constant appears with no input. A memory controller is drawn as its
    // abstraction nodes, added above. A RAM op, like a memref argument, is
    // represented by its memory interfaces, which are labelled with its name,
    // so it is not drawn and neither is its edge.
    if (isa<handshake::EndOp, handshake::SourceOp, handshake::RAMOp>(op))
      continue;
    bool isController = isa<handshake::MemoryControllerOp>(op);

    std::optional<unsigned> bb = getDrawBlock(&op);
    if (!isController) {
      DOTGraph::Subgraph *bbSub = bb ? &getBlockCluster(*bb) : root;
      if (failed(addNode(&op, *bbSub)))
        return failure();
    }

    // Create an edge for each use of each result of the operation. An edge
    // between two different blocks is routed through the blocks' rank dots;
    // an edge within a block stays in its cluster; anything else goes in
    // the root.
    for (OpResult res : op.getResults()) {
      std::optional<unsigned> srcBB = bb;
      if (isController)
        srcBB = abstractionBlocks[mcResultNodes[{&op, res.getResultNumber()}]];
      for (OpOperand &oprd : res.getUses()) {
        std::optional<unsigned> dstBB = getUseBlock(oprd);
        if (srcBB && dstBB && isBackedge(res, oprd.getOwner())) {
          getBlockCluster(*dstBB);
          addBackEdge(oprd, *srcBB, *dstBB);
        } else if (srcBB && dstBB && *srcBB != *dstBB) {
          getBlockCluster(*dstBB);
          addRoutedEdge(oprd, *srcBB, *dstBB);
        } else if (srcBB && dstBB) {
          addEdge(oprd, getBlockCluster(*srcBB));
        } else {
          addEdge(oprd, *root);
        }
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

  if (funcOp == nullptr) {
    modOp->emitError()
        << "No handshake function found in the mlir module (maybe you "
           "specified wrong function name to synthesize)! Aborting..\n";
    return 1;
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
