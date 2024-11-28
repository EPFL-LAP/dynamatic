//===- DOT.cpp - Graphviz's DOT format support ------------------*- C++ -*-===//
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

#include "dynamatic/Support/DOT.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Dialect/Handshake/HandshakeAttributes.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Dialect/Handshake/HandshakeTypes.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/Utils/Utils.h"
#include "dynamatic/Transforms/HandshakeMaterialize.h"
#include "experimental/Transforms/Speculation/SpecAnnotatePaths.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

DOTGraph::Builder DOTGraph::getBuilder() { return Builder(*this); }

SmallVector<const DOTGraph::Edge *>
DOTGraph::getAdjacentEdges(const Node &node) const {
  SmallVector<const Edge *> adjacentEdges;
  const Node *nodeAddr = &node;
  for (const Edge *edge : edges) {
    if (edge->srcNode == nodeAddr || edge->dstNode == nodeAddr)
      adjacentEdges.push_back(edge);
  }
  return adjacentEdges;
}

DOTGraph::~DOTGraph() {
  for (Node *n : nodes)
    delete n;
  for (Edge *e : edges)
    delete e;
}

FailureOr<DOTGraph::Node *>
DOTGraph::Builder::addNode(StringRef id, DOTGraph::Subgraph *subgraph) {
  DOTGraph::Node *node;
  if (auto it = graph.nodesByID.find(id); it != graph.nodesByID.end()) {
    // This is fine as long as the node was added through an earlier edge
    if (!earlyNodes.erase(id)) {
      llvm::errs() << "Duplicate node with ID '" << id << "'\n";
      return failure();
    }
    node = it->second;
  } else {
    node = graph.nodes.emplace_back(new DOTGraph::Node(id));
    graph.nodesByID.insert({node->id, node});
  }

  if (subgraph)
    subgraph->nodes.insert(node);
  return node;
}

DOTGraph::Edge *DOTGraph::Builder::addEdge(StringRef srcID, StringRef dstID,
                                           DOTGraph::Subgraph *subgraph) {
  const DOTGraph::Node *srcNode = getOrAddNode(srcID);
  const DOTGraph::Node *dstNode = getOrAddNode(dstID);
  DOTGraph::Edge *edge =
      graph.edges.emplace_back(new DOTGraph::Edge(srcNode, dstNode));
  graph.successors[srcNode].push_back(edge);
  if (subgraph)
    subgraph->edges.insert(edge);
  return edge;
}

DOTGraph::Subgraph *DOTGraph::Builder::addSubgraph(StringRef id,
                                                   Subgraph *subgraph) {
  if (subgraph)
    return &subgraph->subgraphs.emplace_back(id);
  return &graph.subgraphs.emplace_back(id);
}

DOTGraph::Node *DOTGraph::Builder::getOrAddNode(StringRef id) {
  auto srcNodeIt = graph.nodesByID.find(id.str());
  if (srcNodeIt != graph.nodesByID.end())
    return srcNodeIt->second;

  earlyNodes.insert(id);
  return *addNode(id);
}

LogicalResult DOTGraph::Builder::parseFromFile(StringRef filepath) {
  Parser parser(graph, filepath);
  if (parser)
    return parser.emitError(filepath);
  return success();
}

DOTGraph::Parser::Parser(DOTGraph &graph, StringRef filepath) : Builder(graph) {
  if (failed(tokenize(filepath)))
    return;
  if (parseLiteral("digraph") || parseID(&graph.id) || parseLiteral("{") ||
      parseStatementList() || parseLiteral("}"))
    return;
  parsingFailed = false;
}

LogicalResult DOTGraph::Parser::emitError(StringRef filepath) const {
  assert(error && "no error to emit");
  const Token &tok = tokens[std::min(tokens.size() - 1, tokenIdx)];
  llvm::errs() << "Failed to parse graph @ \"" << filepath << "\", on token '"
               << tok.tok << "':" << tok.line << ":" << tok.pos << "\n";
  if (error)
    llvm::errs() << "\t" << *error << "\n";
  return failure();
}

ParseResult DOTGraph::Parser::parseStatementList() {
  if (!parseOptionalStatement()) {
    parseOptionalLiteral(";");
    return parseStatementList();
  }
  return success();
}

ParseResult DOTGraph::Parser::parseStatement() {
  auto &attr = currentSubgraph ? currentSubgraph->attributes : graph.attributes;

  std::string id;
  if (parseID(&id))
    return setError("expected statement to start with valid ID");

  // First check for all strings that indicate a "special kind of statement"
  if (id == "graph")
    return parseAttrList(&attr);

  if (id == "node" || id == "edge") {
    // Just parse the attributes and drop them immediately after, we don't
    // support this syntax
    DOTGraph::Attributes attributes;
    return parseAttrList(&attributes);
  }
  if (id == "subgraph") {
    std::string subgraph;
    parseOptionalID(&subgraph);
    Subgraph *oldSubgraph = currentSubgraph;
    currentSubgraph = addSubgraph(subgraph, currentSubgraph);
    if (parseLiteral("{") || parseStatementList() || parseLiteral("}"))
      return setError("failed to parse subgraph body");
    currentSubgraph = oldSubgraph;
    return success();
  }

  DOTGraph::Attributes *nodeOrEdgeAttr;
  if (parseOptionalLiteral("->")) {
    // This is a node
    FailureOr<DOTGraph::Node *> node = addNode(id, currentSubgraph);
    if (failed(node))
      return setError("failed to add node to the graph");
    nodeOrEdgeAttr = &(*node)->attributes;
  } else {
    // This is an edge
    std::string dstNodeID;
    if (parseID(&dstNodeID))
      return setError("failed to parse edge destination node ID");

    Edge *edge = addEdge(id, dstNodeID, currentSubgraph);
    nodeOrEdgeAttr = &edge->attributes;
  }
  parseOptionalAttrList(nodeOrEdgeAttr);
  return success();
}

ParseResult DOTGraph::Parser::parseAttrList(DOTGraph::Attributes *attr) {
  if (parseLiteral("["))
    return setError("expected '[' at beginning of attribute list");
  parseOptionalInnerAttrList(attr);
  if (parseLiteral("]"))
    return setError("expected ']' at end of attribute list");
  parseOptionalAttrList(attr);
  return success();
}

ParseResult DOTGraph::Parser::parseInnerAttrList(DOTGraph::Attributes *attr) {
  std::string lhs, rhs;
  if (parseID(&lhs) || parseLiteral("=") || parseID(&rhs))
    return setError("expected 'lhs = rhs' attribute form");
  attr->insert_or_assign(lhs, rhs);
  if (parseOptionalLiteral(";"))
    parseOptionalLiteral(",");
  parseOptionalInnerAttrList(attr);
  return success();
}

ParseResult DOTGraph::Parser::parseLiteral(StringRef literal) {
  if (tokens[tokenIdx++].tok != literal)
    return setError("expected token to be specific literal");
  return success();
}

ParseResult DOTGraph::Parser::parseID(std::string *id) {
  StringRef tokenRef(tokens[tokenIdx++].tok);

  // Check for alphanumeric string (and underscore) not starting with a digit
  if (llvm::all_of(tokenRef,
                   [](char c) { return llvm::isAlnum(c) || c == '_'; }) &&
      !llvm::isDigit(tokenRef.front())) {
    *id = tokenRef;
    return success();
  }

  // Check for double-quoted string (possible escaped quotes taken care of by
  // tokenization logic)
  if (tokenRef.front() == '"') {
    assert(tokenRef.back() == '"' && "unbalanced quoted string");
    *id = tokenRef.drop_front().drop_back().str();
    return success();
  }

  // Check for numeral [-]?(.[0-9]⁺ | [0-9]⁺(.[0-9]*)?)
  bool firstChar = true;
  bool decimalDigits = false;
  bool integerDigits = false;
  for (char c : tokenRef) {
    if (firstChar) {
      firstChar = false;
      if (c == '.') {
        decimalDigits = true;
      } else if (c == '-' || llvm::isDigit(c)) {
        integerDigits = true;
      } else {
        return setError("numeral ID should start with '.', '-', or a digit");
      }
    } else {
      if (decimalDigits) {
        if (!llvm::isDigit(c))
          return setError("numeral should only contain digits after '.'");
      }
      if (integerDigits) {
        if (c == '.') {
          integerDigits = false;
          decimalDigits = true;
        } else if (!llvm::isDigit(c)) {
          return setError("numeral should only contain digits before '.'");
        }
      }
    }
  }
  *id = tokenRef;
  return success();
}

ParseResult DOTGraph::Parser::tokenize(StringRef filepath) {
  static const std::set<char> symbols = {'{', '}', '[', ']',
                                         '=', ':', ';', ','};

  std::ifstream file(filepath.str());
  if (!file.is_open()) {
    llvm::errs() << "Failed to open DOT file @ \"" << filepath << "\"\n";
    return failure();
  }

  size_t lineNum = 0;
  auto error = [&](const llvm::Twine &msg) {
    llvm::errs() << "On line " << lineNum << ": " << msg << "\n";
    return failure();
  };

  std::string line;
  size_t idx = 0;
  auto getLine = [&]() -> bool {
    if (!std::getline(file, line))
      return false;
    idx = 0;
    ++lineNum;
    return true;
  };

  using FNextToken = std::function<FailureOr<Token>()>;
  FNextToken nextToken = [&]() -> FailureOr<Token> {
    while (idx == line.size()) {
      if (!getLine())
        return Token(0, 0);
    }

    // Discard leading whitespaces
    for (; idx < line.size(); ++idx) {
      char c = line[idx];
      if (!isspace(c))
        break;
    }
    if (idx == line.size()) {
      // This is a completely empty line, just skip it
      return nextToken();
    }

    Token token(lineNum, idx);

    // Read character by character until we finish the line (look over next
    // lines for quoted strings with backslashes at the end)
    std::stringstream ss;
    bool quoted = false;
    bool escaped = false;
    do {
      escaped = false;
      for (; idx < line.size(); ++idx) {
        char c = line[idx];
        if (escaped) {
          if (c != '"')
            ss << '\\';
          escaped = false;
          ss << c;
          continue;
        }

        if (quoted) {
          if (c == '\\') {
            escaped = true;
          } else {
            ss << c;
            if (c == '"') {
              token.tok = ss.str();
              ++idx;
              return token;
            }
          }
        } else {
          if (c == '"') {
            quoted = true;
            ss << c;
            continue;
          }
          if (symbols.find(c) != symbols.end()) {
            token.tok = ss.str();
            if (token.tok.empty()) {
              ++idx;
              token.tok = c;
              return token;
            }
            // Don't increment the index so that the next character to be
            // parsed will be the symbol
            return token;
          }
          // Detect edge symbol
          if (c == '-' && idx < line.size() - 1 && line[idx + 1] == '>') {
            token.tok = ss.str();
            if (token.tok.empty()) {
              idx += 2;
              token.tok = "->";
              return token;
            }
            // Don't increment the index so that the next character to be
            // parsed will be the symbol
            return token;
          }
          if (isspace(c)) {
            ++idx;
            token.tok = ss.str();
            return token;
          }
          if (!llvm::isAlnum(c) && c != '_' && c != '.' && c != '-') {
            return error("unquoted string can only contain alphanumeric "
                         "characters or underscores");
          }
          ss << c;
        }
      }
    } while (escaped && quoted && getLine());

    if (quoted)
      return error("unfinished quoted string");
    token.tok = ss.str();
    return token;
  };

  while (true) {
    FailureOr<Token> token = nextToken();
    if (failed(token))
      return failure();
    if (token->tok.empty())
      return success();
    tokens.push_back(*token);
  }
}

//===----------------------------------------------------------------------===//
// DOTPrinter
//===----------------------------------------------------------------------===//

namespace {

/// A list of ports (name and value).
using PortsData = std::vector<std::pair<std::string, Value>>;
/// A list of ports for memory interfaces (name, value, and potential name
/// suffix).
using MemPortsData = std::vector<std::tuple<std::string, Value, std::string>>;
/// In legacy mode, a port represented with a unique name and a bitwidth
using RawPort = std::pair<std::string, unsigned>;
/// Output stream type.
using OS = mlir::raw_indented_ostream;

} // namespace

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
      .Case<handshake::BufferOp>([&](auto) { return "green"; })
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

LogicalResult DOTPrinter::write(mlir::ModuleOp modOp, OS &os) {
  // We support at most one function per module
  auto funcs = modOp.getOps<handshake::FuncOp>();
  if (funcs.empty())
    return success();

  // We only support one function per module
  handshake::FuncOp funcOp = nullptr;
  for (auto op : modOp.getOps<handshake::FuncOp>()) {
    if (op.isExternal())
      continue;
    if (funcOp) {
      return modOp->emitOpError()
             << "we currently only support one non-external "
                "handshake function per module";
    }
    funcOp = op;
  }

  // Name all operations in the IR
  NameAnalysis nameAnalysis = NameAnalysis(modOp);
  if (!nameAnalysis.isAnalysisValid())
    return failure();
  nameAnalysis.nameAllUnnamedOps();

  // Print the graph
  writeFunc(funcOp, os);
  return success();
}

void DOTPrinter::openSubgraph(StringRef name, StringRef label, OS &os) {
  os << "subgraph \"" << name << "\" {\n";
  os.indent();
  os << "label=\"" << label << "\"\n";
}

void DOTPrinter::closeSubgraph(OS &os) {
  os.unindent();
  os << "}\n";
}

void DOTPrinter::writeNode(Operation *op, OS &os) {
  if (isa<handshake::EndOp>(op))
    return;

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

  prettyLabel = opName;

  // Write the node
  os << "\"" << opName << "\""
     << " [mlir_op=\"" << mlirOpName << "\", label=\"" << prettyLabel
     << "\", fillcolor=" << getNodeColor(op) << ", shape=\"" << shape
     << "\", style=\"" << style << "\"]\n";
}

void DOTPrinter::writeEdge(OpOperand &oprd, const PortNames &portNames,
                           OS &os) {
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

  os << "\"" << srcNodeName << "\" -> \"" << dstNodeName << "\" [from=\""
     << srcPortName << "\", from_idx=\"" << srcIdx << "\" to=\"" << dstPortName
     << "\", to_idx=\"" << dstIdx << "\", " << getEdgeStyle(oprd);
  if (isBackedge(val, dstOp))
    os << " color=\"blue\"";
  // Print speculative edge attribute
  if (experimental::speculation::isSpeculative(oprd, true))
    os << ((isBackedge(val, dstOp)) ? ", " : "") << " speculative=1";
  os << "]\n";
}

void DOTPrinter::writeFunc(handshake::FuncOp funcOp, OS &os) {
  std::string splines;
  if (edgeStyle == EdgeStyle::SPLINE)
    splines = "spline";
  else
    splines = "ortho";

  os << "Digraph G {\n";
  os.indent();
  os << "splines=" << splines << "\ncompound=true\n";

  // Collect port names for all operations and the top-level function
  PortNames portNames;
  portNames.try_emplace(funcOp, funcOp);
  for (Operation &op : funcOp.getOps())
    portNames.try_emplace(&op, &op);

  // Function arguments do not belong to any basic block
  os << "// Function arguments\n";
  for (const auto &[idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (isa<MemRefType>(arg.getType()))
      // Arguments with memref types are represented by memory interfaces
      // inside the function so they are not displayed
      continue;

    StringRef argLabel = portNames.at(funcOp).getInputName(idx);
    os << "\"" << argLabel << R"(" [mlir_op="handshake.func", shape=diamond, )"
       << "label=\"" << argLabel << "\", style=\"" << getStyle(arg) << "\"]\n";
  }

  // Function results do not belong to any basic block
  os << "// Function results\n";
  ValueRange results = funcOp.getBodyBlock()->getTerminator()->getOperands();
  for (const auto &[idx, res] : llvm::enumerate(results)) {
    StringRef resLabel = portNames.at(funcOp).getOutputName(idx);
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
    std::string graphLabel = "BB " + std::to_string(blockID);
    openSubgraph(graphName, graphLabel, os);

    os << "// Units in BB " << blockID << "\n";
    for (Operation *op : blockOps)
      writeNode(op, os);

    os << "// Channels in BB " << blockID << "\n";
    for (Operation *op : blockOps) {
      for (OpResult res : op->getResults()) {
        for (OpOperand &oprd : res.getUses()) {
          Operation *userOp = oprd.getOwner();
          std::optional<unsigned> bb = getLogicBB(userOp);
          if (bb && *bb == blockID && !isa<handshake::EndOp>(userOp))
            writeEdge(oprd, portNames, os);
          else
            outgoingEdges[blockID].push_back(&oprd);
        }
      }
    }

    // Close the subgraph
    closeSubgraph(os);
  }

  os << "// Units outside of all basic blocks\n";
  for (Operation *op : blocks.outOfBlocks)
    writeNode(op, os);

  // Print edges coming from function arguments if they haven't been so far
  os << "// Channels from function arguments\n";
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (isa<MemRefType>(arg.getType()))
      continue;
    for (OpOperand &oprd : arg.getUses())
      writeEdge(oprd, portNames, os);
  }

  // Print outgoing edges for each block
  for (auto &[blockID, blockEdges] : outgoingEdges) {
    os << "// Channels outgoing of BB " << blockID << "\n";
    for (OpOperand *oprd : blockEdges)
      writeEdge(*oprd, portNames, os);
  }

  os << "// Channels outside of all basic blocks\n";
  for (Operation *op : blocks.outOfBlocks) {
    for (OpResult res : op->getResults()) {
      for (OpOperand &oprd : res.getUses())
        writeEdge(oprd, portNames, os);
    }
  }

  os.unindent();
  os << "}\n";
}
