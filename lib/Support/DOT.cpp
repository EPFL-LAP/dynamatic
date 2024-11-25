//===- DOT.cpp - Graphviz's DOT format support ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements support for building, parsing, and printing DOT graphs.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/DOT.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <utility>

using namespace mlir;
using namespace dynamatic;

DOTGraph::Builder DOTGraph::getBuilder() { return Builder(*this); }

SmallVector<const DOTGraph::Edge *>
DOTGraph::getAdjacentEdges(const Node &node) const {
  SmallVector<const Edge *> adjacentEdges;
  const Node *nodeAddr = &node;

  std::function<void(const Subgraph &)> addFromSubgraph =
      [&](const Subgraph &subgraph) -> void {
    for (const Edge *edge : subgraph.edges) {
      if (edge->srcNode == nodeAddr || edge->dstNode == nodeAddr)
        adjacentEdges.push_back(edge);
    }
    for (const Subgraph *subsub : subgraph.subgraphs)
      addFromSubgraph(*subsub);
  };

  addFromSubgraph(root);
  return adjacentEdges;
}

bool DOTGraph::WithAttributes::addAttr(const Twine &name, const Twine &value) {
  std::string nameStr = name.str();
  if (auto it = attrs.find(nameStr); it != attrs.end()) {
    // Replace with the new value
    it->getValue() = value.str();
    return false;
  }
  return attrs.insert({nameStr, value.str()}).second;
}

DOTGraph::Subgraph::~Subgraph() {
  // Subgraphs need to be deleted first because they are responsible to free
  // their own nodes/edges/subgraphs
  for (Subgraph *sub : subgraphs)
    delete sub;
  for (Node *n : nodes)
    delete n;
  for (Edge *e : edges)
    delete e;
}

//===----------------------------------------------------------------------===//
// Building
//===----------------------------------------------------------------------===//

DOTGraph::Node *DOTGraph::Builder::addNode(StringRef id,
                                           DOTGraph::Subgraph &subgraph) {
  if (auto it = earlyNodes.find(id); it != earlyNodes.end()) {
    DOTGraph::Node *node = it->second;
    earlyNodes.erase(id);

    // The node was used before it was defined, move it to the proper subgraph
    node->subgraph = &subgraph;
    subgraph.nodes.push_back(node);
    return node;
  }
  // Node IDs must be unique
  if (auto it = graph.nodesByID.find(id); it != graph.nodesByID.end())
    return nullptr;

  // New node definition
  auto *node = new DOTGraph::Node(id, subgraph);
  subgraph.nodes.push_back(node);
  graph.nodesByID.insert({node->id, node});
  return node;
}

DOTGraph::Edge &DOTGraph::Builder::addEdge(StringRef srcID, StringRef dstID,
                                           DOTGraph::Subgraph &subgraph) {
  const DOTGraph::Node &srcNode = getOrAddNode(srcID, subgraph);
  const DOTGraph::Node &dstNode = getOrAddNode(dstID, subgraph);
  auto *edge = new DOTGraph::Edge(srcNode, dstNode, subgraph);
  subgraph.edges.push_back(edge);
  graph.successors[&srcNode].push_back(edge);
  return *edge;
}

DOTGraph::Subgraph &DOTGraph::Builder::addSubgraph(StringRef id,
                                                   Subgraph &subgraph) {
  auto *subsub = new DOTGraph::Subgraph(id, &subgraph);
  subgraph.subgraphs.push_back(subsub);
  return *subsub;
}

DOTGraph::Node &DOTGraph::Builder::getOrAddNode(StringRef id,
                                                Subgraph &subgraph) {
  if (auto srcNodeIt = graph.nodesByID.find(id.str());
      srcNodeIt != graph.nodesByID.end())
    return *srcNodeIt->second;

  auto *node = new DOTGraph::Node(id, getRoot());
  earlyNodes.insert({node->id, node});
  graph.nodesByID.insert({node->id, node});
  return *node;
}

LogicalResult DOTGraph::Builder::parseFromFile(StringRef filepath) {
  Parser parser(graph, filepath);
  if (parser)
    return parser.emitError(filepath);
  return success();
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

void DOTGraph::WithAttributes::print(mlir::raw_indented_ostream &os) const {
  if (attrs.empty())
    return;

  os << " [";
  auto attrIt = attrs.begin();
  for (size_t i = 0, e = attrs.size(); i < e - 1; ++i, ++attrIt)
    os << "\"" << attrIt->getKey() << "\"=\"" << attrIt->getValue() << "\", ";
  os << "\"" << attrIt->getKey() << "\"=\"" << attrIt->getValue() << "\"";
  os << "]";
}

void DOTGraph::print(mlir::raw_ostream &os, EdgeStyle edgeStyle) const {
  mlir::raw_indented_ostream ios(os);

  std::string splines;
  if (edgeStyle == EdgeStyle::SPLINE)
    splines = "spline";
  else
    splines = "ortho";

  ios << "Digraph G {\n";
  ios.indent();
  ios << "splines=" << splines << "\ncompound=true\n";
  root.print(ios, true);
  ios.unindent();
  ios << "}\n";
}

void DOTGraph::Node::print(mlir::raw_indented_ostream &os) const {
  os << '"' << id << '"';
  WithAttributes::print(os);
  os << "\n";
}

void DOTGraph::Edge::print(mlir::raw_indented_ostream &os) const {
  os << '"' << srcNode->id << "\" -> \"" << dstNode->id << '"';
  WithAttributes::print(os);
  os << "\n";
}

void DOTGraph::Subgraph::print(mlir::raw_indented_ostream &os,
                               bool isRoot) const {
  if (!isRoot) {
    os << "subgraph ";
    if (!id.empty())
      os << id << " ";
    os << "{\n";
    os.indent();
  }
  if (!attrs.empty()) {
    os << "graph";
    WithAttributes::print(os);
    os << "\n";
  }

  for (const Node *node : nodes)
    node->print(os);
  for (const Edge *edge : edges)
    edge->print(os);
  for (const Subgraph *subsub : subgraphs)
    subsub->print(os, false);

  if (!isRoot) {
    os.unindent();
    os << "}\n";
  }
}

//===----------------------------------------------------------------------===//
// Parsing
//===----------------------------------------------------------------------===//

DOTGraph::Parser::Parser(DOTGraph &graph, StringRef filepath)
    : Builder(graph), currentSubgraph(&graph.root) {
  if (failed(tokenize(filepath)))
    return;
  if (parseLiteral("digraph") || parseID(&currentSubgraph->id) ||
      parseLiteral("{") || parseStatementList() || parseLiteral("}"))
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
  std::string id;
  if (parseID(&id))
    return setError("expected statement to start with valid ID");

  // First check for all strings that indicate a "special kind of statement"
  if (id == "graph")
    return parseAttrList(*currentSubgraph);

  if (id == "node" || id == "edge") {
    // Just parse the attributes and drop them immediately after, we don't
    // support this syntax
    WithAttributes withAttr;
    return parseAttrList(withAttr);
  }
  if (id == "subgraph") {
    std::string subgraph;
    parseOptionalID(&subgraph);
    Subgraph *oldSubgraph = currentSubgraph;
    currentSubgraph = &addSubgraph(subgraph, *currentSubgraph);
    if (parseLiteral("{") || parseStatementList() || parseLiteral("}"))
      return setError("failed to parse subgraph body");
    currentSubgraph = oldSubgraph;
    return success();
  }

  WithAttributes *dotAttr;
  if (parseOptionalLiteral("->")) {
    // This is a node
    DOTGraph::Node *node = addNode(id, *currentSubgraph);
    if (!node)
      return setError("failed to add node to the graph");
    dotAttr = node;
  } else {
    // This is an edge
    std::string dstNodeID;
    if (parseID(&dstNodeID))
      return setError("failed to parse edge destination node ID");

    Edge &edge = addEdge(id, dstNodeID, *currentSubgraph);
    dotAttr = &edge;
  }
  parseOptionalAttrList(*dotAttr);
  return success();
}

ParseResult DOTGraph::Parser::parseAttrList(WithAttributes &withAttr) {
  if (parseLiteral("["))
    return setError("expected '[' at beginning of attribute list");
  parseOptionalInnerAttrList(withAttr);
  if (parseLiteral("]"))
    return setError("expected ']' at end of attribute list");
  parseOptionalAttrList(withAttr);
  return success();
}

ParseResult DOTGraph::Parser::parseInnerAttrList(WithAttributes &withAttr) {
  std::string lhs, rhs;
  if (parseID(&lhs) || parseLiteral("=") || parseID(&rhs))
    return setError("expected 'lhs = rhs' attribute form");
  withAttr.addAttr(lhs, rhs);
  if (parseOptionalLiteral(";"))
    parseOptionalLiteral(",");
  parseOptionalInnerAttrList(withAttr);
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
