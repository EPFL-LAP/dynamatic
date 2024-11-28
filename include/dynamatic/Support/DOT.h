//===- DOT.h - Graphviz's DOT format support --------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for Graphviz's DOT format, which expresses (directed) graph in text.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_DOT_H
#define DYNAMATIC_SUPPORT_DOT_H

#include "dynamatic/Support/LLVM.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace dynamatic {

/// A directed graph whose nodes and edges are organizes in a tree of subgraph.
/// All nodes, edges, and subgraphs can optionally have a string-keyed
/// dictionary of string attributes. This representation is identical to the
/// Graphviz representation of hierarchical graphs expressed in the DOT format,
/// hence the class name. Multiple edges between the same two nodes are allowed,
/// even if they are going in the same direction.
///
/// The memory for a node, edge, or subgraph is managed by the most nested
/// subgraph in which it is contained.
///
/// Instances of this class do not allow modifications of the underlying graph
/// elements directly; a friend builder class is available for that purpose to
/// programatically create graphs.
///
/// ```cpp
/// DOTGraph graph;
/// Builder builder(graph);
///
/// if (failed(builder.parseFromFile(pathToDOTFile))) {
///   // Parsing failed (note that we do not support the full DOT grammar)
/// }
/// ```
class DOTGraph {
public:
  class Builder;
  class Parser;
  friend Builder;
  friend Parser;

  struct Subgraph;

  /// Type to store attributes on graph nodes/edges/subgraphs.
  using Attributes = llvm::StringMap<std::string>;

  struct WithAttributes {
    /// A map of attributes for the DOT object.
    Attributes attrs;

    /// Adds an attribute (name-value pair) and returns whether this is the
    /// first attribute with this name (if not the existing value is replaced
    /// with the new one).
    bool addAttr(const Twine &name, const Twine &value);

    /// Prints the Graphviz-formatted attributes to the output stream.
    void print(mlir::raw_indented_ostream &os) const;
  };

  /// A DOT node.
  struct Node : public WithAttributes {
    /// The node's unique identifier.
    std::string id;
    /// Most nested subgraph the node belongs to.
    Subgraph *subgraph;

    Node(StringRef id, Subgraph &subgraph) : id(id), subgraph(&subgraph) {
      assert(!id.empty() && "node ID cannot be empty");
    }

    /// Prints the Graphviz-formatted node to the output stream.
    void print(mlir::raw_indented_ostream &os) const;
  };

  /// A DOT edge, between two nodes.
  struct Edge : public WithAttributes {
    /// The edge's source node.
    const Node *srcNode;
    /// The edge's destination node.
    const Node *dstNode;
    /// Most nested subgraph the edge belongs to.
    Subgraph *subgraph;

    Edge(const Node &srcNode, const Node &dstNode, Subgraph &subgraph)
        : srcNode(&srcNode), dstNode(&dstNode), subgraph(&subgraph) {}

    /// Prints the Graphviz-formatted edge to the output stream.
    void print(mlir::raw_indented_ostream &os) const;
  };

  /// A DOT subgraph, which may itself contain a tree of subgraphs.
  struct Subgraph : public WithAttributes {
    /// An optional ID for the subgraph (empty if not provided).
    std::string id;
    /// The nodes that belong immediately to the subgraph (heap-allocated).
    std::vector<Node *> nodes;
    /// The edges that belong immediately to the subgraph (heap-allocated).
    std::vector<Edge *> edges;
    /// A potential list of subgraphs nested within this subgraph.
    std::vector<Subgraph *> subgraphs;
    /// Parent subgraph. `nullptr` for the top-level subgraph.
    Subgraph *parent;

    Subgraph(StringRef id = {}, Subgraph *parent = nullptr)
        : id(id), parent(parent) {}

    /// Prints the Graphviz-formatted subgraph to the output stream.
    void print(mlir::raw_indented_ostream &os, bool isRoot) const;

    ~Subgraph();
  };

  /// Returns a builder object for the current graph, allowing modification of
  /// its internal state.
  Builder getBuilder();

  /// Attempts to retrieve a node by its unique identifier. Returns `nullptr` if
  /// no such node exists, a valid pointer to the node otherwise.
  const Node *getNode(const llvm::Twine &id) const {
    if (auto it = nodesByID.find(id.str()); it != nodesByID.end())
      return it->second;
    return nullptr;
  };

  /// Returns the (potentially empty) list of outgoing edges for a node.
  ArrayRef<const Edge *> getSuccessors(const Node &node) const {
    auto it = successors.find(&node);
    if (it == successors.end())
      return {};
    return it->second;
  };

  /// Returns the root subgraph.
  const Subgraph &getRoot() const { return root; }

  /// Returns the list of edges adjacent to a node (ingoing to it or outgoing
  /// from it).
  SmallVector<const Edge *> getAdjacentEdges(const Node &node) const;

  /// Style in which to render edges in the printed DOTs.
  enum class EdgeStyle {
    /// Render edges as splines (default).
    SPLINE,
    /// Render edges as orthogonal lines.
    ORTHO
  };

  void print(mlir::raw_ostream &os,
             EdgeStyle edgeStyle = EdgeStyle::SPLINE) const;

private:
  /// The root subgraph.
  Subgraph root;
  /// Maps all unique node identifiers to the node object they correspond to.
  llvm::DenseMap<StringRef, Node *> nodesByID;
  /// Maps nodes to their outgoing edges.
  llvm::DenseMap<const Node *, SmallVector<const Edge *>> successors;
};

/// Programmatic builder for DOT graphs. Allows to create the graph one
/// node/edge/subgraph at a time.
class DOTGraph::Builder {
public:
  /// Creates the builder from an empty DOT graph.
  Builder(DOTGraph &graph) : graph(graph) {}

  /// Adds a node with a specific unique identifier to a specific subgraph.
  /// Fails and returns `nullptr` when a node with this identifier was already
  /// added explicitly; otherwise returns a valid pointer to the added node.
  Node *addNode(StringRef id, Subgraph &subgraph);

  /// Adds an edge between two nodes identified by their string identifiers to a
  /// specific subgraph, then returns a reference to it. Note that it is
  /// possible to add an edge whose endpoints do not currently exist; in this
  /// case, nodes with corresponding identifers are added to the same subgraph.
  Edge &addEdge(StringRef srcID, StringRef dstID, Subgraph &subgraph);

  /// Adds a subgraph with a specific unique identifier (which may be empty) to
  /// a specific subgraph, then returns a reference to it.
  Subgraph &addSubgraph(StringRef id, Subgraph &subgraph);

  /// Parses the graph from DOT-formatted file. Fails when the graph could not
  /// be parsed successfully (note that we do not support the full DOT grammar).
  LogicalResult parseFromFile(StringRef filepath);

  /// Returns the root subgraph.
  Subgraph &getRoot() const { return graph.root; }

protected:
  /// The graph being modified.
  DOTGraph &graph;
  /// The set of node that were added indirectly through an earlier edge,
  /// identified by their name.
  llvm::StringMap<Node *> earlyNodes;

  /// Retrieves the node with the identifier if it exists. If it does not, add a
  /// new node to the subgraph and returns the added node.
  DOTGraph::Node &getOrAddNode(StringRef id, Subgraph &subgraph);
};

/// Parses a DOT graph from a DOT-formatted file. Right now this only supports a
/// subset of the full DOT grammar, which is available at
/// https://graphviz.org/doc/info/lang.html. That subset is enough to support
/// all of our current use cases.
class DOTGraph::Parser : public Builder {
public:
  /// Parses a DOT graph from a file located at the provided filepath.
  Parser(DOTGraph &graph, StringRef filepath);

  /// When parsing failed, writes an error to stderr and produced a failure.
  LogicalResult emitError(StringRef filepath) const;

  /// Whether parsing failed.
  operator bool() const { return parsingFailed; }

private:
  /// A DOT token, which is just a string accompanied by its location in the
  /// input file.
  struct Token {
    /// The token (may be enclosed in literal quotes). A valid token is never
    /// empty.
    std::string tok;
    /// The 1-indexed line number on which the token starts.
    size_t line;
    /// The 1-indexed columns number on which the token starts.
    size_t pos;

    /// Creates an empty token from its starting position. The token should be
    /// set manually later.
    Token(size_t line, size_t pos) : line(line), pos(pos) {}
  };

  /// The list of tokens parsed from the DOT-formated file.
  std::vector<Token> tokens;
  /// When parsing, the index of the next token to read in the list of tokens.
  size_t tokenIdx = 0;
  /// An optional error message set during parsing failure.
  std::optional<StringRef> error;
  /// The current subgraph being parsed.
  Subgraph *currentSubgraph;
  /// Whether parsing failed.
  bool parsingFailed = true;

  /// Attempts to tokenize the DOT-formatted file located at the given path.
  ParseResult tokenize(StringRef filepath);

  /// Sets the error message.
  LogicalResult setError(StringLiteral msg) {
    error = msg;
    return failure();
  }

  /// Clears the error message.
  void clearError() { error = std::nullopt; }

  /// Attempts to parse a part of the DOT grammar and returns whether there was
  /// an error parsing it. On error, restores the internal token/error state
  /// to what it was before the call so that parsing may continue.
  template <typename... Args>
  bool parseOptional(ParseResult (Parser::*parser)(Args...), Args... args) {
    size_t currentTokenIdx = tokenIdx;
    if ((*this.*parser)(std::forward<Args>(args)...)) {
      tokenIdx = currentTokenIdx;
      clearError();
      return true;
    }
    return false;
  }

  /// Parses a DOT statement.
  ParseResult parseStatement();

  /// Parses an optional DOT statement.
  bool parseOptionalStatement() {
    return parseOptional(&Parser::parseStatement);
  }

  /// Parses a list of DOT statements.
  ParseResult parseStatementList();

  /// Parses a list of DOT attributes and inserts them in the DOT object wth
  /// attributes.
  ParseResult parseAttrList(WithAttributes &withAttr);

  /// Parses an optional list of DOT attributes and inserts them in the DOT
  /// object wth attributes.
  bool parseOptionalAttrList(WithAttributes &withAttr) {
    return parseOptional<WithAttributes &>(&Parser::parseAttrList, withAttr);
  }

  /// Parses DOT attributes and inserts them in the DOT object wth attributes.
  ParseResult parseInnerAttrList(WithAttributes &withAttr);

  /// Parses optional DOT attributes and inserts them in the DOT object wth
  /// attributes.
  bool parseOptionalInnerAttrList(WithAttributes &withAttr) {
    return parseOptional<WithAttributes &>(&Parser::parseInnerAttrList,
                                           withAttr);
  }

  /// Parses the next token as an ID and stores it in the argument on success.
  ParseResult parseID(std::string *id);

  /// Parses the next token as an optional ID and stores it in the argument on
  /// success.
  bool parseOptionalID(std::string *id) {
    return parseOptional(&Parser::parseID, id);
  }

  /// Parses a literal string.
  ParseResult parseLiteral(StringRef literal);

  /// Parses an optional literal string.
  bool parseOptionalLiteral(StringRef literal) {
    return parseOptional(&Parser::parseLiteral, literal);
  }
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_DOT_H
