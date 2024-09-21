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

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace dynamatic {

/// A directed graph whose nodes and edges can optionally belong to a particular
/// node in a tree of subgraphs. All nodes, edges, and subgraphs can optionally
/// have a string-keyed dictionary of string attributes. This representation is
/// identical to the Graphviz representation of hierarchical graphs expressed in
/// the DOT format, hence the class name. Multiple edges between the same two
/// nodes are allowed, even if they are going in the same direction.
///
/// Memory for nodes and edges is managed by the top-level graph; subgraphs
/// maintain pointers to the subset of the nodes and edges that belong to them
/// but do not free the corresponding memory.
///
/// This class is immutable but a friend builder class is available to
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
  class Builder;
  class Parser;
  friend Builder;
  friend Parser;

public:
  /// Type to store attributes on graph nodes/edges/subgraphs.
  using Attributes = llvm::StringMap<std::string>;

  // A DOT node.
  struct Node {
    /// The node's unique identifier.
    std::string id;
    /// The node's attributes.
    Attributes attributes;

    Node(StringRef id) : id(id) {
      assert(!id.empty() && "node ID cannot be empty");
    }
  };

  // A DOT edge.
  struct Edge {
    // The edge's source node.
    const Node *srcNode;
    // The edge's destination node.
    const Node *dstNode;
    /// The edge's attributes.
    Attributes attributes;

    Edge(const Node *srcNode, const Node *dstNode)
        : srcNode(srcNode), dstNode(dstNode) {
      assert(srcNode && "source node cannot be nullptr");
      assert(dstNode && "destination node cannot be nullptr");
    }
  };

  /// A DOT subgraph, which may itself contain a tree of subgraphs.
  struct Subgraph {
    /// An optional ID for the subgraph (empty if not provided).
    std::string id;
    /// The nodes that belong to the subgraph (but not to potential children
    /// subgraphs).
    mlir::SetVector<Node *> nodes;
    /// The edges that belong to the subgraph (but not to potential children
    /// subgraphs).
    mlir::SetVector<Edge *> edges;
    /// A potential list of subgraphs nested within this subgraph.
    std::vector<Subgraph> subgraphs;
    /// The subgraph's attributes.
    Attributes attributes;

    Subgraph(StringRef id) : id(id) {}
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

  /// Returns the list of all nodes in the graph.
  ArrayRef<Node *> getNodes() const { return nodes; }

  /// Returns the list of all edges in the graph.
  ArrayRef<Edge *> getEdges() const { return edges; }

  /// Returns the list of top-level subgraphs (children of the root graph).
  ArrayRef<Subgraph> getSubgraphs() const { return subgraphs; }

  /// Returns the list of edges adjacent to a node (ingoing to it or outgoing
  /// from it).
  SmallVector<const Edge *> getAdjacentEdges(const Node &node) const;

  /// Deletes all nodes and edges, which are heap-allocated.
  ~DOTGraph();

private:
  /// An optional ID for the graph (empty if not provided).
  std::string id;
  /// The list of all nodes in the graph (heap-allocated).
  std::vector<Node *> nodes;
  /// The list of all edges in the graph (heap-allocated).
  std::vector<Edge *> edges;
  /// The list of all top-level subgraphs.
  std::vector<Subgraph> subgraphs;
  /// The top-level graph's attributes.
  Attributes attributes;

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

  /// Adds a node with a specific unique identifier. If `subgraph` is not null,
  /// the node is added to the specific subgraph. Fails when a node with this
  /// identifier was already added explicitly; otherwise returns the added node.
  FailureOr<Node *> addNode(StringRef id, Subgraph *subgraph = nullptr);

  /// Adds an edge between two nodes identified by their string identifiers.  If
  /// `subgraph` is not null, the edge is added to the specific subgraph. Note
  /// that it is possible to add an edge whose endpoints do not currently exist.
  /// In this case, nodes with corresponding identifers are added to the
  /// top-level graph.
  Edge *addEdge(StringRef srcID, StringRef dstID, Subgraph *subgraph = nullptr);

  /// Adds a subgraph with a specific unique identifier (which may be empty). If
  /// `subgraph` is not null, the subgraph is added to the specific subgraph.
  Subgraph *addSubgraph(StringRef id, Subgraph *subgraph = nullptr);

  /// Parses the graph from DOT-formatted file. Fails when the graph could not
  /// be parsed successfully (note that we do not support the full DOT grammar).
  LogicalResult parseFromFile(StringRef filepath);

protected:
  /// The graph being modified.
  DOTGraph &graph;
  /// The set of node identifiers that were added indirectly through an earlier
  /// edge. Explicitly adding an existing node whose identified is in this set
  /// is not an error, though adding that same node explicitly again is.
  DenseSet<StringRef> earlyNodes;

  /// Retrieves the node with the identifier if it exists. If it does not, add a
  /// new node to the top-level graph and returns a pointer to it.
  DOTGraph::Node *getOrAddNode(StringRef id);
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
  /// The current (nested) subgraph being parsed. `nullptr` when parsing at the
  /// top-level.
  Subgraph *currentSubgraph = nullptr;
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

  /// Parses a list of DOT attributes and inserts them in the attribute map.
  ParseResult parseAttrList(Attributes *attr);

  /// Parses an optional list of DOT attributes and inserts them in the
  /// attribute map.
  bool parseOptionalAttrList(Attributes *attr) {
    return parseOptional(&Parser::parseAttrList, attr);
  }

  /// Parses DOT attributes and inserts them in the attribute map.
  ParseResult parseInnerAttrList(Attributes *attr);

  /// Parses optional DOT attributes and inserts them in the attribute map.
  bool parseOptionalInnerAttrList(Attributes *attr) {
    return parseOptional(&Parser::parseInnerAttrList, attr);
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

/// Implements the logic to convert Handshake-level IR to a DOT. The only public
/// method of this class, print, converts an MLIR module containing a single
/// Handshake function into an equivalent DOT graph printed to a provided output
/// stream. In any legacy mode, the resulting DOT can be used with legacy
/// Dynamatic.
class DOTPrinter {
public:
  /// Style in which to render edges in the printed DOTs.
  enum class EdgeStyle {
    /// Render edges as splines (default).
    SPLINE,
    /// Render edges as orthogonal lines.
    ORTHO
  };

  DOTPrinter(EdgeStyle edgeStyle = EdgeStyle::SPLINE);

  /// Writes the DOT representation of the module to the provided output stream.
  LogicalResult write(mlir::ModuleOp mod, mlir::raw_indented_ostream &os);

private:
  /// Style of edges in the resulting DOTs.
  EdgeStyle edgeStyle;

  using PortNames = DenseMap<Operation *, handshake::PortNamer>;

  /// Writes the node corresponding to an operation.
  void writeNode(Operation *op, mlir::raw_indented_ostream &os);

  /// Writes the edge corresponding to an operation operand. The edge links an
  /// operation result's or block argument to an operation that uses the value.
  void writeEdge(OpOperand &oprd, const PortNames &portNames,
                 mlir::raw_indented_ostream &os);

  /// Writes the graph corresponding to the Handshake function.
  void writeFunc(handshake::FuncOp funcOp, mlir::raw_indented_ostream &os);

  /// Opens a subgraph in the DOT file using the provided name and label.
  void openSubgraph(StringRef name, StringRef label,
                    mlir::raw_indented_ostream &os);

  /// Closes a subgraph in the DOT file.
  void closeSubgraph(mlir::raw_indented_ostream &os);
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_DOT_H
