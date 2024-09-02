//===- DOTPrinter.h - Print DOT to standard output ------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for DOT-printing, which at the moment is only used as part of
// the export-dot tool. Declares the DOTPrinter, which produces the
// Graphviz-formatted representation of an MLIR module on an output stream.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_SUPPORT_DOTPRINTER_H
#define DYNAMATIC_SUPPORT_DOTPRINTER_H

#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Support/IndentedOstream.h"
#include <map>

using namespace mlir;

namespace dynamatic {

struct DOTNode;
struct DOTEdge;

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

/// Holds information about data attributes for a DOT node.
struct DOTNode {
  /// The node's type.
  std::string type;
  /// A mapping between attribute name and value (printed between "").
  std::map<std::string, std::string> stringAttr;
  /// A mapping between attribute name and value (printed without "").
  std::map<std::string, int> intAttr;

  /// Constructs a NodeInfo with a specific type.
  DOTNode(std::string type) : type(std::move(type)) {};

  /// Prints all stored data attributes on the output stream. The function
  /// doesn't insert [brackets] around the attributes; it is the responsibility
  /// of the caller of this method to insert an opening bracket before the call
  /// and a closing bracket after the call.
  void print(mlir::raw_indented_ostream &os);
};

/// Holds information about data attributes for a DOT edge.
struct DOTEdge {
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
  void print(mlir::raw_indented_ostream &os);
};

} // namespace dynamatic

#endif // DYNAMATIC_SUPPORT_DOTPRINTER_H
