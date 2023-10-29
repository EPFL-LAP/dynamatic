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

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Support/TimingModels.h"
#include "mlir/Support/IndentedOstream.h"
#include <map>
#include <set>
#include <unordered_map>

using namespace mlir;
using namespace circt;

namespace dynamatic {

struct NodeInfo;
struct EdgeInfo;

/// Implements the logic to convert Handshake-level IR to a DOT. The only public
/// method of this class, print, converts an MLIR module containing a single
/// Handshake function into an equivalent DOT graph printed to a provided output
/// stream. In any legacy mode, the resulting DOT can be used with legacy
/// Dynamatic.
class DOTPrinter {
public:
  /// Printing mode, dictating the structure and content of the printed DOTs.
  enum class Mode {
    /// Optimized for visualizing the circuit's structure (default).
    VISUAL,
    /// Compatible with legacy dot2vhdl tool.
    LEGACY,
    /// Compatible with legacy buffers and dot2vhdl tools.
    LEGACY_BUFFERS
  };
  /// Style in which to render edges in the printed DOTs.
  enum class EdgeStyle {
    /// Render edges as splines (default).
    SPLINE,
    /// Render edges as orthogonal lines.
    ORTHO
  };

  /// Constructs a DOTPrinter whose behavior is controlled by an overall
  /// printing mode and an edge style. A valid pointer to a timing database must
  /// be provided in any legacy-compatible mode to include node timing
  /// annotations, otherwise the constructor will assert.
  DOTPrinter(Mode mode = Mode::VISUAL, EdgeStyle edgeStyle = EdgeStyle::SPLINE,
             TimingDatabase *timingDB = nullptr);

  /// Prints Handshake-level IR to the provided output stream (or to stdout if
  /// `os` is nullptr).
  LogicalResult print(mlir::ModuleOp mod,
                      mlir::raw_indented_ostream *os = nullptr);

private:
  /// Printing mode (e.g., compatible with legacy tools or not).
  Mode mode;
  /// Style of edges in the resulting DOTs.
  EdgeStyle edgeStyle;
  /// Timing models for dataflow components (required in any legacy-compatible
  /// mode, can safely be nullptr when not in legacy mode).
  TimingDatabase *timingDB = nullptr;

  /// Returns the name of a function's argument given its index.
  std::string getArgumentName(handshake::FuncOp funcOp, size_t idx);

  /// Computes all data attributes of a function argument (indicated by its
  /// index) for use in legacy Dynamatic and prints them to the output stream;
  /// it is the responsibility of the caller of this method to insert an opening
  /// bracket before the call and a closing bracket after the call.
  LogicalResult annotateArgumentNode(handshake::FuncOp funcOp, size_t idx,
                                     mlir::raw_indented_ostream &os);

  /// Computes all data attributes of an edge between a function argument
  /// (indicated by its index) and an operation for use in legacy Dynamatic and
  /// prints them to the output stream; it is the responsibility of the caller
  /// of this method to insert an opening bracket before the call and a closing
  /// bracket after the call.
  LogicalResult annotateArgumentEdge(handshake::FuncOp funcOp, size_t idx,
                                     Operation *dst,
                                     mlir::raw_indented_ostream &os);

  /// Returns the content of the "delay" attribute associated to every graph
  /// node in legacy mode. Requires that `timingDB` points to a valid memory
  /// location.
  std::string getNodeDelayAttr(Operation *op);

  /// Returns the content of the "latency" attribute associated to every graph
  /// node in legacy mode. Requires that `timingDB` points to a valid memory
  /// location.
  std::string getNodeLatencyAttr(Operation *op);

  /// Computes all data attributes of an operation for use in legacy Dynamatic
  /// and prints them to the output stream; it is the responsibility of the
  /// caller of this method to insert an opening bracket before the call and a
  /// closing bracket after the call.
  LogicalResult annotateNode(Operation *op, mlir::raw_indented_ostream &os);

  /// Prints a node corresponding to an operation and, on success, returns a
  /// unique name for the operation in the outName argument.
  LogicalResult printNode(Operation *op, mlir::raw_indented_ostream &os);

  /// Computes all data attributes of an edge for use in legacy Dynamatic and
  /// prints them to the output; it is the responsibility of the caller
  /// of this method to insert an opening bracket before the call and a closing
  /// bracket after the call.
  LogicalResult annotateEdge(Operation *src, Operation *dst, Value val,
                             mlir::raw_indented_ostream &os);

  /// Prints an edge between a source and destination operation, which are
  /// linked by a result of the source that the destination uses as an
  /// operand.
  LogicalResult printEdge(Operation *src, Operation *dst, Value val,
                          mlir::raw_indented_ostream &os);

  /// Prints an instance of a handshake.func to the graph.
  LogicalResult printFunc(handshake::FuncOp funcOp,
                          mlir::raw_indented_ostream &os);

  /// Opens a subgraph in the DOT file using the provided name and label.
  void openSubgraph(std::string &name, std::string &label,
                    mlir::raw_indented_ostream &os);

  /// Closes a subgraph in the DOT file.
  void closeSubgraph(mlir::raw_indented_ostream &os);

  /// Returns whether the DOT printer was setup in a legacy-compatible mode.
  inline bool inLegacyMode() {
    return mode == Mode::LEGACY || mode == Mode::LEGACY_BUFFERS;
  }
};

/// Holds information about data attributes for a DOT node.
struct NodeInfo {
  /// The node's type.
  std::string type;
  /// A mapping between attribute name and value (printed between "").
  std::map<std::string, std::string> stringAttr;
  /// A mapping between attribute name and value (printed without "").
  std::map<std::string, int> intAttr;

  /// Constructs a NodeInfo with a specific type.
  NodeInfo(std::string type) : type(std::move(type)){};

  /// Prints all stored data attributes on the output stream. The function
  /// doesn't insert [brackets] around the attributes; it is the responsibility
  /// of the caller of this method to insert an opening bracket before the call
  /// and a closing bracket after the call.
  void print(mlir::raw_indented_ostream &os);
};

/// Holds information about data attributes for a DOT edge.
struct EdgeInfo {
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
