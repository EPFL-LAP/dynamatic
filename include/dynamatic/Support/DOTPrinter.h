//===- DOTPrinter.h - Print DOT to standard output ------------*- C++ -*-===//
//
// Declarations for DOT-printing, which at the moment is only used as part of
// the export-dot tool. Declares the DOTPrinter, which produces the
// Graphviz-formatted representation of an MLIR module on stdout.
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
/// method of this class, printDOT, converts an MLIR module containing a single
/// Handshake function into an equivalent DOT graph printed on stdout. In legacy
/// mode, the resulting DOT can be used with legacy Dynamatic.
class DOTPrinter {
public:
  /// Constructs a DOTPrinter whose printing behavior is controlled by a couple
  /// flags, plus a path to a file containing timing models for use in legacy
  /// mode.
  DOTPrinter(bool legacy, bool debug, TimingDatabase *timingDB = nullptr);

  /// Prints Handshake-level IR to standard output.
  LogicalResult printDOT(mlir::ModuleOp mod);

private:
  /// Whether to export a legacy-compatible DOT.
  bool legacy;
  /// Whether to pretty-print the exported DOT (pretty-print if false).
  bool debug;
  /// Timing models for dataflow components (required in legacy mode, can safely
  /// be nullptr when not in legacy mode).
  TimingDatabase *timingDB;
  /// The stream to output to.
  mlir::raw_indented_ostream os;

  /// Maintain a mapping of module names and the number of times one of those
  /// modules have been instantiated in the design. This is used to generate
  /// unique names in the output graph.
  std::map<std::string, unsigned> instanceIdMap;

  /// A mapping between operations and their unique name in the .dot file.
  DenseMap<Operation *, std::string> opNameMap;

  /// In legacy mode, holds the set of all ports in the .dot file, represented
  /// by a unique name. Each port name is mapped to its width.
  std::unordered_map<std::string, unsigned> legacyPorts;

  /// In legacy mode, holds the set of all channels in the .dot file,
  /// represented by a pair of uniquely named ports. The first name represents
  /// the source port (out port of a module) while the second name represents
  /// the destination port (in port of a module).
  std::set<std::pair<std::string, std::string>> legacyChannels;

  /// Returns the name of a function's argument given its index.
  std::string getArgumentName(handshake::FuncOp funcOp, size_t idx);

  /// Computes all data attributes of a function argument (indicated by its
  /// index) for use in legacy Dynamatic and prints them to the output stream;
  /// it is the responsibility of the caller of this method to insert an opening
  /// bracket before the call and a closing bracket after the call.
  LogicalResult annotateArgumentNode(handshake::FuncOp funcOp, size_t idx);

  /// Computes all data attributes of an edge between a function argument
  /// (indicated by its index) and an operation for use in legacy Dynamatic and
  /// prints them to the output stream; it is the responsibility of the caller
  /// of this method to insert an opening bracket before the call and a closing
  /// bracket after the call.
  LogicalResult annotateArgumentEdge(handshake::FuncOp funcOp, size_t idx,
                                     Operation *dst);

  /// Returns the name of the node representing the operation.
  std::string getNodeName(Operation *op);

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
  LogicalResult annotateNode(Operation *op);

  /// Prints a node corresponding to an operation and, on success, returns a
  /// unique name for the operation in the outName argument.
  LogicalResult printNode(Operation *op);

  /// Computes all data attributes of an edge for use in legacy Dynamatic and
  /// prints them to the output; it is the responsibility of the caller
  /// of this method to insert an opening bracket before the call and a closing
  /// bracket after the call.
  LogicalResult annotateEdge(Operation *src, Operation *dst, Value val);

  /// Prints an edge between a source and destination operation, which are
  /// linked by a result of the source that the destination uses as an
  /// operand.
  LogicalResult printEdge(Operation *src, Operation *dst, Value val);

  /// Prints an instance of a handshake.func to the graph.
  LogicalResult printFunc(handshake::FuncOp funcOp);

  /// Opens a subgraph in the DOT file using the provided name and label.
  void openSubgraph(std::string &name, std::string &label);

  /// Closes a subgraph in the DOT file.
  void closeSubgraph();

  /// In legacy mode, registers inputs and outputs of a node for later DOT
  /// verification. Input and output ports can be skipped (useful for function
  /// arguments and end node) using their respective flags. Each registered node
  /// is named using the node name passed as argument as a prefix. As a
  /// consequence, a specific node name must never be used in more than one call
  /// to the method. Fails if a port with the same name as an existing port is
  /// generated.
  LogicalResult legacyRegisterPorts(NodeInfo &info, std::string &nodeName,
                                    bool skipInputs = false,
                                    bool skipOutputs = false);

  /// In legacy mode, registers a channel between two ports for later DOT
  /// verification. The edge is named using the source and destination node
  /// names passed as arguments. Fails if a channel with the same name as an
  /// existing channel is generated.
  LogicalResult legacyRegisterChannel(EdgeInfo &info, std::string &srcName,
                                      std::string &dstName);

  /// In legacy mode, verifies that all registered ports are part of a unique
  /// registered channel and that no port is undriven. Additionally, if the flag
  /// is set, verifies that ports linked by a channel have the same bitwidth
  /// (which is not true in general in DOTs produced by legacy Dynamatic). Fails
  /// if one of the above DOT invariants is broken.
  LogicalResult verifyDOT(handshake::FuncOp funcOp,
                          bool failOnWidthMismatch = false);
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
