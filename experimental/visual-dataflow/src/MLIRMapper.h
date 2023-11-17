//===- MLIRMapper.h - Map MLIR module to Graph ------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for MLIR-Mapping, which is only used as part of the
// visual-dataflow. Declares the MLIRMapper, which produces the Graph
// representation of an MLIR module.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_VISUAL_DATAFLOW_MLIRMAPPER_H
#define DYNAMATIC_VISUAL_DATAFLOW_MLIRMAPPER_H

#include "Graph.h"
#include "GraphEdge.h"
#include "dynamatic/Support/TimingModels.h"

using namespace mlir;
using namespace circt;

namespace dynamatic {
namespace experimental {
namespace visual_dataflow {

/// Implements the logic to convert Handshake-level IR to a Graph. The only
/// public method of this class, mapMLIR, converts an MLIR module containing a
/// single Handshake function into an equivalent Graph.
class MLIRMapper {
public:
  /// Constructs an MLIRMapper that will map to the given graph.
  MLIRMapper(Graph *graph);

  /// Map Handshake-level IR to Graph Components.
  LogicalResult mapMLIR(mlir::ModuleOp mod);

private:
  /// The graph to map to.
  Graph *graph;

  /// A mapping between operations and their unique name.
  DenseMap<Operation *, std::string> opNameMap;

  /// Map an operation to a Node in the Graph.
  LogicalResult mapNode(Operation *op);

  /// Map an edge between a source and destination operation, which are
  /// linked by a result of the source that the destination uses as an
  /// operand.
  LogicalResult mapEdge(Operation *src, Operation *dst, Value val,
                        EdgeId *edgeId);

  /// Returns the unique name of the node representing the operation.
  std::string getNodeName(Operation *op);
};

} // namespace visual_dataflow
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_VISUAL_DATAFLOW_MLIRMAPPER_H
