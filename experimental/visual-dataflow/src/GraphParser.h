//===- GraphParser.h - Parse a dataflow graph -------------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Graph parsing.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_VISUAL_DATAFLOW_PARSER_H
#define DYNAMATIC_VISUAL_DATAFLOW_PARSER_H

#include "Graph.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {
namespace experimental {
namespace visual_dataflow {

class GraphParser {
public:
  GraphParser(Graph *graph);

  LogicalResult parse(std::string &filePath);

private:
  Graph *mGraph;
};

} // namespace visual_dataflow
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_VISUAL_DATAFLOW_PARSER_H
