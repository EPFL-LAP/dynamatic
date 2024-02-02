//===- CSVParser.h - Parses a csv transition line ---------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The extracted data provides information about the state of an edge of the
// graph for a given cycle.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_VISUAL_DATAFLOW_CSVPARSER_H
#define DYNAMATIC_VISUAL_DATAFLOW_CSVPARSER_H

#include "Graph.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <algorithm>
#include <sstream>

namespace dynamatic {
namespace experimental {
namespace visual_dataflow {

/// This function reads and interprets a edge transition line form a csv file.
LogicalResult processCSVLine(const std::string &line, size_t lineIndex,
                             Graph &graph, CycleNb *currCycle);

} // namespace visual_dataflow
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_VISUAL_DATAFLOW_CSVPARSER_H
