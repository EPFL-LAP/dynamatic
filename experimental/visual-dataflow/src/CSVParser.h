//===- CSVParser.h - Parses a csv transition line ------------*- C++ -*-===//
//
// The extracted data provides information about the state of an edge of the
// graph for a given cycle .
//===----------------------------------------------------------------------===//

#ifndef VISUAL_DATAFLOW_CSVPARSER_H
#define VISUAL_DATAFLOW_CSVPARSER_H
#include "Graph.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <algorithm>
#include <sstream>

namespace dynamatic {
namespace experimental {
namespace visual_dataflow {

/// This function reads and interprets a edge transition line form a csv file.
void processCSVLine(const std::string &line, Graph *graph, size_t lineIndex);

} // namespace visual_dataflow
} // namespace experimental
} // namespace dynamatic

#endif // VISUAL_DATAFLOW_CSVPARSER_H