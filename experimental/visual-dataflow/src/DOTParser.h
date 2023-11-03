//===- DOTParser.h - Parses a .dot file ------------------------*- C++ -*-===//
//
// The DOTParser class represents a parsing tool used to create a Graph compon-
// ent using a .dot file fed to the input.
//===----------------------------------------------------------------------===//

#ifndef VISUAL_DATAFLOW_DOTPARSER_H
#define VISUAL_DATAFLOW_DOTPARSER_H

#include "Graph.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <fstream>

namespace dynamatic {
namespace experimental {
namespace visual_dataflow {

LogicalResult processDOT(std::ifstream &file, Graph &graph);

} // namespace visual_dataflow
} // namespace experimental
} // namespace dynamatic

#endif // VISUAL_DATAFLOW_DOTPARSER_H
