#ifndef DYNAMATIC_VISUAL_DATAFLOW_PARSER_H
#define DYNAMATIC_VISUAL_DATAFLOW_PARSER_H

#include "Graph.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {
namespace experimental {
namespace visual_dataflow {

class GraphParser {
public:
  GraphParser(std::string filePath);

  LogicalResult parse(Graph *graph);

private:
  std::string mFilePath;
};

} // namespace visual_dataflow
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_VISUAL_DATAFLOW_PARSER_H
