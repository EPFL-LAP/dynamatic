#include "dynamatic/Transforms/BufferPlacement/CircuitGraph.h"

using Path = std::vector<Operation *>;
using PathPair = std::pair<Path, Path>;

struct ReconvergentPath {
  Operation *diverge;
  Operation *reconverge;
  Path path1;
  Path path2;
};

using ReconvergentPathList = std::vector<ReconvergentPath>;

ReconvergentPathList findReconvergentPaths(const CircuitGraph &graph);