#include "dynamatic/Transforms/BufferPlacement/Reconvergence.h"
#include "dynamatic/Analysis/NameAnalysis.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

struct BFSNode {
  Operation *op;     // current node
  Operation *origin; // which successor of divergence node
};

struct OriginParentMap {
  Operation *origin;
  llvm::DenseMap<Operation *, Operation *> parent; // node -> parent
};

// Helper: reconstruct path from parent map
static Path
reconstructPath(Operation *node,
                const llvm::DenseMap<Operation *, Operation *> &parent,
                Operation *stopNode) {
  Path path;
  Operation *cur = node;

  while (cur) {
    path.push_back(cur);
    if (cur == stopNode)
      break; // stop at divergence
    auto it = parent.find(cur);
    if (it == parent.end())
      break;
    cur = it->second;
  }

  std::reverse(path.begin(), path.end());
  return path;
}

ReconvergentPathList findReconvergentPathsFromDiverge(
    Operation *diverge,
    const llvm::DenseMap<Operation *, llvm::DenseMap<Operation *, Value>>
        &adjacency) {

  ReconvergentPathList reconPaths;

  llvm::errs() << "diverge " << getUniqueName(diverge) << "\n";

  auto it = adjacency.find(diverge);
  if (it == adjacency.end() || it->second.size() <= 1)
    return reconPaths;

  std::vector<BFSNode> queue;
  // visitedOrigins[node] = set of origins that have visited `node`
  llvm::DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 4>> visitedOrigins;

  // Replace vector-of-struct with a map: origin -> parent map
  llvm::DenseMap<Operation *, llvm::DenseMap<Operation *, Operation *>>
      parentMaps;

  // Initialize: each successor is its own origin. parent[succ] = diverge
  for (auto &[succ, _] : it->second) {
    queue.push_back({succ, succ});
    visitedOrigins[succ].insert(succ);
    parentMaps[succ][succ] =
        diverge; // parent of succ (for origin succ) is diverge
  }

  // BFS loop (vector used as queue)
  for (size_t idx = 0; idx < queue.size(); ++idx) {
    BFSNode node = queue[idx];
    Operation *cur = node.op;
    Operation *origin = node.origin;

    auto nextIt = adjacency.find(cur);
    if (nextIt == adjacency.end())
      continue;

    for (auto &[succ, _] : nextIt->second) {
      auto &originSet = visitedOrigins[succ];

      // try to insert this origin; if it was already present -> skip (cycle for
      // same origin)
      if (!originSet.insert(origin).second)
        continue;

      // IMPORTANT: set the parent for this origin BEFORE reconstructing paths.
      // This ensures reconstructPath(succ, parentMaps[origin], diverge) will
      // find succ->parent.
      parentMaps[origin][succ] = cur;

      // reconvergence: visited from other origin(s) (originSet now contains
      // 'origin' too)
      if (originSet.size() > 1) {

        if (!isa<handshake::MuxOp>(succ)) {

          for (Operation *prevOrigin : originSet) {
            if (prevOrigin == origin)
              continue;

            // Both parent maps must exist
            auto it1 = parentMaps.find(origin);
            auto it2 = parentMaps.find(prevOrigin);
            if (it1 == parentMaps.end() || it2 == parentMaps.end())
              continue;

            const auto &p1 = it1->second;
            const auto &p2 = it2->second;

            // Reconstruct paths (stop at diverge)
            Path path1 = reconstructPath(succ, p1, diverge);
            Path path2 = reconstructPath(succ, p2, diverge);

            reconPaths.push_back({diverge, succ, path1, path2});

            Operation *newOrigin =
                succ; // the reconvergence node acts as new origin

            originSet.clear();
            originSet.insert(newOrigin);
            // for (auto *origin : originSet)
            //   mergedOrigins.erase(origin); // clear old ones

            // continue BFS, but from succ with origin = newOrigin
            queue.push_back({succ, newOrigin});
            visitedOrigins[succ].clear();
            visitedOrigins[succ].insert(newOrigin);

            continue;
          }
        }

        // continue;
      }

      // enqueue successor for further BFS expansion
      queue.push_back({succ, origin});
    }
  }

  return reconPaths;
}

ReconvergentPathList findReconvergentPaths(const CircuitGraph &graph) {
  ReconvergentPathList result;

  for (auto &[diverge, succMap] : graph.getAdjacency()) {
    if (isa<handshake::ForkOp>(diverge)) {
      // Get reconvergent paths from this divergence node
      ReconvergentPathList localPaths =
          findReconvergentPathsFromDiverge(diverge, graph.getAdjacency());

      // Append all of them to the global result
      result.insert(result.end(), localPaths.begin(), localPaths.end());
    }
  }

  return result;
}
