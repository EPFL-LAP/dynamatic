#include <vector>
#include <unordered_map>
#include <list>
#include <string>
#include <optional>
#include <set>
#include <mlir/IR/Operation.h>
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"


#ifndef DYNAMATIC_TRANSFORMS_LSQSIZING_LSQSIZINGSUPPORT_H
#define DYNAMATIC_TRANSFORMS_LSQSIZING_LSQSIZINGSUPPORT_H

namespace dynamatic {
namespace experimental {
namespace lsqsizing {

// Define a structure for a graph node
struct AdjListNode {
    int latency; // Value stored in the node
    mlir::Operation* op; // Pointer to the operation
    std::list<std::string> edges; // Adjacency list (stores indices of adjacent nodes)
    std::list<std::string> backedges; // Backedge list (stores indices of backedges)
};

class AdjListGraph {
public:
    AdjListGraph(buffer::CFDFC cfdfc, TimingDatabase timingDB, unsigned II);
    AdjListGraph(handshake::FuncOp funcOp, llvm::SetVector<unsigned> cfdfcBBs, TimingDatabase timingDB, unsigned II);

    void addEdge(mlir::Operation* src, mlir::Operation* dest);

    void printGraph();
    void printPath(std::vector<std::string> path);

    std::vector<std::vector<std::string>> findPaths(mlir::Operation *startOp, mlir::Operation *endOp, bool ignoreBackedge = false);
    std::vector<std::vector<std::string>> findPaths(std::string start, std::string end, bool ignoreBackedge = false);

    int getPathLatency(std::vector<std::string> path);

    int findMaxPathLatency(mlir::Operation *startOp, mlir::Operation *endOp, bool ignoreBackedge = false);
    int findMinPathLatency(mlir::Operation *startOp, mlir::Operation *endOp, bool ignoreBackedge = false);

    std::vector<std::string> findLongestNonCyclicPath(mlir::Operation *startOp);

    std::vector<mlir::Operation*> getOperationsWithOpName(std::string opName);

    std::vector<mlir::Operation*> getConnectedOps(mlir::Operation *op);



private:
    std::unordered_map<std::string , AdjListNode> nodes; // Map to store nodes by their index
    
    void addNode(mlir::Operation* op, int latency);
    void addBackedge(mlir::Operation* src, mlir::Operation* dest);
    void insertArtificialNodeOnBackedge(mlir::Operation* src, mlir::Operation* dest, int latency);
    void dfs(std::string& currentNode, std::string& end, std::vector<std::string>& currentPath, std::set<std::string>& visited, std::vector<std::vector<std::string>>& paths, bool ignoreBackedges = false);
    void addChannelEdges(mlir::Value);
    void addChannelBackedges(mlir::Value, int latency);
};

} // namespace lsqsizing
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_LSQSIZING_LSQSIZINGSUPPORT_H