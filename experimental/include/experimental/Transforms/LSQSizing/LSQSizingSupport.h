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
    int latency; // Latency of the operation
    mlir::Operation* op; // Pointer to the operation
    std::set<std::string> edges; // Adjacency list (stores keys of adjacent nodes)
    std::set<std::string> backedges; // Backedge list (stores keys of adjacent nodes connected by backedges)
};

class AdjListGraph {
public:
    // Constructor for the graph, which takes a Vector of BBs, which make up a single CFDFC
    AdjListGraph(handshake::FuncOp funcOp, llvm::SetVector<unsigned> cfdfcBBs, TimingDatabase timingDB, unsigned II);

    // adds an Edge between src and dest
    void addEdge(mlir::Operation* src, mlir::Operation* dest);

    // Prints the graph
    void printGraph();

    // Prints the nodes on a path and their respecitve latencies
    void printPath(std::vector<std::string> path);

    // Finds all paths between two nodes, given the Operation pointers
    std::vector<std::vector<std::string>> findPaths(mlir::Operation *startOp, mlir::Operation *endOp, bool ignoreBackedge = false);

    // Finds all paths between two nodes, given the Operations unique names
    std::vector<std::vector<std::string>> findPaths(std::string start, std::string end, bool ignoreBackedge = false);

    // Returns the latency of a path
    int getPathLatency(std::vector<std::string> path);

    // Finds the path with the highest latency between two nodes
    int findMaxPathLatency(mlir::Operation *startOp, mlir::Operation *endOp, bool ignoreBackedge = false);

    // Finds the path with the lowest latency between two nodes
    int findMinPathLatency(mlir::Operation *startOp, mlir::Operation *endOp, bool ignoreBackedge = false);

    // Finds the longest non-cyclic path starting from a node (ignores backedges)
    std::vector<std::string> findLongestNonCyclicPath(mlir::Operation *startOp);

    // Returns all operations in the graph 
    std::vector<mlir::Operation*> getOperations();

    // Returns all operations in the graph with a specific operation name (not unique name, but type of operation e.g. "handshake.mux")
    std::vector<mlir::Operation*> getOperationsWithOpName(std::string opName);

    // Returns all operations which are connected to a specific operation (includes backedges, but skips the artificial nodes)
    std::vector<mlir::Operation*> getConnectedOps(mlir::Operation *op);

    // TODO: decide if keep recrusive or stack based version
    std::vector<std::string> findLongestNonCyclicPath2(mlir::Operation* startOp);


private:
    // Map to store the nodes by their Operations unique name
    std::unordered_map<std::string , AdjListNode> nodes; 
    
    // Adds a operation with its latency to the graph as a node
    void addNode(mlir::Operation* op, int latency);

    // Adds a backedge between two nodes
    void addBackedge(mlir::Operation* src, mlir::Operation* dest);

    // Adds an artificial node on a backedge, If there is already an edge it will be removed and replaced by a backedge
    void insertArtificialNodeOnBackedge(mlir::Operation* src, mlir::Operation* dest, int latency);

    // Depth first search algorithm for Path finding between two nodes
    void dfs(std::string& currentNode, std::string& end, std::vector<std::string>& currentPath, std::set<std::string>& visited, std::vector<std::vector<std::string>>& paths, bool ignoreBackedges = false);
    
    // TODO: decide if keep recrusive or stack based version    
    void dfsHelper(const std::string& currentNode, std::set<std::string>& visited, std::vector<std::string>& currentPath, int& maxLatency, std::vector<std::string>& bestPath);

    // Adds the edges between nodes for a result value of an operation
    void addChannelEdges(mlir::Value);

    // Adds the backedges between nodes for a result value of an operation
    void addChannelBackedges(mlir::Value, int latency);
};

} // namespace lsqsizing
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_LSQSIZING_LSQSIZINGSUPPORT_H