#include <vector>
#include <unordered_map>
#include <list>
#include <string>
#include <optional>
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

    //TODO check if backedges are already in there and just saved extra in cfdfc structure
    void printGraph();

    std::vector<std::vector<std::string>> findPaths(mlir::Operation *start_op, mlir::Operation *end_op, bool ignore_backedge = false);
    std::vector<std::vector<std::string>> findPaths(std::string start, std::string end, bool ignore_backedge = false);
    int getPathLatency(std::vector<std::string> path);

    int findMaxPathLatency(mlir::Operation *start_op, mlir::Operation *end_op);
    int findMinPathLatency(mlir::Operation *start_op, mlir::Operation *end_op);
    int findMaxLatencyFromStart(mlir::Operation *start_op);

    std::vector<mlir::Operation*> getOperationsWithOpName(std::string op_name);


private:
    std::unordered_map<std::string , AdjListNode> nodes; // Map to store nodes by their index
    
    void addNode(mlir::Operation* op, int latency);
    void addEdge(mlir::Operation* src, mlir::Operation* dest);
    void addBackedge(mlir::Operation* src, mlir::Operation* dest);
    void insertArtificialNodeOnBackedge(mlir::Operation* src, mlir::Operation* dest, int latency);
    std::vector<std::string> findLongestNonCyclicPath(mlir::Operation *start_op);

};

} // namespace lsqsizing
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_LSQSIZING_LSQSIZINGSUPPORT_H