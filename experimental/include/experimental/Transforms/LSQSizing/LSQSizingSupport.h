#include <vector>
#include <unordered_map>
#include <list>
#include <string>
#include <optional>
#include <mlir/IR/Operation.h>


#ifndef DYNAMATIC_TRANSFORMS_LSQSIZING_LSQSIZINGSUPPORT_H
#define DYNAMATIC_TRANSFORMS_LSQSIZING_LSQSIZINGSUPPORT_H

namespace dynamatic {
namespace experimental {
namespace lsqsizing {

// Define a structure for a graph node
struct AdjListNode {
    int latency; // Value stored in the node
    std::optional<mlir::Operation*> op; // Pointer to the operation
    std::list<std::string> adjList; // Adjacency list (stores indices of adjacent nodes)
};

class AdjListGraph {
public:
    AdjListGraph() = default;
    void addNode(mlir::Operation* op, int latency);
    void addEdge(mlir::Operation* src, mlir::Operation* dest);
    void insertArtificialNodeOnEdge(mlir::Operation* src, mlir::Operation* dest, int latency);
    void printGraph();

private:
    std::unordered_map<std::string , AdjListNode> nodes; // Map to store nodes by their index
};

} // namespace lsqsizing
} // namespace experimental
} // namespace dynamatic

#endif // DYNAMATIC_TRANSFORMS_LSQSIZING_LSQSIZINGSUPPORT_H