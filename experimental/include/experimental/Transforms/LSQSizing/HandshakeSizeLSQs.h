//===- HandshakeSizeLSQs.h - Sizes the LSQs --------*- C++ -*-===//
//
// This file declares the --handshake-size-lsqs pass.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_SIZE_LSQS_H
#define DYNAMATIC_TRANSFORMS_SIZE_LSQS_H

#include "dynamatic/Support/DynamaticPass.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "dynamatic/Support/TimingModels.h"


namespace dynamatic {
namespace experimental {
namespace lsqsizing {

#define GEN_PASS_DECL_HANDSHAKESIZELSQS
#define GEN_PASS_DEF_HANDSHAKESIZELSQS
#include "experimental/Transforms/Passes.h.inc"

std::unique_ptr<dynamatic::DynamaticPass>
createHandshakeSizeLSQs(StringRef timingModels = "");



} // namespace lsqsizing
} // namespace experimental
} // namespace dynamatic


#include <iostream>
#include <vector>
#include <unordered_map>
#include <list>
#include <string>

// Define a structure for a graph node
struct AdjListNode {
    int latency; // Value stored in the node
    std::optional<mlir::Operation*> op; // Pointer to the operation
    std::list<std::string> adjList; // Adjacency list (stores indices of adjacent nodes)
};

class AdjListGraph {
public:
    AdjListGraph() = default;
    void addNode(std::string op_name, int latency);
    void addNode(std::string op_name, int latency, mlir::Operation* op);
    void addEdge(std::string src, std::string dest);
    void printGraph();

private:
    std::unordered_map<std::string , AdjListNode> nodes; // Map to store nodes by their index
};

//AdjListGraph::AdjListGraph();



#endif // DYNAMATIC_TRANSFORMS_SIZE_LSQS_H