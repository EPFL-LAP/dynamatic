#include "dynamatic/Dialect/Handshake/HandshakeOps.h"

using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;

class CircuitGraph {

  std::vector<Operation *> vertices;
  llvm::DenseMap<Operation *, llvm::DenseMap<Operation *, Value>> adjacency;

public:
  CircuitGraph(FuncOp funcOp) {
    for (Operation &op : funcOp.getOps()) {
      for (auto result : op.getResults()) {
        for (auto &use : result.getUses()) {
          Operation *nextOp = use.getOwner();

          if (isa<handshake::MemoryControllerOp>(nextOp) ||
              isa<handshake::LSQOp>(nextOp))
            continue;

          adjacency[&op][nextOp] = result;
        }
      }
      vertices.push_back(&op);
    }
  }

  const auto &getVertices() const { return vertices; }

  const auto &getAdjacency() const { return adjacency; }
};